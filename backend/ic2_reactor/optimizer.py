from __future__ import annotations

import heapq
import math
import multiprocessing
import queue
import random
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from functools import lru_cache

from .components import COMPONENTS
from .engine import ReactorSimulator, SimulationOptions
from .gpu_acceleration import (
    CudaBatchScorer,
    CudaBatchScores,
    CudaFixedPointCertificate,
    CudaFixedPointEvaluator,
    cuda_device_info,
    select_screened_layouts,
)
from .gpu_full_simulation import CudaFullSimulator, CudaSimulationResult
from .mark import FUEL_CYCLE_REACTOR_TICKS, classify_mark, mark_family
from .models import Layout, OptimizationRequest
from .skeleton_table import POWER_EMPTY, SkeletonPowerTable, SkeletonSearchNode


@dataclass(slots=True)
class CandidateResult:
    layout: tuple[str, ...]
    mark: str
    average_eu_per_tick: float
    total_eu: float
    safe_game_ticks: int
    safety_margin: float
    component_count: int
    canonical: str

    def score(self) -> tuple:
        """Rank candidates strictly by average generation power."""
        return (self.average_eu_per_tick,)

    def public_dict(self, columns: int) -> dict:
        return {
            "layout": {"ruleset": "ic2-experimental-2.8.221", "columns": columns, "initial_hull_heat": 0, "slots": list(self.layout)},
            "mark": self.mark,
            "average_eu_per_tick": self.average_eu_per_tick,
            "total_eu": self.total_eu,
            "safe_game_ticks": self.safe_game_ticks,
            "safety_margin": self.safety_margin,
            "component_count": self.component_count,
        }


def _transform(layout: tuple[str, ...], columns: int, flip_h: bool, flip_v: bool) -> tuple[str, ...]:
    rows = [list(layout[row * columns:(row + 1) * columns]) for row in range(6)]
    if flip_h:
        rows = [list(reversed(row)) for row in rows]
    if flip_v:
        rows = list(reversed(rows))
    return tuple(item for row in rows for item in row)


def canonical_layout(layout: tuple[str, ...], columns: int) -> str:
    return "|".join(canonical_tuple(layout, columns))


def canonical_tuple(layout: tuple[str, ...], columns: int) -> tuple[str, ...]:
    """Return a display-group key for mirrored layouts, not a simulation-equivalence key."""
    return min(_transform(layout, columns, h, v) for h in (False, True) for v in (False, True))


def power_skeleton(layout: tuple[str, ...]) -> tuple[str, ...]:
    """Keep only components that directly participate in EU pulse production."""
    return tuple(
        item if COMPONENTS[item].kind in {"fuel", "reflector"} else "empty"
        for item in layout
    )


@lru_cache(maxsize=131_072)
def skeleton_eu_per_tick(skeleton: tuple[str, ...], columns: int) -> float:
    """Calculate exact steady EU/t for an unchanged fuel/reflector skeleton."""
    pulses = 0
    for index, item in enumerate(skeleton):
        spec = COMPONENTS[item]
        if spec.kind != "fuel":
            continue
        row, column = divmod(index, columns)
        neighbors: list[int] = []
        if column > 0:
            neighbors.append(index - 1)
        if column + 1 < columns:
            neighbors.append(index + 1)
        if row > 0:
            neighbors.append(index - columns)
        if row < 5:
            neighbors.append(index + columns)
        neighbor_pulses = sum(
            COMPONENTS[skeleton[neighbor]].kind in {"fuel", "reflector"}
            for neighbor in neighbors
        )
        pulses += spec.rod_count * (spec.internal_pulses + neighbor_pulses)
    return pulses * ReactorSimulator.EU_PER_PULSE


@lru_cache(maxsize=131_072)
def skeleton_heat_per_tick(skeleton: tuple[str, ...], columns: int) -> int:
    """Calculate exact heat generated each reactor tick by a static skeleton."""
    total = 0
    for index, item in enumerate(skeleton):
        spec = COMPONENTS[item]
        if spec.kind != "fuel":
            continue
        row, column = divmod(index, columns)
        neighbors: list[int] = []
        if column > 0:
            neighbors.append(index - 1)
        if column + 1 < columns:
            neighbors.append(index + 1)
        if row > 0:
            neighbors.append(index - columns)
        if row < 5:
            neighbors.append(index + columns)
        neighbor_pulses = sum(
            COMPONENTS[skeleton[neighbor]].kind in {"fuel", "reflector"}
            for neighbor in neighbors
        )
        pulses = spec.internal_pulses + neighbor_pulses
        total += 2 * spec.rod_count * pulses * (pulses + 1)
    return total


@lru_cache(maxsize=131_072)
def sustainable_vent_upper_bound(
    skeleton: tuple[str, ...],
    columns: int,
    cooling_caps: tuple[tuple[str, int], ...],
) -> int:
    """Return an optimistic upper bound on indefinitely vented heat per tick.

    Heat exchangers only move heat, while coolant cells, condensators and
    plating provide finite storage.  A self vent cannot reject more than its
    ``self_vent`` value.  A component vent is optimistically allowed to cool
    every free neighboring slot, even if those slots cannot all be occupied by
    coolable components.  Ignoring those conflicts can only overestimate the
    realizable cooling rate, which makes ``generated > bound`` a sound Mark I
    infeasibility certificate.
    """
    free_positions = tuple(index for index, item in enumerate(skeleton) if item == "empty")
    if not free_positions:
        return 0
    free = set(free_positions)
    max_free_degree = 0
    for index in free_positions:
        row, column = divmod(index, columns)
        degree = 0
        if column > 0 and index - 1 in free:
            degree += 1
        if column + 1 < columns and index + 1 in free:
            degree += 1
        if row > 0 and index - columns in free:
            degree += 1
        if row < 5 and index + columns in free:
            degree += 1
        max_free_degree = max(max_free_degree, degree)

    optimistic_components: list[int] = []
    for item, cap in cooling_caps:
        spec = COMPONENTS[item]
        if spec.kind != "vent" or cap <= 0:
            continue
        per_component = spec.self_vent + spec.side_vent * max_free_degree
        if per_component > 0:
            optimistic_components.extend([per_component] * min(cap, len(free_positions)))
    optimistic_components.sort(reverse=True)
    return sum(optimistic_components[:len(free_positions)])


def _layout_neighbors(index: int, columns: int, slots: int) -> tuple[int, ...]:
    row, column = divmod(index, columns)
    values: list[int] = []
    if column > 0:
        values.append(index - 1)
    if column + 1 < columns:
        values.append(index + 1)
    if row > 0:
        values.append(index - columns)
    if index + columns < slots:
        values.append(index + columns)
    return tuple(values)


def _maximum_flow(node_count: int, edges: list[tuple[int, int, int]], source: int, sink: int) -> int:
    """Small integer Dinic implementation used by the thermal relaxation."""
    graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(start: int, end: int, capacity: int) -> None:
        if capacity <= 0:
            return
        forward = [end, capacity, len(graph[end])]
        reverse = [start, 0, len(graph[start])]
        graph[start].append(forward)
        graph[end].append(reverse)

    for start, end, capacity in edges:
        add_edge(start, end, capacity)

    total = 0
    while True:
        level = [-1] * node_count
        level[source] = 0
        queue_values = [source]
        for node in queue_values:
            for end, capacity, _reverse in graph[node]:
                if capacity > 0 and level[end] < 0:
                    level[end] = level[node] + 1
                    queue_values.append(end)
        if level[sink] < 0:
            return total

        cursor = [0] * node_count

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                end, capacity, reverse_index = edge
                if capacity > 0 and level[end] == level[node] + 1:
                    pushed = send(end, min(amount, capacity))
                    if pushed:
                        edge[1] -= pushed
                        graph[end][reverse_index][1] += pushed
                        return pushed
                cursor[node] += 1
            return 0

        while True:
            pushed = send(source, 10**9)
            if not pushed:
                break
            total += pushed


def sustainable_heat_flow_upper_bound(layout: tuple[str, ...], columns: int) -> int:
    """Bound sustainable heat rejection while relaxing order and heat ratios.

    The network preserves every hard per-tick transfer/vent limit but allows
    heat to choose any favorable direction and ignores row-major ordering,
    percentage thresholds and finite capacities.  Every safe periodic run
    induces an average flow in this relaxed network.  Consequently a maximum
    flow below generated heat is a sound infeasibility proof; reaching the
    generated amount is only a necessary condition and still requires exact
    simulation.
    """
    slots = len(layout)
    generated = skeleton_heat_per_tick(power_skeleton(layout), columns)
    if generated <= 0:
        return 0

    hull = 0
    slot_node = lambda index: index + 1
    source = slots + 1
    sink = slots + 2
    edges: list[tuple[int, int, int]] = []
    infinite = generated

    for index, item in enumerate(layout):
        spec = COMPONENTS[item]
        node = slot_node(index)
        neighbors = _layout_neighbors(index, columns, slots)

        if spec.kind == "fuel":
            neighbor_pulses = sum(
                COMPONENTS[layout[neighbor]].kind in {"fuel", "reflector"}
                for neighbor in neighbors
            )
            pulses = spec.internal_pulses + neighbor_pulses
            heat = 2 * spec.rod_count * pulses * (pulses + 1)
            edges.append((source, node, heat))
            edges.append((node, hull, infinite))
            for neighbor in neighbors:
                if COMPONENTS[layout[neighbor]].accepts_heat:
                    edges.append((node, slot_node(neighbor), infinite))
            continue

        if spec.self_vent:
            edges.append((node, sink, spec.self_vent))
        if spec.hull_draw:
            edges.append((hull, node, spec.hull_draw))
        if spec.side_vent:
            for neighbor in neighbors:
                if COMPONENTS[layout[neighbor]].is_coolable:
                    edges.append((slot_node(neighbor), sink, spec.side_vent))
        if spec.kind == "exchanger":
            if spec.exchange_hull:
                # The official low-percentage branches use half of the side
                # exchange range even for hull exchange.  That can exceed the
                # nominal hull limit (e.g. 6 rather than 4 for the basic
                # exchanger), so the relaxation must include that larger hard
                # upper bound to avoid false infeasibility certificates.
                hull_limit = max(spec.exchange_hull, spec.exchange_side // 2, 1)
                edges.append((hull, node, hull_limit))
                edges.append((node, hull, hull_limit))
            if spec.exchange_side:
                for neighbor in neighbors:
                    if COMPONENTS[layout[neighbor]].accepts_heat:
                        neighbor_node = slot_node(neighbor)
                        edges.append((node, neighbor_node, spec.exchange_side))
                        edges.append((neighbor_node, node, spec.exchange_side))

    return _maximum_flow(slots + 3, edges, source, sink)


def theoretical_eu_per_tick(layout: tuple[str, ...], columns: int) -> float:
    """Return the power bound shared by all cooling completions of a skeleton."""
    return skeleton_eu_per_tick(power_skeleton(layout), columns)


def has_degrading_power_component(layout: tuple[str, ...], columns: int) -> bool:
    """Whether a finite reflector is pulsed and therefore cannot be Mark I-I."""
    for index, item in enumerate(layout):
        spec = COMPONENTS[item]
        if spec.kind != "reflector" or spec.max_damage <= 0:
            continue
        row, column = divmod(index, columns)
        neighbors = []
        if column > 0:
            neighbors.append(index - 1)
        if column + 1 < columns:
            neighbors.append(index + 1)
        if row > 0:
            neighbors.append(index - columns)
        if row < 5:
            neighbors.append(index + columns)
        if any(COMPONENTS[layout[neighbor]].kind == "fuel" for neighbor in neighbors):
            return True
    return False


def _allowed_and_caps(request: OptimizationRequest) -> tuple[list[str], dict[str, int]]:
    slots = request.columns * 6
    if request.fuel.mode == "total_rods":
        fuels = ["uranium_single", "uranium_dual", "uranium_quad"] if request.fuel.total_rods else []
        caps = {item: slots for item in fuels}
    else:
        fuel_limits = {
            "uranium_single": request.fuel.single,
            "uranium_dual": request.fuel.dual,
            "uranium_quad": request.fuel.quad,
        }
        fuels = [item for item, limit in fuel_limits.items() if limit > 0]
        caps = {item: min(fuel_limits[item], slots) for item in fuels}
    nonfuel = [item for item, limit in request.component_limits.items() if limit > 0]
    # Every enumerated layout must contain fuel, so no individual non-fuel
    # component can occupy more than ``slots - 1`` cells.  Normalizing here
    # keeps equivalent oversized requests on the same DP/counting state space.
    max_nonfuel = max(0, slots - 1)
    caps.update({
        item: min(request.component_limits[item], max_nonfuel)
        for item in nonfuel
    })
    return [*fuels, *nonfuel], caps


def _exhaustive_shards(
    request: OptimizationRequest,
    *,
    power_only: bool = False,
    target_shards: int | None = None,
) -> list[tuple[tuple[int, str], ...]]:
    """Split the labelled layout space into disjoint fixed-cell prefixes.

    Without a target the legacy two-centre-cell split is retained.  Mark I
    search supplies a target based on CPU workers and keeps expanding cells
    from the centre outwards until there are enough independent tasks.  When
    cooling items are included, completions of the same power skeleton land in
    different shards and can therefore use the whole process pool.
    """
    columns = request.columns
    allowed, caps = _allowed_and_caps(request)
    if power_only:
        allowed = [item for item in allowed if COMPONENTS[item].kind in {"fuel", "reflector"}]
    empty_value = POWER_EMPTY if power_only else "empty"
    values = [empty_value, *allowed]
    slots = columns * 6
    if target_shards is None:
        positions = (2 * columns + columns // 2, 3 * columns + columns // 2)
    else:
        center_row = 2.5
        center_column = (columns - 1) / 2
        positions = tuple(sorted(
            range(slots),
            key=lambda index: (
                abs(index // columns - center_row) + abs(index % columns - center_column),
                index,
            ),
        ))

    prefixes: list[tuple[tuple[int, str], ...]] = [()]
    required = max(1, target_shards or 1)
    minimum_depth = 2 if target_shards is None else 1
    for depth, position in enumerate(positions, start=1):
        expanded: list[tuple[tuple[int, str], ...]] = []
        for prefix in prefixes:
            used: dict[str, int] = {}
            rods = 0
            for _, item in prefix:
                if item in {"empty", POWER_EMPTY}:
                    continue
                used[item] = used.get(item, 0) + 1
                rods += COMPONENTS[item].rod_count
            for item in values:
                if item not in {"empty", POWER_EMPTY} and used.get(item, 0) >= caps[item]:
                    continue
                next_rods = rods + (
                    0 if item == POWER_EMPTY else COMPONENTS[item].rod_count
                )
                if request.fuel.mode == "total_rods" and next_rods > request.fuel.total_rods:
                    continue
                expanded.append((*prefix, (position, item)))
        prefixes = expanded
        if depth >= minimum_depth and len(prefixes) >= required:
            break

    # Prefer fuel-bearing assignments so leaderboards begin producing useful
    # results immediately. Every shard is still evaluated in full.
    prefixes.sort(key=lambda shard: (
        not any(
            item != POWER_EMPTY and COMPONENTS[item].rod_count
            for _, item in shard
        ),
        shard,
    ))
    return prefixes


def estimate_exhaustive_space(request: OptimizationRequest) -> int:
    """Count inventory-valid labelled layouts (before symmetry reduction)."""
    slots = request.columns * 6
    types: list[tuple[int, int, bool]] = []  # cap, rod cost, is fuel
    if request.fuel.mode == "separate":
        types.extend((cap, 0, True) for cap in (request.fuel.single, request.fuel.dual, request.fuel.quad) if cap > 0)
    elif request.fuel.total_rods > 0:
        types.extend((request.fuel.total_rods // rods, rods, True) for rods in (1, 2, 4))
    types.extend((cap, 0, False) for cap in request.component_limits.values() if cap > 0)

    # dp[(occupied slots, used rods, has fuel)] = number of ways to choose labelled positions.
    dp: dict[tuple[int, int, bool], int] = {(0, 0, False): 1}
    for cap, rod_cost, is_fuel in types:
        next_dp: dict[tuple[int, int, bool], int] = {}
        for (used, rods, has_fuel), ways in dp.items():
            for count in range(min(cap, slots - used) + 1):
                next_rods = rods + count * rod_cost
                if request.fuel.mode == "total_rods" and next_rods > request.fuel.total_rods:
                    break
                key = (used + count, next_rods, has_fuel or (is_fuel and count > 0))
                next_dp[key] = next_dp.get(key, 0) + ways * math.comb(slots - used, count)
        dp = next_dp
    return sum(ways for (_, _, has_fuel), ways in dp.items() if has_fuel)


@lru_cache(maxsize=16_384)
def count_cooling_completions(free_slots: int, caps: tuple[int, ...]) -> int:
    """Count labelled cooling assignments for a fixed power skeleton."""
    dp: dict[int, int] = {0: 1}
    for cap in caps:
        next_dp: dict[int, int] = {}
        for used, ways in dp.items():
            for count in range(min(cap, free_slots - used) + 1):
                occupied = used + count
                next_dp[occupied] = (
                    next_dp.get(occupied, 0)
                    + ways * math.comb(free_slots - used, count)
                )
        dp = next_dp
    return sum(dp.values())


def _evaluate_layout_uncached(
    layout: tuple[str, ...],
    columns: int,
    max_reactor_ticks: int,
    cancel_check=None,
) -> CandidateResult:
    """Process-safe full candidate evaluation used by heuristic worker processes."""
    certificate = prove_simple_fixed_point(layout, columns, max_reactor_ticks)
    if certificate is not None:
        return certificate
    simulator = ReactorSimulator(Layout(columns=columns, initial_hull_heat=0, slots=list(layout)))
    run = simulator.simulate(SimulationOptions(
        max_game_ticks=max_reactor_ticks * 20,
        auto_refuel=True,
        stop_on_stable=True,
        record_components=False,
        record_history=False,
        cancel_check=cancel_check,
    ))
    safe_ticks = run.summary.first_intervention_tick or run.summary.game_ticks
    return CandidateResult(
        layout=layout,
        mark=run.summary.mark or "未分类",
        average_eu_per_tick=run.summary.average_eu_per_tick,
        total_eu=run.summary.average_eu_per_tick * safe_ticks,
        safe_game_ticks=safe_ticks,
        safety_margin=1.0 - run.summary.peak_hull_heat / run.summary.max_hull_heat,
        component_count=sum(item != "empty" for item in layout),
        canonical=canonical_layout(layout, columns),
    )


@lru_cache(maxsize=8_192)
def prove_simple_fixed_point(
    layout: tuple[str, ...],
    columns: int,
    max_reactor_ticks: int,
) -> CandidateResult | None:
    """Prove a simple complete layout reaches a safe period-one thermal state.

    The verifier executes the production transition exactly; it does not use
    nominal cooling sums.  It is deliberately restricted to layouts without
    exchangers or finite heat stores and returns ``None`` whenever it cannot
    prove a fixed point quickly.  ``None`` therefore means unknown, never
    infeasible, and callers must retain the normal simulator fallback.
    """
    stable_reactor_ticks = 2 * FUEL_CYCLE_REACTOR_TICKS
    if max_reactor_ticks < stable_reactor_ticks:
        return None
    simple_kinds = {"empty", "fuel", "vent", "plating", "reflector"}
    if any(COMPONENTS[item].kind not in simple_kinds for item in layout):
        return None

    simulator = ReactorSimulator(Layout(columns=columns, initial_hull_heat=0, slots=list(layout)))
    previous_state = simulator.state_signature(include_fuel_damage=False)
    eu_per_tick = 0.0
    for _ in range(4):
        eu_per_tick, _generated, _vented = simulator.step(auto_refuel=True)
        if (
            simulator.first_critical_tick is not None
            or simulator.first_component_break_tick is not None
            or simulator.meltdown_tick is not None
        ):
            return None
        current_state = simulator.state_signature(include_fuel_damage=False)
        if current_state == previous_state:
            safe_game_ticks = stable_reactor_ticks * 20
            mark = classify_mark(None, None, True, simulator.uses_single_use_coolant)
            return CandidateResult(
                layout=layout,
                mark=mark or "未分类",
                average_eu_per_tick=eu_per_tick,
                total_eu=eu_per_tick * safe_game_ticks,
                safe_game_ticks=safe_game_ticks,
                safety_margin=1.0 - simulator.peak_hull_heat / simulator.max_hull_heat,
                component_count=sum(item != "empty" for item in layout),
                canonical=canonical_layout(layout, columns),
            )
        previous_state = current_state
    return None


@lru_cache(maxsize=8_192)
def _fixed_point_certificate(
    layout: tuple[str, ...],
    columns: int,
    max_reactor_ticks: int,
) -> CandidateResult:
    """Process-local bounded cache for previously proved simulation results."""
    return _evaluate_layout_uncached(layout, columns, max_reactor_ticks)


def evaluate_layout(
    layout: tuple[str, ...],
    columns: int,
    max_reactor_ticks: int,
    cancel_check=None,
    use_certificate: bool = True,
) -> CandidateResult:
    """Evaluate a layout, reusing a bounded certificate when cancellation is absent."""
    if cancel_check is None and use_certificate:
        return _fixed_point_certificate(layout, columns, max_reactor_ticks)
    return _evaluate_layout_uncached(layout, columns, max_reactor_ticks, cancel_check)


def candidate_from_cuda_fixed_point(
    layout: tuple[str, ...],
    columns: int,
    certificate: CudaFixedPointCertificate,
) -> CandidateResult:
    """Convert an exact CUDA fixed-point proof into the public candidate form."""
    power = certificate.average_eu_per_tick
    return CandidateResult(
        layout=layout,
        mark="Mark I-I",
        average_eu_per_tick=power,
        total_eu=power * certificate.safe_game_ticks,
        safe_game_ticks=certificate.safe_game_ticks,
        safety_margin=(
            1.0 - certificate.peak_hull_heat / certificate.max_hull_heat
        ),
        component_count=sum(item != "empty" for item in layout),
        canonical=canonical_layout(layout, columns),
    )


def candidate_from_cuda_simulation(
    layout: tuple[str, ...],
    columns: int,
    simulation: CudaSimulationResult,
) -> CandidateResult:
    """Convert a complete per-thread CUDA simulation into a candidate."""
    power = simulation.average_eu_per_tick
    return CandidateResult(
        layout=layout,
        mark=simulation.mark or "未分类",
        average_eu_per_tick=power,
        total_eu=power * simulation.safe_game_ticks,
        safe_game_ticks=simulation.safe_game_ticks,
        safety_margin=(
            1.0 - simulation.peak_hull_heat / simulation.max_hull_heat
        ),
        component_count=sum(item != "empty" for item in layout),
        canonical=canonical_layout(layout, columns),
    )


def _rank_candidates(
    values: list[CandidateResult],
    limit: int = 10,
) -> list[CandidateResult]:
    board: dict[str, CandidateResult] = {}
    for result in values:
        previous = board.get(result.canonical)
        if previous is None or result.score() > previous.score():
            board[result.canonical] = result
    ordered = sorted(board.values(), key=lambda item: item.canonical)
    ordered.sort(key=lambda item: item.score(), reverse=True)
    return ordered[:limit]


def construct_simple_cooling_candidates(
    base_layout: tuple[str, ...],
    columns: int,
    free_positions: tuple[int, ...],
    cooling_remaining: dict[str, int],
    generated_heat: int,
    max_reactor_ticks: int,
    target_count: int,
) -> list[CandidateResult]:
    """Construct a few direct-vent layouts and retain only proved fixed points.

    This is a witness finder, not an infeasibility solver.  Returning an empty
    list never authorizes pruning; the exhaustive cooling generator remains
    the mandatory fallback.
    """
    if target_count <= 0:
        return []

    fuel_heat: dict[int, int] = {}
    for index, item in enumerate(base_layout):
        spec = COMPONENTS[item]
        if spec.kind != "fuel":
            continue
        neighbors = _layout_neighbors(index, columns, len(base_layout))
        pulses = spec.internal_pulses + sum(
            COMPONENTS[base_layout[neighbor]].kind in {"fuel", "reflector"}
            for neighbor in neighbors
        )
        fuel_heat[index] = 2 * spec.rod_count * pulses * (pulses + 1)

    adjacency_heat = {
        position: sum(
            heat
            for fuel_position, heat in fuel_heat.items()
            if position in _layout_neighbors(fuel_position, columns, len(base_layout))
        )
        for position in free_positions
    }
    position_orders = [
        tuple(sorted(free_positions, key=lambda p: (-adjacency_heat[p], p))),
        tuple(sorted(free_positions, key=lambda p: (-adjacency_heat[p], -p))),
        tuple(sorted(free_positions, key=lambda p: (adjacency_heat[p], p))),
        tuple(sorted(free_positions, reverse=True)),
    ]

    fixed_nominal_vent = sum(
        COMPONENTS[item].self_vent
        for item in base_layout
        if COMPONENTS[item].kind == "vent"
    )
    attempted: set[tuple[str, ...]] = set()
    proved_by_canonical: dict[str, CandidateResult] = {}
    attempt_budget = 12

    def try_layout(layout: tuple[str, ...]) -> None:
        nonlocal attempt_budget
        if attempt_budget <= 0 or layout in attempted:
            return
        attempted.add(layout)
        attempt_budget -= 1
        result = prove_simple_fixed_point(layout, columns, max_reactor_ticks)
        if result is not None:
            proved_by_canonical.setdefault(result.canonical, result)

    if fixed_nominal_vent >= generated_heat:
        try_layout(base_layout)

    vent_instances = [
        item
        for item, count in cooling_remaining.items()
        if COMPONENTS[item].kind == "vent" and COMPONENTS[item].self_vent > 0
        for _ in range(min(count, len(free_positions)))
    ]
    if not vent_instances or len(proved_by_canonical) >= target_count:
        return list(proved_by_canonical.values())[:target_count]
    component_orders = [
        tuple(sorted(vent_instances, key=lambda item: (-COMPONENTS[item].self_vent, item))),
        tuple(sorted(
            vent_instances,
            key=lambda item: (
                COMPONENTS[item].hull_draw == 0,
                -min(COMPONENTS[item].self_vent, COMPONENTS[item].hull_draw or COMPONENTS[item].self_vent),
                item,
            ),
        )),
    ]

    for positions in position_orders:
        for components in component_orders:
            if attempt_budget <= 0 or len(proved_by_canonical) >= target_count:
                break
            layout = list(base_layout)
            nominal_vent = fixed_nominal_vent
            for position, item in zip(positions, components, strict=False):
                layout[position] = item
                nominal_vent += COMPONENTS[item].self_vent
                if nominal_vent >= generated_heat:
                    try_layout(tuple(layout))
                if attempt_budget <= 0 or len(proved_by_canonical) >= target_count:
                    break
        if attempt_budget <= 0 or len(proved_by_canonical) >= target_count:
            break
    return list(proved_by_canonical.values())[:target_count]


def _run_mark_i_two_level_shard(
    request: OptimizationRequest,
    shard_id: int,
    fixed_items: tuple[tuple[int, str], ...],
    progress_queue,
    cancel_event,
    shared_power_floor=None,
) -> dict:
    """Enumerate power skeletons first, then their cooling completions."""
    gpu_evaluator: CudaFixedPointEvaluator | None = None
    gpu_full_simulator: CudaFullSimulator | None = None
    gpu_error: str | None = None
    if request.accelerator == "cuda_full":
        try:
            gpu_full_simulator = CudaFullSimulator(
                ticks_per_launch=request.gpu_ticks_per_launch
            )
        except Exception as exc:
            gpu_error = str(exc)
            raise
    elif request.accelerator != "cpu":
        try:
            gpu_evaluator = CudaFixedPointEvaluator()
        except Exception as exc:
            gpu_error = str(exc)
            if request.accelerator == "cuda":
                raise
    allowed, caps = _allowed_and_caps(request)
    power_items = [
        item for item in allowed
        if COMPONENTS[item].kind in {"fuel", "reflector"}
    ]
    power_items.sort(key=lambda item: (
        COMPONENTS[item].kind != "fuel",
        -COMPONENTS[item].rod_count,
        item,
    ))
    cooling_items = [
        item for item in allowed
        if COMPONENTS[item].kind not in {"fuel", "reflector"}
    ]
    cooling_items.sort()
    initial_cooling_caps = tuple(caps[item] for item in cooling_items)
    cooling_cap_items = tuple(zip(cooling_items, initial_cooling_caps, strict=True))
    cooling_remaining = dict(zip(cooling_items, initial_cooling_caps, strict=True))
    fixed = {
        position: item
        for position, item in fixed_items
        if item != POWER_EMPTY
    }
    slots = request.columns * 6
    power_caps = tuple(caps[item] for item in power_items)
    fixed_skeleton = ["empty"] * slots
    for position, item in fixed_items:
        if item in {"empty", POWER_EMPTY}:
            continue
        if COMPONENTS[item].kind in {"fuel", "reflector"}:
            fixed_skeleton[position] = item
        else:
            cooling_remaining[item] -= 1
    cooling_caps = tuple(cooling_remaining[item] for item in cooling_items)
    power_table = SkeletonPowerTable(
        columns=request.columns,
        power_items=tuple(power_items),
        power_caps=power_caps,
        fixed_items=fixed_items,
        total_rods=(
            request.fuel.total_rods
            if request.fuel.mode == "total_rods"
            else None
        ),
    )

    checked = 0
    pruned = 0
    evaluated = 0
    gpu_certified = 0
    gpu_fallback = 0
    gpu_batches = 0
    gpu_full_simulated = 0
    visits = 0
    last_report = time.monotonic()
    cancelled = False
    cancel_cache = False
    last_cancel_check = 0.0
    shared_floor_cache = -1.0
    last_floor_check = 0.0
    boards: dict[str, list[CandidateResult]] = {"I": []}
    precertified_layouts: set[tuple[str, ...]] = set()
    pending_layouts: list[tuple[str, ...]] = []

    def cancellation_requested() -> bool:
        nonlocal cancel_cache, last_cancel_check
        now = time.monotonic()
        if now - last_cancel_check >= 0.1:
            cancel_cache = cancel_event.is_set()
            last_cancel_check = now
        return cancel_cache

    def current_power_floor() -> float:
        nonlocal shared_floor_cache, last_floor_check
        local_board = boards["I"]
        local_floor = (
            local_board[-1].average_eu_per_tick
            if len(local_board) >= request.result_limit
            else -1.0
        )
        now = time.monotonic()
        if shared_power_floor is not None and now - last_floor_check >= 0.1:
            shared_floor_cache = shared_power_floor.value
            last_floor_check = now
        return max(local_floor, shared_floor_cache)

    def cannot_enter_board(power_upper_bound: float, floor: float) -> bool:
        """Return whether a branch cannot improve the requested leaderboard.

        Top-1 only needs one globally optimal representative, so equality is
        already dominated after the first candidate.  For top-K, equal-power
        branches remain eligible because they can fill or reorder tied places.
        """
        if floor < 0:
            return False
        return (
            power_upper_bound <= floor
            if request.result_limit == 1
            else power_upper_bound < floor
        )

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 0.25:
            progress_queue.put((
                "progress", shard_id, checked, pruned, evaluated,
                gpu_certified, gpu_fallback, gpu_batches, gpu_full_simulated,
            ))
            last_report = now

    def accept_result(result: CandidateResult) -> None:
        previous = boards["I"]
        ranked = _rank_candidates([*previous, result], request.result_limit)
        if ranked != previous:
            boards["I"] = ranked
            progress_queue.put(("candidate", result))

    def flush_pending() -> None:
        nonlocal evaluated, cancelled
        nonlocal gpu_certified, gpu_fallback, gpu_batches, gpu_full_simulated
        if not pending_layouts or cancelled:
            return
        batch = pending_layouts[:]
        pending_layouts.clear()
        full_results: list[CudaSimulationResult] | None = None
        if gpu_full_simulator is not None:
            full_results = gpu_full_simulator.simulate(
                batch,
                request.columns,
                request.max_reactor_ticks,
                cancel_check=cancellation_requested,
            )
            if full_results is None:
                cancelled = True
                return
            certificates = [None] * len(batch)
            gpu_batches += 1
        elif gpu_evaluator is not None:
            certificates = gpu_evaluator.certify(
                batch,
                request.columns,
                request.max_reactor_ticks,
            )
            gpu_batches += 1
        else:
            certificates = [None] * len(batch)
        for index, (raw, certificate) in enumerate(zip(batch, certificates, strict=True)):
            if cancellation_requested():
                cancelled = True
                break
            if full_results is not None:
                gpu_full_simulated += 1
                result = candidate_from_cuda_simulation(
                    raw,
                    request.columns,
                    full_results[index],
                )
            elif certificate is None:
                if gpu_evaluator is not None:
                    gpu_fallback += 1
                result = evaluate_layout(
                    raw,
                    request.columns,
                    request.max_reactor_ticks,
                    cancel_check=cancellation_requested,
                )
            else:
                gpu_certified += 1
                result = candidate_from_cuda_fixed_point(
                    raw,
                    request.columns,
                    certificate,
                )
            if cancellation_requested():
                cancelled = True
                break
            evaluated += 1
            if mark_family(result.mark) == "I":
                accept_result(result)
        report()

    @lru_cache(maxsize=None)
    def count_power_subtree(
        step: int,
        remaining: tuple[int, ...],
        used_power: int,
    ) -> int:
        """Count full labelled layouts below one power-table prefix exactly."""
        if step == slots:
            if not power_table.has_fuel(remaining):
                return 0
            nonfixed_power = used_power - power_table.fixed_power_count
            free_slots = slots - power_table.fixed_cell_count - nonfixed_power
            return count_cooling_completions(free_slots, cooling_caps)
        total = 0
        for code in power_table.allowed_codes(step, remaining):
            next_remaining = power_table.consume(code, remaining)
            total += count_power_subtree(
                step + 1,
                next_remaining,
                used_power + int(code != 0),
            )
        return total

    minimum_fuel_heat = min(
        (
            2
            * COMPONENTS[item].rod_count
            * COMPONENTS[item].internal_pulses
            * (COMPONENTS[item].internal_pulses + 1)
            for item in power_items
            if COMPONENTS[item].kind == "fuel"
        ),
        default=0,
    )
    global_vent_upper = sustainable_vent_upper_bound(
        tuple(fixed_skeleton),
        request.columns,
        cooling_cap_items,
    )
    if minimum_fuel_heat and global_vent_upper < minimum_fuel_heat:
        count = count_power_subtree(0, power_table.initial_remaining, 0)
        checked += count
        pruned += count
        report(force=True)
        return {
            "shard_id": shard_id,
            "checked": checked,
            "pruned": pruned,
            "evaluated": evaluated,
            "boards": boards,
            "cancelled": cancelled,
            "skeleton_table_states": 0,
            "skeleton_table_cache_hit": False,
            "gpu_certified": gpu_certified,
            "gpu_fallback": gpu_fallback,
            "gpu_batches": gpu_batches,
            "gpu_full_simulated": gpu_full_simulated,
            "gpu_error": gpu_error,
        }

    def generate_cooling(
        layout: list[str],
        free_positions: tuple[int, ...],
        offset: int,
        skeleton_power: float,
        skeleton_heat: int,
    ) -> None:
        nonlocal checked, pruned, evaluated, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancel_event.is_set():
            cancelled = True
            return
        if cancelled:
            return

        floor = current_power_floor()
        if cannot_enter_board(skeleton_power, floor):
            count = count_cooling_completions(
                len(free_positions) - offset,
                tuple(cooling_remaining[item] for item in cooling_items),
            )
            checked += count
            pruned += count
            report()
            return

        if offset == len(free_positions):
            checked += 1
            raw = tuple(layout)
            if raw in precertified_layouts:
                precertified_layouts.remove(raw)
                report()
                return
            if (
                sustainable_heat_flow_upper_bound(raw, request.columns)
                < skeleton_heat
            ):
                pruned += 1
                report()
                return
            pending_layouts.append(raw)
            batch_size = (
                request.gpu_exhaustive_batch_size
                if gpu_evaluator is not None or gpu_full_simulator is not None
                else 1
            )
            if len(pending_layouts) >= batch_size:
                flush_pending()
            return


        position = free_positions[offset]
        for item in ["empty", *cooling_items]:
            if item == "empty":
                layout[position] = "empty"
                generate_cooling(layout, free_positions, offset + 1, skeleton_power, skeleton_heat)
            elif cooling_remaining[item] > 0:
                cooling_remaining[item] -= 1
                layout[position] = item
                generate_cooling(layout, free_positions, offset + 1, skeleton_power, skeleton_heat)
                cooling_remaining[item] += 1
            if cancelled:
                break
        layout[position] = "empty"

    def finish_skeleton(raw_skeleton: tuple[str, ...]) -> None:
        nonlocal checked, pruned, evaluated
        free_positions = tuple(
            index
            for index, item in enumerate(raw_skeleton)
            if item == "empty" and index not in fixed
        )
        completion_count = count_cooling_completions(len(free_positions), cooling_caps)
        skeleton_power = skeleton_eu_per_tick(raw_skeleton, request.columns)
        skeleton_heat = skeleton_heat_per_tick(raw_skeleton, request.columns)
        vent_upper = sustainable_vent_upper_bound(
            raw_skeleton,
            request.columns,
            cooling_cap_items,
        )
        if (
            has_degrading_power_component(raw_skeleton, request.columns)
            or skeleton_heat > vent_upper
            or cannot_enter_board(skeleton_power, current_power_floor())
        ):
            checked += completion_count
            pruned += completion_count
            report()
            return
        layout = list(raw_skeleton)
        for position, item in fixed.items():
            if item != "empty" and COMPONENTS[item].kind not in {"fuel", "reflector"}:
                layout[position] = item
        constructed = construct_simple_cooling_candidates(
            tuple(layout),
            request.columns,
            free_positions,
            cooling_remaining,
            skeleton_heat,
            request.max_reactor_ticks,
            request.result_limit,
        )
        for result in constructed:
            evaluated += 1
            precertified_layouts.add(result.layout)
            accept_result(result)
        if request.result_limit == 1 and constructed:
            # One proved witness realizes this skeleton's exact power.  Every
            # other cooling completion can only tie it in a Top-1 request.
            checked += completion_count
            pruned += completion_count - 1
            precertified_layouts.discard(constructed[0].layout)
            report()
            return
        generate_cooling(layout, free_positions, 0, skeleton_power, skeleton_heat)
        flush_pending()

    try:
        power_table.build(cancel_check=cancellation_requested)
    except InterruptedError:
        cancelled = True

    heap: list[tuple[int, int, SkeletonSearchNode]] = []
    serial = 0
    root = None if cancelled else power_table.root()
    if root is not None:
        heapq.heappush(heap, (-root.bound, serial, root))
    while heap and not cancelled:
        _negative_bound, _serial, node = heapq.heappop(heap)
        visits += 1
        if visits % 4096 == 0 and cancellation_requested():
            cancelled = True
            break
        floor = current_power_floor()
        if cannot_enter_board(node.bound, floor):
            pending = [node, *(entry[2] for entry in heap)]
            count = sum(
                count_power_subtree(
                    item.step,
                    item.remaining,
                    item.power_components,
                )
                for item in pending
            )
            checked += count
            pruned += count
            heap.clear()
            report()
            break
        if node.step == slots:
            finish_skeleton(power_table.materialize(node.choices))
            continue
        for child in power_table.expand(node):
            serial += 1
            heapq.heappush(heap, (-child.bound, serial, child))

    report(force=True)
    flush_pending()
    return {
        "shard_id": shard_id,
        "checked": checked,
        "pruned": pruned,
        "evaluated": evaluated,
        "boards": boards,
        "cancelled": cancelled,
        "skeleton_table_states": len(power_table.memo),
        "skeleton_table_cache_hit": power_table.loaded_from_disk,
        "gpu_certified": gpu_certified,
        "gpu_fallback": gpu_fallback,
        "gpu_batches": gpu_batches,
        "gpu_full_simulated": gpu_full_simulated,
        "gpu_error": gpu_error,
    }


def _run_exhaustive_shard(
    request_data: dict,
    shard_id: int,
    fixed_items: tuple[tuple[int, str], ...],
    progress_queue,
    cancel_event,
    shared_power_floor=None,
) -> dict:
    """Process worker for one disjoint exhaustive-search shard."""
    request = OptimizationRequest.model_validate(request_data)
    if request.marks == ["I"]:
        return _run_mark_i_two_level_shard(
            request,
            shard_id,
            fixed_items,
            progress_queue,
            cancel_event,
            shared_power_floor,
        )
    allowed, remaining = _allowed_and_caps(request)
    gpu_evaluator: CudaFixedPointEvaluator | None = None
    gpu_full_simulator: CudaFullSimulator | None = None
    gpu_error: str | None = None
    if request.accelerator == "cuda_full":
        try:
            gpu_full_simulator = CudaFullSimulator(
                ticks_per_launch=request.gpu_ticks_per_launch
            )
        except Exception as exc:
            gpu_error = str(exc)
            raise
    elif request.accelerator != "cpu":
        try:
            gpu_evaluator = CudaFixedPointEvaluator()
        except Exception as exc:
            gpu_error = str(exc)
            if request.accelerator == "cuda":
                raise
    fixed = dict(fixed_items)
    slots = request.columns * 6
    layout = ["empty"] * slots
    rods = 0
    has_fuel = False
    for position, item in fixed_items:
        layout[position] = item
        if item != "empty":
            remaining[item] -= 1
            rods += COMPONENTS[item].rod_count
            has_fuel = has_fuel or COMPONENTS[item].rod_count > 0

    checked = 0
    pruned = 0
    evaluated = 0
    gpu_certified = 0
    gpu_fallback = 0
    gpu_batches = 0
    gpu_full_simulated = 0
    visits = 0
    last_report = time.monotonic()
    cancelled = False
    cancel_cache = False
    last_cancel_check = 0.0
    boards: dict[str, list[CandidateResult]] = {mark: [] for mark in request.marks}
    pending_layouts: list[tuple[str, ...]] = []

    def cancellation_requested() -> bool:
        nonlocal cancel_cache, last_cancel_check
        now = time.monotonic()
        if now - last_cancel_check >= 0.1:
            cancel_cache = cancel_event.is_set()
            last_cancel_check = now
        return cancel_cache

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 0.25:
            progress_queue.put((
                "progress", shard_id, checked, pruned, evaluated,
                gpu_certified, gpu_fallback, gpu_batches, gpu_full_simulated,
            ))
            last_report = now

    def accept_result(result: CandidateResult) -> None:
        family = mark_family(result.mark)
        if family not in boards:
            return
        previous = boards[family]
        ranked = _rank_candidates([*previous, result], request.result_limit)
        if ranked != previous:
            boards[family] = ranked
            progress_queue.put(("candidate", result))

    def flush_pending() -> None:
        nonlocal checked, evaluated, cancelled
        nonlocal gpu_certified, gpu_fallback, gpu_batches, gpu_full_simulated
        if not pending_layouts or cancelled:
            return
        batch = pending_layouts[:]
        pending_layouts.clear()
        full_results: list[CudaSimulationResult] | None = None
        certificates: list[CudaFixedPointCertificate | None]
        if gpu_full_simulator is not None:
            full_results = gpu_full_simulator.simulate(
                batch,
                request.columns,
                request.max_reactor_ticks,
                cancel_check=cancellation_requested,
            )
            if full_results is None:
                cancelled = True
                return
            certificates = [None] * len(batch)
            gpu_batches += 1
        elif gpu_evaluator is not None:
            certificates = gpu_evaluator.certify(
                batch,
                request.columns,
                request.max_reactor_ticks,
            )
            gpu_batches += 1
        else:
            certificates = [None] * len(batch)
        for index, (raw, certificate) in enumerate(zip(batch, certificates, strict=True)):
            if cancellation_requested():
                cancelled = True
                break
            checked += 1
            if full_results is not None:
                gpu_full_simulated += 1
                result = candidate_from_cuda_simulation(
                    raw,
                    request.columns,
                    full_results[index],
                )
            elif certificate is None:
                if gpu_evaluator is not None:
                    gpu_fallback += 1
                result = evaluate_layout(
                    raw,
                    request.columns,
                    request.max_reactor_ticks,
                    cancel_check=cancellation_requested,
                )
            else:
                gpu_certified += 1
                result = candidate_from_cuda_fixed_point(
                    raw,
                    request.columns,
                    certificate,
                )
            if cancellation_requested():
                cancelled = True
                break
            evaluated += 1
            accept_result(result)
        report()

    def generate(position: int, current_rods: int, current_has_fuel: bool) -> None:
        nonlocal checked, pruned, evaluated, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancel_event.is_set():
            cancelled = True
            return
        if position == slots:
            if not current_has_fuel:
                return
            pending_layouts.append(tuple(layout))
            batch_size = (
                request.gpu_exhaustive_batch_size
                if gpu_evaluator is not None or gpu_full_simulator is not None
                else 1
            )
            if len(pending_layouts) >= batch_size:
                flush_pending()
            return
        if cancelled:
            return
        if position in fixed:
            generate(position + 1, current_rods, current_has_fuel)
            return

        for item in ["empty", *allowed]:
            if item == "empty":
                layout[position] = "empty"
                generate(position + 1, current_rods, current_has_fuel)
                continue
            if cancelled or remaining[item] <= 0:
                continue
            rod_cost = COMPONENTS[item].rod_count
            if request.fuel.mode == "total_rods" and current_rods + rod_cost > request.fuel.total_rods:
                continue
            remaining[item] -= 1
            layout[position] = item
            generate(position + 1, current_rods + rod_cost, current_has_fuel or rod_cost > 0)
            remaining[item] += 1
        layout[position] = "empty"

    generate(0, rods, has_fuel)
    flush_pending()
    report(force=True)
    return {
        "shard_id": shard_id,
        "checked": checked,
        "pruned": pruned,
        "evaluated": evaluated,
        "boards": boards,
        "cancelled": cancelled,
        "gpu_certified": gpu_certified,
        "gpu_fallback": gpu_fallback,
        "gpu_batches": gpu_batches,
        "gpu_full_simulated": gpu_full_simulated,
        "gpu_error": gpu_error,
    }


class OptimizationJob:
    def __init__(self, request: OptimizationRequest):
        self.id = uuid.uuid4().hex
        self.request = request
        self.status = "queued"
        self.progress = 0.0
        self.evaluated = 0
        self.checked = 0
        self.pruned = 0
        self.skeleton_table_states = 0
        self.skeleton_table_cache_hits = 0
        self.accelerator = "cpu"
        self.accelerator_detail: str | None = None
        self.accelerator_fallback_reason: str | None = None
        self.gpu_screened = 0
        self.gpu_batches = 0
        self.gpu_exhaustive_certified = 0
        self.gpu_exhaustive_fallback = 0
        self.gpu_full_simulated = 0
        self.generation = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.message = "等待开始"
        self.error: str | None = None
        self.proven_global = False
        self.exhaustive_estimate = estimate_exhaustive_space(request) if request.solver == "exhaustive" else None
        self.cancel_event = threading.Event()
        self.process_cancel_event = None
        self._heuristic_cache: dict[tuple[str, ...], CandidateResult | None] = {}
        self.leaderboards: dict[str, list[CandidateResult]] = {mark: [] for mark in request.marks}

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "evaluated": self.evaluated,
            "checked": self.checked,
            "pruned": self.pruned,
            "generation": self.generation,
            "message": self.message,
            "error": self.error,
            "proven_global": self.proven_global,
            "estimate": str(self.exhaustive_estimate) if self.exhaustive_estimate is not None else None,
            "cpu_workers": self.request.cpu_workers,
            "accelerator_requested": self.request.accelerator,
            "accelerator": self.accelerator,
            "accelerator_detail": self.accelerator_detail,
            "accelerator_fallback_reason": self.accelerator_fallback_reason,
            "gpu_screened": self.gpu_screened,
            "gpu_batches": self.gpu_batches,
            "gpu_exhaustive_certified": self.gpu_exhaustive_certified,
            "gpu_exhaustive_fallback": self.gpu_exhaustive_fallback,
            "gpu_full_simulated": self.gpu_full_simulated,
            "result_limit": self.request.result_limit,
            "skeleton_table_states": self.skeleton_table_states,
            "skeleton_table_cache_hits": self.skeleton_table_cache_hits,
            "elapsed_seconds": (self.finished_at or time.time()) - self.started_at if self.started_at else 0,
            "leaderboards": {
                mark: [candidate.public_dict(self.request.columns) for candidate in values]
                for mark, values in self.leaderboards.items()
            },
        }

    def _fuel_allowed(self) -> list[str]:
        if self.request.fuel.mode == "total_rods":
            return ["uranium_single", "uranium_dual", "uranium_quad"] if self.request.fuel.total_rods > 0 else []
        result = []
        if self.request.fuel.single:
            result.append("uranium_single")
        if self.request.fuel.dual:
            result.append("uranium_dual")
        if self.request.fuel.quad:
            result.append("uranium_quad")
        return result

    def _within_limits(self, layout: tuple[str, ...]) -> bool:
        counts: dict[str, int] = {}
        for item in layout:
            counts[item] = counts.get(item, 0) + 1
        if self.request.fuel.mode == "total_rods":
            rods = sum(COMPONENTS[item].rod_count for item in layout)
            if rods > self.request.fuel.total_rods:
                return False
        else:
            if counts.get("uranium_single", 0) > self.request.fuel.single:
                return False
            if counts.get("uranium_dual", 0) > self.request.fuel.dual:
                return False
            if counts.get("uranium_quad", 0) > self.request.fuel.quad:
                return False
        return all(counts.get(item, 0) <= limit for item, limit in self.request.component_limits.items()) and all(
            item in {"empty", "uranium_single", "uranium_dual", "uranium_quad"} or item in self.request.component_limits
            for item in layout
        )

    def _random_layout(self, rng: random.Random) -> tuple[str, ...]:
        slots = self.request.columns * 6
        values = ["empty"]
        values.extend(self._fuel_allowed())
        values.extend(item for item, limit in self.request.component_limits.items() if limit > 0)
        if len(values) == 1:
            return tuple("empty" for _ in range(slots))
        for _ in range(500):
            layout = tuple(rng.choice(values) if rng.random() < 0.7 else "empty" for _ in range(slots))
            if any(COMPONENTS[item].kind == "fuel" for item in layout) and self._within_limits(layout):
                return layout
        result = ["empty"] * slots
        fuel = self._fuel_allowed()
        if fuel:
            result[rng.randrange(slots)] = fuel[0]
        return tuple(result)

    def _mutate(self, layout: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
        result = list(layout)
        mode = rng.randrange(3)
        if mode == 0:
            # 局部交换
            a, b = rng.sample(range(len(result)), 2)
            result[a], result[b] = result[b], result[a]
        elif mode == 1:
            # 将一个现有组件移动到空格。
            occupied = [index for index, item in enumerate(result) if item != "empty"]
            empty = [index for index, item in enumerate(result) if item == "empty"]
            if occupied and empty:
                source, target = rng.choice(occupied), rng.choice(empty)
                result[target], result[source] = result[source], "empty"
        else:
            # 在库存约束内替换组件。
            values = ["empty", *self._fuel_allowed(), *(item for item, limit in self.request.component_limits.items() if limit > 0)]
            result[rng.randrange(len(result))] = rng.choice(values)
        candidate = tuple(result)
        return candidate if self._within_limits(candidate) and any(COMPONENTS[x].kind == "fuel" for x in candidate) else layout

    def _evaluate(self, layout: tuple[str, ...]) -> CandidateResult | None:
        if self.cancel_event.is_set() or not self._within_limits(layout):
            return None
        result = evaluate_layout(layout, self.request.columns, self.request.max_reactor_ticks)
        family = mark_family(result.mark)
        if family not in self.request.marks:
            return None
        return result

    def _accept(self, result: CandidateResult | None, *, count_evaluation: bool = True) -> None:
        if count_evaluation:
            self.evaluated += 1
        if result is None:
            return
        family = mark_family(result.mark)
        if family is None or family not in self.leaderboards:
            return
        self.leaderboards[family] = _rank_candidates(
            [*self.leaderboards[family], result],
            self.request.result_limit,
        )

    def _run_heuristic(self) -> None:
        rng = random.Random(self.request.seed + self.evaluated)
        self.accelerator = "cpu"
        self.accelerator_detail = None
        self.accelerator_fallback_reason = None
        gpu_scorer: CudaBatchScorer | None = None
        if self.request.accelerator != "cpu":
            info = cuda_device_info()
            if info.available:
                try:
                    gpu_scorer = CudaBatchScorer()
                    self.accelerator = "cuda"
                    self.accelerator_detail = info.label
                except Exception as exc:
                    self.accelerator_fallback_reason = str(exc)
            else:
                self.accelerator_fallback_reason = info.reason
            if gpu_scorer is None and self.request.accelerator == "cuda":
                raise RuntimeError(self.accelerator_fallback_reason or "请求了 CUDA，但 CUDA 不可用")

        # GPU mode uses a few larger islands. Tiny per-island batches waste most
        # of a high-end GPU and do not improve genetic diversity in practice.
        island_limit = 4 if gpu_scorer is not None else self.request.cpu_workers
        island_count = min(island_limit, max(1, self.request.population // 10))
        base_size, remainder = divmod(self.request.population, island_count)
        island_sizes = [base_size + (island < remainder) for island in range(island_count)]

        def gpu_screen_pools(
            pools: list[list[tuple[str, ...]]],
            keeps: list[int],
            preserved: list[list[tuple[str, ...]]] | None = None,
        ) -> list[list[tuple[str, ...]]]:
            if gpu_scorer is None:
                return [pool[:keep] for pool, keep in zip(pools, keeps, strict=True)]
            unique_pools = [list(dict.fromkeys(pool)) for pool in pools]
            flattened = [layout for pool in unique_pools for layout in pool]
            if not flattened:
                return [[] for _ in pools]
            scores = gpu_scorer.score(flattened, self.request.columns)
            self.gpu_screened += len(flattened)
            self.gpu_batches += 1
            result: list[list[tuple[str, ...]]] = []
            offset = 0
            for pool_index, (pool, keep) in enumerate(zip(unique_pools, keeps, strict=True)):
                end = offset + len(pool)
                pool_scores = CudaBatchScores(
                    scores.power[offset:end],
                    scores.generated_heat[offset:end],
                    scores.cooling_proxy[offset:end],
                )
                required = list(dict.fromkeys((preserved or [[] for _ in pools])[pool_index]))
                required = required[:keep]
                required_set = set(required)
                candidates = [layout for layout in pool if layout not in required_set]
                candidate_indices = [index for index, layout in enumerate(pool) if layout not in required_set]
                candidate_scores = CudaBatchScores(
                    pool_scores.power[candidate_indices],
                    pool_scores.generated_heat[candidate_indices],
                    pool_scores.cooling_proxy[candidate_indices],
                )
                selected = select_screened_layouts(
                    candidates,
                    candidate_scores,
                    keep - len(required),
                    mark_i_only=self.request.marks == ["I"],
                )
                result.append([*required, *selected])
                offset = end
            return result

        if gpu_scorer is None:
            islands = [
                [self._random_layout(rng) for _ in range(size)]
                for size in island_sizes
            ]
        else:
            target_total = max(
                4_096,
                self.request.population * self.request.gpu_batch_multiplier,
            )
            pool_base, pool_remainder = divmod(target_total, island_count)
            initial_pools = []
            for island, size in enumerate(island_sizes):
                pool = [self._random_layout(rng) for _ in range(size)]
                target = pool_base + (island < pool_remainder)
                while len(pool) < target:
                    pool.append(self._mutate(rng.choice(pool), rng))
                initial_pools.append(pool)
            islands = gpu_screen_pools(initial_pools, island_sizes)
            for island, size in zip(islands, island_sizes, strict=True):
                while len(island) < size:
                    island.append(self._random_layout(rng))
        deadline = time.time() + self.request.time_budget_seconds
        executor = ProcessPoolExecutor(max_workers=self.request.cpu_workers) if self.request.cpu_workers > 1 else None
        for generation in range(self.request.generations):
            if self.cancel_event.is_set() or time.time() >= deadline:
                break
            self.generation = generation + 1
            population = [layout for island in islands for layout in island]
            unique_population = list(dict.fromkeys(population))
            scored: list[tuple[tuple, tuple[str, ...]]] = []
            if executor is None:
                for layout in unique_population:
                    if self.cancel_event.is_set() or time.time() >= deadline:
                        break
                    if layout in self._heuristic_cache:
                        result = self._heuristic_cache[layout]
                    else:
                        result = self._evaluate(layout)
                        self._heuristic_cache[layout] = result
                        self._accept(result)
                    scored.append((result.score() if result else (-1,), layout))
            else:
                valid = [layout for layout in unique_population if self._within_limits(layout)]
                unseen: list[tuple[str, ...]] = []
                for layout in valid:
                    if layout in self._heuristic_cache:
                        result = self._heuristic_cache[layout]
                        scored.append((result.score() if result else (-1,), layout))
                    else:
                        unseen.append(layout)
                futures = {
                    executor.submit(
                        evaluate_layout,
                        layout,
                        self.request.columns,
                        self.request.max_reactor_ticks,
                        None,
                        False,
                    ): layout
                    for layout in unseen
                }
                try:
                    for future in as_completed(futures, timeout=max(0.01, deadline - time.time())):
                        if self.cancel_event.is_set() or time.time() >= deadline:
                            break
                        result = future.result()
                        if mark_family(result.mark) not in self.request.marks:
                            result = None
                        self._heuristic_cache[futures[future]] = result
                        self._accept(result)
                        scored.append((result.score() if result else (-1,), futures[future]))
                except TimeoutError:
                    pass
                for future in futures:
                    future.cancel()
            score_by_layout = {layout: score for score, layout in scored}
            next_islands: list[list[tuple[str, ...]]] = []
            mutation_pools: list[list[tuple[str, ...]]] = []
            preserved_elites: list[list[tuple[str, ...]]] = []
            leaders: list[tuple[str, ...]] = []
            for island in islands:
                island_scored = sorted(
                    ((score_by_layout.get(layout, (-1,)), layout) for layout in island),
                    reverse=True,
                    key=lambda item: item[0],
                )
                elite_count = max(2, len(island) // 5)
                elites = [layout for _, layout in island_scored[:elite_count]]
                leaders.append(elites[0])
                if gpu_scorer is None:
                    next_island = list(elites)
                    while len(next_island) < len(island):
                        next_island.append(self._mutate(rng.choice(elites), rng))
                    next_islands.append(next_island)
                else:
                    target = max(
                        len(island) * self.request.gpu_batch_multiplier,
                        4_096 // island_count,
                    )
                    pool = list(elites)
                    while len(pool) < target:
                        pool.append(self._mutate(rng.choice(elites), rng))
                    mutation_pools.append(pool)
                    preserved_elites.append(elites)
            if gpu_scorer is not None:
                next_islands = gpu_screen_pools(
                    mutation_pools,
                    [len(island) for island in islands],
                    preserved_elites,
                )
            # 每五代做环形迁移：各岛最佳个体替换下一岛的最后一个个体。
            if island_count > 1 and (generation + 1) % 5 == 0:
                for island, leader in enumerate(leaders):
                    next_islands[(island + 1) % island_count][-1] = leader
            islands = next_islands
            self.progress = min(0.999, (generation + 1) / self.request.generations)
            gpu_status = f" · GPU 已筛选 {self.gpu_screened:,} 个" if gpu_scorer is not None else ""
            self.message = (
                f"第 {generation + 1} 代 · {island_count} 个岛 · "
                f"已精确评估 {self.evaluated} 个布局{gpu_status}"
            )
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_exhaustive(self) -> None:
        self.accelerator = "cpu"
        self.accelerator_detail = None
        self.accelerator_fallback_reason = None
        if self.request.accelerator != "cpu":
            info = cuda_device_info()
            if info.available:
                self.accelerator = (
                    "cuda_full"
                    if self.request.accelerator == "cuda_full"
                    else "cuda"
                )
                self.accelerator_detail = info.label
            else:
                self.accelerator_fallback_reason = info.reason
                if self.request.accelerator in {"cuda", "cuda_full"}:
                    raise RuntimeError(info.reason or "请求了 CUDA，但 CUDA 不可用")
        estimate = self.exhaustive_estimate or 0
        total = max(1, estimate)
        mark_i_two_level = self.request.marks == ["I"]
        shards = (
            [()]
            if self.accelerator in {"cuda", "cuda_full"}
            else _exhaustive_shards(
                self.request,
                power_only=mark_i_two_level,
                target_shards=self.request.cpu_workers * 4 if mark_i_two_level else None,
            )
        )
        # One CUDA context owns large batches efficiently. Multiple WDDM
        # processes contending for the same GPU are slower and waste VRAM.
        worker_limit = (
            1
            if self.accelerator in {"cuda", "cuda_full"}
            else self.request.cpu_workers
        )
        worker_count = min(worker_limit, len(shards))
        request_data = self.request.model_dump(mode="json")
        shard_progress: dict[int, tuple[int, int, int]] = {}
        shard_gpu_progress: dict[int, tuple[int, int, int, int]] = {}

        def update_progress(shard_id: int, checked: int, pruned: int, evaluated: int) -> None:
            old_checked, old_pruned, old_evaluated = shard_progress.get(shard_id, (0, 0, 0))
            self.checked += checked - old_checked
            self.pruned += pruned - old_pruned
            self.evaluated += evaluated - old_evaluated
            shard_progress[shard_id] = (checked, pruned, evaluated)
            self.progress = min(0.999, self.checked / total)
            self.message = (
                f"{worker_count} 进程并行枚举 · 已检查 {self.checked:,} 个方案"
                f" · 精确验证 {self.evaluated:,} 个 · 数学跳过 {self.pruned:,} 个"
            )

        def update_gpu_progress(
            shard_id: int,
            certified: int,
            fallback: int,
            batches: int,
            full_simulated: int = 0,
        ) -> None:
            old_certified, old_fallback, old_batches, old_full_simulated = shard_gpu_progress.get(
                shard_id, (0, 0, 0, 0)
            )
            self.gpu_exhaustive_certified += certified - old_certified
            self.gpu_exhaustive_fallback += fallback - old_fallback
            self.gpu_screened += (
                certified + fallback + full_simulated
                - old_certified - old_fallback - old_full_simulated
            )
            self.gpu_batches += batches - old_batches
            self.gpu_full_simulated += full_simulated - old_full_simulated
            shard_gpu_progress[shard_id] = (
                certified, fallback, batches, full_simulated
            )
            if self.accelerator == "cuda_full":
                self.message = (
                    f"CUDA 完整模拟穷举 · 已检查 {self.checked:,} 个方案"
                    f" · GPU 完整模拟 {self.gpu_full_simulated:,} 个"
                    f" · 数学跳过 {self.pruned:,} 个"
                )
            elif self.accelerator == "cuda":
                self.message = (
                    f"CUDA 批量穷举 · 已检查 {self.checked:,} 个方案"
                    f" · GPU 证书 {self.gpu_exhaustive_certified:,} 个"
                    f" · CPU 回退 {self.gpu_exhaustive_fallback:,} 个"
                    f" · 数学跳过 {self.pruned:,} 个"
                )

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        self.process_cancel_event = manager.Event()
        shared_power_floor = manager.Value("d", -1.0) if mark_i_two_level else None
        executor = ProcessPoolExecutor(max_workers=worker_count)
        futures = {
            executor.submit(
                _run_exhaustive_shard,
                request_data,
                shard_id,
                shard,
                progress_queue,
                self.process_cancel_event,
                shared_power_floor,
            ): shard_id
            for shard_id, shard in enumerate(shards)
        }
        pending = set(futures)

        def handle_message(message: tuple) -> None:
            if message[0] == "progress":
                _, shard_id, checked, pruned, evaluated, *gpu_values = message
                update_progress(shard_id, checked, pruned, evaluated)
                if gpu_values:
                    update_gpu_progress(shard_id, *gpu_values)
            elif message[0] == "candidate":
                self._accept(message[1], count_evaluation=False)
                if (
                    shared_power_floor is not None
                    and len(self.leaderboards["I"]) >= self.request.result_limit
                ):
                    shared_power_floor.value = self.leaderboards["I"][-1].average_eu_per_tick

        try:
            while pending:
                if self.cancel_event.is_set():
                    self.process_cancel_event.set()
                    for future in pending:
                        future.cancel()
                try:
                    handle_message(progress_queue.get(timeout=0.2))
                except queue.Empty:
                    pass

                finished = [future for future in pending if future.done()]
                for future in finished:
                    pending.remove(future)
                    if future.cancelled():
                        continue
                    result = future.result()
                    self.skeleton_table_states += result.get("skeleton_table_states", 0)
                    self.skeleton_table_cache_hits += int(
                        result.get("skeleton_table_cache_hit", False)
                    )
                    update_gpu_progress(
                        result["shard_id"],
                        result.get("gpu_certified", 0),
                        result.get("gpu_fallback", 0),
                        result.get("gpu_batches", 0),
                        result.get("gpu_full_simulated", 0),
                    )
                    if result.get("gpu_error") and self.request.accelerator == "auto":
                        self.accelerator = "cpu"
                        self.accelerator_detail = None
                        self.accelerator_fallback_reason = result["gpu_error"]
                    update_progress(
                        result["shard_id"], result["checked"], result["pruned"], result["evaluated"]
                    )
                    for values in result["boards"].values():
                        for candidate in values:
                            self._accept(candidate, count_evaluation=False)
                    if (
                        shared_power_floor is not None
                        and len(self.leaderboards["I"]) >= self.request.result_limit
                    ):
                        shared_power_floor.value = self.leaderboards["I"][-1].average_eu_per_tick

            while True:
                try:
                    handle_message(progress_queue.get_nowait())
                except queue.Empty:
                    break
            if not self.cancel_event.is_set():
                self.proven_global = True
        finally:
            if self.process_cancel_event is not None:
                self.process_cancel_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.process_cancel_event = None
            manager.shutdown()

    def cancel(self) -> None:
        self.cancel_event.set()
        if self.process_cancel_event is not None:
            self.process_cancel_event.set()

    def run(self) -> None:
        self.status = "running"
        self.started_at = time.time()
        try:
            if self.request.solver == "exhaustive":
                self._run_exhaustive()
            else:
                self._run_heuristic()
            if self.cancel_event.is_set():
                self.status = "cancelled"
                self.message = "优化已取消，保留当前候选"
            else:
                self.status = "completed"
                self.progress = 1.0
                self.message = "优化完成"
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
            self.message = "优化失败"
        finally:
            self.finished_at = time.time()


class OptimizationManager:
    def __init__(self):
        self.jobs: dict[str, OptimizationJob] = {}
        self.lock = threading.Lock()

    def create(self, request: OptimizationRequest) -> OptimizationJob:
        job = OptimizationJob(request)
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=job.run, name=f"optimizer-{job.id[:8]}", daemon=True).start()
        return job

    def get(self, job_id: str) -> OptimizationJob:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError("优化任务不存在") from exc

    def latest(self) -> OptimizationJob:
        with self.lock:
            if not self.jobs:
                raise KeyError("暂无优化任务")
            return next(reversed(self.jobs.values()))

    def resume(self, job_id: str) -> OptimizationJob:
        job = self.get(job_id)
        if job.request.solver != "heuristic":
            raise ValueError("穷举任务不能续算；请新建任务并完整枚举")
        if job.status in {"queued", "running"}:
            raise ValueError("任务仍在运行")
        job.cancel_event.clear()
        job.status = "queued"
        job.progress = 0.0
        job.proven_global = False
        job.message = "准备继续改进当前候选"
        threading.Thread(target=job.run, name=f"optimizer-{job.id[:8]}-resume", daemon=True).start()
        return job
