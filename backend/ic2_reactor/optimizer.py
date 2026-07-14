from __future__ import annotations

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
from .mark import mark_family
from .models import Layout, OptimizationRequest


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


def _power_vertex_value(item: str) -> int:
    spec = COMPONENTS[item]
    return (
        int(ReactorSimulator.EU_PER_PULSE) * spec.rod_count * spec.internal_pulses
        if spec.kind == "fuel"
        else 0
    )


def _power_edge_value(first: str, second: str) -> int:
    first_spec = COMPONENTS[first]
    second_spec = COMPONENTS[second]
    pulses = 0
    if first_spec.kind == "fuel" and second_spec.kind in {"fuel", "reflector"}:
        pulses += first_spec.rod_count
    if second_spec.kind == "fuel" and first_spec.kind in {"fuel", "reflector"}:
        pulses += second_spec.rod_count
    return int(ReactorSimulator.EU_PER_PULSE) * pulses


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
        caps = {item: fuel_limits[item] for item in fuels}
    nonfuel = [item for item, limit in request.component_limits.items() if limit > 0]
    caps.update({item: request.component_limits[item] for item in nonfuel})
    return [*fuels, *nonfuel], caps


def _exhaustive_shards(
    request: OptimizationRequest,
    *,
    power_only: bool = False,
) -> list[tuple[tuple[int, str], ...]]:
    """Split the search into disjoint assignments on two central cells.

    Central cells distribute the labelled search space more evenly than a
    top-left prefix does.
    """
    columns = request.columns
    positions = (2 * columns + columns // 2, 3 * columns + columns // 2)
    allowed, caps = _allowed_and_caps(request)
    if power_only:
        allowed = [item for item in allowed if COMPONENTS[item].kind in {"fuel", "reflector"}]
    values = ["empty", *allowed]
    shards: list[tuple[tuple[int, str], ...]] = []
    for first in values:
        for second in values:
            used: dict[str, int] = {}
            rods = 0
            valid = True
            for item in (first, second):
                if item == "empty":
                    continue
                used[item] = used.get(item, 0) + 1
                rods += COMPONENTS[item].rod_count
                if used[item] > caps[item]:
                    valid = False
            if request.fuel.mode == "total_rods" and rods > request.fuel.total_rods:
                valid = False
            if valid:
                shards.append(((positions[0], first), (positions[1], second)))
    # Prefer fuel-bearing assignments so leaderboards begin producing useful
    # results immediately. Every shard is still evaluated in full.
    shards.sort(key=lambda shard: (
        not any(COMPONENTS[item].rod_count for _, item in shard),
        shard,
    ))
    return shards


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


def _rank_candidates(values: list[CandidateResult]) -> list[CandidateResult]:
    board: dict[str, CandidateResult] = {}
    for result in values:
        previous = board.get(result.canonical)
        if previous is None or result.score() > previous.score():
            board[result.canonical] = result
    ordered = sorted(board.values(), key=lambda item: item.canonical)
    ordered.sort(key=lambda item: item.score(), reverse=True)
    return ordered[:10]


def _run_mark_i_two_level_shard(
    request: OptimizationRequest,
    shard_id: int,
    fixed_items: tuple[tuple[int, str], ...],
    progress_queue,
    cancel_event,
    shared_power_floor=None,
) -> dict:
    """Enumerate power skeletons first, then their cooling completions."""
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
    power_remaining = {item: caps[item] for item in power_items}
    cooling_caps = tuple(caps[item] for item in cooling_items)
    cooling_cap_items = tuple(zip(cooling_items, cooling_caps, strict=True))
    cooling_remaining = dict(zip(cooling_items, cooling_caps, strict=True))
    fixed = dict(fixed_items)
    slots = request.columns * 6
    power_caps = tuple(caps[item] for item in power_items)
    skeleton = ["empty"] * slots
    rods = 0
    has_fuel = False
    for position, item in fixed_items:
        if item == "empty":
            continue
        skeleton[position] = item
        power_remaining[item] -= 1
        rods += COMPONENTS[item].rod_count
        has_fuel = has_fuel or COMPONENTS[item].rod_count > 0

    checked = 0
    pruned = 0
    evaluated = 0
    visits = 0
    last_report = time.monotonic()
    cancelled = False
    cancel_cache = False
    last_cancel_check = 0.0
    shared_floor_cache = -1.0
    last_floor_check = 0.0
    boards: dict[str, list[CandidateResult]] = {"I": []}
    grid_edges = tuple(
        (index, neighbor)
        for index in range(slots)
        for neighbor in (
            *((index + 1,) if index % request.columns + 1 < request.columns else ()),
            *((index + request.columns,) if index + request.columns < slots else ()),
        )
    )

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
        local_floor = local_board[-1].average_eu_per_tick if len(local_board) >= 10 else -1.0
        now = time.monotonic()
        if shared_power_floor is not None and now - last_floor_check >= 0.1:
            shared_floor_cache = shared_power_floor.value
            last_floor_check = now
        return max(local_floor, shared_floor_cache)

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 0.25:
            progress_queue.put(("progress", shard_id, checked, pruned, evaluated))
            last_report = now

    def accept_result(result: CandidateResult) -> None:
        previous = boards["I"]
        ranked = _rank_candidates([*previous, result])
        if ranked != previous:
            boards["I"] = ranked
            progress_queue.put(("candidate", result))

    def power_increment(position: int, item: str) -> int:
        value = _power_vertex_value(item)
        row, column = divmod(position, request.columns)
        if column > 0:
            value += _power_edge_value(skeleton[position - 1], item)
        if row > 0:
            value += _power_edge_value(skeleton[position - request.columns], item)
        return value

    def optimistic_power_bound(position: int, current_power: int, current_rods: int) -> int:
        """Bound every unfinished skeleton without underestimating its power."""
        options = ["empty"]
        for item in power_items:
            if power_remaining[item] <= 0:
                continue
            rod_cost = COMPONENTS[item].rod_count
            if request.fuel.mode == "total_rods" and current_rods + rod_cost > request.fuel.total_rods:
                continue
            options.append(item)

        def known_label(index: int) -> str | None:
            if index < position or index in fixed:
                return skeleton[index]
            return None

        upper = current_power
        max_vertex = max(_power_vertex_value(item) for item in options)
        for index in range(position, slots):
            upper += (
                _power_vertex_value(skeleton[index])
                if index in fixed
                else max_vertex
            )

        for first, second in grid_edges:
            if second < position:
                continue
            first_label = known_label(first)
            second_label = known_label(second)
            if first_label is not None and second_label is not None:
                upper += _power_edge_value(first_label, second_label)
            elif first_label is not None:
                upper += max(_power_edge_value(first_label, item) for item in options)
            elif second_label is not None:
                upper += max(_power_edge_value(item, second_label) for item in options)
            else:
                upper += max(_power_edge_value(a, b) for a in options for b in options)
        return upper

    def count_remaining_layouts(
        position: int,
        current_rods: int,
        current_has_fuel: bool,
    ) -> int:
        """Count a pruned partial skeleton and every cooling completion below it."""
        remaining_positions = sum(index not in fixed for index in range(position, slots))
        used_power_slots = sum(
            cap - power_remaining[item]
            for item, cap in zip(power_items, power_caps, strict=True)
        )
        # (new power slots, new rods, has fuel) -> labelled assignments.
        dp: dict[tuple[int, int, bool], int] = {(0, 0, current_has_fuel): 1}
        for item in power_items:
            spec = COMPONENTS[item]
            cap = power_remaining[item]
            next_dp: dict[tuple[int, int, bool], int] = {}
            for (used, extra_rods, has_fuel), ways in dp.items():
                for count in range(min(cap, remaining_positions - used) + 1):
                    rods = extra_rods + count * spec.rod_count
                    if (
                        request.fuel.mode == "total_rods"
                        and current_rods + rods > request.fuel.total_rods
                    ):
                        break
                    key = (used + count, rods, has_fuel or (spec.kind == "fuel" and count > 0))
                    next_dp[key] = (
                        next_dp.get(key, 0)
                        + ways * math.comb(remaining_positions - used, count)
                    )
            dp = next_dp

        total = 0
        for (additional_power, _extra_rods, has_fuel), ways in dp.items():
            if not has_fuel:
                continue
            free_slots = slots - used_power_slots - additional_power
            total += ways * count_cooling_completions(free_slots, cooling_caps)
        return total

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
        if floor >= 0 and skeleton_power < floor:
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
            if (
                sustainable_heat_flow_upper_bound(raw, request.columns)
                < skeleton_heat
            ):
                pruned += 1
                report()
                return
            result = evaluate_layout(
                raw,
                request.columns,
                request.max_reactor_ticks,
                cancel_check=cancellation_requested,
            )
            if cancellation_requested():
                cancelled = True
                return
            evaluated += 1
            if mark_family(result.mark) == "I":
                accept_result(result)
            report()
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

    def finish_skeleton() -> None:
        nonlocal checked, pruned
        raw_skeleton = tuple(skeleton)
        free_positions = tuple(index for index, item in enumerate(raw_skeleton) if item == "empty")
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
            or (current_power_floor() >= 0 and skeleton_power < current_power_floor())
        ):
            checked += completion_count
            pruned += completion_count
            report()
            return
        layout = list(raw_skeleton)
        generate_cooling(layout, free_positions, 0, skeleton_power, skeleton_heat)

    def generate_skeleton(
        position: int,
        current_rods: int,
        current_has_fuel: bool,
        current_power: int,
    ) -> None:
        nonlocal checked, pruned, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancel_event.is_set():
            cancelled = True
            return
        if cancelled:
            return
        floor = current_power_floor()
        if floor >= 0 and optimistic_power_bound(position, current_power, current_rods) < floor:
            count = count_remaining_layouts(position, current_rods, current_has_fuel)
            checked += count
            pruned += count
            report()
            return
        if position == slots:
            if current_has_fuel:
                finish_skeleton()
            return
        if position in fixed:
            generate_skeleton(
                position + 1,
                current_rods,
                current_has_fuel,
                current_power + power_increment(position, skeleton[position]),
            )
            return

        for item in [*power_items, "empty"]:
            if item == "empty":
                skeleton[position] = "empty"
                generate_skeleton(position + 1, current_rods, current_has_fuel, current_power)
            elif power_remaining[item] > 0:
                rod_cost = COMPONENTS[item].rod_count
                if request.fuel.mode == "total_rods" and current_rods + rod_cost > request.fuel.total_rods:
                    continue
                power_remaining[item] -= 1
                skeleton[position] = item
                generate_skeleton(
                    position + 1,
                    current_rods + rod_cost,
                    current_has_fuel or rod_cost > 0,
                    current_power + power_increment(position, item),
                )
                power_remaining[item] += 1
            if cancelled:
                break
        skeleton[position] = "empty"

    generate_skeleton(0, rods, has_fuel, 0)
    report(force=True)
    return {
        "shard_id": shard_id,
        "checked": checked,
        "pruned": pruned,
        "evaluated": evaluated,
        "boards": boards,
        "cancelled": cancelled,
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
    visits = 0
    last_report = time.monotonic()
    cancelled = False
    cancel_cache = False
    last_cancel_check = 0.0
    boards: dict[str, list[CandidateResult]] = {mark: [] for mark in request.marks}

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
            progress_queue.put(("progress", shard_id, checked, pruned, evaluated))
            last_report = now

    def generate(position: int, current_rods: int, current_has_fuel: bool) -> None:
        nonlocal checked, pruned, evaluated, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancel_event.is_set():
            cancelled = True
            return
        if position == slots:
            if not current_has_fuel:
                return
            checked += 1
            raw = tuple(layout)
            result = evaluate_layout(
                raw,
                request.columns,
                request.max_reactor_ticks,
                cancel_check=cancellation_requested,
            )
            if cancellation_requested():
                cancelled = True
                return
            evaluated += 1
            family = mark_family(result.mark)
            if family in boards:
                previous = boards[family]
                ranked = _rank_candidates([*previous, result])
                # A later mirrored direction can replace an earlier result
                # without changing the canonical-key sequence.
                if ranked != previous:
                    boards[family] = ranked
                    progress_queue.put(("candidate", result))
            report()
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
    report(force=True)
    return {
        "shard_id": shard_id,
        "checked": checked,
        "pruned": pruned,
        "evaluated": evaluated,
        "boards": boards,
        "cancelled": cancelled,
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
        self.leaderboards[family] = _rank_candidates([*self.leaderboards[family], result])

    def _run_heuristic(self) -> None:
        rng = random.Random(self.request.seed + self.evaluated)
        island_count = min(self.request.cpu_workers, max(1, self.request.population // 10))
        base_size, remainder = divmod(self.request.population, island_count)
        islands = [
            [self._random_layout(rng) for _ in range(base_size + (island < remainder))]
            for island in range(island_count)
        ]
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
                next_island = list(elites)
                while len(next_island) < len(island):
                    next_island.append(self._mutate(rng.choice(elites), rng))
                next_islands.append(next_island)
            # 每五代做环形迁移：各岛最佳个体替换下一岛的最后一个个体。
            if island_count > 1 and (generation + 1) % 5 == 0:
                for island, leader in enumerate(leaders):
                    next_islands[(island + 1) % island_count][-1] = leader
            islands = next_islands
            self.progress = min(0.999, (generation + 1) / self.request.generations)
            self.message = f"第 {generation + 1} 代 · {island_count} 个岛 · 已评估 {self.evaluated} 个布局"
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_exhaustive(self) -> None:
        estimate = self.exhaustive_estimate or 0
        total = max(1, estimate)
        mark_i_two_level = self.request.marks == ["I"]
        shards = _exhaustive_shards(self.request, power_only=mark_i_two_level)
        worker_count = min(self.request.cpu_workers, len(shards))
        request_data = self.request.model_dump(mode="json")
        shard_progress: dict[int, tuple[int, int, int]] = {}

        def update_progress(shard_id: int, checked: int, pruned: int, evaluated: int) -> None:
            old_checked, old_pruned, old_evaluated = shard_progress.get(shard_id, (0, 0, 0))
            self.checked += checked - old_checked
            self.pruned += pruned - old_pruned
            self.evaluated += evaluated - old_evaluated
            shard_progress[shard_id] = (checked, pruned, evaluated)
            self.progress = min(0.999, self.checked / total)
            self.message = (
                f"{worker_count} 进程并行枚举 · 已检查 {self.checked:,} 个方案"
                f" · 热学模拟 {self.evaluated:,} 个 · 数学跳过 {self.pruned:,} 个"
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
                _, shard_id, checked, pruned, evaluated = message
                update_progress(shard_id, checked, pruned, evaluated)
            elif message[0] == "candidate":
                self._accept(message[1], count_evaluation=False)
                if shared_power_floor is not None and len(self.leaderboards["I"]) >= 10:
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
                    update_progress(
                        result["shard_id"], result["checked"], result["pruned"], result["evaluated"]
                    )
                    for values in result["boards"].values():
                        for candidate in values:
                            self._accept(candidate, count_evaluation=False)
                    if shared_power_floor is not None and len(self.leaderboards["I"]) >= 10:
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
