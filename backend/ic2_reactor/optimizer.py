from __future__ import annotations

import json
import math
import multiprocessing
import os
import queue
import random
import threading
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .components import COMPONENTS
from .cpu_scheduling import (
    cpu_scheduling_plan,
    initialize_compute_worker,
    initialize_gpu_service,
)
from .engine import ReactorSimulator, SimulationOptions
from .kernel_abi import (
    PackedEvaluationBatch,
    PackedLayoutBatch,
    decode_mark,
    pack_layouts,
)
from .mark import FUEL_CYCLE_REACTOR_TICKS, TEN_PERCENT_CYCLE, mark_family
from .models import OptimizationRequest


EVALUATION_BATCH_SIZE = 64
CUDA_MAX_BATCH_SIZE = 16_384
CUDA_MIN_BATCH_SIZE = 8_192
CUDA_MAX_DIRECT_REACTOR_TICKS = 40_000
EXHAUSTIVE_WARM_START_LAYOUTS = 512
CHECKPOINT_DIRECTORY = Path(".data/checkpoints")


def _wait_for_worker_control(cancel_event, pause_event=None) -> bool:
    """Block at a safe point while paused; return whether cancellation won."""
    while pause_event is not None and pause_event.is_set():
        if cancel_event.is_set():
            return True
        time.sleep(0.05)
    return cancel_event.is_set()


def _search_batch_size(request: OptimizationRequest) -> int:
    """Choose a bounded batch that fills CUDA without excessive checkpoints."""
    if request.compute_backend != "cuda":
        return EVALUATION_BATCH_SIZE
    checkpoints = (
        (request.unresolved_max_reactor_ticks or request.max_reactor_ticks)
        // FUEL_CYCLE_REACTOR_TICKS
        + 1
    )
    checkpoint_bytes_per_layout = checkpoints * (
        request.columns * 6 * 9 + 8
    )
    memory_limited = (512 * 1024 * 1024) // max(1, checkpoint_bytes_per_layout)
    return max(256, min(CUDA_MAX_BATCH_SIZE, memory_limited))


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


@lru_cache(maxsize=28)
def _transform_indices(columns: int, flip_h: bool, flip_v: bool) -> tuple[int, ...]:
    """Precompute a mirror permutation for one of the seven chamber widths."""
    return tuple(
        (5 - row if flip_v else row) * columns
        + (columns - 1 - column if flip_h else column)
        for row in range(6)
        for column in range(columns)
    )


def _transform(layout: tuple[str, ...], columns: int, flip_h: bool, flip_v: bool) -> tuple[str, ...]:
    return tuple(layout[index] for index in _transform_indices(columns, flip_h, flip_v))


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


@lru_cache(maxsize=16)
def _fuel_intrinsic_heat(item: str) -> int:
    """Minimum heat from one fuel package with no active neighbours."""
    spec = COMPONENTS[item]
    pulses = spec.internal_pulses
    return 2 * spec.rod_count * pulses * (pulses + 1)


def _partial_skeleton_heat_increment(
    skeleton: list[str],
    position: int,
    item: str,
    columns: int,
) -> int:
    """Exact heat increase when one power-active item enters a partial grid."""
    spec = COMPONENTS[item]
    if spec.kind not in {"fuel", "reflector"}:
        return 0
    slots = len(skeleton)
    neighbors = _layout_neighbors(position, columns, slots)
    active_neighbors = sum(
        COMPONENTS[skeleton[neighbor]].kind in {"fuel", "reflector"}
        for neighbor in neighbors
    )
    increase = 0
    if spec.kind == "fuel":
        pulses = spec.internal_pulses + active_neighbors
        increase += 2 * spec.rod_count * pulses * (pulses + 1)

    # The new item adds one pulse to every adjacent existing fuel. Its old
    # pulse count is read from the grid before ``item`` is installed.
    for neighbor in neighbors:
        neighbor_spec = COMPONENTS[skeleton[neighbor]]
        if neighbor_spec.kind != "fuel":
            continue
        old_neighbors = sum(
            COMPONENTS[skeleton[other]].kind in {"fuel", "reflector"}
            for other in _layout_neighbors(neighbor, columns, slots)
        )
        old_pulses = neighbor_spec.internal_pulses + old_neighbors
        increase += 4 * neighbor_spec.rod_count * (old_pulses + 1)
    return increase


@lru_cache(maxsize=16_384)
def _optimistic_cooling_upper_bound(
    free_slots: int,
    cooling_caps: tuple[tuple[str, int], ...],
) -> int:
    """Topology-free sustainable cooling upper bound for unfinished grids."""
    if free_slots <= 0:
        return 0
    values: list[int] = []
    for item, cap in cooling_caps:
        spec = COMPONENTS[item]
        if spec.kind != "vent" or cap <= 0:
            continue
        # A component vent can touch at most four components. Ignoring grid
        # conflicts only enlarges this bound and therefore remains safe.
        per_component = spec.self_vent + 4 * spec.side_vent
        if per_component > 0:
            values.extend([per_component] * min(cap, free_slots))
    values.sort(reverse=True)
    return sum(values[:free_slots])


@lru_cache(maxsize=65_536)
def _minimum_total_fuel_margins(
    needed_rods: int,
    available_slots: int,
    fuel_caps: tuple[tuple[str, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Return minimum intrinsic heat for each exact package-slot count."""
    # (used slots, used rods) -> minimum intrinsic heat
    dp: dict[tuple[int, int], int] = {(0, 0): 0}
    for item, cap in fuel_caps:
        spec = COMPONENTS[item]
        base_heat = _fuel_intrinsic_heat(item)
        next_dp: dict[tuple[int, int], int] = {}
        for (used_slots, used_rods), heat in dp.items():
            maximum = min(
                cap,
                available_slots - used_slots,
                (needed_rods - used_rods) // spec.rod_count,
            )
            for count in range(maximum + 1):
                key = (
                    used_slots + count,
                    used_rods + count * spec.rod_count,
                )
                next_dp[key] = min(
                    next_dp.get(key, math.inf),
                    heat + count * base_heat,
                )
        dp = next_dp
    return tuple(sorted(
        (used_slots, heat)
        for (used_slots, rods), heat in dp.items()
        if rods == needed_rods
    ))


def _partial_mark_i_heat_infeasible(
    request: OptimizationRequest,
    power_remaining: dict[str, int],
    cooling_caps: tuple[tuple[str, int], ...],
    current_rods: int,
    current_power_slots: int,
    available_slots: int,
    current_heat: int,
) -> bool:
    """Prove every completion of a partial skeleton thermally impossible."""
    slots = request.columns * 6
    free_before_required_fuel = slots - current_power_slots

    if request.fuel.usage != "exact":
        # Optional future power components can only add heat and consume slots.
        return current_heat > _optimistic_cooling_upper_bound(
            free_before_required_fuel,
            cooling_caps,
        )

    if request.fuel.mode == "separate":
        required = tuple(
            (item, power_remaining.get(item, 0)) for item in FUEL_ITEMS
        )
        additional_slots = sum(count for _item, count in required)
        if additional_slots > available_slots:
            return True
        additional_heat = sum(
            count * _fuel_intrinsic_heat(item) for item, count in required
        )
        return (
            current_heat + additional_heat
            > _optimistic_cooling_upper_bound(
                free_before_required_fuel - additional_slots,
                cooling_caps,
            )
        )

    needed_rods = request.fuel.total_rods - current_rods
    options = _minimum_total_fuel_margins(
        needed_rods,
        available_slots,
        tuple(
            (item, power_remaining.get(item, 0))
            for item in FUEL_ITEMS
            if power_remaining.get(item, 0) > 0
        ),
    )
    if not options:
        return True
    return all(
        current_heat + additional_heat
        > _optimistic_cooling_upper_bound(
            free_before_required_fuel - additional_slots,
            cooling_caps,
        )
        for additional_slots, additional_heat in options
    )


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


@lru_cache(maxsize=512)
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


def sustainable_heat_flow_upper_bound(
    layout: tuple[str, ...],
    columns: int,
    generated_heat: int | None = None,
) -> int:
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
    generated = (
        skeleton_heat_per_tick(power_skeleton(layout), columns)
        if generated_heat is None
        else generated_heat
    )
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


FUEL_ITEMS = ("uranium_single", "uranium_dual", "uranium_quad")


def _fuel_requirement_feasible(
    request: OptimizationRequest,
    remaining: dict[str, int],
    current_rods: int,
    available_slots: int,
) -> bool:
    """Whether an exact fuel target can still fit in unfinished slots."""
    if request.fuel.usage != "exact":
        return True
    if request.fuel.mode == "separate":
        return sum(remaining.get(item, 0) for item in FUEL_ITEMS) <= available_slots
    rods_needed = request.fuel.total_rods - current_rods
    return 0 <= rods_needed <= available_slots * 4


def _fuel_requirement_complete(
    request: OptimizationRequest,
    remaining: dict[str, int],
    current_rods: int,
    has_fuel: bool,
) -> bool:
    if not has_fuel:
        return False
    if request.fuel.usage != "exact":
        return True
    if request.fuel.mode == "separate":
        return all(remaining.get(item, 0) == 0 for item in FUEL_ITEMS)
    return current_rods == request.fuel.total_rods


def _allowed_and_caps(request: OptimizationRequest) -> tuple[list[str], dict[str, int]]:
    slots = request.columns * 6
    if request.fuel.mode == "total_rods":
        fuels = list(FUEL_ITEMS) if request.fuel.total_rods else []
        caps = {
            item: min(slots, request.fuel.total_rods // COMPONENTS[item].rod_count)
            for item in fuels
        }
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
    target_shards: int = 1,
) -> list[tuple[tuple[int, str], ...]]:
    """Split the search into enough disjoint central-cell assignments.

    Central cells distribute the labelled search space more evenly than a
    top-left prefix does.  Extra fixed cells create several times more shards
    than worker processes so late-running subtrees do not leave most CPUs idle.
    """
    columns = request.columns
    slots = columns * 6
    first_positions = (2 * columns + columns // 2, 3 * columns + columns // 2)
    remaining_positions = sorted(
        (position for position in range(slots) if position not in first_positions),
        key=lambda position: (
            abs(position // columns - 2.5),
            abs(position % columns - (columns - 1) / 2),
            position,
        ),
    )
    positions = (*first_positions, *remaining_positions)
    allowed, caps = _allowed_and_caps(request)
    if power_only:
        allowed = [item for item in allowed if COMPONENTS[item].kind in {"fuel", "reflector"}]
    # Enumerate occupied central assignments first. This only changes search
    # order: empty assignments remain in the shard list, so the proof still
    # covers the complete labelled layout space.
    values = [*allowed, "empty"]
    states: list[tuple[tuple[tuple[int, str], ...], dict[str, int], int]] = [
        ((), {}, 0)
    ]
    minimum_depth = min(2, slots)
    target_shards = max(1, target_shards)
    for depth, position in enumerate(positions, start=1):
        next_states: list[
            tuple[tuple[tuple[int, str], ...], dict[str, int], int]
        ] = []
        for assignments, used, rods in states:
            for item in values:
                next_rods = rods
                next_used = used
                if item != "empty":
                    count = used.get(item, 0) + 1
                    if count > caps[item]:
                        continue
                    next_rods += COMPONENTS[item].rod_count
                    if (
                        request.fuel.mode == "total_rods"
                        and next_rods > request.fuel.total_rods
                    ):
                        continue
                    next_used = dict(used)
                    next_used[item] = count
                next_states.append((
                    (*assignments, (position, item)),
                    next_used,
                    next_rods,
                ))
        remaining_slots = slots - depth
        states = [
            state
            for state in next_states
            if _fuel_requirement_feasible(
                request,
                {
                    item: caps.get(item, 0) - state[1].get(item, 0)
                    for item in FUEL_ITEMS
                },
                state[2],
                remaining_slots,
            )
        ]
        if depth >= minimum_depth and len(states) >= target_shards:
            break

    shards = [assignments for assignments, _used, _rods in states]
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
    # minimum, cap, rod cost, is fuel
    types: list[tuple[int, int, int, bool]] = []
    if request.fuel.mode == "separate":
        for cap, rods in zip(
            (request.fuel.single, request.fuel.dual, request.fuel.quad),
            (1, 2, 4),
            strict=True,
        ):
            if cap > 0:
                minimum = cap if request.fuel.usage == "exact" else 0
                types.append((minimum, cap, rods, True))
    elif request.fuel.total_rods > 0:
        types.extend((0, request.fuel.total_rods // rods, rods, True) for rods in (1, 2, 4))
    types.extend((0, cap, 0, False) for cap in request.component_limits.values() if cap > 0)

    # dp[(occupied slots, used rods, has fuel)] = number of ways to choose labelled positions.
    dp: dict[tuple[int, int, bool], int] = {(0, 0, False): 1}
    for minimum, cap, rod_cost, is_fuel in types:
        next_dp: dict[tuple[int, int, bool], int] = {}
        for (used, rods, has_fuel), ways in dp.items():
            maximum = min(cap, slots - used)
            for count in range(minimum, maximum + 1):
                next_rods = rods + count * rod_cost
                if request.fuel.mode == "total_rods" and next_rods > request.fuel.total_rods:
                    break
                key = (used + count, next_rods, has_fuel or (is_fuel and count > 0))
                next_dp[key] = next_dp.get(key, 0) + ways * math.comb(slots - used, count)
        dp = next_dp
    return sum(
        ways
        for (_used, rods, has_fuel), ways in dp.items()
        if has_fuel
        and (
            request.fuel.usage != "exact"
            or request.fuel.mode != "total_rods"
            or rods == request.fuel.total_rods
        )
    )


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
    simulator = ReactorSimulator.from_slots(columns, layout)
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
        component_count=len(layout) - layout.count("empty"),
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


def evaluate_layout_batch(
    layouts: tuple[tuple[str, ...], ...],
    columns: int,
    max_reactor_ticks: int,
    use_certificate: bool = False,
    cancel_check=None,
    compute_backend: str = "scalar",
    compute_workers: int | None = None,
) -> list[CandidateResult]:
    """Evaluate a batch through the selected scalar or packed CPU backend."""
    if compute_backend == "numba_cpu":
        if cancel_check is not None and cancel_check():
            return []
        # Import lazily so the reference backend and normal application startup
        # do not require Numba unless acceleration is explicitly selected.
        from .numba_backend import NumbaPackedEvaluator

        packed = NumbaPackedEvaluator(compute_workers).evaluate(
            pack_layouts(layouts, columns),
            max_reactor_ticks,
        )
    elif compute_backend == "cuda":
        if cancel_check is not None and cancel_check():
            return []
        if (
            len(layouts) < CUDA_MIN_BATCH_SIZE
            or max_reactor_ticks > CUDA_MAX_DIRECT_REACTOR_TICKS
        ):
            # A GPU lane is much slower than one CPU core for this branch-heavy
            # serial state machine.  Measured crossover on the target machine
            # is about 8K long-running layouts; longer kernels also risk the
            # Windows display-driver watchdog.  Keep those cases on CPU.
            from .numba_backend import NumbaPackedEvaluator

            packed = NumbaPackedEvaluator(compute_workers).evaluate(
                pack_layouts(layouts, columns),
                max_reactor_ticks,
            )
        else:
            try:
                from .cuda_backend import CudaPackedEvaluator

                packed = CudaPackedEvaluator().evaluate(
                    pack_layouts(layouts, columns),
                    max_reactor_ticks,
                )
            except Exception as exc:
                # GPU failures must never turn a candidate into a mathematical
                # exclusion. Recompute the complete batch on the proven CPU path.
                from .numba_backend import NumbaPackedEvaluator

                warnings.warn(
                    f"CUDA evaluation failed; falling back to Numba CPU: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                packed = NumbaPackedEvaluator(compute_workers).evaluate(
                    pack_layouts(layouts, columns),
                    max_reactor_ticks,
                )
    else:
        packed = None

    if packed is not None:
        if cancel_check is not None and cancel_check():
            return []
        return _candidate_results_from_packed(layouts, columns, packed)
    if compute_backend != "scalar":
        raise ValueError(f"unknown compute backend: {compute_backend}")

    results: list[CandidateResult] = []
    for layout in layouts:
        if cancel_check is not None and cancel_check():
            break
        if cancel_check is not None:
            # A cancellation callback already disables the certificate cache.
            # Keep the historical four-argument call shape for test/reference
            # evaluators that implement the scalar contract.
            result = evaluate_layout(layout, columns, max_reactor_ticks, cancel_check)
        else:
            result = evaluate_layout(
                layout,
                columns,
                max_reactor_ticks,
                use_certificate=use_certificate,
            )
        results.append(result)
        if cancel_check is not None and cancel_check():
            break
    return results


def _candidate_results_from_packed(
    layouts: tuple[tuple[str, ...], ...],
    columns: int,
    packed: PackedEvaluationBatch,
) -> list[CandidateResult]:
    """Reattach layouts and display metadata to a numeric evaluator result."""
    return [
        CandidateResult(
            layout=layout,
            mark=(
                decode_mark(
                    int(packed.mark_family[index]),
                    int(packed.mark_level[index]),
                    int(packed.mark_flags[index]),
                )
                or "未分类"
            ),
            average_eu_per_tick=float(packed.average_eu_per_tick[index]),
            total_eu=float(packed.total_eu[index]),
            safe_game_ticks=int(packed.safe_game_ticks[index]),
            safety_margin=float(packed.safety_margin[index]),
            component_count=len(layout) - layout.count("empty"),
            canonical=canonical_layout(layout, columns),
        )
        for index, layout in enumerate(layouts)
    ]


def _rank_candidates(
    values: list[CandidateResult],
    result_limit: int = 10,
) -> list[CandidateResult]:
    board: dict[str, CandidateResult] = {}
    for result in values:
        previous = board.get(result.canonical)
        if previous is None or result.score() > previous.score():
            board[result.canonical] = result
    ordered = sorted(board.values(), key=lambda item: item.canonical)
    ordered.sort(key=lambda item: item.score(), reverse=True)
    return ordered[:result_limit]


def _possible_unresolved_marks(max_reactor_ticks: int) -> set[str]:
    """Mark families still reachable after a horizon with no intervention."""
    possible = {"I", "II"}
    if max_reactor_ticks < FUEL_CYCLE_REACTOR_TICKS:
        possible.update(("III", "IV"))
    if max_reactor_ticks < TEN_PERCENT_CYCLE:
        possible.add("V")
    return possible


def _slice_packed_evaluations(
    values: PackedEvaluationBatch,
    start: int,
    end: int,
) -> PackedEvaluationBatch:
    """Copy one request's rows out of a combined GPU service response."""
    return PackedEvaluationBatch(
        mark_family=values.mark_family[start:end].copy(),
        mark_level=values.mark_level[start:end].copy(),
        mark_flags=values.mark_flags[start:end].copy(),
        stop_reason=values.stop_reason[start:end].copy(),
        reactor_ticks=values.reactor_ticks[start:end].copy(),
        safe_game_ticks=values.safe_game_ticks[start:end].copy(),
        average_eu_per_tick=values.average_eu_per_tick[start:end].copy(),
        total_eu=values.total_eu[start:end].copy(),
        safety_margin=values.safety_margin[start:end].copy(),
    )


def _cuda_evaluator_service(
    request_data: dict,
    request_queue,
    response_queues,
    cancel_event,
    failure_event,
    service_cpu_set_ids: tuple[int, ...] = (),
    pause_event=None,
) -> None:
    """Own the only CUDA context and serve packed batches from CPU producers."""
    initialize_gpu_service(service_cpu_set_ids)
    request = OptimizationRequest.model_validate(request_data)
    try:
        from .cuda_backend import CudaPackedEvaluator
        from .numba_backend import NumbaPackedEvaluator

        cuda_evaluator = CudaPackedEvaluator()
        cpu_evaluator = NumbaPackedEvaluator(
            max(1, min(request.cpu_workers, len(service_cpu_set_ids) or request.cpu_workers))
        )
        stop_after_group = False
        deferred_item = None
        while True:
            if _wait_for_worker_control(cancel_event, pause_event):
                return
            if deferred_item is None:
                item = request_queue.get()
            else:
                item = deferred_item
                deferred_item = None
            if item is None:
                return
            shard_id, batch_id, packed_batch, max_reactor_ticks = item
            group = [(shard_id, batch_id, packed_batch)]
            total_rows = packed_batch.batch_size

            # Let partial final batches from several producers coalesce into a
            # GPU-sized launch. Normal full batches pass through immediately.
            deadline = time.monotonic() + 0.02
            while total_rows < CUDA_MAX_BATCH_SIZE:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    next_item = request_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if next_item is None:
                    stop_after_group = True
                    break
                next_shard, next_batch_id, next_batch, next_ticks = next_item
                if (
                    next_ticks != max_reactor_ticks
                    or next_batch.columns != packed_batch.columns
                    or total_rows + next_batch.batch_size > CUDA_MAX_BATCH_SIZE
                ):
                    # Keep the mismatching request locally. Re-inserting into
                    # a bounded full queue could deadlock the sole consumer.
                    deferred_item = next_item
                    break
                group.append((next_shard, next_batch_id, next_batch))
                total_rows += next_batch.batch_size

            if _wait_for_worker_control(cancel_event, pause_event):
                for target_shard, target_batch, _ in group:
                    response_queues[target_shard].put(
                        (target_batch, None, "cancelled")
                    )
                if stop_after_group:
                    return
                continue

            if len(group) == 1:
                combined = packed_batch
            else:
                combined = PackedLayoutBatch(
                    columns=packed_batch.columns,
                    component_codes=np.ascontiguousarray(np.concatenate([
                        part.component_codes for _, _, part in group
                    ])),
                    initial_hull_heat=np.ascontiguousarray(np.concatenate([
                        part.initial_hull_heat for _, _, part in group
                    ])),
                )

            try:
                if (
                    combined.batch_size >= CUDA_MIN_BATCH_SIZE
                    and max_reactor_ticks <= CUDA_MAX_DIRECT_REACTOR_TICKS
                ):
                    packed_result = cuda_evaluator.evaluate(combined, max_reactor_ticks)
                else:
                    packed_result = cpu_evaluator.evaluate(combined, max_reactor_ticks)
            except Exception as exc:
                warnings.warn(
                    f"CUDA service failed; recomputing batch on Numba CPU: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                packed_result = cpu_evaluator.evaluate(combined, max_reactor_ticks)

            offset = 0
            for target_shard, target_batch, part in group:
                end = offset + part.batch_size
                response_queues[target_shard].put((
                    target_batch,
                    _slice_packed_evaluations(packed_result, offset, end),
                    None,
                ))
                offset = end
            if stop_after_group:
                return
    except BaseException:
        failure_event.set()
        raise


class _WorkerBatchEvaluator:
    """Route a shard's simulations locally or through the single GPU service."""

    def __init__(
        self,
        request: OptimizationRequest,
        shard_id: int,
        request_queue=None,
        response_queue=None,
        failure_event=None,
    ):
        self.request = request
        self.shard_id = shard_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.failure_event = failure_event
        self.sequence = 0

    def _local_numba(
        self,
        layouts: tuple[tuple[str, ...], ...],
        max_reactor_ticks: int,
        cancel_check,
    ) -> list[CandidateResult]:
        return evaluate_layout_batch(
            layouts,
            self.request.columns,
            max_reactor_ticks,
            False,
            cancel_check,
            "numba_cpu",
            1,
        )

    def __call__(
        self,
        layouts: tuple[tuple[str, ...], ...],
        max_reactor_ticks: int,
        cancel_check,
    ) -> list[CandidateResult]:
        if self.request_queue is None:
            compute_workers = (
                1
                if self.shard_id >= 0 and self.request.compute_backend == "numba_cpu"
                else self.request.cpu_workers
            )
            return evaluate_layout_batch(
                layouts,
                self.request.columns,
                max_reactor_ticks,
                False,
                cancel_check,
                self.request.compute_backend,
                compute_workers,
            )
        if cancel_check is not None and cancel_check():
            return []

        self.sequence += 1
        batch_id = self.sequence
        packed_input = pack_layouts(layouts, self.request.columns)
        while True:
            if cancel_check is not None and cancel_check():
                return []
            if self.failure_event is not None and self.failure_event.is_set():
                return self._local_numba(layouts, max_reactor_ticks, cancel_check)
            try:
                self.request_queue.put(
                    (
                        self.shard_id,
                        batch_id,
                        packed_input,
                        max_reactor_ticks,
                    ),
                    timeout=0.2,
                )
                break
            except queue.Full:
                continue
            except Exception:
                return self._local_numba(layouts, max_reactor_ticks, cancel_check)

        while True:
            if cancel_check is not None and cancel_check():
                return []
            if self.failure_event is not None and self.failure_event.is_set():
                return self._local_numba(layouts, max_reactor_ticks, cancel_check)
            try:
                response_id, packed_result, error = self.response_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if response_id != batch_id:
                raise RuntimeError("GPU service returned an out-of-order batch")
            if error is not None or packed_result is None:
                if error == "cancelled":
                    return []
                return self._local_numba(layouts, max_reactor_ticks, cancel_check)
            return _candidate_results_from_packed(
                layouts,
                self.request.columns,
                packed_result,
            )


def _unresolved_can_enter_boards(
    result: CandidateResult,
    boards: dict[str, list[CandidateResult]],
    max_reactor_ticks: int,
    result_limit: int = 10,
) -> bool:
    """Whether an unclassified layout can still enter a requested board.

    Before the first intervention the fuel/reflector skeleton is unchanged, so
    the average EU/t observed at the finite horizon is an upper bound on its
    eventual safe-period average.  Board floors therefore make a sound filter;
    equality remains competitive because canonical tie ordering may change.
    """
    for family in _possible_unresolved_marks(max_reactor_ticks).intersection(boards):
        board = boards[family]
        if (
            len(board) < result_limit
            or result.average_eu_per_tick >= board[-1].average_eu_per_tick
        ):
            return True
    return False


def _evaluate_search_batch(
    layouts: tuple[tuple[str, ...], ...],
    request: OptimizationRequest,
    boards: dict[str, list[CandidateResult]],
    cancel_check,
    batch_evaluator=None,
) -> list[CandidateResult]:
    """Evaluate once, then extend only still-competitive unclassified rows."""
    if batch_evaluator is None:
        batch_evaluator = _WorkerBatchEvaluator(request, -1)
    results = batch_evaluator(layouts, request.max_reactor_ticks, cancel_check)
    extension_limit = request.unresolved_max_reactor_ticks
    if (
        len(results) != len(layouts)
        or extension_limit is None
        or extension_limit <= request.max_reactor_ticks
    ):
        return results

    # Include newly classified base-horizon rows when deciding whether the
    # unresolved rows in this same batch can still enter a leaderboard.
    projected = {family: list(values) for family, values in boards.items()}
    for result in results:
        family = mark_family(result.mark)
        if family in projected:
            projected[family] = _rank_candidates(
                [*projected[family], result],
                request.result_limit,
            )

    extension_indices = [
        index
        for index, result in enumerate(results)
        if mark_family(result.mark) is None
        and _unresolved_can_enter_boards(
            result,
            projected,
            request.max_reactor_ticks,
            request.result_limit,
        )
    ]
    if not extension_indices:
        return results

    extension_layouts = tuple(layouts[index] for index in extension_indices)
    extended = batch_evaluator(extension_layouts, extension_limit, cancel_check)
    if len(extended) != len(extension_layouts):
        return []
    for index, result in zip(extension_indices, extended, strict=True):
        results[index] = result
    return results


def _run_mark_i_two_level_shard(
    request: OptimizationRequest,
    shard_id: int,
    fixed_items: tuple[tuple[int, str], ...],
    progress_queue,
    cancel_event,
    shared_power_floor=None,
    simulation_request_queue=None,
    simulation_response_queue=None,
    simulation_failure_event=None,
    pause_event=None,
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
    fixed_power_slots = sum(item != "empty" for _position, item in fixed_items)
    initial_skeleton_heat = skeleton_heat_per_tick(tuple(skeleton), request.columns)

    checked = 0
    pruned = 0
    evaluated = 0
    unresolved = 0
    unresolved_power_ceiling = -1.0
    visits = 0
    last_report = time.monotonic()
    cancelled = False
    cancel_cache = False
    last_cancel_check = 0.0
    shared_floor_cache = -1.0
    last_floor_check = 0.0
    boards: dict[str, list[CandidateResult]] = {"I": []}
    pending_layouts: list[tuple[str, ...]] = []
    batch_evaluator = _WorkerBatchEvaluator(
        request,
        shard_id,
        simulation_request_queue,
        simulation_response_queue,
        simulation_failure_event,
    )
    grid_edges = tuple(
        (index, neighbor)
        for index in range(slots)
        for neighbor in (
            *((index + 1,) if index % request.columns + 1 < request.columns else ()),
            *((index + request.columns,) if index + request.columns < slots else ()),
        )
    )
    grid_degrees = tuple(
        len(_layout_neighbors(index, request.columns, slots)) for index in range(slots)
    )

    def cancellation_requested() -> bool:
        nonlocal cancel_cache, last_cancel_check
        if pause_event is not None and pause_event.is_set():
            cancel_cache = _wait_for_worker_control(cancel_event, pause_event)
            last_cancel_check = time.monotonic()
            return cancel_cache
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

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 0.25:
            progress_queue.put(("progress", shard_id, checked, pruned, evaluated, unresolved))
            last_report = now

    def accept_result(result: CandidateResult) -> None:
        previous = boards["I"]
        ranked = _rank_candidates([*previous, result], request.result_limit)
        if ranked != previous:
            boards["I"] = ranked
            progress_queue.put(("candidate", result))

    def flush_pending_layouts() -> None:
        nonlocal evaluated, unresolved, unresolved_power_ceiling, cancelled
        if not pending_layouts or cancelled:
            return
        batch = tuple(pending_layouts)
        pending_layouts.clear()
        results = _evaluate_search_batch(
            batch,
            request,
            boards,
            cancellation_requested,
            batch_evaluator,
        )
        if cancellation_requested() or len(results) != len(batch):
            cancelled = True
            return
        for result in results:
            evaluated += 1
            family = mark_family(result.mark)
            if family == "I":
                accept_result(result)
            elif family is None:
                unresolved += 1
                unresolved_power_ceiling = max(
                    unresolved_power_ceiling,
                    result.average_eu_per_tick,
                )
        report()

    def power_increment(position: int, item: str) -> int:
        value = _power_vertex_value(item)
        row, column = divmod(position, request.columns)
        if column > 0:
            value += _power_edge_value(skeleton[position - 1], item)
        if row > 0:
            value += _power_edge_value(skeleton[position - request.columns], item)
        return value

    def optimistic_power_bound(position: int, current_power: int, current_rods: int) -> int:
        """Inventory-aware directed-contribution upper bound.

        An edge's EU contribution is the sum of one directed pulse term from
        each fuel endpoint whose neighbor is fuel/reflector.  Every unfinished
        fuel is optimistically given four such neighbors.  This preserves an
        upper bound while respecting remaining fuel packages and total rods;
        reflectors need not be allocated because assuming all neighbors are
        power-active only makes the bound larger.
        """
        upper = current_power

        # Directed terms from already processed fuels to unfinished neighbors.
        directed_per_rod = int(ReactorSimulator.EU_PER_PULSE)
        for first, second in grid_edges:
            if first < position <= second:
                first_spec = COMPONENTS[skeleton[first]]
                if first_spec.kind == "fuel":
                    upper += directed_per_rod * first_spec.rod_count

        variable_slots = 0
        for index in range(position, slots):
            if index not in fixed:
                variable_slots += 1
                continue
            item = skeleton[index]
            spec = COMPONENTS[item]
            if spec.kind == "fuel":
                upper += (
                    _power_vertex_value(item)
                    + directed_per_rod * spec.rod_count * grid_degrees[index]
                )

        remaining_fuels = [
            item for item in power_items
            if COMPONENTS[item].kind == "fuel" and power_remaining[item] > 0
        ]
        if request.fuel.mode == "separate":
            contributions: list[int] = []
            for item in remaining_fuels:
                spec = COMPONENTS[item]
                contribution = _power_vertex_value(item) + 4 * directed_per_rod * spec.rod_count
                contributions.extend([contribution] * min(power_remaining[item], variable_slots))
            contributions.sort(reverse=True)
            upper += sum(contributions[:variable_slots])
        else:
            remaining_rods = request.fuel.total_rods - current_rods
            # (used slots, used rods) -> best optimistic future contribution.
            dp: dict[tuple[int, int], int] = {(0, 0): 0}
            for item in remaining_fuels:
                spec = COMPONENTS[item]
                contribution = _power_vertex_value(item) + 4 * directed_per_rod * spec.rod_count
                next_dp: dict[tuple[int, int], int] = {}
                for (used_slots, used_rods), value in dp.items():
                    max_count = min(
                        power_remaining[item],
                        variable_slots - used_slots,
                        (remaining_rods - used_rods) // spec.rod_count,
                    )
                    for count in range(max_count + 1):
                        key = (
                            used_slots + count,
                            used_rods + count * spec.rod_count,
                        )
                        next_dp[key] = max(
                            next_dp.get(key, -1),
                            value + count * contribution,
                        )
                dp = next_dp
            upper += max(dp.values(), default=0)
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
                maximum = min(cap, remaining_positions - used)
                if (
                    request.fuel.usage == "exact"
                    and request.fuel.mode == "separate"
                    and spec.kind == "fuel"
                ):
                    counts = (cap,) if cap <= maximum else ()
                else:
                    counts = range(maximum + 1)
                for count in counts:
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
        for (additional_power, extra_rods, has_fuel), ways in dp.items():
            if not has_fuel:
                continue
            if (
                request.fuel.usage == "exact"
                and request.fuel.mode == "total_rods"
                and current_rods + extra_rods != request.fuel.total_rods
            ):
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
        nonlocal checked, pruned, evaluated, unresolved, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancellation_requested():
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
                sustainable_heat_flow_upper_bound(raw, request.columns, skeleton_heat)
                < skeleton_heat
            ):
                pruned += 1
                report()
                return
            pending_layouts.append(raw)
            if len(pending_layouts) >= _search_batch_size(request):
                flush_pending_layouts()
            report()
            return


        position = free_positions[offset]
        # Try occupied completions first so a strong incumbent is available
        # early. Empty remains the final branch; global completeness is intact.
        for item in [*cooling_items, "empty"]:
            if item != "empty" and cooling_remaining[item] > 0:
                cooling_remaining[item] -= 1
                layout[position] = item
                generate_cooling(layout, free_positions, offset + 1, skeleton_power, skeleton_heat)
                cooling_remaining[item] += 1
            elif item == "empty":
                layout[position] = "empty"
                generate_cooling(layout, free_positions, offset + 1, skeleton_power, skeleton_heat)
            if cancelled:
                break
        layout[position] = "empty"

    def finish_skeleton(skeleton_heat: int) -> None:
        nonlocal checked, pruned
        raw_skeleton = tuple(skeleton)
        free_positions = tuple(index for index, item in enumerate(raw_skeleton) if item == "empty")
        completion_count = count_cooling_completions(len(free_positions), cooling_caps)
        skeleton_power = skeleton_eu_per_tick(raw_skeleton, request.columns)
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
        current_power_slots: int,
        current_heat: int,
    ) -> None:
        nonlocal checked, pruned, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancellation_requested():
            cancelled = True
            return
        if cancelled:
            return
        remaining_positions = sum(
            index not in fixed for index in range(position, slots)
        )
        if not _fuel_requirement_feasible(
            request,
            power_remaining,
            current_rods,
            remaining_positions,
        ):
            return
        if _partial_mark_i_heat_infeasible(
            request,
            power_remaining,
            cooling_cap_items,
            current_rods,
            current_power_slots,
            remaining_positions,
            current_heat,
        ):
            count = count_remaining_layouts(
                position,
                current_rods,
                current_has_fuel,
            )
            checked += count
            pruned += count
            report()
            return
        floor = current_power_floor()
        if floor >= 0 and optimistic_power_bound(position, current_power, current_rods) < floor:
            count = count_remaining_layouts(position, current_rods, current_has_fuel)
            checked += count
            pruned += count
            report()
            return
        if position == slots:
            if _fuel_requirement_complete(
                request,
                power_remaining,
                current_rods,
                current_has_fuel,
            ):
                finish_skeleton(current_heat)
            return
        if position in fixed:
            generate_skeleton(
                position + 1,
                current_rods,
                current_has_fuel,
                current_power + power_increment(position, skeleton[position]),
                current_power_slots,
                current_heat,
            )
            return

        for item in [*power_items, "empty"]:
            if item == "empty":
                skeleton[position] = "empty"
                generate_skeleton(
                    position + 1,
                    current_rods,
                    current_has_fuel,
                    current_power,
                    current_power_slots,
                    current_heat,
                )
            elif power_remaining[item] > 0:
                rod_cost = COMPONENTS[item].rod_count
                if request.fuel.mode == "total_rods" and current_rods + rod_cost > request.fuel.total_rods:
                    continue
                heat_increment = _partial_skeleton_heat_increment(
                    skeleton,
                    position,
                    item,
                    request.columns,
                )
                power_remaining[item] -= 1
                skeleton[position] = item
                generate_skeleton(
                    position + 1,
                    current_rods + rod_cost,
                    current_has_fuel or rod_cost > 0,
                    current_power + power_increment(position, item),
                    current_power_slots + 1,
                    current_heat + heat_increment,
                )
                power_remaining[item] += 1
            if cancelled:
                break
        skeleton[position] = "empty"

    generate_skeleton(
        0,
        rods,
        has_fuel,
        0,
        fixed_power_slots,
        initial_skeleton_heat,
    )
    flush_pending_layouts()
    report(force=True)
    return {
        "shard_id": shard_id,
        "checked": checked,
        "pruned": pruned,
        "evaluated": evaluated,
        "unresolved": unresolved,
        "unresolved_power_ceiling": unresolved_power_ceiling,
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
    simulation_request_queue=None,
    simulation_response_queue=None,
    simulation_failure_event=None,
    pause_event=None,
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
            simulation_request_queue,
            simulation_response_queue,
            simulation_failure_event,
            pause_event,
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
    unresolved = 0
    unresolved_power_ceiling = -1.0
    visits = 0
    last_report = time.monotonic()
    cancelled = False
    cancel_cache = False
    last_cancel_check = 0.0
    boards: dict[str, list[CandidateResult]] = {mark: [] for mark in request.marks}
    pending_layouts: list[tuple[str, ...]] = []
    batch_evaluator = _WorkerBatchEvaluator(
        request,
        shard_id,
        simulation_request_queue,
        simulation_response_queue,
        simulation_failure_event,
    )

    def cancellation_requested() -> bool:
        nonlocal cancel_cache, last_cancel_check
        if pause_event is not None and pause_event.is_set():
            cancel_cache = _wait_for_worker_control(cancel_event, pause_event)
            last_cancel_check = time.monotonic()
            return cancel_cache
        now = time.monotonic()
        if now - last_cancel_check >= 0.1:
            cancel_cache = cancel_event.is_set()
            last_cancel_check = now
        return cancel_cache

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 0.25:
            progress_queue.put(("progress", shard_id, checked, pruned, evaluated, unresolved))
            last_report = now

    def flush_pending_layouts() -> None:
        nonlocal evaluated, unresolved, unresolved_power_ceiling, cancelled
        if not pending_layouts or cancelled:
            return
        batch = tuple(pending_layouts)
        pending_layouts.clear()
        results = _evaluate_search_batch(
            batch,
            request,
            boards,
            cancellation_requested,
            batch_evaluator,
        )
        if cancellation_requested() or len(results) != len(batch):
            cancelled = True
            return
        for result in results:
            evaluated += 1
            family = mark_family(result.mark)
            if family in boards:
                previous = boards[family]
                ranked = _rank_candidates(
                    [*previous, result],
                    request.result_limit,
                )
                # A later mirrored direction can replace an earlier result
                # without changing the canonical-key sequence.
                if ranked != previous:
                    boards[family] = ranked
                    progress_queue.put(("candidate", result))
            elif family is None:
                unresolved += 1
                unresolved_power_ceiling = max(
                    unresolved_power_ceiling,
                    result.average_eu_per_tick,
                )
        report()

    def generate(position: int, current_rods: int, current_has_fuel: bool) -> None:
        nonlocal checked, pruned, evaluated, unresolved, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancellation_requested():
            cancelled = True
            return
        remaining_positions = sum(
            index not in fixed for index in range(position, slots)
        )
        if not _fuel_requirement_feasible(
            request,
            remaining,
            current_rods,
            remaining_positions,
        ):
            return
        if position == slots:
            if not _fuel_requirement_complete(
                request,
                remaining,
                current_rods,
                current_has_fuel,
            ):
                return
            checked += 1
            pending_layouts.append(tuple(layout))
            if len(pending_layouts) >= _search_batch_size(request):
                flush_pending_layouts()
            report()
            return
        if cancelled:
            return
        if position in fixed:
            generate(position + 1, current_rods, current_has_fuel)
            return

        for item in [*allowed, "empty"]:
            if item != "empty":
                if cancelled or remaining[item] <= 0:
                    continue
                rod_cost = COMPONENTS[item].rod_count
                if request.fuel.mode == "total_rods" and current_rods + rod_cost > request.fuel.total_rods:
                    continue
                remaining[item] -= 1
                layout[position] = item
                generate(position + 1, current_rods + rod_cost, current_has_fuel or rod_cost > 0)
                remaining[item] += 1
            else:
                layout[position] = "empty"
                generate(position + 1, current_rods, current_has_fuel)
        layout[position] = "empty"

    generate(0, rods, has_fuel)
    flush_pending_layouts()
    report(force=True)
    return {
        "shard_id": shard_id,
        "checked": checked,
        "pruned": pruned,
        "evaluated": evaluated,
        "unresolved": unresolved,
        "unresolved_power_ceiling": unresolved_power_ceiling,
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
        self.unresolved = 0
        self.unresolved_power_ceiling = -1.0
        self.warm_start_evaluated = 0
        self.enumeration_processes = 0
        self.simulation_processes = 0
        self.reserved_cpu_cores = 0
        self.generation = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.message = "等待开始"
        self.error: str | None = None
        self.proven_global = False
        self.proven_within_horizon = False
        self.exhaustive_estimate = estimate_exhaustive_space(request) if request.solver == "exhaustive" else None
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.process_cancel_event = None
        self.process_pause_event = None
        self.paused_at: float | None = None
        self.paused_seconds = 0.0
        self.checkpoint_path = CHECKPOINT_DIRECTORY / f"{self.id}.json"
        self._heuristic_cache: dict[tuple[str, ...], CandidateResult | None] = {}
        self.leaderboards: dict[str, list[CandidateResult]] = {mark: [] for mark in request.marks}

    def _wait_if_paused(self) -> float:
        """Wait in the job thread and return wall time spent paused."""
        started = time.time() if self.pause_event.is_set() else None
        while self.pause_event.is_set() and not self.cancel_event.is_set():
            time.sleep(0.05)
        return 0.0 if started is None else time.time() - started

    def _control_requested(self) -> bool:
        self._wait_if_paused()
        return self.cancel_event.is_set()

    def persist_checkpoint(self) -> Path:
        """Atomically persist observable progress for audit and safe pause."""
        payload = {
            "schema_version": 2,
            "job_id": self.id,
            "checkpoint_kind": "process_local_pause",
            "saved_at": time.time(),
            "paused": self.status == "paused",
            "restart_resumable": False,
            "resume_requirement": "the original server and worker processes must remain alive",
            "request": self.request.model_dump(mode="json"),
            "snapshot": self.snapshot(),
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.checkpoint_path)
        return self.checkpoint_path

    def pause(self) -> None:
        if self.status != "running":
            raise ValueError("只有运行中的任务可以暂停")
        self.pause_event.set()
        if self.process_pause_event is not None:
            self.process_pause_event.set()
        self.paused_at = time.time()
        self.status = "paused"
        self.message = "任务已暂停，进度检查点已落盘"
        self.persist_checkpoint()

    def resume_in_place(self) -> None:
        if self.status != "paused":
            raise ValueError("任务当前没有暂停")
        now = time.time()
        if self.paused_at is not None:
            self.paused_seconds += now - self.paused_at
        self.paused_at = None
        self.status = "running"
        self.message = "任务已从内存检查点继续"
        self.pause_event.clear()
        if self.process_pause_event is not None:
            self.process_pause_event.clear()
        self.persist_checkpoint()

    def snapshot(self) -> dict:
        elapsed_end = self.finished_at or (
            self.paused_at if self.status == "paused" and self.paused_at is not None else time.time()
        )
        elapsed_seconds = (
            max(0.0, elapsed_end - self.started_at - self.paused_seconds)
            if self.started_at is not None
            else 0.0
        )
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "evaluated": self.evaluated,
            "checked": self.checked,
            "pruned": self.pruned,
            "unresolved": self.unresolved,
            "unresolved_power_ceiling": (
                self.unresolved_power_ceiling if self.unresolved else None
            ),
            "warm_start_evaluated": self.warm_start_evaluated,
            "generation": self.generation,
            "message": self.message,
            "error": self.error,
            "proven_global": self.proven_global,
            "proven_within_horizon": self.proven_within_horizon,
            "estimate": str(self.exhaustive_estimate) if self.exhaustive_estimate is not None else None,
            "cpu_workers": self.request.cpu_workers,
            "compute_backend": self.request.compute_backend,
            "enumeration_processes": self.enumeration_processes,
            "simulation_processes": self.simulation_processes,
            "reserved_cpu_cores": self.reserved_cpu_cores,
            "elapsed_seconds": elapsed_seconds,
            "paused_at": self.paused_at,
            "checkpoint_path": (
                str(self.checkpoint_path)
                if self.status == "paused" or self.checkpoint_path.exists()
                else None
            ),
            "restart_resumable": False,
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
            if (
                self.request.fuel.usage == "exact"
                and rods != self.request.fuel.total_rods
            ) or rods > self.request.fuel.total_rods:
                return False
        else:
            expected = {
                "uranium_single": self.request.fuel.single,
                "uranium_dual": self.request.fuel.dual,
                "uranium_quad": self.request.fuel.quad,
            }
            for item, limit in expected.items():
                count = counts.get(item, 0)
                if (
                    self.request.fuel.usage == "exact"
                    and count != limit
                ) or count > limit:
                    return False
        return all(counts.get(item, 0) <= limit for item, limit in self.request.component_limits.items()) and all(
            item in {"empty", "uranium_single", "uranium_dual", "uranium_quad"} or item in self.request.component_limits
            for item in layout
        )

    def _random_layout(self, rng: random.Random) -> tuple[str, ...]:
        slots = self.request.columns * 6
        if self.request.fuel.usage == "exact":
            result = ["empty"] * slots
            if self.request.fuel.mode == "separate":
                fuel_items = [
                    *(["uranium_single"] * self.request.fuel.single),
                    *(["uranium_dual"] * self.request.fuel.dual),
                    *(["uranium_quad"] * self.request.fuel.quad),
                ]
            else:
                fuel_items = []
                remaining_rods = self.request.fuel.total_rods
                while remaining_rods > 0:
                    slots_left = slots - len(fuel_items)
                    choices = [
                        item
                        for item in FUEL_ITEMS
                        if COMPONENTS[item].rod_count <= remaining_rods
                        and (
                            remaining_rods - COMPONENTS[item].rod_count + 3
                        ) // 4 <= slots_left - 1
                    ]
                    item = rng.choice(choices)
                    fuel_items.append(item)
                    remaining_rods -= COMPONENTS[item].rod_count

            positions = list(range(slots))
            rng.shuffle(positions)
            for item, position in zip(fuel_items, positions, strict=False):
                result[position] = item
            free_positions = positions[len(fuel_items):]
            nonfuel_remaining = {
                item: limit
                for item, limit in self.request.component_limits.items()
                if limit > 0
            }
            # Exact layouts also seed large exhaustive searches. Fill every
            # remaining slot that inventory permits so the warm start tests
            # dense cooling arrangements first without changing any cap.
            while free_positions:
                choices = [
                    item for item, count in nonfuel_remaining.items() if count > 0
                ]
                if not choices:
                    break
                item = rng.choice(choices)
                result[free_positions.pop()] = item
                nonfuel_remaining[item] -= 1
            return tuple(result)

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
            if self.request.fuel.usage == "exact":
                mutable = [
                    index
                    for index, item in enumerate(result)
                    if COMPONENTS[item].kind != "fuel"
                ]
                values = [
                    "empty",
                    *(item for item, limit in self.request.component_limits.items() if limit > 0),
                ]
                if mutable:
                    result[rng.choice(mutable)] = rng.choice(values)
            else:
                values = ["empty", *self._fuel_allowed(), *(item for item, limit in self.request.component_limits.items() if limit > 0)]
                result[rng.randrange(len(result))] = rng.choice(values)
        candidate = tuple(result)
        return candidate if self._within_limits(candidate) and any(COMPONENTS[x].kind == "fuel" for x in candidate) else layout

    def _evaluate(self, layout: tuple[str, ...]) -> CandidateResult | None:
        if self._control_requested() or not self._within_limits(layout):
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
        island_count = min(self.request.cpu_workers, max(1, self.request.population // 10))
        base_size, remainder = divmod(self.request.population, island_count)
        islands = [
            [self._random_layout(rng) for _ in range(base_size + (island < remainder))]
            for island in range(island_count)
        ]
        deadline = time.time() + self.request.time_budget_seconds

        def should_stop() -> bool:
            nonlocal deadline
            deadline += self._wait_if_paused()
            return self.cancel_event.is_set() or time.time() >= deadline

        use_packed_backend = self.request.compute_backend in {"numba_cpu", "cuda"}
        use_search_batches = use_packed_backend or (
            self.request.unresolved_max_reactor_ticks is not None
            and self.request.unresolved_max_reactor_ticks
            > self.request.max_reactor_ticks
        )
        executor = (
            ProcessPoolExecutor(max_workers=self.request.cpu_workers)
            if self.request.cpu_workers > 1 and not use_packed_backend
            else None
        )
        for generation in range(self.request.generations):
            if should_stop():
                break
            self.generation = generation + 1
            population = [layout for island in islands for layout in island]
            unique_population = list(dict.fromkeys(population))
            scored: list[tuple[tuple, tuple[str, ...]]] = []
            if executor is None and not use_search_batches:
                for layout in unique_population:
                    if should_stop():
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
                batch_size = max(
                    1,
                    min(
                        _search_batch_size(self.request),
                        (
                            len(unseen)
                            if use_packed_backend
                            else math.ceil(len(unseen) / max(1, self.request.cpu_workers * 4))
                        ),
                    ),
                )
                batches = [
                    tuple(unseen[offset:offset + batch_size])
                    for offset in range(0, len(unseen), batch_size)
                ]
                if executor is None:
                    for batch in batches:
                        if should_stop():
                            break
                        results = _evaluate_search_batch(
                            batch,
                            self.request,
                            self.leaderboards,
                            should_stop,
                        )
                        for result in results:
                            layout = result.layout
                            accepted = result if mark_family(result.mark) in self.request.marks else None
                            self._heuristic_cache[layout] = accepted
                            self._accept(accepted)
                            scored.append((accepted.score() if accepted else (-1,), layout))
                else:
                    board_snapshot = {
                        family: list(values)
                        for family, values in self.leaderboards.items()
                    }
                    futures = {
                        executor.submit(
                            _evaluate_search_batch,
                            batch,
                            self.request,
                            board_snapshot,
                            None,
                        ): batch
                        for batch in batches
                    }
                    try:
                        for future in as_completed(futures, timeout=max(0.01, deadline - time.time())):
                            if should_stop():
                                break
                            for result in future.result():
                                layout = result.layout
                                accepted = result if mark_family(result.mark) in self.request.marks else None
                                self._heuristic_cache[layout] = accepted
                                self._accept(accepted)
                                scored.append((accepted.score() if accepted else (-1,), layout))
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
        if (
            mark_i_two_level
            and self.request.fuel.usage == "exact"
            and estimate >= 100_000
            and self.request.component_limits
            and not self._control_requested()
        ):
            self.message = "正在用满布局样本建立全局穷举功率下界"
            rng = random.Random(self.request.seed)
            layouts: set[tuple[str, ...]] = set()
            for _ in range(EXHAUSTIVE_WARM_START_LAYOUTS * 8):
                layouts.add(self._random_layout(rng))
                if len(layouts) >= EXHAUSTIVE_WARM_START_LAYOUTS:
                    break
            warm_results = evaluate_layout_batch(
                tuple(layouts),
                self.request.columns,
                self.request.max_reactor_ticks,
                False,
                self._control_requested,
                self.request.compute_backend,
                self.request.cpu_workers,
            )
            self.warm_start_evaluated = len(warm_results)
            for result in warm_results:
                if mark_family(result.mark) == "I":
                    self._accept(result, count_evaluation=False)
            if self._control_requested():
                return
        cpu_plan = cpu_scheduling_plan(2)
        scheduling_capacity = (
            len(cpu_plan.worker_cpu_set_ids)
            if cpu_plan.worker_cpu_set_ids
            else max(1, cpu_plan.available_logical_processors - 2)
        )
        enumeration_capacity = min(self.request.cpu_workers, scheduling_capacity)
        shards = _exhaustive_shards(
            self.request,
            power_only=mark_i_two_level,
            target_shards=enumeration_capacity * 4,
        )
        # Every exhaustive backend uses multiple enumeration/pruning
        # processes. Numba runs one compute thread inside each producer;
        # CUDA producers instead feed one dedicated GPU process.
        worker_count = min(enumeration_capacity, len(shards))
        self.enumeration_processes = worker_count
        self.simulation_processes = int(self.request.compute_backend == "cuda")
        self.reserved_cpu_cores = min(2, max(0, cpu_plan.available_physical_cores - 1))
        request_data = self.request.model_dump(mode="json")
        shard_progress: dict[int, tuple[int, int, int, int]] = {}
        any_worker_cancelled = False

        def update_progress(
            shard_id: int,
            checked: int,
            pruned: int,
            evaluated: int,
            unresolved: int,
        ) -> None:
            old_checked, old_pruned, old_evaluated, old_unresolved = shard_progress.get(
                shard_id, (0, 0, 0, 0)
            )
            # Queue delivery and ProcessPool future completion are independent.
            # A delayed progress snapshot may arrive after the final worker
            # result, so every per-shard counter must remain monotonic.
            checked = max(checked, old_checked)
            pruned = max(pruned, old_pruned)
            evaluated = max(evaluated, old_evaluated)
            unresolved = max(unresolved, old_unresolved)
            self.checked += checked - old_checked
            self.pruned += pruned - old_pruned
            self.evaluated += evaluated - old_evaluated
            self.unresolved += unresolved - old_unresolved
            shard_progress[shard_id] = (checked, pruned, evaluated, unresolved)
            self.progress = min(0.999, self.checked / total)
            worker_label = (
                f"{worker_count} CPU 枚举进程 + 1 GPU 模拟服务"
                if self.request.compute_backend == "cuda"
                else f"{worker_count} 进程并行枚举"
            )
            self.message = (
                f"{worker_label} · 已检查 {self.checked:,} 个方案"
                f" · 热学模拟 {self.evaluated:,} 个 · 数学跳过 {self.pruned:,} 个"
                f" · 未决 {self.unresolved:,} 个"
            )

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        self.process_cancel_event = manager.Event()
        self.process_pause_event = manager.Event()
        initial_power_floor = (
            self.leaderboards["I"][-1].average_eu_per_tick
            if mark_i_two_level
            and len(self.leaderboards["I"]) >= self.request.result_limit
            else -1.0
        )
        shared_power_floor = (
            manager.Value("d", initial_power_floor) if mark_i_two_level else None
        )
        simulation_request_queue = None
        simulation_response_queues = None
        simulation_failure_event = None
        simulation_process = None
        if self.request.compute_backend == "cuda":
            simulation_request_queue = manager.Queue(maxsize=max(4, worker_count * 2))
            simulation_response_queues = [manager.Queue(maxsize=2) for _ in shards]
            simulation_failure_event = manager.Event()
            simulation_process = multiprocessing.get_context("spawn").Process(
                target=_cuda_evaluator_service,
                args=(
                    request_data,
                    simulation_request_queue,
                    simulation_response_queues,
                    self.process_cancel_event,
                    simulation_failure_event,
                    cpu_plan.reserved_cpu_set_ids,
                    self.process_pause_event,
                ),
                name="ic2-cuda-evaluator",
            )
            simulation_process.start()
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=(
                multiprocessing.get_context("spawn")
                if self.request.compute_backend == "cuda"
                else None
            ),
            initializer=(
                initialize_compute_worker
                if cpu_plan.worker_cpu_set_ids
                else None
            ),
            initargs=(
                (cpu_plan.worker_cpu_set_ids,)
                if cpu_plan.worker_cpu_set_ids
                else ()
            ),
        )
        futures = {
            executor.submit(
                _run_exhaustive_shard,
                request_data,
                shard_id,
                shard,
                progress_queue,
                self.process_cancel_event,
                shared_power_floor,
                simulation_request_queue,
                (
                    simulation_response_queues[shard_id]
                    if simulation_response_queues is not None
                    else None
                ),
                simulation_failure_event,
                self.process_pause_event,
            ): shard_id
            for shard_id, shard in enumerate(shards)
        }
        pending = set(futures)

        def handle_message(message: tuple) -> None:
            if message[0] == "progress":
                _, shard_id, checked, pruned, evaluated, unresolved = message
                update_progress(shard_id, checked, pruned, evaluated, unresolved)
            elif message[0] == "candidate":
                self._accept(message[1], count_evaluation=False)
                if (
                    shared_power_floor is not None
                    and len(self.leaderboards["I"]) >= self.request.result_limit
                ):
                    shared_power_floor.value = self.leaderboards["I"][-1].average_eu_per_tick

        try:
            while pending:
                if (
                    simulation_process is not None
                    and not simulation_process.is_alive()
                    and simulation_process.exitcode is not None
                    and simulation_failure_event is not None
                ):
                    simulation_failure_event.set()
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
                    any_worker_cancelled = any_worker_cancelled or result["cancelled"]
                    self.unresolved_power_ceiling = max(
                        self.unresolved_power_ceiling,
                        result["unresolved_power_ceiling"],
                    )
                    update_progress(
                        result["shard_id"],
                        result["checked"],
                        result["pruned"],
                        result["evaluated"],
                        result["unresolved"],
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
            counts_complete = (
                self.checked == estimate
                and self.evaluated + self.pruned == self.checked
            )
            self.proven_within_horizon = (
                not self.cancel_event.is_set()
                and not any_worker_cancelled
                and counts_complete
            )
            unresolved_can_match = False
            if self.unresolved:
                possible = _possible_unresolved_marks(self.request.max_reactor_ticks)
                for family in possible.intersection(self.request.marks):
                    board = self.leaderboards[family]
                    if (
                        len(board) < self.request.result_limit
                        or self.unresolved_power_ceiling
                        >= board[-1].average_eu_per_tick
                    ):
                        unresolved_can_match = True
                        break
            self.proven_global = self.proven_within_horizon and not unresolved_can_match
        finally:
            if self.process_cancel_event is not None:
                self.process_cancel_event.set()
            if self.process_pause_event is not None:
                self.process_pause_event.clear()
            if simulation_request_queue is not None:
                try:
                    simulation_request_queue.put_nowait(None)
                except Exception:
                    pass
            executor.shutdown(wait=True, cancel_futures=True)
            if simulation_process is not None:
                simulation_process.join(timeout=10)
                if simulation_process.is_alive():
                    simulation_process.terminate()
                    simulation_process.join(timeout=5)
            self.process_cancel_event = None
            self.process_pause_event = None
            manager.shutdown()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()
        if self.process_cancel_event is not None:
            self.process_cancel_event.set()
        if self.process_pause_event is not None:
            self.process_pause_event.clear()

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
            if self.status in {"completed", "cancelled", "failed"}:
                self.persist_checkpoint()


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
        if job.status == "paused":
            job.resume_in_place()
            return job
        if job.request.solver != "heuristic":
            raise ValueError("穷举任务不能续算；请新建任务并完整枚举")
        if job.status in {"queued", "running"}:
            raise ValueError("任务仍在运行")
        job.cancel_event.clear()
        job.status = "queued"
        job.progress = 0.0
        job.proven_global = False
        job.proven_within_horizon = False
        job.message = "准备继续改进当前候选"
        threading.Thread(target=job.run, name=f"optimizer-{job.id[:8]}-resume", daemon=True).start()
        return job
