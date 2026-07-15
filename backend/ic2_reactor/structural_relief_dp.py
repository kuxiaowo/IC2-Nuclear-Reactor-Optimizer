"""Exact bounded-width DP for geometry-aware aggregate relief maxima.

The DP scans a rectangular graph along its shorter side.  It never emits a
power skeleton.  Instead it embeds one aggregate active-degree signature and
simultaneously chooses ordinary free cells versus side-vent cells.  The value
is the exact number of directed side-vent-to-ordinary edges.  Optimistic
exchanger cells are added afterwards by a one-dimensional knapsack because
their placement and routing are deliberately relaxed.

Consequently the final relief is an upper bound for every real cooling layout
with the aggregate signature.  A value below the required relief closes the
entire signature; a larger value is only a relaxation witness.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .mathematical_model import AggregatePattern, ReactorProblem


@dataclass(frozen=True, slots=True)
class StructuralReliefDPResult:
    proven: bool
    maximum_relief: int | None
    excluded: bool | None
    best_side_vent_cells: int | None
    best_effective_side_edges: int | None
    best_exchanger_cells: int | None
    side_edge_frontier: tuple[tuple[int, int], ...]
    states_visited: int
    peak_layer_states: int
    transitions: int
    frontier_width: int
    elapsed_seconds: float
    stop_reason: str


class RectangularStructuralReliefDP:
    """Maximise the structural relief relaxation in one bounded-width pass."""

    def __init__(self, problem: ReactorProblem) -> None:
        graph = problem.graph
        if graph.rows is None or graph.columns is None:
            raise ValueError("structural relief DP requires rectangular metadata")
        expected = graph.rectangular(graph.rows, graph.columns)
        if graph.edges != expected.edges:
            raise ValueError("structural relief DP requires standard rectangular adjacency")
        self.problem = problem
        self.rows = graph.rows
        self.columns = graph.columns
        self.column_major = self.rows <= self.columns
        self.width = min(self.rows, self.columns)

    def _position(self, step: int) -> tuple[int, int]:
        major, minor = divmod(step, self.width)
        return major, minor

    def solve(
        self,
        pattern: AggregatePattern,
        *,
        ideal_receiver_sink: int = 20,
        side_rate: int = 4,
        exchanger_slot_cost: int = 20,
        exchanger_relief: int = 72,
        time_limit_seconds: float | None = None,
    ) -> StructuralReliefDPResult:
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        if min(
            ideal_receiver_sink,
            side_rate,
            exchanger_slot_cost,
            exchanger_relief,
        ) <= 0:
            raise ValueError("relief capacities must be positive")
        minimum_side_cost = (
            ideal_receiver_sink - side_rate * self.problem.graph.maximum_degree
        )
        if minimum_side_cost <= 0:
            raise ValueError(
                "bounded side-vent count requires sink > side_rate*maximum_degree"
            )
        if pattern.slack < 0:
            raise ValueError("aggregate pattern slack must be non-negative")

        started = perf_counter()
        deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )
        requested = tuple(pattern.fuel_degree_counts)
        fuel_ids = tuple(dict.fromkeys(item for item, _degree, _count in requested))
        known_fuels = {
            item.id for item in self.problem.power_components if item.rods > 0
        }
        if unknown := set(fuel_ids) - known_fuels:
            raise ValueError(f"pattern has unknown fuel labels: {sorted(unknown)}")
        state_keys = tuple((item, degree) for item, degree, _count in requested)
        if len(state_keys) != len(set(state_keys)):
            raise ValueError("pattern repeats a fuel-degree state")
        if any(
            count < 0
            or degree < 0
            or degree > self.problem.graph.maximum_degree
            for _item, degree, count in requested
        ):
            raise ValueError("pattern has an invalid degree or count")
        initial_remaining = tuple(count for _item, _degree, count in requested)
        fuel_cells = sum(initial_remaining)
        unknown_active = pattern.active_cells - fuel_cells
        if unknown_active < 0 or pattern.active_cells > self.problem.graph.size:
            raise ValueError("pattern active-cell count is inconsistent")

        ordinary_code = 0
        code_by_fuel = {item: index + 1 for index, item in enumerate(fuel_ids)}
        fuel_by_code = {code: item for item, code in code_by_fuel.items()}
        next_code = len(fuel_ids) + 1
        unknown_code = next_code if unknown_active else None
        if unknown_code is not None:
            next_code += 1
        side_code = next_code
        choices = (
            ordinary_code,
            side_code,
            *fuel_by_code.keys(),
            *((unknown_code,) if unknown_code is not None else ()),
        )
        active_codes = frozenset((*fuel_by_code.keys(), *((unknown_code,) if unknown_code is not None else ())))
        key_index = {key: index for index, key in enumerate(state_keys)}
        maximum_side_cells = min(
            self.problem.graph.size - pattern.active_cells,
            pattern.slack // minimum_side_cost,
        )

        def active(code: int) -> int:
            return int(code in active_codes)

        def side_edge(first: int, second: int) -> int:
            return int(
                (first == side_code and second == ordinary_code)
                or (second == side_code and first == ordinary_code)
            )

        def finalize(
            entry: tuple[int, int],
            remaining: tuple[int, ...],
            unknown_remaining: int,
        ) -> tuple[tuple[int, ...], int] | None:
            code, degree = entry
            if code in (ordinary_code, side_code):
                return remaining, unknown_remaining
            if code == unknown_code:
                if unknown_remaining <= 0:
                    return None
                return remaining, unknown_remaining - 1
            item = fuel_by_code[code]
            index = key_index.get((item, degree))
            if index is None or remaining[index] <= 0:
                return None
            values = list(remaining)
            values[index] -= 1
            return tuple(values), unknown_remaining

        initial_frontier = ((ordinary_code, 0),) * self.width
        # State -> maximum number of already closed effective side edges.
        layer: dict[tuple, int] = {
            (initial_frontier, initial_remaining, unknown_active, 0): 0
        }
        states_visited = peak = transitions = 0

        for step in range(self.problem.graph.size):
            if deadline is not None and perf_counter() >= deadline:
                return StructuralReliefDPResult(
                    proven=False,
                    maximum_relief=None,
                    excluded=None,
                    best_side_vent_cells=None,
                    best_effective_side_edges=None,
                    best_exchanger_cells=None,
                    side_edge_frontier=(),
                    states_visited=states_visited,
                    peak_layer_states=max(peak, len(layer)),
                    transitions=transitions,
                    frontier_width=self.width,
                    elapsed_seconds=perf_counter() - started,
                    stop_reason="time_limit",
                )
            major, minor = self._position(step)
            following: dict[tuple, int] = {}
            states_visited += len(layer)
            peak = max(peak, len(layer))
            for (
                frontier,
                remaining,
                unknown_remaining,
                side_cells,
            ), edge_value in layer.items():
                for code in choices:
                    transitions += 1
                    next_side_cells = side_cells + int(code == side_code)
                    if next_side_cells > maximum_side_cells:
                        continue
                    values = list(frontier)
                    old_code, old_degree = values[minor]
                    increment = 0
                    if major > 0:
                        increment += side_edge(old_code, code)
                    if minor > 0:
                        left_code, left_degree = values[minor - 1]
                        increment += side_edge(left_code, code)
                        if active(code) and active(left_code):
                            values[minor - 1] = (left_code, left_degree + 1)

                    next_remaining = remaining
                    next_unknown = unknown_remaining
                    if major > 0:
                        finished = finalize(
                            (old_code, old_degree + active(code)),
                            next_remaining,
                            next_unknown,
                        )
                        if finished is None:
                            continue
                        next_remaining, next_unknown = finished
                    partial_degree = (
                        (
                            (active(old_code) if major > 0 else 0)
                            + (active(values[minor - 1][0]) if minor > 0 else 0)
                        )
                        if active(code)
                        else 0
                    )
                    values[minor] = (code, partial_degree)
                    next_frontier = tuple(values)

                    valid = True
                    for fuel_code, item in fuel_by_code.items():
                        unfinalized = sum(
                            entry_code == fuel_code
                            for entry_code, _degree in next_frontier
                        )
                        remaining_for_item = sum(
                            next_remaining[index]
                            for index, (state_item, _degree) in enumerate(state_keys)
                            if state_item == item
                        )
                        if unfinalized > remaining_for_item:
                            valid = False
                            break
                    if not valid:
                        continue
                    if unknown_code is not None and sum(
                        entry_code == unknown_code
                        for entry_code, _degree in next_frontier
                    ) > next_unknown:
                        continue
                    key = (
                        next_frontier,
                        next_remaining,
                        next_unknown,
                        next_side_cells,
                    )
                    following[key] = max(
                        following.get(key, -1),
                        edge_value + increment,
                    )
            layer = following

        side_frontier: dict[int, int] = {}
        states_visited += len(layer)
        peak = max(peak, len(layer))
        for (
            frontier,
            remaining,
            unknown_remaining,
            side_cells,
        ), edge_value in layer.items():
            valid_remaining = remaining
            valid_unknown = unknown_remaining
            valid = True
            for entry in frontier:
                finished = finalize(entry, valid_remaining, valid_unknown)
                if finished is None:
                    valid = False
                    break
                valid_remaining, valid_unknown = finished
            if valid and not any(valid_remaining) and valid_unknown == 0:
                side_frontier[side_cells] = max(
                    side_frontier.get(side_cells, -1),
                    edge_value,
                )

        best: tuple[int, int, int, int] | None = None
        free_cells = self.problem.graph.size - pattern.active_cells
        for side_cells, effective_edges in side_frontier.items():
            side_loss = ideal_receiver_sink * side_cells - side_rate * effective_edges
            if side_loss > pattern.slack:
                continue
            maximum_exchangers = min(
                free_cells - side_cells,
                (pattern.slack - side_loss) // exchanger_slot_cost,
            )
            relief = side_rate * effective_edges + exchanger_relief * maximum_exchangers
            candidate = (relief, side_cells, effective_edges, maximum_exchangers)
            if best is None or candidate > best:
                best = candidate

        maximum_relief = None if best is None else best[0]
        return StructuralReliefDPResult(
            proven=True,
            maximum_relief=maximum_relief,
            excluded=(
                True
                if maximum_relief is None
                else maximum_relief < pattern.required_relief
            ),
            best_side_vent_cells=(None if best is None else best[1]),
            best_effective_side_edges=(None if best is None else best[2]),
            best_exchanger_cells=(None if best is None else best[3]),
            side_edge_frontier=tuple(sorted(side_frontier.items())),
            states_visited=states_visited,
            peak_layer_states=peak,
            transitions=transitions,
            frontier_width=self.width,
            elapsed_seconds=perf_counter() - started,
            stop_reason="complete",
        )
