"""Exact bounded-width counting DP for aggregate active-degree signatures."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .mathematical_model import AggregatePattern, ReactorProblem


@dataclass(frozen=True, slots=True)
class DegreeSignatureCount:
    count: int | None
    proven: bool
    states_visited: int
    peak_layer_states: int
    transitions: int
    frontier_width: int
    elapsed_seconds: float
    stop_reason: str


class RectangularDegreeSignatureCounter:
    """Count abstract power skeletons matching one aggregate pattern exactly."""

    def __init__(self, problem: ReactorProblem) -> None:
        graph = problem.graph
        if graph.rows is None or graph.columns is None:
            raise ValueError("degree-signature DP requires rectangular metadata")
        expected = problem.graph.rectangular(graph.rows, graph.columns)
        if graph.edges != expected.edges:
            raise ValueError("degree-signature DP requires standard rectangular adjacency")
        self.problem = problem
        self.rows = graph.rows
        self.columns = graph.columns
        self.column_major = self.rows <= self.columns
        self.width = min(self.rows, self.columns)
        self.length = max(self.rows, self.columns)

    def _position(self, step: int) -> tuple[int, int, int]:
        major, minor = divmod(step, self.width)
        if self.column_major:
            row, column = minor, major
        else:
            row, column = major, minor
        return row * self.columns + column, major, minor

    def count(
        self,
        pattern: AggregatePattern,
        *,
        time_limit_seconds: float | None = None,
    ) -> DegreeSignatureCount:
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
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
        target_by_label = {
            item: sum(
                count
                for state_item, _degree, count in requested
                if state_item == item
            )
            for item in fuel_ids
        }
        fuel_cells = sum(initial_remaining)
        unknown_active = pattern.active_cells - fuel_cells
        if unknown_active < 0 or pattern.active_cells > self.problem.graph.size:
            raise ValueError("pattern active-cell count is inconsistent")

        # Code zero is inactive, fuel codes follow, and the final code is one
        # abstract active reflector/unknown label when needed.
        code_by_fuel = {item: index + 1 for index, item in enumerate(fuel_ids)}
        fuel_by_code = {code: item for item, code in code_by_fuel.items()}
        unknown_code = len(fuel_ids) + 1 if unknown_active else None
        choices = (0, *fuel_by_code.keys(), *((unknown_code,) if unknown_code else ()))
        key_index = {key: index for index, key in enumerate(state_keys)}

        def active(code: int) -> int:
            return int(code != 0)

        def finalize(
            entry: tuple[int, int],
            remaining: tuple[int, ...],
            unknown_remaining: int,
        ) -> tuple[tuple[int, ...], int] | None:
            code, degree = entry
            if code == 0:
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

        initial_frontier = ((0, 0),) * self.width
        # State -> number of prefixes producing it.
        layer: dict[tuple, int] = {
            (initial_frontier, initial_remaining, unknown_active): 1
        }
        states_visited = peak = transitions = 0

        for step in range(self.problem.graph.size):
            if deadline is not None and perf_counter() >= deadline:
                return DegreeSignatureCount(
                    count=None,
                    proven=False,
                    states_visited=states_visited,
                    peak_layer_states=max(peak, len(layer)),
                    transitions=transitions,
                    frontier_width=self.width,
                    elapsed_seconds=perf_counter() - started,
                    stop_reason="time_limit",
                )
            _vertex, major, minor = self._position(step)
            following: dict[tuple, int] = {}
            states_visited += len(layer)
            peak = max(peak, len(layer))
            for (frontier, remaining, unknown_remaining), prefix_count in layer.items():
                for code in choices:
                    transitions += 1
                    values = list(frontier)
                    old_code, old_degree = values[minor]
                    current_active = active(code)
                    if minor > 0 and current_active:
                        previous_code, previous_degree = values[minor - 1]
                        if previous_code != 0:
                            values[minor - 1] = (
                                previous_code,
                                previous_degree + 1,
                            )
                    next_remaining = remaining
                    next_unknown = unknown_remaining
                    if major > 0:
                        finished = finalize(
                            (old_code, old_degree + current_active),
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
                        if code != 0
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
                    key = (next_frontier, next_remaining, next_unknown)
                    following[key] = following.get(key, 0) + prefix_count
            layer = following

        total = 0
        states_visited += len(layer)
        peak = max(peak, len(layer))
        for (frontier, remaining, unknown_remaining), prefix_count in layer.items():
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
                total += prefix_count

        return DegreeSignatureCount(
            count=total,
            proven=True,
            states_visited=states_visited,
            peak_layer_states=peak,
            transitions=transitions,
            frontier_width=self.width,
            elapsed_seconds=perf_counter() - started,
            stop_reason="complete",
        )
