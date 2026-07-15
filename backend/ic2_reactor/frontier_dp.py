"""Exact bounded-width frontier dynamic program for rectangular power models.

The state width is ``min(rows, columns)``.  No reactor size, rod count or
component label is hard-coded; the DP consumes :class:`ReactorProblem` data.
Only pulse-producing labels are represented.  Every cooling-layer choice is
the single ``empty`` power label and is delegated to the thermal subproblem.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import heapq
from time import perf_counter

from .mathematical_model import ReactorProblem, evaluate_power_skeleton


NEGATIVE_INFINITY = -10**18


@dataclass(frozen=True, slots=True)
class FrontierDPResult:
    feasible: bool
    proven_optimal: bool
    maximum_power: int | None
    skeleton: tuple[str, ...] | None
    states: int
    frontier_width: int
    scan_length: int
    elapsed_seconds: float
    reason: str


@dataclass(frozen=True, slots=True)
class RankedPowerSkeleton:
    power: int
    skeleton: tuple[str, ...]


class RectangularFrontierPowerDP:
    """Exact point/edge objective DP parameterised by the shorter grid side."""

    def __init__(
        self,
        problem: ReactorProblem,
        *,
        fixed_labels: Mapping[int, str] | None = None,
    ) -> None:
        graph = problem.graph
        if graph.rows is None or graph.columns is None:
            raise ValueError("frontier DP requires rectangular graph metadata")
        if graph.size != graph.rows * graph.columns:
            raise ValueError("inconsistent rectangular graph dimensions")
        expected_edges = set()
        for row in range(graph.rows):
            for column in range(graph.columns):
                vertex = row * graph.columns + column
                if column + 1 < graph.columns:
                    expected_edges.add((vertex, vertex + 1))
                if row + 1 < graph.rows:
                    expected_edges.add((vertex, vertex + graph.columns))
        if set(graph.edges) != expected_edges:
            raise ValueError("frontier DP requires the standard rectangular adjacency")

        self.problem = problem
        self.rows = graph.rows
        self.columns = graph.columns
        self.column_major = self.rows <= self.columns
        self.width = min(self.rows, self.columns)
        self.length = max(self.rows, self.columns)
        self.slots = graph.size
        self.types = problem.power_components
        self.code_by_id = {item.id: code for code, item in enumerate(self.types)}
        raw_fixed = {} if fixed_labels is None else dict(fixed_labels)
        if unknown_vertices := set(raw_fixed) - set(graph.vertices):
            raise ValueError(f"fixed vertices outside graph: {sorted(unknown_vertices)}")
        if unknown_labels := set(raw_fixed.values()) - self.code_by_id.keys():
            raise ValueError(f"unknown fixed power labels: {sorted(unknown_labels)}")
        self.fixed_codes = {
            vertex: self.code_by_id[label] for vertex, label in raw_fixed.items()
        }

        limits = dict(problem.component_limits)
        capped_codes = tuple(
            code
            for code, item in enumerate(self.types)
            if code != 0 and limits.get(item.id) is not None
        )
        self.capped_codes = capped_codes
        self.cap_index = {code: index for index, code in enumerate(capped_codes)}
        self.caps = tuple(int(limits[self.types[code].id]) for code in capped_codes)
        self.memo: dict[tuple, int] = {}
        self.choice: dict[tuple, int] = {}
        self._deadline: float | None = None
        self._cancel_check: Callable[[], bool] | None = None
        self._visits = 0

    def _position(self, step: int) -> tuple[int, int, int, int]:
        major, minor = divmod(step, self.width)
        if self.column_major:
            row, column = minor, major
        else:
            row, column = major, minor
        return row * self.columns + column, major, minor, step

    def _edge_power(self, first_code: int, second_code: int) -> int:
        first = self.types[first_code]
        second = self.types[second_code]
        pulse_units = 0
        if first.rods > 0 and second.accepts_pulse:
            pulse_units += first.rods
        if second.rods > 0 and first.accepts_pulse:
            pulse_units += second.rods
        return self.problem.eu_per_pulse * pulse_units

    def _increment(
        self,
        code: int,
        major: int,
        minor: int,
        frontier: tuple[int, ...],
    ) -> int:
        item = self.types[code]
        value = self.problem.eu_per_pulse * item.rods * item.internal_pulses
        if major > 0:
            value += self._edge_power(frontier[minor], code)
        if minor > 0:
            # The previous minor position has already been overwritten with
            # the current major slice's label.
            value += self._edge_power(frontier[minor - 1], code)
        return value

    def _allowed_codes(
        self,
        vertex: int,
        rods_used: int,
        cap_counts: tuple[int, ...],
    ):
        forced = self.fixed_codes.get(vertex)
        codes = (forced,) if forced is not None else range(len(self.types))
        for code in codes:
            item = self.types[code]
            if rods_used + item.rods > self.problem.rod_budget:
                continue
            cap_position = self.cap_index.get(code)
            if cap_position is not None and cap_counts[cap_position] >= self.caps[cap_position]:
                continue
            yield code

    def _best(
        self,
        step: int,
        frontier: tuple[int, ...],
        rods_used: int,
        cap_counts: tuple[int, ...],
    ) -> int:
        key = (step, frontier, rods_used, cap_counts)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        self._visits += 1
        if self._visits % 4096 == 0:
            if self._deadline is not None and perf_counter() >= self._deadline:
                raise TimeoutError
            if self._cancel_check is not None and self._cancel_check():
                raise InterruptedError
        remaining_slots = self.slots - step
        if rods_used > self.problem.rod_budget:
            return NEGATIVE_INFINITY
        if (
            self.problem.exact_rods
            and rods_used + remaining_slots * self.problem.max_rods_per_cell
            < self.problem.rod_budget
        ):
            return NEGATIVE_INFINITY
        if step == self.slots:
            valid_rods = (
                rods_used == self.problem.rod_budget
                if self.problem.exact_rods
                else 1 <= rods_used <= self.problem.rod_budget
            )
            result = 0 if valid_rods else NEGATIVE_INFINITY
            self.memo[key] = result
            return result

        vertex, major, minor, _ = self._position(step)
        best = NEGATIVE_INFINITY
        best_code: int | None = None
        for code in self._allowed_codes(vertex, rods_used, cap_counts):
            item = self.types[code]
            next_frontier = list(frontier)
            next_frontier[minor] = code
            next_caps = list(cap_counts)
            cap_position = self.cap_index.get(code)
            if cap_position is not None:
                next_caps[cap_position] += 1
            increment = self._increment(code, major, minor, frontier)
            suffix = self._best(
                step + 1,
                tuple(next_frontier),
                rods_used + item.rods,
                tuple(next_caps),
            )
            if suffix != NEGATIVE_INFINITY and increment + suffix > best:
                best = increment + suffix
                best_code = code
        self.memo[key] = best
        if best_code is not None:
            self.choice[key] = best_code
        return best

    def solve(
        self,
        *,
        time_limit_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> FrontierDPResult:
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        self._deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )
        self._cancel_check = cancel_check
        initial_frontier = (0,) * self.width
        initial_caps = (0,) * len(self.capped_codes)
        try:
            maximum = self._best(0, initial_frontier, 0, initial_caps)
        except TimeoutError:
            return FrontierDPResult(
                feasible=False,
                proven_optimal=False,
                maximum_power=None,
                skeleton=None,
                states=len(self.memo),
                frontier_width=self.width,
                scan_length=self.length,
                elapsed_seconds=perf_counter() - started,
                reason="time_limit",
            )
        except InterruptedError:
            return FrontierDPResult(
                feasible=False,
                proven_optimal=False,
                maximum_power=None,
                skeleton=None,
                states=len(self.memo),
                frontier_width=self.width,
                scan_length=self.length,
                elapsed_seconds=perf_counter() - started,
                reason="cancelled",
            )
        finally:
            self._deadline = None
            self._cancel_check = None

        if maximum == NEGATIVE_INFINITY:
            return FrontierDPResult(
                feasible=False,
                proven_optimal=True,
                maximum_power=None,
                skeleton=None,
                states=len(self.memo),
                frontier_width=self.width,
                scan_length=self.length,
                elapsed_seconds=perf_counter() - started,
                reason="infeasible",
            )

        skeleton = ["empty"] * self.slots
        step = rods_used = 0
        frontier = initial_frontier
        cap_counts = initial_caps
        while step < self.slots:
            key = (step, frontier, rods_used, cap_counts)
            code = self.choice[key]
            vertex, _major, minor, _ = self._position(step)
            skeleton[vertex] = self.types[code].id
            next_frontier = list(frontier)
            next_frontier[minor] = code
            next_caps = list(cap_counts)
            cap_position = self.cap_index.get(code)
            if cap_position is not None:
                next_caps[cap_position] += 1
            rods_used += self.types[code].rods
            frontier = tuple(next_frontier)
            cap_counts = tuple(next_caps)
            step += 1

        metrics = evaluate_power_skeleton(self.problem, skeleton)
        if metrics.power != maximum:
            raise AssertionError("frontier reconstruction disagrees with exact objective")
        return FrontierDPResult(
            feasible=True,
            proven_optimal=True,
            maximum_power=maximum,
            skeleton=tuple(skeleton),
            states=len(self.memo),
            frontier_width=self.width,
            scan_length=self.length,
            elapsed_seconds=perf_counter() - started,
            reason="optimal",
        )

    def complexity_signature(self) -> dict[str, int]:
        """Return the parameters controlling the pseudo-polynomial state bound."""

        label_count = len(self.types)
        inventory_factor = 1
        for cap in self.caps:
            inventory_factor *= cap + 1
        return {
            "vertices": self.slots,
            "frontier_width": self.width,
            "labels": label_count,
            "rod_states": self.problem.rod_budget + 1,
            "inventory_factor": inventory_factor,
            "coarse_state_bound": (
                (self.slots + 1)
                * label_count**self.width
                * (self.problem.rod_budget + 1)
                * inventory_factor
            ),
        }

    def ranked_skeletons(self, limit: int | None = None):
        """Yield exact feasible skeletons in non-increasing power order.

        ``solve`` must first complete the suffix table.  The priority queue is
        an A* enumeration whose key is ``power_so_far + exact_best_suffix``;
        consequently the first returned skeleton is globally best and every
        prefix of the output is the best possible power-ordered prefix.
        """

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        root_key = (0, (0,) * self.width, 0, (0,) * len(self.capped_codes))
        root_bound = self.memo.get(root_key)
        if root_bound is None:
            raise RuntimeError("solve() must complete before ranked enumeration")
        if root_bound == NEGATIVE_INFINITY or limit == 0:
            return

        # Entries: -exact bound, serial, step, frontier, rods, cap counts,
        # accumulated power, scan-order label choices.
        serial = 0
        heap = [(
            -root_bound,
            serial,
            0,
            (0,) * self.width,
            0,
            (0,) * len(self.capped_codes),
            0,
            (),
        )]
        yielded = 0
        while heap and (limit is None or yielded < limit):
            (
                _negative_bound,
                _serial,
                step,
                frontier,
                rods_used,
                cap_counts,
                power_so_far,
                choices,
            ) = heapq.heappop(heap)
            if step == self.slots:
                skeleton = ["empty"] * self.slots
                for choice_step, code in enumerate(choices):
                    vertex, _major, _minor, _ = self._position(choice_step)
                    skeleton[vertex] = self.types[code].id
                yielded += 1
                yield RankedPowerSkeleton(power_so_far, tuple(skeleton))
                continue

            vertex, major, minor, _ = self._position(step)
            for code in self._allowed_codes(vertex, rods_used, cap_counts):
                item = self.types[code]
                next_frontier_values = list(frontier)
                next_frontier_values[minor] = code
                next_frontier = tuple(next_frontier_values)
                next_caps_values = list(cap_counts)
                cap_position = self.cap_index.get(code)
                if cap_position is not None:
                    next_caps_values[cap_position] += 1
                next_caps = tuple(next_caps_values)
                increment = self._increment(code, major, minor, frontier)
                next_power = power_so_far + increment
                next_rods = rods_used + item.rods
                suffix = self._best(step + 1, next_frontier, next_rods, next_caps)
                if suffix == NEGATIVE_INFINITY:
                    continue
                serial += 1
                heapq.heappush(heap, (
                    -(next_power + suffix),
                    serial,
                    step + 1,
                    next_frontier,
                    next_rods,
                    next_caps,
                    next_power,
                    (*choices, code),
                ))
