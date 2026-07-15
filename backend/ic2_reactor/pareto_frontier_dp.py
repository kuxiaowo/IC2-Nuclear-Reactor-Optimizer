"""Pareto-minimal bounded-width master for static power/heat relaxations.

The table represents all power skeletons but stores only an antichain for each
continuation key.  A key contains exactly the separator labels/degrees and the
used rod count.  Point coordinates contain accumulated power, accumulated heat
and monotone residual resources (power-label inventory and future cooling-slot
capacity).  This is a master relaxation: thermal/dynamic cuts that distinguish
past prefixes must add their automaton state to the key and trigger a rebuild.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

from .frontier_automata import (
    FrontierConstraintAutomaton,
    FrontierTransitionContext,
    rectangular_frontier_order,
    rectangular_frontier_orders,
)
from .mathematical_model import ReactorProblem
from .state_quotient import (
    ContinuationParetoTable,
    ParetoPoint,
    append_packed_code,
    unpack_packed_codes,
)


@dataclass(frozen=True, slots=True)
class ParetoPowerHeatSkeleton:
    power: int
    generated_heat: int
    active_cells: int
    residual_inventory: tuple[int, ...]
    skeleton: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrontierLayerStatistics:
    placed_vertices: int
    continuation_keys: int
    pareto_points: int
    maximum_antichain_width: int
    automaton_state_tuples: int
    raw_transitions: int
    retained_insertions: int
    dominated_rejections: int
    removed_points: int
    upper_bound_rejections: int
    equivalent_successor_merges: int = 0


@dataclass(frozen=True, slots=True)
class ParetoFrontierDPResult:
    proven: bool
    frontier: tuple[ParetoPowerHeatSkeleton, ...]
    states_visited: int
    raw_transitions: int
    dominated_rejections: int
    upper_bound_rejections: int
    removed_points: int
    peak_layer_points: int
    peak_antichain_width: int
    layer_statistics: tuple[FrontierLayerStatistics, ...]
    frontier_width: int
    elapsed_seconds: float
    stop_reason: str
    equivalent_successor_merges: int = 0
    automaton_transition_cache_hits: int = 0
    automaton_transition_cache_misses: int = 0

    @property
    def peak_antichain_points(self) -> int:
        """Backward-compatible alias for the formerly misnamed layer peak."""

        return self.peak_layer_points


@dataclass(slots=True)
class _StateOrderedEntry:
    full_key: tuple
    point: ParetoPoint
    state_coordinates: tuple[tuple[int, ...], ...]


class _StateOrderedContinuationTable:
    """Pareto table extended by automaton-provided monotone state orders.

    Ordinary automata remain exact-key coordinates.  An automaton may expose
    ``state_dominance_key`` plus componentwise-larger
    ``state_dominance_coordinates``.  Entries are compared only inside one
    such family, so this cannot create cross-continuation pruning.
    """

    def __init__(self, automata: Sequence[FrontierConstraintAutomaton]) -> None:
        self.automata = tuple(automata)
        self._families: dict[tuple, list[_StateOrderedEntry]] = {}
        self.insertions = 0
        self.dominated_rejections = 0
        self.removed_points = 0
        self.equal_replacements = 0

    def _family_and_coordinates(
        self,
        full_key: tuple,
    ) -> tuple[tuple, tuple[tuple[int, ...], ...]]:
        frontier, rods, states = full_key
        family_states = []
        coordinates = []
        for automaton, state in zip(self.automata, states, strict=True):
            family = getattr(automaton, "state_dominance_key", None)
            ordered = getattr(automaton, "state_dominance_coordinates", None)
            if family is None or ordered is None:
                family_states.append(state)
                coordinates.append(())
            else:
                family_states.append(family(state))
                coordinates.append(tuple(ordered(state)))
        return (
            (frontier, rods, tuple(family_states)),
            tuple(coordinates),
        )

    @staticmethod
    def _weakly_dominates(
        first: _StateOrderedEntry,
        second: _StateOrderedEntry,
    ) -> bool:
        left = first.point
        right = second.point
        return (
            left.power >= right.power
            and left.generated_heat <= right.generated_heat
            and all(
                first_value >= second_value
                for first_value, second_value in zip(
                    left.residual_capacities,
                    right.residual_capacities,
                    strict=True,
                )
            )
            and all(
                first_value >= second_value
                for first_group, second_group in zip(
                    first.state_coordinates,
                    second.state_coordinates,
                    strict=True,
                )
                for first_value, second_value in zip(
                    first_group,
                    second_group,
                    strict=True,
                )
            )
        )

    @classmethod
    def _dominates(
        cls,
        first: _StateOrderedEntry,
        second: _StateOrderedEntry,
    ) -> bool:
        return cls._weakly_dominates(first, second) and not cls._weakly_dominates(
            second,
            first,
        )

    @classmethod
    def _mathematically_equal(
        cls,
        first: _StateOrderedEntry,
        second: _StateOrderedEntry,
    ) -> bool:
        return cls._weakly_dominates(first, second) and cls._weakly_dominates(
            second,
            first,
        )

    def insert(self, full_key: tuple, point: ParetoPoint) -> bool:
        self.insertions += 1
        family_key, coordinates = self._family_and_coordinates(full_key)
        candidate = _StateOrderedEntry(full_key, point, coordinates)
        frontier = self._families.setdefault(family_key, [])
        for index, existing in enumerate(frontier):
            if self._dominates(existing, candidate):
                self.dominated_rejections += 1
                return False
            if self._mathematically_equal(existing, candidate):
                if point.tie_key < existing.point.tie_key:
                    frontier[index] = candidate
                    self.equal_replacements += 1
                    return True
                self.dominated_rejections += 1
                return False
        retained = [
            existing
            for existing in frontier
            if not self._dominates(candidate, existing)
        ]
        self.removed_points += len(frontier) - len(retained)
        retained.append(candidate)
        self._families[family_key] = retained
        return True

    def frontier_items(self):
        for entries in self._families.values():
            by_full_key: dict[tuple, list[ParetoPoint]] = {}
            for entry in entries:
                by_full_key.setdefault(entry.full_key, []).append(entry.point)
            yield from by_full_key.items()

    @property
    def key_count(self) -> int:
        return len(self._families)

    @property
    def point_count(self) -> int:
        return sum(len(entries) for entries in self._families.values())

    @property
    def maximum_frontier_width(self) -> int:
        return max((len(entries) for entries in self._families.values()), default=0)


class RectangularParetoPowerHeatDP:
    """Compute exact nondominated static power/heat/resource representatives."""

    def __init__(
        self,
        problem: ReactorProblem,
        *,
        automata: Sequence[FrontierConstraintAutomaton] = (),
        placement_order: Sequence[int] | None = None,
        transition_cache_limit: int = 200_000,
        track_active_slot_resource: bool = True,
    ) -> None:
        graph = problem.graph
        if graph.rows is None or graph.columns is None:
            raise ValueError("Pareto frontier DP requires rectangular metadata")
        expected = graph.rectangular(graph.rows, graph.columns)
        if graph.edges != expected.edges:
            raise ValueError("Pareto frontier DP requires standard rectangular adjacency")
        if transition_cache_limit <= 0:
            raise ValueError("transition cache limit must be positive")
        self.problem = problem
        self.rows = graph.rows
        self.columns = graph.columns
        self.width = min(self.rows, self.columns)
        requested_order = (
            rectangular_frontier_order(graph)
            if placement_order is None
            else tuple(placement_order)
        )
        if requested_order not in rectangular_frontier_orders(graph):
            raise ValueError(
                "Pareto frontier DP placement order must be a minimum-width "
                "rectangular raster order"
            )
        self.placement_order = requested_order
        self.types = problem.power_components
        self.slots = graph.size
        self.automata = tuple(automata)
        self.transition_cache_limit = transition_cache_limit
        self.track_active_slot_resource = track_active_slot_resource
        self.automaton_initial_resources = tuple(
            tuple(automaton.initial_resources()) for automaton in self.automata
        )
        self.automaton_resource_widths = tuple(
            len(resources) for resources in self.automaton_initial_resources
        )
        self.active_by_code = tuple(item.accepts_pulse for item in self.types)
        self.degree_by_code = tuple(item.rods > 0 for item in self.types)
        # Placement-only automata retain every cut-relevant label distinction
        # in their own exact future-function state.  The geometric frontier can
        # then quotient labels solely by static power behaviour; in particular,
        # all non-power cooling components share the empty behaviour.
        quotient_graph_labels = all(
            getattr(automaton, "placement_only", False)
            for automaton in self.automata
        )
        representative_by_behaviour: dict[tuple[int, int, bool], int] = {}
        canonical_codes = []
        for code, item in enumerate(self.types):
            signature = (
                item.rods,
                item.internal_pulses if item.rods else 0,
                item.accepts_pulse,
            )
            representative = representative_by_behaviour.setdefault(
                signature,
                code,
            )
            canonical_codes.append(representative if quotient_graph_labels else code)
        self.power_frontier_code = tuple(canonical_codes)
        self._suffix_power_bounds = self._build_suffix_power_bounds()

        limits = dict(problem.component_limits)
        self.capped_codes = tuple(
            code
            for code, item in enumerate(self.types)
            if limits.get(item.id) is not None
        )
        self.cap_index = {
            code: index for index, code in enumerate(self.capped_codes)
        }
        self.caps = tuple(
            int(limits[self.types[code].id]) for code in self.capped_codes
        )
        self.base_residual_dimensions = (
            len(self.caps) + int(self.track_active_slot_resource)
        )

    def _build_suffix_power_bounds(self) -> tuple[tuple[int | None, ...], ...]:
        """Optimistic exact-rod power bounds for every unplaced scan suffix."""

        budget = self.problem.rod_budget
        unreachable: int | None = None
        rows = [[unreachable] * (budget + 1) for _ in range(self.slots + 1)]
        rows[self.slots][0] = 0
        for step in range(self.slots - 1, -1, -1):
            vertex, _major, _minor = self._position(step)
            maximum_degree = len(self.problem.graph.neighbours[vertex])
            # The suffix bound observes only rods and optimistic static power.
            # Full-layout domains can contain many thermally distinct cooling
            # labels with identical power behaviour; evaluating that duplicate
            # choice cannot strengthen the bound.
            choices = tuple(dict.fromkeys(
                (
                    item.rods,
                    self._final_contribution(code, maximum_degree)[0],
                )
                for code, item in enumerate(self.types)
            ))
            following = rows[step + 1]
            current = rows[step]
            for rods in range(budget + 1):
                best: int | None = None
                for item_rods, contribution in choices:
                    if item_rods > rods:
                        continue
                    suffix = following[rods - item_rods]
                    if suffix is None:
                        continue
                    value = suffix + contribution
                    if best is None or value > best:
                        best = value
                current[rods] = best
        return tuple(tuple(row) for row in rows)

    def _open_frontier_power_upper_bound(
        self,
        frontier: Sequence[tuple[int, int]],
        major: int,
        minor: int,
    ) -> int:
        """Bound the not-yet-finalized placed vertices in ``frontier``."""

        total = 0
        for frontier_minor, (code, _partial_degree) in enumerate(frontier):
            if frontier_minor <= minor:
                frontier_major = major
            elif major > 0:
                frontier_major = major - 1
            else:
                # Initial placeholders have not been placed yet.
                continue
            vertex, _unused_major, _unused_minor = self._position(
                frontier_major * self.width + frontier_minor
            )
            maximum_degree = len(self.problem.graph.neighbours[vertex])
            total += self._final_contribution(code, maximum_degree)[0]
        return total

    def _position(self, step: int) -> tuple[int, int, int]:
        major, minor = divmod(step, self.width)
        return self.placement_order[step], major, minor

    def _final_contribution(self, code: int, degree: int) -> tuple[int, int]:
        item = self.types[code]
        if item.rods <= 0:
            return 0, 0
        pulses = item.internal_pulses + degree
        return (
            self.problem.eu_per_pulse * item.rods * pulses,
            self.problem.heat_scale
            * item.rods
            * pulses
            * (pulses + 1),
        )

    def solve(
        self,
        *,
        incumbent_lower_bound: int | None = None,
        time_limit_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ParetoFrontierDPResult:
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )
        empty_code = next(
            code for code, item in enumerate(self.types) if item.id == "empty"
        )
        initial_frontier = ((empty_code, 0),) * self.width
        # Residual coordinates: capped power-label inventories, then the number
        # of slots not yet consumed by active power labels.
        initial_residual = (
            *self.caps,
            *((self.slots,) if self.track_active_slot_resource else ()),
            *(
                value
                for resources in self.automaton_initial_resources
                for value in resources
            ),
        )
        state_ordered = any(
            hasattr(automaton, "state_dominance_key")
            and hasattr(automaton, "state_dominance_coordinates")
            for automaton in self.automata
        )

        def new_layer_table():
            return (
                _StateOrderedContinuationTable(self.automata)
                if state_ordered
                else ContinuationParetoTable()
            )

        layer = new_layer_table()
        layer.insert(
            (
                initial_frontier,
                0,
                tuple(automaton.initial_state() for automaton in self.automata),
            ),
            ParetoPoint(0, 0, initial_residual, (0,)),
        )
        states_visited = raw_transitions = dominated = upper_pruned = removed = 0
        equivalent_merges = 0
        transition_cache: OrderedDict[tuple, object] = OrderedDict()
        transition_cache_hits = transition_cache_misses = 0
        cache_missing = object()
        peak_layer_points = layer.point_count
        peak_antichain_width = layer.maximum_frontier_width
        layer_statistics = [
            FrontierLayerStatistics(
                placed_vertices=0,
                continuation_keys=layer.key_count,
                pareto_points=layer.point_count,
                maximum_antichain_width=layer.maximum_frontier_width,
                automaton_state_tuples=1,
                raw_transitions=0,
                retained_insertions=1,
                dominated_rejections=0,
                removed_points=0,
                upper_bound_rejections=0,
            )
        ]

        for step in range(self.slots):
            if (
                (deadline is not None and perf_counter() >= deadline)
                or (cancel_check is not None and cancel_check())
            ):
                return ParetoFrontierDPResult(
                    proven=False,
                    frontier=(),
                    states_visited=states_visited,
                    raw_transitions=raw_transitions,
                    dominated_rejections=dominated,
                    upper_bound_rejections=upper_pruned,
                    removed_points=removed,
                    peak_layer_points=peak_layer_points,
                    peak_antichain_width=peak_antichain_width,
                    layer_statistics=tuple(layer_statistics),
                    frontier_width=self.width,
                    elapsed_seconds=perf_counter() - started,
                    stop_reason=(
                        "cancelled"
                        if cancel_check is not None and cancel_check()
                        else "time_limit"
                    ),
                    equivalent_successor_merges=equivalent_merges,
                    automaton_transition_cache_hits=transition_cache_hits,
                    automaton_transition_cache_misses=transition_cache_misses,
                )
            _vertex, major, minor = self._position(step)
            following = new_layer_table()
            transitions_before = raw_transitions
            upper_before = upper_pruned
            equivalent_before = equivalent_merges
            states_visited += sum(
                len(points) for _key, points in layer.frontier_items()
            )
            for (
                frontier,
                rods_used,
                automaton_states,
            ), points in layer.frontier_items():
                remaining_slots = self.slots - step - 1
                for point in points:
                    successors: dict[tuple, int] = {}
                    for code, item in enumerate(self.types):
                        raw_transitions += 1
                        next_rods = rods_used + item.rods
                        if next_rods > self.problem.rod_budget:
                            continue
                        if (
                            self.problem.exact_rods
                            and next_rods
                            + remaining_slots * self.problem.max_rods_per_cell
                            < self.problem.rod_budget
                        ):
                            continue
                        residual = list(point.residual_capacities)
                        cap_position = self.cap_index.get(code)
                        if cap_position is not None:
                            if residual[cap_position] <= 0:
                                continue
                            residual[cap_position] -= 1
                        if self.track_active_slot_resource and self.active_by_code[code]:
                            active_slot_resource = len(self.caps)
                            if residual[active_slot_resource] <= 0:
                                continue
                            residual[active_slot_resource] -= 1

                        values = list(frontier)
                        old_code, old_degree = values[minor]
                        current_active = self.active_by_code[code]
                        if minor > 0 and current_active:
                            left_code, left_degree = values[minor - 1]
                            if self.degree_by_code[left_code]:
                                values[minor - 1] = (left_code, left_degree + 1)

                        added_power = added_heat = 0
                        if major > 0:
                            added_power, added_heat = self._final_contribution(
                                old_code,
                                old_degree + int(current_active),
                            )
                        partial_degree = (
                            (
                                (int(self.active_by_code[old_code]) if major > 0 else 0)
                                + (
                                    int(self.active_by_code[values[minor - 1][0]])
                                    if minor > 0
                                    else 0
                                )
                            )
                            if self.degree_by_code[code]
                            else 0
                        )
                        values[minor] = (
                            self.power_frontier_code[code],
                            partial_degree,
                        )
                        next_frontier = tuple(values)
                        finalized_entry = (
                            (old_code, old_degree + int(current_active))
                            if major > 0
                            else None
                        )
                        next_automaton_states = []
                        placed_neighbours = []
                        if major > 0:
                            previous_vertex, _unused_major, _unused_minor = self._position(
                                step - self.width
                            )
                            placed_neighbours.append((previous_vertex, old_code))
                        if minor > 0:
                            preceding_vertex, _unused_major, _unused_minor = self._position(
                                step - 1
                            )
                            placed_neighbours.append(
                                (preceding_vertex, values[minor - 1][0])
                            )
                        finalized_vertex = (
                            self._position(step - self.width)[0]
                            if major > 0
                            else None
                        )
                        context = FrontierTransitionContext(
                            step=step,
                            vertex=_vertex,
                            placed_code=code,
                            major=major,
                            minor=minor,
                            placed_neighbours=tuple(placed_neighbours),
                            previous_frontier=frontier,
                            next_frontier=next_frontier,
                            finalized_vertex=finalized_vertex,
                            finalized_entry=finalized_entry,
                        )
                        automata_valid = True
                        resource_cursor = self.base_residual_dimensions
                        for automaton_index, (
                            automaton,
                            automaton_state,
                            resource_width,
                        ) in enumerate(zip(
                            self.automata,
                            automaton_states,
                            self.automaton_resource_widths,
                            strict=True,
                        )):
                            automaton_resources = tuple(
                                residual[
                                    resource_cursor:resource_cursor + resource_width
                                ]
                            )
                            cache_key = None
                            cached: object = cache_missing
                            if getattr(automaton, "placement_only", False):
                                cache_key = (
                                    automaton_index,
                                    step,
                                    automaton_state,
                                    automaton_resources,
                                    code,
                                )
                                cached = transition_cache.get(
                                    cache_key,
                                    cache_missing,
                                )
                            if cached is cache_missing:
                                transition_cache_misses += int(cache_key is not None)
                                transition = automaton.advance(
                                    automaton_state,
                                    automaton_resources,
                                    context,
                                )
                                if cache_key is not None:
                                    transition_cache[cache_key] = transition
                                    if len(transition_cache) > self.transition_cache_limit:
                                        transition_cache.popitem(last=False)
                            else:
                                transition_cache_hits += 1
                                transition_cache.move_to_end(cache_key)
                                transition = cached
                            if transition is None:
                                automata_valid = False
                                break
                            if len(transition.resources) != resource_width:
                                raise ValueError(
                                    "frontier automaton changed its resource dimension"
                                )
                            residual[
                                resource_cursor:resource_cursor + resource_width
                            ] = transition.resources
                            resource_cursor += resource_width
                            next_automaton_states.append(transition.state)
                        if not automata_valid:
                            continue
                        if incumbent_lower_bound is not None:
                            rods_left = self.problem.rod_budget - next_rods
                            suffix_row = self._suffix_power_bounds[step + 1]
                            if self.problem.exact_rods:
                                suffix_power = suffix_row[rods_left]
                            else:
                                reachable = [
                                    value
                                    for value in suffix_row[: rods_left + 1]
                                    if value is not None
                                ]
                                suffix_power = max(reachable) if reachable else None
                            if suffix_power is None:
                                upper_pruned += 1
                                continue
                            power_upper_bound = (
                                point.power
                                + added_power
                                + self._open_frontier_power_upper_bound(
                                    next_frontier,
                                    major,
                                    minor,
                                )
                                + suffix_power
                            )
                            if power_upper_bound <= incumbent_lower_bound:
                                upper_pruned += 1
                                continue
                        successor_signature = (
                            next_frontier,
                            next_rods,
                            tuple(next_automaton_states),
                            point.power + added_power,
                            point.generated_heat + added_heat,
                            tuple(residual),
                        )
                        previous_code = successors.get(successor_signature)
                        if previous_code is not None:
                            equivalent_merges += 1
                            if code < previous_code:
                                successors[successor_signature] = code
                            continue
                        successors[successor_signature] = code
                    for successor_signature, code in successors.items():
                        (
                            next_frontier,
                            next_rods,
                            next_automaton_states,
                            next_power,
                            next_heat,
                            next_residual,
                        ) = successor_signature
                        following.insert(
                            (
                                next_frontier,
                                next_rods,
                                next_automaton_states,
                            ),
                            ParetoPoint(
                                next_power,
                                next_heat,
                                next_residual,
                                append_packed_code(
                                    point.tie_key,
                                    code,
                                    len(self.types),
                                ),
                            ),
                        )
            dominated += following.dominated_rejections
            removed += following.removed_points
            layer = following
            layer_statistics.append(FrontierLayerStatistics(
                placed_vertices=step + 1,
                continuation_keys=layer.key_count,
                pareto_points=layer.point_count,
                maximum_antichain_width=layer.maximum_frontier_width,
                automaton_state_tuples=len({
                    key[2] for key, _points in layer.frontier_items()
                }),
                raw_transitions=raw_transitions - transitions_before,
                retained_insertions=(
                    following.insertions - following.dominated_rejections
                ),
                dominated_rejections=following.dominated_rejections,
                removed_points=following.removed_points,
                upper_bound_rejections=upper_pruned - upper_before,
                equivalent_successor_merges=(
                    equivalent_merges - equivalent_before
                ),
            ))
            peak_layer_points = max(peak_layer_points, layer.point_count)
            peak_antichain_width = max(
                peak_antichain_width,
                layer.maximum_frontier_width,
            )

        final_table = ContinuationParetoTable()
        final_major = self.slots // self.width - 1
        for (
            frontier,
            rods_used,
            automaton_states,
        ), points in layer.frontier_items():
            valid_rods = (
                rods_used == self.problem.rod_budget
                if self.problem.exact_rods
                else 1 <= rods_used <= self.problem.rod_budget
            )
            if not valid_rods:
                continue
            finalized_frontier = tuple(
                (
                    self._position(final_major * self.width + minor)[0],
                    code,
                    degree,
                )
                for minor, (code, degree) in enumerate(frontier)
            )
            final_power = sum(
                self._final_contribution(code, degree)[0]
                for code, degree in frontier
            )
            final_heat = sum(
                self._final_contribution(code, degree)[1]
                for code, degree in frontier
            )
            for point in points:
                resource_cursor = self.base_residual_dimensions
                accepted = True
                for automaton, automaton_state, resource_width in zip(
                    self.automata,
                    automaton_states,
                    self.automaton_resource_widths,
                    strict=True,
                ):
                    automaton_resources = tuple(
                        point.residual_capacities[
                            resource_cursor:resource_cursor + resource_width
                        ]
                    )
                    if not automaton.accepts(
                        automaton_state,
                        automaton_resources,
                        finalized_frontier,
                    ):
                        accepted = False
                        break
                    resource_cursor += resource_width
                if not accepted:
                    continue
                complete_power = point.power + final_power
                if (
                    incumbent_lower_bound is not None
                    and complete_power <= incumbent_lower_bound
                ):
                    upper_pruned += 1
                    continue
                final_table.insert(
                    "complete",
                    ParetoPoint(
                        complete_power,
                        point.generated_heat + final_heat,
                        point.residual_capacities[:self.base_residual_dimensions],
                        point.tie_key,
                    ),
                )
        dominated += final_table.dominated_rejections
        removed += final_table.removed_points
        result = []
        for point in final_table.frontier("complete"):
            labels = ["empty"] * self.slots
            codes = unpack_packed_codes(
                point.tie_key,
                self.slots,
                len(self.types),
            )
            for step, code in enumerate(codes):
                vertex, _major, _minor = self._position(step)
                labels[vertex] = self.types[code].id
            skeleton = tuple(labels)
            result.append(ParetoPowerHeatSkeleton(
                power=point.power,
                generated_heat=point.generated_heat,
                active_cells=(
                    self.slots - point.residual_capacities[len(self.caps)]
                    if self.track_active_slot_resource
                    else sum(self.active_by_code[code] for code in codes)
                ),
                residual_inventory=point.residual_capacities[:len(self.caps)],
                skeleton=skeleton,
            ))
        return ParetoFrontierDPResult(
            proven=True,
            frontier=tuple(result),
            states_visited=states_visited + layer.point_count,
            raw_transitions=raw_transitions,
            dominated_rejections=dominated,
            upper_bound_rejections=upper_pruned,
            removed_points=removed,
            peak_layer_points=peak_layer_points,
            peak_antichain_width=peak_antichain_width,
            layer_statistics=tuple(layer_statistics),
            frontier_width=self.width,
            elapsed_seconds=perf_counter() - started,
            stop_reason="complete",
            equivalent_successor_merges=equivalent_merges,
            automaton_transition_cache_hits=transition_cache_hits,
            automaton_transition_cache_misses=transition_cache_misses,
        )
