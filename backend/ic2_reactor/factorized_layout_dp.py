"""Full-label feasibility DP whose key stores behaviour, never raw label history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping, Sequence

from .frontier_automata import (
    FrontierConstraintAutomaton,
    FrontierTransitionContext,
)
from .state_quotient import (
    ContinuationParetoTable,
    ParetoPoint,
    append_packed_code,
    unpack_packed_codes,
)


@dataclass(frozen=True, slots=True)
class FactorizedLayoutLayerStatistics:
    placed_vertices: int
    continuation_keys: int
    pareto_points: int
    maximum_antichain_width: int
    automaton_state_tuples: int
    raw_transitions: int
    equivalent_successor_merges: int
    dominated_rejections: int
    removed_points: int


@dataclass(frozen=True, slots=True)
class FactorizedLayoutDPResult:
    proven: bool
    feasible: bool
    layout: tuple[str, ...] | None
    raw_transitions: int
    equivalent_successor_merges: int
    dominated_rejections: int
    removed_points: int
    peak_layer_points: int
    peak_antichain_width: int
    layer_statistics: tuple[FactorizedLayoutLayerStatistics, ...]
    elapsed_seconds: float
    stop_reason: str


class FactorizedLayoutFeasibilityDP:
    """Intersect local-factor cuts and inventories over a full label domain.

    Power labels are normally fixed by an outer skeleton master.  Free slots
    choose from ``free_labels``.  Prefix labels themselves are not part of the
    continuation key: local factor automata retain exactly the residual future
    functions that still depend on them.  Component upper inventories and cut
    scores are monotone Pareto resources.
    """

    def __init__(
        self,
        placement_order: Sequence[int],
        label_domain: Sequence[str],
        *,
        free_labels: Sequence[str],
        fixed_labels: Mapping[int, str] | None = None,
        component_limits: Mapping[str, int | None] | None = None,
        automata: Sequence[FrontierConstraintAutomaton] = (),
    ) -> None:
        self.placement_order = tuple(placement_order)
        size = len(self.placement_order)
        if set(self.placement_order) != set(range(size)):
            raise ValueError("placement order must be a permutation of 0..n-1")
        self.labels = tuple(label_domain)
        if not self.labels or len(self.labels) != len(set(self.labels)):
            raise ValueError("label domain must be non-empty and unique")
        self.code_by_label = {
            label: code for code, label in enumerate(self.labels)
        }
        if unknown := set(free_labels) - self.code_by_label.keys():
            raise ValueError(f"free labels are outside the domain: {sorted(unknown)}")
        self.free_codes = tuple(self.code_by_label[label] for label in free_labels)
        if not self.free_codes:
            raise ValueError("at least one free label is required")
        raw_fixed = {} if fixed_labels is None else dict(fixed_labels)
        if unknown_vertices := set(raw_fixed) - set(self.placement_order):
            raise ValueError(f"fixed vertices are outside the layout: {sorted(unknown_vertices)}")
        if unknown_labels := set(raw_fixed.values()) - self.code_by_label.keys():
            raise ValueError(f"fixed labels are outside the domain: {sorted(unknown_labels)}")
        self.fixed_codes = {
            vertex: self.code_by_label[label]
            for vertex, label in raw_fixed.items()
        }

        limits = {} if component_limits is None else dict(component_limits)
        if unknown := set(limits) - self.code_by_label.keys():
            raise ValueError(f"component limits use unknown labels: {sorted(unknown)}")
        if any(limit is not None and limit < 0 for limit in limits.values()):
            raise ValueError("component limits must be non-negative or None")
        fixed_counts = Counter(self.fixed_codes.values())
        free_vertex_count = size - len(self.fixed_codes)
        self.inventory_impossible = any(
            fixed_counts[code] > int(limits[label])
            for code, label in enumerate(self.labels)
            if limits.get(label) is not None
        )
        remaining_by_code = {
            code: int(limits[label]) - fixed_counts[code]
            for code, label in enumerate(self.labels)
            if limits.get(label) is not None
        }
        self.capped_codes = tuple(
            code
            for code, remaining in remaining_by_code.items()
            if (
                remaining >= 0
                and code in self.free_codes
                and remaining < free_vertex_count
            )
        )
        self.cap_index = {
            code: index for index, code in enumerate(self.capped_codes)
        }
        self.caps = tuple(
            remaining_by_code[code] for code in self.capped_codes
        )

        self.automata = tuple(automata)
        unsupported = tuple(
            automaton
            for automaton in self.automata
            if not getattr(automaton, "placement_only", False)
        )
        if unsupported:
            raise TypeError(
                "full-label behaviour DP accepts only placement-factor automata; "
                "geometric context automata must first be compiled to factors"
            )
        self.automaton_initial_resources = tuple(
            tuple(automaton.initial_resources()) for automaton in self.automata
        )
        self.automaton_resource_widths = tuple(
            len(resources) for resources in self.automaton_initial_resources
        )
        self.base_resource_dimensions = len(self.caps)

    def solve(
        self,
        *,
        time_limit_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> FactorizedLayoutDPResult:
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )
        initial_resources = (
            *self.caps,
            *(
                value
                for resources in self.automaton_initial_resources
                for value in resources
            ),
        )
        layer = ContinuationParetoTable()
        layer.insert(
            tuple(automaton.initial_state() for automaton in self.automata),
            ParetoPoint(0, 0, initial_resources, (0,)),
        )
        raw_transitions = equivalent_merges = dominated = removed = 0
        peak_points = layer.point_count
        peak_width = layer.maximum_frontier_width
        statistics = [
            FactorizedLayoutLayerStatistics(
                placed_vertices=0,
                continuation_keys=layer.key_count,
                pareto_points=layer.point_count,
                maximum_antichain_width=layer.maximum_frontier_width,
                automaton_state_tuples=layer.key_count,
                raw_transitions=0,
                equivalent_successor_merges=0,
                dominated_rejections=0,
                removed_points=0,
            )
        ]

        if self.inventory_impossible:
            return FactorizedLayoutDPResult(
                proven=True,
                feasible=False,
                layout=None,
                raw_transitions=0,
                equivalent_successor_merges=0,
                dominated_rejections=0,
                removed_points=0,
                peak_layer_points=layer.point_count,
                peak_antichain_width=layer.maximum_frontier_width,
                layer_statistics=tuple(statistics),
                elapsed_seconds=perf_counter() - started,
                stop_reason="fixed_inventory_infeasible",
            )

        for step, vertex in enumerate(self.placement_order):
            if (
                (deadline is not None and perf_counter() >= deadline)
                or (cancel_check is not None and cancel_check())
            ):
                return FactorizedLayoutDPResult(
                    proven=False,
                    feasible=False,
                    layout=None,
                    raw_transitions=raw_transitions,
                    equivalent_successor_merges=equivalent_merges,
                    dominated_rejections=dominated,
                    removed_points=removed,
                    peak_layer_points=peak_points,
                    peak_antichain_width=peak_width,
                    layer_statistics=tuple(statistics),
                    elapsed_seconds=perf_counter() - started,
                    stop_reason=(
                        "cancelled"
                        if cancel_check is not None and cancel_check()
                        else "time_limit"
                    ),
                )
            following = ContinuationParetoTable()
            transitions_before = raw_transitions
            equivalent_before = equivalent_merges
            allowed_codes = (
                (self.fixed_codes[vertex],)
                if vertex in self.fixed_codes
                else self.free_codes
            )
            for automaton_states, points in layer.frontier_items():
                for point in points:
                    successors: dict[
                        tuple[tuple[object, ...], tuple[int, ...]],
                        int,
                    ] = {}
                    for code in allowed_codes:
                        raw_transitions += 1
                        resources = list(point.residual_capacities)
                        cap_position = self.cap_index.get(code)
                        if cap_position is not None and vertex not in self.fixed_codes:
                            if resources[cap_position] <= 0:
                                continue
                            resources[cap_position] -= 1
                        context = FrontierTransitionContext(
                            step=step,
                            vertex=vertex,
                            placed_code=code,
                            major=step,
                            minor=0,
                            placed_neighbours=(),
                            previous_frontier=(),
                            next_frontier=(),
                            finalized_vertex=None,
                            finalized_entry=None,
                        )
                        next_states = []
                        cursor = self.base_resource_dimensions
                        valid = True
                        for automaton, state, width in zip(
                            self.automata,
                            automaton_states,
                            self.automaton_resource_widths,
                            strict=True,
                        ):
                            transition = automaton.advance(
                                state,
                                tuple(resources[cursor:cursor + width]),
                                context,
                            )
                            if transition is None:
                                valid = False
                                break
                            if len(transition.resources) != width:
                                raise ValueError(
                                    "factor automaton changed its resource dimension"
                                )
                            resources[cursor:cursor + width] = transition.resources
                            cursor += width
                            next_states.append(transition.state)
                        if not valid:
                            continue
                        signature = (tuple(next_states), tuple(resources))
                        previous_code = successors.get(signature)
                        if previous_code is not None:
                            equivalent_merges += 1
                            if code < previous_code:
                                successors[signature] = code
                            continue
                        successors[signature] = code
                    for (next_states, resources), code in successors.items():
                        following.insert(
                            next_states,
                            ParetoPoint(
                                0,
                                0,
                                resources,
                                append_packed_code(
                                    point.tie_key,
                                    code,
                                    len(self.labels),
                                ),
                            ),
                        )
            dominated += following.dominated_rejections
            removed += following.removed_points
            layer = following
            statistics.append(FactorizedLayoutLayerStatistics(
                placed_vertices=step + 1,
                continuation_keys=layer.key_count,
                pareto_points=layer.point_count,
                maximum_antichain_width=layer.maximum_frontier_width,
                automaton_state_tuples=layer.key_count,
                raw_transitions=raw_transitions - transitions_before,
                equivalent_successor_merges=(
                    equivalent_merges - equivalent_before
                ),
                dominated_rejections=following.dominated_rejections,
                removed_points=following.removed_points,
            ))
            peak_points = max(peak_points, layer.point_count)
            peak_width = max(peak_width, layer.maximum_frontier_width)
            if layer.point_count == 0:
                return FactorizedLayoutDPResult(
                    proven=True,
                    feasible=False,
                    layout=None,
                    raw_transitions=raw_transitions,
                    equivalent_successor_merges=equivalent_merges,
                    dominated_rejections=dominated,
                    removed_points=removed,
                    peak_layer_points=peak_points,
                    peak_antichain_width=peak_width,
                    layer_statistics=tuple(statistics),
                    elapsed_seconds=perf_counter() - started,
                    stop_reason="infeasible",
                )

        accepted: list[ParetoPoint] = []
        for automaton_states, points in layer.frontier_items():
            for point in points:
                cursor = self.base_resource_dimensions
                valid = True
                for automaton, state, width in zip(
                    self.automata,
                    automaton_states,
                    self.automaton_resource_widths,
                    strict=True,
                ):
                    if not automaton.accepts(
                        state,
                        tuple(point.residual_capacities[cursor:cursor + width]),
                        (),
                    ):
                        valid = False
                        break
                    cursor += width
                if valid:
                    accepted.append(point)
        if not accepted:
            return FactorizedLayoutDPResult(
                proven=True,
                feasible=False,
                layout=None,
                raw_transitions=raw_transitions,
                equivalent_successor_merges=equivalent_merges,
                dominated_rejections=dominated,
                removed_points=removed,
                peak_layer_points=peak_points,
                peak_antichain_width=peak_width,
                layer_statistics=tuple(statistics),
                elapsed_seconds=perf_counter() - started,
                stop_reason="infeasible",
            )
        representative = min(accepted, key=lambda point: point.tie_key)
        layout = [self.labels[0]] * len(self.placement_order)
        codes = unpack_packed_codes(
            representative.tie_key,
            len(self.placement_order),
            len(self.labels),
        )
        for step, code in enumerate(codes):
            layout[self.placement_order[step]] = self.labels[code]
        return FactorizedLayoutDPResult(
            proven=True,
            feasible=True,
            layout=tuple(layout),
            raw_transitions=raw_transitions,
            equivalent_successor_merges=equivalent_merges,
            dominated_rejections=dominated,
            removed_points=removed,
            peak_layer_points=peak_points,
            peak_antichain_width=peak_width,
            layer_statistics=tuple(statistics),
            elapsed_seconds=perf_counter() - started,
            stop_reason="feasible",
        )
