"""All-at-once terminal-cut representation of the average heat-flow test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from .frontier_automata import (
    FrontierAutomatonTransition,
    FrontierTransitionContext,
)
from .mathematical_model import ReactorProblem, evaluate_power_skeleton
from .terminal_cut_quotient import (
    TerminalCutFactorScope,
    TerminalCutScheduleProfile,
    TerminalCutSignature,
    terminal_cut_schedule_profile,
)
from .thermal_relaxation import HeatFlowComponent


HULL_TERMINAL: tuple[str] = ("hull",)


def storage_terminal(vertex: int) -> tuple[str, int]:
    return ("storage", int(vertex))


def average_flow_combined_terminal_factor_scopes(
    problem: ReactorProblem,
) -> tuple[TerminalCutFactorScope, ...]:
    """One exact radius-one cut factor per possible component centre."""

    return tuple(
        TerminalCutFactorScope(
            (vertex, *problem.graph.neighbours[vertex]),
            (
                HULL_TERMINAL,
                storage_terminal(vertex),
                *(
                    storage_terminal(neighbour)
                    for neighbour in problem.graph.neighbours[vertex]
                ),
            ),
        )
        for vertex in problem.graph.vertices
    )


@dataclass(frozen=True, slots=True)
class AverageFlowTerminalCutState:
    """Exact label boundary and saturated cut function after one scan step."""

    live_assignments: tuple[tuple[int, int], ...]
    signature: TerminalCutSignature


@dataclass(frozen=True, slots=True)
class AverageFlowTerminalAutomatonProfile:
    schedule: TerminalCutScheduleProfile
    peak_live_label_variables: int
    peak_live_label_behaviour_product: int
    live_cut_terminals_after_step: tuple[int, ...]
    cut_vector_entries_after_step: tuple[int, ...]
    discrete_state_bound: int


class AverageFlowTerminalCutAutomaton:
    """Enforce the complete average max-flow condition in a layout DP.

    Every generator node is minimized analytically.  The remaining directed
    cut capacity is a sum of one radius-one factor per grid cell.  The state
    stores the exact saturated terminal cut function and only those local
    label behaviours still read by an unfinished factor.  Its sole Pareto
    resource is ``maximum_heat - heat_already_generated``.
    """

    placement_only = True

    def __init__(
        self,
        problem: ReactorProblem,
        catalogue: Mapping[str, HeatFlowComponent],
        labels: Sequence[str],
        *,
        placement_order: Sequence[int],
    ) -> None:
        self.problem = problem
        self.catalogue = dict(catalogue)
        self.labels = tuple(labels)
        if not self.labels or len(self.labels) != len(set(self.labels)):
            raise ValueError("average-flow label domain must be non-empty and unique")
        if missing := set(self.labels) - self.catalogue.keys():
            raise ValueError(f"heat catalogue is missing labels: {sorted(missing)}")
        self.placement_order = tuple(placement_order)
        if set(self.placement_order) != set(problem.graph.vertices):
            raise ValueError("average-flow placement order must permute graph vertices")
        self.rank = {
            vertex: step for step, vertex in enumerate(self.placement_order)
        }
        self.code_by_label = {
            label: code for code, label in enumerate(self.labels)
        }
        self.power_by_label = {
            item.id: item for item in problem.power_components
        }
        maximum_internal = max(
            item.internal_pulses
            for item in problem.power_components
            if item.rods > 0
        )
        maximum_pulses = problem.graph.maximum_degree + maximum_internal
        self.maximum_heat = (
            problem.heat_scale
            * problem.rod_budget
            * maximum_pulses
            * (maximum_pulses + 1)
        )
        if self.maximum_heat <= 0:
            raise ValueError("average-flow heat upper bound must be positive")

        self.stars = tuple(
            (vertex, *problem.graph.neighbours[vertex])
            for vertex in problem.graph.vertices
        )
        self.cut_scopes = tuple(
            (
                HULL_TERMINAL,
                storage_terminal(vertex),
                *(
                    storage_terminal(neighbour)
                    for neighbour in problem.graph.neighbours[vertex]
                ),
            )
            for vertex in problem.graph.vertices
        )
        self.factor_event = tuple(
            max(self.rank[vertex] for vertex in star)
            for star in self.stars
        )
        buckets: list[list[int]] = [[] for _ in self.placement_order]
        for center, event in enumerate(self.factor_event):
            buckets[event].append(center)
        self.centers_by_step = tuple(tuple(bucket) for bucket in buckets)

        terminal_order: tuple[Hashable, ...] = (
            HULL_TERMINAL,
            *(storage_terminal(vertex) for vertex in problem.graph.vertices),
        )
        first_event: dict[Hashable, int] = {}
        last_event: dict[Hashable, int] = {}
        for center, event in enumerate(self.factor_event):
            for terminal in self.cut_scopes[center]:
                first_event[terminal] = min(first_event.get(terminal, event), event)
                last_event[terminal] = max(last_event.get(terminal, event), event)
        self.introductions_by_step = tuple(
            tuple(
                terminal
                for terminal in terminal_order
                if first_event.get(terminal) == step
            )
            for step in range(len(self.placement_order))
        )
        self.forgets_by_step = tuple(
            tuple(
                terminal
                for terminal in terminal_order
                if last_event.get(terminal) == step
            )
            for step in range(len(self.placement_order))
        )
        self.live_cut_terminals_after_step = tuple(
            sum(
                first_event[terminal] <= step < last_event[terminal]
                for terminal in terminal_order
            )
            for step in range(len(self.placement_order))
        )

        incident_centers = tuple(
            tuple(
                center
                for center, star in enumerate(self.stars)
                if vertex in star
            )
            for vertex in problem.graph.vertices
        )
        self.label_last_use = tuple(
            max(self.factor_event[center] for center in incident_centers[vertex])
            for vertex in problem.graph.vertices
        )
        center_behaviour = []
        neighbour_behaviour = []
        for label in self.labels:
            power = self.power_by_label.get(label)
            spec = self.catalogue[label]
            center_behaviour.append((
                0 if power is None else power.rods,
                0 if power is None else power.internal_pulses,
                spec.accepts_heat,
                spec.self_vent,
                spec.hull_draw,
                spec.exchange_side,
                spec.exchange_hull,
            ))
            neighbour_behaviour.append((
                bool(power and power.accepts_pulse),
                spec.accepts_heat,
                spec.side_vent,
            ))
        self.center_behaviour = tuple(center_behaviour)
        self.neighbour_behaviour = tuple(neighbour_behaviour)
        canonical_after: list[dict[int, tuple[int, ...]]] = []
        label_products = []
        for step in range(len(self.placement_order)):
            mappings: dict[int, tuple[int, ...]] = {}
            product_bound = 1
            for vertex in problem.graph.vertices:
                if not (
                    self.rank[vertex] <= step < self.label_last_use[vertex]
                ):
                    continue
                future_roles = tuple(
                    (center, center == vertex)
                    for center in incident_centers[vertex]
                    if self.factor_event[center] > step
                )
                representative_by_signature: dict[tuple, int] = {}
                canonical = []
                for code in range(len(self.labels)):
                    signature = tuple(
                        (
                            center,
                            (
                                self.center_behaviour[code]
                                if is_center
                                else self.neighbour_behaviour[code]
                            ),
                        )
                        for center, is_center in future_roles
                    )
                    canonical.append(representative_by_signature.setdefault(
                        signature,
                        code,
                    ))
                mappings[vertex] = tuple(canonical)
                product_bound *= len(set(canonical))
            canonical_after.append(mappings)
            label_products.append(product_bound)
        self.canonical_code_after = tuple(canonical_after)

        schedule = terminal_cut_schedule_profile(
            self.placement_order,
            average_flow_combined_terminal_factor_scopes(problem),
        )
        cut_entries_after = tuple(
            1 << count for count in self.live_cut_terminals_after_step
        )
        per_step_state_bounds = tuple(
            label_product * pow(self.maximum_heat + 1, entries)
            for label_product, entries in zip(
                label_products,
                cut_entries_after,
                strict=True,
            )
        )
        self.profile = AverageFlowTerminalAutomatonProfile(
            schedule=schedule,
            peak_live_label_variables=max(
                (len(mapping) for mapping in self.canonical_code_after),
                default=0,
            ),
            peak_live_label_behaviour_product=max(label_products, default=1),
            live_cut_terminals_after_step=self.live_cut_terminals_after_step,
            cut_vector_entries_after_step=cut_entries_after,
            discrete_state_bound=max(per_step_state_bounds, default=1),
        )

    def initial_state(self) -> AverageFlowTerminalCutState:
        return AverageFlowTerminalCutState(
            (),
            TerminalCutSignature.zero(saturation=self.maximum_heat),
        )

    def initial_resources(self) -> tuple[int, ...]:
        return (self.maximum_heat,)

    def pareto_resource_chain_bounds(self) -> tuple[tuple[int, int], ...]:
        return ((0, self.maximum_heat),)

    @staticmethod
    def state_dominance_key(
        state: AverageFlowTerminalCutState,
    ) -> tuple[tuple[tuple[int, int], ...], tuple[Hashable, ...]]:
        """Return exactly the non-ordered part of future equivalence."""

        return state.live_assignments, state.signature.terminals

    @staticmethod
    def state_dominance_coordinates(
        state: AverageFlowTerminalCutState,
    ) -> tuple[int, ...]:
        """Larger constrained-cut capacity is never worse for any suffix."""

        return state.signature.values

    def _center_factor(
        self,
        center: int,
        assignments: Mapping[int, int],
    ) -> tuple[int, tuple[int, ...]]:
        neighbours = tuple(self.problem.graph.neighbours[center])
        center_code = assignments[center]
        centre_label = self.labels[center_code]
        spec = self.catalogue[centre_label]
        neighbour_codes = tuple(assignments[vertex] for vertex in neighbours)
        neighbour_labels = tuple(self.labels[code] for code in neighbour_codes)
        power = self.power_by_label.get(centre_label)
        heat = 0
        if power is not None and power.rods > 0:
            degree = sum(
                bool(
                    (neighbour_power := self.power_by_label.get(label))
                    and neighbour_power.accepts_pulse
                )
                for label in neighbour_labels
            )
            pulses = power.internal_pulses + degree
            heat = (
                self.problem.heat_scale
                * power.rods
                * pulses
                * (pulses + 1)
            )

        costs = []
        for mask in range(1 << (len(neighbours) + 2)):
            hull_source = bool(mask & 1)
            centre_source = bool(mask & 2)
            neighbour_sources = tuple(
                bool(mask >> (position + 2) & 1)
                for position in range(len(neighbours))
            )
            cost = 0
            if heat and (
                not hull_source
                or any(
                    self.catalogue[label].accepts_heat and not source_side
                    for label, source_side in zip(
                        neighbour_labels,
                        neighbour_sources,
                        strict=True,
                    )
                )
            ):
                cost += heat
            if spec.accepts_heat and centre_source:
                cost += spec.self_vent + sum(
                    self.catalogue[label].side_vent
                    for label in neighbour_labels
                )
            if (
                spec.accepts_heat
                and spec.hull_draw
                and hull_source
                and not centre_source
            ):
                cost += spec.hull_draw
            if spec.exchange_side:
                cost += spec.exchange_side * sum(
                    self.catalogue[label].accepts_heat
                    and centre_source != source_side
                    for label, source_side in zip(
                        neighbour_labels,
                        neighbour_sources,
                        strict=True,
                    )
                )
            if spec.exchange_hull and centre_source != hull_source:
                cost += spec.exchange_hull
            costs.append(cost)
        return heat, tuple(costs)

    def advance(
        self,
        state: AverageFlowTerminalCutState,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if len(resources) != 1:
            raise ValueError("average-flow terminal automaton requires one resource")
        if context.step >= len(self.placement_order):
            raise ValueError("average-flow frontier step is outside the order")
        if context.vertex != self.placement_order[context.step]:
            raise ValueError("average-flow frontier scan order differs from automaton")
        assignments = dict(state.live_assignments)
        assignments[context.vertex] = context.placed_code
        signature = state.signature
        for terminal in self.introductions_by_step[context.step]:
            signature = signature.add_terminal(terminal)
        remaining_heat = resources[0]
        for center in self.centers_by_step[context.step]:
            try:
                heat, costs = self._center_factor(center, assignments)
            except KeyError as error:  # pragma: no cover - constructor invariant
                raise AssertionError(
                    "average-flow automaton forgot a live label"
                ) from error
            remaining_heat -= heat
            if remaining_heat < 0:  # pragma: no cover - analytic upper invariant
                raise AssertionError("generated heat exceeded its analytic upper bound")
            signature = signature.add_factor(self.cut_scopes[center], costs)
        for terminal in self.forgets_by_step[context.step]:
            signature = signature.forget(terminal)

        live = []
        for vertex, code in assignments.items():
            if self.label_last_use[vertex] <= context.step:
                continue
            canonical = self.canonical_code_after[context.step].get(vertex)
            if canonical is None:  # pragma: no cover - last-use invariant
                raise AssertionError("live average-flow label has no future quotient")
            live.append((vertex, canonical[code]))
        return FrontierAutomatonTransition(
            AverageFlowTerminalCutState(tuple(sorted(live)), signature),
            (remaining_heat,),
        )

    def accepts(
        self,
        state: AverageFlowTerminalCutState,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool:
        _ = final_frontier
        if len(resources) != 1:
            raise ValueError("average-flow terminal automaton requires one resource")
        if state.live_assignments or state.signature.terminals:
            raise AssertionError("average-flow terminal state did not fully eliminate")
        generated_heat = self.maximum_heat - resources[0]
        return state.signature.minimum_cut >= generated_heat


def average_flow_terminal_factor_scopes(
    problem: ReactorProblem,
    catalogue: Mapping[str, HeatFlowComponent],
    labels: Sequence[str],
) -> tuple[TerminalCutFactorScope, ...]:
    """Compile only the symbolic scopes; no layout assignments are visited."""

    domain = tuple(labels)
    if not domain or len(domain) != len(set(domain)):
        raise ValueError("average-flow label domain must be non-empty and unique")
    if missing := set(domain) - catalogue.keys():
        raise ValueError(f"heat catalogue is missing labels: {sorted(missing)}")
    specs = tuple(catalogue[label] for label in domain)
    power_by_id = {item.id: item for item in problem.power_components}
    fuel_possible = any(
        power_by_id.get(label) is not None and power_by_id[label].rods > 0
        for label in domain
    )
    acceptance_possible = any(spec.accepts_heat for spec in specs)
    external_possible = any(spec.accepts_heat and spec.self_vent for spec in specs)
    side_vent_possible = any(spec.side_vent for spec in specs)
    hull_draw_possible = any(
        spec.accepts_heat and spec.hull_draw for spec in specs
    )
    side_exchange_possible = any(spec.exchange_side for spec in specs)
    hull_exchange_possible = any(spec.exchange_hull for spec in specs)

    factors: list[TerminalCutFactorScope] = []
    for vertex in problem.graph.vertices:
        neighbours = tuple(problem.graph.neighbours[vertex])
        star = (vertex, *neighbours)
        if fuel_possible:
            generator_cut_scope: tuple[Hashable, ...] = (
                HULL_TERMINAL,
                *(
                    storage_terminal(neighbour)
                    for neighbour in neighbours
                    if acceptance_possible
                ),
            )
            factors.append(TerminalCutFactorScope(star, generator_cut_scope))
        if acceptance_possible and (external_possible or side_vent_possible):
            factors.append(TerminalCutFactorScope(
                star if side_vent_possible else (vertex,),
                (storage_terminal(vertex),),
            ))
        if hull_draw_possible:
            factors.append(TerminalCutFactorScope(
                (vertex,),
                (HULL_TERMINAL, storage_terminal(vertex)),
            ))
        if side_exchange_possible and acceptance_possible:
            for neighbour in neighbours:
                factors.append(TerminalCutFactorScope(
                    (vertex, neighbour),
                    (storage_terminal(vertex), storage_terminal(neighbour)),
                ))
        if hull_exchange_possible:
            factors.append(TerminalCutFactorScope(
                (vertex,),
                (storage_terminal(vertex), HULL_TERMINAL),
            ))
    return tuple(factors)


def average_flow_terminal_schedule_profile(
    problem: ReactorProblem,
    catalogue: Mapping[str, HeatFlowComponent],
    labels: Sequence[str],
    placement_order: Sequence[int],
) -> TerminalCutScheduleProfile:
    return terminal_cut_schedule_profile(
        placement_order,
        average_flow_terminal_factor_scopes(problem, catalogue, labels),
    )


def eliminated_average_flow_minimum_cut(
    problem: ReactorProblem,
    layout: Sequence[str],
    catalogue: Mapping[str, HeatFlowComponent],
) -> tuple[int, int]:
    """Reference proof of generator elimination for a small fixed layout.

    The routine intentionally materializes ``2 ** (|V| + 1)`` cut values and
    is only a validation oracle.  The production algorithm uses the same
    factors incrementally and never exceeds its live terminal separator.
    """

    if len(layout) != problem.graph.size:
        raise ValueError("layout length does not match graph")
    if missing := set(layout) - catalogue.keys():
        raise ValueError(f"heat catalogue is missing labels: {sorted(missing)}")
    power_by_id = {item.id: item for item in problem.power_components}
    skeleton = tuple(label if label in power_by_id else "empty" for label in layout)
    metrics = evaluate_power_skeleton(problem, skeleton)
    terminals: tuple[Hashable, ...] = (
        HULL_TERMINAL,
        *(storage_terminal(vertex) for vertex in problem.graph.vertices),
    )
    signature = TerminalCutSignature.zero(
        terminals,
        saturation=metrics.generated_heat or None,
    )

    for vertex, label in enumerate(layout):
        spec = catalogue[label]
        neighbours = tuple(problem.graph.neighbours[vertex])
        power_spec = power_by_id.get(label)
        if power_spec is not None and power_spec.rods > 0:
            degree = metrics.degrees[vertex]
            pulses = power_spec.internal_pulses + degree
            heat = (
                problem.heat_scale
                * power_spec.rods
                * pulses
                * (pulses + 1)
            )
            targets: tuple[Hashable, ...] = (
                HULL_TERMINAL,
                *(
                    storage_terminal(neighbour)
                    for neighbour in neighbours
                    if catalogue[layout[neighbour]].accepts_heat
                ),
            )
            signature = signature.add_factor(
                targets,
                tuple(
                    0 if mask == (1 << len(targets)) - 1 else heat
                    for mask in range(1 << len(targets))
                ),
            )

        if spec.accepts_heat:
            external = spec.self_vent + sum(
                catalogue[layout[neighbour]].side_vent
                for neighbour in neighbours
            )
            signature = signature.add_to_fixed_sink(
                storage_terminal(vertex),
                external,
            )
            if spec.hull_draw:
                signature = signature.add_directed_edge(
                    HULL_TERMINAL,
                    storage_terminal(vertex),
                    spec.hull_draw,
                )

        if spec.exchange_side:
            for neighbour in neighbours:
                if not catalogue[layout[neighbour]].accepts_heat:
                    continue
                signature = signature.add_factor(
                    (storage_terminal(vertex), storage_terminal(neighbour)),
                    (0, spec.exchange_side, spec.exchange_side, 0),
                )
        if spec.exchange_hull:
            signature = signature.add_factor(
                (storage_terminal(vertex), HULL_TERMINAL),
                (0, spec.exchange_hull, spec.exchange_hull, 0),
            )

    for terminal in terminals:
        signature = signature.forget(terminal)
    return metrics.generated_heat, signature.minimum_cut
