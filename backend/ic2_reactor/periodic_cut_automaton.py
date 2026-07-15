"""Compile a fixed periodic-prefix Hoffman cut into bounded local factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .frontier_automata import (
    FactorAutomatonSelection,
    FrontierConstraintAutomaton,
    LocalScoreFactor,
    select_factor_automaton,
)
from .mathematical_model import ReactorProblem
from .periodic_prefix import PeriodicPrefixCutTemplate, PrefixHeatComponent


def _ordered_split(total: int, receivers: Sequence[int]) -> tuple[dict[int, int], int]:
    shares = {receiver: 0 for receiver in receivers}
    remaining = total
    pending = list(receivers)
    while pending and remaining > 0:
        amount = remaining // len(pending)
        remaining -= amount
        shares[pending.pop(0)] += amount
    return shares, remaining


@dataclass(frozen=True, slots=True)
class PeriodicPrefixCutFactorization:
    labels: tuple[str, ...]
    factors: tuple[LocalScoreFactor, ...]
    automaton: FrontierConstraintAutomaton
    selection: FactorAutomatonSelection

    def score(self, layout: Sequence[str]) -> int:
        """Return ``upper_bound_out - lower_bound_in`` from the local factors."""

        if len(layout) != len(self.automaton.placement_order):
            raise ValueError("layout length differs from the factor placement order")
        code_by_label = {label: code for code, label in enumerate(self.labels)}
        if unknown := set(layout) - code_by_label.keys():
            raise ValueError(f"factorized cut has unknown labels: {sorted(unknown)}")
        codes = tuple(code_by_label[label] for label in layout)
        return sum(
            factor.evaluate(tuple(codes[vertex] for vertex in factor.scope))
            for factor in self.factors
        )


def compile_periodic_prefix_cut(
    problem: ReactorProblem,
    cut: PeriodicPrefixCutTemplate,
    catalogue: Mapping[str, PrefixHeatComponent],
    labels: Sequence[str],
    *,
    base_hull_capacity: int,
    placement_order: Sequence[int],
) -> PeriodicPrefixCutFactorization:
    """Compile one exact Hoffman inequality without enumerating full layouts.

    All ordinary time-expanded capacities are unary in a slot label.  Exact
    fuel injection is a radius-one star factor because it depends on the fuel
    type, pulse-active neighbours and the ordered set of accepting neighbours.
    The resulting inequality is accepted exactly when the factor sum is
    non-negative.
    """

    if base_hull_capacity <= 0:
        raise ValueError("base hull capacity must be positive")
    label_domain = tuple(labels)
    if not label_domain or len(label_domain) != len(set(label_domain)):
        raise ValueError("factor label domain must be non-empty and unique")
    if missing := set(label_domain) - catalogue.keys():
        raise ValueError(f"prefix catalogue is missing labels: {sorted(missing)}")
    if any(catalogue[label].optional_fuel_acceptance for label in label_domain):
        raise ValueError(
            "optional fuel acceptance must be split into resolved label variants"
        )
    order = tuple(problem.graph.update_order)
    event_count = len(order)
    if tuple(sorted(placement_order)) != tuple(problem.graph.vertices):
        raise ValueError("placement order must be a permutation of graph vertices")

    hull = -1
    reservoirs = (hull, *problem.graph.vertices)

    def reservoir_label(reservoir: int) -> str:
        return "hull" if reservoir == hull else f"slot_{reservoir}"

    def before(event: int, reservoir: int) -> str:
        return f"e{event}:{reservoir_label(reservoir)}:in"

    def after(event: int, reservoir: int) -> str:
        return f"e{event}:{reservoir_label(reservoir)}:out"

    def middle_in(event: int, vertex: int) -> str:
        return f"e{event}:slot_{vertex}:action_pool_in"

    def middle_out(event: int, vertex: int) -> str:
        return f"e{event}:slot_{vertex}:action_pool_out"

    environment = "environment"
    known_names = {environment}
    for event, vertex in enumerate(order):
        for reservoir in reservoirs:
            known_names.add(before(event, reservoir))
            known_names.add(after(event, reservoir))
        known_names.add(middle_in(event, vertex))
        known_names.add(middle_out(event, vertex))
    source_side = set(cut.source_side_nodes)
    if unknown := source_side - known_names:
        raise ValueError(f"prefix cut has unknown canonical nodes: {sorted(unknown)}")

    def upper_coefficient(start: str, end: str) -> int:
        return int(start in source_side and end not in source_side)

    def fixed_flow_coefficient(start: str, end: str) -> int:
        if start in source_side and end not in source_side:
            return 1
        if start not in source_side and end in source_side:
            return -1
        return 0

    code_count = len(label_domain)
    unary = [
        [0] * code_count
        for _vertex in problem.graph.vertices
    ]
    constant = 0

    def add_capacity(coefficient: int, reservoir: int) -> None:
        nonlocal constant
        if not coefficient:
            return
        if reservoir == hull:
            constant += coefficient * base_hull_capacity
            for vertex in problem.graph.vertices:
                for code, label in enumerate(label_domain):
                    unary[vertex][code] += (
                        coefficient * catalogue[label].hull_capacity_bonus
                    )
        else:
            for code, label in enumerate(label_domain):
                unary[reservoir][code] += (
                    coefficient * catalogue[label].heat_capacity
                )

    def add_unary_edge(
        start: str,
        end: str,
        vertex: int,
        attribute: str,
    ) -> None:
        coefficient = upper_coefficient(start, end)
        if not coefficient:
            return
        for code, label in enumerate(label_domain):
            unary[vertex][code] += coefficient * int(
                getattr(catalogue[label], attribute)
            )

    # Canonical per-event storage gates.
    for event in range(event_count):
        for reservoir in reservoirs:
            add_capacity(
                upper_coefficient(before(event, reservoir), after(event, reservoir)),
                reservoir,
            )

    injection_factors: list[LocalScoreFactor] = []
    power_by_id = {item.id: item for item in problem.power_components}
    for event, vertex in enumerate(order):
        next_event = (event + 1) % event_count
        for reservoir in reservoirs:
            if reservoir == vertex:
                continue
            add_capacity(
                upper_coefficient(
                    after(event, reservoir),
                    before(next_event, reservoir),
                ),
                reservoir,
            )

        add_capacity(
            upper_coefficient(after(event, vertex), middle_in(event, vertex)),
            vertex,
        )
        add_unary_edge(
            after(event, hull),
            middle_in(event, vertex),
            vertex,
            "hull_draw",
        )
        add_capacity(
            upper_coefficient(middle_in(event, vertex), middle_out(event, vertex)),
            vertex,
        )
        add_capacity(
            upper_coefficient(
                middle_out(event, vertex),
                before(next_event, vertex),
            ),
            vertex,
        )
        add_unary_edge(
            middle_out(event, vertex),
            environment,
            vertex,
            "self_vent",
        )

        for neighbour in problem.graph.neighbours[vertex]:
            add_unary_edge(
                after(event, neighbour),
                environment,
                vertex,
                "side_vent",
            )
            add_unary_edge(
                after(event, vertex),
                before(next_event, neighbour),
                vertex,
                "exchange_side",
            )
            add_unary_edge(
                after(event, neighbour),
                before(next_event, vertex),
                vertex,
                "exchange_side",
            )
        add_unary_edge(
            after(event, vertex),
            before(next_event, hull),
            vertex,
            "exchange_hull",
        )
        add_unary_edge(
            after(event, hull),
            before(next_event, vertex),
            vertex,
            "exchange_hull",
        )

        neighbours = tuple(problem.graph.neighbours[vertex])
        scope = (vertex, *neighbours)

        center_signatures = tuple(
            (
                None
                if (power_spec := power_by_id.get(label)) is None
                else (
                    power_spec.rods,
                    power_spec.internal_pulses,
                    power_spec.accepts_pulse,
                )
            )
            for label in label_domain
        )
        neighbour_signatures = tuple(
            (
                bool(
                    power_by_id.get(label)
                    and power_by_id[label].accepts_pulse
                ),
                catalogue[label].accepts_fuel_heat,
            )
            for label in label_domain
        )

        def injection_score(codes: tuple[int, ...]) -> int:
            center_label = label_domain[codes[0]]
            power_spec = power_by_id.get(center_label)
            if power_spec is None or power_spec.rods <= 0:
                return 0
            neighbour_labels = tuple(label_domain[code] for code in codes[1:])
            degree = sum(
                bool(
                    power_by_id.get(label)
                    and power_by_id[label].accepts_pulse
                )
                for label in neighbour_labels
            )
            pulses = power_spec.internal_pulses + degree
            per_rod_heat = problem.heat_scale * pulses * (pulses + 1)
            acceptors = tuple(
                neighbour
                for neighbour, label in zip(neighbours, neighbour_labels, strict=True)
                if catalogue[label].accepts_fuel_heat
            )
            total_shares = {receiver: 0 for receiver in acceptors}
            hull_share = 0
            for _rod in range(power_spec.rods):
                shares, remainder = _ordered_split(per_rod_heat, acceptors)
                for receiver, amount in shares.items():
                    total_shares[receiver] += amount
                hull_share += remainder
            score = sum(
                fixed_flow_coefficient(
                    environment,
                    before(next_event, receiver),
                ) * amount
                for receiver, amount in total_shares.items()
            )
            score += fixed_flow_coefficient(
                environment,
                before(next_event, hull),
            ) * hull_share
            return score

        factor = LocalScoreFactor.tabulate_quotiented(
            scope,
            code_count,
            (center_signatures,) + (neighbour_signatures,) * len(neighbours),
            injection_score,
        )
        if any(factor.values):
            injection_factors.append(factor)

    # Store the global constant in one unary factor.  This avoids a separate
    # offset coordinate and preserves exact factor-sum equality.
    first_vertex = problem.graph.vertices[0]
    unary[first_vertex] = [value + constant for value in unary[first_vertex]]
    unary_factors = tuple(
        LocalScoreFactor((vertex,), code_count, tuple(values))
        for vertex, values in zip(problem.graph.vertices, unary, strict=True)
        if any(values)
    )
    factors = (*unary_factors, *injection_factors)
    automaton, selection = select_factor_automaton(
        placement_order,
        factors,
        threshold=0,
    )
    return PeriodicPrefixCutFactorization(
        labels=label_domain,
        factors=tuple(factors),
        automaton=automaton,
        selection=selection,
    )
