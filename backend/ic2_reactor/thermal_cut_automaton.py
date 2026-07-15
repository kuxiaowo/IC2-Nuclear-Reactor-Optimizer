"""Compile an average heat-flow min-cut into bounded local factors."""

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
from .thermal_relaxation import HeatFlowComponent, ThermalCutTemplate


@dataclass(frozen=True, slots=True)
class ThermalCutFactorization:
    labels: tuple[str, ...]
    factors: tuple[LocalScoreFactor, ...]
    automaton: FrontierConstraintAutomaton
    selection: FactorAutomatonSelection

    def score(self, layout: Sequence[str]) -> int:
        """Return ``cut_capacity - generated_heat`` from local factors."""

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


def compile_thermal_cut(
    problem: ReactorProblem,
    cut: ThermalCutTemplate,
    catalogue: Mapping[str, HeatFlowComponent],
    labels: Sequence[str],
    *,
    placement_order: Sequence[int],
) -> ThermalCutFactorization:
    """Compile ``generated_heat <= cut_capacity`` as radius-one factors."""

    label_domain = tuple(labels)
    if not label_domain or len(label_domain) != len(set(label_domain)):
        raise ValueError("factor label domain must be non-empty and unique")
    if missing := set(label_domain) - catalogue.keys():
        raise ValueError(f"heat catalogue is missing labels: {sorted(missing)}")
    if tuple(sorted(placement_order)) != tuple(problem.graph.vertices):
        raise ValueError("placement order must be a permutation of graph vertices")
    vertices = set(problem.graph.vertices)
    storage_source = set(cut.source_storage_slots)
    generator_source = set(cut.source_generator_slots)
    if unknown := (storage_source | generator_source) - vertices:
        raise ValueError(f"thermal cut has vertices outside the graph: {sorted(unknown)}")

    power_by_id = {item.id: item for item in problem.power_components}
    code_count = len(label_domain)
    factors = []
    for vertex in problem.graph.vertices:
        neighbours = tuple(problem.graph.neighbours[vertex])
        scope = (vertex, *neighbours)

        center_signatures = tuple(
            (
                (
                    None
                    if (power_spec := power_by_id.get(label)) is None
                    else (
                        power_spec.rods,
                        power_spec.internal_pulses,
                        power_spec.accepts_pulse,
                    )
                ),
                catalogue[label].accepts_heat,
                catalogue[label].self_vent,
                catalogue[label].hull_draw,
                catalogue[label].exchange_side,
                catalogue[label].exchange_hull,
            )
            for label in label_domain
        )
        neighbour_signatures = tuple(
            (
                bool(
                    power_by_id.get(label)
                    and power_by_id[label].accepts_pulse
                ),
                catalogue[label].accepts_heat,
                catalogue[label].side_vent,
            )
            for label in label_domain
        )

        def star_score(codes: tuple[int, ...]) -> int:
            center_label = label_domain[codes[0]]
            neighbour_labels = tuple(label_domain[code] for code in codes[1:])
            center_spec = catalogue[center_label]
            power_spec = power_by_id.get(center_label)
            heat = 0
            if power_spec is not None and power_spec.rods > 0:
                degree = sum(
                    bool(
                        power_by_id.get(label)
                        and power_by_id[label].accepts_pulse
                    )
                    for label in neighbour_labels
                )
                pulses = power_spec.internal_pulses + degree
                heat = (
                    problem.heat_scale
                    * power_spec.rods
                    * pulses
                    * (pulses + 1)
                )

            # Every layout pays its exact generated heat.  Remaining terms are
            # precisely the fixed cut's source-to-sink edge capacities.
            score = -heat
            if heat:
                if vertex not in generator_source:
                    score += heat  # source -> generator
                if vertex in generator_source and not cut.hull_source_side:
                    score += heat  # generator -> hull
                if vertex in generator_source:
                    score += heat * sum(
                        catalogue[label].accepts_heat
                        and neighbour not in storage_source
                        for neighbour, label in zip(
                            neighbours,
                            neighbour_labels,
                            strict=True,
                        )
                    )

            if center_spec.accepts_heat and vertex in storage_source:
                score += center_spec.self_vent
                score += sum(catalogue[label].side_vent for label in neighbour_labels)
            if (
                center_spec.accepts_heat
                and center_spec.hull_draw
                and cut.hull_source_side
                and vertex not in storage_source
            ):
                score += center_spec.hull_draw

            if center_spec.exchange_side:
                score += center_spec.exchange_side * sum(
                    catalogue[label].accepts_heat
                    and ((vertex in storage_source) != (neighbour in storage_source))
                    for neighbour, label in zip(
                        neighbours,
                        neighbour_labels,
                        strict=True,
                    )
                )
            if center_spec.exchange_hull and (
                (vertex in storage_source) != cut.hull_source_side
            ):
                score += center_spec.exchange_hull
            return score

        factor = LocalScoreFactor.tabulate_quotiented(
            scope,
            code_count,
            (center_signatures,) + (neighbour_signatures,) * len(neighbours),
            star_score,
        )
        if any(factor.values):
            factors.append(factor)

    automaton, selection = select_factor_automaton(
        placement_order,
        factors,
        threshold=0,
    )
    return ThermalCutFactorization(
        labels=label_domain,
        factors=tuple(factors),
        automaton=automaton,
        selection=selection,
    )
