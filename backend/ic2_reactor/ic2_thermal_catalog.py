"""IC2 catalogue adapter for the ruleset-independent heat-flow model."""

from __future__ import annotations

from .components import COMPONENTS
from .mathematical_model import ReactorProblem
from .periodic_prefix import PrefixHeatComponent
from .thermal_relaxation import HeatFlowComponent


IC2_HEAT_FLOW_CATALOGUE: dict[str, HeatFlowComponent] = {
    component_id: HeatFlowComponent(
        accepts_heat=spec.accepts_heat,
        self_vent=spec.self_vent,
        side_vent=spec.side_vent,
        hull_draw=spec.hull_draw,
        exchange_side=spec.exchange_side,
        exchange_hull=spec.exchange_hull,
        optional_heat_acceptance=spec.kind == "condensator",
    )
    for component_id, spec in COMPONENTS.items()
}


IC2_PERIODIC_PREFIX_CATALOGUE: dict[str, PrefixHeatComponent] = {
    component_id: PrefixHeatComponent(
        # Condensators are monotone stores: their heat cannot be removed.  In
        # a repeated state they have zero useful net flow and may be treated
        # as non-accepting, which matches their eventual full state.
        heat_capacity=(
            spec.max_heat
            if spec.accepts_heat and spec.kind != "condensator"
            else 0
        ),
        accepts_fuel_heat=spec.accepts_heat and spec.kind != "condensator",
        self_vent=spec.self_vent,
        side_vent=spec.side_vent,
        hull_draw=spec.hull_draw,
        exchange_side=spec.exchange_side,
        exchange_hull=spec.exchange_hull,
        hull_capacity_bonus=spec.hull_capacity_bonus,
    )
    for component_id, spec in COMPONENTS.items()
}


IDEAL_VENT = "ideal_mandatory_vent"
IDEAL_OPTIONAL_STORAGE = "ideal_optional_storage"
IDEAL_SIDE_VENT = "ideal_side_vent"
IDEAL_SIDE_EXCHANGER = "ideal_optional_side_exchanger"
IDEAL_HULL_EXCHANGER = "ideal_optional_hull_exchanger"
IDEAL_MIXED_EXCHANGER = "ideal_optional_mixed_exchanger"


def ic2_optimistic_thermal_problem(
    problem: ReactorProblem,
) -> tuple[ReactorProblem, dict[str, HeatFlowComponent], dict[str, str]]:
    """Collapse IC2 non-power labels to three component-wise dominators.

    Acceptance is optional for the two heat-storing dominators.  This covers
    both ordinary stores (which must accept while safe) and condensators
    (which may reject once full), so the quotient is an optimistic relaxation.
    """

    power_ids = {item.id for item in problem.power_components}
    mapping = {item: item for item in power_ids}
    groups: dict[str, list[str]] = {
        "empty": ["empty"],
        IDEAL_VENT: [],
        IDEAL_OPTIONAL_STORAGE: [],
        IDEAL_SIDE_VENT: [],
        IDEAL_SIDE_EXCHANGER: [],
        IDEAL_HULL_EXCHANGER: [],
        IDEAL_MIXED_EXCHANGER: [],
    }
    for label in problem.layout_components:
        if label == "empty":
            continue
        spec = COMPONENTS[label]
        if spec.kind == "exchanger":
            if spec.exchange_side == 36:
                target = IDEAL_SIDE_EXCHANGER
            elif spec.exchange_hull == 72:
                target = IDEAL_HULL_EXCHANGER
            else:
                target = IDEAL_MIXED_EXCHANGER
        elif spec.kind == "condensator":
            target = IDEAL_OPTIONAL_STORAGE
        elif spec.side_vent > 0 and not spec.accepts_heat:
            target = IDEAL_SIDE_VENT
        elif spec.accepts_heat:
            target = IDEAL_VENT
        else:
            target = "empty"
        mapping[label] = target
        groups[target].append(label)

    raw_limits = dict(problem.component_limits)

    def aggregate_limit(labels: list[str]) -> int | None:
        limits = [raw_limits.get(label) for label in labels]
        if not limits or any(limit is None for limit in limits):
            return None
        return sum(int(limit) for limit in limits)

    quotient_limits = []
    for item in problem.power_components:
        if item.id != "empty" and item.id in raw_limits:
            quotient_limits.append((item.id, raw_limits[item.id]))
    for group_id, members in groups.items():
        limit = aggregate_limit(members)
        if limit is not None:
            quotient_limits.append((group_id, limit))

    quotient = ReactorProblem(
        graph=problem.graph,
        rod_budget=problem.rod_budget,
        exact_rods=problem.exact_rods,
        power_components=problem.power_components,
        cooling_components=(),
        layout_components=tuple(
            group_id
            for group_id in (
                IDEAL_VENT,
                IDEAL_OPTIONAL_STORAGE,
                IDEAL_SIDE_VENT,
                IDEAL_SIDE_EXCHANGER,
                IDEAL_HULL_EXCHANGER,
                IDEAL_MIXED_EXCHANGER,
            )
            if groups[group_id]
        ),
        component_limits=tuple(quotient_limits),
        eu_per_pulse=problem.eu_per_pulse,
        heat_scale=problem.heat_scale,
        ruleset=f"{problem.ruleset}:optimistic-thermal-quotient",
    )
    catalogue = {
        item.id: HeatFlowComponent() for item in problem.power_components
    }
    catalogue.update({
        IDEAL_VENT: HeatFlowComponent(
            accepts_heat=True,
            self_vent=20,
            hull_draw=36,
        ),
        IDEAL_OPTIONAL_STORAGE: HeatFlowComponent(
            accepts_heat=True,
            optional_heat_acceptance=True,
        ),
        IDEAL_SIDE_VENT: HeatFlowComponent(side_vent=4),
        IDEAL_SIDE_EXCHANGER: HeatFlowComponent(
            accepts_heat=True,
            exchange_side=36,
        ),
        IDEAL_HULL_EXCHANGER: HeatFlowComponent(
            accepts_heat=True,
            exchange_hull=72,
        ),
        IDEAL_MIXED_EXCHANGER: HeatFlowComponent(
            accepts_heat=True,
            exchange_side=24,
            exchange_hull=8,
        ),
    })
    return quotient, catalogue, mapping
