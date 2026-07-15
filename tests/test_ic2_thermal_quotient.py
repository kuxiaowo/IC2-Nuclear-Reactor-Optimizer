from __future__ import annotations

from ic2_reactor.ic2_thermal_catalog import (
    IC2_HEAT_FLOW_CATALOGUE,
    ic2_optimistic_thermal_problem,
)
from ic2_reactor.mathematical_model import ic2_mark_i_problem


def test_optimistic_thermal_quotient_dominates_every_enabled_component() -> None:
    original = ic2_mark_i_problem()
    quotient, catalogue, mapping = ic2_optimistic_thermal_problem(original)
    labels = tuple(dict.fromkeys((
        *(item.id for item in quotient.power_components),
        *quotient.layout_components,
    )))
    assert len(labels) == 11
    assert set(mapping) == set(item.id for item in original.power_components) | set(
        original.layout_components
    )

    attributes = (
        "self_vent",
        "side_vent",
        "hull_draw",
        "exchange_side",
        "exchange_hull",
    )
    for original_label, quotient_label in mapping.items():
        actual = IC2_HEAT_FLOW_CATALOGUE[original_label]
        relaxed = catalogue[quotient_label]
        assert not actual.accepts_heat or relaxed.accepts_heat
        assert all(
            getattr(relaxed, attribute) >= getattr(actual, attribute)
            for attribute in attributes
        )
        if actual.optional_heat_acceptance:
            assert relaxed.optional_heat_acceptance
