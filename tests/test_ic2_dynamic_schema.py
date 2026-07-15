from __future__ import annotations

from ic2_reactor.ic2_dynamic_schema import (
    IC2DynamicStateSchema,
    ic2_dynamic_state_schema,
    ic2_layout_dynamic_schema_signature,
    ic2_layout_structural_signature,
    ic2_permanent_catalogue_quotient,
    ic2_permanent_structural_quotient,
    ic2_permanent_search_representative,
    ic2_structural_signature,
)


def test_all_enabled_permanent_labels_collapse_to_twelve_state_schemas() -> None:
    quotient = ic2_permanent_catalogue_quotient()
    assert len(quotient.labels) == 22
    assert quotient.schema_count == 12
    assert set(quotient.removed_finite_reflectors) == {
        "neutron_reflector",
        "thick_neutron_reflector",
    }
    no_local_state = next(
        labels
        for schema, labels in quotient.schemas
        if schema == IC2DynamicStateSchema(0, 0)
    )
    assert set(no_local_state) == {
        "empty",
        "uranium_single",
        "uranium_dual",
        "uranium_quad",
        "component_heat_vent",
        "iridium_reflector",
    }


def test_schema_equivalence_preserves_dynamic_field_shape_not_event_behavior() -> None:
    assert ic2_dynamic_state_schema("heat_vent") == ic2_dynamic_state_schema(
        "advanced_heat_vent"
    )
    assert ic2_dynamic_state_schema("coolant_10k") == ic2_dynamic_state_schema(
        "advanced_heat_exchanger"
    )
    assert ic2_layout_dynamic_schema_signature((
        "heat_vent",
        "coolant_10k",
    )) == ic2_layout_dynamic_schema_signature((
        "advanced_heat_vent",
        "advanced_heat_exchanger",
    ))


def test_finite_reflectors_have_one_lossless_permanent_representative() -> None:
    assert ic2_permanent_search_representative("neutron_reflector") == (
        "iridium_reflector"
    )
    assert ic2_permanent_search_representative("thick_neutron_reflector") == (
        "iridium_reflector"
    )
    assert ic2_permanent_search_representative("uranium_single") == (
        "uranium_single"
    )


def test_joint_dynamic_and_power_signature_has_sixteen_exact_classes() -> None:
    quotient = ic2_permanent_structural_quotient()
    assert len(quotient.labels) == 22
    assert quotient.signature_count == 16
    assert max(len(labels) for _signature, labels in quotient.groups) == 4
    assert any(
        set(labels) == {"empty", "component_heat_vent"}
        for _signature, labels in quotient.groups
    )
    assert any(
        set(labels) == {
            "heat_vent",
            "advanced_heat_vent",
            "reactor_heat_vent",
            "overclocked_heat_vent",
        }
        for _signature, labels in quotient.groups
    )
    assert ic2_structural_signature("uranium_single") != (
        ic2_structural_signature("uranium_dual")
    )
    layout = ("empty", "heat_vent", "uranium_quad")
    assert ic2_layout_structural_signature(layout) == tuple(
        ic2_structural_signature(label) for label in layout
    )
