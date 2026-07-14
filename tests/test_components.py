from ic2_reactor.components import COMPONENTS


def test_component_registry_matches_221_values():
    vents = {
        "heat_vent": (6, 0, 0, 1_000),
        "advanced_heat_vent": (12, 0, 0, 1_000),
        "reactor_heat_vent": (5, 5, 0, 1_000),
        "component_heat_vent": (0, 0, 4, 0),
        "overclocked_heat_vent": (20, 36, 0, 1_000),
    }
    for component_id, expected in vents.items():
        spec = COMPONENTS[component_id]
        assert (spec.self_vent, spec.hull_draw, spec.side_vent, spec.max_heat) == expected

    exchangers = {
        "heat_exchanger": (12, 4, 2_500),
        "advanced_heat_exchanger": (24, 8, 10_000),
        "reactor_heat_exchanger": (0, 72, 5_000),
        "component_heat_exchanger": (36, 0, 5_000),
    }
    for component_id, expected in exchangers.items():
        spec = COMPONENTS[component_id]
        assert (spec.exchange_side, spec.exchange_hull, spec.max_heat) == expected

    assert [COMPONENTS[x].max_heat for x in ("coolant_10k", "coolant_30k", "coolant_60k")] == [10_000, 30_000, 60_000]
    assert [COMPONENTS[x].max_heat for x in ("rsh_condensator", "lzh_condensator")] == [20_000, 100_000]
    assert [COMPONENTS[x].hull_capacity_bonus for x in ("reactor_plating", "heat_capacity_plating", "containment_plating")] == [1_000, 2_000, 500]
    assert [COMPONENTS[x].max_damage for x in ("neutron_reflector", "thick_neutron_reflector", "iridium_reflector")] == [30_000, 120_000, 0]
