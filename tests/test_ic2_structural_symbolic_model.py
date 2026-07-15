from __future__ import annotations

from itertools import product
from random import Random

from ic2_reactor.engine import ReactorSimulator
from ic2_reactor.ic2_dynamic_schema import ic2_layout_structural_signature
from ic2_reactor.ic2_structural_symbolic_model import (
    compile_ic2_structural_no_exchange_symbolic_model,
    compile_ic2_structural_symbolic_model,
    ic2_structural_local_event_work_bound,
)
from ic2_reactor.models import Layout


def test_structural_event_model_matches_every_label_refinement_and_random_state() -> None:
    representative = (
        "uranium_single",
        "rsh_condensator",
        "empty",
        "heat_vent",
        "coolant_30k",
        "empty",
        *("containment_plating",) * 12,
    )
    model = compile_ic2_structural_no_exchange_symbolic_model(
        ic2_layout_structural_signature(representative),
        columns=3,
        exact_rods=1,
    )
    assert tuple(
        len(domain) for domain in model.label_domains if len(domain) > 1
    ) == (2, 4, 2)
    assert model.event_composition.compiled_alternative_count == 23
    assert model.event_composition.skipped_identity_event_count == 15
    assert model.event_composition.explicit_family_count == 16

    layouts = product(*model.label_domains)
    rng = Random(1_407)
    for layout in layouts:
        for _sample in range(24):
            hull = rng.randrange(model.critical_heat)
            component_heat = {
                1: rng.randrange(20_001),
                3: rng.randrange(1_001),
                4: rng.randrange(30_001),
            }
            state_code = model.encode(hull, component_heat)
            assignment = model.assignment(layout, state_code)
            symbolic_failure = model.manager.evaluate(
                model.transition_failure_root,
                assignment,
            )

            simulator = ReactorSimulator(Layout(columns=3, slots=list(layout)))
            simulator.hull_heat = hull
            simulator.peak_hull_heat = hull
            for vertex, value in component_heat.items():
                simulator.slots[vertex].heat = value
            simulator.step(auto_refuel=True)
            actual_failure = (
                simulator.first_critical_tick is not None
                or simulator.first_component_break_tick is not None
                or simulator.meltdown_tick is not None
            )
            assert symbolic_failure == actual_failure
            if actual_failure:
                assert model.next_code(layout, state_code) == (
                    model.encoded_state_count - 1
                )
            else:
                expected = model.encode(
                    simulator.hull_heat,
                    {
                        vertex: simulator.slots[vertex].heat
                        for vertex in component_heat
                    },
                )
                assert model.next_code(layout, state_code) == expected


def test_structural_exchanger_choice_matches_all_refinements_and_signed_hull() -> None:
    representative = (
        "reactor_heat_exchanger",
        "heat_vent",
        *("iridium_reflector",) * 16,
    )
    model = compile_ic2_structural_symbolic_model(
        ic2_layout_structural_signature(representative),
        columns=3,
        exact_rods=0,
    )
    assert tuple(
        len(domain) for domain in model.label_domains if len(domain) > 1
    ) == (2, 4)
    assert model.compiled_alternative_count == 6
    assert len(model.local_events) == 2
    assert len(model.local_event_live_node_counts) == 2
    assert len(model.local_event_support_sizes) == 2
    assert max(model.local_event_support_sizes) < len(model.manager.variables)
    assert model.layout_circuit.manager.model_count(
        model.parameter_constraint_root,
        model.parameter_variables,
    ) == 8

    rng = Random(8_311)
    for layout in product(*model.label_domains):
        samples = [
            (71, {0: 29, 1: 10}),
            *(
                (
                    rng.randrange(-72, model.critical_heat),
                    {0: rng.randrange(5_001), 1: rng.randrange(1_001)},
                )
                for _sample in range(30)
            ),
        ]
        for hull, component_heat in samples:
            state_code = model.encode(hull, component_heat)
            symbolic_next, symbolic_failure = model.step_code(layout, state_code)
            simulator = ReactorSimulator(Layout(columns=3, slots=list(layout)))
            simulator.hull_heat = hull
            simulator.peak_hull_heat = hull
            for vertex, value in component_heat.items():
                simulator.slots[vertex].heat = value
            simulator.step(auto_refuel=True)
            actual_failure = (
                simulator.first_critical_tick is not None
                or simulator.first_component_break_tick is not None
                or simulator.meltdown_tick is not None
            )
            assert symbolic_failure == actual_failure
            if actual_failure:
                assert symbolic_next == model.encoded_state_count - 1
            else:
                expected = model.encode(
                    simulator.hull_heat,
                    {
                        vertex: simulator.slots[vertex].heat
                        for vertex in component_heat
                    },
                )
                assert symbolic_next == expected


def test_local_event_work_bound_replaces_layout_product_by_event_sum() -> None:
    representative = (
        "reactor_heat_exchanger",
        "heat_vent",
        *("iridium_reflector",) * 16,
    )
    bound = ic2_structural_local_event_work_bound(
        ic2_layout_structural_signature(representative),
        columns=3,
    )
    assert bound.parameter_bit_count == 3
    assert bound.unconstrained_refinement_count == 8
    assert bound.eventful_slot_count == 2
    assert bound.local_alternative_count == 6
    assert bound.maximum_local_domain_size == 4
    assert bound.catalogue_maximum_local_alternatives == 72

    largest_54_cell_event_domain = ic2_layout_structural_signature(
        ("heat_vent",) * 54
    )
    full_bound = ic2_structural_local_event_work_bound(
        largest_54_cell_event_domain,
        columns=9,
    )
    assert full_bound.unconstrained_refinement_count == 4 ** 54
    assert full_bound.local_alternative_count == 4 * 54 == 216
    assert full_bound.catalogue_maximum_local_alternatives == 216
    assert full_bound.catalogue_local_support_bit_upper_bound == 104


def test_infeasible_structural_rod_budget_compiles_no_local_alternatives() -> None:
    representative = (
        "reactor_heat_exchanger",
        "heat_vent",
        *("iridium_reflector",) * 16,
    )
    model = compile_ic2_structural_symbolic_model(
        ic2_layout_structural_signature(representative),
        columns=3,
        exact_rods=1,
    )
    assert model.parameter_constraint_root == 0
    assert model.compiled_alternative_count == 0
    assert not model.local_events

