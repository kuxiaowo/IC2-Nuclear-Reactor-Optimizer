from __future__ import annotations

from random import Random

from ic2_reactor.engine import ReactorSimulator
from ic2_reactor.ic2_symbolic_no_exchange import (
    IC2NoExchangeSymbolicModel,
    compile_ic2_no_exchange_symbolic_model,
)
from ic2_reactor.models import Layout
from ic2_reactor.symbolic_safety import symbolic_failure_attractor


def assert_transition_matches(
    model: IC2NoExchangeSymbolicModel,
    hull: int,
    component_heat: dict[int, int],
) -> None:
    code = model.encode(hull, component_heat)
    assignment = model.assignment(code)
    symbolic_failure = model.manager.evaluate(
        model.transition_failure_root,
        assignment,
    )

    simulator = ReactorSimulator(Layout(columns=3, slots=list(model.layout)))
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
        assert model.next_code(code) == model.encoded_state_count - 1
        return
    expected = model.encode(
        simulator.hull_heat,
        {
            vertex: simulator.slots[vertex].heat
            for vertex in component_heat
        },
    )
    assert model.next_code(code) == expected


def assert_random_transitions_match(
    layout: tuple[str, ...],
    heat_vertices: tuple[int, ...],
) -> None:
    model = compile_ic2_no_exchange_symbolic_model(layout)
    rng = Random(79)
    for _sample in range(2_000):
        hull = rng.randrange(model.critical_heat)
        component_heat = {
            vertex: rng.randrange(
                next(
                    field.maximum_safe_value
                    for field in model.fields
                    if field.kind == "heat" and field.vertex == vertex
                ) + 1
            )
            for vertex in heat_vertices
        }
        assert_transition_matches(model, hull, component_heat)


def test_fuel_then_self_vent_order_matches_official_step() -> None:
    layout = ("uranium_single", "heat_vent", *("empty",) * 16)
    assert_random_transitions_match(layout, (1,))


def test_hull_draw_before_later_fuel_matches_official_step() -> None:
    layout = ("reactor_heat_vent", "uranium_single", *("empty",) * 16)
    assert_random_transitions_match(layout, (0,))


def test_side_vent_after_fuel_and_coolant_matches_official_step() -> None:
    layout = (
        "uranium_single",
        "coolant_10k",
        "component_heat_vent",
        *("empty",) * 15,
    )
    assert_random_transitions_match(layout, (1,))


def test_hull_addition_overflow_is_failure() -> None:
    layout = ("uranium_quad",) + (("empty",) * 17)
    model = compile_ic2_no_exchange_symbolic_model(layout)
    # A near-maximum encoded hull value plus generated heat must not wrap to
    # a low, apparently safe value in the symbolic transition circuit.
    code = (1 << model.fields[0].width) - 1
    assignment = model.assignment(code)
    assert model.manager.evaluate(model.transition_failure_root, assignment)


def test_condensator_fill_and_redistribution_match_official_step() -> None:
    layout = (
        "uranium_quad",
        "rsh_condensator",
        "empty",
        "heat_vent",
        *("empty",) * 14,
    )
    assert_random_transitions_match(layout, (1, 3))

    model = compile_ic2_no_exchange_symbolic_model(layout)
    for condensator_heat in range(19_880, 20_001):
        for vent_heat in (0, 950, 999, 1_000):
            assert_transition_matches(
                model,
                0,
                {1: condensator_heat, 3: vent_heat},
            )


def test_side_vent_cannot_remove_condensator_heat() -> None:
    layout = (
        "component_heat_vent",
        "rsh_condensator",
        *("empty",) * 16,
    )
    assert_random_transitions_match(layout, (1,))


def test_direct_event_circuit_feeds_exact_infinite_safety_fixed_point() -> None:
    layout = ("uranium_single", "advanced_heat_vent", *("empty",) * 16)
    model = compile_ic2_no_exchange_symbolic_model(layout)
    proof = symbolic_failure_attractor(
        model.manager,
        model.state_variables,
        model.next_functions,
        model.bad_root,
    )
    initial = model.assignment(model.encode(0, {1: 0}))
    assert model.manager.evaluate(proof.safe_invariant_root, initial)
    assert proof.verify(model.manager, model.next_functions)
