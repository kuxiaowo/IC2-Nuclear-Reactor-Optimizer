from __future__ import annotations

from random import Random

from ic2_reactor.engine import ReactorSimulator
from ic2_reactor.ic2_symbolic_no_exchange import compile_ic2_symbolic_model
from ic2_reactor.models import Layout


def assert_official_transition(
    model,
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
    for vertex, heat in component_heat.items():
        simulator.slots[vertex].heat = heat
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


def test_side_exchange_event_circuit_matches_official_random_states() -> None:
    layout = ("component_heat_exchanger", "heat_vent", *("empty",) * 16)
    model = compile_ic2_symbolic_model(layout)
    assert model.event_compactions > 0
    assert model.manager.allocated_node_count < model.peak_allocated_nodes
    rng = Random(17)
    for _sample in range(300):
        hull = rng.randrange(model.critical_heat)
        source = rng.randrange(5_001)
        target = rng.randrange(1_001)
        assert_official_transition(model, hull, {0: source, 1: target})


def test_signed_hull_exchange_and_later_vent_match_official_events() -> None:
    layout = (
        "reactor_heat_exchanger",
        "reactor_heat_vent",
        "heat_capacity_plating",
        *("empty",) * 15,
    )
    model = compile_ic2_symbolic_model(layout)
    # Locked counterexample: the exchanger makes hull heat -1, then the later
    # reactor vent observes that signed value and returns the hull to zero.
    assert_official_transition(model, 71, {0: 29, 1: 10})

    rng = Random(149)
    for _sample in range(300):
        assert_official_transition(
            model,
            rng.randrange(-72, model.critical_heat),
            {0: rng.randrange(5_001), 1: rng.randrange(1_001)},
        )


def test_exchanger_condensator_one_way_transfer_matches_official_events() -> None:
    layout = ("component_heat_exchanger", "rsh_condensator", *("empty",) * 16)
    model = compile_ic2_symbolic_model(layout)
    rng = Random(157)
    for _sample in range(300):
        assert_official_transition(
            model,
            rng.randrange(-72, model.critical_heat),
            {0: rng.randrange(5_001), 1: rng.randrange(20_001)},
        )
    for condensator_heat in range(19_964, 20_001):
        assert_official_transition(
            model,
            0,
            {0: 2_500, 1: condensator_heat},
        )
