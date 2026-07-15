from __future__ import annotations

from ic2_reactor.ic2_symbolic_hull import compile_ic2_hull_only_symbolic_model
from ic2_reactor.ic2_symbolic_oracle import compile_ic2_oracle_symbolic_model
from ic2_reactor.symbolic_safety import symbolic_failure_attractor


def assert_matches_oracle(layout: tuple[str, ...]) -> None:
    direct = compile_ic2_hull_only_symbolic_model(layout)
    oracle = compile_ic2_oracle_symbolic_model(layout, maximum_state_bits=14)
    assert direct.encoded_state_count == oracle.encoded_state_count
    for code in range(direct.encoded_state_count):
        direct_assignment = direct.assignment(code)
        oracle_assignment = oracle.assignment(code)
        direct_bad = direct.manager.evaluate(direct.bad_root, direct_assignment)
        oracle_bad = oracle.manager.evaluate(oracle.bad_root, oracle_assignment)
        assert direct_bad == oracle_bad
        if direct_bad:
            continue
        for direct_variable, oracle_variable in zip(
            direct.state_variables,
            oracle.state_variables,
            strict=True,
        ):
            assert direct.manager.evaluate(
                direct.next_functions[direct_variable],
                direct_assignment,
            ) == oracle.manager.evaluate(
                oracle.next_functions[oracle_variable],
                oracle_assignment,
            )


def test_direct_hull_circuit_matches_oracle_for_empty_layout() -> None:
    layout = ("empty",) * 18
    assert_matches_oracle(layout)
    model = compile_ic2_hull_only_symbolic_model(layout)
    proof = symbolic_failure_attractor(
        model.manager,
        model.state_variables,
        model.next_functions,
        model.bad_root,
    )
    assert model.manager.evaluate(proof.safe_invariant_root, model.assignment(0))


def test_direct_hull_circuit_matches_fuel_and_reflector_pulses() -> None:
    layout = (
        "uranium_quad",
        "iridium_reflector",
        *("empty",) * 16,
    )
    assert_matches_oracle(layout)
    model = compile_ic2_hull_only_symbolic_model(layout)
    assert model.generated_heat == 160
    proof = symbolic_failure_attractor(
        model.manager,
        model.state_variables,
        model.next_functions,
        model.bad_root,
    )
    assert model.manager.evaluate(proof.attractor_root, model.assignment(0))
