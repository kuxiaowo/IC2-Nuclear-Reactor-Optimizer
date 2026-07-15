from __future__ import annotations

from ic2_reactor.ic2_symbolic_oracle import compile_ic2_oracle_symbolic_model
from ic2_reactor.symbolic_safety import symbolic_failure_attractor


def test_empty_ic2_layout_initial_state_is_in_symbolic_safe_invariant() -> None:
    layout = ("empty",) * 18
    model = compile_ic2_oracle_symbolic_model(layout, maximum_state_bits=14)
    proof = symbolic_failure_attractor(
        model.manager,
        model.state_variables,
        model.next_functions,
        model.bad_root,
    )
    assert proof.verify(model.manager, model.next_functions)
    assert model.manager.evaluate(
        proof.safe_invariant_root,
        model.assignment(model.initial_code),
    )


def test_uncoolable_quad_fuel_is_in_symbolic_failure_attractor() -> None:
    layout = ("uranium_quad", *("empty",) * 17)
    model = compile_ic2_oracle_symbolic_model(layout, maximum_state_bits=14)
    proof = symbolic_failure_attractor(
        model.manager,
        model.state_variables,
        model.next_functions,
        model.bad_root,
    )
    assert proof.verify(model.manager, model.next_functions)
    assert model.manager.evaluate(
        proof.attractor_root,
        model.assignment(model.initial_code),
    )
    assert not model.manager.evaluate(
        proof.safe_invariant_root,
        model.assignment(model.initial_code),
    )
