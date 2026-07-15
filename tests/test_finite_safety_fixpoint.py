from __future__ import annotations

from ic2_reactor.finite_safety_fixpoint import (
    finite_failure_attractor,
    maximize_safe_initial_parameters,
)


def test_failure_attractor_proves_ranking_and_safe_invariant() -> None:
    # 0 -> 1 -> bad(4), while 2 <-> 3 is an infinite safe cycle.
    states = (0, 1, 2, 3, 4)
    successor = {0: 1, 1: 4, 2: 3, 3: 2}
    proof = finite_failure_attractor(states, successor, (4,))
    assert proof.verify(successor)
    assert proof.failure_attractor == frozenset((0, 1, 4))
    assert proof.safe_invariant == frozenset((2, 3))
    assert proof.rank_by_state() == {4: 0, 1: 1, 0: 2}


def test_frozen_parameters_are_optimized_after_one_joint_fixed_point() -> None:
    # Parameter/layout A starts in the safe cycle; B has higher static power
    # but starts in the failure attractor.  The fixed point is shared.
    proof = finite_failure_attractor(
        (0, 1, 2, 3, 4),
        {0: 1, 1: 4, 2: 3, 3: 2},
        (4,),
    )
    result = maximize_safe_initial_parameters(
        ("layout_a", "layout_b"),
        {"layout_a": 2, "layout_b": 0},
        {"layout_a": 5, "layout_b": 9},
        proof,
    )
    assert result.feasible_parameters == ("layout_a",)
    assert result.optimum_value == 5
    assert result.optimal_parameters == ("layout_a",)
