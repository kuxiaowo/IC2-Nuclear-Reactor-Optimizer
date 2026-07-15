from __future__ import annotations

from dataclasses import dataclass
from random import Random

from ic2_reactor.cycle_proof import (
    ConstantMemoryCycleVerifier,
    DeterministicCycleSession,
    DeterministicCycleVerifier,
    IC2TransitionSystem,
    TransitionObservation,
    ic2_safe_state_space_bound,
)


@dataclass(frozen=True)
class ToySystem:
    fail_at: int | None = None

    def initial_state(self, layout: object) -> int:
        return 0

    def step(self, layout: object, state: int) -> TransitionObservation[int]:
        if self.fail_at is not None and state + 1 == self.fail_at:
            return TransitionObservation(state + 1, failed=True, failure_reason="toy_failure")
        # 0 -> 1 -> 2 -> 3 -> 2 gives transient 2 and period 2.
        next_state = 2 if state == 3 else state + 1
        return TransitionObservation(next_state, metrics={"state": next_state})

    def state_key(self, state: int) -> int:
        return state


def test_generic_cycle_verifier_proves_transient_and_period() -> None:
    proof = DeterministicCycleVerifier().verify(ToySystem(), None, max_steps=20)
    assert proof.safe
    assert proof.conclusive
    assert proof.transient_length == 2
    assert proof.period_length == 2
    assert proof.checked_steps == 4


def test_generic_cycle_verifier_distinguishes_failure_from_horizon() -> None:
    failure = DeterministicCycleVerifier().verify(ToySystem(fail_at=2), None, max_steps=20)
    assert failure.conclusive
    assert not failure.safe
    assert failure.failure_step == 2
    assert failure.failure_reason == "toy_failure"

    horizon = DeterministicCycleVerifier().verify(ToySystem(), None, max_steps=1)
    assert not horizon.conclusive
    assert horizon.outcome == "horizon"


def test_cycle_session_resumes_without_replaying_unknown_prefix() -> None:
    session = DeterministicCycleSession(ToySystem(), None)

    first = session.advance(1)
    assert first.outcome == "horizon"
    assert first.checked_steps == 1

    second = session.advance(1)
    assert second.outcome == "horizon"
    assert second.checked_steps == 2

    terminal = session.advance(2)
    assert terminal.safe
    assert terminal.transient_length == 2
    assert terminal.period_length == 2
    assert terminal.checked_steps == 4
    assert session.progress_steps == 4
    assert session.advance(1) is terminal


@dataclass(frozen=True)
class BoundedToySystem(ToySystem):
    def safe_state_upper_bound(self, layout: object) -> int:
        return 4


def test_complete_verifier_uses_declared_finite_state_bound() -> None:
    proof = DeterministicCycleVerifier().verify_complete(
        BoundedToySystem(),
        None,
    )
    assert proof.safe
    assert proof.complete_state_bound_used
    assert proof.state_space_upper_bound == 4


def test_constant_memory_verifier_returns_an_observed_repeat() -> None:
    proof = ConstantMemoryCycleVerifier().verify(
        ToySystem(),
        None,
        max_steps=20,
    )
    assert proof.safe
    assert proof.conclusive
    # Brent need not return the minimum transient, but these are actual equal
    # reachable states x_3 == x_5.
    assert proof.transient_length == 3
    assert proof.period_length == 2
    assert proof.checked_steps == 5


def test_constant_memory_complete_horizon_uses_finite_bound() -> None:
    proof = ConstantMemoryCycleVerifier().verify_complete(
        BoundedToySystem(),
        None,
    )
    assert proof.safe
    assert proof.complete_state_bound_used
    assert proof.state_space_upper_bound == 4


def test_constant_memory_bound_on_random_finite_maps() -> None:
    @dataclass(frozen=True)
    class FiniteMap:
        following: tuple[int, ...]

        def initial_state(self, layout: object) -> int:
            return 0

        def step(self, layout: object, state: int) -> TransitionObservation[int]:
            return TransitionObservation(self.following[state])

        def state_key(self, state: int) -> int:
            return state

        def safe_state_upper_bound(self, layout: object) -> int:
            return len(self.following)

    rng = Random(53)
    for size in range(1, 25):
        for _sample in range(30):
            system = FiniteMap(tuple(rng.randrange(size) for _ in range(size)))
            proof = ConstantMemoryCycleVerifier().verify_complete(system, None)
            assert proof.safe
            assert proof.checked_steps <= 3 * size
            assert proof.transient_length is not None
            assert proof.period_length is not None
            state = 0
            states = [state]
            for _step in range(proof.checked_steps):
                state = system.following[state]
                states.append(state)
            assert states[proof.transient_length] == states[
                proof.transient_length + proof.period_length
            ]


def test_ic2_safe_state_product_is_explicit_and_layout_dependent() -> None:
    empty = ("empty",) * 18
    base = ic2_safe_state_space_bound(empty)
    assert base.hull_states == 8_500
    assert base.safe_states == 8_500
    assert IC2TransitionSystem(3).safe_state_upper_bound(empty) == 8_500

    with_vent = ("heat_vent", *empty[1:])
    vent = ic2_safe_state_space_bound(with_vent)
    assert vent.slot_state_factors[0] == 1_001
    assert vent.safe_states == 8_500 * 1_001


def test_ic2_cycle_key_omits_layout_constants_and_irrelevant_fields() -> None:
    empty = ("empty",) * 18
    system = IC2TransitionSystem(3)
    empty_state = system.initial_state(empty)
    assert system.state_key(empty_state) == (0,)

    layout = ("heat_vent", "neutron_reflector", *empty[2:])
    state = system.initial_state(layout)
    # Hull heat, one heat-accepting component and one finite reflector.
    assert system.state_key(state) == (0, 0, 0)
