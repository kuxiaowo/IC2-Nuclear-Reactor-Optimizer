"""Exact least/greatest fixed-point model for deterministic infinite safety."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, Hashable, Mapping, Sequence, TypeVar


StateT = TypeVar("StateT", bound=Hashable)
ParameterT = TypeVar("ParameterT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class FailureAttractorProof(Generic[StateT]):
    """Partition of a finite deterministic graph with a checkable ranking."""

    states: frozenset[StateT]
    bad_states: frozenset[StateT]
    failure_attractor: frozenset[StateT]
    safe_invariant: frozenset[StateT]
    failure_ranks: tuple[tuple[StateT, int], ...]
    reverse_edges_processed: int

    def rank_by_state(self) -> dict[StateT, int]:
        return dict(self.failure_ranks)

    def verify(self, successor: Mapping[StateT, StateT]) -> bool:
        """Check the inductive invariant and decreasing failure ranking."""

        if self.failure_attractor | self.safe_invariant != self.states:
            return False
        if self.failure_attractor & self.safe_invariant:
            return False
        if not self.bad_states <= self.failure_attractor:
            return False
        ranks = self.rank_by_state()
        if set(ranks) != set(self.failure_attractor):
            return False
        if any(ranks[state] != 0 for state in self.bad_states):
            return False
        for state in self.states - self.bad_states:
            following = successor.get(state)
            if following not in self.states:
                return False
            if state in self.safe_invariant:
                if following not in self.safe_invariant:
                    return False
            elif ranks.get(following) != ranks[state] - 1:
                return False
        return True


def finite_failure_attractor(
    states: Sequence[StateT],
    successor: Mapping[StateT, StateT],
    bad_states: Sequence[StateT],
) -> FailureAttractorProof[StateT]:
    """Compute ``mu Y. bad union pre(Y)`` in linear explicit-state time.

    The transition is deterministic.  Every attracted non-bad state receives
    the exact number of remaining transitions to the first bad state.  The
    complement is a greatest safe invariant and therefore contains precisely
    the states whose unique infinite trajectory never fails.
    """

    domain = frozenset(states)
    if not domain:
        raise ValueError("finite safety domain must be non-empty")
    bad = frozenset(bad_states)
    if not bad <= domain:
        raise ValueError("bad states must belong to the finite domain")
    missing = {
        state
        for state in domain - bad
        if state not in successor or successor[state] not in domain
    }
    if missing:
        raise ValueError(
            "every non-bad state must have one successor inside the domain"
        )

    predecessors: dict[StateT, list[StateT]] = {
        state: [] for state in domain
    }
    edge_count = 0
    for state in domain - bad:
        following = successor[state]
        predecessors[following].append(state)
        edge_count += 1

    ranks: dict[StateT, int] = {state: 0 for state in bad}
    queue = deque(bad)
    while queue:
        following = queue.popleft()
        following_rank = ranks[following]
        for state in predecessors[following]:
            if state in ranks:
                continue
            ranks[state] = following_rank + 1
            queue.append(state)
    attractor = frozenset(ranks)
    invariant = domain - attractor
    proof = FailureAttractorProof(
        states=domain,
        bad_states=bad,
        failure_attractor=attractor,
        safe_invariant=invariant,
        failure_ranks=tuple(sorted(
            ranks.items(),
            key=lambda item: (item[1], repr(item[0])),
        )),
        reverse_edges_processed=edge_count,
    )
    if not proof.verify(successor):  # pragma: no cover - construction theorem
        raise AssertionError("constructed finite safety proof failed verification")
    return proof


@dataclass(frozen=True, slots=True)
class SafeParameterOptimum(Generic[ParameterT]):
    feasible_parameters: tuple[ParameterT, ...]
    optimum_value: int | None
    optimal_parameters: tuple[ParameterT, ...]


def maximize_safe_initial_parameters(
    parameters: Sequence[ParameterT],
    initial_states: Mapping[ParameterT, StateT],
    objective_values: Mapping[ParameterT, int],
    proof: FailureAttractorProof[StateT],
) -> SafeParameterOptimum[ParameterT]:
    """Optimize frozen layout parameters after one joint safety fixed point."""

    domain = tuple(dict.fromkeys(parameters))
    if set(domain) - initial_states.keys() or set(domain) - objective_values.keys():
        raise ValueError("every parameter needs an initial state and objective")
    if unknown := set(initial_states[parameter] for parameter in domain) - proof.states:
        raise ValueError(f"initial states are outside the fixed point: {unknown}")
    feasible = tuple(
        parameter
        for parameter in domain
        if initial_states[parameter] in proof.safe_invariant
    )
    optimum = max(
        (int(objective_values[parameter]) for parameter in feasible),
        default=None,
    )
    optimal = tuple(
        parameter
        for parameter in feasible
        if int(objective_values[parameter]) == optimum
    )
    return SafeParameterOptimum(feasible, optimum, optimal)
