"""Exact safety fixed points over a sequence of small local events.

Materialising one closed tick function ``s' = F(p, s)`` can be much larger
than the official row-major program that defines it.  In particular, composing
a clipped subtraction after an exchanger creates a large OBDD even though both
local operations are small.  This module keeps the program partitioned.

For an event ``E`` with local failure predicate ``f_E`` and deterministic bit
substitutions ``E(v)``, the exact failure-or-target predecessor is

``Pre_E(Y) = f_E OR Y[v <- E(v)]``.

Applying this operator in reverse event order is exactly the predecessor of a
whole reactor tick.  No intermediate state is enumerated and no monolithic
current/next relation or closed transition function is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from .robdd import ROBDDManager


@dataclass(frozen=True, slots=True)
class SymbolicLocalEvent:
    """One deterministic event represented only by its non-identity bits."""

    name: str
    changed_next_functions: tuple[tuple[Hashable, int], ...]
    failure_root: int = 0

    @classmethod
    def from_mapping(
        cls,
        name: str,
        changed_next_functions: Mapping[Hashable, int],
        failure_root: int = 0,
    ) -> SymbolicLocalEvent:
        return cls(name, tuple(changed_next_functions.items()), failure_root)

    def substitutions(self) -> dict[Hashable, int]:
        return dict(self.changed_next_functions)


def _validate_local_events(
    manager: ROBDDManager,
    events: Sequence[SymbolicLocalEvent],
    frozen_variables: Sequence[Hashable],
) -> tuple[tuple[SymbolicLocalEvent, ...], frozenset[Hashable]]:
    ordered = tuple(events)
    frozen = frozenset(frozen_variables)
    manager_variables = set(manager.variables)
    if len(frozen) != len(tuple(frozen_variables)):
        raise ValueError("frozen local-event variables must be unique")
    if not frozen <= manager_variables:
        raise ValueError("frozen local-event variables are outside the manager")
    for event in ordered:
        substitutions = event.substitutions()
        if not event.name:
            raise ValueError("local symbolic events need non-empty names")
        if len(substitutions) != len(event.changed_next_functions):
            raise ValueError("a local event changes one variable more than once")
        if unknown := set(substitutions) - manager_variables:
            raise ValueError(f"local event changes unknown variables: {unknown}")
        if set(substitutions) & frozen:
            raise ValueError("local events cannot modify frozen variables")
    return ordered, frozen


def local_failure_preimage(
    manager: ROBDDManager,
    event: SymbolicLocalEvent,
    target_root: int,
) -> int:
    """Return states that fail in ``event`` or enter ``target_root`` after it."""

    substitutions = event.substitutions()
    if substitutions and not manager.support(target_root).isdisjoint(substitutions):
        predecessor = manager.compose(target_root, substitutions)
    else:
        predecessor = target_root
    return manager.apply("or", event.failure_root, predecessor)


def sequential_failure_preimage(
    manager: ROBDDManager,
    events: Sequence[SymbolicLocalEvent],
    target_root: int,
    *,
    frozen_variables: Sequence[Hashable] = (),
) -> int:
    """Exact whole-pass predecessor by reverse local substitution/elimination."""

    ordered, _frozen = _validate_local_events(
        manager,
        events,
        frozen_variables,
    )
    predecessor = target_root
    for event in reversed(ordered):
        predecessor = local_failure_preimage(manager, event, predecessor)
    return predecessor


@dataclass(frozen=True, slots=True)
class LocalEventFailureAttractorProof:
    """Replayable least-fixed-point certificate for partitioned events."""

    state_variables: tuple[Hashable, ...]
    frozen_variables: tuple[Hashable, ...]
    events: tuple[SymbolicLocalEvent, ...]
    bad_root: int
    expansion_layer_roots: tuple[int, ...]
    attractor_root: int
    safe_invariant_root: int
    iterations: int
    peak_reachable_nodes: int
    allocated_nodes: int

    def verify(self, manager: ROBDDManager) -> bool:
        if set(self.state_variables) != set(manager.variables):
            return False
        try:
            events, frozen = _validate_local_events(
                manager,
                self.events,
                self.frozen_variables,
            )
        except ValueError:
            return False
        if frozen != frozenset(self.frozen_variables):
            return False
        if not self.expansion_layer_roots:
            return False
        if self.expansion_layer_roots[0] != self.bad_root:
            return False
        attractor = 0
        for index, layer in enumerate(self.expansion_layer_roots):
            if index == 0:
                expected = self.bad_root
            else:
                expected = manager.apply(
                    "and",
                    sequential_failure_preimage(
                        manager,
                        events,
                        attractor,
                        frozen_variables=self.frozen_variables,
                    ),
                    manager.negate(attractor),
                )
            if layer != expected:
                return False
            attractor = manager.apply("or", attractor, layer)
        if attractor != self.attractor_root:
            return False
        if manager.negate(attractor) != self.safe_invariant_root:
            return False
        outside_predecessor = manager.apply(
            "and",
            sequential_failure_preimage(
                manager,
                events,
                attractor,
                frozen_variables=self.frozen_variables,
            ),
            manager.negate(attractor),
        )
        return outside_predecessor == 0


def local_event_failure_attractor(
    manager: ROBDDManager,
    events: Sequence[SymbolicLocalEvent],
    bad_root: int,
    *,
    frozen_variables: Sequence[Hashable] = (),
    maximum_iterations: int | None = None,
) -> LocalEventFailureAttractorProof:
    """Compute the exact least failure attractor without closing the event list."""

    ordered, frozen = _validate_local_events(
        manager,
        events,
        frozen_variables,
    )
    if maximum_iterations is not None and maximum_iterations <= 0:
        raise ValueError("maximum local-event iterations must be positive or None")

    attractor = bad_root
    layers = [bad_root]
    peak_nodes = manager.reachable_node_count(bad_root)
    while True:
        if maximum_iterations is not None and len(layers) > maximum_iterations:
            raise TimeoutError("local-event failure attractor limit reached")
        following = manager.apply(
            "and",
            sequential_failure_preimage(
                manager,
                ordered,
                attractor,
                frozen_variables=tuple(frozen),
            ),
            manager.negate(attractor),
        )
        if following == 0:
            break
        layers.append(following)
        attractor = manager.apply("or", attractor, following)
        peak_nodes = max(
            peak_nodes,
            manager.reachable_node_count(following),
            manager.reachable_node_count(attractor),
        )

    proof = LocalEventFailureAttractorProof(
        state_variables=tuple(manager.variables),
        frozen_variables=tuple(
            variable for variable in manager.variables if variable in frozen
        ),
        events=ordered,
        bad_root=bad_root,
        expansion_layer_roots=tuple(layers),
        attractor_root=attractor,
        safe_invariant_root=manager.negate(attractor),
        iterations=len(layers) - 1,
        peak_reachable_nodes=peak_nodes,
        allocated_nodes=manager.allocated_node_count,
    )
    if not proof.verify(manager):  # pragma: no cover
        raise AssertionError("constructed local-event safety proof failed verification")
    return proof
