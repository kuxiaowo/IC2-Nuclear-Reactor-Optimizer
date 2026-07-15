"""Partitioned ROBDD fixed point for deterministic infinite safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from .robdd import ROBDDManager


@dataclass(frozen=True, slots=True)
class SymbolicFailureAttractorProof:
    state_variables: tuple[Hashable, ...]
    bad_root: int
    rank_layer_roots: tuple[int, ...]
    attractor_root: int
    safe_invariant_root: int
    iterations: int
    peak_reachable_nodes: int
    allocated_nodes: int

    def verify(
        self,
        manager: ROBDDManager,
        next_functions: Mapping[Hashable, int],
    ) -> bool:
        if self.bad_root == 0:
            return (
                not self.rank_layer_roots
                and self.attractor_root == 0
                and self.safe_invariant_root == 1
            )
        if not self.rank_layer_roots or self.rank_layer_roots[0] != self.bad_root:
            return False
        attractor = 0
        previous = None
        for index, layer in enumerate(self.rank_layer_roots):
            if index == 0:
                expected = self.bad_root
            else:
                assert previous is not None
                expected = manager.apply(
                    "and",
                    manager.compose(previous, next_functions),
                    manager.negate(attractor),
                )
            if layer != expected:
                return False
            attractor = manager.apply("or", attractor, layer)
            previous = layer
        if attractor != self.attractor_root:
            return False
        if manager.negate(attractor) != self.safe_invariant_root:
            return False
        new_bad_predecessor = manager.apply(
            "and",
            manager.compose(attractor, next_functions),
            manager.negate(attractor),
        )
        if new_bad_predecessor != 0:
            return False
        safe_successor = manager.compose(self.safe_invariant_root, next_functions)
        if manager.apply(
            "and",
            self.safe_invariant_root,
            manager.negate(safe_successor),
        ) != 0:
            return False
        return True


@dataclass(frozen=True, slots=True)
class CompactingSymbolicSafetyResult:
    manager: ROBDDManager
    next_functions: dict[Hashable, int]
    proof: SymbolicFailureAttractorProof
    compactions: int
    peak_allocated_nodes: int


def symbolic_failure_attractor(
    manager: ROBDDManager,
    state_variables: Sequence[Hashable],
    next_functions: Mapping[Hashable, int],
    bad_root: int,
    *,
    maximum_iterations: int | None = None,
) -> SymbolicFailureAttractorProof:
    """Compute disjoint failure-rank layers by functional preimage.

    ``next_functions[v]`` is the partitioned Boolean next-state function for
    bit ``v``.  Hence ``Pre(Y)`` is direct simultaneous composition
    ``Y(next_functions)``; no monolithic current/next transition relation or
    existential next-state elimination is constructed.
    """

    variables = tuple(state_variables)
    if not variables or len(variables) != len(set(variables)):
        raise ValueError("symbolic state variables must be non-empty and unique")
    if set(variables) != set(next_functions):
        raise ValueError("every symbolic state bit needs one next-state function")
    if set(variables) - set(manager.variables):
        raise ValueError("symbolic state variables are outside the ROBDD manager")
    if maximum_iterations is not None and maximum_iterations <= 0:
        raise ValueError("maximum symbolic iterations must be positive or None")

    attractor = 0
    layer = bad_root
    layers = []
    peak_nodes = 0
    while layer != 0:
        if maximum_iterations is not None and len(layers) >= maximum_iterations:
            raise TimeoutError("symbolic failure attractor iteration limit reached")
        layers.append(layer)
        attractor = manager.apply("or", attractor, layer)
        peak_nodes = max(
            peak_nodes,
            manager.reachable_node_count(layer),
            manager.reachable_node_count(attractor),
        )
        layer = manager.apply(
            "and",
            manager.compose(layer, next_functions),
            manager.negate(attractor),
        )
    proof = SymbolicFailureAttractorProof(
        state_variables=variables,
        bad_root=bad_root,
        rank_layer_roots=tuple(layers),
        attractor_root=attractor,
        safe_invariant_root=manager.negate(attractor),
        iterations=max(0, len(layers) - 1),
        peak_reachable_nodes=peak_nodes,
        allocated_nodes=manager.allocated_node_count,
    )
    if not proof.verify(manager, next_functions):  # pragma: no cover
        raise AssertionError("constructed symbolic safety proof failed verification")
    return proof


def compacting_symbolic_failure_attractor(
    manager: ROBDDManager,
    state_variables: Sequence[Hashable],
    next_functions: Mapping[Hashable, int],
    bad_root: int,
    *,
    minimum_compaction_nodes: int = 1_000,
    compaction_ratio: int = 8,
    maximum_iterations: int | None = None,
) -> CompactingSymbolicSafetyResult:
    """Fixed point with exact live-root compaction between iterations."""

    variables = tuple(state_variables)
    if not variables or len(variables) != len(set(variables)):
        raise ValueError("symbolic state variables must be non-empty and unique")
    if set(variables) != set(next_functions):
        raise ValueError("every symbolic state bit needs one next-state function")
    if set(variables) - set(manager.variables):
        raise ValueError("symbolic state variables are outside the ROBDD manager")
    if minimum_compaction_nodes <= 0 or compaction_ratio <= 1:
        raise ValueError("symbolic compaction thresholds are invalid")
    if maximum_iterations is not None and maximum_iterations <= 0:
        raise ValueError("maximum symbolic iterations must be positive or None")

    current_manager = manager
    current_next = dict(next_functions)
    bad = bad_root
    attractor = 0
    layer = bad
    layers: list[int] = []
    peak_reachable = 0
    peak_allocated = current_manager.allocated_node_count
    compactions = 0
    while layer != 0:
        if maximum_iterations is not None and len(layers) >= maximum_iterations:
            raise TimeoutError("symbolic failure attractor iteration limit reached")
        layers.append(layer)
        attractor = current_manager.apply("or", attractor, layer)
        peak_reachable = max(
            peak_reachable,
            current_manager.reachable_node_count(layer),
            current_manager.reachable_node_count(attractor),
        )
        layer = current_manager.apply(
            "and",
            current_manager.compose(layer, current_next),
            current_manager.negate(attractor),
        )
        peak_allocated = max(
            peak_allocated,
            current_manager.allocated_node_count,
        )
        roots = (
            bad,
            attractor,
            layer,
            *layers,
            *(current_next[variable] for variable in variables),
        )
        if current_manager.allocated_node_count < minimum_compaction_nodes:
            continue
        live = current_manager.reachable_union_node_count(roots)
        if current_manager.allocated_node_count <= compaction_ratio * max(1, live):
            continue
        current_manager, compacted = current_manager.compact_roots(roots)
        cursor = 0
        bad, attractor, layer = compacted[cursor:cursor + 3]
        cursor += 3
        layers = list(compacted[cursor:cursor + len(layers)])
        cursor += len(layers)
        current_next = dict(zip(
            variables,
            compacted[cursor:],
            strict=True,
        ))
        compactions += 1

    proof = SymbolicFailureAttractorProof(
        state_variables=variables,
        bad_root=bad,
        rank_layer_roots=tuple(layers),
        attractor_root=attractor,
        safe_invariant_root=current_manager.negate(attractor),
        iterations=max(0, len(layers) - 1),
        peak_reachable_nodes=peak_reachable,
        allocated_nodes=current_manager.allocated_node_count,
    )
    if not proof.verify(current_manager, current_next):  # pragma: no cover
        raise AssertionError("compacted symbolic safety proof failed verification")
    peak_allocated = max(peak_allocated, current_manager.allocated_node_count)
    verification_roots = (
        proof.bad_root,
        *proof.rank_layer_roots,
        proof.attractor_root,
        proof.safe_invariant_root,
        *(current_next[variable] for variable in variables),
    )
    current_manager, compacted = current_manager.compact_roots(verification_roots)
    cursor = 0
    compacted_bad = compacted[cursor]
    cursor += 1
    compacted_layers = compacted[cursor:cursor + len(proof.rank_layer_roots)]
    cursor += len(compacted_layers)
    compacted_attractor, compacted_safe = compacted[cursor:cursor + 2]
    cursor += 2
    current_next = dict(zip(
        variables,
        compacted[cursor:],
        strict=True,
    ))
    proof = SymbolicFailureAttractorProof(
        state_variables=variables,
        bad_root=compacted_bad,
        rank_layer_roots=tuple(compacted_layers),
        attractor_root=compacted_attractor,
        safe_invariant_root=compacted_safe,
        iterations=proof.iterations,
        peak_reachable_nodes=proof.peak_reachable_nodes,
        allocated_nodes=current_manager.allocated_node_count,
    )
    compactions += 1
    return CompactingSymbolicSafetyResult(
        manager=current_manager,
        next_functions=current_next,
        proof=proof,
        compactions=compactions,
        peak_allocated_nodes=peak_allocated,
    )
