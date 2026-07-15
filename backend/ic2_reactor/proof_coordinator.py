"""Protocol-driven anytime branch-and-proof coordinator.

The coordinator is domain agnostic.  It knows only integer objective bounds,
complete partitions and proof artifacts.  Reactor-specific CP-SAT, frontier
DP, Benders cuts and cycle checks are supplied by a ``NodeProcessor`` adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import heapq
from time import perf_counter
from typing import Generic, Protocol, TypeVar


PayloadT = TypeVar("PayloadT")
WitnessPayloadT = TypeVar("WitnessPayloadT")


@dataclass(frozen=True, slots=True)
class ProofArtifact:
    kind: str
    statement: str
    data: tuple[tuple[str, int | float | str | bool | None], ...] = ()


@dataclass(frozen=True, slots=True)
class SearchNode(Generic[PayloadT]):
    node_id: str
    strict_upper_bound: int
    payload: PayloadT

    def __post_init__(self) -> None:
        if self.strict_upper_bound < 0:
            raise ValueError("strict_upper_bound must be non-negative")


@dataclass(frozen=True, slots=True)
class CertifiedWitness(Generic[WitnessPayloadT]):
    objective_value: int
    payload: WitnessPayloadT
    proof: ProofArtifact


class NodeDisposition(StrEnum):
    INFEASIBLE = "infeasible"
    EXHAUSTED = "exhausted"
    BRANCHED = "branched"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NodeOutcome(Generic[PayloadT, WitnessPayloadT]):
    disposition: NodeDisposition
    children: tuple[SearchNode[PayloadT], ...] = ()
    witness: CertifiedWitness[WitnessPayloadT] | None = None
    proof: ProofArtifact | None = None


class NodeProcessor(Protocol[PayloadT, WitnessPayloadT]):
    def process(
        self,
        node: SearchNode[PayloadT],
        incumbent_lower_bound: int,
        time_limit_seconds: float,
    ) -> NodeOutcome[PayloadT, WitnessPayloadT]: ...


@dataclass(frozen=True, slots=True)
class ClosedNodeRecord:
    node_id: str
    upper_bound: int
    reason: str
    proof: ProofArtifact


@dataclass(frozen=True, slots=True)
class ProofSearchReport(Generic[PayloadT, WitnessPayloadT]):
    lower_bound: int
    upper_bound: int
    proven_global: bool
    best_witness: CertifiedWitness[WitnessPayloadT] | None
    open_nodes: tuple[SearchNode[PayloadT], ...]
    closed_nodes: tuple[ClosedNodeRecord, ...]
    processed_nodes: int
    elapsed_seconds: float
    stop_reason: str


class ProofCoordinator(Generic[PayloadT, WitnessPayloadT]):
    """Maintain ``lower <= optimum <= max(open upper)`` at every return."""

    def run(
        self,
        root: SearchNode[PayloadT],
        processor: NodeProcessor[PayloadT, WitnessPayloadT],
        *,
        time_limit_seconds: float,
        known_witness: CertifiedWitness[WitnessPayloadT] | None = None,
    ) -> ProofSearchReport[PayloadT, WitnessPayloadT]:
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        started = perf_counter()
        deadline = started + time_limit_seconds
        lower = known_witness.objective_value if known_witness is not None else 0
        best_witness = known_witness
        heap: list[tuple[int, int, SearchNode[PayloadT]]] = [
            (-root.strict_upper_bound, 0, root)
        ]
        serial = 0
        processed = 0
        deferred: list[SearchNode[PayloadT]] = []
        closed: list[ClosedNodeRecord] = []
        stop_reason = "exhausted"

        def close_by_bound(node: SearchNode[PayloadT]) -> None:
            proof = ProofArtifact(
                kind="objective_bound",
                statement=(
                    f"node upper bound {node.strict_upper_bound} does not exceed "
                    f"certified incumbent {lower}"
                ),
                data=(("upper", node.strict_upper_bound), ("lower", lower)),
            )
            closed.append(ClosedNodeRecord(
                node_id=node.node_id,
                upper_bound=node.strict_upper_bound,
                reason="bound",
                proof=proof,
            ))

        while heap:
            if perf_counter() >= deadline:
                stop_reason = "time_limit"
                break
            _negative_upper, _serial, node = heapq.heappop(heap)
            if node.strict_upper_bound <= lower:
                close_by_bound(node)
                continue
            remaining = deadline - perf_counter()
            outcome = processor.process(node, lower, remaining)
            processed += 1

            if outcome.witness is not None:
                witness = outcome.witness
                if not 0 <= witness.objective_value <= node.strict_upper_bound:
                    raise ValueError("witness objective lies outside its node bound")
                if witness.objective_value > lower:
                    lower = witness.objective_value
                    best_witness = witness

            if outcome.disposition == NodeDisposition.UNKNOWN:
                if outcome.children:
                    raise ValueError("UNKNOWN may not silently discard or replace a node")
                deferred.append(node)
                continue

            if outcome.disposition in {
                NodeDisposition.INFEASIBLE,
                NodeDisposition.EXHAUSTED,
            }:
                if outcome.proof is None:
                    raise ValueError("closing a node requires a proof artifact")
                if outcome.children:
                    raise ValueError("a closed node may not also return children")
                closed.append(ClosedNodeRecord(
                    node_id=node.node_id,
                    upper_bound=node.strict_upper_bound,
                    reason=outcome.disposition.value,
                    proof=outcome.proof,
                ))
                continue

            if outcome.disposition != NodeDisposition.BRANCHED:
                raise ValueError(f"unsupported node disposition: {outcome.disposition}")
            if outcome.proof is None or not outcome.children:
                raise ValueError("a complete branch partition needs children and a proof")
            child_ids = [child.node_id for child in outcome.children]
            if len(child_ids) != len(set(child_ids)):
                raise ValueError("branch child ids must be unique")
            if any(child.strict_upper_bound > node.strict_upper_bound for child in outcome.children):
                raise ValueError("child upper bound may not exceed parent upper bound")
            closed.append(ClosedNodeRecord(
                node_id=node.node_id,
                upper_bound=node.strict_upper_bound,
                reason="partitioned",
                proof=outcome.proof,
            ))
            for child in outcome.children:
                if child.strict_upper_bound <= lower:
                    close_by_bound(child)
                else:
                    serial += 1
                    heapq.heappush(heap, (-child.strict_upper_bound, serial, child))

        remaining_nodes = [entry[2] for entry in heap]
        remaining_nodes.extend(deferred)
        still_open: list[SearchNode[PayloadT]] = []
        for node in remaining_nodes:
            if node.strict_upper_bound <= lower:
                close_by_bound(node)
            else:
                still_open.append(node)
        still_open.sort(key=lambda item: (-item.strict_upper_bound, item.node_id))
        upper = max((node.strict_upper_bound for node in still_open), default=lower)
        proven = not still_open and best_witness is not None
        if proven:
            stop_reason = "global_optimum"
        elif stop_reason == "exhausted" and deferred:
            stop_reason = "unknown_nodes"
        return ProofSearchReport(
            lower_bound=lower,
            upper_bound=upper,
            proven_global=proven,
            best_witness=best_witness,
            open_nodes=tuple(still_open),
            closed_nodes=tuple(closed),
            processed_nodes=processed,
            elapsed_seconds=perf_counter() - started,
            stop_reason=stop_reason,
        )
