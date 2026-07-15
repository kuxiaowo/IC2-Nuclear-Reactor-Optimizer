"""Domain-independent logic-based Benders node processor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from .proof_coordinator import (
    CertifiedWitness,
    NodeDisposition,
    NodeOutcome,
    ProofArtifact,
    SearchNode,
)


BaseT = TypeVar("BaseT")
CutT = TypeVar("CutT")
CandidateT = TypeVar("CandidateT")
WitnessT = TypeVar("WitnessT")


@dataclass(frozen=True, slots=True)
class BendersCut(Generic[CutT]):
    cut_id: str
    payload: CutT
    proof: ProofArtifact


@dataclass(frozen=True, slots=True)
class BendersDomain(Generic[BaseT, CutT]):
    base: BaseT
    cuts: tuple[BendersCut[CutT], ...] = ()
    iteration: int = 0
    # Set only by the coordinator after a subproblem returns UNKNOWN.  The
    # exact candidate singleton stays open while the no-good residual can
    # continue independently.
    fixed_candidate: object | None = None


@dataclass(frozen=True, slots=True)
class MasterCandidate(Generic[CandidateT, CutT]):
    objective_value: int
    payload: CandidateT
    exact_no_good: BendersCut[CutT]
    residual_upper_bound: int


class MasterStatus(StrEnum):
    CANDIDATE = "candidate"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MasterAnswer(Generic[CandidateT, CutT]):
    status: MasterStatus
    candidate: MasterCandidate[CandidateT, CutT] | None = None
    proof: ProofArtifact | None = None
    strict_upper_bound: int | None = None


class BendersMaster(Protocol[BaseT, CutT, CandidateT]):
    def solve(
        self,
        domain: BendersDomain[BaseT, CutT],
        incumbent_lower_bound: int,
        time_limit_seconds: float,
    ) -> MasterAnswer[CandidateT, CutT]: ...


class SubproblemStatus(StrEnum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SubproblemAnswer(Generic[CutT, WitnessT]):
    status: SubproblemStatus
    witness_payload: WitnessT | None = None
    proof: ProofArtifact | None = None
    generalized_cut: BendersCut[CutT] | None = None


class BendersSubproblem(Protocol[CandidateT, CutT, WitnessT]):
    def check(
        self,
        candidate: CandidateT,
        time_limit_seconds: float,
    ) -> SubproblemAnswer[CutT, WitnessT]: ...


class LogicBendersProcessor(Generic[BaseT, CutT, CandidateT, WitnessT]):
    """Convert master/subproblem answers into certified coordinator outcomes."""

    def __init__(
        self,
        master: BendersMaster[BaseT, CutT, CandidateT],
        subproblem: BendersSubproblem[CandidateT, CutT, WitnessT],
        *,
        master_fraction: float = 0.5,
    ) -> None:
        if not 0 < master_fraction < 1:
            raise ValueError("master_fraction must lie strictly between zero and one")
        self.master = master
        self.subproblem = subproblem
        self.master_fraction = master_fraction

    def process(
        self,
        node: SearchNode[BendersDomain[BaseT, CutT]],
        incumbent_lower_bound: int,
        time_limit_seconds: float,
    ) -> NodeOutcome[BendersDomain[BaseT, CutT], WitnessT]:
        fixed = node.payload.fixed_candidate
        if fixed is None:
            master_budget = max(1e-6, time_limit_seconds * self.master_fraction)
            answer = self.master.solve(node.payload, incumbent_lower_bound, master_budget)
            if answer.status == MasterStatus.UNKNOWN:
                if (
                    answer.strict_upper_bound is not None
                    and answer.strict_upper_bound < node.strict_upper_bound
                ):
                    child = SearchNode(
                        node_id=f"{node.node_id}/t{node.payload.iteration + 1}",
                        strict_upper_bound=max(0, answer.strict_upper_bound),
                        payload=BendersDomain(
                            base=node.payload.base,
                            cuts=node.payload.cuts,
                            iteration=node.payload.iteration + 1,
                        ),
                    )
                    return NodeOutcome(
                        NodeDisposition.BRANCHED,
                        children=(child,),
                        proof=answer.proof or ProofArtifact(
                            kind="master_bound_tightening",
                            statement="the exact master returned a stricter dual bound",
                        ),
                    )
                return NodeOutcome(NodeDisposition.UNKNOWN)
            if answer.status == MasterStatus.INFEASIBLE:
                if answer.proof is None:
                    raise ValueError("master INFEASIBLE requires a proof")
                return NodeOutcome(NodeDisposition.INFEASIBLE, proof=answer.proof)
            candidate = answer.candidate
            if answer.status != MasterStatus.CANDIDATE or candidate is None:
                raise ValueError("master CANDIDATE answer is missing its candidate")
        else:
            if not isinstance(fixed, MasterCandidate):
                raise TypeError("fixed_candidate must be a MasterCandidate")
            candidate = fixed
            master_budget = 0.0
        if candidate.objective_value > node.strict_upper_bound:
            raise ValueError("master candidate exceeds node upper bound")
        sub_budget = max(1e-6, time_limit_seconds - master_budget)
        sub = self.subproblem.check(candidate.payload, sub_budget)
        if sub.status == SubproblemStatus.UNKNOWN:
            if fixed is not None:
                return NodeOutcome(NodeDisposition.UNKNOWN)
            next_iteration = node.payload.iteration + 1
            singleton = SearchNode(
                node_id=f"{node.node_id}/u{next_iteration}",
                strict_upper_bound=candidate.objective_value,
                payload=BendersDomain(
                    base=node.payload.base,
                    cuts=node.payload.cuts,
                    iteration=next_iteration,
                    fixed_candidate=candidate,
                ),
            )
            children = [singleton]
            residual_upper = min(
                node.strict_upper_bound,
                candidate.residual_upper_bound,
            )
            if residual_upper >= 0:
                children.append(SearchNode(
                    node_id=f"{node.node_id}/r{next_iteration}",
                    strict_upper_bound=residual_upper,
                    payload=BendersDomain(
                        base=node.payload.base,
                        cuts=(*node.payload.cuts, candidate.exact_no_good),
                        iteration=next_iteration,
                    ),
                ))
            return NodeOutcome(
                NodeDisposition.BRANCHED,
                children=tuple(children),
                proof=ProofArtifact(
                    kind="unknown_singleton_partition",
                    statement=(
                        "the exact no-good partitions the domain into the "
                        "unresolved candidate singleton and its complete residual"
                    ),
                    data=(("iteration", node.payload.iteration),),
                ),
            )

        witness: CertifiedWitness[WitnessT] | None = None
        if sub.status == SubproblemStatus.FEASIBLE:
            if sub.proof is None or sub.witness_payload is None:
                raise ValueError("feasible subproblem requires witness payload and proof")
            witness = CertifiedWitness(
                objective_value=candidate.objective_value,
                payload=sub.witness_payload,
                proof=sub.proof,
            )
            cut = candidate.exact_no_good
        elif sub.status == SubproblemStatus.INFEASIBLE:
            if sub.generalized_cut is None or sub.proof is None:
                raise ValueError("infeasible subproblem requires a proven generalized cut")
            cut = sub.generalized_cut
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported subproblem status: {sub.status}")

        if fixed is not None:
            return NodeOutcome(
                (
                    NodeDisposition.EXHAUSTED
                    if sub.status == SubproblemStatus.FEASIBLE
                    else NodeDisposition.INFEASIBLE
                ),
                witness=witness,
                proof=(
                    ProofArtifact(
                        kind="fixed_candidate_resolved",
                        statement="the previously unknown singleton is now certified",
                    )
                    if sub.status == SubproblemStatus.FEASIBLE
                    else sub.proof
                ),
            )

        residual_upper = min(node.strict_upper_bound, candidate.residual_upper_bound)
        partition_proof = ProofArtifact(
            kind="logic_benders_partition",
            statement=(
                "the certified cut removes only the checked feasible singleton "
                "or a subproblem-proven infeasible region; the child is the exact residual"
            ),
            data=(
                ("cut_id", cut.cut_id),
                ("iteration", node.payload.iteration),
                ("cut_proof_kind", cut.proof.kind),
                ("cut_proof_statement", cut.proof.statement),
                ("cut_payload", repr(cut.payload)),
                *tuple(
                    (f"cut_proof_{key}", value)
                    for key, value in cut.proof.data
                ),
            ),
        )
        if residual_upper < 0:
            return NodeOutcome(
                NodeDisposition.EXHAUSTED,
                witness=witness,
                proof=partition_proof,
            )
        child_domain = BendersDomain(
            base=node.payload.base,
            cuts=(*node.payload.cuts, cut),
            iteration=node.payload.iteration + 1,
        )
        child = SearchNode(
            node_id=f"{node.node_id}/b{child_domain.iteration}",
            strict_upper_bound=residual_upper,
            payload=child_domain,
        )
        return NodeOutcome(
            NodeDisposition.BRANCHED,
            children=(child,),
            witness=witness,
            proof=partition_proof,
        )
