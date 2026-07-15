from __future__ import annotations

from ic2_reactor.logic_benders import (
    BendersCut,
    BendersDomain,
    LogicBendersProcessor,
    MasterAnswer,
    MasterCandidate,
    MasterStatus,
    SubproblemAnswer,
    SubproblemStatus,
)
from ic2_reactor.proof_coordinator import ProofArtifact, ProofCoordinator, SearchNode


class IntegerMaster:
    """Master proposes the largest integer not removed by a cut."""

    def solve(self, domain, incumbent_lower_bound, time_limit_seconds):
        excluded = {cut.payload for cut in domain.cuts}
        remaining = [value for value in domain.base if value not in excluded]
        if not remaining:
            return MasterAnswer(
                MasterStatus.INFEASIBLE,
                proof=ProofArtifact("finite_exhaustion", "all master values were cut"),
            )
        value = max(remaining)
        residual = max((item for item in remaining if item != value), default=-1)
        no_good = BendersCut(
            f"not_{value}",
            value,
            ProofArtifact("exact_no_good", f"exclude exactly {value}"),
        )
        return MasterAnswer(
            MasterStatus.CANDIDATE,
            MasterCandidate(value, value, no_good, residual),
        )


class EvenSubproblem:
    def check(self, candidate, time_limit_seconds):
        if candidate % 2 == 0:
            return SubproblemAnswer(
                SubproblemStatus.FEASIBLE,
                witness_payload=candidate,
                proof=ProofArtifact("parity_witness", f"{candidate} is even"),
            )
        cut = BendersCut(
            f"odd_{candidate}",
            candidate,
            ProofArtifact("odd_cut", f"{candidate} is proven odd"),
        )
        return SubproblemAnswer(
            SubproblemStatus.INFEASIBLE,
            proof=cut.proof,
            generalized_cut=cut,
        )


def test_generic_logic_benders_proves_optimum_without_domain_knowledge_in_coordinator() -> None:
    domain = BendersDomain(tuple(range(8)))
    processor = LogicBendersProcessor(IntegerMaster(), EvenSubproblem())
    report = ProofCoordinator().run(
        SearchNode("root", 7, domain),
        processor,
        time_limit_seconds=2,
    )
    assert report.proven_global
    assert report.lower_bound == report.upper_bound == 6
    assert report.best_witness.payload == 6
    assert any(record.proof.kind == "logic_benders_partition" for record in report.closed_nodes)


class UnknownTopSubproblem(EvenSubproblem):
    def check(self, candidate, time_limit_seconds):
        if candidate == 3:
            return SubproblemAnswer(SubproblemStatus.UNKNOWN)
        return super().check(candidate, time_limit_seconds)


def test_unknown_candidate_isolated_without_blocking_residual_domain() -> None:
    processor = LogicBendersProcessor(IntegerMaster(), UnknownTopSubproblem())
    report = ProofCoordinator().run(
        SearchNode("root", 3, BendersDomain(tuple(range(4)))),
        processor,
        time_limit_seconds=2,
    )
    assert not report.proven_global
    assert report.lower_bound == 2
    assert report.upper_bound == 3
    assert len(report.open_nodes) == 1
    assert report.open_nodes[0].payload.fixed_candidate is not None
    assert any(
        record.proof.kind == "unknown_singleton_partition"
        for record in report.closed_nodes
    )


class BoundOnlyMaster:
    def solve(self, domain, incumbent_lower_bound, time_limit_seconds):
        return MasterAnswer(
            MasterStatus.UNKNOWN,
            proof=ProofArtifact("dual_bound", "master proves upper bound two"),
            strict_upper_bound=2,
        )


def test_master_timeout_dual_bound_tightens_open_node() -> None:
    report = ProofCoordinator().run(
        SearchNode("root", 3, BendersDomain(tuple(range(4)))),
        LogicBendersProcessor(BoundOnlyMaster(), EvenSubproblem()),
        time_limit_seconds=1,
    )
    assert report.lower_bound == 0
    assert report.upper_bound == 2
    assert len(report.open_nodes) == 1
    assert any(record.proof.kind == "dual_bound" for record in report.closed_nodes)
