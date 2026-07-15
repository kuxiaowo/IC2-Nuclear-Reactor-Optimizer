from __future__ import annotations

from dataclasses import dataclass

from ic2_reactor.proof_coordinator import (
    CertifiedWitness,
    NodeDisposition,
    NodeOutcome,
    ProofArtifact,
    ProofCoordinator,
    SearchNode,
)


@dataclass(frozen=True)
class EvenIntegerProcessor:
    """Toy exact problem: maximise an even integer in an interval."""

    def process(self, node, incumbent_lower_bound, time_limit_seconds):
        low, high = node.payload
        if low == high:
            if low % 2:
                return NodeOutcome(
                    NodeDisposition.INFEASIBLE,
                    proof=ProofArtifact("parity", f"{low} is odd"),
                )
            witness = CertifiedWitness(
                low,
                low,
                ProofArtifact("direct_check", f"{low} is feasible and even"),
            )
            return NodeOutcome(
                NodeDisposition.EXHAUSTED,
                witness=witness,
                proof=ProofArtifact("singleton", "singleton domain exhausted"),
            )
        middle = (low + high) // 2
        children = (
            SearchNode(f"{low}:{middle}", middle, (low, middle)),
            SearchNode(f"{middle + 1}:{high}", high, (middle + 1, high)),
        )
        return NodeOutcome(
            NodeDisposition.BRANCHED,
            children=children,
            proof=ProofArtifact("interval_partition", "two children exactly partition parent"),
        )


def test_generic_coordinator_proves_global_optimum_from_complete_partitions() -> None:
    report = ProofCoordinator().run(
        SearchNode("0:7", 7, (0, 7)),
        EvenIntegerProcessor(),
        time_limit_seconds=2,
    )
    assert report.proven_global
    assert report.lower_bound == report.upper_bound == 6
    assert report.best_witness.payload == 6
    assert not report.open_nodes


class UnknownProcessor:
    def process(self, node, incumbent_lower_bound, time_limit_seconds):
        return NodeOutcome(NodeDisposition.UNKNOWN)


def test_unknown_is_retained_as_open_and_never_relabelled_infeasible() -> None:
    known = CertifiedWitness(3, "known", ProofArtifact("cycle", "reachable cycle"))
    report = ProofCoordinator().run(
        SearchNode("root", 10, object()),
        UnknownProcessor(),
        time_limit_seconds=1,
        known_witness=known,
    )
    assert not report.proven_global
    assert report.lower_bound == 3
    assert report.upper_bound == 10
    assert report.stop_reason == "unknown_nodes"
    assert [node.node_id for node in report.open_nodes] == ["root"]


class InvalidCloser:
    def process(self, node, incumbent_lower_bound, time_limit_seconds):
        return NodeOutcome(NodeDisposition.INFEASIBLE)


def test_coordinator_rejects_unproved_node_closure() -> None:
    try:
        ProofCoordinator().run(
            SearchNode("root", 1, None),
            InvalidCloser(),
            time_limit_seconds=1,
        )
    except ValueError as error:
        assert "requires a proof" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unproved closure was accepted")


class UpperBoundFirstProcessor:
    def __init__(self) -> None:
        self.visited = []

    def process(self, node, incumbent_lower_bound, time_limit_seconds):
        self.visited.append(node.node_id)
        if node.node_id == "root":
            return NodeOutcome(
                NodeDisposition.BRANCHED,
                children=(
                    SearchNode("low", 5, "low"),
                    SearchNode("high", 10, "high"),
                ),
                proof=ProofArtifact("partition", "high and low partition root"),
            )
        if node.node_id == "high":
            return NodeOutcome(
                NodeDisposition.UNKNOWN,
                witness=CertifiedWitness(
                    8,
                    "safe-eight",
                    ProofArtifact("cycle", "reachable safe cycle at eight"),
                ),
            )
        raise AssertionError("the lower-bound child should have been pruned")


def test_anytime_coordinator_visits_highest_bound_and_prunes_after_witness() -> None:
    processor = UpperBoundFirstProcessor()
    report = ProofCoordinator().run(
        SearchNode("root", 10, "root"),
        processor,
        time_limit_seconds=1,
    )
    assert processor.visited == ["root", "high"]
    assert report.lower_bound == 8
    assert report.upper_bound == 10
    assert [node.node_id for node in report.open_nodes] == ["high"]
    assert any(
        record.node_id == "low" and record.reason == "bound"
        for record in report.closed_nodes
    )
