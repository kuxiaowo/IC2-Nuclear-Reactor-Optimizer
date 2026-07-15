from __future__ import annotations

from ic2_reactor.mathematical_model import (
    AggregatePattern,
    Graph,
    PowerComponent,
    ReactorProblem,
)
from ic2_reactor.structural_master import AggregateDegreeEmbeddingMaster


def path_problem() -> ReactorProblem:
    return ReactorProblem(
        graph=Graph.from_edges(3, ((0, 1), (1, 2))),
        rod_budget=3,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
            PowerComponent("reflector", 0, 0, True),
        ),
        cooling_components=(),
    )


def pattern(degrees: tuple[int, ...], *, active_cells: int | None = None) -> AggregatePattern:
    counts = tuple(
        ("fuel", degree, degrees.count(degree))
        for degree in sorted(set(degrees))
    )
    return AggregatePattern(
        active_cells=len(degrees) if active_cells is None else active_cells,
        generated_heat=0,
        slack=0,
        required_relief=0,
        maximum_available_relief=0,
        margin=0,
        fuel_degree_counts=counts,
    )


def test_exact_geometry_rejects_bipartite_degree_sequence_not_embeddable_in_path() -> None:
    # (2, 2, 2) needs a triangle; the path has no such induced subgraph.
    result = AggregateDegreeEmbeddingMaster(path_problem()).solve(
        pattern((2, 2, 2)),
        seconds=2,
        workers=1,
    )
    assert result.proven
    assert result.possible is False
    assert result.status == "INFEASIBLE"


def test_exact_geometry_constructs_a_valid_embedding() -> None:
    result = AggregateDegreeEmbeddingMaster(path_problem()).solve(
        pattern((2, 1, 1)),
        seconds=2,
        workers=1,
    )
    assert result.proven
    assert result.possible is True
    assert result.skeleton is not None
    assert result.skeleton.count("fuel") == 3


def test_unknown_active_vertices_are_an_optimistic_relaxation() -> None:
    # The requested degree-two fuel can use two unspecified active neighbours.
    result = AggregateDegreeEmbeddingMaster(path_problem()).solve(
        pattern((2,), active_cells=3),
        seconds=2,
        workers=1,
    )
    assert result.possible is True
    assert result.skeleton is not None
    assert result.skeleton.count("active_unknown") == 2


def test_geometry_aware_relief_counts_only_effective_side_edges() -> None:
    candidate = pattern((0,), active_cells=1)
    candidate = AggregatePattern(
        active_cells=candidate.active_cells,
        generated_heat=24,
        slack=16,
        required_relief=5,
        maximum_available_relief=16,
        margin=-11,
        fuel_degree_counts=candidate.fuel_degree_counts,
    )
    result = AggregateDegreeEmbeddingMaster(path_problem()).maximize_optimistic_relief(
        candidate,
        exchanger_relief=1,
        seconds=2,
        workers=1,
    )
    assert result.proven_optimal
    assert result.maximum_relief == 4
    assert result.effective_side_edges == 1
    assert result.baseline_capacity_loss == 16
    assert result.excluded is True


def test_geometry_aware_relief_keeps_optimistic_exchanger_bound() -> None:
    candidate = pattern((0,), active_cells=1)
    candidate = AggregatePattern(
        active_cells=candidate.active_cells,
        generated_heat=0,
        slack=40,
        required_relief=145,
        maximum_available_relief=160,
        margin=-15,
        fuel_degree_counts=candidate.fuel_degree_counts,
    )
    result = AggregateDegreeEmbeddingMaster(path_problem()).maximize_optimistic_relief(
        candidate,
        exchanger_relief=72,
        seconds=2,
        workers=1,
    )
    assert result.proven_optimal
    assert result.maximum_relief == 144
    assert result.exchanger_cells == 2
    assert result.excluded is True
