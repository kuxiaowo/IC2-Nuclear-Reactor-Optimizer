from __future__ import annotations

from itertools import product

from ic2_reactor.mathematical_model import (
    AggregatePattern,
    Graph,
    PowerComponent,
    ReactorProblem,
)
from ic2_reactor.structural_relief_dp import RectangularStructuralReliefDP


def small_problem() -> ReactorProblem:
    return ReactorProblem(
        graph=Graph.rectangular(2, 2),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
    )


def target(required_relief: int) -> AggregatePattern:
    return AggregatePattern(
        active_cells=1,
        generated_heat=2,
        slack=20,
        required_relief=required_relief,
        maximum_available_relief=0,
        margin=0,
        fuel_degree_counts=(("fuel", 0, 1),),
    )


def brute_maximum(pattern: AggregatePattern) -> int:
    problem = small_problem()
    best = -1
    for labels in product(("ordinary", "side", "fuel"), repeat=problem.graph.size):
        if labels.count("fuel") != 1:
            continue
        fuel = labels.index("fuel")
        if sum(labels[n] == "fuel" for n in problem.graph.neighbours[fuel]) != 0:
            continue
        side_cells = labels.count("side")
        effective_edges = sum(
            {labels[first], labels[second]} == {"ordinary", "side"}
            for first, second in problem.graph.edges
        )
        side_loss = 20 * side_cells - 4 * effective_edges
        if side_loss > pattern.slack:
            continue
        exchangers = min(
            problem.graph.size - pattern.active_cells - side_cells,
            (pattern.slack - side_loss) // 20,
        )
        best = max(best, 4 * effective_edges + 7 * exchangers)
    return best


def test_structural_relief_dp_matches_complete_small_enumeration() -> None:
    pattern = target(required_relief=9)
    result = RectangularStructuralReliefDP(small_problem()).solve(
        pattern,
        exchanger_relief=7,
    )
    assert result.proven
    assert result.maximum_relief == brute_maximum(pattern) == 8
    assert result.best_side_vent_cells == 1
    assert result.best_effective_side_edges == 2
    assert result.best_exchanger_cells == 0
    assert result.excluded is True


def test_relief_at_the_exact_upper_bound_is_not_excluded() -> None:
    result = RectangularStructuralReliefDP(small_problem()).solve(
        target(required_relief=8),
        exchanger_relief=7,
    )
    assert result.maximum_relief == 8
    assert result.excluded is False
