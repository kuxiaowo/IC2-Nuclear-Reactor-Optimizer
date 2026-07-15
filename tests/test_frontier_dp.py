from __future__ import annotations

from itertools import product

from ic2_reactor.frontier_dp import RectangularFrontierPowerDP
from ic2_reactor.mathematical_model import (
    Graph,
    PowerHeatMaster,
    PowerComponent,
    ReactorProblem,
    evaluate_power_skeleton,
)


POWER_TYPES = (
    PowerComponent("empty", 0, 0, False),
    PowerComponent("fuel", 1, 1, True),
    PowerComponent("reflector", 0, 0, True),
)


def problem(rows: int, columns: int, rods: int, **kwargs) -> ReactorProblem:
    return ReactorProblem(
        graph=Graph.rectangular(rows, columns),
        rod_budget=rods,
        exact_rods=True,
        power_components=POWER_TYPES,
        cooling_components=(),
        **kwargs,
    )


def brute_force(instance: ReactorProblem) -> tuple[int, tuple[str, ...]]:
    labels = tuple(item.id for item in instance.power_components)
    best = (-1, ())
    for skeleton in product(labels, repeat=instance.graph.size):
        metrics = evaluate_power_skeleton(instance, skeleton)
        if metrics.rods == instance.rod_budget and metrics.power > best[0]:
            best = (metrics.power, skeleton)
    return best


def test_frontier_dp_matches_brute_force_when_rows_are_short_side() -> None:
    instance = problem(2, 3, 2)
    expected, _ = brute_force(instance)
    result = RectangularFrontierPowerDP(instance).solve()
    assert result.proven_optimal
    assert result.maximum_power == expected
    assert result.frontier_width == 2
    assert evaluate_power_skeleton(instance, result.skeleton).power == expected


def test_frontier_dp_rotates_scan_when_columns_are_short_side() -> None:
    instance = problem(3, 2, 2)
    expected, _ = brute_force(instance)
    result = RectangularFrontierPowerDP(instance).solve()
    assert result.maximum_power == expected
    assert result.frontier_width == 2
    assert result.scan_length == 3


def test_frontier_dp_honours_generic_inventory_and_fixed_labels() -> None:
    instance = problem(
        2,
        3,
        2,
        component_limits=(("reflector", 1),),
    )
    solver = RectangularFrontierPowerDP(instance, fixed_labels={0: "fuel"})
    result = solver.solve()
    assert result.proven_optimal
    assert result.skeleton[0] == "fuel"
    assert result.skeleton.count("reflector") <= 1
    assert result.skeleton.count("fuel") == 2
    assert solver.complexity_signature()["frontier_width"] == 2


def test_frontier_dp_reports_infeasible_instead_of_inventing_layout() -> None:
    instance = problem(
        2,
        2,
        3,
        component_limits=(("fuel", 2),),
    )
    result = RectangularFrontierPowerDP(instance).solve()
    assert result.proven_optimal
    assert not result.feasible
    assert result.reason == "infeasible"


def test_frontier_dp_cross_checks_cp_sat_across_sizes_and_catalogue() -> None:
    catalogue = (
        PowerComponent("empty", 0, 0, False),
        PowerComponent("single", 1, 1, True),
        PowerComponent("dual", 2, 2, True),
        PowerComponent("mirror", 0, 0, True),
    )
    for rows, columns, rods in ((2, 4, 3), (4, 2, 3), (3, 3, 4)):
        instance = ReactorProblem(
            graph=Graph.rectangular(rows, columns),
            rod_budget=rods,
            exact_rods=True,
            power_components=catalogue,
            cooling_components=(),
        )
        dynamic_program = RectangularFrontierPowerDP(instance).solve()
        cp_sat = PowerHeatMaster(instance).solve(
            seconds=5,
            workers=1,
            use_cooling_envelope=False,
        )
        assert dynamic_program.proven_optimal
        assert cp_sat.proven_optimal
        assert dynamic_program.maximum_power == cp_sat.power


def test_ranked_frontier_enumerator_returns_true_k_best_power_order() -> None:
    instance = problem(2, 3, 2)
    labels = tuple(item.id for item in instance.power_components)
    brute = sorted(
        (
            evaluate_power_skeleton(instance, skeleton).power,
            skeleton,
        )
        for skeleton in product(labels, repeat=instance.graph.size)
        if evaluate_power_skeleton(instance, skeleton).rods == instance.rod_budget
    )
    brute.reverse()
    solver = RectangularFrontierPowerDP(instance)
    solver.solve()
    ranked = list(solver.ranked_skeletons(limit=20))
    assert [item.power for item in ranked] == [item[0] for item in brute[:20]]
    assert len({item.skeleton for item in ranked}) == len(ranked)
