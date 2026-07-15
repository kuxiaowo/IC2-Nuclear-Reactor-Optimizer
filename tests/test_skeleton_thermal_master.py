from __future__ import annotations

from ic2_reactor.mathematical_model import Graph, PowerComponent, ReactorProblem
from ic2_reactor.periodic_prefix import PrefixHeatComponent, periodic_prefix_flow_bound
from ic2_reactor.skeleton_thermal_master import IdealSkeletonThermalMaster
from ic2_reactor.thermal_relaxation import HeatFlowComponent


def problem(heat_scale: int) -> tuple[ReactorProblem, dict[str, HeatFlowComponent]]:
    instance = ReactorProblem(
        graph=Graph.from_edges(3, ((0, 1), (1, 2))),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("sink", "exchange"),
        eu_per_pulse=1,
        heat_scale=heat_scale,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
        "exchange": HeatFlowComponent(accepts_heat=True, exchange_side=10),
    }
    return instance, catalogue


def test_ideal_skeleton_master_proves_tier_impossible_for_all_completions() -> None:
    instance, catalogue = problem(heat_scale=3)
    result = IdealSkeletonThermalMaster(instance, catalogue).solve(
        exact_power=1,
        seconds=2,
        workers=1,
    )
    assert result.status == "INFEASIBLE"
    assert result.proven_optimal


def test_ideal_skeleton_master_returns_only_an_optimistic_skeleton_candidate() -> None:
    instance, catalogue = problem(heat_scale=1)
    result = IdealSkeletonThermalMaster(instance, catalogue).solve(
        exact_power=1,
        seconds=2,
        workers=1,
    )
    assert result.proven_optimal
    assert result.layout is not None
    assert result.layout.count("fuel") == 1
    assert set(result.layout) <= {"empty", "fuel"}

    alternative = IdealSkeletonThermalMaster(instance, catalogue).solve(
        exact_power=1,
        excluded_skeletons=(result.layout,),
        seconds=2,
        workers=1,
    )
    assert alternative.proven_optimal
    assert alternative.layout is not None
    assert alternative.layout != result.layout


def test_dynamic_prefix_cut_projects_to_optional_ideal_cooling_slots() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(2, ((0, 1),)),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("weak",),
        eu_per_pulse=1,
        heat_scale=1,
    )
    heat_catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "weak": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    prefix_catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "weak": PrefixHeatComponent(1, True, self_vent=2),
    }
    failed = tuple(
        periodic_prefix_flow_bound(
            problem,
            layout,
            prefix_catalogue,
            base_hull_capacity=10,
        ).cut_template
        for layout in (
            ("weak", "fuel"),
            ("empty", "fuel"),
            ("fuel", "weak"),
            ("fuel", "empty"),
        )
    )
    result = IdealSkeletonThermalMaster(
        problem,
        heat_catalogue,
        prefix_catalogue=prefix_catalogue,
        base_hull_capacity=10,
    ).solve(
        prefix_cuts=failed,
        exact_power=1,
        seconds=2,
        workers=1,
    )
    assert result.status == "INFEASIBLE"
