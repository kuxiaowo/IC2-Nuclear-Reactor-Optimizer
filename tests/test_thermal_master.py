from __future__ import annotations

from itertools import product

import pytest
from ortools.sat.python import cp_model

from ic2_reactor.mathematical_model import (
    AggregatePattern,
    Graph,
    PowerComponent,
    ReactorProblem,
    evaluate_power_skeleton,
)
from ic2_reactor.periodic_prefix import (
    PrefixHeatComponent,
    periodic_prefix_flow_bound,
)
from ic2_reactor.thermal_master import PowerSkeletonNoGood, ThermalCutMaster
from ic2_reactor.thermal_relaxation import HeatFlowComponent, layout_heat_flow_bound
from ic2_reactor.thermal_relaxation import ThermalCutTemplate


def small_problem() -> tuple[ReactorProblem, dict[str, HeatFlowComponent]]:
    problem = ReactorProblem(
        graph=Graph.from_edges(2, ((0, 1),)),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("sink",),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    return problem, catalogue


def test_linearized_cut_matches_direct_capacity_for_every_small_layout() -> None:
    problem, catalogue = small_problem()
    cut = layout_heat_flow_bound(
        problem,
        ("fuel", "empty"),
        catalogue,
    ).cut_template
    master = ThermalCutMaster(problem, catalogue)
    power_ids = {item.id for item in problem.power_components}

    for layout in product(master.labels, repeat=problem.graph.size):
        skeleton = tuple(label if label in power_ids else "empty" for label in layout)
        metrics = evaluate_power_skeleton(problem, skeleton)
        if metrics.rods != problem.rod_budget:
            continue
        model, variables = master.build(cuts=(cut,))
        for vertex, label in enumerate(layout):
            model.add(variables["one_hot"][vertex][master.code_by_id[label]] == 1)
        solver = cp_model.CpSolver()
        status = solver.solve(model)
        direct = cut.evaluate(problem, layout, catalogue)
        assert (status == cp_model.OPTIMAL) == direct.necessary_condition_satisfied
        if status == cp_model.OPTIMAL:
            assert solver.value(variables["cut_capacities"][0]) == direct.cut_capacity
            assert solver.value(variables["heat"]) == direct.generated_heat


def test_thermal_master_cut_removes_failed_family_and_keeps_repair() -> None:
    problem, catalogue = small_problem()
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    result = ThermalCutMaster(problem, catalogue).solve(
        cuts=(failed.cut_template,),
        seconds=2,
        workers=1,
    )
    assert result.proven_optimal
    assert result.power == 1
    assert result.generated_heat == 2
    assert result.layout is not None
    assert "sink" in result.layout
    assert result.cut_capacities == (2,)


def test_full_label_inventory_limit_is_enforced() -> None:
    base, catalogue = small_problem()
    problem = ReactorProblem(
        graph=base.graph,
        rod_budget=base.rod_budget,
        exact_rods=True,
        power_components=base.power_components,
        cooling_components=(),
        layout_components=base.layout_components,
        component_limits=(("sink", 0),),
        eu_per_pulse=1,
        heat_scale=1,
    )
    cuts = tuple(
        layout_heat_flow_bound(problem, layout, catalogue).cut_template
        for layout in (("fuel", "empty"), ("empty", "fuel"))
    )
    result = ThermalCutMaster(problem, catalogue).solve(
        cuts=cuts,
        seconds=2,
        workers=1,
    )
    assert not result.feasible
    assert result.status == "INFEASIBLE"


def rich_problem() -> tuple[ReactorProblem, dict[str, HeatFlowComponent]]:
    problem = ReactorProblem(
        graph=Graph.from_edges(3, ((0, 1), (1, 2))),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("sink", "side", "draw", "exchange"),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
        "side": HeatFlowComponent(side_vent=1),
        "draw": HeatFlowComponent(accepts_heat=True, self_vent=1, hull_draw=3),
        "exchange": HeatFlowComponent(
            accepts_heat=True,
            exchange_side=2,
            exchange_hull=1,
        ),
    }
    return problem, catalogue


def test_all_cut_arc_families_match_direct_network_formula() -> None:
    problem, catalogue = rich_problem()
    cut_sources = (
        ("fuel", "empty", "empty"),
        ("fuel", "draw", "side"),
        ("exchange", "fuel", "sink"),
        ("draw", "exchange", "fuel"),
    )
    cuts = tuple(
        layout_heat_flow_bound(problem, layout, catalogue).cut_template
        for layout in cut_sources
    )
    master = ThermalCutMaster(problem, catalogue)
    power_ids = {item.id for item in problem.power_components}

    for layout in product(master.labels, repeat=problem.graph.size):
        skeleton = tuple(label if label in power_ids else "empty" for label in layout)
        if evaluate_power_skeleton(problem, skeleton).rods != 1:
            continue
        model, variables = master.build(cuts=cuts, enforce_cuts=False)
        for vertex, label in enumerate(layout):
            model.add(variables["one_hot"][vertex][master.code_by_id[label]] == 1)
        solver = cp_model.CpSolver()
        assert solver.solve(model) == cp_model.OPTIMAL
        assert tuple(
            solver.value(capacity) for capacity in variables["cut_capacities"]
        ) == tuple(
            cut.evaluate(problem, layout, catalogue).cut_capacity for cut in cuts
        )


def test_compact_full_flow_is_equivalent_to_max_flow_on_every_rich_layout() -> None:
    problem, catalogue = rich_problem()
    master = ThermalCutMaster(problem, catalogue)
    power_ids = {item.id for item in problem.power_components}
    for layout in product(master.labels, repeat=problem.graph.size):
        skeleton = tuple(label if label in power_ids else "empty" for label in layout)
        if evaluate_power_skeleton(problem, skeleton).rods != 1:
            continue
        model, variables = master.build(enforce_full_flow=True)
        for vertex, label in enumerate(layout):
            model.add(variables["one_hot"][vertex][master.code_by_id[label]] == 1)
        solver = cp_model.CpSolver()
        status = solver.solve(model)
        expected = layout_heat_flow_bound(
            problem,
            layout,
            catalogue,
        ).necessary_condition_satisfied
        assert (status == cp_model.OPTIMAL) == expected


def test_explicit_side_and_exchange_multicut_model_is_valid() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(2, ((0, 1),)),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("side", "exchange"),
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "side": HeatFlowComponent(side_vent=1),
        "exchange": HeatFlowComponent(accepts_heat=True, exchange_side=2),
    }
    cuts = (
        ThermalCutTemplate((0,), (), False),
        ThermalCutTemplate((1,), (0,), True),
    )
    model, _variables = ThermalCutMaster(problem, catalogue).build(
        cuts=cuts,
        enforce_cuts=False,
    )
    assert model.validate() == ""


def test_ordered_distribution_flow_rejects_unvented_hull_and_keeps_sink() -> None:
    problem, catalogue = small_problem()
    master = ThermalCutMaster(problem, catalogue)
    for layout, expected in (
        (("fuel", "empty"), False),
        (("fuel", "sink"), True),
        (("empty", "fuel"), False),
        (("sink", "fuel"), True),
    ):
        model, variables = master.build(enforce_ordered_distribution_flow=True)
        for vertex, label in enumerate(layout):
            model.add(variables["one_hot"][vertex][master.code_by_id[label]] == 1)
        solver = cp_model.CpSolver()
        assert (solver.solve(model) == cp_model.OPTIMAL) == expected


def test_embedded_periodic_circulation_matches_fixed_layout_network() -> None:
    problem, heat_catalogue = small_problem()
    prefix_catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "sink": PrefixHeatComponent(
            heat_capacity=2,
            accepts_fuel_heat=True,
            self_vent=2,
        ),
    }
    master = ThermalCutMaster(
        problem,
        heat_catalogue,
        prefix_catalogue=prefix_catalogue,
        base_hull_capacity=10,
    )
    power_ids = {item.id for item in problem.power_components}
    for layout in product(master.labels, repeat=problem.graph.size):
        skeleton = tuple(label if label in power_ids else "empty" for label in layout)
        if evaluate_power_skeleton(problem, skeleton).rods != problem.rod_budget:
            continue
        model, variables = master.build(enforce_periodic_prefix_flow=True)
        for vertex, label in enumerate(layout):
            model.add(variables["one_hot"][vertex][master.code_by_id[label]] == 1)
        solver = cp_model.CpSolver()
        embedded_feasible = solver.solve(model) == cp_model.OPTIMAL
        fixed = periodic_prefix_flow_bound(
            problem,
            layout,
            prefix_catalogue,
            base_hull_capacity=10,
        )
        assert embedded_feasible == fixed.feasible


def test_power_tier_can_be_restricted_to_proven_aggregate_patterns() -> None:
    problem, catalogue = small_problem()

    def aggregate(degree: int) -> AggregatePattern:
        return AggregatePattern(
            active_cells=1,
            generated_heat=2,
            slack=0,
            required_relief=0,
            maximum_available_relief=0,
            margin=0,
            fuel_degree_counts=(("fuel", degree, 1),),
        )

    master = ThermalCutMaster(problem, catalogue)
    allowed = master.solve(
        exact_power=1,
        conditional_aggregate_patterns={1: (aggregate(0),)},
        seconds=2,
        workers=1,
    )
    assert allowed.feasible

    wrong_signature = master.solve(
        exact_power=1,
        conditional_aggregate_patterns={1: (aggregate(1),)},
        seconds=2,
        workers=1,
    )
    assert wrong_signature.status == "INFEASIBLE"

    excluded_tier = master.solve(
        exact_power=1,
        conditional_aggregate_patterns={1: ()},
        seconds=2,
        workers=1,
    )
    assert excluded_tier.status == "INFEASIBLE"


def test_unknown_without_incumbent_keeps_explicit_power_upper_bound() -> None:
    problem, catalogue = small_problem()
    result = ThermalCutMaster(problem, catalogue).solve(
        minimum_power=1,
        maximum_power_limit=1,
        seconds=1e-9,
        workers=1,
    )
    assert result.status == "UNKNOWN"
    assert not result.feasible
    assert result.strict_power_upper_bound == 1


def test_generic_weighted_label_limit_is_enforced() -> None:
    problem, catalogue = small_problem()
    result = ThermalCutMaster(problem, catalogue).solve(
        weighted_label_limits=(({"sink": 2}, 0),),
        seconds=2,
        workers=1,
    )
    assert result.proven_optimal
    assert result.layout is not None
    assert "sink" not in result.layout


def test_fixed_power_skeleton_leaves_only_cooling_slots_free() -> None:
    problem, catalogue = small_problem()
    result = ThermalCutMaster(problem, catalogue).solve(
        fixed_power_skeleton=("fuel", "empty"),
        enforce_ordered_distribution_flow=True,
        seconds=2,
        workers=1,
    )
    assert result.proven_optimal
    assert result.layout == ("fuel", "sink")
    assert result.power == 1


def test_fixed_power_skeleton_rejects_invalid_input() -> None:
    problem, catalogue = small_problem()
    master = ThermalCutMaster(problem, catalogue)
    with pytest.raises(ValueError, match="length"):
        master.build(fixed_power_skeleton=("fuel",))
    with pytest.raises(ValueError, match="unknown power"):
        master.build(fixed_power_skeleton=("sink", "empty"))


def test_linearized_periodic_prefix_cut_matches_direct_hofmann_inequality() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(2, ((0, 1),)),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("weak", "strong"),
        eu_per_pulse=1,
        heat_scale=1,
    )
    prefix_catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "weak": PrefixHeatComponent(1, True, self_vent=2),
        "strong": PrefixHeatComponent(2, True, self_vent=2),
    }
    heat_catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "weak": HeatFlowComponent(accepts_heat=True, self_vent=2),
        "strong": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    failed = periodic_prefix_flow_bound(
        problem,
        ("weak", "fuel"),
        prefix_catalogue,
        base_hull_capacity=10,
    )
    master = ThermalCutMaster(
        problem,
        heat_catalogue,
        prefix_catalogue=prefix_catalogue,
        base_hull_capacity=10,
    )
    power_ids = {item.id for item in problem.power_components}
    for layout in product(master.labels, repeat=problem.graph.size):
        skeleton = tuple(label if label in power_ids else "empty" for label in layout)
        if evaluate_power_skeleton(problem, skeleton).rods != 1:
            continue
        direct = failed.cut_template.evaluate(
            problem,
            layout,
            prefix_catalogue,
            base_hull_capacity=10,
        )
        model, variables = master.build(prefix_cuts=(failed.cut_template,))
        for vertex, label in enumerate(layout):
            model.add(variables["one_hot"][vertex][master.code_by_id[label]] == 1)
        solver = cp_model.CpSolver()
        status = solver.solve(model)
        assert (status == cp_model.OPTIMAL) == direct.necessary_condition_satisfied
        if status == cp_model.OPTIMAL:
            assert solver.value(variables["prefix_cut_violations"][0]) == (
                direct.lower_bound_into_source_side
                - direct.upper_bound_out_of_source_side
            )


def test_periodic_prefix_multicuts_cover_every_ordered_arc_family() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(3, ((0, 1), (1, 2))),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("sink", "side", "draw", "exchange", "plate"),
        eu_per_pulse=1,
        heat_scale=3,
    )
    prefix_catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "sink": PrefixHeatComponent(2, True, self_vent=2),
        "side": PrefixHeatComponent(side_vent=1),
        "draw": PrefixHeatComponent(3, True, self_vent=1, hull_draw=2),
        "exchange": PrefixHeatComponent(
            4,
            True,
            exchange_side=2,
            exchange_hull=1,
        ),
        "plate": PrefixHeatComponent(hull_capacity_bonus=5),
    }
    heat_catalogue = {
        label: HeatFlowComponent(
            accepts_heat=spec.heat_capacity > 0,
            self_vent=spec.self_vent,
            side_vent=spec.side_vent,
            hull_draw=spec.hull_draw,
            exchange_side=spec.exchange_side,
            exchange_hull=spec.exchange_hull,
        )
        for label, spec in prefix_catalogue.items()
    }
    cut_layouts = (
        ("fuel", "empty", "empty"),
        ("fuel", "draw", "side"),
        ("exchange", "fuel", "sink"),
        ("draw", "exchange", "fuel"),
    )
    cuts = tuple(
        periodic_prefix_flow_bound(
            problem,
            layout,
            prefix_catalogue,
            base_hull_capacity=10,
        ).cut_template
        for layout in cut_layouts
    )
    master = ThermalCutMaster(
        problem,
        heat_catalogue,
        prefix_catalogue=prefix_catalogue,
        base_hull_capacity=10,
    )
    power_ids = {item.id for item in problem.power_components}
    for layout in product(master.labels, repeat=problem.graph.size):
        skeleton = tuple(label if label in power_ids else "empty" for label in layout)
        if evaluate_power_skeleton(problem, skeleton).rods != 1:
            continue
        expected = all(
            cut.evaluate(
                problem,
                layout,
                prefix_catalogue,
                base_hull_capacity=10,
            ).necessary_condition_satisfied
            for cut in cuts
        )
        model, variables = master.build(prefix_cuts=cuts)
        for vertex, label in enumerate(layout):
            model.add(variables["one_hot"][vertex][master.code_by_id[label]] == 1)
        solver = cp_model.CpSolver()
        assert (solver.solve(model) == cp_model.OPTIMAL) == expected


def test_fixed_skeleton_infeasibility_core_becomes_a_partial_family_no_good() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(3, ((0, 1), (1, 2))),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
            PowerComponent("reflector", 0, 0, True),
        ),
        cooling_components=(),
        layout_components=("sink",),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "reflector": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    master = ThermalCutMaster(problem, catalogue)
    failed = master.solve(
        fixed_power_skeleton=("fuel", "reflector", "empty"),
        extract_fixed_skeleton_core=True,
        enforce_ordered_distribution_flow=True,
        seconds=2,
        workers=1,
    )
    assert failed.status == "INFEASIBLE"
    assert failed.fixed_skeleton_core
    assert len(failed.fixed_skeleton_core) <= problem.graph.size

    residual = master.solve(
        excluded_power_cores=(PowerSkeletonNoGood(failed.fixed_skeleton_core),),
        enforce_ordered_distribution_flow=True,
        seconds=2,
        workers=1,
    )
    assert residual.proven_optimal
    assert residual.feasible
    assert residual.layout is not None
