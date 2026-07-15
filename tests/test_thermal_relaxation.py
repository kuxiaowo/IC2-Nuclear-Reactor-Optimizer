from __future__ import annotations

from itertools import product

import pytest

from ic2_reactor.mathematical_model import Graph, PowerComponent, ReactorProblem
from ic2_reactor.thermal_cut_automaton import compile_thermal_cut
from ic2_reactor.thermal_relaxation import (
    HeatFlowComponent,
    ThermalCutTemplate,
    componentwise_cooling_dominator,
    evaluate_skeleton_thermal_cut,
    layout_heat_flow_bound,
    skeleton_heat_flow_bound,
)


def test_heat_flow_relaxation_uses_supplied_generic_catalogue() -> None:
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
    bound = layout_heat_flow_bound(problem, ("fuel", "sink"), catalogue)
    # The sink accepts heat, but is not a pulse-accepting power component.
    assert bound.generated_heat == 2
    assert bound.maximum_removable_heat == 2
    assert bound.necessary_condition_satisfied
    cut = bound.cut_template.evaluate(problem, ("fuel", "sink"), catalogue)
    assert cut.cut_capacity == bound.maximum_removable_heat
    assert cut.necessary_condition_satisfied


def test_min_cut_template_is_a_generalized_layout_inequality() -> None:
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
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    assert failed.maximum_removable_heat == 0
    assert not failed.necessary_condition_satisfied
    same_cut = failed.cut_template.evaluate(
        problem,
        ("fuel", "empty"),
        catalogue,
    )
    assert same_cut.cut_capacity == 0
    assert same_cut.deficit == 2

    # The fixed node partition remains a valid inequality after relabelling:
    # a sink added to the other slot raises this cut's capacity to the heat.
    repaired = failed.cut_template.evaluate(problem, ("fuel", "sink"), catalogue)
    assert repaired.cut_capacity == 2
    assert repaired.necessary_condition_satisfied


def test_heat_flow_catalogue_validation_is_ruleset_independent() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(1, ()),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
    )
    with pytest.raises(ValueError, match="unknown components"):
        layout_heat_flow_bound(problem, ("fuel",), {"empty": HeatFlowComponent()})


def test_ideal_skeleton_bound_excludes_every_cooling_completion() -> None:
    problem = ReactorProblem(
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
        heat_scale=3,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
        "exchange": HeatFlowComponent(accepts_heat=True, exchange_side=10),
    }
    ideal = componentwise_cooling_dominator(problem, catalogue)
    assert ideal.self_vent == 2
    assert ideal.exchange_side == 10

    skeleton = ("empty", "fuel", "empty")
    failed = skeleton_heat_flow_bound(problem, skeleton, catalogue)
    assert failed.generated_heat == 6
    assert failed.maximum_removable_heat == 4
    assert not failed.necessary_condition_satisfied

    # The minimum-cut partition remains a skeleton-family inequality.
    reevaluated = evaluate_skeleton_thermal_cut(
        problem,
        skeleton,
        catalogue,
        failed.cut_template,
    )
    assert reevaluated.cut_capacity == 4
    assert reevaluated.deficit == 2


def test_compiled_average_cut_factors_equal_direct_cut_for_all_small_layouts() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(3, ((0, 1), (1, 2))),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("thermal",),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "thermal": HeatFlowComponent(
            accepts_heat=True,
            self_vent=2,
            side_vent=1,
            hull_draw=2,
            exchange_side=3,
            exchange_hull=4,
        ),
    }
    labels = ("empty", "fuel", "thermal")
    cuts = (
        ThermalCutTemplate((), (), False),
        ThermalCutTemplate((0, 2), (1,), False),
        ThermalCutTemplate((1,), (0, 2), True),
        ThermalCutTemplate((0, 1, 2), (0, 1, 2), True),
    )
    layouts = tuple(
        layout
        for layout in product(labels, repeat=problem.graph.size)
        if layout.count("fuel") == 1
    )
    for cut in cuts:
        compiled = compile_thermal_cut(
            problem,
            cut,
            catalogue,
            labels,
            placement_order=problem.graph.update_order,
        )
        assert compiled.selection.selected_representation in {
            "conditioned_residual_functions",
            "per_variable_function_quotient",
        }
        for layout in layouts:
            direct = cut.evaluate(problem, layout, catalogue)
            assert compiled.score(layout) == (
                direct.cut_capacity - direct.generated_heat
            )
