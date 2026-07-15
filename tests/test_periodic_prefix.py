from __future__ import annotations

from itertools import product

from ic2_reactor.mathematical_model import Graph, PowerComponent, ReactorProblem
from ic2_reactor.periodic_cut_automaton import compile_periodic_prefix_cut
from ic2_reactor.periodic_prefix import (
    PeriodicPrefixCutTemplate,
    PrefixHeatComponent,
    _build_periodic_network,
    periodic_prefix_flow_bound,
)
from ic2_reactor.thermal_relaxation import HeatFlowComponent, layout_heat_flow_bound


def line_problem(size: int) -> ReactorProblem:
    return ReactorProblem(
        graph=Graph.from_edges(size, tuple((i, i + 1) for i in range(size - 1))),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("vent",),
        eu_per_pulse=1,
        heat_scale=1,
    )


def prefix_catalogue(capacity: int, vent: int) -> dict[str, PrefixHeatComponent]:
    return {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "vent": PrefixHeatComponent(
            heat_capacity=capacity,
            accepts_fuel_heat=True,
            self_vent=vent,
        ),
    }


def test_prefix_capacity_rejects_batch_overflow_that_average_flow_misses() -> None:
    problem = line_problem(2)
    layout = ("vent", "fuel")
    average = layout_heat_flow_bound(
        problem,
        layout,
        {
            "empty": HeatFlowComponent(),
            "fuel": HeatFlowComponent(),
            "vent": HeatFlowComponent(accepts_heat=True, self_vent=2),
        },
    )
    assert average.necessary_condition_satisfied

    prefix = periodic_prefix_flow_bound(
        problem,
        layout,
        prefix_catalogue(capacity=1, vent=2),
        base_hull_capacity=10,
    )
    assert not prefix.feasible
    assert prefix.generated_heat == 2
    assert prefix.deficit > 0
    cut = prefix.cut_template.evaluate(
        problem,
        layout,
        prefix_catalogue(capacity=1, vent=2),
        base_hull_capacity=10,
    )
    assert cut.deficit == prefix.deficit == 1


def test_prefix_min_cut_is_reusable_after_component_relabelling() -> None:
    base = line_problem(2)
    problem = ReactorProblem(
        graph=base.graph,
        rod_budget=base.rod_budget,
        exact_rods=True,
        power_components=base.power_components,
        cooling_components=(),
        layout_components=("weak", "strong"),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "weak": PrefixHeatComponent(1, True, self_vent=2),
        "strong": PrefixHeatComponent(2, True, self_vent=2),
    }
    failed = periodic_prefix_flow_bound(
        problem,
        ("weak", "fuel"),
        catalogue,
        base_hull_capacity=10,
    )
    assert not failed.feasible
    repaired = failed.cut_template.evaluate(
        problem,
        ("strong", "fuel"),
        catalogue,
        base_hull_capacity=10,
    )
    assert repaired.necessary_condition_satisfied


def test_cyclic_prefix_network_carries_heat_to_an_earlier_vent_next_tick() -> None:
    result = periodic_prefix_flow_bound(
        line_problem(2),
        ("vent", "fuel"),
        prefix_catalogue(capacity=2, vent=2),
        base_hull_capacity=10,
    )
    assert result.feasible
    assert result.required_circulation == result.routed_circulation == 2


def test_exact_ordered_split_across_two_small_receivers_is_feasible() -> None:
    result = periodic_prefix_flow_bound(
        line_problem(3),
        ("vent", "fuel", "vent"),
        prefix_catalogue(capacity=1, vent=1),
        base_hull_capacity=10,
    )
    assert result.feasible
    assert result.generated_heat == 2


def test_compiled_hoffman_factors_equal_direct_cut_for_every_small_layout() -> None:
    base = line_problem(3)
    problem = ReactorProblem(
        graph=base.graph,
        rod_budget=base.rod_budget,
        exact_rods=True,
        power_components=base.power_components,
        cooling_components=(),
        layout_components=("weak", "strong"),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "weak": PrefixHeatComponent(1, True, self_vent=1),
        "strong": PrefixHeatComponent(2, True, self_vent=2),
    }
    failed = periodic_prefix_flow_bound(
        problem,
        ("weak", "fuel", "weak"),
        catalogue,
        base_hull_capacity=10,
    )
    labels = ("empty", "fuel", "weak", "strong")
    compiled = compile_periodic_prefix_cut(
        problem,
        failed.cut_template,
        catalogue,
        labels,
        base_hull_capacity=10,
        placement_order=problem.graph.update_order,
    )
    assert compiled.selection.selected_representation in {
        "conditioned_residual_functions",
        "per_variable_function_quotient",
    }
    checked = 0
    for layout in product(labels, repeat=problem.graph.size):
        if layout.count("fuel") != 1:
            continue
        direct = failed.cut_template.evaluate(
            problem,
            layout,
            catalogue,
            base_hull_capacity=10,
        )
        assert compiled.score(layout) == (
            direct.upper_bound_out_of_source_side
            - direct.lower_bound_into_source_side
        )
        checked += 1
    assert checked == 3 * 3**2


def test_compiled_hoffman_factors_cover_all_capacity_edge_families() -> None:
    base = line_problem(3)
    problem = ReactorProblem(
        graph=base.graph,
        rod_budget=1,
        exact_rods=True,
        power_components=base.power_components,
        cooling_components=(),
        layout_components=("thermal",),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "thermal": PrefixHeatComponent(
            heat_capacity=3,
            accepts_fuel_heat=True,
            self_vent=2,
            side_vent=1,
            hull_draw=2,
            exchange_side=1,
            exchange_hull=3,
            hull_capacity_bonus=4,
        ),
    }
    labels = ("empty", "fuel", "thermal")
    builder, _generated = _build_periodic_network(
        problem,
        ("thermal", "fuel", "empty"),
        catalogue,
        base_hull_capacity=7,
    )
    source_sets = (
        (),
        tuple(builder.names),
        tuple(builder.names[::2]),
        tuple(builder.names[1::3]),
        tuple(builder.names[2::5]),
    )
    layouts = tuple(
        layout
        for layout in product(labels, repeat=problem.graph.size)
        if layout.count("fuel") == 1
    )
    for source_nodes in source_sets:
        cut = PeriodicPrefixCutTemplate(source_nodes)
        compiled = compile_periodic_prefix_cut(
            problem,
            cut,
            catalogue,
            labels,
            base_hull_capacity=7,
            placement_order=problem.graph.update_order,
        )
        for layout in layouts:
            direct = cut.evaluate(
                problem,
                layout,
                catalogue,
                base_hull_capacity=7,
            )
            assert compiled.score(layout) == (
                direct.upper_bound_out_of_source_side
                - direct.lower_bound_into_source_side
            )
