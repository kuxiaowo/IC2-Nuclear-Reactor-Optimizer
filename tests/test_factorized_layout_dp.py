from __future__ import annotations

from itertools import product

import pytest

from ic2_reactor.factorized_layout_dp import FactorizedLayoutFeasibilityDP
from ic2_reactor.factorized_cooling_master import FactorizedCoolingCutMaster
from ic2_reactor.frontier_automata import (
    ExcludedLayoutsAutomaton,
    JointLocalFactorThresholdAutomaton,
    LocalScoreFactor,
    normalize_factor_constraint,
    rectangular_frontier_order,
    select_factor_automaton,
)
from ic2_reactor.mathematical_model import (
    Graph,
    PowerComponent,
    ReactorProblem,
    evaluate_power_skeleton,
)
from ic2_reactor.periodic_prefix import PrefixHeatComponent, periodic_prefix_flow_bound
from ic2_reactor.thermal_cut_automaton import compile_thermal_cut
from ic2_reactor.thermal_relaxation import HeatFlowComponent, layout_heat_flow_bound


def thermal_completion_problem() -> tuple[
    ReactorProblem,
    dict[str, HeatFlowComponent],
]:
    problem = ReactorProblem(
        graph=Graph.rectangular(1, 2),
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
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "weak": HeatFlowComponent(),
        "strong": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    return problem, catalogue


def test_full_label_behaviour_dp_matches_every_cooling_completion() -> None:
    problem, catalogue = thermal_completion_problem()
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    labels = ("empty", "fuel", "weak", "strong")
    order = rectangular_frontier_order(problem.graph)
    compiled = compile_thermal_cut(
        problem,
        failed.cut_template,
        catalogue,
        labels,
        placement_order=order,
    )
    result = FactorizedLayoutFeasibilityDP(
        order,
        labels,
        free_labels=("empty", "weak", "strong"),
        fixed_labels={0: "fuel"},
        automata=(compiled.automaton,),
    ).solve()
    assert result.proven and result.feasible
    assert result.layout is not None
    assert failed.cut_template.evaluate(
        problem,
        result.layout,
        catalogue,
    ).necessary_condition_satisfied

    brute = tuple(
        ("fuel", cooling)
        for cooling in ("empty", "weak", "strong")
        if failed.cut_template.evaluate(
            problem,
            ("fuel", cooling),
            catalogue,
        ).necessary_condition_satisfied
    )
    assert brute == (("fuel", "strong"),)
    assert result.layout in brute


def test_full_label_behaviour_dp_proves_cut_infeasibility_with_inventory() -> None:
    problem, catalogue = thermal_completion_problem()
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    labels = ("empty", "fuel", "weak", "strong")
    order = rectangular_frontier_order(problem.graph)
    compiled = compile_thermal_cut(
        problem,
        failed.cut_template,
        catalogue,
        labels,
        placement_order=order,
    )
    result = FactorizedLayoutFeasibilityDP(
        order,
        labels,
        free_labels=("empty", "weak", "strong"),
        fixed_labels={0: "fuel"},
        component_limits={"strong": 0},
        automata=(compiled.automaton,),
    ).solve()
    assert result.proven
    assert not result.feasible
    assert result.layout is None


def test_full_label_behaviour_dp_timeout_never_becomes_infeasibility_proof() -> None:
    problem, catalogue = thermal_completion_problem()
    labels = ("empty", "fuel", "weak", "strong")
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    order = rectangular_frontier_order(problem.graph)
    compiled = compile_thermal_cut(
        problem,
        failed.cut_template,
        catalogue,
        labels,
        placement_order=order,
    )
    result = FactorizedLayoutFeasibilityDP(
        order,
        labels,
        free_labels=("empty", "weak", "strong"),
        fixed_labels={0: "fuel"},
        automata=(compiled.automaton,),
    ).solve(time_limit_seconds=1e-12)
    assert not result.proven
    assert not result.feasible
    assert result.stop_reason == "time_limit"


def test_full_label_dp_eliminates_nonbinding_and_fixed_inventory_dimensions() -> None:
    nonbinding = FactorizedLayoutFeasibilityDP(
        (0, 1, 2),
        ("A", "B"),
        free_labels=("A", "B"),
        fixed_labels={0: "A"},
        component_limits={"A": 3, "B": 2},
    )
    # A has two remaining units for two free vertices; B also has two.  Neither
    # upper bound can bind, so neither belongs in a Pareto resource vector.
    assert nonbinding.base_resource_dimensions == 0
    assert nonbinding.solve().proven

    impossible = FactorizedLayoutFeasibilityDP(
        (0, 1),
        ("A", "B"),
        free_labels=("A", "B"),
        fixed_labels={0: "A", 1: "A"},
        component_limits={"A": 1},
    ).solve()
    assert impossible.proven and not impossible.feasible
    assert impossible.raw_transitions == 0
    assert impossible.stop_reason == "fixed_inventory_infeasible"


def test_full_label_dp_groups_identical_successors_before_pareto_insertion() -> None:
    result = FactorizedLayoutFeasibilityDP(
        (0, 1),
        ("A", "B", "C"),
        free_labels=("A", "B", "C"),
    ).solve()
    assert result.proven and result.feasible
    assert result.layout == ("A", "A")
    assert result.raw_transitions == 6
    assert result.equivalent_successor_merges == 4
    assert result.dominated_rejections == 0


def test_full_label_behaviour_dp_matches_two_signed_cuts_and_inventories() -> None:
    labels = ("A", "B", "C")
    first_factor = LocalScoreFactor.tabulate(
        (0, 1),
        len(labels),
        lambda codes: 1 if codes[0] != codes[1] else -1,
    )
    second_factor = LocalScoreFactor.tabulate(
        (1, 2),
        len(labels),
        lambda codes: 1 if codes[0] == codes[1] else -1,
    )
    first, _first_selection = select_factor_automaton(
        (0, 1, 2),
        (first_factor,),
        threshold=0,
    )
    second, _second_selection = select_factor_automaton(
        (0, 1, 2),
        (second_factor,),
        threshold=0,
    )
    joint = JointLocalFactorThresholdAutomaton(
        (0, 1, 2),
        (
            normalize_factor_constraint((first_factor,), threshold=0),
            normalize_factor_constraint((second_factor,), threshold=0),
        ),
    )

    for limits in ({}, {"B": 0}, {"B": 0, "C": 0}):
        brute = []
        for layout in product(labels, repeat=3):
            if layout[0] != "A":
                continue
            if any(layout.count(label) > limit for label, limit in limits.items()):
                continue
            codes = tuple(labels.index(label) for label in layout)
            if (
                first_factor.evaluate((codes[0], codes[1])) >= 0
                and second_factor.evaluate((codes[1], codes[2])) >= 0
            ):
                brute.append(layout)

        result = FactorizedLayoutFeasibilityDP(
            (0, 1, 2),
            labels,
            free_labels=labels,
            fixed_labels={0: "A"},
            component_limits=limits,
            automata=(first, second),
        ).solve()
        assert result.proven
        assert result.feasible == bool(brute)
        if brute:
            assert result.layout in brute
        else:
            assert result.layout is None

        joint_result = FactorizedLayoutFeasibilityDP(
            (0, 1, 2),
            labels,
            free_labels=labels,
            fixed_labels={0: "A"},
            component_limits=limits,
            automata=(joint,),
        ).solve()
        assert joint_result.proven
        assert joint_result.feasible == bool(brute)
        if brute:
            assert joint_result.layout in brute
        else:
            assert joint_result.layout is None


def test_exact_layout_no_goods_share_one_prefix_automaton() -> None:
    labels = ("A", "B")
    order = (0, 1)
    excluded = (("A", "A"), ("A", "B"), ("B", "A"))
    no_goods = ExcludedLayoutsAutomaton(order, labels, excluded)
    assert no_goods.excluded_layout_count == 3
    assert no_goods.trie_node_count == 6
    assert no_goods.trie_edge_count == 5
    result = FactorizedLayoutFeasibilityDP(
        order,
        labels,
        free_labels=labels,
        automata=(no_goods,),
    ).solve()
    assert result.proven and result.feasible
    assert result.layout == ("B", "B")

    all_excluded = ExcludedLayoutsAutomaton(
        order,
        labels,
        (*excluded, ("B", "B")),
    )
    closed = FactorizedLayoutFeasibilityDP(
        order,
        labels,
        free_labels=labels,
        automata=(all_excluded,),
    ).solve()
    assert closed.proven
    assert not closed.feasible


def test_reusable_factorized_cooling_master_compiles_and_caches_cuts() -> None:
    problem, catalogue = thermal_completion_problem()
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    master = FactorizedCoolingCutMaster(problem, catalogue)
    first = master.solve(
        ("fuel", "empty"),
        average_cuts=(failed.cut_template,),
    )
    assert first.proven and first.feasible
    assert first.layout == ("fuel", "strong")
    assert master.cached_average_cuts == 1
    # Only one generic automaton is needed to obtain the reusable factor
    # tables.  Every scan direction is then compiled from the much smaller
    # skeleton-conditioned factors instead of caching redundant generic
    # order automata.
    assert master.cached_average_order_automata == 1
    assert master.last_order_selection.candidate_order_count == 2
    assert set(master.last_order_selection.placement_order) == set(
        problem.graph.vertices
    )
    assert master.last_order_selection.structural_state_product_bound >= 1
    assert master.last_order_selection.raw_factor_table_entries >= (
        master.last_order_selection.quotient_factor_table_entries
    )
    assert master.last_order_selection.specialized_factor_cache_misses > 0
    assert (
        first.peak_layer_points
        <= master.last_order_selection.peak_pareto_points_without_no_goods_bound
    )
    assert (
        first.raw_transitions
        <= master.last_order_selection.raw_transitions_without_no_goods_bound
    )
    first_bound = master.last_order_selection.structural_state_product_bound

    repeated = master.solve(
        ("fuel", "empty"),
        average_cuts=(failed.cut_template, failed.cut_template),
    )
    assert repeated.proven and repeated.layout == first.layout
    assert master.last_order_selection.structural_state_product_bound == first_bound
    assert master.last_order_selection.submitted_cut_count == 2
    assert master.last_order_selection.distinct_factor_constraint_count == 1
    assert master.last_order_selection.specialized_factor_cache_hits > 0
    assert master.last_order_selection.specialized_factor_cache_misses == 0

    closed = master.solve(
        ("fuel", "empty"),
        average_cuts=(failed.cut_template,),
        excluded_layouts=(first.layout,),
    )
    assert closed.proven
    assert not closed.feasible
    assert master.cached_average_cuts == 1
    assert master.last_order_selection.submitted_layout_no_goods == 1
    assert master.last_order_selection.relevant_layout_no_goods == 1
    assert master.last_order_selection.no_good_trie_nodes == 3


def test_compile_only_cooling_audit_never_enters_layout_dp() -> None:
    problem, catalogue = thermal_completion_problem()
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    master = FactorizedCoolingCutMaster(problem, catalogue)
    result = master.compile_cuts(
        average_cuts=(failed.cut_template,),
        time_limit_seconds=1,
    )
    assert result.proven
    assert result.stop_reason == "compiled"
    assert result.selection is not None
    assert result.selection.submitted_cut_count == 1
    assert result.selection.distinct_factor_constraint_count == 1
    assert result.selection.quotient_factor_table_entries > 0
    assert result.selection.submitted_layout_no_goods == 0

    specialized = FactorizedCoolingCutMaster(
        problem,
        catalogue,
    ).compile_cuts_for_skeleton(
        ("fuel", "empty"),
        average_cuts=(failed.cut_template,),
        time_limit_seconds=1,
    )
    assert specialized.proven
    assert specialized.stop_reason == "skeleton_conditioned_cuts_compiled"
    assert specialized.selection is not None
    assert specialized.selection.distinct_factor_constraint_count == 1


def test_joint_full_layout_master_matches_complete_layout_enumeration() -> None:
    problem, catalogue = thermal_completion_problem()
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    master = FactorizedCoolingCutMaster(problem, catalogue)
    result = master.solve_joint_layouts(
        average_cuts=(failed.cut_template,),
    )
    assert result.proven
    assert result.equivalent_successor_merges > 0

    labels = master.labels
    brute_metrics = set()
    brute_layouts = []
    for layout in product(labels, repeat=problem.graph.size):
        if sum(
            next(
                item.rods
                for item in master.full_layout_problem.power_components
                if item.id == label
            )
            for label in layout
        ) != problem.rod_budget:
            continue
        if not failed.cut_template.evaluate(
            problem,
            layout,
            catalogue,
        ).necessary_condition_satisfied:
            continue
        metrics = evaluate_power_skeleton(master.full_layout_problem, layout)
        brute_metrics.add((metrics.power, metrics.generated_heat, metrics.active_cells))
        brute_layouts.append(layout)

    assert brute_layouts
    actual = {
        (point.power, point.generated_heat, point.active_cells)
        for point in result.frontier
    }
    # The tiny instance has one static Pareto value; every returned witness is
    # nevertheless checked directly against the submitted thermal cut.
    assert actual == brute_metrics
    for point in result.frontier:
        assert point.skeleton in brute_layouts


def test_joint_full_layout_master_rebuilds_after_exact_no_good() -> None:
    problem, catalogue = thermal_completion_problem()
    master = FactorizedCoolingCutMaster(problem, catalogue)
    first = master.solve_joint_layouts()
    assert first.proven and first.frontier
    excluded = first.frontier[0].skeleton

    following = master.solve_joint_layouts(excluded_layouts=(excluded,))
    assert following.proven and following.frontier
    assert all(point.skeleton != excluded for point in following.frontier)
    assert master.last_order_selection.relevant_layout_no_goods == 1


def test_joint_master_scan_order_accounts_for_no_good_prefix_sharing() -> None:
    problem = ReactorProblem(
        graph=Graph.rectangular(1, 3),
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
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "weak": HeatFlowComponent(),
    }
    master = FactorizedCoolingCutMaster(problem, catalogue)
    result = master.solve_joint_layouts(excluded_layouts=(
        ("fuel", "empty", "empty"),
        ("fuel", "empty", "weak"),
    ))
    assert result.proven and result.frontier
    # Forward scanning shares the first two labels of both no-goods; reverse
    # scanning branches at its first label and creates a larger trie.
    assert master.last_order_selection.placement_order == (0, 1, 2)
    assert master.last_order_selection.relevant_layout_no_goods == 2


def test_local_specialization_cache_uses_factor_behaviour_not_power_name() -> None:
    problem = ReactorProblem(
        graph=Graph.rectangular(1, 2),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel_a", 1, 1, True),
            PowerComponent("fuel_b", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("strong",),
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel_a": HeatFlowComponent(),
        "fuel_b": HeatFlowComponent(),
        "strong": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    cut = layout_heat_flow_bound(
        problem,
        ("fuel_a", "empty"),
        catalogue,
    ).cut_template
    master = FactorizedCoolingCutMaster(problem, catalogue)
    first = master.compile_cuts_for_skeleton(
        ("fuel_a", "empty"),
        average_cuts=(cut,),
    )
    assert first.proven and first.selection is not None
    assert first.selection.specialized_factor_cache_misses > 0

    second = master.compile_cuts_for_skeleton(
        ("fuel_b", "empty"),
        average_cuts=(cut,),
    )
    assert second.proven and second.selection is not None
    assert second.selection.specialized_factor_cache_misses == 0
    assert second.selection.specialized_factor_cache_hits > 0


def test_reusable_factorized_cooling_master_accepts_hoffman_cuts() -> None:
    problem, heat_catalogue = thermal_completion_problem()
    prefix_catalogue = {
        "empty": PrefixHeatComponent(),
        "fuel": PrefixHeatComponent(),
        "weak": PrefixHeatComponent(1, True, self_vent=2),
        "strong": PrefixHeatComponent(2, True, self_vent=2),
    }
    failed = periodic_prefix_flow_bound(
        problem,
        ("weak", "fuel"),
        prefix_catalogue,
        base_hull_capacity=10,
    )
    assert not failed.feasible
    master = FactorizedCoolingCutMaster(
        problem,
        heat_catalogue,
        prefix_catalogue=prefix_catalogue,
        base_hull_capacity=10,
    )
    result = master.solve(
        ("empty", "fuel"),
        prefix_cuts=(failed.cut_template,),
    )
    assert result.proven and result.feasible
    assert result.layout is not None
    direct = failed.cut_template.evaluate(
        problem,
        result.layout,
        prefix_catalogue,
        base_hull_capacity=10,
    )
    assert direct.necessary_condition_satisfied
    assert master.cached_prefix_cuts == 1


def test_reusable_factorized_cooling_master_validates_no_goods() -> None:
    problem, catalogue = thermal_completion_problem()
    master = FactorizedCoolingCutMaster(problem, catalogue)
    with pytest.raises(ValueError, match="wrong size"):
        master.solve(
            ("fuel", "empty"),
            excluded_layouts=(("fuel",),),
        )
    with pytest.raises(ValueError, match="unknown labels"):
        master.solve(
            ("fuel", "empty"),
            excluded_layouts=(("fuel", "missing"),),
        )


def test_reusable_cooling_master_time_limit_includes_cut_compilation() -> None:
    problem, catalogue = thermal_completion_problem()
    failed = layout_heat_flow_bound(problem, ("fuel", "empty"), catalogue)
    result = FactorizedCoolingCutMaster(problem, catalogue).solve(
        ("fuel", "empty"),
        average_cuts=(failed.cut_template,),
        time_limit_seconds=1e-12,
    )
    assert not result.proven and not result.feasible
    assert result.stop_reason == "cut_compilation_time_limit"
