from __future__ import annotations

import random
from itertools import product

from ic2_reactor.anytime_math_optimizer import (
    CertifiedAnytimeSolver,
    IC2CoolingLNS,
    sample_skeletons_at_power,
)
from ic2_reactor.engine import ReactorSimulator
from ic2_reactor.mathematical_model import (
    AggregatePattern,
    Graph,
    IC2CycleOracle,
    PowerHeatMaster,
    aggregate_overload_analysis,
    bipartite_degree_sequence_possible,
    closed_form_upper_bound,
    component_exchanger_relief_bound,
    derive_ic2_top_tier_cut,
    evaluate_power_skeleton,
    ic2_mark_i_problem,
    minimum_heat_for_pulse_units,
    route_conditioned_overload_analysis,
    PowerComponent,
    ReactorProblem,
)
from ic2_reactor.models import Layout
from ic2_reactor.ic2_thermal_catalog import IC2_HEAT_FLOW_CATALOGUE
from ic2_reactor.thermal_relaxation import layout_heat_flow_bound


WITNESS_380 = tuple(
    {
        ".": "empty",
        "S": "uranium_single",
        "Q": "uranium_quad",
        "R": "iridium_reflector",
        "O": "overclocked_heat_vent",
        "C": "component_heat_vent",
        "X": "component_heat_exchanger",
        "P": "reactor_plating",
    }[symbol]
    for symbol in "".join((
        "QCOXOOCRP",
        ".COOCORSR",
        "POQOOQOOX",
        "COOCOOCOO",
        "OQOOQOOQO",
        "COCPOCPOC",
    ))
)


def test_rectangular_graph_and_arbitrary_graph_are_parameterised() -> None:
    grid = Graph.rectangular(2, 3)
    assert grid.size == 6
    assert grid.maximum_degree == 3
    assert grid.edges == ((0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5))

    path = Graph.from_edges(4, ((0, 1), (1, 2), (2, 3)), update_order=(3, 2, 1, 0))
    assert path.neighbours == ((1,), (0, 2), (1, 3), (2,))
    assert path.update_order == (3, 2, 1, 0)


def test_arbitrary_graph_at_most_rods_matches_complete_enumeration() -> None:
    catalogue = (
        PowerComponent("empty", 0, 0, False),
        PowerComponent("single", 1, 1, True),
        PowerComponent("dual", 2, 2, True),
        PowerComponent("mirror", 0, 0, True),
    )
    problem = ReactorProblem(
        graph=Graph.from_edges(5, ((0, 1), (1, 2), (1, 3), (3, 4))),
        rod_budget=3,
        exact_rods=False,
        power_components=catalogue,
        cooling_components=(),
        component_limits=(("mirror", 1),),
    )
    labels = tuple(item.id for item in catalogue)
    best = max(
        evaluate_power_skeleton(problem, skeleton).power
        for skeleton in product(labels, repeat=problem.graph.size)
        if 1 <= evaluate_power_skeleton(problem, skeleton).rods <= 3
        and skeleton.count("mirror") <= 1
    )
    result = PowerHeatMaster(problem).solve(
        seconds=5,
        workers=1,
        use_cooling_envelope=False,
    )
    assert result.proven_optimal
    assert result.power == best


def test_convex_heat_bound_for_25_rods_and_96_pulse_units() -> None:
    assert minimum_heat_for_pulse_units(25, 96) == 936


def test_bipartite_degree_sequence_filter_is_structural_not_geometric_search() -> None:
    assert not bipartite_degree_sequence_possible((2, 2, 2, 0, 0))
    assert bipartite_degree_sequence_possible((2, 1, 1, 0))
    assert not bipartite_degree_sequence_possible((3, 1, 1, 1, 1))
    assert bipartite_degree_sequence_possible((1,), unknown_vertices=1)


def test_closed_form_bound_proves_480_without_search() -> None:
    result = closed_form_upper_bound(ic2_mark_i_problem())
    assert result.power_upper_bound == 480
    assert result.heat_at_bound == 936
    assert result.cooling_upper_bound == 940
    assert result.minimum_fuel_cells == 7


def test_static_equations_match_independent_simulator_on_random_skeletons() -> None:
    rng = random.Random(5841)
    problem = ic2_mark_i_problem()
    labels = tuple(item.id for item in problem.power_components)
    for _ in range(20):
        skeleton = tuple(rng.choice(labels) for _ in range(problem.graph.size))
        metrics = evaluate_power_skeleton(problem, skeleton)
        simulator = ReactorSimulator(Layout(columns=9, slots=list(skeleton)))
        power, heat, _vented = simulator.step(auto_refuel=True)
        assert metrics.power == power
        assert metrics.generated_heat == heat


def test_cp_sat_master_proves_same_480_root_bound() -> None:
    solution = PowerHeatMaster(ic2_mark_i_problem()).solve(seconds=5, workers=1)
    assert solution.status == "OPTIMAL"
    assert solution.proven_optimal
    assert solution.power == 480
    assert solution.generated_heat == 936
    assert solution.strict_power_upper_bound == 480


def test_inventory_limits_are_inputs_not_hardcoded_constants() -> None:
    limited = ic2_mark_i_problem(component_limits={
        "overclocked_heat_vent": 8,
        "component_heat_vent": 8,
        "advanced_heat_vent": 8,
        "heat_vent": 8,
        "reactor_heat_vent": 8,
    })
    assert closed_form_upper_bound(limited).power_upper_bound == 320
    solution = PowerHeatMaster(limited).solve(seconds=5, workers=1)
    assert solution.proven_optimal
    assert solution.power == 320


def test_checked_top_tier_cuts_remove_480_through_460() -> None:
    proof = derive_ic2_top_tier_cut(ic2_mark_i_problem())
    assert proof is not None
    assert proof.power_upper_bound == 455
    assert proof.excluded_power_levels == (480, 475, 470, 465, 460)
    assert dict(proof.checks)["seven_active_degree_sum"] == 7
    assert dict(proof.checks)["aggregate_patterns_470"] == 13
    assert dict(proof.checks)["minimum_470_relief_margin"] > 0
    assert dict(proof.checks)["aggregate_patterns_465"] == 82
    assert dict(proof.checks)["minimum_465_relief_margin"] > 0
    assert dict(proof.checks)["aggregate_patterns_460"] == 219
    assert dict(proof.checks)["minimum_460_relief_margin"] > 0
    assert dict(proof.checks)["surviving_overload_patterns_460"] == 0
    assert dict(proof.checks)["structurally_surviving_patterns_460"] == 0
    analysis_455 = aggregate_overload_analysis(ic2_mark_i_problem(), 455)
    assert analysis_455.excluded is False
    assert len(analysis_455.surviving_patterns) == 3
    assert len(analysis_455.structurally_surviving_patterns) == 3
    assert {
        pattern.margin for pattern in analysis_455.structurally_surviving_patterns
    } == {-80, -20, -8}


def test_pattern_specific_exchanger_bound_accounts_for_direct_fuel_edges() -> None:
    problem = ic2_mark_i_problem()
    survivors = aggregate_overload_analysis(
        problem,
        455,
    ).structurally_surviving_patterns
    bounds = [
        component_exchanger_relief_bound(problem, pattern)
        for pattern in survivors
    ]
    assert sorted(bounds) == [72, 72, 88]

    hotter = AggregatePattern(
        active_cells=1,
        generated_heat=336,
        slack=0,
        required_relief=0,
        maximum_available_relief=0,
        margin=0,
        fuel_degree_counts=(("uranium_quad", 3, 1),),
    )
    assert component_exchanger_relief_bound(problem, hotter) == 88


def test_route_conditioning_does_not_mix_incompatible_exchanger_advantages() -> None:
    problem = ic2_mark_i_problem()
    patterns = aggregate_overload_analysis(
        problem,
        455,
    ).structurally_surviving_patterns
    analyses = [
        route_conditioned_overload_analysis(problem, pattern)
        for pattern in patterns
    ]
    assert [analysis.profile_count for analysis in analyses] == [30, 48, 60]
    assert [analysis.minimum_margin for analysis in analyses] == [-48, 4, 16]
    assert [analysis.excluded for analysis in analyses] == [False, True, True]
    assert len(analyses[0].surviving_profiles) == 8
    assert not analyses[1].surviving_profiles
    assert not analyses[2].surviving_profiles


def test_cycle_oracle_certifies_reachable_nontrivial_380_cycle() -> None:
    certificate = IC2CycleOracle().verify(WITNESS_380, columns=9, max_ticks=1_000)
    assert certificate.outcome == "safe_cycle"
    assert certificate.safe
    assert certificate.conclusive
    assert certificate.power == 380
    assert certificate.generated_heat == 616
    assert certificate.transient_length == 380
    assert certificate.period_length == 18


def test_layout_max_flow_is_a_sound_one_way_filter() -> None:
    problem = ic2_mark_i_problem()
    witness_bound = layout_heat_flow_bound(problem, WITNESS_380, IC2_HEAT_FLOW_CATALOGUE)
    assert witness_bound.generated_heat == 616
    assert witness_bound.necessary_condition_satisfied

    uncoolable = ["empty"] * 54
    for index in range(6):
        uncoolable[index * 9] = "uranium_quad"
    uncoolable[53] = "uranium_single"
    bad_bound = layout_heat_flow_bound(problem, uncoolable, IC2_HEAT_FLOW_CATALOGUE)
    assert bad_bound.generated_heat > 0
    assert not bad_bound.necessary_condition_satisfied
    assert bad_bound.deficit > 0


def test_lns_accepts_generic_skeleton_and_rechecks_known_witness() -> None:
    problem = ic2_mark_i_problem()
    skeleton = tuple(
        item if item in {component.id for component in problem.power_components} else "empty"
        for item in WITNESS_380
    )
    result = IC2CoolingLNS(problem).search(
        skeleton,
        seconds=1,
        horizon=400,
        population=8,
        initial_layouts=(WITNESS_380,),
    )
    assert result.certificate is not None
    assert result.certificate.safe
    assert result.certificate.power == 380


def test_unsatisfiable_power_tier_is_closed_by_static_proof() -> None:
    results, status = sample_skeletons_at_power(
        ic2_mark_i_problem(),
        power=485,
        limit=1,
        seconds=5,
        workers=1,
        seed=9,
    )
    assert results == []
    assert status == "exhausted"


def test_anytime_report_keeps_unsearched_tiers_open() -> None:
    report = CertifiedAnytimeSolver(ic2_mark_i_problem()).solve(
        time_limit_seconds=3,
        workers=1,
        known_layouts=(WITNESS_380,),
        skeletons_per_tier=1,
        cooling_seconds_per_skeleton=0.01,
        thermal_horizon=20,
    )
    assert report.lower_bound == 380
    assert report.upper_bound >= report.lower_bound
    assert report.analytical_cut_upper_bound == 455
    assert report.analytical_proof is not None
    assert dict(report.analytical_proof.checks)["aggregate_patterns_470"] == 13
    assert not report.proven_global
    assert report.best_cycle is not None
    assert "not claimed infeasible" in report.statement


def test_exact_cycle_and_anytime_root_obey_a_shared_deadline() -> None:
    certificate = IC2CycleOracle().verify(
        WITNESS_380,
        columns=9,
        max_ticks=100_000,
        time_limit_seconds=1e-12,
    )
    assert not certificate.safe
    assert not certificate.conclusive
    assert certificate.outcome == "time_limit"

    report = CertifiedAnytimeSolver(ic2_mark_i_problem()).solve(
        time_limit_seconds=1e-12,
        workers=1,
    )
    assert report.static_master_status == "SKIPPED_TIME_LIMIT"
    assert not report.proven_global
    assert report.upper_bound >= report.lower_bound

    skeleton = tuple(
        "uranium_quad" if index < 6 else (
            "uranium_single" if index == 6 else "empty"
        )
        for index in range(54)
    )
    cooling = IC2CoolingLNS(ic2_mark_i_problem()).search(
        skeleton,
        seconds=1e-12,
        horizon=400,
        population=8,
    )
    assert cooling.certificate is None
    assert cooling.evaluated == 0
