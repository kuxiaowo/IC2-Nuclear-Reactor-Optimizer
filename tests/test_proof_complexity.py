from __future__ import annotations

import pytest

from ic2_reactor.frontier_automata import FactorAutomatonSelection
from ic2_reactor.factorized_layout_dp import FactorizedLayoutFeasibilityDP
from ic2_reactor.factorized_cooling_master import CoolingOrderSelection
from ic2_reactor.mathematical_model import Graph, PowerComponent, ReactorProblem
from ic2_reactor.pareto_frontier_dp import RectangularParetoPowerHeatDP
from ic2_reactor.proof_complexity import (
    average_flow_primal_encoding_size,
    global_dynamic_state_work_bound,
    certify_exact_proof_capacity,
    choose_exact_elimination_strategy,
    joint_layout_analytic_work_bound,
    proof_work_envelope,
    project_frontier_memory,
    project_proof_budget,
    summarize_factor_compilation,
    summarize_compiled_cooling_domains,
    summarize_factorized_layout_work,
    summarize_frontier_work,
    terminal_cut_quotient_work_bound,
)
from ic2_reactor.terminal_cut_quotient import TerminalCutScheduleProfile


def cooling_selection(**overrides) -> CoolingOrderSelection:
    values = dict(
        candidate_order_count=1,
        placement_order=(0,),
        structural_state_product_bound=3,
        submitted_cut_count=1,
        distinct_factor_constraint_count=1,
        raw_factor_table_entries=2,
        quotient_factor_table_entries=2,
        per_variable_representations=1,
        residual_function_representations=0,
        submitted_layout_no_goods=0,
        relevant_layout_no_goods=0,
        no_good_trie_nodes=0,
        joint_per_variable_representation=False,
        separate_constraint_state_bounds=(3,),
        cut_score_chain_bounds=((0, 2),),
        cut_score_antichain_width_bound=1,
        cut_score_antichain_width_exact=True,
        resource_antichain_width_bound=1,
        resource_antichain_width_exact=True,
        peak_pareto_points_without_no_goods_bound=20,
        raw_transitions_without_no_goods_bound=100,
    )
    values.update(overrides)
    return CoolingOrderSelection(**values)


def test_compiled_cooling_domain_bounds_sum_only_declared_complete_work() -> None:
    with_no_goods = cooling_selection(
        submitted_layout_no_goods=2,
        relevant_layout_no_goods=2,
        no_good_trie_nodes=4,
    )
    infeasible = cooling_selection(
        fixed_inventory_infeasible=True,
        raw_transitions_without_no_goods_bound=0,
        peak_pareto_points_without_no_goods_bound=0,
    )
    complete = summarize_compiled_cooling_domains(
        (with_no_goods, infeasible),
        all_open_domains_accounted=True,
    )
    assert complete.work_count_complete
    assert complete.bounded_domains == 2
    assert complete.inventory_infeasible_domains == 1
    assert complete.total_raw_transition_bound == 500
    assert complete.peak_continuation_key_bound == 15
    assert complete.peak_pareto_point_bound == 100

    undeclared = summarize_compiled_cooling_domains(
        (with_no_goods,),
        all_open_domains_accounted=False,
    )
    assert not undeclared.work_count_complete

    missing = summarize_compiled_cooling_domains(
        (cooling_selection(raw_transitions_without_no_goods_bound=None),),
        all_open_domains_accounted=True,
    )
    assert not missing.work_count_complete
    assert missing.missing_domain_indices == (0,)


def test_joint_layout_bound_counts_quotient_states_not_layout_strings() -> None:
    problem = tiny_problem()
    selection = cooling_selection(
        placement_order=(0, 1),
        submitted_cut_count=0,
        structural_state_product_bound=1,
        cut_score_chain_bounds=(),
        cut_score_antichain_width_bound=1,
    )
    bound = joint_layout_analytic_work_bound(problem, selection)
    assert bound.bound_complete
    assert bound.full_label_count == 3
    # Empty and sink have one static behaviour; fuel has the other.  The
    # capacity calculation never raises this to three power behaviours.
    assert bound.static_power_behaviour_classes == 2
    assert bound.degree_tracked_power_behaviour_classes == 1
    assert bound.continuation_key_bound == (
        bound.graph_separator_state_bound * bound.rod_state_bound
    )
    assert bound.maximum_metric_pairs_per_rod_state <= (
        bound.independent_degree_metric_pairs_total
    )
    assert bound.raw_label_transition_bound is not None

    complete_conditioned = summarize_compiled_cooling_domains(
        (cooling_selection(raw_transitions_without_no_goods_bound=1),),
        all_open_domains_accounted=True,
    )
    choice = choose_exact_elimination_strategy(bound, complete_conditioned)
    assert choice.comparison_complete
    assert choice.selected_strategy == "conditioned_skeleton_domains"

    incomplete_conditioned = summarize_compiled_cooling_domains(
        (cooling_selection(),),
        all_open_domains_accounted=False,
    )
    undecided = choose_exact_elimination_strategy(bound, incomplete_conditioned)
    assert not undecided.comparison_complete
    assert undecided.selected_strategy is None


def tiny_problem() -> ReactorProblem:
    return ReactorProblem(
        graph=Graph.rectangular(1, 2),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("empty", "sink"),
        eu_per_pulse=1,
        heat_scale=1,
    )


def test_work_envelope_uses_generating_functions_not_raw_label_power() -> None:
    envelope = proof_work_envelope(
        tiny_problem(),
        incumbent_lower_bound=0,
        static_upper_bound=2,
    )
    # Two positions can hold the single fuel.
    assert envelope.rod_feasible_power_skeletons == 2
    # The other position has two zero-rod full labels: empty or sink.
    assert envelope.rod_feasible_full_layouts == 4
    assert envelope.open_power_tiers == (1, 2)
    assert envelope.frontier_width == 1
    assert envelope.frontier_transition_bound == 2 * envelope.frontier_state_bound


def test_six_hour_projection_is_an_explicit_capacity_inequality() -> None:
    projection = project_proof_budget(
        100,
        2.0,
        workers=10,
        parallel_efficiency=0.5,
        wall_time_budget_seconds=30,
    )
    assert projection.required_core_seconds == 200
    assert projection.available_effective_core_seconds == 150
    assert projection.projected_wall_seconds == 40
    assert projection.maximum_units_in_budget == 75
    assert not projection.fits_budget


def test_six_hour_projection_rejects_unmeasured_zero_cost() -> None:
    with pytest.raises(ValueError, match="positive"):
        project_proof_budget(1, 0, workers=1)


def test_terminal_cut_work_bound_counts_vector_not_cut_templates() -> None:
    profile = TerminalCutScheduleProfile(
        factor_count=10,
        placement_steps=4,
        maximum_layout_scope=3,
        maximum_cut_scope=3,
        live_terminals_during_step=(2, 3, 3, 1),
        peak_live_terminals=3,
        peak_cut_vector_entries=8,
    )
    bound = terminal_cut_quotient_work_bound(profile, saturation=255)
    assert bound.cut_vector_entries == 8
    assert bound.bits_per_value == 8
    assert bound.packed_bytes_per_signature == 8
    assert bound.primitive_values_per_full_vector_pass == 8
    assert bound.coarse_full_scan_value_operations_bound == 0
    assert bound.log2_raw_vector_count_bound == 64


def test_average_flow_primal_size_is_linear_in_graph_size() -> None:
    size = average_flow_primal_encoding_size(tiny_problem())
    assert size.vertices == 2
    assert size.undirected_edges == 1
    assert size.flow_variables == 14
    assert size.capacity_product_variables == 6
    assert size.additional_integer_variables == 20
    assert size.additional_constraints == 27


def test_global_dynamic_fallback_is_summed_by_generating_function() -> None:
    problem = tiny_problem()
    bound = global_dynamic_state_work_bound(
        problem,
        {"empty": 1, "fuel": 1, "sink": 3},
    )
    # Exactly one fuel.  For either fuel position the other slot is empty
    # (weight 1) or sink (weight 3): total local-state weight is 8.
    assert bound.rod_feasible_layout_count == 4
    assert bound.total_safe_state_step_bound == 8 * 8_500
    assert bound.maximum_single_layout_safe_state_bound == 3 * 8_500
    assert bound.hull_bonus_bucket_count == 1


def test_measured_frontier_ledger_and_memory_projection_are_explicit() -> None:
    result = RectangularParetoPowerHeatDP(tiny_problem()).solve()
    ledger = summarize_frontier_work(result)
    assert ledger.proven
    assert ledger.completed_layers == 2
    assert ledger.raw_transitions == result.raw_transitions
    assert ledger.peak_pareto_points >= ledger.peak_continuation_keys
    assert ledger.measured_transitions_per_second > 0

    memory = project_frontier_memory(
        ledger.peak_continuation_keys,
        ledger.peak_pareto_points,
        conservative_bytes_per_key=100,
        conservative_bytes_per_point=50,
        fixed_overhead_bytes=25,
        memory_budget_bytes=1_000,
    )
    assert memory.projected_peak_bytes == (
        25
        + 100 * ledger.peak_continuation_keys
        + 50 * ledger.peak_pareto_points
    )
    assert memory.fits_budget


def test_factor_compilation_ledger_records_both_exact_representations() -> None:
    ledger = summarize_factor_compilation((
        FactorAutomatonSelection(
            "per_variable_function_quotient",
            3_200_000,
            243,
            1_000,
            2_000,
            5,
            20,
            True,
        ),
        FactorAutomatonSelection(
            "conditioned_residual_functions",
            64,
            8,
            16,
            4,
            2,
            7,
            True,
        ),
    ))
    assert ledger.cut_count == 2
    assert ledger.raw_factor_table_entries == 3_200_064
    assert ledger.quotient_factor_table_entries == 251
    assert ledger.table_reduction_ratio is not None
    assert ledger.table_reduction_ratio > 10_000
    assert ledger.per_variable_representations == 1
    assert ledger.residual_function_representations == 1


def test_capacity_certificate_refuses_a_numeric_fit_with_missing_premises() -> None:
    time_projection = project_proof_budget(
        10,
        0.1,
        workers=2,
        parallel_efficiency=1,
        wall_time_budget_seconds=10,
    )
    memory_projection = project_frontier_memory(
        1,
        1,
        conservative_bytes_per_key=10,
        conservative_bytes_per_point=10,
        memory_budget_bytes=100,
    )
    incomplete = certify_exact_proof_capacity(
        time_projection,
        memory_projection,
        work_count_complete=False,
        conservative_unit_cost_measured=True,
        conservative_memory_cost_measured=True,
        parallel_efficiency_measured=True,
    )
    assert not incomplete.certified_fit
    assert "work count" in incomplete.failure_reasons[0]

    complete = certify_exact_proof_capacity(
        time_projection,
        memory_projection,
        work_count_complete=True,
        conservative_unit_cost_measured=True,
        conservative_memory_cost_measured=True,
        parallel_efficiency_measured=True,
    )
    assert complete.certified_fit
    assert not complete.failure_reasons

    unmeasured_parallel = certify_exact_proof_capacity(
        time_projection,
        memory_projection,
        work_count_complete=True,
        conservative_unit_cost_measured=True,
        conservative_memory_cost_measured=True,
        parallel_efficiency_measured=False,
    )
    assert not unmeasured_parallel.certified_fit
    assert "parallel efficiency" in unmeasured_parallel.failure_reasons[0]


def test_factorized_layout_ledger_counts_future_equivalent_labels() -> None:
    result = FactorizedLayoutFeasibilityDP(
        (0, 1),
        ("A", "B", "C"),
        free_labels=("A", "B", "C"),
    ).solve()
    ledger = summarize_factorized_layout_work(result)
    assert ledger.proven
    assert ledger.completed_layers == 2
    assert ledger.raw_label_transitions == 6
    assert ledger.equivalent_successor_merges == 4
    assert ledger.peak_pareto_points == 1
