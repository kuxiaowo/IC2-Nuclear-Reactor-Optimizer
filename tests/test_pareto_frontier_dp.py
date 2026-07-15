from __future__ import annotations

from itertools import product
from typing import Callable

from ic2_reactor.mathematical_model import (
    AggregatePattern,
    Graph,
    PowerComponent,
    ReactorProblem,
    evaluate_power_skeleton,
)
from ic2_reactor.frontier_automata import (
    AggregateDegreeAutomaton,
    JointLocalFactorThresholdAutomaton,
    LocalFactorThresholdAutomaton,
    LocalRewardThresholdAutomaton,
    LocalScoreFactor,
    condition_factor_constraint,
    factorwise_constraint_implies,
    factor_hypergraph_frontier_orders,
    mobius_decompose_local_factor,
    ResidualFactorThresholdAutomaton,
    FrontierTransitionContext,
    select_factor_automaton,
    rectangular_frontier_order,
    rectangular_frontier_orders,
    normalize_factor_constraint,
    remove_factorwise_implied_constraints,
)
from ic2_reactor.pareto_frontier_dp import RectangularParetoPowerHeatDP
from ic2_reactor.factorized_layout_dp import FactorizedLayoutFeasibilityDP
from ic2_reactor.state_quotient import ContinuationParetoTable, ParetoPoint


def small_problem(rows: int = 2, columns: int = 3) -> ReactorProblem:
    return ReactorProblem(
        graph=Graph.rectangular(rows, columns),
        rod_budget=2,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
            PowerComponent("reflector", 0, 0, True),
        ),
        cooling_components=(),
        component_limits=(("reflector", 2),),
        eu_per_pulse=1,
        heat_scale=1,
    )


def test_minimum_width_rectangle_exposes_all_four_scan_directions() -> None:
    graph = Graph.rectangular(2, 3)
    orders = rectangular_frontier_orders(graph)
    assert len(orders) == 4
    assert orders[0] == rectangular_frontier_order(graph)
    assert all(set(order) == set(graph.vertices) for order in orders)


def test_every_minimum_width_scan_order_has_the_same_exact_frontier() -> None:
    problem = small_problem()
    frontiers = []
    for order in rectangular_frontier_orders(problem.graph):
        result = RectangularParetoPowerHeatDP(
            problem,
            placement_order=order,
        ).solve()
        assert result.proven
        frontiers.append({
            (
                point.power,
                point.generated_heat,
                point.active_cells,
                point.residual_inventory,
            )
            for point in result.frontier
        })
    assert frontiers and all(frontier == frontiers[0] for frontier in frontiers)


def brute_frontier(
    problem: ReactorProblem,
    accept: Callable[[tuple[str, ...]], bool] = lambda _skeleton: True,
) -> set[tuple[int, int, int, int]]:
    table = ContinuationParetoTable()
    labels = tuple(item.id for item in problem.power_components)
    for skeleton in product(labels, repeat=problem.graph.size):
        if skeleton.count("fuel") != 2 or skeleton.count("reflector") > 2:
            continue
        if not accept(skeleton):
            continue
        metrics = evaluate_power_skeleton(problem, skeleton)
        active = sum(label != "empty" for label in skeleton)
        table.insert(
            "complete",
            ParetoPoint(
                metrics.power,
                metrics.generated_heat,
                (2 - skeleton.count("reflector"), problem.graph.size - active),
                skeleton,
            ),
        )
    return {
        (
            point.power,
            point.generated_heat,
            problem.graph.size - point.residual_capacities[-1],
            point.residual_capacities[0],
        )
        for point in table.frontier("complete")
    }


def test_pareto_frontier_dp_matches_complete_enumeration() -> None:
    problem = small_problem()
    result = RectangularParetoPowerHeatDP(problem).solve()
    assert result.proven
    actual = {
        (
            point.power,
            point.generated_heat,
            point.active_cells,
            point.residual_inventory[0],
        )
        for point in result.frontier
    }
    assert actual == brute_frontier(problem)
    assert result.dominated_rejections > 0
    assert len(result.layer_statistics) == problem.graph.size + 1
    assert result.layer_statistics[-1].placed_vertices == problem.graph.size
    assert result.peak_layer_points == max(
        layer.pareto_points for layer in result.layer_statistics
    )
    assert result.peak_antichain_width == max(
        layer.maximum_antichain_width for layer in result.layer_statistics
    )
    for point in result.frontier:
        metrics = evaluate_power_skeleton(problem, point.skeleton)
        assert metrics.power == point.power
        assert metrics.generated_heat == point.generated_heat


def test_non_pulse_accepting_fuel_still_receives_neighbour_pulses() -> None:
    problem = ReactorProblem(
        graph=Graph.rectangular(1, 2),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("silent_fuel", 1, 1, False),
            PowerComponent("reflector", 0, 0, True),
        ),
        cooling_components=(),
        eu_per_pulse=1,
        heat_scale=1,
    )
    result = RectangularParetoPowerHeatDP(problem).solve()
    assert result.proven and result.frontier
    for point in result.frontier:
        metrics = evaluate_power_skeleton(problem, point.skeleton)
        assert point.power == metrics.power
        assert point.generated_heat == metrics.generated_heat
    assert any(
        point.power == 2
        and set(point.skeleton) == {"silent_fuel", "reflector"}
        for point in result.frontier
    )


def test_pareto_frontier_dp_timeout_is_never_reported_as_proof() -> None:
    result = RectangularParetoPowerHeatDP(small_problem()).solve(
        time_limit_seconds=1e-12,
    )
    assert not result.proven
    assert not result.frontier
    assert result.stop_reason == "time_limit"


def test_exact_suffix_power_bound_prunes_every_state_that_cannot_beat_incumbent() -> None:
    problem = small_problem()
    baseline = RectangularParetoPowerHeatDP(problem).solve()
    maximum_power = max(point.power for point in baseline.frontier)

    closed = RectangularParetoPowerHeatDP(problem).solve(
        incumbent_lower_bound=maximum_power,
    )
    assert closed.proven
    assert not closed.frontier
    assert closed.upper_bound_rejections > 0

    improving = RectangularParetoPowerHeatDP(problem).solve(
        incumbent_lower_bound=maximum_power - 1,
    )
    assert improving.proven
    assert improving.frontier
    assert all(point.power == maximum_power for point in improving.frontier)


def test_aggregate_degree_cut_is_compiled_into_the_frontier_key() -> None:
    problem = small_problem()
    pattern = AggregatePattern(
        active_cells=2,
        generated_heat=12,
        slack=0,
        required_relief=0,
        maximum_available_relief=0,
        margin=0,
        fuel_degree_counts=(("fuel", 1, 2),),
    )
    result = RectangularParetoPowerHeatDP(
        problem,
        automata=(AggregateDegreeAutomaton(problem, pattern),),
    ).solve()
    assert result.proven
    assert result.frontier
    for point in result.frontier:
        assert point.active_cells == 2
        assert point.skeleton.count("fuel") == 2
        assert point.skeleton.count("reflector") == 0
        fuel_vertices = [
            vertex
            for vertex, label in enumerate(point.skeleton)
            if label == "fuel"
        ]
        assert all(
            sum(
                point.skeleton[neighbour] != "empty"
                for neighbour in problem.graph.neighbours[vertex]
            ) == 1
            for vertex in fuel_vertices
        )


def test_local_edge_threshold_is_compiled_without_listing_matching_layouts() -> None:
    fuel_code = 1
    reflector_code = 2
    required_edges = 2
    for problem in (small_problem(2, 3), small_problem(3, 2)):
        automaton = LocalRewardThresholdAutomaton(
            required_edges,
            edge_reward=lambda _u, left, _v, right: int(
                {left, right} == {fuel_code, reflector_code}
            ),
        )
        result = RectangularParetoPowerHeatDP(
            problem,
            automata=(automaton,),
        ).solve()
        assert result.proven
        assert result.frontier
        actual = {
            (
                point.power,
                point.generated_heat,
                point.active_cells,
                point.residual_inventory[0],
            )
            for point in result.frontier
        }

        def enough_edges(skeleton: tuple[str, ...]) -> bool:
            return sum(
                {skeleton[first], skeleton[second]} == {"fuel", "reflector"}
                for first, second in problem.graph.edges
            ) >= required_edges

        assert actual == brute_frontier(problem, enough_edges)
        for point in result.frontier:
            assert enough_edges(point.skeleton)


def test_signed_local_factor_cut_matches_filtered_complete_enumeration() -> None:
    for problem in (small_problem(2, 3), small_problem(3, 2)):
        code_count = len(problem.power_components)
        factors = tuple(
            LocalScoreFactor.tabulate(
                (first, second),
                code_count,
                lambda codes: (
                    3
                    if set(codes) == {1, 2}
                    else (-1 if codes[0] != 0 and codes[1] != 0 else 0)
                ),
            )
            for first, second in problem.graph.edges
        )
        threshold = 3

        def score(skeleton: tuple[str, ...]) -> int:
            codes = {
                item.id: code
                for code, item in enumerate(problem.power_components)
            }
            encoded = tuple(codes[label] for label in skeleton)
            return sum(
                factor.evaluate(tuple(encoded[v] for v in factor.scope))
                for factor in factors
            )

        expected = brute_frontier(
            problem,
            lambda skeleton: score(skeleton) >= threshold,
        )
        for automaton_type in (
            LocalFactorThresholdAutomaton,
            ResidualFactorThresholdAutomaton,
        ):
            automaton = automaton_type(
                rectangular_frontier_order(problem.graph),
                factors,
                threshold=threshold,
            )
            result = RectangularParetoPowerHeatDP(
                problem,
                automata=(automaton,),
            ).solve()
            assert result.proven
            actual = {
                (
                    point.power,
                    point.generated_heat,
                    point.active_cells,
                    point.residual_inventory[0],
                )
                for point in result.frontier
            }
            assert actual == expected


def test_local_factor_boundary_automatically_quotients_equivalent_codes() -> None:
    # In the only future factor, codes 1 and 2 have identical slices for the
    # first variable.  The automaton proves and applies that quotient itself.
    factor = LocalScoreFactor.tabulate(
        (0, 2),
        3,
        lambda codes: int(codes[0] == 0) + int(codes[1] == 0),
    )
    automaton = LocalFactorThresholdAutomaton(
        (0, 1, 2),
        (factor,),
        threshold=0,
    )
    assert automaton.canonical_live_codes(0, 0) == (0, 1, 1)
    profile = automaton.complexity_profile()
    assert profile.maximum_scope_size == 2
    assert profile.peak_live_variables == 1
    assert profile.peak_raw_label_product == 3
    assert profile.peak_quotient_label_product == 2
    selected, selection = select_factor_automaton(
        (0, 1, 2),
        (factor,),
        threshold=0,
    )
    assert selection.residual_peak_product < selection.assignment_peak_product
    assert selection.selected_representation == "conditioned_residual_functions"
    assert isinstance(selected, ResidualFactorThresholdAutomaton)


def test_projected_factor_never_materializes_duplicate_label_slices() -> None:
    signatures = (
        (("cold",), ("cold",), ("hot",), ("hot",)),
        ((False,), (True,), (False,), (True,)),
        ((0,), (0,), (1,), (1,)),
    )

    def evaluator(codes: tuple[int, ...]) -> int:
        return (
            10 * int(codes[0] >= 2)
            + 3 * (codes[1] % 2)
            + int(codes[2] >= 2)
        )

    projected = LocalScoreFactor.tabulate_quotiented(
        (0, 1, 2),
        4,
        signatures,
        evaluator,
    )
    dense = LocalScoreFactor.tabulate((0, 1, 2), 4, evaluator)
    assert len(projected.values) == 2 * 2 * 2
    assert len(dense.values) == 4**3
    for codes in product(range(4), repeat=3):
        assert projected.evaluate(codes) == dense.evaluate(codes)
    for code in range(4):
        projected_residual = projected.condition(1, code)
        dense_residual = dense.condition(1, code)
        assert isinstance(projected_residual, LocalScoreFactor)
        assert isinstance(dense_residual, LocalScoreFactor)
        for remaining in product(range(4), repeat=2):
            assert projected_residual.evaluate(remaining) == dense_residual.evaluate(
                remaining
            )
    _automaton, selection = select_factor_automaton(
        (0, 1, 2),
        (projected,),
        threshold=0,
    )
    assert selection.raw_factor_table_entries == 4**3
    assert selection.quotient_factor_table_entries == 2**3


def test_factor_constraint_normalization_extracts_additive_constants() -> None:
    shifted = LocalScoreFactor((0,), 2, (5, 6))
    base = LocalScoreFactor((0,), 2, (0, 1))
    first = normalize_factor_constraint((shifted,), threshold=0)
    second = normalize_factor_constraint((base,), threshold=-5)
    assert first == second

    stronger = normalize_factor_constraint((base,), threshold=0)
    assert stronger.factors == first.factors
    assert stronger.threshold > first.threshold

    # Equal-scope factors are added before a second constant extraction.  The
    # two minima occur at opposite labels, but their sum is constant one.
    complementary = LocalScoreFactor((0,), 2, (1, 0))
    tautology = normalize_factor_constraint(
        (base, complementary),
        threshold=1,
    )
    assert tautology.factors == ()
    assert tautology.threshold == 0


def test_factorwise_implication_removes_only_proved_weaker_cut() -> None:
    lower = normalize_factor_constraint(
        (LocalScoreFactor((0,), 2, (0, 1)),),
        threshold=1,
    )
    amplified = normalize_factor_constraint(
        (LocalScoreFactor((0,), 2, (0, 2)),),
        threshold=1,
    )
    assert factorwise_constraint_implies(lower, amplified)
    assert not factorwise_constraint_implies(amplified, lower)
    assert remove_factorwise_implied_constraints((amplified, lower)) == (lower,)

    for code in range(2):
        lower_holds = sum(
            factor.evaluate((code,)) for factor in lower.factors
        ) >= lower.threshold
        amplified_holds = sum(
            factor.evaluate((code,)) for factor in amplified.factors
        ) >= amplified.threshold
        assert not lower_holds or amplified_holds


def test_factor_hypergraph_order_closes_sparse_pairs_early() -> None:
    factors = tuple(
        LocalScoreFactor.tabulate(
            scope,
            2,
            lambda codes: int(codes[0] != codes[1]),
        )
        for scope in ((0, 5), (1, 4), (2, 3))
    )
    raster = LocalFactorThresholdAutomaton(
        (0, 1, 2, 3, 4, 5),
        factors,
    ).complexity_profile().peak_quotient_label_product
    generated = factor_hypergraph_frontier_orders(
        tuple(range(6)),
        factors,
        beam_width=8,
    )
    assert generated
    assert all(set(order) == set(range(6)) for order in generated)
    cut_aware = min(
        LocalFactorThresholdAutomaton(order, factors)
        .complexity_profile()
        .peak_quotient_label_product
        for order in generated
    )
    assert raster == 8
    assert cut_aware == 2

    # Reordering changes only the separator representation.  Both exact DPs
    # decide the same physical factor inequality.
    for order in ((0, 1, 2, 3, 4, 5), generated[0]):
        automaton, _selection = select_factor_automaton(
            order,
            factors,
            threshold=2,
        )
        result = FactorizedLayoutFeasibilityDP(
            order,
            ("A", "B"),
            free_labels=("A", "B"),
            automata=(automaton,),
        ).solve()
        assert result.proven and result.feasible
        assert result.layout is not None
        codes = tuple(("A", "B").index(label) for label in result.layout)
        assert sum(
            factor.evaluate(tuple(codes[vertex] for vertex in factor.scope))
            for factor in factors
        ) >= 2


def test_joint_monotone_cut_score_saturates_at_threshold() -> None:
    constraint = normalize_factor_constraint(
        (LocalScoreFactor((0,), 2, (0, 10)),),
        threshold=3,
    )
    automaton = JointLocalFactorThresholdAutomaton((0,), (constraint,))
    transition = automaton.advance(
        automaton.initial_state(),
        automaton.initial_resources(),
        FrontierTransitionContext(
            step=0,
            vertex=0,
            placed_code=1,
            major=0,
            minor=0,
            placed_neighbours=(),
            previous_frontier=(),
            next_frontier=(),
            finalized_vertex=None,
            finalized_entry=None,
        ),
    )
    assert transition is not None
    assert transition.resources == (3,)
    assert automaton.pareto_resource_chain_bounds() == ((0, 3),)


def test_mobius_decomposition_proves_exact_lower_interaction_order() -> None:
    factor = LocalScoreFactor.tabulate(
        (0, 1, 2),
        3,
        lambda codes: (
            7
            + 2 * codes[0]
            - codes[1]
            + 3 * codes[0] * codes[2]
        ),
    )
    constant, pieces = mobius_decompose_local_factor(factor)
    assert constant == 7
    assert pieces
    assert max(len(piece.scope) for piece in pieces) == 2
    for codes in product(range(3), repeat=3):
        reconstructed = constant + sum(
            piece.evaluate(tuple(codes[vertex] for vertex in piece.scope))
            for piece in pieces
        )
        assert reconstructed == factor.evaluate(codes)


def test_conditioned_factor_constraint_preserves_every_completion() -> None:
    factor = LocalScoreFactor.tabulate(
        (0, 1, 2),
        3,
        lambda codes: 7 * codes[0] - 3 * codes[1] + codes[2],
    )
    unary = LocalScoreFactor.tabulate(
        (1,),
        3,
        lambda codes: 5 - 2 * codes[0],
    )
    conditioned = condition_factor_constraint(
        (factor, unary),
        {0: 2, 2: 1},
        threshold=9,
        allowed_codes=(0, 2),
    )
    for middle in (0, 2):
        original_score = factor.evaluate((2, middle, 1)) + unary.evaluate((middle,))
        residual_score = sum(
            item.evaluate((middle,)) for item in conditioned.factors
        )
        assert (original_score >= 9) == (
            residual_score >= conditioned.threshold
        )


def test_residual_factor_state_merges_joint_assignments_with_same_future_function() -> None:
    factor = LocalScoreFactor.tabulate(
        (0, 1, 2),
        2,
        lambda codes: int((codes[0] ^ codes[1] ^ codes[2]) == 0),
    )
    local = LocalFactorThresholdAutomaton((0, 1, 2), (factor,), threshold=1)
    residual = ResidualFactorThresholdAutomaton((0, 1, 2), (factor,), threshold=1)

    def context(step: int, code: int) -> FrontierTransitionContext:
        return FrontierTransitionContext(
            step=step,
            vertex=step,
            placed_code=code,
            major=step,
            minor=0,
            placed_neighbours=(),
            previous_frontier=((0, 0),),
            next_frontier=((code, 0),),
            finalized_vertex=None,
            finalized_entry=None,
        )

    def prefix(automaton, codes: tuple[int, ...]):
        state = automaton.initial_state()
        resources = automaton.initial_resources()
        for step, code in enumerate(codes):
            transition = automaton.advance(state, resources, context(step, code))
            assert transition is not None
            state, resources = transition.state, transition.resources
        return state, resources

    # 00 and 11 induce the same unary residual function of x2, although their
    # individual stored labels differ.  Only the residual-function quotient
    # recognizes this joint equivalence.
    assert prefix(local, (0, 0))[0] != prefix(local, (1, 1))[0]
    assert prefix(residual, (0, 0)) == prefix(residual, (1, 1))

    selected, report = select_factor_automaton(
        (0, 1, 2),
        (factor,),
        threshold=1,
    )
    assert report.residual_peak_product < report.assignment_peak_product
    assert report.selected_representation == "conditioned_residual_functions"
    assert isinstance(selected, ResidualFactorThresholdAutomaton)

    shifted_factor = LocalScoreFactor.tabulate(
        (0, 1),
        2,
        lambda codes: 5 * codes[0] + codes[1],
    )
    balancing_factor = LocalScoreFactor(
        (1,),
        2,
        (-10, 10),
    )
    shifted = ResidualFactorThresholdAutomaton(
        (0, 1),
        (shifted_factor, balancing_factor),
        threshold=3,
    )
    zero_state, zero_resources = prefix(shifted, (0,))
    five_state, five_resources = prefix(shifted, (1,))
    # The two residual slices [0, 1] and [5, 6] differ only by a constant.
    # Their normalized future function is one state; the additive five moves
    # into the monotone score coordinate, where it can participate in Pareto
    # dominance instead of multiplying discrete states.
    assert zero_state == five_state
    assert five_resources[0] - zero_resources[0] == 5
