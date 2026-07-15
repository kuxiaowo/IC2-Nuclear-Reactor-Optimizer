from __future__ import annotations

from itertools import product
from random import Random

from ic2_reactor.terminal_cut_quotient import (
    TerminalCutSignature,
    directed_network_terminal_cut_signature,
    terminal_cut_frontier_orders,
)
from ic2_reactor.thermal_relaxation import _maximum_flow_with_cut
from ic2_reactor.thermal_relaxation import HeatFlowComponent, layout_heat_flow_bound
from ic2_reactor.thermal_terminal_cut import (
    AverageFlowTerminalCutAutomaton,
    average_flow_terminal_factor_scopes,
    average_flow_terminal_schedule_profile,
    eliminated_average_flow_minimum_cut,
)
from ic2_reactor.factorized_layout_dp import FactorizedLayoutFeasibilityDP
from ic2_reactor.factorized_cooling_master import FactorizedCoolingCutMaster
from ic2_reactor.mathematical_model import evaluate_power_skeleton
from ic2_reactor.pareto_frontier_dp import _StateOrderedContinuationTable
from ic2_reactor.state_quotient import ParetoPoint
from ic2_reactor.thermal_terminal_cut import AverageFlowTerminalCutState
from ic2_reactor.frontier_automata import rectangular_frontier_order
from ic2_reactor.mathematical_model import Graph, PowerComponent, ReactorProblem


def constrained_cut_values(
    node_count: int,
    edges: tuple[tuple[int, int, int], ...],
    source: int,
    sink: int,
    terminals: tuple[int, ...],
    saturation: int | None = None,
) -> tuple[int, ...]:
    internal = tuple(
        node
        for node in range(node_count)
        if node not in {source, sink, *terminals}
    )
    result = []
    for terminal_mask in range(1 << len(terminals)):
        best = None
        fixed = {
            terminal: bool(terminal_mask >> position & 1)
            for position, terminal in enumerate(terminals)
        }
        for internal_sides in product((False, True), repeat=len(internal)):
            sides = {source: True, sink: False, **fixed}
            sides.update(dict(zip(internal, internal_sides, strict=True)))
            capacity = sum(
                value
                for start, end, value in edges
                if sides[start] and not sides[end]
            )
            best = capacity if best is None else min(best, capacity)
        assert best is not None
        result.append(best if saturation is None else min(saturation, best))
    return tuple(result)


def test_terminal_cut_signature_matches_every_constrained_partition() -> None:
    rng = Random(7)
    for node_count in range(3, 7):
        source, sink = node_count - 2, node_count - 1
        for _sample in range(20):
            edges = tuple(
                (start, end, rng.randrange(6))
                for start in range(node_count)
                for end in range(node_count)
                if start != end and rng.random() < 0.25
            )
            terminals = tuple(range(min(2, node_count - 2)))
            signature = directed_network_terminal_cut_signature(
                node_count,
                edges,
                source=source,
                sink=sink,
                terminals=terminals,
            )
            assert signature.values == constrained_cut_values(
                node_count,
                edges,
                source,
                sink,
                terminals,
            )


def test_gluing_and_forgetting_equal_direct_network_elimination() -> None:
    saturation = 8
    first = (
        TerminalCutSignature.zero(("a", "b"), saturation=saturation)
        .add_from_fixed_source("a", 5)
        .add_directed_edge("a", "b", 3)
    )
    second = (
        TerminalCutSignature.zero(("a", "b"), saturation=saturation)
        .add_to_fixed_sink("a", 1)
        .add_to_fixed_sink("b", 4)
    )
    glued = first.combine(second).forget("a").forget("b")
    edges = ((2, 0, 5), (0, 1, 3), (0, 3, 1), (1, 3, 4))
    direct = directed_network_terminal_cut_signature(
        4,
        edges,
        source=2,
        sink=3,
        saturation=saturation,
    )
    assert glued == direct
    assert glued.minimum_cut == 4


def test_saturation_preserves_the_required_flow_decision() -> None:
    edges = ((3, 0, 9), (0, 1, 7), (1, 4, 6), (0, 2, 3), (2, 4, 3))
    flow, _cut = _maximum_flow_with_cut(5, edges, 3, 4)
    assert flow == 9
    for threshold in range(1, 12):
        signature = directed_network_terminal_cut_signature(
            5,
            edges,
            source=3,
            sink=4,
            saturation=threshold,
        )
        assert (signature.minimum_cut >= threshold) == (flow >= threshold)


def test_cut_factor_and_condition_are_exact() -> None:
    signature = (
        TerminalCutSignature.zero(("x", "y"), saturation=10)
        .add_factor(("x", "y"), (1, 3, 5, 7))
        .add_directed_edge("x", "y", 4)
    )
    # Masks: 00, 01, 10, 11.  The edge crosses only at x=1,y=0.
    assert signature.values == (1, 7, 5, 7)
    assert signature.condition({"x": True}).values == (7, 7)


def test_generator_elimination_matches_original_average_flow_network() -> None:
    problem = ReactorProblem(
        graph=Graph.rectangular(2, 3),
        rod_budget=2,
        exact_rods=False,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("empty", "vent", "side", "exchange"),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "vent": HeatFlowComponent(accepts_heat=True, self_vent=2, hull_draw=1),
        "side": HeatFlowComponent(side_vent=1),
        "exchange": HeatFlowComponent(
            accepts_heat=True,
            exchange_side=2,
            exchange_hull=1,
        ),
    }
    rng = Random(19)
    labels = tuple(catalogue)
    for _sample in range(40):
        layout = tuple(rng.choice(labels) for _ in problem.graph.vertices)
        expected = layout_heat_flow_bound(problem, layout, catalogue)
        generated, minimum_cut = eliminated_average_flow_minimum_cut(
            problem,
            layout,
            catalogue,
        )
        assert generated == expected.generated_heat
        assert minimum_cut == expected.maximum_removable_heat


def test_average_flow_schedule_profile_is_structural_only() -> None:
    problem = ReactorProblem(
        graph=Graph.rectangular(2, 3),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("empty", "vent"),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "vent": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    profile = average_flow_terminal_schedule_profile(
        problem,
        catalogue,
        tuple(catalogue),
        rectangular_frontier_order(problem.graph),
    )
    assert profile.factor_count > 0
    assert profile.distinct_cut_terminals > 0
    assert profile.peak_live_terminals <= problem.graph.size + 1
    assert profile.peak_cut_vector_entries == 1 << profile.peak_live_terminals
    assert profile.coarse_full_scan_value_operations_bound > 0
    factors = average_flow_terminal_factor_scopes(
        problem,
        catalogue,
        tuple(catalogue),
    )
    orders = terminal_cut_frontier_orders(
        problem.graph.vertices,
        factors,
        beam_width=8,
    )
    assert orders
    assert all(set(order) == set(problem.graph.vertices) for order in orders)


def test_terminal_cut_automaton_enforces_all_average_cuts_at_once() -> None:
    problem = ReactorProblem(
        graph=Graph.rectangular(2, 2),
        rod_budget=4,
        exact_rods=False,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
        ),
        cooling_components=(),
        layout_components=("empty", "vent", "side", "exchange"),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "vent": HeatFlowComponent(accepts_heat=True, self_vent=2, hull_draw=1),
        "side": HeatFlowComponent(side_vent=1),
        "exchange": HeatFlowComponent(
            accepts_heat=True,
            exchange_side=2,
            exchange_hull=1,
        ),
    }
    labels = tuple(catalogue)
    order = rectangular_frontier_order(problem.graph)
    automaton = AverageFlowTerminalCutAutomaton(
        problem,
        catalogue,
        labels,
        placement_order=order,
    )
    assert automaton.profile.schedule.factor_count == problem.graph.size
    for layout in product(labels, repeat=problem.graph.size):
        expected = layout_heat_flow_bound(
            problem,
            layout,
            catalogue,
        ).necessary_condition_satisfied
        result = FactorizedLayoutFeasibilityDP(
            order,
            labels,
            free_labels=labels,
            fixed_labels={
                vertex: label for vertex, label in enumerate(layout)
            },
            automata=(automaton,),
        ).solve()
        assert result.proven
        assert result.feasible == expected


def test_joint_layout_master_can_use_complete_average_flow_quotient() -> None:
    problem = ReactorProblem(
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
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    master = FactorizedCoolingCutMaster(problem, catalogue)
    result = master.solve_joint_layouts(enforce_complete_average_flow=True)
    assert result.proven
    assert master.last_order_selection.complete_average_flow
    assert result.frontier
    power_ids = {item.id for item in problem.power_components}
    feasible = []
    for layout in product(tuple(catalogue), repeat=problem.graph.size):
        skeleton = tuple(
            label if label in power_ids else "empty" for label in layout
        )
        metrics = evaluate_power_skeleton(problem, skeleton)
        if metrics.rods != problem.rod_budget:
            continue
        if layout_heat_flow_bound(
            problem,
            layout,
            catalogue,
        ).necessary_condition_satisfied:
            feasible.append((metrics.power, layout))
    assert max(item.power for item in result.frontier) == max(
        power for power, _layout in feasible
    )
    assert all(
        layout_heat_flow_bound(
            problem,
            item.skeleton,
            catalogue,
        ).necessary_condition_satisfied
        for item in result.frontier
    )


def test_pointwise_stronger_terminal_cut_signature_dominates() -> None:
    class OrderedSignatureAutomaton:
        state_dominance_key = staticmethod(
            AverageFlowTerminalCutAutomaton.state_dominance_key
        )
        state_dominance_coordinates = staticmethod(
            AverageFlowTerminalCutAutomaton.state_dominance_coordinates
        )

    table = _StateOrderedContinuationTable((OrderedSignatureAutomaton(),))
    stronger = AverageFlowTerminalCutState(
        (),
        TerminalCutSignature(("x",), (3, 4), saturation=5),
    )
    weaker = AverageFlowTerminalCutState(
        (),
        TerminalCutSignature(("x",), (2, 4), saturation=5),
    )
    point = ParetoPoint(10, 2, (3,), (0,))
    assert table.insert(((0,), 1, (stronger,)), point)
    assert not table.insert(((0,), 1, (weaker,)), point)
    assert table.point_count == 1
