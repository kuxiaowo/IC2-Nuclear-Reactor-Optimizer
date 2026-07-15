from __future__ import annotations

from itertools import product

from ic2_reactor.robdd import ROBDDManager
from ic2_reactor.robdd_bitvector import unsigned_equals_constant
from ic2_reactor.factorized_layout_dp import FactorizedLayoutFeasibilityDP
from ic2_reactor.symbolic_parameter_cut import (
    ROBDDLabelDomainCutAutomaton,
    ROBDDLayoutCutAutomaton,
    ROBDDSequentialParameterCut,
    structural_parameter_region_automaton,
)


def test_sequential_parameter_cut_merges_equal_residuals_and_counts_suffixes() -> None:
    groups = (
        (("cell", 0, 0), ("cell", 0, 1)),
        (("cell", 1, 0), ("cell", 1, 1)),
    )
    manager = ROBDDManager(tuple(variable for group in groups for variable in group))
    bits = tuple(
        tuple(manager.variable(variable) for variable in group)
        for group in groups
    )
    equal = manager.disjunction(*(
        manager.apply(
            "and",
            unsigned_equals_constant(manager, bits[0], code),
            unsigned_equals_constant(manager, bits[1], code),
        )
        for code in range(3)
    ))
    cut = ROBDDSequentialParameterCut(
        manager,
        manager.negate(equal),
        groups,
        (3, 3),
    )
    assert cut.completion_count(0, cut.initial_state) == 6
    for first in range(3):
        state = cut.transition(0, cut.initial_state, first)
        assert cut.completion_count(1, state) == 2
    for codes in product(range(3), repeat=2):
        assert cut.accepts(codes) == (codes[0] != codes[1])


def test_sequential_parameter_cut_rejects_invalid_multivalued_codes() -> None:
    group = (("choice", 0), ("choice", 1))
    manager = ROBDDManager(group)
    cut = ROBDDSequentialParameterCut(manager, 1, (group,), (3,))
    assert cut.completion_count(0, cut.initial_state) == 3
    assert cut.accepts((0,))
    assert cut.accepts((2,))
    assert not cut.accepts((3,))


def test_symbolic_region_cut_runs_inside_full_label_frontier_dp() -> None:
    groups = (
        (("cell", 0, 0), ("cell", 0, 1)),
        (("cell", 1, 0), ("cell", 1, 1)),
    )
    manager = ROBDDManager(tuple(variable for group in groups for variable in group))
    bits = tuple(
        tuple(manager.variable(variable) for variable in group)
        for group in groups
    )
    equal = manager.disjunction(*(
        manager.apply(
            "and",
            unsigned_equals_constant(manager, bits[0], code),
            unsigned_equals_constant(manager, bits[1], code),
        )
        for code in range(3)
    ))
    automaton = ROBDDLayoutCutAutomaton(ROBDDSequentialParameterCut(
        manager,
        manager.negate(equal),
        groups,
        (3, 3),
    ))
    result = FactorizedLayoutFeasibilityDP(
        (0, 1),
        ("A", "B", "C"),
        free_labels=("A", "B", "C"),
        automata=(automaton,),
    ).solve()
    assert result.proven and result.feasible
    assert result.layout == ("A", "B")


def test_structural_region_cut_supports_singletons_and_any_scan_order() -> None:
    groups = (
        (("cell", 0, 0),),
        (),
        (("cell", 2, 0), ("cell", 2, 1)),
    )
    manager = ROBDDManager((groups[0][0], *groups[2]))
    first_is_b = manager.variable(groups[0][0])
    third_bits = tuple(manager.variable(variable) for variable in groups[2])
    third_is_c = unsigned_equals_constant(manager, third_bits, 2)
    accepted = manager.apply("xor", first_is_b, third_is_c)
    automaton = structural_parameter_region_automaton(
        manager=manager,
        accepted_root=accepted,
        cell_parameter_variables=groups,
        cell_label_domains=(("A", "B"), ("C",), ("A", "B", "C")),
        global_labels=("A", "B", "C"),
        placement_order=(2, 1, 0),
    )
    assert isinstance(automaton, ROBDDLabelDomainCutAutomaton)
    result = FactorizedLayoutFeasibilityDP(
        (2, 1, 0),
        ("A", "B", "C"),
        free_labels=("A", "B", "C"),
        automata=(automaton,),
    ).solve()
    assert result.proven and result.feasible and result.layout is not None
    assert result.layout[1] == "C"
    assert (result.layout[0] == "B") != (result.layout[2] == "C")
