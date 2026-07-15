from __future__ import annotations

from itertools import product

from ic2_reactor.factorized_layout_dp import FactorizedLayoutFeasibilityDP
from ic2_reactor.robdd import ROBDDManager
from ic2_reactor.symbolic_parameter_optimization import (
    optimize_safe_frozen_parameters,
)
from ic2_reactor.symbolic_parameter_cut import (
    structural_parameter_region_automaton,
)
from ic2_reactor.symbolic_local_events import (
    SymbolicLocalEvent,
    local_event_failure_attractor,
    sequential_failure_preimage,
)


def assignment(p: bool, x: bool, y: bool) -> dict[str, bool]:
    return {"p": p, "x": x, "y": y}


def test_reverse_local_preimage_matches_every_explicit_event_execution() -> None:
    manager = ROBDDManager(("p", "x", "y"))
    p = manager.variable("p")
    x = manager.variable("x")
    y = manager.variable("y")
    events = (
        SymbolicLocalEvent.from_mapping(
            "conditional-xor",
            {"x": manager.apply("xor", x, p)},
        ),
        SymbolicLocalEvent.from_mapping(
            "failing-y-flip",
            {"y": manager.negate(y)},
            manager.apply("and", x, y),
        ),
    )
    target = manager.apply("and", x, y)
    predecessor = sequential_failure_preimage(
        manager,
        events,
        target,
        frozen_variables=("p",),
    )

    for p_value, x_value, y_value in product((False, True), repeat=3):
        after_x = x_value != p_value
        failed = after_x and y_value
        after_y = not y_value
        expected = failed or (after_x and after_y)
        assert manager.evaluate(
            predecessor,
            assignment(p_value, x_value, y_value),
        ) == expected


def test_local_event_attractor_matches_explicit_finite_graph() -> None:
    manager = ROBDDManager(("p", "x", "y"))
    p = manager.variable("p")
    x = manager.variable("x")
    y = manager.variable("y")
    events = (
        SymbolicLocalEvent.from_mapping(
            "conditional-xor",
            {"x": manager.apply("xor", x, p)},
        ),
        SymbolicLocalEvent.from_mapping(
            "failing-y-flip",
            {"y": manager.negate(y)},
            manager.apply("and", x, y),
        ),
    )
    bad = manager.apply("and", x, y)
    proof = local_event_failure_attractor(
        manager,
        events,
        bad,
        frozen_variables=("p",),
    )
    assert proof.verify(manager)

    def explicitly_fails(p_value: bool, x_value: bool, y_value: bool) -> bool:
        seen = set()
        state = (x_value, y_value)
        while state not in seen:
            seen.add(state)
            current_x, current_y = state
            if current_x and current_y:
                return True
            current_x = current_x != p_value
            if current_x and current_y:
                return True
            current_y = not current_y
            state = (current_x, current_y)
        return False

    for p_value, x_value, y_value in product((False, True), repeat=3):
        observed = manager.evaluate(
            proof.attractor_root,
            assignment(p_value, x_value, y_value),
        )
        assert observed == explicitly_fails(p_value, x_value, y_value)

    optimum = optimize_safe_frozen_parameters(
        manager,
        proof,
        parameter_variables=("p",),
        dynamic_variables=("x", "y"),
        initial_dynamic_assignment={"x": False, "y": False},
        objective_bits=(p,),
    )
    assert optimum.optimum_value == 0
    assert dict(optimum.witness or ()) == {"p": False}

    automaton = structural_parameter_region_automaton(
        manager=manager,
        accepted_root=optimum.feasible_parameter_root,
        cell_parameter_variables=(("p",), ()),
        cell_label_domains=(("A", "B"), ("C",)),
        global_labels=("A", "B", "C"),
        placement_order=(1, 0),
    )
    outer = FactorizedLayoutFeasibilityDP(
        (1, 0),
        ("A", "B", "C"),
        free_labels=("A", "B", "C"),
        automata=(automaton,),
    ).solve()
    assert outer.proven and outer.feasible
    assert outer.layout == ("A", "C")
