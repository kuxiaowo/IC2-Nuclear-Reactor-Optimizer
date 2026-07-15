from __future__ import annotations

from itertools import product

from ic2_reactor.ic2_symbolic_layout import compile_ic2_symbolic_layout_circuit
from ic2_reactor.mathematical_model import (
    evaluate_power_skeleton,
    ic2_mark_i_problem,
)
from ic2_reactor.symbolic_parameter_optimization import (
    maximize_unsigned_boolean_objective,
)


def test_symbolic_layout_circuit_matches_all_small_static_layouts() -> None:
    labels = (
        "empty",
        "uranium_single",
        "uranium_dual",
        "uranium_quad",
        "iridium_reflector",
    )
    exact_rods = 5
    circuit = compile_ic2_symbolic_layout_circuit(
        rows=2,
        columns=2,
        exact_rods=exact_rods,
        labels=labels,
    )
    problem = ic2_mark_i_problem(
        rows=2,
        columns=2,
        rod_budget=exact_rods,
        enabled_components=set(labels),
    )
    feasible_metrics = []
    for layout in product(labels, repeat=4):
        assignment = circuit.assignment(layout)
        metrics = evaluate_power_skeleton(problem, layout)
        assert circuit.manager.evaluate(circuit.valid_labels_root, assignment)
        assert circuit.manager.evaluate(circuit.rod_budget_root, assignment) == (
            metrics.rods == exact_rods
        )
        assert circuit.value(circuit.rod_count_bits, assignment) == metrics.rods
        assert circuit.value(circuit.pulse_unit_bits, assignment) == metrics.pulse_units
        assert circuit.value(circuit.power_bits, assignment) == metrics.power
        assert circuit.value(circuit.generated_heat_bits, assignment) == metrics.generated_heat
        if metrics.rods == exact_rods:
            feasible_metrics.append(metrics)

    optimum, optimum_root = maximize_unsigned_boolean_objective(
        circuit.manager,
        circuit.feasible_layout_root,
        circuit.power_bits,
    )
    assert optimum == max(item.power for item in feasible_metrics)
    witness = circuit.manager.satisfying_assignment(optimum_root)
    assert witness is not None
    assert circuit.value(circuit.power_bits, witness) == optimum


def test_invalid_binary_component_code_is_rejected() -> None:
    circuit = compile_ic2_symbolic_layout_circuit(
        rows=1,
        columns=1,
        exact_rods=0,
        labels=("empty", "uranium_single", "uranium_dual"),
    )
    variables = circuit.layout_variables[0]
    invalid = {variable: True for variable in variables}
    assert not circuit.manager.evaluate(circuit.valid_labels_root, invalid)
