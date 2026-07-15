from __future__ import annotations

from itertools import product

from ic2_reactor.ic2_dynamic_schema import (
    IC2DynamicStateSchema,
    ic2_layout_dynamic_schema_signature,
    ic2_layout_structural_signature,
)
from ic2_reactor.ic2_schema_parameterization import (
    compile_ic2_schema_conditioned_layout_circuit,
)
from ic2_reactor.components import COMPONENTS


def test_schema_conditioned_static_circuit_matches_every_small_refinement() -> None:
    schema = IC2DynamicStateSchema(0, 0)
    circuit = compile_ic2_schema_conditioned_layout_circuit(
        rows=2,
        columns=2,
        schemas=(schema,) * 4,
        exact_rods=1,
    )
    total = {
        variable: False for variable in circuit.manager.variables
    }
    for layout in product(circuit.label_domains[0], repeat=4):
        assignment = circuit.assignment(layout)
        total.update(assignment)
        active = tuple(
            COMPONENTS[label].kind in {"fuel", "reflector"}
            for label in layout
        )
        neighbours = ((1, 2), (0, 3), (0, 3), (1, 2))
        degrees = tuple(
            sum(active[target] for target in adjacent)
            for adjacent in neighbours
        )
        rods = power = generated_heat = 0
        for label, degree in zip(layout, degrees, strict=True):
            spec = COMPONENTS[label]
            rods += spec.rod_count
            if spec.kind == "fuel":
                pulses = spec.internal_pulses + degree
                power += 5 * spec.rod_count * pulses
                generated_heat += 2 * spec.rod_count * pulses * (pulses + 1)
        assert circuit.manager.evaluate(circuit.valid_parameter_root, total)
        assert circuit.manager.evaluate(circuit.rod_budget_root, total) == (
            rods == 1
        )
        assert circuit.value(circuit.rod_count_bits, assignment) == rods
        assert circuit.value(circuit.power_bits, assignment) == power
        assert circuit.value(
            circuit.generated_heat_bits,
            assignment,
        ) == generated_heat


def test_schema_conditioning_uses_no_parameter_bits_for_singleton_domains() -> None:
    layout = (
        "empty",
        "heat_vent",
        "heat_exchanger",
        "coolant_30k",
    )
    schemas = ic2_layout_dynamic_schema_signature(layout)
    circuit = compile_ic2_schema_conditioned_layout_circuit(
        rows=2,
        columns=2,
        schemas=schemas,
        exact_rods=0,
    )
    assert tuple(map(len, circuit.label_domains)) == (6, 4, 1, 1)
    assert tuple(map(len, circuit.cell_parameter_variables)) == (3, 2, 0, 0)
    assert len(circuit.parameter_variables) == 5
    assert circuit.assignment(layout).keys() == set(circuit.parameter_variables)
    invalid = {
        variable: True
        for variable in circuit.manager.variables
    }
    assert not circuit.manager.evaluate(circuit.valid_parameter_root, invalid)


def test_joint_structural_conditioning_reduces_each_inner_domain_to_four() -> None:
    layout = (
        "empty",
        "heat_vent",
        "reactor_heat_exchanger",
        "coolant_10k",
    )
    circuit = compile_ic2_schema_conditioned_layout_circuit(
        rows=2,
        columns=2,
        structural_signatures=ic2_layout_structural_signature(layout),
        exact_rods=0,
    )
    assert tuple(map(len, circuit.label_domains)) == (2, 4, 2, 2)
    assert tuple(map(len, circuit.cell_parameter_variables)) == (1, 2, 1, 1)
    assert max(map(len, circuit.cell_parameter_variables)) == 2
    assert circuit.structural_signatures is not None
    assert circuit.assignment(layout).keys() == set(circuit.parameter_variables)
