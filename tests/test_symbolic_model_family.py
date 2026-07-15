from __future__ import annotations

from ic2_reactor.ic2_symbolic_no_exchange import (
    compile_ic2_no_exchange_symbolic_model,
)
from ic2_reactor.symbolic_model_family import merge_symbolic_ic2_model_family


def test_actual_ic2_layout_family_directly_selects_best_infinite_safe_power() -> None:
    suffix = ("advanced_heat_vent", *("empty",) * 16)
    layouts = (
        ("empty", *suffix),
        ("uranium_single", *suffix),
        ("uranium_dual", *suffix),
        ("uranium_quad", *suffix),
    )
    # With no pulse-active neighbour: powers are 0, 5, 20 and 60 EU/t.
    # The advanced vent removes 12/t; only empty and single are safe forever.
    family = merge_symbolic_ic2_model_family(
        tuple(compile_ic2_no_exchange_symbolic_model(layout) for layout in layouts),
        (0, 5, 20, 60),
    )
    proof, optimum = family.prove_optimum()
    assert proof.verify(family.manager, family.next_functions)
    assert optimum.optimum_value == 5
    assert optimum.safe_initial_parameter_count == 2
    assert optimum.feasible_parameter_count == 2
    assert optimum.optimum_parameter_count == 1
    assert family.witness_layout(optimum) == layouts[1]

    compacted, compacted_optimum = family.prove_optimum_compacting()
    assert compacted.compactions > 0
    assert compacted.proof.verify(compacted.manager, compacted.next_functions)
    assert compacted_optimum.optimum_value == 5
    assert family.witness_layout(compacted_optimum) == layouts[1]

    forward, forward_value, forward_root = family.prove_optimum_forward()
    assert forward.verify(
        family.manager,
        {
            variable: family.next_functions[variable]
            for variable in family.dynamic_variables
        },
    )
    assert forward.complete
    assert forward_value == 5
    assert family.manager.model_count(
        forward.safe_parameter_root,
        family.parameter_variables,
    ) == 2
    assert family.manager.model_count(
        forward_root,
        family.parameter_variables,
    ) == 1
