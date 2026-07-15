from __future__ import annotations

from random import Random

from ic2_reactor.robdd import ROBDDManager
from ic2_reactor.symbolic_forward_trajectory import (
    symbolic_forward_trajectory_safety,
)


def _assignment(code: int, variables: tuple[str, ...]) -> dict[str, bool]:
    return {
        variable: bool(code >> bit & 1)
        for bit, variable in enumerate(variables)
    }


def test_forward_relation_classifies_failure_and_safe_cycles_together() -> None:
    parameters = ("p",)
    dynamics = ("s0", "s1")
    manager = ROBDDManager((*parameters, *dynamics))
    p = manager.variable("p")
    s0 = manager.variable("s0")
    s1 = manager.variable("s1")

    # p=0: 00 -> 00, hence safe.  p=1: 00 -> 01 -> 11, hence bad.
    proof = symbolic_forward_trajectory_safety(
        manager,
        parameter_variables=parameters,
        dynamic_variables=dynamics,
        next_functions={
            "s0": p,
            "s1": manager.apply("and", p, s0),
        },
        bad_root=manager.apply("and", s0, s1),
        initial_dynamic_assignment={"s0": False, "s1": False},
    )

    assert proof.complete
    assert proof.inspected_steps == 3
    assert manager.model_count(proof.safe_parameter_root, parameters) == 1
    assert manager.model_count(proof.failed_parameter_root, parameters) == 1
    # Only (p=0,00), (p=1,00) and (p=1,01) were stored.  The other five
    # parameter/state pairs in the full cube were never explored.
    assert manager.model_count(
        proof.visited_graph_root,
        (*parameters, *dynamics),
    ) == 3


def test_forward_cutoff_preserves_the_unresolved_parameter_region() -> None:
    parameters = ("p",)
    dynamics = ("s",)
    manager = ROBDDManager((*parameters, *dynamics))
    proof = symbolic_forward_trajectory_safety(
        manager,
        parameter_variables=parameters,
        dynamic_variables=dynamics,
        next_functions={"s": manager.variable("p")},
        bad_root=manager.variable("s"),
        initial_dynamic_assignment={"s": False},
        maximum_steps=1,
    )
    assert not proof.complete
    assert proof.safe_parameter_root == 0
    assert proof.failed_parameter_root == 0
    assert proof.unknown_parameter_root == 1


def test_random_parameterized_maps_match_explicit_orbit_simulation() -> None:
    rng = Random(942_701)
    variables = ("p0", "p1", "s0", "s1")
    parameters = variables[:2]
    dynamics = variables[2:]

    for _case in range(40):
        manager = ROBDDManager(variables)
        bad_values = tuple(bool(rng.getrandbits(1)) for _ in range(16))
        next_values = tuple(
            tuple(bool(rng.getrandbits(1)) for _ in range(16))
            for _bit in dynamics
        )
        bad = manager.from_truth_table(bad_values)
        next_functions = {
            variable: manager.from_truth_table(values)
            for variable, values in zip(dynamics, next_values, strict=True)
        }
        initial = {variable: bool(rng.getrandbits(1)) for variable in dynamics}
        valid_parameter_codes = tuple(
            code for code in range(4) if rng.getrandbits(1)
        )
        constraint = manager.from_assignments(tuple(
            _assignment(code, parameters) for code in valid_parameter_codes
        ))

        proof = symbolic_forward_trajectory_safety(
            manager,
            parameter_variables=parameters,
            dynamic_variables=dynamics,
            next_functions=next_functions,
            bad_root=bad,
            initial_dynamic_assignment=initial,
            parameter_constraint_root=constraint,
        )
        assert proof.complete

        for parameter_code in range(4):
            parameter_assignment = _assignment(parameter_code, parameters)
            if parameter_code not in valid_parameter_codes:
                expected = None
            else:
                state = sum(
                    int(initial[variable]) << bit
                    for bit, variable in enumerate(dynamics)
                )
                seen: set[int] = set()
                while True:
                    full_code = parameter_code | state << len(parameters)
                    if bad_values[full_code]:
                        expected = "failed"
                        break
                    if state in seen:
                        expected = "safe"
                        break
                    seen.add(state)
                    state = sum(
                        int(values[full_code]) << bit
                        for bit, values in enumerate(next_values)
                    )

            total = {
                **parameter_assignment,
                **{variable: False for variable in dynamics},
            }
            observed_safe = manager.evaluate(proof.safe_parameter_root, total)
            observed_failed = manager.evaluate(proof.failed_parameter_root, total)
            assert observed_safe == (expected == "safe")
            assert observed_failed == (expected == "failed")
