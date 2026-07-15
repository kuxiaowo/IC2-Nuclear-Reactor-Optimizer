from __future__ import annotations

from random import Random

from ic2_reactor.finite_safety_fixpoint import finite_failure_attractor
from ic2_reactor.robdd import ROBDDManager
from ic2_reactor.symbolic_parameter_optimization import (
    optimize_safe_frozen_parameters,
)
from ic2_reactor.symbolic_safety import symbolic_failure_attractor


def bit_assignment(value: int, variables: tuple[str, ...]) -> dict[str, bool]:
    return {
        variable: bool(value >> index & 1)
        for index, variable in enumerate(variables)
    }


def boolean_function_from_table(
    manager: ROBDDManager,
    variables: tuple[str, ...],
    values: tuple[bool, ...],
) -> int:
    return manager.from_assignments(tuple(
        bit_assignment(value, variables)
        for value, enabled in enumerate(values)
        if enabled
    ))


def test_robdd_is_canonical_and_quantification_is_exact() -> None:
    manager = ROBDDManager(("x", "y"))
    x = manager.variable("x")
    y = manager.variable("y")
    rebuilt_x = manager.disjunction(
        manager.conjunction(x, y),
        manager.conjunction(x, manager.negate(y)),
    )
    assert rebuilt_x == x
    assert manager.existential(manager.conjunction(x, y), ("y",)) == x
    xor = manager.apply("xor", x, y)
    swapped = manager.apply("xor", y, x)
    assert xor == swapped


def test_robdd_root_compaction_preserves_functions_and_drops_transients() -> None:
    manager = ROBDDManager(("x", "y", "z"))
    x = manager.variable("x")
    y = manager.variable("y")
    z = manager.variable("z")
    live = manager.apply("xor", x, y)
    _transient = manager.apply("and", manager.apply("or", x, z), y)
    compacted, (following,) = manager.compact_roots((live,))
    assert compacted.allocated_node_count == manager.reachable_node_count(live)
    assert compacted.allocated_node_count < manager.allocated_node_count
    for value in range(8):
        assignment = bit_assignment(value, ("x", "y", "z"))
        assert compacted.evaluate(following, assignment) == manager.evaluate(
            live,
            assignment,
        )


def test_robdd_canonical_forest_key_is_exact_across_managers() -> None:
    first = ROBDDManager(("x", "y", "z"))
    first_xor = first.apply("xor", first.variable("x"), first.variable("y"))
    first_and = first.apply("and", first_xor, first.variable("z"))
    _manager, _roots, first_key = first.canonicalize_roots(
        (first_xor, first_and)
    )

    second = ROBDDManager(("x", "y", "z"))
    # Deliberately construct unrelated garbage and the live functions in a
    # different operation order; the exact forest key must ignore both.
    _garbage = second.apply("or", second.variable("x"), second.variable("z"))
    second_xor = second.apply("xor", second.variable("y"), second.variable("x"))
    second_and = second.apply("and", second.variable("z"), second_xor)
    _manager, _roots, second_key = second.canonicalize_roots(
        (second_xor, second_and)
    )
    assert second_key == first_key

    _manager, _roots, unequal_key = second.canonicalize_roots(
        (second_and, second_xor)
    )
    assert unequal_key != first_key


def test_robdd_import_preserves_functions_with_prefixed_parameters() -> None:
    source = ROBDDManager(("x", "y"))
    root = source.apply("xor", source.variable("x"), source.variable("y"))
    target = ROBDDManager(("parameter", "x", "y"))
    (imported,) = target.import_roots(source, (root,))
    for value in range(8):
        target_assignment = bit_assignment(value, ("parameter", "x", "y"))
        source_assignment = {
            "x": target_assignment["x"],
            "y": target_assignment["y"],
        }
        assert target.evaluate(imported, target_assignment) == source.evaluate(
            root,
            source_assignment,
        )


def test_robdd_constant_restriction_is_exact() -> None:
    manager = ROBDDManager(("x", "y", "z"))
    root = manager.apply(
        "xor",
        manager.variable("x"),
        manager.apply("and", manager.variable("y"), manager.variable("z")),
    )
    restricted = manager.restrict(root, {"y": True, "z": False})
    assert restricted == manager.variable("x")
    assert manager.model_count(root) == 4
    assert manager.model_count(restricted, ("x", "y", "z")) == 4


def test_partitioned_symbolic_fixed_point_matches_explicit_graph() -> None:
    variables = ("s0", "s1")
    manager = ROBDDManager(variables)
    # 0 -> 1 -> bad(3), while 2 -> 2 is permanently safe.
    following = (1, 3, 2, 3)
    next_functions = {
        variable: boolean_function_from_table(
            manager,
            variables,
            tuple(bool(target >> bit & 1) for target in following),
        )
        for bit, variable in enumerate(variables)
    }
    bad = manager.cube(bit_assignment(3, variables))
    proof = symbolic_failure_attractor(
        manager,
        variables,
        next_functions,
        bad,
    )
    assert proof.verify(manager, next_functions)
    assert tuple(
        value
        for value in range(4)
        if manager.evaluate(proof.attractor_root, bit_assignment(value, variables))
    ) == (0, 1, 3)
    assert manager.evaluate(proof.safe_invariant_root, bit_assignment(2, variables))


def test_empty_bad_set_has_empty_failure_attractor_certificate() -> None:
    manager = ROBDDManager(("state",))
    state = manager.variable("state")
    proof = symbolic_failure_attractor(
        manager,
        ("state",),
        {"state": state},
        0,
    )
    assert proof.rank_layer_roots == ()
    assert proof.attractor_root == 0
    assert proof.safe_invariant_root == 1
    assert proof.verify(manager, {"state": state})


def test_symbolic_fixed_point_matches_random_explicit_deterministic_maps() -> None:
    rng = Random(67)
    for bit_count in range(1, 5):
        variables = tuple(f"s{bit}" for bit in range(bit_count))
        state_count = 1 << bit_count
        for _sample in range(30):
            following = tuple(rng.randrange(state_count) for _ in range(state_count))
            bad_states = tuple(
                state for state in range(state_count) if rng.random() < 0.25
            ) or (rng.randrange(state_count),)
            explicit = finite_failure_attractor(
                tuple(range(state_count)),
                {
                    state: following[state]
                    for state in range(state_count)
                    if state not in bad_states
                },
                bad_states,
            )
            manager = ROBDDManager(variables)
            functions = {
                variable: boolean_function_from_table(
                    manager,
                    variables,
                    tuple(
                        bool(following[state] >> bit & 1)
                        for state in range(state_count)
                    ),
                )
                for bit, variable in enumerate(variables)
            }
            bad_root = manager.from_assignments(tuple(
                bit_assignment(state, variables) for state in bad_states
            ))
            symbolic = symbolic_failure_attractor(
                manager,
                variables,
                functions,
                bad_root,
            )
            symbolic_attractor = {
                state
                for state in range(state_count)
                if manager.evaluate(
                    symbolic.attractor_root,
                    bit_assignment(state, variables),
                )
            }
            assert symbolic_attractor == set(explicit.failure_attractor)


def test_frozen_parameter_fixed_point_directly_maximizes_safe_objective() -> None:
    variables = ("p0", "p1", "heat")
    manager = ROBDDManager(variables)
    p0 = manager.variable("p0")
    p1 = manager.variable("p1")
    heat = manager.variable("heat")
    # p0 injects heat forever; p1 is harmless.  Identity next functions make
    # both parameter bits frozen, so one fixed point proves all four choices.
    next_functions = {
        "p0": p0,
        "p1": p1,
        "heat": manager.apply("or", heat, p0),
    }
    proof = symbolic_failure_attractor(
        manager,
        variables,
        next_functions,
        heat,
    )
    optimum = optimize_safe_frozen_parameters(
        manager,
        proof,
        parameter_variables=("p0", "p1"),
        dynamic_variables=("heat",),
        initial_dynamic_assignment={"heat": False},
        parameter_constraint_root=manager.apply("or", p0, p1),
        objective_bits=(p0, p1),
    )
    assert optimum.optimum_value == 2
    assert dict(optimum.witness or ()) == {"p0": False, "p1": True}
    assert optimum.verify(manager, proof)


def test_parameter_objective_rejects_dynamic_support() -> None:
    manager = ROBDDManager(("parameter", "state"))
    parameter = manager.variable("parameter")
    state = manager.variable("state")
    proof = symbolic_failure_attractor(
        manager,
        ("parameter", "state"),
        {"parameter": parameter, "state": state},
        state,
    )
    try:
        optimize_safe_frozen_parameters(
            manager,
            proof,
            parameter_variables=("parameter",),
            dynamic_variables=("state",),
            initial_dynamic_assignment={"state": False},
            objective_bits=(state,),
        )
    except ValueError as error:
        assert "objective depends" in str(error)
    else:  # pragma: no cover
        raise AssertionError("dynamic objective support must be rejected")


def test_random_frozen_parameter_optima_match_explicit_enumeration() -> None:
    rng = Random(113)
    variables = ("p0", "p1", "s0", "s1")
    for _sample in range(40):
        manager = ROBDDManager(variables)
        following_dynamic = {
            parameter: tuple(rng.randrange(4) for _state in range(4))
            for parameter in range(4)
        }
        bad_table = tuple(rng.random() < 0.2 for _combined in range(16))
        constraint = tuple(rng.random() < 0.8 for _parameter in range(4))
        objective = tuple(rng.randrange(8) for _parameter in range(4))

        next_functions = {
            "p0": manager.variable("p0"),
            "p1": manager.variable("p1"),
        }
        for bit, variable in enumerate(("s0", "s1")):
            next_functions[variable] = manager.from_truth_table(tuple(
                bool(following_dynamic[combined & 3][combined >> 2] >> bit & 1)
                for combined in range(16)
            ))
        proof = symbolic_failure_attractor(
            manager,
            variables,
            next_functions,
            manager.from_truth_table(bad_table),
        )
        constraint_root = manager.from_truth_table(tuple(
            constraint[combined & 3] for combined in range(16)
        ))
        objective_bits = tuple(
            manager.from_truth_table(tuple(
                bool(objective[combined & 3] >> bit & 1)
                for combined in range(16)
            ))
            for bit in range(3)
        )
        result = optimize_safe_frozen_parameters(
            manager,
            proof,
            parameter_variables=("p0", "p1"),
            dynamic_variables=("s0", "s1"),
            initial_dynamic_assignment={"s0": False, "s1": False},
            parameter_constraint_root=constraint_root,
            objective_bits=objective_bits,
        )

        safe_parameters = []
        for parameter in range(4):
            dynamic = 0
            visited = set()
            safe = True
            while dynamic not in visited:
                combined = parameter | (dynamic << 2)
                if bad_table[combined]:
                    safe = False
                    break
                visited.add(dynamic)
                dynamic = following_dynamic[parameter][dynamic]
            assignment = bit_assignment(parameter, ("p0", "p1"))
            assignment.update({"s0": False, "s1": False})
            assert manager.evaluate(
                result.safe_initial_parameter_root,
                assignment,
            ) == safe
            if safe and constraint[parameter]:
                safe_parameters.append(parameter)
        expected = (
            max(objective[item] for item in safe_parameters)
            if safe_parameters
            else None
        )
        assert result.optimum_value == expected
        if expected is not None:
            witness = dict(result.witness or ())
            parameter = int(witness["p0"]) | (int(witness["p1"]) << 1)
            assert parameter in safe_parameters
            assert objective[parameter] == expected
