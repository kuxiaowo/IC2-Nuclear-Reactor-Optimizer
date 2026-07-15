from __future__ import annotations

from ic2_reactor.robdd import ROBDDManager
from ic2_reactor.robdd_bitvector import (
    constant_bits,
    select_bits,
    signed_case_sum_bits,
    unsigned_add,
    unsigned_add_constant,
    unsigned_at_least_constant,
    unsigned_equals_constant,
    unsigned_subtract_constant_floor_zero,
    unsigned_lookup,
)


def test_unsigned_bitvector_circuits_match_integers() -> None:
    width = 6
    variables = tuple(f"b{bit}" for bit in range(width))
    manager = ROBDDManager(variables)
    bits = tuple(manager.variable(variable) for variable in variables)
    for constant in (0, 1, 7, 31, 64, 79):
        result, overflow = unsigned_add_constant(manager, bits, constant)
        for value in range(1 << width):
            assignment = {
                variable: bool(value >> bit & 1)
                for bit, variable in enumerate(variables)
            }
            actual = sum(
                int(manager.evaluate(root, assignment)) << bit
                for bit, root in enumerate(result)
            )
            assert actual == (value + constant) % (1 << width)
            assert manager.evaluate(overflow, assignment) == (
                value + constant >= 1 << width
            )
    for threshold in (0, 1, 17, 63, 64, 80):
        root = unsigned_at_least_constant(manager, bits, threshold)
        for value in range(1 << width):
            assignment = {
                variable: bool(value >> bit & 1)
                for bit, variable in enumerate(variables)
            }
            assert manager.evaluate(root, assignment) == (value >= threshold)
    for expected in (-1, 0, 17, 63, 64):
        root = unsigned_equals_constant(manager, bits, expected)
        for value in range(1 << width):
            assignment = {
                variable: bool(value >> bit & 1)
                for bit, variable in enumerate(variables)
            }
            assert manager.evaluate(root, assignment) == (value == expected)


def test_general_add_select_and_saturating_subtract() -> None:
    width = 5
    variables = tuple((*[f"a{bit}" for bit in range(width)], *[f"b{bit}" for bit in range(width)]))
    manager = ROBDDManager(variables)
    left = tuple(manager.variable(f"a{bit}") for bit in range(width))
    right = tuple(manager.variable(f"b{bit}") for bit in range(width))
    result, overflow = unsigned_add(manager, left, right)
    selected = select_bits(manager, manager.variable("a0"), left, right)
    subtracted = unsigned_subtract_constant_floor_zero(manager, left, 7)
    assert constant_bits(7, width) == (1, 1, 1, 0, 0)
    for first in range(1 << width):
        for second in range(1 << width):
            assignment = {
                **{f"a{bit}": bool(first >> bit & 1) for bit in range(width)},
                **{f"b{bit}": bool(second >> bit & 1) for bit in range(width)},
            }
            actual_sum = sum(
                int(manager.evaluate(root, assignment)) << bit
                for bit, root in enumerate(result)
            )
            assert actual_sum == (first + second) % (1 << width)
            assert manager.evaluate(overflow, assignment) == (
                first + second >= 1 << width
            )
            chosen = sum(
                int(manager.evaluate(root, assignment)) << bit
                for bit, root in enumerate(selected)
            )
            assert chosen == (first if first & 1 else second)
            difference = sum(
                int(manager.evaluate(root, assignment)) << bit
                for bit, root in enumerate(subtracted)
            )
            assert difference == max(0, first - 7)


def test_satisfying_assignment_uses_one_model_without_enumeration() -> None:
    manager = ROBDDManager(("a", "b", "c"))
    root = manager.apply(
        "and",
        manager.variable("a"),
        manager.variable("c"),
    )
    witness = manager.satisfying_assignment(root)
    assert witness == {"a": True, "b": False, "c": True}
    assert manager.satisfying_assignment(0) is None


def test_unsigned_lookup_accepts_composed_boolean_inputs() -> None:
    manager = ROBDDManager(("x", "y"))
    x = manager.variable("x")
    y = manager.variable("y")
    inputs = (manager.apply("xor", x, y), y)
    table = tuple((value * value + 1) % 8 for value in range(4))
    outputs = unsigned_lookup(manager, inputs, table, output_width=3)
    for first in (False, True):
        for second in (False, True):
            assignment = {"x": first, "y": second}
            index = int(first != second) | (int(second) << 1)
            observed = sum(
                int(manager.evaluate(root, assignment)) << bit
                for bit, root in enumerate(outputs)
            )
            assert observed == table[index]


def test_signed_case_sum_avoids_case_cartesian_product_exactly() -> None:
    manager = ROBDDManager(("a", "b"))
    a = manager.variable("a")
    b = manager.variable("b")
    first_cases = ((-2, a), (1, manager.negate(a)))
    second_cases = ((3, b), (-1, manager.negate(b)))
    bits, bias = signed_case_sum_bits(
        manager,
        ((2, first_cases), (3, second_cases)),
    )
    assert bias == 5
    for first in (False, True):
        for second in (False, True):
            assignment = {"a": first, "b": second}
            observed = sum(
                int(manager.evaluate(root, assignment)) << bit
                for bit, root in enumerate(bits)
            ) - bias
            expected = (-2 if first else 1) + (3 if second else -1)
            assert observed == expected
