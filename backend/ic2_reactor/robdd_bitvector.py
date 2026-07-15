"""Unsigned little-endian bit-vector circuits over the local ROBDD manager."""

from __future__ import annotations

from typing import Sequence

from .robdd import ROBDDManager


def _pad(bits: Sequence[int], width: int) -> tuple[int, ...]:
    raw = tuple(bits)
    if width < len(raw):
        raise ValueError("cannot shrink an unsigned bit-vector by padding")
    return (*raw, *((0,) * (width - len(raw))))


def unsigned_add(
    manager: ROBDDManager,
    left: Sequence[int],
    right: Sequence[int],
    *,
    width: int | None = None,
) -> tuple[tuple[int, ...], int]:
    """Return a fixed-width unsigned sum and overflow predicate."""

    left_raw = tuple(left)
    right_raw = tuple(right)
    if not left_raw or not right_raw:
        raise ValueError("bit-vector addition requires non-empty operands")
    result_width = max(len(left_raw), len(right_raw)) if width is None else int(width)
    if result_width < max(len(left_raw), len(right_raw)):
        raise ValueError("bit-vector sum width is smaller than an operand")
    left_bits = _pad(left_raw, result_width)
    right_bits = _pad(right_raw, result_width)
    carry = 0
    result = []
    for first, second in zip(left_bits, right_bits, strict=True):
        partial = manager.apply("xor", first, second)
        result.append(manager.apply("xor", partial, carry))
        carry = manager.disjunction(
            manager.apply("and", first, second),
            manager.apply("and", first, carry),
            manager.apply("and", second, carry),
        )
    return tuple(result), carry


def unsigned_add_constant(
    manager: ROBDDManager,
    bits: Sequence[int],
    value: int,
) -> tuple[tuple[int, ...], int]:
    """Return fixed-width sum bits and an unsigned overflow predicate."""

    raw = tuple(bits)
    if not raw:
        raise ValueError("bit-vector addition requires at least one bit")
    if value < 0:
        raise ValueError("unsigned bit-vector constant must be non-negative")
    constant_bits = tuple(
        int(bool(value >> position & 1)) for position in range(len(raw))
    )
    result, carry = unsigned_add(manager, raw, constant_bits, width=len(raw))
    return result, (1 if value >> len(raw) else carry)


def unsigned_at_least_constant(
    manager: ROBDDManager,
    bits: Sequence[int],
    threshold: int,
) -> int:
    """Return the predicate represented unsigned value is at least threshold."""

    raw = tuple(bits)
    if not raw:
        raise ValueError("bit-vector comparison requires at least one bit")
    if threshold <= 0:
        return 1
    if threshold >= 1 << len(raw):
        return 0
    less = 0
    equal = 1
    for position in range(len(raw) - 1, -1, -1):
        bit = raw[position]
        if threshold >> position & 1:
            less = manager.apply(
                "or",
                less,
                manager.apply("and", equal, manager.negate(bit)),
            )
            equal = manager.apply("and", equal, bit)
        else:
            equal = manager.apply("and", equal, manager.negate(bit))
    return manager.negate(less)


def unsigned_equals_constant(
    manager: ROBDDManager,
    bits: Sequence[int],
    value: int,
) -> int:
    """Return the predicate represented unsigned value equals ``value``."""

    raw = tuple(bits)
    if not raw:
        raise ValueError("bit-vector equality requires at least one bit")
    if value < 0 or value >= 1 << len(raw):
        return 0
    return manager.conjunction(*(
        bit if value >> position & 1 else manager.negate(bit)
        for position, bit in enumerate(raw)
    ))


def select_bits(
    manager: ROBDDManager,
    condition: int,
    when_true: Sequence[int],
    when_false: Sequence[int],
) -> tuple[int, ...]:
    first = tuple(when_true)
    second = tuple(when_false)
    if len(first) != len(second):
        raise ValueError("selected bit-vectors must have equal width")
    return tuple(
        manager.ite(condition, left, right)
        for left, right in zip(first, second, strict=True)
    )


def unsigned_subtract_constant_floor_zero(
    manager: ROBDDManager,
    bits: Sequence[int],
    value: int,
) -> tuple[int, ...]:
    """Return ``max(unsigned(bits) - value, 0)``."""

    raw = tuple(bits)
    if not raw:
        raise ValueError("bit-vector subtraction requires at least one bit")
    if value < 0:
        raise ValueError("unsigned subtraction constant must be non-negative")
    if value == 0:
        return raw
    if value >= 1 << len(raw):
        return (0,) * len(raw)
    enough = unsigned_at_least_constant(manager, raw, value)
    modular, _overflow = unsigned_add_constant(
        manager,
        raw,
        (1 << len(raw)) - value,
    )
    return tuple(manager.apply("and", enough, bit) for bit in modular)


def signed_case_sum_bits(
    manager: ROBDDManager,
    partitions: Sequence[tuple[int, Sequence[tuple[int, int]]]],
) -> tuple[tuple[int, ...], int]:
    """Add deterministic signed case partitions using one unsigned offset.

    Each item is ``(bound, ((value, condition), ...))`` and promises
    ``-bound <= value <= bound``.  The conditions of an item are expected to
    select exactly one value on the caller's relevant domain.  The result
    encodes ``sum(values) + total_bound`` without constructing the Cartesian
    product of the individual value cases.
    """

    raw = tuple((int(bound), tuple(cases)) for bound, cases in partitions)
    if any(
        bound < 0
        or any(not -bound <= value <= bound for value, _condition in cases)
        for bound, cases in raw
    ):
        raise ValueError("signed case partition exceeds its declared bound")
    total_bound = sum(bound for bound, _cases in raw)
    width = max(1, (2 * total_bound).bit_length())
    total = constant_bits(0, width)
    for bound, cases in raw:
        encoded = tuple(
            manager.disjunction(*(
                condition
                for value, condition in cases
                if (value + bound) >> bit & 1
            ))
            for bit in range(width)
        )
        total, overflow = unsigned_add(manager, total, encoded, width=width)
        if overflow != 0:  # pragma: no cover - follows from the declared bounds
            raise AssertionError("signed case sum exceeded its analytic width")
    return total, total_bound


def constant_bits(value: int, width: int) -> tuple[int, ...]:
    if value < 0 or width <= 0:
        raise ValueError("bit-vector constant requires non-negative value and width")
    if value >= 1 << width:
        raise ValueError("bit-vector constant does not fit its width")
    return tuple(int(bool(value >> bit & 1)) for bit in range(width))


def unsigned_lookup(
    manager: ROBDDManager,
    input_bits: Sequence[int],
    values: Sequence[int],
    *,
    output_width: int,
) -> tuple[int, ...]:
    """Compile a small total integer lookup over arbitrary Boolean inputs.

    ``values[i]`` is selected by the little-endian integer represented by
    ``input_bits``.  This is useful for exact bounded operations such as
    division of the at-most-112 units produced by one IC2 fuel rod.
    """

    bits = tuple(input_bits)
    table = tuple(int(value) for value in values)
    if not bits or len(table) != 1 << len(bits):
        raise ValueError("unsigned lookup table has the wrong input size")
    if output_width <= 0 or any(
        value < 0 or value >= 1 << output_width for value in table
    ):
        raise ValueError("unsigned lookup output does not fit its width")

    def compile_boolean(rows: tuple[bool, ...], level: int = 0) -> int:
        if not any(rows):
            return 0
        if all(rows):
            return 1
        if level >= len(bits):  # pragma: no cover - constant cases above
            raise AssertionError("nonconstant lookup exhausted input bits")
        low = compile_boolean(rows[0::2], level + 1)
        high = compile_boolean(rows[1::2], level + 1)
        return manager.ite(bits[level], high, low)

    return tuple(
        compile_boolean(tuple(bool(value >> bit & 1) for value in table))
        for bit in range(output_width)
    )
