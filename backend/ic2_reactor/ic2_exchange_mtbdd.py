"""Exact interval-compressed ROBDD circuits for IC2 exchanger amounts.

The official rule uses IEEE-754 binary64 intermediates and integer truncation;
replacing it with rational algebra is not exact.  Exhaustive capacity-product
tables are also unnecessary.  For fixed source heat, target rounded percent
has three ordered sign regions.  ``combined`` crosses only four special
thresholds below one; above one the truncated/clamped magnitude is monotone.
Consequently one row has O(limit) constant intervals, found by binary search
using the official operation itself.  Equal target-row functions and source
decision suffixes are then shared by the ROBDD unique table.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Hashable, Mapping, Sequence

from .engine import ReactorSimulator
from .robdd import ROBDDManager
from .robdd_bitvector import unsigned_at_least_constant


@dataclass(frozen=True, slots=True)
class IC2ExchangeAmountCircuit:
    source_capacity: int
    target_capacity: int
    target_heat_minimum: int
    target_heat_maximum: int
    limit: int
    rounded_base: bool
    low_range: int | None
    manager: ROBDDManager
    source_bits: tuple[int, ...]
    target_bits: tuple[int, ...]
    amount_code_bits: tuple[int, ...]
    source_row_classes: tuple[int, ...]
    interval_count: int
    unique_row_count: int
    oracle_evaluations: int
    negative_amount_never_exceeds_target: bool

    def amount(self, assignment: Mapping[Hashable, bool]) -> int:
        code = sum(
            int(self.manager.evaluate(root, assignment)) << bit
            for bit, root in enumerate(self.amount_code_bits)
        )
        return code - self.limit


def _first_true(end: int, predicate) -> int:
    """First integer in [0,end) satisfying a false-then-true predicate."""

    low, high = 0, end
    while low < high:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle + 1
    return low


def _merge_intervals(
    intervals: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    merged: list[tuple[int, int, int]] = []
    for start, end, value in intervals:
        if start >= end:
            continue
        if merged and merged[-1][1] == start and merged[-1][2] == value:
            merged[-1] = (merged[-1][0], end, value)
        else:
            merged.append((start, end, value))
    return tuple(merged)


def _row_intervals(
    source_heat: int,
    source_capacity: int,
    target_capacity: int,
    target_heat_minimum: int,
    target_heat_maximum: int,
    limit: int,
    *,
    rounded_base: bool,
    low_range: int | None,
    evaluation_counter: list[int],
) -> tuple[tuple[int, int, int], ...]:
    source_ratio = source_heat * 100.0 / source_capacity
    source_tenth = floor(source_ratio * 10.0 + 0.5) / 10.0
    end = target_heat_maximum - target_heat_minimum + 1

    def physical_heat(code: int) -> int:
        return target_heat_minimum + code

    def target_ratio(code: int) -> float:
        return physical_heat(code) * 100.0 / target_capacity

    def target_tenth(code: int) -> float:
        return floor(target_ratio(code) * 10.0 + 0.5) / 10.0

    def combined(code: int) -> float:
        return target_ratio(code) + source_ratio / 2.0

    def amount(code: int) -> int:
        evaluation_counter[0] += 1
        return ReactorSimulator._exchange_amount(
            source_ratio,
            target_ratio(code),
            target_capacity,
            limit,
            rounded_base=rounded_base,
            low_range=low_range,
        )

    boundaries = {0, end}
    boundaries.add(_first_true(end, lambda heat: target_tenth(heat) >= source_tenth))
    boundaries.add(_first_true(end, lambda heat: target_tenth(heat) > source_tenth))
    for threshold in (0.25, 0.5, 0.75, 1.0):
        boundaries.add(_first_true(end, lambda heat, t=threshold: combined(heat) >= t))
    ordered = sorted(boundary for boundary in boundaries if 0 <= boundary <= end)

    intervals: list[tuple[int, int, int]] = []
    for start, stop in zip(ordered, ordered[1:]):
        if start >= stop:
            continue
        # Below combined=1 every threshold band overrides the raw magnitude
        # with a constant.  A rounded-percent sign band is also constant zero.
        if combined(start) < 1.0 or target_tenth(start) == source_tenth:
            intervals.append((start, stop, amount(start)))
            continue

        # In a fixed nonzero sign region above combined=1, the absolute
        # truncated/clamped amount is monotone.  Locate each next run exactly.
        cursor = start
        while cursor < stop:
            value = amount(cursor)
            magnitude = abs(value)
            following = _first_true(
                stop - cursor - 1,
                lambda offset: abs(amount(cursor + 1 + offset)) > magnitude,
            )
            next_cursor = cursor + 1 + following
            if next_cursor > stop:
                next_cursor = stop
            intervals.append((cursor, next_cursor, value))
            cursor = next_cursor
    result = _merge_intervals(intervals)
    if not result or result[0][0] != 0 or result[-1][1] != end:
        raise AssertionError("exchange interval partition does not cover its row")
    return result


def compile_ic2_exchange_amount_circuit(
    manager: ROBDDManager,
    source_bits: Sequence[int],
    target_bits: Sequence[int],
    *,
    source_capacity: int,
    target_capacity: int,
    target_heat_minimum: int = 0,
    target_heat_maximum: int | None = None,
    limit: int,
    rounded_base: bool = False,
    low_range: int | None = None,
) -> IC2ExchangeAmountCircuit:
    """Compile the exact signed amount as offset code ``amount + limit``."""

    source = tuple(source_bits)
    target = tuple(target_bits)
    minimum_target_heat = int(target_heat_minimum)
    maximum_target_heat = (
        target_capacity if target_heat_maximum is None else int(target_heat_maximum)
    )
    if (
        source_capacity <= 0
        or target_capacity <= 0
        or minimum_target_heat > maximum_target_heat
        or limit <= 0
    ):
        raise ValueError("exchange capacities and limit must be positive")
    if not source or not target:
        raise ValueError("exchange circuit needs source and target bits")
    if (
        source_capacity >= 1 << len(source)
        or maximum_target_heat - minimum_target_heat >= 1 << len(target)
    ):
        raise ValueError("exchange state does not fit its bit-vector")
    if low_range is not None and low_range < 0:
        raise ValueError("exchange low range must be non-negative")

    evaluation_counter = [0]
    rows = tuple(
        _row_intervals(
            heat,
            source_capacity,
            target_capacity,
            minimum_target_heat,
            maximum_target_heat,
            limit,
            rounded_base=rounded_base,
            low_range=low_range,
            evaluation_counter=evaluation_counter,
        )
        for heat in range(source_capacity + 1)
    )
    code_width = max(1, (2 * limit).bit_length())
    interval_predicates: dict[tuple[int, int], int] = {}

    def interval_root(start: int, end: int) -> int:
        key = (start, end)
        found = interval_predicates.get(key)
        if found is not None:
            return found
        found = manager.apply(
            "and",
            unsigned_at_least_constant(manager, target, start),
            manager.negate(unsigned_at_least_constant(manager, target, end)),
        )
        interval_predicates[key] = found
        return found

    row_roots_by_bit: list[list[int]] = [[] for _bit in range(code_width)]
    for intervals in rows:
        for bit in range(code_width):
            roots = [
                interval_root(start, end)
                for start, end, value in intervals
                if (value + limit) >> bit & 1
            ]
            row_roots_by_bit[bit].append(manager.disjunction(*roots))

    row_class_by_signature: dict[tuple[tuple[int, int, int], ...], int] = {}
    source_row_classes = []
    for row in rows:
        source_row_classes.append(row_class_by_signature.setdefault(
            row,
            len(row_class_by_signature),
        ))

    padded_size = 1 << len(source)
    zero_code = limit
    amount_bits = []
    for bit, valid_rows in enumerate(row_roots_by_bit):
        invalid_root = int(bool(zero_code >> bit & 1))
        roots = tuple((*valid_rows, *((invalid_root,) * (padded_size - len(valid_rows)))))
        cache: dict[tuple[int, tuple[int, ...]], int] = {}

        def multiplex(level: int, values: tuple[int, ...]) -> int:
            if all(value == values[0] for value in values):
                return values[0]
            key = (level, values)
            found = cache.get(key)
            if found is not None:
                return found
            low = multiplex(level + 1, values[0::2])
            high = multiplex(level + 1, values[1::2])
            result = manager.ite(source[level], high, low)
            cache[key] = result
            return result

        amount_bits.append(multiplex(0, roots))

    return IC2ExchangeAmountCircuit(
        source_capacity=source_capacity,
        target_capacity=target_capacity,
        target_heat_minimum=minimum_target_heat,
        target_heat_maximum=maximum_target_heat,
        limit=limit,
        rounded_base=rounded_base,
        low_range=low_range,
        manager=manager,
        source_bits=source,
        target_bits=target,
        amount_code_bits=tuple(amount_bits),
        source_row_classes=tuple(source_row_classes),
        interval_count=sum(len(row) for row in rows),
        unique_row_count=len(row_class_by_signature),
        oracle_evaluations=evaluation_counter[0],
        negative_amount_never_exceeds_target=all(
            value >= 0 or start >= -value
            for row in rows
            for start, _end, value in row
        ),
    )
