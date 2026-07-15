"""Symbolic IC2 layout constraints and exact static power/heat circuits."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Hashable, Mapping, Sequence

from .components import COMPONENTS
from .robdd import ROBDDManager
from .robdd_bitvector import (
    constant_bits,
    select_bits,
    unsigned_add,
    unsigned_at_least_constant,
    unsigned_equals_constant,
)


def _exact_boolean_count(
    manager: ROBDDManager,
    roots: Sequence[int],
) -> tuple[int, ...]:
    """Return roots for exactly 0..len(roots) true inputs."""

    distribution = [1]
    for root in roots:
        following = [0] * (len(distribution) + 1)
        negated = manager.negate(root)
        for count, previous in enumerate(distribution):
            following[count] = manager.apply(
                "or",
                following[count],
                manager.apply("and", previous, negated),
            )
            following[count + 1] = manager.apply(
                "or",
                following[count + 1],
                manager.apply("and", previous, root),
            )
        distribution = following
    return tuple(distribution)


def _sum_conditioned_constants(
    manager: ROBDDManager,
    terms: Sequence[tuple[int, int]],
    width: int,
) -> tuple[int, ...]:
    total = (0,) * width
    zeros = (0,) * width
    for condition, value in terms:
        contribution = select_bits(
            manager,
            condition,
            constant_bits(value, width),
            zeros,
        )
        total, overflow = unsigned_add(manager, total, contribution, width=width)
        if overflow != 0:  # pragma: no cover - widths use analytic maxima
            raise AssertionError("symbolic IC2 aggregate width is insufficient")
    return total


@dataclass(frozen=True, slots=True)
class IC2SymbolicLayoutCircuit:
    rows: int
    columns: int
    labels: tuple[str, ...]
    exact_rods: int
    manager: ROBDDManager
    layout_variables: tuple[tuple[Hashable, ...], ...]
    valid_labels_root: int
    rod_budget_root: int
    feasible_layout_root: int
    rod_count_bits: tuple[int, ...]
    pulse_unit_bits: tuple[int, ...]
    power_bits: tuple[int, ...]
    generated_heat_bits: tuple[int, ...]

    def assignment(self, layout: Sequence[str]) -> dict[Hashable, bool]:
        raw = tuple(layout)
        if len(raw) != self.rows * self.columns:
            raise ValueError("symbolic layout has the wrong number of cells")
        code_by_label = {label: code for code, label in enumerate(self.labels)}
        try:
            codes = tuple(code_by_label[label] for label in raw)
        except KeyError as error:
            raise ValueError(f"symbolic layout uses unknown label: {error.args[0]}") from error
        return {
            variable: bool(codes[vertex] >> bit & 1)
            for vertex, variables in enumerate(self.layout_variables)
            for bit, variable in enumerate(variables)
        }

    def value(self, bits: Sequence[int], assignment: Mapping[Hashable, bool]) -> int:
        return sum(
            int(self.manager.evaluate(root, assignment)) << bit
            for bit, root in enumerate(bits)
        )


def compile_ic2_symbolic_layout_circuit(
    *,
    rows: int,
    columns: int,
    exact_rods: int,
    labels: Sequence[str],
    manager: ROBDDManager | None = None,
    layout_variables: Sequence[Sequence[Hashable]] | None = None,
) -> IC2SymbolicLayoutCircuit:
    """Compile all layouts collectively; no label assignment is enumerated."""

    if rows <= 0 or columns <= 0 or exact_rods < 0:
        raise ValueError("symbolic layout dimensions/rod budget are invalid")
    label_tuple = tuple(labels)
    if not label_tuple or len(label_tuple) != len(set(label_tuple)):
        raise ValueError("symbolic layout labels must be non-empty and unique")
    if unknown := set(label_tuple) - COMPONENTS.keys():
        raise ValueError(f"unknown symbolic IC2 labels: {sorted(unknown)}")
    code_width = max(1, ceil(log2(len(label_tuple))))
    cell_count = rows * columns
    owns_manager = manager is None
    if layout_variables is None:
        if manager is not None:
            raise ValueError("shared manager requires explicit layout variables")
        variable_rows = tuple(
            tuple(("layout", vertex, bit) for bit in range(code_width))
            for vertex in range(cell_count)
        )
        manager = ROBDDManager(tuple(
            variable for variables in variable_rows for variable in variables
        ))
    else:
        variable_rows = tuple(tuple(variables) for variables in layout_variables)
        if len(variable_rows) != cell_count or any(
            len(variables) != code_width for variables in variable_rows
        ):
            raise ValueError("symbolic layout variable matrix has the wrong shape")
        if manager is None:
            raise ValueError("explicit layout variables require a shared manager")
        flat = {variable for variables in variable_rows for variable in variables}
        if len(flat) != cell_count * code_width or not flat <= set(manager.variables):
            raise ValueError("symbolic layout variables are duplicate or unknown")
    assert manager is not None

    code_bits = tuple(
        tuple(manager.variable(variable) for variable in variables)
        for variables in variable_rows
    )
    label_roots = tuple(
        tuple(
            unsigned_equals_constant(manager, bits, code)
            for code in range(len(label_tuple))
        )
        for bits in code_bits
    )
    valid_labels = manager.conjunction(*(
        manager.negate(unsigned_at_least_constant(manager, bits, len(label_tuple)))
        for bits in code_bits
    ))

    maximum_rods = cell_count * max(COMPONENTS[label].rod_count for label in label_tuple)
    rod_width = max(1, maximum_rods.bit_length())
    rod_terms = [
        (label_roots[vertex][code], COMPONENTS[label].rod_count)
        for vertex in range(cell_count)
        for code, label in enumerate(label_tuple)
        if COMPONENTS[label].rod_count
    ]
    rod_bits = _sum_conditioned_constants(manager, rod_terms, rod_width)
    rod_budget = unsigned_equals_constant(manager, rod_bits, exact_rods)
    feasible = manager.apply("and", valid_labels, rod_budget)

    active = tuple(
        manager.disjunction(*(
            label_roots[vertex][code]
            for code, label in enumerate(label_tuple)
            if COMPONENTS[label].kind in {"fuel", "reflector"}
        ))
        for vertex in range(cell_count)
    )

    def neighbours(vertex: int) -> tuple[int, ...]:
        row, column = divmod(vertex, columns)
        result = []
        if column:
            result.append(vertex - 1)
        if column + 1 < columns:
            result.append(vertex + 1)
        if row:
            result.append(vertex - columns)
        if row + 1 < rows:
            result.append(vertex + columns)
        return tuple(result)

    maximum_pulse_units = cell_count * 4 * (3 + 4)
    pulse_width = max(1, maximum_pulse_units.bit_length())
    maximum_power = maximum_pulse_units * 5
    power_width = max(1, maximum_power.bit_length())
    maximum_heat = cell_count * 2 * 4 * (3 + 4) * (3 + 4 + 1)
    heat_width = max(1, maximum_heat.bit_length())
    pulse_terms: list[tuple[int, int]] = []
    power_terms: list[tuple[int, int]] = []
    heat_terms: list[tuple[int, int]] = []
    for vertex in range(cell_count):
        adjacent = neighbours(vertex)
        degree_roots = _exact_boolean_count(
            manager,
            tuple(active[item] for item in adjacent),
        )
        for code, label in enumerate(label_tuple):
            spec = COMPONENTS[label]
            if spec.kind != "fuel":
                continue
            for degree, degree_root in enumerate(degree_roots):
                condition = manager.apply(
                    "and",
                    label_roots[vertex][code],
                    degree_root,
                )
                pulses = spec.internal_pulses + degree
                pulse_units = spec.rod_count * pulses
                pulse_terms.append((condition, pulse_units))
                power_terms.append((condition, 5 * pulse_units))
                heat_terms.append((
                    condition,
                    2 * spec.rod_count * pulses * (pulses + 1),
                ))

    pulse_bits = _sum_conditioned_constants(manager, pulse_terms, pulse_width)
    power_bits = _sum_conditioned_constants(manager, power_terms, power_width)
    heat_bits = _sum_conditioned_constants(manager, heat_terms, heat_width)
    if owns_manager:
        manager, roots = manager.compact_roots((
            valid_labels,
            rod_budget,
            feasible,
            *rod_bits,
            *pulse_bits,
            *power_bits,
            *heat_bits,
        ))
        cursor = 0
        valid_labels, rod_budget, feasible = roots[cursor:cursor + 3]
        cursor += 3
        rod_bits = roots[cursor:cursor + len(rod_bits)]
        cursor += len(rod_bits)
        pulse_bits = roots[cursor:cursor + len(pulse_bits)]
        cursor += len(pulse_bits)
        power_bits = roots[cursor:cursor + len(power_bits)]
        cursor += len(power_bits)
        heat_bits = roots[cursor:cursor + len(heat_bits)]

    return IC2SymbolicLayoutCircuit(
        rows=rows,
        columns=columns,
        labels=label_tuple,
        exact_rods=exact_rods,
        manager=manager,
        layout_variables=variable_rows,
        valid_labels_root=valid_labels,
        rod_budget_root=rod_budget,
        feasible_layout_root=feasible,
        rod_count_bits=rod_bits,
        pulse_unit_bits=tuple(pulse_bits),
        power_bits=tuple(power_bits),
        generated_heat_bits=tuple(heat_bits),
    )
