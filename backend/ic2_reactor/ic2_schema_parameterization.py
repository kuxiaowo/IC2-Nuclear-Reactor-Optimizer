"""Exact per-cell label parameters conditioned on dynamic state schemas.

The full permanent catalogue has 22 labels but only 12 dynamic field schemas.
Conditioning on a schema skeleton makes every cell's heat width/capacity and
plating bonus constant while retaining all real label behaviours inside that
schema.  This is the natural interface between the outer frontier quotient
over schemas and a layout-variable event circuit: no schema skeleton member or
within-schema label assignment is enumerated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Hashable, Mapping, Sequence

from .components import COMPONENTS
from .ic2_dynamic_schema import (
    IC2DynamicStateSchema,
    IC2StructuralSignature,
    ic2_permanent_catalogue_quotient,
    ic2_permanent_structural_quotient,
)
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
        if overflow != 0:  # pragma: no cover - analytic widths below
            raise AssertionError("schema-conditioned aggregate width is insufficient")
    return total


@dataclass(frozen=True, slots=True)
class IC2SchemaConditionedLayoutCircuit:
    rows: int
    columns: int
    schemas: tuple[IC2DynamicStateSchema, ...]
    structural_signatures: tuple[IC2StructuralSignature, ...] | None
    label_domains: tuple[tuple[str, ...], ...]
    exact_rods: int
    manager: ROBDDManager
    parameter_variables: tuple[Hashable, ...]
    cell_parameter_variables: tuple[tuple[Hashable, ...], ...]
    label_roots: tuple[tuple[int, ...], ...]
    valid_parameter_root: int
    rod_budget_root: int
    feasible_parameter_root: int
    rod_count_bits: tuple[int, ...]
    power_bits: tuple[int, ...]
    generated_heat_bits: tuple[int, ...]

    def assignment(self, layout: Sequence[str]) -> dict[Hashable, bool]:
        raw = tuple(layout)
        if len(raw) != len(self.schemas):
            raise ValueError("schema-conditioned layout has the wrong size")
        result: dict[Hashable, bool] = {}
        for vertex, (label, domain, variables) in enumerate(zip(
            raw,
            self.label_domains,
            self.cell_parameter_variables,
            strict=True,
        )):
            try:
                code = domain.index(label)
            except ValueError as error:
                raise ValueError(
                    f"label at cell {vertex} is outside its dynamic schema"
                ) from error
            result.update({
                variable: bool(code >> bit & 1)
                for bit, variable in enumerate(variables)
            })
        return result

    def value(self, bits: Sequence[int], assignment: Mapping[Hashable, bool]) -> int:
        total = {
            variable: bool(assignment.get(variable, False))
            for variable in self.manager.variables
        }
        return sum(
            int(self.manager.evaluate(root, total)) << bit
            for bit, root in enumerate(bits)
        )


def compile_ic2_schema_conditioned_layout_circuit(
    *,
    rows: int,
    columns: int,
    exact_rods: int,
    schemas: Sequence[IC2DynamicStateSchema] | None = None,
    structural_signatures: Sequence[IC2StructuralSignature] | None = None,
    manager: ROBDDManager | None = None,
    cell_parameter_variables: Sequence[Sequence[Hashable]] | None = None,
) -> IC2SchemaConditionedLayoutCircuit:
    """Compile every label refinement of one schema skeleton collectively."""

    cell_count = rows * columns
    if (schemas is None) == (structural_signatures is None):
        raise ValueError(
            "provide exactly one of schemas or structural signatures"
        )
    signature_tuple = (
        None
        if structural_signatures is None
        else tuple(structural_signatures)
    )
    schema_tuple = (
        tuple(schemas)
        if signature_tuple is None
        else tuple(signature.dynamic_schema for signature in signature_tuple)
    )
    if (
        rows <= 0
        or columns <= 0
        or len(schema_tuple) != cell_count
        or exact_rods < 0
    ):
        raise ValueError("schema-conditioned layout dimensions are invalid")
    if signature_tuple is None:
        quotient = ic2_permanent_catalogue_quotient()
        domain_by_schema = dict(quotient.schemas)
        if unknown := set(schema_tuple) - domain_by_schema.keys():
            raise ValueError(f"unknown permanent dynamic schemas: {sorted(unknown)}")
        domains = tuple(domain_by_schema[schema] for schema in schema_tuple)
    else:
        quotient = ic2_permanent_structural_quotient()
        domain_by_signature = dict(quotient.groups)
        if unknown := set(signature_tuple) - domain_by_signature.keys():
            raise ValueError(f"unknown permanent structural signatures: {sorted(unknown)}")
        domains = tuple(
            domain_by_signature[signature] for signature in signature_tuple
        )
    widths = tuple(
        0 if len(domain) == 1 else ceil(log2(len(domain)))
        for domain in domains
    )

    owns_manager = manager is None
    if cell_parameter_variables is None:
        if manager is not None:
            raise ValueError("shared manager requires explicit cell parameter variables")
        variable_rows = tuple(
            tuple(("schema_layout", vertex, bit) for bit in range(width))
            for vertex, width in enumerate(widths)
        )
        flat_variables = tuple(
            variable for variables in variable_rows for variable in variables
        )
        if not flat_variables:
            raise ValueError(
                "a fully singleton schema skeleton needs a shared dynamic manager"
            )
        manager = ROBDDManager(flat_variables)
    else:
        if manager is None:
            raise ValueError("explicit parameter variables require a shared manager")
        variable_rows = tuple(tuple(variables) for variables in cell_parameter_variables)
        if len(variable_rows) != cell_count or any(
            len(variables) != width
            for variables, width in zip(variable_rows, widths, strict=True)
        ):
            raise ValueError("schema-conditioned parameter matrix has wrong widths")
        flat_variables = tuple(
            variable for variables in variable_rows for variable in variables
        )
        if (
            len(flat_variables) != len(set(flat_variables))
            or not set(flat_variables) <= set(manager.variables)
        ):
            raise ValueError("schema-conditioned parameters are duplicate or unknown")
    assert manager is not None

    code_bits = tuple(
        tuple(manager.variable(variable) for variable in variables)
        for variables in variable_rows
    )
    label_roots = tuple(
        (1,)
        if len(domain) == 1
        else tuple(
            unsigned_equals_constant(manager, bits, code)
            for code in range(len(domain))
        )
        for domain, bits in zip(domains, code_bits, strict=True)
    )
    valid = manager.conjunction(*(
        1
        if len(domain) == 1
        else manager.negate(unsigned_at_least_constant(manager, bits, len(domain)))
        for domain, bits in zip(domains, code_bits, strict=True)
    ))

    maximum_rods = sum(
        max(COMPONENTS[label].rod_count for label in domain)
        for domain in domains
    )
    rod_width = max(1, maximum_rods.bit_length())
    rod_terms = tuple(
        (label_roots[vertex][code], COMPONENTS[label].rod_count)
        for vertex, domain in enumerate(domains)
        for code, label in enumerate(domain)
        if COMPONENTS[label].rod_count
    )
    rod_bits = _sum_conditioned_constants(manager, rod_terms, rod_width)
    rod_budget = unsigned_equals_constant(manager, rod_bits, exact_rods)
    feasible = manager.apply("and", valid, rod_budget)

    active = tuple(
        manager.disjunction(*(
            label_roots[vertex][code]
            for code, label in enumerate(domain)
            if COMPONENTS[label].kind in {"fuel", "reflector"}
        ))
        for vertex, domain in enumerate(domains)
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

    maximum_pulse_units = cell_count * 4 * 7
    maximum_power = maximum_pulse_units * 5
    maximum_heat = cell_count * 2 * 4 * 7 * 8
    power_width = max(1, maximum_power.bit_length())
    heat_width = max(1, maximum_heat.bit_length())
    power_terms: list[tuple[int, int]] = []
    heat_terms: list[tuple[int, int]] = []
    for vertex, domain in enumerate(domains):
        degree_roots = _exact_boolean_count(
            manager,
            tuple(active[item] for item in neighbours(vertex)),
        )
        for code, label in enumerate(domain):
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
                power_terms.append((condition, 5 * pulse_units))
                heat_terms.append((
                    condition,
                    2 * spec.rod_count * pulses * (pulses + 1),
                ))
    power_bits = _sum_conditioned_constants(manager, power_terms, power_width)
    heat_bits = _sum_conditioned_constants(manager, heat_terms, heat_width)

    parameter_variables = tuple(
        variable for variables in variable_rows for variable in variables
    )
    if owns_manager:
        manager, roots = manager.compact_roots((
            valid,
            rod_budget,
            feasible,
            *rod_bits,
            *power_bits,
            *heat_bits,
        ))
        cursor = 0
        valid, rod_budget, feasible = roots[cursor:cursor + 3]
        cursor += 3
        rod_bits = roots[cursor:cursor + len(rod_bits)]
        cursor += len(rod_bits)
        power_bits = roots[cursor:cursor + len(power_bits)]
        cursor += len(power_bits)
        heat_bits = roots[cursor:cursor + len(heat_bits)]

    return IC2SchemaConditionedLayoutCircuit(
        rows=rows,
        columns=columns,
        schemas=schema_tuple,
        structural_signatures=signature_tuple,
        label_domains=domains,
        exact_rods=exact_rods,
        manager=manager,
        parameter_variables=parameter_variables,
        cell_parameter_variables=variable_rows,
        label_roots=label_roots,
        valid_parameter_root=valid,
        rod_budget_root=rod_budget,
        feasible_parameter_root=feasible,
        rod_count_bits=tuple(rod_bits),
        power_bits=tuple(power_bits),
        generated_heat_bits=tuple(heat_bits),
    )
