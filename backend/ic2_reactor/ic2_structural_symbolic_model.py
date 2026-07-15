"""Layout-variable IC2 event circuit conditioned on structural signatures.

This first production slice covers every permanent non-exchanger component.
The structural skeleton fixes power behaviour, heat-field widths and plating
bonuses.  Real labels inside each signature are frozen Boolean parameters and
select local events through sequential mux composition, never by enumerating
the Cartesian product of complete layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log2, prod
from typing import Hashable, Mapping, Sequence

from .components import COMPONENTS
from .ic2_dynamic_schema import (
    IC2StructuralSignature,
    ic2_permanent_structural_quotient,
)
from .ic2_schema_parameterization import (
    IC2SchemaConditionedLayoutCircuit,
    compile_ic2_schema_conditioned_layout_circuit,
)
from .ic2_exchange_mtbdd import (
    IC2ExchangeAmountCircuit,
    compile_ic2_exchange_amount_circuit,
)
from .ic2_symbolic_oracle import IC2SymbolicField
from .robdd import ROBDDManager
from .robdd_bitvector import (
    constant_bits,
    select_bits,
    signed_case_sum_bits,
    unsigned_add,
    unsigned_add_constant,
    unsigned_at_least_constant,
    unsigned_equals_constant,
    unsigned_lookup,
    unsigned_subtract_constant_floor_zero,
)
from .symbolic_event_composition import (
    FrozenParameterEventComposition,
)
from .symbolic_local_events import SymbolicLocalEvent


@dataclass(frozen=True, slots=True)
class IC2StructuralSymbolicModel:
    rows: int
    columns: int
    structural_signatures: tuple[IC2StructuralSignature, ...]
    layout_circuit: IC2SchemaConditionedLayoutCircuit
    manager: ROBDDManager
    parameter_variables: tuple[Hashable, ...]
    state_variables: tuple[Hashable, ...]
    fields: tuple[IC2SymbolicField, ...]
    next_functions: dict[Hashable, int]
    bad_root: int
    transition_failure_root: int
    parameter_constraint_root: int
    critical_heat: int
    encoded_state_count: int
    event_composition: FrozenParameterEventComposition
    peak_behavior_class_count: int
    terminal_behavior_class_count: int
    peak_branch_allocated_nodes: int
    peak_canonical_branch_nodes: int

    @property
    def label_domains(self) -> tuple[tuple[str, ...], ...]:
        return self.layout_circuit.label_domains

    def state_assignment(self, code: int) -> dict[Hashable, bool]:
        if not 0 <= code < self.encoded_state_count:
            raise ValueError("structural symbolic state code is outside the model")
        return {
            variable: bool(code >> bit & 1)
            for bit, variable in enumerate(self.state_variables)
        }

    def assignment(
        self,
        layout: Sequence[str],
        state_code: int,
    ) -> dict[Hashable, bool]:
        return {
            **self.layout_circuit.assignment(layout),
            **self.state_assignment(state_code),
        }

    def encode(
        self,
        hull_heat: int,
        component_heat: Mapping[int, int] | None = None,
    ) -> int:
        heat = {} if component_heat is None else dict(component_heat)
        code = 0
        for field in self.fields:
            value = hull_heat if field.kind == "hull" else heat.get(int(field.vertex), 0)
            if not field.minimum_safe_value <= value <= field.maximum_safe_value:
                raise ValueError("encoded thermal value is outside the safe field domain")
            code |= int(value - field.minimum_safe_value) << field.offset
        return code

    def next_code(self, layout: Sequence[str], state_code: int) -> int:
        assignment = self.assignment(layout, state_code)
        return sum(
            int(self.manager.evaluate(self.next_functions[variable], assignment)) << bit
            for bit, variable in enumerate(self.state_variables)
        )


@dataclass(frozen=True, slots=True)
class IC2StructuralLocalEventModel:
    """Exact layout-parametric IC2 tick kept as official local events."""

    rows: int
    columns: int
    structural_signatures: tuple[IC2StructuralSignature, ...]
    layout_circuit: IC2SchemaConditionedLayoutCircuit
    manager: ROBDDManager
    parameter_variables: tuple[Hashable, ...]
    state_variables: tuple[Hashable, ...]
    fields: tuple[IC2SymbolicField, ...]
    local_events: tuple[SymbolicLocalEvent, ...]
    bad_root: int
    parameter_constraint_root: int
    critical_heat: int
    encoded_state_count: int
    compiled_alternative_count: int
    skipped_identity_event_count: int
    peak_alternative_allocated_nodes: int
    peak_alternative_live_nodes: int
    local_event_live_node_counts: tuple[int, ...]
    local_event_support_sizes: tuple[int, ...]

    @property
    def label_domains(self) -> tuple[tuple[str, ...], ...]:
        return self.layout_circuit.label_domains

    def state_assignment(self, code: int) -> dict[Hashable, bool]:
        if not 0 <= code < self.encoded_state_count:
            raise ValueError("structural symbolic state code is outside the model")
        return {
            variable: bool(code >> bit & 1)
            for bit, variable in enumerate(self.state_variables)
        }

    def assignment(
        self,
        layout: Sequence[str],
        state_code: int,
    ) -> dict[Hashable, bool]:
        return {
            **self.layout_circuit.assignment(layout),
            **self.state_assignment(state_code),
        }

    def encode(
        self,
        hull_heat: int,
        component_heat: Mapping[int, int] | None = None,
    ) -> int:
        heat = {} if component_heat is None else dict(component_heat)
        code = 0
        for field in self.fields:
            value = hull_heat if field.kind == "hull" else heat.get(int(field.vertex), 0)
            if not field.minimum_safe_value <= value <= field.maximum_safe_value:
                raise ValueError("encoded thermal value is outside the safe field domain")
            code |= int(value - field.minimum_safe_value) << field.offset
        return code

    def step_code(self, layout: Sequence[str], state_code: int) -> tuple[int, bool]:
        """Evaluate the partitioned event program without closing its functions."""

        assignment = self.assignment(layout, state_code)
        for event in self.local_events:
            if self.manager.evaluate(event.failure_root, assignment):
                return self.encoded_state_count - 1, True
            updates = {
                variable: self.manager.evaluate(root, assignment)
                for variable, root in event.changed_next_functions
            }
            assignment.update(updates)
        failed = self.manager.evaluate(self.bad_root, assignment)
        if failed:
            return self.encoded_state_count - 1, True
        code = sum(
            int(assignment[variable]) << bit
            for bit, variable in enumerate(self.state_variables)
        )
        return code, False


@dataclass(frozen=True, slots=True)
class IC2LocalEventCompilationBound:
    """Layout-free count of local circuits required by one structural skeleton."""

    cell_count: int
    parameter_bit_count: int
    unconstrained_refinement_count: int
    eventful_slot_count: int
    local_alternative_count: int
    maximum_local_domain_size: int
    catalogue_maximum_local_alternatives: int
    catalogue_local_support_bit_upper_bound: int


def ic2_structural_local_event_work_bound(
    structural_signatures: Sequence[IC2StructuralSignature],
    *,
    rows: int = 6,
    columns: int,
) -> IC2LocalEventCompilationBound:
    """Count local event alternatives without compiling or enumerating layouts.

    A closed family has the product of all within-signature label-domain sizes.
    The partitioned compiler builds each event label once at its cell, hence its
    circuit count is the sum of eventful domain sizes.  This is an exact count
    of compilation units, not an estimate of their individual ROBDD sizes.
    """

    signatures = tuple(structural_signatures)
    if rows <= 0 or columns <= 0 or len(signatures) != rows * columns:
        raise ValueError("IC2 local-event work-bound dimensions are invalid")
    quotient = ic2_permanent_structural_quotient()
    domains_by_signature = dict(quotient.groups)
    if unknown := set(signatures) - domains_by_signature.keys():
        raise ValueError(f"unknown IC2 structural signatures: {sorted(unknown)}")
    domains = tuple(domains_by_signature[signature] for signature in signatures)
    event_sizes = tuple(
        len(domain)
        for domain in domains
        if any(
            COMPONENTS[label].kind in {"fuel", "vent", "exchanger"}
            for label in domain
        )
    )
    catalogue_event_domain_maximum = max(
        len(domain)
        for _signature, domain in quotient.groups
        if any(
            COMPONENTS[label].kind in {"fuel", "vent", "exchanger"}
            for label in domain
        )
    )
    maximum_heat_width = max(1, max(
        signature.dynamic_schema.heat_maximum
        for signature, _domain in quotient.groups
    ).bit_length())
    maximum_plating_bonus = max(
        signature.dynamic_schema.hull_capacity_bonus
        for signature, _domain in quotient.groups
    )
    maximum_hull_capacity = 10_000 + len(signatures) * maximum_plating_bonus
    maximum_critical_heat = floor(maximum_hull_capacity * 0.85)
    signed_hull_width = (maximum_critical_heat + 71).bit_length()
    maximum_grid_degree = (
        (2 if rows > 1 else 0) + (2 if columns > 1 else 0)
    )
    maximum_parameter_width = ceil(log2(catalogue_event_domain_maximum))
    return IC2LocalEventCompilationBound(
        cell_count=len(signatures),
        parameter_bit_count=sum(
            0 if len(domain) == 1 else ceil(log2(len(domain)))
            for domain in domains
        ),
        unconstrained_refinement_count=prod(map(len, domains)),
        eventful_slot_count=len(event_sizes),
        local_alternative_count=sum(event_sizes),
        maximum_local_domain_size=max(event_sizes, default=0),
        catalogue_maximum_local_alternatives=(
            len(signatures) * catalogue_event_domain_maximum
        ),
        catalogue_local_support_bit_upper_bound=(
            maximum_parameter_width
            + (maximum_grid_degree + 1) * maximum_heat_width
            + signed_hull_width
        ),
    )


def _compile_ic2_structural_symbolic_model(
    structural_signatures: Sequence[IC2StructuralSignature],
    *,
    rows: int = 6,
    columns: int,
    exact_rods: int,
    allow_exchangers: bool,
    partitioned_events_only: bool = False,
) -> IC2StructuralSymbolicModel | IC2StructuralLocalEventModel:
    """Compile all label refinements of one structural skeleton."""

    signatures = tuple(structural_signatures)
    if rows != 6 or not 3 <= columns <= 9 or len(signatures) != rows * columns:
        raise ValueError("IC2 structural symbolic dimensions are invalid")
    quotient = ic2_permanent_structural_quotient()
    domains_by_signature = dict(quotient.groups)
    if unknown := set(signatures) - domains_by_signature.keys():
        raise ValueError(f"unknown IC2 structural signatures: {sorted(unknown)}")
    domains = tuple(domains_by_signature[signature] for signature in signatures)
    if not allow_exchangers and (exchangers := {
        label
        for domain in domains
        for label in domain
        if COMPONENTS[label].kind == "exchanger"
    }):
        raise ValueError(
            "non-exchange structural compiler received exchanger alternatives: "
            f"{sorted(exchangers)}"
        )

    widths = tuple(
        0 if len(domain) == 1 else ceil(log2(len(domain)))
        for domain in domains
    )
    cell_parameter_variables = tuple(
        tuple(("layout", vertex, bit) for bit in range(width))
        for vertex, width in enumerate(widths)
    )
    parameters = tuple(
        variable
        for variables in cell_parameter_variables
        for variable in variables
    )
    maximum_hull_heat = 10_000 + sum(
        signature.dynamic_schema.hull_capacity_bonus
        for signature in signatures
    )
    critical_heat = floor(maximum_hull_heat * 0.85)
    minimum_hull_heat = -72 if allow_exchangers else 0
    raw_fields: list[tuple[str, int | None, int, int]] = [
        ("hull", None, minimum_hull_heat, critical_heat - 1)
    ]
    raw_fields.extend(
        ("heat", vertex, 0, signature.dynamic_schema.heat_maximum)
        for vertex, signature in enumerate(signatures)
        if signature.dynamic_schema.heat_maximum > 0
    )
    fields = []
    state_variables: list[Hashable] = []
    offset = 0
    for kind, vertex, minimum, maximum in raw_fields:
        width = max(1, (maximum - minimum).bit_length())
        fields.append(IC2SymbolicField(
            kind,
            vertex,
            maximum,
            offset,
            width,
            minimum,
        ))
        state_variables.extend(
            ("state", kind, vertex, bit) for bit in range(width)
        )
        offset += width
    state_tuple = tuple(state_variables)
    manager = ROBDDManager((*parameters, *state_tuple))
    layout_circuit = compile_ic2_schema_conditioned_layout_circuit(
        rows=rows,
        columns=columns,
        exact_rods=exact_rods,
        structural_signatures=signatures,
        manager=manager,
        cell_parameter_variables=cell_parameter_variables,
    )

    bits_by_field = {
        (field.kind, field.vertex): tuple(
            manager.variable(state_tuple[field.offset + bit])
            for bit in range(field.width)
        )
        for field in fields
    }
    hull_field = fields[0]
    input_hull_bits = bits_by_field[("hull", None)]
    hull_code_offset = -hull_field.minimum_safe_value
    field_by_vertex = {
        int(field.vertex): field
        for field in fields
        if field.kind == "heat" and field.vertex is not None
    }
    input_heat_bits = {
        vertex: bits_by_field[("heat", vertex)]
        for vertex in field_by_vertex
    }
    condensator = {
        vertex: all(COMPONENTS[label].kind == "condensator" for label in domains[vertex])
        for vertex in field_by_vertex
    }
    active = tuple(
        signature.power_behavior.accepts_pulse for signature in signatures
    )
    hull_capacity_at_event = []
    running_hull_capacity = 10_000
    for signature in signatures:
        hull_capacity_at_event.append(running_hull_capacity)
        running_hull_capacity += signature.dynamic_schema.hull_capacity_bonus

    bad = unsigned_at_least_constant(
        manager,
        input_hull_bits,
        critical_heat - hull_field.minimum_safe_value,
    )
    for vertex, field in field_by_vertex.items():
        bad = manager.apply(
            "or",
            bad,
            unsigned_at_least_constant(
                manager,
                input_heat_bits[vertex],
                field.maximum_safe_value + 1,
            ),
        )

    def neighbours(vertex: int) -> tuple[int, ...]:
        row, column = divmod(vertex, columns)
        result = []
        if column > 0:
            result.append(vertex - 1)
        if column + 1 < columns:
            result.append(vertex + 1)
        if row > 0:
            result.append(vertex - columns)
        if row + 1 < rows:
            result.append(vertex + columns)
        return tuple(result)

    def exact_boolean_counts(roots: Sequence[int]) -> tuple[int, ...]:
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

    def event_alternative(
        vertex: int,
        label: str,
        *,
        base_hull_bits: Sequence[int] | None = None,
        base_heat_bits: Mapping[int, Sequence[int]] | None = None,
    ) -> tuple[dict[Hashable, int], int]:
        spec = COMPONENTS[label]
        hull_bits = tuple(
            input_hull_bits if base_hull_bits is None else base_hull_bits
        )
        heat_bits = {
            target: tuple(bits)
            for target, bits in (
                input_heat_bits if base_heat_bits is None else base_heat_bits
            ).items()
        }
        failure = 0

        def add_constant_to_store(target: int, value: int) -> None:
            nonlocal failure
            if value == 0:
                return
            field = field_by_vertex[target]
            following, overflow = unsigned_add_constant(
                manager,
                heat_bits[target],
                value,
            )
            failed = manager.apply(
                "or",
                overflow,
                unsigned_at_least_constant(
                    manager,
                    following,
                    field.maximum_safe_value + 1,
                ),
            )
            failure = manager.apply("or", failure, failed)
            heat_bits[target] = following

        def add_constant_to_hull(value: int) -> None:
            nonlocal failure, hull_bits
            if value == 0:
                return
            following, overflow = unsigned_add_constant(manager, hull_bits, value)
            hull_bits = select_bits(
                manager,
                unsigned_at_least_constant(manager, following, hull_code_offset),
                following,
                constant_bits(hull_code_offset, len(hull_bits)),
            )
            failure = manager.apply("or", failure, overflow)

        def add_bits_to_hull(value_bits: Sequence[int]) -> None:
            nonlocal failure, hull_bits
            following, overflow = unsigned_add(
                manager,
                hull_bits,
                value_bits,
                width=len(hull_bits),
            )
            hull_bits = select_bits(
                manager,
                unsigned_at_least_constant(manager, following, hull_code_offset),
                following,
                constant_bits(hull_code_offset, len(hull_bits)),
            )
            failure = manager.apply("or", failure, overflow)

        def distribute_one_fuel_rod(acceptors: Sequence[int], heat: int) -> None:
            nonlocal failure
            remaining_width = max(1, heat.bit_length())
            remaining = constant_bits(heat, remaining_width)
            active_acceptors = tuple(
                manager.negate(unsigned_at_least_constant(
                    manager,
                    heat_bits[target],
                    field_by_vertex[target].maximum_safe_value,
                ))
                if condensator[target]
                else 1
                for target in acceptors
            )
            for position, target in enumerate(acceptors):
                is_active = active_acceptors[position]
                suffix_counts = exact_boolean_counts(active_acceptors[position:])
                store = heat_bits[target]
                field = field_by_vertex[target]
                updated_store = store
                updated_remaining = remaining
                for count in range(1, len(suffix_counts)):
                    condition = manager.apply("and", is_active, suffix_counts[count])
                    amount = unsigned_lookup(
                        manager,
                        remaining,
                        tuple(value // count for value in range(1 << remaining_width)),
                        output_width=remaining_width,
                    )
                    base_remaining = unsigned_lookup(
                        manager,
                        remaining,
                        tuple(
                            value - value // count
                            for value in range(1 << remaining_width)
                        ),
                        output_width=remaining_width,
                    )
                    if condensator[target]:
                        full_width = field.width + 1
                        total, _overflow = unsigned_add(
                            manager,
                            (*store, 0),
                            (*amount, *((0,) * (full_width - len(amount)))),
                            width=full_width,
                        )
                        over_capacity = unsigned_at_least_constant(
                            manager,
                            total,
                            field.maximum_safe_value + 1,
                        )
                        stored_full = select_bits(
                            manager,
                            over_capacity,
                            constant_bits(field.maximum_safe_value, full_width),
                            total,
                        )
                        candidate_store = stored_full[:field.width]
                        returned_full = unsigned_subtract_constant_floor_zero(
                            manager,
                            total,
                            field.maximum_safe_value,
                        )
                        candidate_remaining, _carry = unsigned_add(
                            manager,
                            base_remaining,
                            returned_full[:remaining_width],
                            width=remaining_width,
                        )
                    else:
                        candidate_store, overflow = unsigned_add(
                            manager,
                            store,
                            amount,
                            width=field.width,
                        )
                        failed = manager.apply(
                            "or",
                            overflow,
                            unsigned_at_least_constant(
                                manager,
                                candidate_store,
                                field.maximum_safe_value + 1,
                            ),
                        )
                        failure = manager.apply(
                            "or",
                            failure,
                            manager.apply("and", condition, failed),
                        )
                        candidate_remaining = base_remaining
                    updated_store = select_bits(
                        manager,
                        condition,
                        candidate_store,
                        updated_store,
                    )
                    updated_remaining = select_bits(
                        manager,
                        condition,
                        candidate_remaining,
                        updated_remaining,
                    )
                heat_bits[target] = updated_store
                remaining = updated_remaining
            add_bits_to_hull(remaining)

        def amount_conditions(
            circuit: IC2ExchangeAmountCircuit,
        ) -> tuple[tuple[int, int], ...]:
            return tuple(
                (
                    value,
                    unsigned_equals_constant(
                        manager,
                        circuit.amount_code_bits,
                        value + circuit.limit,
                    ),
                )
                for value in range(-circuit.limit, circuit.limit + 1)
            )

        def normalize_increment_cases(
            cases: Sequence[tuple[int, int]],
        ) -> tuple[tuple[int, int], ...]:
            by_increment: dict[int, int] = {}
            for increment, condition in cases:
                by_increment[increment] = manager.apply(
                    "or",
                    by_increment.get(increment, 0),
                    condition,
                )
            return tuple(sorted(by_increment.items()))

        def exchange_with_target(
            target: int,
            circuit: IC2ExchangeAmountCircuit,
            active_root: int,
        ) -> tuple[tuple[int, int], ...]:
            nonlocal failure
            field = field_by_vertex[target]
            store = heat_bits[target]
            updated_store = store
            cases: list[tuple[int, int]] = []
            if condensator[target]:
                cases.append((0, manager.negate(active_root)))

            for value, amount_root in amount_conditions(circuit):
                condition = manager.apply("and", active_root, amount_root)
                if condensator[target]:
                    if value <= 0:
                        cases.append((0, condition))
                        continue
                    full_width = field.width + 1
                    total, _overflow = unsigned_add_constant(
                        manager,
                        (*store, 0),
                        value,
                    )
                    over_capacity = unsigned_at_least_constant(
                        manager,
                        total,
                        field.maximum_safe_value + 1,
                    )
                    stored_full = select_bits(
                        manager,
                        over_capacity,
                        constant_bits(field.maximum_safe_value, full_width),
                        total,
                    )
                    updated_store = select_bits(
                        manager,
                        condition,
                        stored_full[:field.width],
                        updated_store,
                    )
                    for accepted in range(1, value):
                        heat_condition = unsigned_equals_constant(
                            manager,
                            store,
                            field.maximum_safe_value - accepted,
                        )
                        cases.append((
                            -accepted,
                            manager.conjunction(condition, heat_condition),
                        ))
                    full_acceptance = manager.negate(unsigned_at_least_constant(
                        manager,
                        store,
                        field.maximum_safe_value - value + 1,
                    ))
                    cases.append((
                        -value,
                        manager.conjunction(condition, full_acceptance),
                    ))
                    continue

                if value >= 0:
                    candidate, overflow = unsigned_add_constant(manager, store, value)
                    failed = manager.apply(
                        "or",
                        overflow,
                        unsigned_at_least_constant(
                            manager,
                            candidate,
                            field.maximum_safe_value + 1,
                        ),
                    )
                    failure = manager.apply(
                        "or",
                        failure,
                        manager.apply("and", condition, failed),
                    )
                    updated_store = select_bits(
                        manager,
                        condition,
                        candidate,
                        updated_store,
                    )
                    cases.append((-value, condition))
                else:
                    magnitude = -value
                    candidate = unsigned_subtract_constant_floor_zero(
                        manager,
                        store,
                        magnitude,
                    )
                    updated_store = select_bits(
                        manager,
                        condition,
                        candidate,
                        updated_store,
                    )
                    for transferred in range(magnitude):
                        heat_condition = unsigned_equals_constant(
                            manager,
                            store,
                            transferred,
                        )
                        cases.append((
                            transferred,
                            manager.conjunction(condition, heat_condition),
                        ))
                    cases.append((
                        magnitude,
                        manager.conjunction(
                            condition,
                            unsigned_at_least_constant(manager, store, magnitude),
                        ),
                    ))
            heat_bits[target] = updated_store
            return normalize_increment_cases(cases)

        def exchange_with_hull(
            circuit: IC2ExchangeAmountCircuit,
        ) -> tuple[tuple[int, int], ...]:
            nonlocal failure, hull_bits
            if not circuit.negative_amount_never_exceeds_target:
                raise ValueError("exchange rule could drive encoded hull heat negative")
            updated_hull = hull_bits
            cases = []
            for value, condition in amount_conditions(circuit):
                if value >= 0:
                    candidate, overflow = unsigned_add_constant(
                        manager,
                        hull_bits,
                        value,
                    )
                    failure = manager.apply(
                        "or",
                        failure,
                        manager.apply("and", condition, overflow),
                    )
                else:
                    candidate = unsigned_subtract_constant_floor_zero(
                        manager,
                        hull_bits,
                        -value,
                    )
                updated_hull = select_bits(
                    manager,
                    condition,
                    candidate,
                    updated_hull,
                )
                cases.append((-value, condition))
            hull_bits = updated_hull
            return normalize_increment_cases(cases)

        def apply_exchange_source_delta(
            distribution: Mapping[int, int],
        ) -> None:
            nonlocal failure
            store = heat_bits[vertex]
            field = field_by_vertex[vertex]
            updated_store = store
            for delta, condition in distribution.items():
                if delta >= 0:
                    candidate, overflow = unsigned_add_constant(manager, store, delta)
                    failed = manager.apply(
                        "or",
                        overflow,
                        unsigned_at_least_constant(
                            manager,
                            candidate,
                            field.maximum_safe_value + 1,
                        ),
                    )
                    failure = manager.apply(
                        "or",
                        failure,
                        manager.apply("and", condition, failed),
                    )
                else:
                    candidate = unsigned_subtract_constant_floor_zero(
                        manager,
                        store,
                        -delta,
                    )
                updated_store = select_bits(
                    manager,
                    condition,
                    candidate,
                    updated_store,
                )
            heat_bits[vertex] = updated_store

        def process_exchanger() -> None:
            source_store = heat_bits[vertex]
            active_side_targets = tuple(
                target
                for target in neighbours(vertex)
                if target in heat_bits
            ) if spec.exchange_side else ()
            delta_partitions: list[tuple[int, tuple[tuple[int, int], ...]]] = []

            if spec.exchange_side:
                for target in active_side_targets:
                    active_root = (
                        manager.negate(unsigned_at_least_constant(
                            manager,
                            heat_bits[target],
                            field_by_vertex[target].maximum_safe_value,
                        ))
                        if condensator[target]
                        else 1
                    )
                    circuit = compile_ic2_exchange_amount_circuit(
                        manager,
                        source_store,
                        heat_bits[target],
                        source_capacity=spec.max_heat,
                        target_capacity=field_by_vertex[target].maximum_safe_value,
                        limit=spec.exchange_side,
                    )
                    delta_partitions.append((
                        spec.exchange_side,
                        exchange_with_target(target, circuit, active_root),
                    ))
            if spec.exchange_hull:
                circuit = compile_ic2_exchange_amount_circuit(
                    manager,
                    source_store,
                    hull_bits,
                    source_capacity=spec.max_heat,
                    target_capacity=hull_capacity_at_event[vertex],
                    target_heat_minimum=hull_field.minimum_safe_value,
                    target_heat_maximum=critical_heat - 1,
                    limit=spec.exchange_hull,
                    rounded_base=True,
                    low_range=spec.exchange_side,
                )
                delta_partitions.append((
                    spec.exchange_hull,
                    exchange_with_hull(circuit),
                ))
            delta_code, delta_bias = signed_case_sum_bits(
                manager,
                delta_partitions,
            )
            delta_distribution = {
                code - delta_bias: condition
                for code in range(2 * delta_bias + 1)
                if (
                    condition := unsigned_equals_constant(
                        manager,
                        delta_code,
                        code,
                    )
                ) != 0
            }
            apply_exchange_source_delta(delta_distribution)

        if spec.kind == "fuel":
            pulses = spec.internal_pulses + sum(
                active[target] for target in neighbours(vertex)
            )
            per_rod_heat = 2 * pulses * (pulses + 1)
            acceptors = tuple(
                target for target in neighbours(vertex) if target in heat_bits
            )
            if not acceptors:
                add_constant_to_hull(spec.rod_count * per_rod_heat)
            elif any(condensator[target] for target in acceptors):
                for _rod in range(spec.rod_count):
                    distribute_one_fuel_rod(acceptors, per_rod_heat)
            else:
                remaining = per_rod_heat
                shares = []
                for count in range(len(acceptors), 0, -1):
                    amount = remaining // count
                    remaining -= amount
                    shares.append(amount)
                for target, share in zip(acceptors, shares, strict=True):
                    add_constant_to_store(target, spec.rod_count * share)
        elif spec.kind == "vent":
            adjacent = neighbours(vertex)
            if spec.side_vent:
                for target in adjacent:
                    if target in heat_bits and not condensator[target]:
                        heat_bits[target] = unsigned_subtract_constant_floor_zero(
                            manager,
                            heat_bits[target],
                            spec.side_vent,
                        )
            else:
                if spec.hull_draw:
                    store = heat_bits[vertex]
                    field = field_by_vertex[vertex]
                    enough_hull = unsigned_at_least_constant(
                        manager,
                        hull_bits,
                        hull_code_offset + spec.hull_draw,
                    )
                    physical_hull = unsigned_subtract_constant_floor_zero(
                        manager,
                        hull_bits,
                        hull_code_offset,
                    )
                    hull_for_store = (
                        *physical_hull[:field.width],
                        *((0,) * max(0, field.width - len(physical_hull))),
                    )[:field.width]
                    drawn = select_bits(
                        manager,
                        enough_hull,
                        constant_bits(spec.hull_draw, field.width),
                        hull_for_store,
                    )
                    following, overflow = unsigned_add(
                        manager,
                        store,
                        drawn,
                        width=field.width,
                    )
                    failed = manager.apply(
                        "or",
                        overflow,
                        unsigned_at_least_constant(
                            manager,
                            following,
                            field.maximum_safe_value + 1,
                        ),
                    )
                    failure = manager.apply("or", failure, failed)
                    updated_store = following
                    for magnitude in range(1, hull_code_offset + 1):
                        negative_hull = unsigned_equals_constant(
                            manager,
                            hull_bits,
                            hull_code_offset - magnitude,
                        )
                        updated_store = select_bits(
                            manager,
                            negative_hull,
                            unsigned_subtract_constant_floor_zero(
                                manager,
                                store,
                                magnitude,
                            ),
                            updated_store,
                        )
                    heat_bits[vertex] = updated_store
                    hull_bits = select_bits(
                        manager,
                        enough_hull,
                        unsigned_subtract_constant_floor_zero(
                            manager,
                            hull_bits,
                            spec.hull_draw,
                        ),
                        constant_bits(hull_code_offset, len(hull_bits)),
                    )
                if spec.self_vent:
                    heat_bits[vertex] = unsigned_subtract_constant_floor_zero(
                        manager,
                        heat_bits[vertex],
                        spec.self_vent,
                    )
        elif spec.kind == "exchanger":
            process_exchanger()

        following_by_field = {("hull", None): hull_bits}
        following_by_field.update(
            {("heat", target): bits for target, bits in heat_bits.items()}
        )
        functions = {}
        for field in fields:
            following = following_by_field[(field.kind, field.vertex)]
            for bit in range(field.width):
                functions[state_tuple[field.offset + bit]] = following[bit]
        return functions, failure

    if partitioned_events_only:
        shared_manager = manager
        shared_identity = {
            variable: shared_manager.variable(variable)
            for variable in state_tuple
        }
        local_events = []
        local_event_nodes = []
        local_event_supports = []
        skipped_identity_events = 0
        compiled_alternatives = 0
        peak_alternative_allocated = 0
        peak_alternative_live = 0
        for vertex, domain in enumerate(domains):
            if all(
                COMPONENTS[label].kind not in {"fuel", "vent", "exchanger"}
                for label in domain
            ):
                skipped_identity_events += 1
                continue

            selected = dict(shared_identity)
            selected_failure = 0
            conditions = layout_circuit.label_roots[vertex]
            for label, condition in zip(domain, conditions, strict=True):
                if shared_manager.apply(
                    "and",
                    layout_circuit.feasible_parameter_root,
                    condition,
                ) == 0:
                    continue
                # Compile one raw local alternative in isolation.  It sees the
                # pre-event state variables, never functions accumulated from
                # earlier cells.
                alternative_manager = ROBDDManager(state_tuple)
                manager = alternative_manager
                alternative_identity = tuple(
                    manager.variable(variable) for variable in state_tuple
                )
                alternative_hull = tuple(
                    alternative_identity[hull_field.offset + bit]
                    for bit in range(hull_field.width)
                )
                alternative_heat = {
                    target: tuple(
                        alternative_identity[field.offset + bit]
                        for bit in range(field.width)
                    )
                    for target, field in field_by_vertex.items()
                }
                functions, failure = event_alternative(
                    vertex,
                    label,
                    base_hull_bits=alternative_hull,
                    base_heat_bits=alternative_heat,
                )
                changed_indices = tuple(
                    index
                    for index, variable in enumerate(state_tuple)
                    if functions[variable] != alternative_identity[index]
                )
                roots = tuple(
                    functions[state_tuple[index]] for index in changed_indices
                )
                peak_alternative_allocated = max(
                    peak_alternative_allocated,
                    manager.allocated_node_count,
                )
                normalized_manager, normalized, _key = manager.canonicalize_roots(
                    (*roots, failure)
                )
                peak_alternative_live = max(
                    peak_alternative_live,
                    normalized_manager.allocated_node_count,
                )
                imported = shared_manager.import_roots(
                    normalized_manager,
                    normalized,
                )
                manager = shared_manager
                imported_by_index = dict(zip(
                    changed_indices,
                    imported[:-1],
                    strict=True,
                ))
                for index, variable in enumerate(state_tuple):
                    selected[variable] = manager.ite(
                        condition,
                        imported_by_index.get(index, shared_identity[variable]),
                        selected[variable],
                    )
                selected_failure = manager.apply(
                    "or",
                    selected_failure,
                    manager.apply("and", condition, imported[-1]),
                )
                compiled_alternatives += 1

            changed = {
                variable: selected[variable]
                for variable in state_tuple
                if selected[variable] != shared_identity[variable]
            }
            if not changed and selected_failure == 0:
                skipped_identity_events += 1
                continue
            local_event = SymbolicLocalEvent.from_mapping(
                f"slot-{vertex}",
                changed,
                selected_failure,
            )
            local_events.append(local_event)
            event_roots = (
                *(root for _variable, root in local_event.changed_next_functions),
                local_event.failure_root,
            )
            local_event_nodes.append(manager.reachable_union_node_count(event_roots))
            local_event_supports.append(len(frozenset().union(*(
                manager.support(root) for root in event_roots
            ))))

        manager = shared_manager
        return IC2StructuralLocalEventModel(
            rows=rows,
            columns=columns,
            structural_signatures=signatures,
            layout_circuit=layout_circuit,
            manager=manager,
            parameter_variables=parameters,
            state_variables=state_tuple,
            fields=tuple(fields),
            local_events=tuple(local_events),
            bad_root=bad,
            parameter_constraint_root=layout_circuit.feasible_parameter_root,
            critical_heat=critical_heat,
            encoded_state_count=1 << len(state_tuple),
            compiled_alternative_count=compiled_alternatives,
            skipped_identity_event_count=skipped_identity_events,
            peak_alternative_allocated_nodes=peak_alternative_allocated,
            peak_alternative_live_nodes=peak_alternative_live,
            local_event_live_node_counts=tuple(local_event_nodes),
            local_event_support_sizes=tuple(local_event_supports),
        )

    shared_manager = manager
    thermal_manager = ROBDDManager(state_tuple)
    manager = thermal_manager
    thermal_identity = tuple(manager.variable(variable) for variable in state_tuple)
    thermal_hull_bits = tuple(
        thermal_identity[hull_field.offset + bit]
        for bit in range(hull_field.width)
    )
    thermal_heat_bits = {
        target: tuple(
            thermal_identity[field.offset + bit]
            for bit in range(field.width)
        )
        for target, field in field_by_vertex.items()
    }
    thermal_bad = unsigned_at_least_constant(
        manager,
        thermal_hull_bits,
        critical_heat - hull_field.minimum_safe_value,
    )
    for target, field in field_by_vertex.items():
        thermal_bad = manager.apply(
            "or",
            thermal_bad,
            unsigned_at_least_constant(
                manager,
                thermal_heat_bits[target],
                field.maximum_safe_value + 1,
            ),
        )
    thermal_manager, initial_roots, initial_key = thermal_manager.canonicalize_roots(
        (*thermal_identity, thermal_bad)
    )
    thermal_identity = initial_roots[:-1]
    thermal_bad = initial_roots[-1]
    manager = shared_manager
    # Each future-behaviour class owns a thermal-only manager.  The dictionary
    # key is the complete canonical forest, never a probabilistic digest.
    ForestKey = tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]
    Branch = tuple[ROBDDManager, tuple[int, ...], int, int]
    branches: dict[ForestKey, Branch] = {
        initial_key: (
            thermal_manager,
            thermal_identity,
            thermal_bad,
            layout_circuit.feasible_parameter_root,
        )
    }
    skipped_identity_events = 0
    compiled_alternatives = 0
    peak_behavior_classes = 1
    peak_branch_allocated = thermal_manager.allocated_node_count
    peak_canonical_branch = thermal_manager.allocated_node_count

    def insert_normalized_branch(
        target: dict[ForestKey, Branch],
        branch_manager: ROBDDManager,
        functions: Sequence[int],
        failure: int,
        region: int,
    ) -> ForestKey:
        nonlocal peak_branch_allocated, peak_canonical_branch
        peak_branch_allocated = max(
            peak_branch_allocated,
            branch_manager.allocated_node_count,
        )
        normalized_manager, roots, key = branch_manager.canonicalize_roots(
            (*functions, failure)
        )
        peak_canonical_branch = max(
            peak_canonical_branch,
            normalized_manager.allocated_node_count,
        )
        existing = target.get(key)
        if existing is None:
            target[key] = (
                normalized_manager,
                roots[:-1],
                roots[-1],
                region,
            )
        else:
            old_manager, old_functions, old_failure, old_region = existing
            target[key] = (
                old_manager,
                old_functions,
                old_failure,
                shared_manager.apply("or", old_region, region),
            )
        return key

    for vertex, domain in enumerate(domains):
        if all(
            COMPONENTS[label].kind not in {"fuel", "vent", "exchanger"}
            for label in domain
        ):
            skipped_identity_events += 1
            continue
        conditions = layout_circuit.label_roots[vertex]
        following_branches: dict[ForestKey, Branch] = {}
        layer_changed = False
        for branch_key, (
            branch_manager,
            branch_functions,
            branch_failure,
            region,
        ) in branches.items():
            for label, condition in zip(domain, conditions, strict=True):
                refined_region = shared_manager.apply("and", region, condition)
                if refined_region == 0:
                    continue
                # Clone only this class's live forest.  Alternative arithmetic
                # therefore cannot retain transient nodes from any sibling.
                work_manager, work_roots = branch_manager.compact_roots(
                    (*branch_functions, branch_failure)
                )
                manager = work_manager
                functions = work_roots[:-1]
                prior_failure = work_roots[-1]
                base_hull = tuple(
                    functions[hull_field.offset + bit]
                    for bit in range(hull_field.width)
                )
                base_heat = {
                    target_vertex: tuple(
                        functions[field.offset + bit]
                        for bit in range(field.width)
                    )
                    for target_vertex, field in field_by_vertex.items()
                }
                event_functions, event_failure = event_alternative(
                    vertex,
                    label,
                    base_hull_bits=base_hull,
                    base_heat_bits=base_heat,
                )
                following_functions = tuple(
                    event_functions[variable] for variable in state_tuple
                )
                following_failure = manager.apply(
                    "or",
                    prior_failure,
                    event_failure,
                )
                key = insert_normalized_branch(
                    following_branches,
                    manager,
                    following_functions,
                    following_failure,
                    refined_region,
                )
                compiled_alternatives += 1
                if key != branch_key:
                    layer_changed = True
        manager = shared_manager
        branches = following_branches
        peak_behavior_classes = max(peak_behavior_classes, len(branches))
        if not layer_changed:
            skipped_identity_events += 1

    final_branches: dict[ForestKey, Branch] = {}
    for branch_manager, functions, prior_failure, region in branches.values():
        manager = branch_manager
        hull_bits = tuple(
            functions[hull_field.offset + bit]
            for bit in range(hull_field.width)
        )
        failure = manager.apply(
            "or",
            prior_failure,
            unsigned_at_least_constant(
                manager,
                hull_bits,
                critical_heat - hull_field.minimum_safe_value,
            ),
        )
        insert_normalized_branch(
            final_branches,
            manager,
            functions,
            failure,
            region,
        )
    branches = final_branches
    compiled_alternatives += len(branches)
    peak_behavior_classes = max(peak_behavior_classes, len(branches))
    imported_branches = []
    for branch_manager, functions, failure, region in branches.values():
        imported = shared_manager.import_roots(
            branch_manager,
            (*functions, failure),
        )
        imported_branches.append((region, imported[:-1], imported[-1]))
    identity = {
        variable: shared_manager.variable(variable) for variable in state_tuple
    }
    raw_next = dict(identity)
    transition_failure = 0
    for region, functions, failure in imported_branches:
        transition_failure = shared_manager.apply(
            "or",
            transition_failure,
            shared_manager.apply("and", region, failure),
        )
        for index, variable in enumerate(state_tuple):
            raw_next[variable] = shared_manager.ite(
                region,
                functions[index],
                raw_next[variable],
            )
    next_functions = {
        variable: shared_manager.apply(
            "or",
            raw_next[variable],
            transition_failure,
        )
        for variable in state_tuple
    }
    manager = shared_manager
    composition = FrozenParameterEventComposition(
        parameter_variables=parameters,
        dynamic_variables=state_tuple,
        valid_parameter_root=layout_circuit.valid_parameter_root,
        raw_next_functions=raw_next,
        transition_failure_root=transition_failure,
        poisoned_next_functions=next_functions,
        event_count=len(domains) + 1,
        skipped_identity_event_count=skipped_identity_events,
        compiled_alternative_count=compiled_alternatives,
        represented_parameter_count=manager.model_count(
            layout_circuit.valid_parameter_root,
            parameters,
        ),
        explicit_family_count=manager.model_count(
            layout_circuit.feasible_parameter_root,
            parameters,
        ),
        allocated_nodes=manager.allocated_node_count,
    )
    return IC2StructuralSymbolicModel(
        rows=rows,
        columns=columns,
        structural_signatures=signatures,
        layout_circuit=layout_circuit,
        manager=manager,
        parameter_variables=parameters,
        state_variables=state_tuple,
        fields=tuple(fields),
        next_functions=composition.poisoned_next_functions,
        bad_root=bad,
        transition_failure_root=composition.transition_failure_root,
        parameter_constraint_root=layout_circuit.feasible_parameter_root,
        critical_heat=critical_heat,
        encoded_state_count=1 << len(state_tuple),
        event_composition=composition,
        peak_behavior_class_count=peak_behavior_classes,
        terminal_behavior_class_count=len(branches),
        peak_branch_allocated_nodes=peak_branch_allocated,
        peak_canonical_branch_nodes=peak_canonical_branch,
    )


def compile_ic2_structural_no_exchange_symbolic_model(
    structural_signatures: Sequence[IC2StructuralSignature],
    *,
    rows: int = 6,
    columns: int,
    exact_rods: int,
) -> IC2StructuralSymbolicModel:
    """Compile all permanent non-exchanger refinements collectively."""

    return _compile_ic2_structural_symbolic_model(
        structural_signatures,
        rows=rows,
        columns=columns,
        exact_rods=exact_rods,
        allow_exchangers=False,
    )


def compile_ic2_structural_closed_symbolic_model(
    structural_signatures: Sequence[IC2StructuralSignature],
    *,
    rows: int = 6,
    columns: int,
    exact_rods: int,
) -> IC2StructuralSymbolicModel:
    """Diagnostic closed-function compiler, including exchanger alternatives.

    Prefer :func:`compile_ic2_structural_symbolic_model`; closing a long event
    sequence can be exponentially larger than its exact local representation.
    """

    return _compile_ic2_structural_symbolic_model(
        structural_signatures,
        rows=rows,
        columns=columns,
        exact_rods=exact_rods,
        allow_exchangers=True,
    )


def compile_ic2_structural_local_event_model(
    structural_signatures: Sequence[IC2StructuralSignature],
    *,
    rows: int = 6,
    columns: int,
    exact_rods: int,
) -> IC2StructuralLocalEventModel:
    """Compile all refinements as an exact, unclosed row-major event program."""

    result = _compile_ic2_structural_symbolic_model(
        structural_signatures,
        rows=rows,
        columns=columns,
        exact_rods=exact_rods,
        allow_exchangers=True,
        partitioned_events_only=True,
    )
    if not isinstance(result, IC2StructuralLocalEventModel):  # pragma: no cover
        raise AssertionError("partitioned IC2 compiler returned a closed model")
    return result


def compile_ic2_structural_symbolic_model(
    structural_signatures: Sequence[IC2StructuralSignature],
    *,
    rows: int = 6,
    columns: int,
    exact_rods: int,
) -> IC2StructuralLocalEventModel:
    """Compile the preferred exact layout-parametric local event model."""

    return compile_ic2_structural_local_event_model(
        structural_signatures,
        rows=rows,
        columns=columns,
        exact_rods=exact_rods,
    )
