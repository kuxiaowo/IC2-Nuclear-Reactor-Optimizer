"""Direct IC2 symbolic event compiler excluding heat exchangers."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from time import perf_counter
from typing import Hashable, Mapping, Sequence

from .components import COMPONENTS
from .ic2_exchange_mtbdd import (
    IC2ExchangeAmountCircuit,
    compile_ic2_exchange_amount_circuit,
)
from .ic2_symbolic_oracle import IC2SymbolicField
from .robdd import ROBDDManager
from .robdd_bitvector import (
    constant_bits,
    select_bits,
    unsigned_add,
    unsigned_add_constant,
    unsigned_at_least_constant,
    unsigned_equals_constant,
    unsigned_lookup,
    unsigned_subtract_constant_floor_zero,
)


@dataclass(frozen=True, slots=True)
class IC2NoExchangeSymbolicModel:
    layout: tuple[str, ...]
    manager: ROBDDManager
    state_variables: tuple[Hashable, ...]
    fields: tuple[IC2SymbolicField, ...]
    next_functions: dict[Hashable, int]
    bad_root: int
    transition_failure_root: int
    generated_heat: int
    critical_heat: int
    encoded_state_count: int
    compile_seconds: float
    peak_allocated_nodes: int
    event_compactions: int

    def assignment(self, code: int) -> dict[Hashable, bool]:
        if not 0 <= code < self.encoded_state_count:
            raise ValueError("symbolic state code is outside the model")
        return {
            variable: bool(code >> bit & 1)
            for bit, variable in enumerate(self.state_variables)
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

    def next_code(self, code: int) -> int:
        assignment = self.assignment(code)
        return sum(
            int(self.manager.evaluate(self.next_functions[variable], assignment)) << bit
            for bit, variable in enumerate(self.state_variables)
        )


def _compile_ic2_symbolic_model(
    layout: Sequence[str],
    *,
    allow_exchangers: bool,
) -> IC2NoExchangeSymbolicModel:
    """Compile the exact row-major heat pass as partitioned Boolean circuits."""

    started = perf_counter()
    layout_tuple = tuple(layout)
    if not layout_tuple or len(layout_tuple) % 6:
        raise ValueError("IC2 symbolic layout must contain six complete rows")
    columns = len(layout_tuple) // 6
    if not 3 <= columns <= 9:
        raise ValueError("IC2 symbolic layout columns must lie in 3..9")
    if unknown := set(layout_tuple) - COMPONENTS.keys():
        raise ValueError(f"unknown IC2 symbolic components: {sorted(unknown)}")
    if unsupported := {
        label
        for label in layout_tuple
        if (COMPONENTS[label].kind == "exchanger" and not allow_exchangers)
        or (
            COMPONENTS[label].kind != "fuel"
            and COMPONENTS[label].max_damage > 0
        )
    }:
        raise ValueError(
            "symbolic compiler does not support the selected exchangers or "
            f"finite damage: {sorted(unsupported)}"
        )

    maximum_hull_heat = 10_000 + sum(
        COMPONENTS[label].hull_capacity_bonus for label in layout_tuple
    )
    critical_heat = floor(maximum_hull_heat * 0.85)
    minimum_hull_heat = -72 if allow_exchangers else 0
    raw_fields: list[tuple[str, int | None, int, int]] = [
        ("hull", None, minimum_hull_heat, critical_heat - 1)
    ]
    raw_fields.extend(
        ("heat", vertex, 0, COMPONENTS[label].max_heat)
        for vertex, label in enumerate(layout_tuple)
        if COMPONENTS[label].accepts_heat
    )
    fields = []
    variables: list[Hashable] = []
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
        variables.extend((kind, vertex, bit) for bit in range(width))
        offset += width
    manager = ROBDDManager(tuple(variables))
    bits_by_field = {
        (field.kind, field.vertex): tuple(
            manager.variable(variables[field.offset + bit])
            for bit in range(field.width)
        )
        for field in fields
    }
    field_by_vertex = {
        int(field.vertex): field
        for field in fields
        if field.kind == "heat" and field.vertex is not None
    }
    hull_field = fields[0]
    hull_bits = bits_by_field[("hull", None)]
    heat_bits = {
        vertex: bits_by_field[("heat", vertex)]
        for vertex in field_by_vertex
    }

    hull_code_offset = -hull_field.minimum_safe_value
    bad = unsigned_at_least_constant(
        manager,
        hull_bits,
        critical_heat - hull_field.minimum_safe_value,
    )
    for vertex, field in field_by_vertex.items():
        bad = manager.apply(
            "or",
            bad,
            unsigned_at_least_constant(
                manager,
                heat_bits[vertex],
                field.maximum_safe_value - field.minimum_safe_value + 1,
            ),
        )
    failure = bad
    generated_heat = 0

    def neighbours(vertex: int) -> tuple[int, ...]:
        row, column = divmod(vertex, columns)
        result = []
        if column > 0:
            result.append(vertex - 1)
        if column + 1 < columns:
            result.append(vertex + 1)
        if row > 0:
            result.append(vertex - columns)
        if row < 5:
            result.append(vertex + columns)
        return tuple(result)

    active = tuple(
        COMPONENTS[label].kind in {"fuel", "reflector"}
        for label in layout_tuple
    )

    def add_constant_to_store(vertex: int, value: int) -> None:
        nonlocal failure
        if value == 0:
            return
        field = field_by_vertex[vertex]
        following, overflow = unsigned_add_constant(
            manager,
            heat_bits[vertex],
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
        heat_bits[vertex] = following

    def add_to_hull(value: int) -> None:
        nonlocal failure, hull_bits
        if value:
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

    def distribute_one_fuel_rod(
        acceptors: Sequence[int],
        heat: int,
    ) -> None:
        """Compile the exact conditional condensator distribution loop."""

        nonlocal failure
        remaining_width = max(1, heat.bit_length())
        remaining = constant_bits(heat, remaining_width)
        active_acceptors = tuple(
            (
                manager.negate(unsigned_at_least_constant(
                    manager,
                    heat_bits[target],
                    COMPONENTS[layout_tuple[target]].max_heat,
                ))
                if COMPONENTS[layout_tuple[target]].kind == "condensator"
                else 1
            )
            for target in acceptors
        )
        for position, target in enumerate(acceptors):
            active = active_acceptors[position]
            suffix_counts = exact_boolean_counts(active_acceptors[position:])
            store = heat_bits[target]
            field = field_by_vertex[target]
            updated_store = store
            updated_remaining = remaining
            for count in range(1, len(suffix_counts)):
                condition = manager.apply("and", active, suffix_counts[count])
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
                if COMPONENTS[layout_tuple[target]].kind == "condensator":
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

    def advance_delta_distribution(
        distribution: Mapping[int, int],
        cases: Sequence[tuple[int, int]],
    ) -> dict[int, int]:
        following: dict[int, int] = {}
        for old_delta, old_condition in distribution.items():
            for increment, case_condition in cases:
                new_delta = old_delta + increment
                joint = manager.apply("and", old_condition, case_condition)
                following[new_delta] = manager.apply(
                    "or",
                    following.get(new_delta, 0),
                    joint,
                )
        return {value: root for value, root in following.items() if root != 0}

    def exchange_with_target(
        target: int,
        circuit: IC2ExchangeAmountCircuit,
        active_root: int,
    ) -> tuple[tuple[int, int], ...]:
        """Update one side target and return source-heat delta cases."""

        nonlocal failure
        target_spec = COMPONENTS[layout_tuple[target]]
        field = field_by_vertex[target]
        store = heat_bits[target]
        updated_store = store
        cases: list[tuple[int, int]] = []
        if target_spec.kind == "condensator":
            cases.append((0, manager.negate(active_root)))

        for value, amount_root in amount_conditions(circuit):
            condition = manager.apply("and", active_root, amount_root)
            if target_spec.kind == "condensator":
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
        """Update hull target and return source-heat delta cases."""

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
        vertex: int,
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

    def process_exchanger(vertex: int, hull_capacity: int) -> None:
        spec = COMPONENTS[layout_tuple[vertex]]
        source_store = heat_bits[vertex]
        delta_distribution: dict[int, int] = {0: 1}
        if spec.exchange_side:
            for target in neighbours(vertex):
                target_spec = COMPONENTS[layout_tuple[target]]
                if not target_spec.accepts_heat:
                    continue
                active_root = (
                    manager.negate(unsigned_at_least_constant(
                        manager,
                        heat_bits[target],
                        target_spec.max_heat,
                    ))
                    if target_spec.kind == "condensator"
                    else 1
                )
                circuit = compile_ic2_exchange_amount_circuit(
                    manager,
                    source_store,
                    heat_bits[target],
                    source_capacity=spec.max_heat,
                    target_capacity=target_spec.max_heat,
                    limit=spec.exchange_side,
                )
                delta_distribution = advance_delta_distribution(
                    delta_distribution,
                    exchange_with_target(target, circuit, active_root),
                )
        if spec.exchange_hull:
            circuit = compile_ic2_exchange_amount_circuit(
                manager,
                source_store,
                hull_bits,
                source_capacity=spec.max_heat,
                target_capacity=hull_capacity,
                target_heat_minimum=hull_field.minimum_safe_value,
                target_heat_maximum=critical_heat - 1,
                limit=spec.exchange_hull,
                rounded_base=True,
                low_range=spec.exchange_side,
            )
            delta_distribution = advance_delta_distribution(
                delta_distribution,
                exchange_with_hull(circuit),
            )
        apply_exchange_source_delta(vertex, delta_distribution)

    peak_allocated_nodes = manager.allocated_node_count
    event_compactions = 0

    def compact_event_state() -> None:
        nonlocal bad, event_compactions, failure, heat_bits, hull_bits, manager
        nonlocal peak_allocated_nodes
        peak_allocated_nodes = max(
            peak_allocated_nodes,
            manager.allocated_node_count,
        )
        if manager.allocated_node_count < 200_000:
            return
        heat_vertices = tuple(sorted(heat_bits))
        roots = (bad, failure, *hull_bits, *(bit for vertex in heat_vertices for bit in heat_bits[vertex]))
        live_nodes = manager.reachable_union_node_count(roots)
        if (
            manager.allocated_node_count <= 4 * max(1, live_nodes)
        ):
            return
        manager, compacted = manager.compact_roots(roots)
        cursor = 0
        bad, failure = compacted[cursor:cursor + 2]
        cursor += 2
        hull_bits = compacted[cursor:cursor + len(hull_bits)]
        cursor += len(hull_bits)
        following_heat: dict[int, tuple[int, ...]] = {}
        for vertex in heat_vertices:
            width = len(heat_bits[vertex])
            following_heat[vertex] = compacted[cursor:cursor + width]
            cursor += width
        heat_bits = following_heat
        event_compactions += 1

    current_hull_capacity = 10_000
    for vertex, label in enumerate(layout_tuple):
        spec = COMPONENTS[label]
        adjacent = neighbours(vertex)
        if spec.kind == "fuel":
            pulses = spec.internal_pulses + sum(active[item] for item in adjacent)
            per_rod_heat = 2 * pulses * (pulses + 1)
            generated_heat += spec.rod_count * per_rod_heat
            acceptors = tuple(
                item for item in adjacent if COMPONENTS[layout_tuple[item]].accepts_heat
            )
            if not acceptors:
                add_to_hull(spec.rod_count * per_rod_heat)
            elif any(
                COMPONENTS[layout_tuple[target]].kind == "condensator"
                for target in acceptors
            ):
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
            if spec.side_vent:
                for target in adjacent:
                    if (
                        target in heat_bits
                        and COMPONENTS[layout_tuple[target]].kind != "condensator"
                    ):
                        heat_bits[target] = unsigned_subtract_constant_floor_zero(
                            manager,
                            heat_bits[target],
                            spec.side_vent,
                        )
                compact_event_state()
                continue
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
            process_exchanger(vertex, current_hull_capacity)
        elif spec.kind == "plating":
            current_hull_capacity += spec.hull_capacity_bonus
        compact_event_state()

    hull_failed = unsigned_at_least_constant(
        manager,
        hull_bits,
        critical_heat - hull_field.minimum_safe_value,
    )
    failure = manager.apply("or", failure, hull_failed)
    following_by_field = {("hull", None): hull_bits}
    following_by_field.update(
        {("heat", vertex): bits for vertex, bits in heat_bits.items()}
    )
    next_functions = {}
    for field in fields:
        following = following_by_field[(field.kind, field.vertex)]
        for bit in range(field.width):
            variable = variables[field.offset + bit]
            next_functions[variable] = manager.apply(
                "or",
                following[bit],
                failure,
            )
    function_variables = tuple(next_functions)
    peak_allocated_nodes = max(
        peak_allocated_nodes,
        manager.allocated_node_count,
    )
    manager, compacted_roots = manager.compact_roots((
        bad,
        failure,
        *(next_functions[variable] for variable in function_variables),
    ))
    bad, failure, *compacted_functions = compacted_roots
    next_functions = dict(zip(
        function_variables,
        compacted_functions,
        strict=True,
    ))
    return IC2NoExchangeSymbolicModel(
        layout=layout_tuple,
        manager=manager,
        state_variables=tuple(variables),
        fields=tuple(fields),
        next_functions=next_functions,
        bad_root=bad,
        transition_failure_root=failure,
        generated_heat=generated_heat,
        critical_heat=critical_heat,
        encoded_state_count=1 << offset,
        compile_seconds=perf_counter() - started,
        peak_allocated_nodes=peak_allocated_nodes,
        event_compactions=event_compactions,
    )


def compile_ic2_no_exchange_symbolic_model(
    layout: Sequence[str],
) -> IC2NoExchangeSymbolicModel:
    """Compile the supported official event semantics and reject exchangers."""

    return _compile_ic2_symbolic_model(layout, allow_exchangers=False)


def compile_ic2_symbolic_model(
    layout: Sequence[str],
) -> IC2NoExchangeSymbolicModel:
    """Compile fixed-layout official thermal events, including exchangers."""

    return _compile_ic2_symbolic_model(layout, allow_exchangers=True)
