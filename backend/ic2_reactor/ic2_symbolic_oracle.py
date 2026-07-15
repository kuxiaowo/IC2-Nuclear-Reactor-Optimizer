"""Small-state oracle compiler from the locked IC2 transition to ROBDDs.

This module is a semantic validator, not the production 54-slot compiler.  It
enumerates every encoded thermal state only below an explicit bit limit, calls
the official simulator for one transition, and compiles the resulting truth
tables.  The production compiler must build the same bit functions from
partitioned event circuits without state enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from time import perf_counter
from typing import Hashable, Sequence

from .components import COMPONENTS
from .engine import ReactorSimulator
from .models import Layout
from .robdd import ROBDDManager


@dataclass(frozen=True, slots=True)
class IC2SymbolicField:
    kind: str
    vertex: int | None
    maximum_safe_value: int
    offset: int
    width: int
    minimum_safe_value: int = 0


@dataclass(frozen=True, slots=True)
class IC2OracleSymbolicModel:
    layout: tuple[str, ...]
    manager: ROBDDManager
    state_variables: tuple[Hashable, ...]
    fields: tuple[IC2SymbolicField, ...]
    next_functions: dict[Hashable, int]
    bad_root: int
    initial_code: int
    encoded_state_count: int
    compile_seconds: float

    def assignment(self, code: int) -> dict[Hashable, bool]:
        if not 0 <= code < self.encoded_state_count:
            raise ValueError("symbolic state code is outside the model")
        return {
            variable: bool(code >> bit & 1)
            for bit, variable in enumerate(self.state_variables)
        }


def compile_ic2_oracle_symbolic_model(
    layout: Sequence[str],
    *,
    maximum_state_bits: int = 18,
) -> IC2OracleSymbolicModel:
    """Compile a fixed small layout by exhaustive one-step semantic queries."""

    started = perf_counter()
    layout_tuple = tuple(layout)
    if not layout_tuple or len(layout_tuple) % 6:
        raise ValueError("IC2 symbolic layout must contain six complete rows")
    columns = len(layout_tuple) // 6
    if not 3 <= columns <= 9:
        raise ValueError("IC2 symbolic layout columns must lie in 3..9")
    if unknown := set(layout_tuple) - COMPONENTS.keys():
        raise ValueError(f"unknown IC2 symbolic components: {sorted(unknown)}")
    if maximum_state_bits <= 0:
        raise ValueError("maximum symbolic state bits must be positive")

    maximum_hull_heat = 10_000 + sum(
        COMPONENTS[label].hull_capacity_bonus for label in layout_tuple
    )
    critical_heat = floor(maximum_hull_heat * 0.85)
    field_specs: list[tuple[str, int | None, int]] = [
        ("hull", None, critical_heat - 1)
    ]
    for vertex, label in enumerate(layout_tuple):
        spec = COMPONENTS[label]
        if spec.accepts_heat:
            field_specs.append(("heat", vertex, spec.max_heat))
        if spec.kind != "fuel" and spec.max_damage > 0:
            field_specs.append(("damage", vertex, spec.max_damage - 1))

    fields = []
    variables: list[Hashable] = []
    offset = 0
    for kind, vertex, maximum in field_specs:
        width = max(1, int(maximum).bit_length())
        fields.append(IC2SymbolicField(kind, vertex, maximum, offset, width))
        variables.extend(
            (kind, vertex, bit) for bit in range(width)
        )
        offset += width
    if offset > maximum_state_bits:
        raise ValueError(
            f"oracle symbolic model needs {offset} bits, above limit "
            f"{maximum_state_bits}"
        )
    state_count = 1 << offset
    canonical_bad = state_count - 1
    base_layout = Layout(columns=columns, slots=list(layout_tuple))
    bad_values = [False] * state_count
    next_codes = [0] * state_count

    def field_value(code: int, field: IC2SymbolicField) -> int:
        return (code >> field.offset) & ((1 << field.width) - 1)

    def encode(simulator: ReactorSimulator) -> int:
        code = 0
        for field in fields:
            if field.kind == "hull":
                value = simulator.hull_heat
            elif field.kind == "heat":
                assert field.vertex is not None
                value = simulator.slots[field.vertex].heat
            else:
                assert field.vertex is not None
                value = simulator.slots[field.vertex].damage
            if not 0 <= value <= field.maximum_safe_value:
                return canonical_bad
            code |= int(value) << field.offset
        return code

    for code in range(state_count):
        decoded = tuple(field_value(code, field) for field in fields)
        if any(
            value > field.maximum_safe_value
            for value, field in zip(decoded, fields, strict=True)
        ):
            bad_values[code] = True
            next_codes[code] = code
            continue
        simulator = ReactorSimulator(base_layout)
        for value, field in zip(decoded, fields, strict=True):
            if field.kind == "hull":
                simulator.hull_heat = value
                simulator.peak_hull_heat = value
            elif field.kind == "heat":
                assert field.vertex is not None
                simulator.slots[field.vertex].heat = value
            else:
                assert field.vertex is not None
                simulator.slots[field.vertex].damage = value
        simulator.step(auto_refuel=True)
        failed = (
            simulator.first_critical_tick is not None
            or simulator.first_component_break_tick is not None
            or simulator.meltdown_tick is not None
        )
        next_codes[code] = canonical_bad if failed else encode(simulator)

    manager = ROBDDManager(tuple(variables))
    bad_root = manager.from_truth_table(bad_values)
    next_functions = {
        variable: manager.from_truth_table(tuple(
            bool(code >> bit & 1) for code in next_codes
        ))
        for bit, variable in enumerate(variables)
    }
    return IC2OracleSymbolicModel(
        layout=layout_tuple,
        manager=manager,
        state_variables=tuple(variables),
        fields=tuple(fields),
        next_functions=next_functions,
        bad_root=bad_root,
        initial_code=0,
        encoded_state_count=state_count,
        compile_seconds=perf_counter() - started,
    )
