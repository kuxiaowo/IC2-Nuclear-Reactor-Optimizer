"""Direct symbolic compiler for IC2 layouts whose only thermal store is hull."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from time import perf_counter
from typing import Hashable, Sequence

from .components import COMPONENTS
from .robdd import ROBDDManager
from .robdd_bitvector import unsigned_add_constant, unsigned_at_least_constant


@dataclass(frozen=True, slots=True)
class IC2HullOnlySymbolicModel:
    layout: tuple[str, ...]
    manager: ROBDDManager
    state_variables: tuple[Hashable, ...]
    next_functions: dict[Hashable, int]
    bad_root: int
    generated_heat: int
    critical_heat: int
    encoded_state_count: int
    compile_seconds: float

    def assignment(self, code: int) -> dict[Hashable, bool]:
        if not 0 <= code < self.encoded_state_count:
            raise ValueError("hull symbolic code is outside the model")
        return {
            variable: bool(code >> bit & 1)
            for bit, variable in enumerate(self.state_variables)
        }


def compile_ic2_hull_only_symbolic_model(
    layout: Sequence[str],
) -> IC2HullOnlySymbolicModel:
    """Compile exact one-tick functions without enumerating thermal states."""

    started = perf_counter()
    layout_tuple = tuple(layout)
    if not layout_tuple or len(layout_tuple) % 6:
        raise ValueError("IC2 hull-only layout must contain six complete rows")
    columns = len(layout_tuple) // 6
    if not 3 <= columns <= 9:
        raise ValueError("IC2 hull-only layout columns must lie in 3..9")
    if unknown := set(layout_tuple) - COMPONENTS.keys():
        raise ValueError(f"unknown IC2 hull-only components: {sorted(unknown)}")
    if heat_stores := {
        label for label in layout_tuple if COMPONENTS[label].accepts_heat
    }:
        raise ValueError(
            f"hull-only compiler cannot encode component heat stores: {sorted(heat_stores)}"
        )
    if finite_damage := {
        label
        for label in layout_tuple
        if COMPONENTS[label].kind != "fuel" and COMPONENTS[label].max_damage > 0
    }:
        raise ValueError(
            f"hull-only compiler cannot encode finite damage: {sorted(finite_damage)}"
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
        if row < 5:
            result.append(vertex + columns)
        return tuple(result)

    active = tuple(
        COMPONENTS[label].kind in {"fuel", "reflector"}
        for label in layout_tuple
    )
    generated_heat = 0
    for vertex, label in enumerate(layout_tuple):
        spec = COMPONENTS[label]
        if spec.kind != "fuel":
            continue
        pulses = spec.internal_pulses + sum(active[item] for item in neighbours(vertex))
        generated_heat += 2 * spec.rod_count * pulses * (pulses + 1)

    maximum_hull_heat = 10_000 + sum(
        COMPONENTS[label].hull_capacity_bonus for label in layout_tuple
    )
    critical_heat = floor(maximum_hull_heat * 0.85)
    width = max(1, critical_heat.bit_length())
    variables: tuple[Hashable, ...] = tuple(
        ("hull", bit) for bit in range(width)
    )
    manager = ROBDDManager(variables)
    current_bits = tuple(manager.variable(variable) for variable in variables)
    bad_root = unsigned_at_least_constant(manager, current_bits, critical_heat)
    sum_bits, overflow = unsigned_add_constant(
        manager,
        current_bits,
        generated_heat,
    )
    # Crossing the critical threshold or overflowing is a conclusive failure.
    # Map both to the same all-one canonical bad code as the semantic oracle.
    following_bad = manager.apply(
        "or",
        overflow,
        unsigned_at_least_constant(manager, sum_bits, critical_heat),
    )
    following_bits = tuple(
        manager.apply("or", bit, following_bad) for bit in sum_bits
    )
    return IC2HullOnlySymbolicModel(
        layout=layout_tuple,
        manager=manager,
        state_variables=variables,
        next_functions=dict(zip(variables, following_bits, strict=True)),
        bad_root=bad_root,
        generated_heat=generated_heat,
        critical_heat=critical_heat,
        encoded_state_count=1 << width,
        compile_seconds=perf_counter() - started,
    )
