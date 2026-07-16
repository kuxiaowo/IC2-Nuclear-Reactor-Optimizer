from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .components import COMPONENTS, COMPONENT_IDS
from .mark import mark_family
from .models import StopReason


# These integer codes are a versioned in-process ABI between the search layer
# and future CPU/GPU simulation kernels.  They intentionally do not depend on
# dictionary iteration after module import.
COMPONENT_ID_BY_CODE: tuple[str, ...] = COMPONENT_IDS
COMPONENT_CODE_BY_ID: dict[str, int] = {
    component_id: code for code, component_id in enumerate(COMPONENT_ID_BY_CODE)
}

KIND_CODE: dict[str, int] = {
    "empty": 0,
    "fuel": 1,
    "vent": 2,
    "exchanger": 3,
    "coolant": 4,
    "condensator": 5,
    "plating": 6,
    "reflector": 7,
}

MARK_UNCLASSIFIED = 0
MARK_I = 1
MARK_II = 2
MARK_III = 3
MARK_IV = 4
MARK_V = 5
MARK_FLAG_SUC = 1
MARK_FLAG_STABLE = 2

STOP_REASON_CODE: dict[StopReason, int] = {
    StopReason.MELTDOWN: 1,
    StopReason.TICK_LIMIT: 2,
    StopReason.STABLE: 3,
    StopReason.CANCELLED: 4,
}


def encode_mark(mark: str | None, stable: bool) -> tuple[int, int, int]:
    """Encode the public Mark string into fixed-width kernel result fields."""
    family = mark_family(mark)
    family_code = {
        None: MARK_UNCLASSIFIED,
        "I": MARK_I,
        "II": MARK_II,
        "III": MARK_III,
        "IV": MARK_IV,
        "V": MARK_V,
    }[family]
    level = 0
    if family == "II" and mark is not None:
        level_text = mark.removeprefix("Mark II-").removesuffix("-SUC")
        level = 16 if level_text == "E" else int(level_text)
    flags = 0
    if mark and mark.endswith("-SUC"):
        flags |= MARK_FLAG_SUC
    if stable:
        flags |= MARK_FLAG_STABLE
    return family_code, level, flags


def decode_mark(family: int, level: int, flags: int) -> str | None:
    """Decode fixed-width kernel result fields into the public Mark string."""
    suffix = "-SUC" if flags & MARK_FLAG_SUC else ""
    if family == MARK_UNCLASSIFIED:
        return None
    if family == MARK_I:
        return f"Mark I-I{suffix}"
    if family == MARK_II:
        level_text = "E" if level >= 16 else str(max(1, level))
        return f"Mark II-{level_text}{suffix}"
    family_text = {
        MARK_III: "III",
        MARK_IV: "IV",
        MARK_V: "V",
    }.get(family)
    if family_text is None:
        raise ValueError(f"unknown packed Mark family: {family}")
    return f"Mark {family_text}{suffix}"


def _readonly(values: Sequence[int], dtype) -> NDArray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ComponentKernelTable:
    """Structure-of-arrays component constants suitable for device upload."""

    kind: NDArray[np.uint8]
    max_heat: NDArray[np.int32]
    max_damage: NDArray[np.int32]
    rod_count: NDArray[np.uint8]
    internal_pulses: NDArray[np.uint8]
    self_vent: NDArray[np.int16]
    hull_draw: NDArray[np.int16]
    side_vent: NDArray[np.int16]
    exchange_side: NDArray[np.int16]
    exchange_hull: NDArray[np.int16]
    hull_capacity_bonus: NDArray[np.int32]
    accepts_heat: NDArray[np.uint8]
    is_coolable: NDArray[np.uint8]


def _build_component_table() -> ComponentKernelTable:
    specs = [COMPONENTS[component_id] for component_id in COMPONENT_ID_BY_CODE]
    return ComponentKernelTable(
        kind=_readonly([KIND_CODE[spec.kind] for spec in specs], np.uint8),
        max_heat=_readonly([spec.max_heat for spec in specs], np.int32),
        max_damage=_readonly([spec.max_damage for spec in specs], np.int32),
        rod_count=_readonly([spec.rod_count for spec in specs], np.uint8),
        internal_pulses=_readonly([spec.internal_pulses for spec in specs], np.uint8),
        self_vent=_readonly([spec.self_vent for spec in specs], np.int16),
        hull_draw=_readonly([spec.hull_draw for spec in specs], np.int16),
        side_vent=_readonly([spec.side_vent for spec in specs], np.int16),
        exchange_side=_readonly([spec.exchange_side for spec in specs], np.int16),
        exchange_hull=_readonly([spec.exchange_hull for spec in specs], np.int16),
        hull_capacity_bonus=_readonly([spec.hull_capacity_bonus for spec in specs], np.int32),
        accepts_heat=_readonly([spec.accepts_heat for spec in specs], np.uint8),
        is_coolable=_readonly([spec.is_coolable for spec in specs], np.uint8),
    )


COMPONENT_KERNEL_TABLE = _build_component_table()


@lru_cache(maxsize=7)
def packed_neighbor_table(columns: int) -> NDArray[np.int16]:
    """Return ``(slots, 4)`` neighbors in official left/right/up/down order."""
    if not 3 <= columns <= 9:
        raise ValueError("columns must be between 3 and 9")
    slots = columns * 6
    neighbors = np.full((slots, 4), -1, dtype=np.int16)
    for index in range(slots):
        row, column = divmod(index, columns)
        values: list[int] = []
        if column > 0:
            values.append(index - 1)
        if column + 1 < columns:
            values.append(index + 1)
        if row > 0:
            values.append(index - columns)
        if row < 5:
            values.append(index + columns)
        neighbors[index, :len(values)] = values
    neighbors.setflags(write=False)
    return neighbors


@dataclass(slots=True)
class PackedLayoutBatch:
    """Contiguous fixed-width input batch shared by scalar and device backends."""

    columns: int
    component_codes: NDArray[np.uint8]
    initial_hull_heat: NDArray[np.int32]

    def __post_init__(self) -> None:
        expected_slots = self.columns * 6
        if self.component_codes.dtype != np.uint8 or self.component_codes.ndim != 2:
            raise TypeError("component_codes must be a two-dimensional uint8 array")
        if self.component_codes.shape[1] != expected_slots:
            raise ValueError("component_codes width must equal 6 * columns")
        if not self.component_codes.flags.c_contiguous:
            raise ValueError("component_codes must be C-contiguous")
        if self.initial_hull_heat.dtype != np.int32 or self.initial_hull_heat.ndim != 1:
            raise TypeError("initial_hull_heat must be a one-dimensional int32 array")
        if self.initial_hull_heat.shape[0] != self.component_codes.shape[0]:
            raise ValueError("initial_hull_heat length must match the layout batch")
        if not self.initial_hull_heat.flags.c_contiguous:
            raise ValueError("initial_hull_heat must be C-contiguous")

    @property
    def batch_size(self) -> int:
        return self.component_codes.shape[0]

    @property
    def slots(self) -> int:
        return self.component_codes.shape[1]


@dataclass(slots=True)
class PackedEvaluationBatch:
    """String-free result buffers expected from a future parallel backend."""

    mark_family: NDArray[np.uint8]
    mark_level: NDArray[np.uint8]
    mark_flags: NDArray[np.uint8]
    stop_reason: NDArray[np.uint8]
    reactor_ticks: NDArray[np.int64]
    safe_game_ticks: NDArray[np.int64]
    average_eu_per_tick: NDArray[np.float64]
    total_eu: NDArray[np.float64]
    safety_margin: NDArray[np.float64]

    @property
    def batch_size(self) -> int:
        return self.mark_family.shape[0]


class PackedEvaluator(Protocol):
    """Backend contract; a CUDA implementation can satisfy it without search changes."""

    def evaluate(
        self,
        batch: PackedLayoutBatch,
        max_reactor_ticks: int,
    ) -> PackedEvaluationBatch: ...


def pack_layouts(
    layouts: Sequence[Sequence[str]],
    columns: int,
    initial_hull_heat: int | Sequence[int] = 0,
) -> PackedLayoutBatch:
    """Encode validated string layouts into the stable numeric kernel ABI."""
    slots = columns * 6
    component_codes = np.empty((len(layouts), slots), dtype=np.uint8)
    for row, layout in enumerate(layouts):
        if len(layout) != slots:
            raise ValueError("every layout must contain 6 * columns slots")
        try:
            component_codes[row] = [COMPONENT_CODE_BY_ID[item] for item in layout]
        except KeyError as exc:
            raise ValueError(f"unknown component id: {exc.args[0]}") from exc

    if isinstance(initial_hull_heat, int):
        hull_heat = np.full(len(layouts), initial_hull_heat, dtype=np.int32)
    else:
        hull_heat = np.ascontiguousarray(initial_hull_heat, dtype=np.int32)
        if hull_heat.shape != (len(layouts),):
            raise ValueError("initial_hull_heat length must match layouts")
    return PackedLayoutBatch(columns, component_codes, hull_heat)


def unpack_layout(batch: PackedLayoutBatch, index: int) -> tuple[str, ...]:
    """Decode one batch row for scalar cross-validation and diagnostics."""
    return tuple(COMPONENT_ID_BY_CODE[int(code)] for code in batch.component_codes[index])
