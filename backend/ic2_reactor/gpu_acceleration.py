from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .components import COMPONENTS, COMPONENT_IDS


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    available: bool
    name: str | None = None
    compute_capability: tuple[int, int] | None = None
    reason: str | None = None

    @property
    def label(self) -> str | None:
        if self.name is None:
            return None
        if self.compute_capability is None:
            return self.name
        major, minor = self.compute_capability
        return f"{self.name} (CC {major}.{minor})"


@dataclass(frozen=True, slots=True)
class CudaBatchScores:
    power: np.ndarray
    generated_heat: np.ndarray
    cooling_proxy: np.ndarray


@dataclass(frozen=True, slots=True)
class CudaFixedPointCertificate:
    average_eu_per_tick: float
    safe_game_ticks: int
    peak_hull_heat: int
    max_hull_heat: int


@lru_cache(maxsize=1)
def cuda_device_info() -> CudaDeviceInfo:
    """Probe CUDA lazily so the normal CPU installation stays usable."""
    try:
        from numba import cuda

        if not cuda.is_available():
            return CudaDeviceInfo(
                False,
                reason="未检测到可用 CUDA 运行时；请安装 numba-cuda 与 CUDA Toolkit",
            )
        device = cuda.get_current_device()
        capability = tuple(int(value) for value in device.compute_capability)
        return CudaDeviceInfo(True, str(device.name), capability)
    except Exception as exc:  # CUDA discovery failures vary by driver/toolkit.
        return CudaDeviceInfo(False, reason=f"CUDA 初始化失败：{exc}")


_CUDA_SCORE_KERNEL = None
_CUDA_FIXED_POINT_KERNEL = None


def _score_kernel():
    global _CUDA_SCORE_KERNEL
    if _CUDA_SCORE_KERNEL is not None:
        return _CUDA_SCORE_KERNEL

    from numba import cuda

    @cuda.jit
    def kernel(
        layouts,
        columns,
        rod_counts,
        internal_pulses,
        power_participant,
        self_vent,
        side_vent,
        coolable,
        output_power,
        output_heat,
        output_cooling,
    ):
        layout_index = cuda.grid(1)
        if layout_index >= layouts.shape[0]:
            return

        slots = layouts.shape[1]
        pulse_count = 0
        generated_heat = 0
        cooling_capacity = 0
        for position in range(slots):
            code = layouts[layout_index, position]
            rods = rod_counts[code]
            row = position // columns
            column = position - row * columns

            if rods > 0:
                neighbors = 0
                if column > 0:
                    neighbors += power_participant[layouts[layout_index, position - 1]]
                if column + 1 < columns:
                    neighbors += power_participant[layouts[layout_index, position + 1]]
                if row > 0:
                    neighbors += power_participant[layouts[layout_index, position - columns]]
                if position + columns < slots:
                    neighbors += power_participant[layouts[layout_index, position + columns]]
                pulses = internal_pulses[code] + neighbors
                pulse_count += rods * pulses
                generated_heat += 2 * rods * pulses * (pulses + 1)

            cooling_capacity += self_vent[code]
            spread = side_vent[code]
            if spread > 0:
                if column > 0:
                    cooling_capacity += spread * coolable[layouts[layout_index, position - 1]]
                if column + 1 < columns:
                    cooling_capacity += spread * coolable[layouts[layout_index, position + 1]]
                if row > 0:
                    cooling_capacity += spread * coolable[layouts[layout_index, position - columns]]
                if position + columns < slots:
                    cooling_capacity += spread * coolable[layouts[layout_index, position + columns]]

        output_power[layout_index] = pulse_count * 5
        output_heat[layout_index] = generated_heat
        output_cooling[layout_index] = cooling_capacity

    _CUDA_SCORE_KERNEL = kernel
    return kernel


def _fixed_point_kernel():
    """Build the exact four-cycle fixed-point certificate kernel lazily."""
    global _CUDA_FIXED_POINT_KERNEL
    if _CUDA_FIXED_POINT_KERNEL is not None:
        return _CUDA_FIXED_POINT_KERNEL

    from numba import cuda

    @cuda.jit
    def kernel(
        layouts,
        columns,
        slot_count,
        kinds,
        max_heat,
        rod_counts,
        internal_pulses,
        power_participant,
        accepts_heat,
        self_vent,
        hull_draw,
        side_vent,
        hull_bonus,
        supported,
        output_fixed_tick,
        output_power,
        output_peak,
        output_max_hull,
    ):
        layout_index = cuda.grid(1)
        if layout_index >= layouts.shape[0]:
            return

        slots = slot_count
        for position in range(slots):
            if supported[layouts[layout_index, position]] == 0:
                return

        hull = 0
        peak = 0
        power = 0
        for tick in range(1, 5):
            previous_hull = hull
            max_hull_value = 10_000
            changed = False

            # Heat pass: exact row-major ordering used by ReactorSimulator.
            for position in range(slots):
                code = layouts[layout_index, position]
                kind = kinds[code]
                row = position // columns
                column = position - row * columns

                if kind == 1:  # fuel
                    rods = rod_counts[code]
                    for _rod in range(rods):
                        pulses = internal_pulses[code]
                        if column > 0:
                            pulses += power_participant[layouts[layout_index, position - 1]]
                        if column + 1 < columns:
                            pulses += power_participant[layouts[layout_index, position + 1]]
                        if row > 0:
                            pulses += power_participant[layouts[layout_index, position - columns]]
                        if position + columns < slots:
                            pulses += power_participant[layouts[layout_index, position + columns]]
                        remaining = 2 * pulses * (pulses + 1)

                        acceptor_count = 0
                        if column > 0 and accepts_heat[layouts[layout_index, position - 1]]:
                            acceptor_count += 1
                        if column + 1 < columns and accepts_heat[layouts[layout_index, position + 1]]:
                            acceptor_count += 1
                        if row > 0 and accepts_heat[layouts[layout_index, position - columns]]:
                            acceptor_count += 1
                        if position + columns < slots and accepts_heat[layouts[layout_index, position + columns]]:
                            acceptor_count += 1

                        visited = 0
                        for direction in range(4):
                            target = -1
                            if direction == 0 and column > 0:
                                target = position - 1
                            elif direction == 1 and column + 1 < columns:
                                target = position + 1
                            elif direction == 2 and row > 0:
                                target = position - columns
                            elif direction == 3 and position + columns < slots:
                                target = position + columns
                            if target < 0 or accepts_heat[layouts[layout_index, target]] == 0:
                                continue
                            left = acceptor_count - visited
                            amount = remaining // left
                            remaining -= amount
                            visited += 1
                            target_code = layouts[layout_index, target]
                            target_heat = layouts[layout_index, slots + target] + amount
                            if target_heat > max_heat[target_code]:
                                # The CPU path must handle component removal.
                                return
                            if target_heat != layouts[layout_index, slots + target]:
                                changed = True
                            layouts[layout_index, slots + target] = target_heat
                        if remaining > 0:
                            hull += remaining
                            changed = True

                elif kind == 2:  # vent
                    spread = side_vent[code]
                    if spread > 0:
                        for direction in range(4):
                            target = -1
                            if direction == 0 and column > 0:
                                target = position - 1
                            elif direction == 1 and column + 1 < columns:
                                target = position + 1
                            elif direction == 2 and row > 0:
                                target = position - columns
                            elif direction == 3 and position + columns < slots:
                                target = position + columns
                            if target < 0 or accepts_heat[layouts[layout_index, target]] == 0:
                                continue
                            before = layouts[layout_index, slots + target]
                            after = before - spread
                            if after < 0:
                                after = 0
                            if after != before:
                                changed = True
                            layouts[layout_index, slots + target] = after
                    else:
                        before = layouts[layout_index, slots + position]
                        draw = hull_draw[code]
                        if draw > hull:
                            draw = hull
                        target_heat = before + draw
                        if target_heat > max_heat[code]:
                            return
                        hull -= draw
                        after = target_heat - self_vent[code]
                        if after < 0:
                            after = 0
                        if after != before or hull != previous_hull:
                            changed = True
                        layouts[layout_index, slots + position] = after
                elif kind == 6:  # plating
                    max_hull_value += hull_bonus[code]

            # Energy pass has no thermal side effects with automatic refuelling.
            current_power = 0
            for position in range(slots):
                code = layouts[layout_index, position]
                rods = rod_counts[code]
                if rods == 0:
                    continue
                row = position // columns
                column = position - row * columns
                neighbors = 0
                if column > 0:
                    neighbors += power_participant[layouts[layout_index, position - 1]]
                if column + 1 < columns:
                    neighbors += power_participant[layouts[layout_index, position + 1]]
                if row > 0:
                    neighbors += power_participant[layouts[layout_index, position - columns]]
                if position + columns < slots:
                    neighbors += power_participant[layouts[layout_index, position + columns]]
                current_power += rods * (internal_pulses[code] + neighbors) * 5

            if hull > peak:
                peak = hull
            if hull >= (max_hull_value * 85) // 100:
                return

            # ``changed`` is conservative; compare every heat cell to the
            # snapshot stored in the second auxiliary plane for an exact test.
            same = hull == previous_hull
            for position in range(slots):
                before = layouts[layout_index, 2 * slots + position]
                after = layouts[layout_index, slots + position]
                if after != before:
                    same = False
                layouts[layout_index, 2 * slots + position] = after
            if tick == 1:
                # The initial auxiliary plane is zero-initialized.
                same = hull == 0
                for position in range(slots):
                    if layouts[layout_index, slots + position] != 0:
                        same = False
                    layouts[layout_index, 2 * slots + position] = layouts[
                        layout_index, slots + position
                    ]
            power = current_power
            if same:
                output_fixed_tick[layout_index] = tick
                output_power[layout_index] = power
                output_peak[layout_index] = peak
                output_max_hull[layout_index] = max_hull_value
                return

    _CUDA_FIXED_POINT_KERNEL = kernel
    return kernel


class CudaBatchScorer:
    """Reusable RTX-friendly batch scorer for heuristic candidate screening.

    The kernel computes exact static EU/t and generated heat. ``cooling_proxy``
    is deliberately only a ranking hint; every retained layout is still passed
    through the production CPU simulator before it can enter a leaderboard.
    """

    THREADS_PER_BLOCK = 256

    def __init__(self) -> None:
        info = cuda_device_info()
        if not info.available:
            raise RuntimeError(info.reason or "CUDA 不可用")

        from numba import cuda

        self.info = info
        self._cuda = cuda
        self._codes = {item: index for index, item in enumerate(COMPONENT_IDS)}
        specs = [COMPONENTS[item] for item in COMPONENT_IDS]
        self._rod_counts = cuda.to_device(np.asarray([spec.rod_count for spec in specs], dtype=np.int32))
        self._internal_pulses = cuda.to_device(
            np.asarray([spec.internal_pulses for spec in specs], dtype=np.int32)
        )
        self._power_participant = cuda.to_device(
            np.asarray([spec.kind in {"fuel", "reflector"} for spec in specs], dtype=np.int32)
        )
        self._self_vent = cuda.to_device(np.asarray([spec.self_vent for spec in specs], dtype=np.int32))
        self._side_vent = cuda.to_device(np.asarray([spec.side_vent for spec in specs], dtype=np.int32))
        self._coolable = cuda.to_device(np.asarray([spec.is_coolable for spec in specs], dtype=np.int32))

    def score(self, layouts: Sequence[tuple[str, ...]], columns: int) -> CudaBatchScores:
        count = len(layouts)
        if count == 0:
            empty = np.empty(0, dtype=np.int32)
            return CudaBatchScores(empty, empty.copy(), empty.copy())
        slots = columns * 6
        if any(len(layout) != slots for layout in layouts):
            raise ValueError("布局尺寸与反应堆列数不匹配")

        encoded = np.fromiter(
            (self._codes[item] for layout in layouts for item in layout),
            dtype=np.int16,
            count=count * slots,
        ).reshape(count, slots)
        device_layouts = self._cuda.to_device(encoded)
        output_power = self._cuda.device_array(count, dtype=np.int32)
        output_heat = self._cuda.device_array(count, dtype=np.int32)
        output_cooling = self._cuda.device_array(count, dtype=np.int32)
        blocks = (count + self.THREADS_PER_BLOCK - 1) // self.THREADS_PER_BLOCK
        # Small legal search spaces cannot fill this GPU. The dispatch is still
        # correct, and surfacing the generic occupancy warning on every batch
        # would make the server log misleadingly noisy.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _score_kernel()[blocks, self.THREADS_PER_BLOCK](
                device_layouts,
                columns,
                self._rod_counts,
                self._internal_pulses,
                self._power_participant,
                self._self_vent,
                self._side_vent,
                self._coolable,
                output_power,
                output_heat,
                output_cooling,
            )
        return CudaBatchScores(
            output_power.copy_to_host(),
            output_heat.copy_to_host(),
            output_cooling.copy_to_host(),
        )


class CudaFixedPointEvaluator:
    """Batch exact fixed-point prover used by exhaustive search.

    It covers the branch-free IC2 subset made of fuel, vents, plating and an
    infinite reflector. Layouts outside that subset, layouts that break a
    component, and layouts that do not reach a period-one thermal state within
    four cycles are returned as ``None`` for mandatory CPU evaluation.
    """

    THREADS_PER_BLOCK = 256

    def __init__(self) -> None:
        info = cuda_device_info()
        if not info.available:
            raise RuntimeError(info.reason or "CUDA 不可用")

        from numba import cuda

        self.info = info
        self._cuda = cuda
        self._codes = {item: index for index, item in enumerate(COMPONENT_IDS)}
        specs = [COMPONENTS[item] for item in COMPONENT_IDS]
        kind_codes = {
            "empty": 0,
            "fuel": 1,
            "vent": 2,
            "exchanger": 3,
            "coolant": 4,
            "condensator": 5,
            "plating": 6,
            "reflector": 7,
        }
        supported = [
            spec.kind in {"empty", "fuel", "vent", "plating"}
            or (spec.kind == "reflector" and spec.max_damage == 0)
            for spec in specs
        ]
        self._kinds = cuda.to_device(np.asarray([kind_codes[spec.kind] for spec in specs], dtype=np.int32))
        self._max_heat = cuda.to_device(np.asarray([spec.max_heat for spec in specs], dtype=np.int32))
        self._rod_counts = cuda.to_device(np.asarray([spec.rod_count for spec in specs], dtype=np.int32))
        self._internal_pulses = cuda.to_device(
            np.asarray([spec.internal_pulses for spec in specs], dtype=np.int32)
        )
        self._power_participant = cuda.to_device(
            np.asarray([spec.kind in {"fuel", "reflector"} for spec in specs], dtype=np.int32)
        )
        self._accepts_heat = cuda.to_device(
            np.asarray([spec.accepts_heat for spec in specs], dtype=np.int32)
        )
        self._self_vent = cuda.to_device(np.asarray([spec.self_vent for spec in specs], dtype=np.int32))
        self._hull_draw = cuda.to_device(np.asarray([spec.hull_draw for spec in specs], dtype=np.int32))
        self._side_vent = cuda.to_device(np.asarray([spec.side_vent for spec in specs], dtype=np.int32))
        self._hull_bonus = cuda.to_device(
            np.asarray([spec.hull_capacity_bonus for spec in specs], dtype=np.int32)
        )
        self._supported = cuda.to_device(np.asarray(supported, dtype=np.int32))

    def certify(
        self,
        layouts: Sequence[tuple[str, ...]],
        columns: int,
        max_reactor_ticks: int,
    ) -> list[CudaFixedPointCertificate | None]:
        count = len(layouts)
        if count == 0:
            return []
        slots = columns * 6
        if any(len(layout) != slots for layout in layouts):
            raise ValueError("布局尺寸与反应堆列数不匹配")
        if max_reactor_ticks < 20_000:
            return [None] * count

        # Plane 0 stores immutable component codes, plane 1 current heat, and
        # plane 2 the previous-cycle heat snapshot.
        working = np.zeros((count, slots * 3), dtype=np.int32)
        working[:, :slots] = np.fromiter(
            (self._codes[item] for layout in layouts for item in layout),
            dtype=np.int32,
            count=count * slots,
        ).reshape(count, slots)
        device_working = self._cuda.to_device(working)
        fixed_tick = self._cuda.to_device(np.zeros(count, dtype=np.int32))
        power = self._cuda.device_array(count, dtype=np.int32)
        peak = self._cuda.device_array(count, dtype=np.int32)
        max_hull = self._cuda.device_array(count, dtype=np.int32)
        blocks = (count + self.THREADS_PER_BLOCK - 1) // self.THREADS_PER_BLOCK
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _fixed_point_kernel()[blocks, self.THREADS_PER_BLOCK](
                device_working,
                columns,
                slots,
                self._kinds,
                self._max_heat,
                self._rod_counts,
                self._internal_pulses,
                self._power_participant,
                self._accepts_heat,
                self._self_vent,
                self._hull_draw,
                self._side_vent,
                self._hull_bonus,
                self._supported,
                fixed_tick,
                power,
                peak,
                max_hull,
            )
        host_fixed_tick = fixed_tick.copy_to_host()
        host_power = power.copy_to_host()
        host_peak = peak.copy_to_host()
        host_max_hull = max_hull.copy_to_host()
        safe_game_ticks = (40_000 if max_reactor_ticks >= 40_000 else 20_000) * 20
        return [
            CudaFixedPointCertificate(
                average_eu_per_tick=float(host_power[index]),
                safe_game_ticks=safe_game_ticks,
                peak_hull_heat=int(host_peak[index]),
                max_hull_heat=int(host_max_hull[index]),
            )
            if host_fixed_tick[index] > 0
            else None
            for index in range(count)
        ]


def select_screened_layouts(
    layouts: Sequence[tuple[str, ...]],
    scores: CudaBatchScores,
    keep: int,
    *,
    mark_i_only: bool,
) -> list[tuple[str, ...]]:
    """Keep a power-diverse GPU-ranked subset without claiming feasibility."""
    if keep <= 0 or not layouts:
        return []
    if len(layouts) <= keep:
        return list(layouts)

    indices = np.arange(len(layouts), dtype=np.int64)
    margin = scores.cooling_proxy.astype(np.int64) - scores.generated_heat.astype(np.int64)
    by_power = np.lexsort((-indices, margin, scores.power))[::-1]
    if not mark_i_only:
        return [layouts[int(index)] for index in by_power[:keep]]

    feasible = margin >= 0
    by_thermal = np.lexsort((-indices, margin, scores.power, feasible))[::-1]
    selected: list[int] = []
    seen: set[int] = set()
    # Half preserves the theoretical-power frontier; half favors plausible
    # Mark-I thermal balance. This proxy is never used as a proof or result.
    for order, target in ((by_power, (keep + 1) // 2), (by_thermal, keep)):
        added = 0
        for raw_index in order:
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            selected.append(index)
            added += 1
            if len(selected) >= keep or (order is by_power and added >= target):
                break
        if len(selected) >= keep:
            break
    return [layouts[index] for index in selected]
