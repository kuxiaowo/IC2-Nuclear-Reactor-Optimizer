from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from .components import COMPONENTS, COMPONENT_IDS
from .gpu_acceleration import CudaDeviceInfo, cuda_device_info
from .mark import classify_mark


@dataclass(frozen=True, slots=True)
class CudaSimulationResult:
    mark: str | None
    average_eu_per_tick: float
    safe_game_ticks: int
    peak_hull_heat: int
    max_hull_heat: int
    reactor_ticks: int
    first_critical_tick: int | None
    first_component_break_tick: int | None
    meltdown_tick: int | None
    stable: bool


_FULL_SIMULATION_KERNEL = None


def _build_full_simulation_kernel():
    global _FULL_SIMULATION_KERNEL
    if _FULL_SIMULATION_KERNEL is not None:
        return _FULL_SIMULATION_KERNEL

    from numba import cuda, int32, uint64

    @cuda.jit(device=True, inline=True)
    def neighbor_at(position, direction, columns, slots):
        row = position // columns
        column = position - row * columns
        if direction == 0:
            return position - 1 if column > 0 else -1
        if direction == 1:
            return position + 1 if column + 1 < columns else -1
        if direction == 2:
            return position - columns if row > 0 else -1
        return position + columns if position + columns < slots else -1

    @cuda.jit(device=True, inline=True)
    def can_store(codes, heat, layout_index, position, kinds, max_heat, accepts_heat):
        code = codes[layout_index, position]
        if code == 0 or accepts_heat[code] == 0:
            return False
        if kinds[code] == 5:  # condensator
            return heat[layout_index, position] < max_heat[code]
        return True

    @cuda.jit(device=True)
    def alter_heat(
        codes,
        heat,
        damage,
        layout_index,
        position,
        delta,
        kinds,
        max_heat,
        accepts_heat,
    ):
        code = codes[layout_index, position]
        if code == 0 or accepts_heat[code] == 0:
            return delta, 0
        if kinds[code] == 5:  # condensator
            if delta < 0:
                return delta, 0
            capacity = max_heat[code] - heat[layout_index, position]
            accepted = delta if delta < capacity else capacity
            heat[layout_index, position] += accepted
            return delta - accepted, 0

        target = heat[layout_index, position] + delta
        if target > max_heat[code]:
            remainder = max_heat[code] - target + 1
            codes[layout_index, position] = 0
            heat[layout_index, position] = 0
            damage[layout_index, position] = 0
            return remainder, 1
        if target < 0:
            heat[layout_index, position] = 0
            return target, 0
        heat[layout_index, position] = target
        return 0, 0

    @cuda.jit(device=True, inline=True)
    def exchange_amount(source_heat, source_capacity, target_heat, target_capacity, limit, rounded, low_range):
        source_ratio = source_heat * 100.0 / source_capacity
        target_ratio = target_heat * 100.0 / target_capacity
        combined = target_ratio + source_ratio / 2.0
        raw = target_capacity / 100.0 * combined
        amount = int(raw + 0.5) if rounded else int(raw)
        if amount > limit:
            amount = limit
        threshold_range = low_range if low_range >= 0 else limit
        if combined < 1.0:
            amount = threshold_range // 2
        if combined < 0.75:
            amount = threshold_range // 4
        if combined < 0.5:
            amount = threshold_range // 8
        if combined < 0.25:
            amount = 1
        source_tenth = int(source_ratio * 10.0 + 0.5)
        target_tenth = int(target_ratio * 10.0 + 0.5)
        if target_tenth > source_tenth:
            amount = -amount
        elif target_tenth == source_tenth:
            amount = 0
        return amount

    @cuda.jit(device=True)
    def state_hash(codes, heat, damage, hull, layout_index, slots, kinds):
        value = uint64(1469598103934665603)
        value = (value ^ uint64(hull)) * uint64(1099511628211)
        for position in range(slots):
            code = codes[layout_index, position]
            value = (value ^ uint64(code)) * uint64(1099511628211)
            value = (value ^ uint64(heat[layout_index, position])) * uint64(1099511628211)
            if kinds[code] != 1:  # fuel damage is excluded by ReactorSimulator
                value = (value ^ uint64(damage[layout_index, position])) * uint64(1099511628211)
        return value

    @cuda.jit(device=True)
    def state_equals_history(
        codes,
        heat,
        damage,
        hull,
        layout_index,
        slots,
        kinds,
        history_codes,
        history_heat,
        history_damage,
        history_hull,
        checkpoint,
    ):
        if hull != history_hull[checkpoint, layout_index]:
            return False
        for position in range(slots):
            code = codes[layout_index, position]
            if code != history_codes[checkpoint, layout_index, position]:
                return False
            if heat[layout_index, position] != history_heat[checkpoint, layout_index, position]:
                return False
            if kinds[code] != 1 and damage[layout_index, position] != history_damage[
                checkpoint, layout_index, position
            ]:
                return False
        return True

    @cuda.jit
    def simulate_chunk(
        codes,
        heat,
        damage,
        previous_codes,
        previous_heat,
        previous_damage,
        columns,
        slots,
        max_reactor_ticks,
        chunk_ticks,
        kinds,
        max_heat,
        max_damage,
        rod_counts,
        internal_pulses,
        accepts_heat,
        self_vent,
        hull_draw,
        side_vent,
        exchange_side,
        exchange_hull,
        hull_bonus,
        hull_values,
        max_hull_values,
        peak_values,
        reactor_ticks,
        first_critical,
        first_break,
        meltdown_tick,
        safe_eu,
        safe_cycles,
        status,
        history_codes,
        history_heat,
        history_damage,
        history_hull,
        history_hash,
        history_count,
    ):
        layout_index = cuda.grid(1)
        if layout_index >= codes.shape[0] or status[layout_index] != 0:
            return

        for _chunk_step in range(chunk_ticks):
            tick = reactor_ticks[layout_index]
            if tick >= max_reactor_ticks:
                status[layout_index] = 3  # tick limit
                return

            hull = hull_values[layout_index]
            for position in range(slots):
                previous_codes[layout_index, position] = codes[layout_index, position]
                previous_heat[layout_index, position] = heat[layout_index, position]
                previous_damage[layout_index, position] = damage[layout_index, position]
            previous_hull = hull
            tick += 1
            current_max_hull = 10_000

            # Official row-major heat pass.
            for position in range(slots):
                code = codes[layout_index, position]
                if code == 0:
                    continue
                kind = kinds[code]
                if kind == 1:  # fuel
                    for _rod in range(rod_counts[code]):
                        pulses = internal_pulses[code]
                        for direction in range(4):
                            neighbor = neighbor_at(position, direction, columns, slots)
                            if neighbor < 0:
                                continue
                            neighbor_code = codes[layout_index, neighbor]
                            neighbor_kind = kinds[neighbor_code]
                            if neighbor_kind == 1:
                                pulses += 1
                            elif neighbor_kind == 7:
                                pulses += 1
                                durability = max_damage[neighbor_code]
                                if durability > 0:
                                    if damage[layout_index, neighbor] + 1 >= durability:
                                        codes[layout_index, neighbor] = 0
                                        heat[layout_index, neighbor] = 0
                                        damage[layout_index, neighbor] = 0
                                        if first_break[layout_index] == 0:
                                            first_break[layout_index] = tick
                                    else:
                                        damage[layout_index, neighbor] += 1

                        remaining = 2 * pulses * (pulses + 1)
                        acceptors = cuda.local.array(4, dtype=int32)
                        acceptor_count = 0
                        for direction in range(4):
                            neighbor = neighbor_at(position, direction, columns, slots)
                            if neighbor >= 0 and can_store(
                                codes, heat, layout_index, neighbor, kinds, max_heat, accepts_heat
                            ):
                                acceptors[acceptor_count] = neighbor
                                acceptor_count += 1
                        for acceptor_index in range(acceptor_count):
                            left = acceptor_count - acceptor_index
                            amount = remaining // left
                            remaining -= amount
                            neighbor = acceptors[acceptor_index]
                            remainder, broke = alter_heat(
                                codes,
                                heat,
                                damage,
                                layout_index,
                                neighbor,
                                amount,
                                kinds,
                                max_heat,
                                accepts_heat,
                            )
                            remaining += remainder
                            if broke and first_break[layout_index] == 0:
                                first_break[layout_index] = tick
                        if remaining > 0:
                            hull += remaining

                elif kind == 2:  # vent
                    if side_vent[code] > 0:
                        for direction in range(4):
                            neighbor = neighbor_at(position, direction, columns, slots)
                            if neighbor < 0 or not can_store(
                                codes, heat, layout_index, neighbor, kinds, max_heat, accepts_heat
                            ):
                                continue
                            _remainder, broke = alter_heat(
                                codes,
                                heat,
                                damage,
                                layout_index,
                                neighbor,
                                -side_vent[code],
                                kinds,
                                max_heat,
                                accepts_heat,
                            )
                            if broke and first_break[layout_index] == 0:
                                first_break[layout_index] = tick
                    else:
                        drawn = hull_draw[code]
                        if drawn > hull:
                            drawn = hull
                        remainder, broke = alter_heat(
                            codes,
                            heat,
                            damage,
                            layout_index,
                            position,
                            drawn,
                            kinds,
                            max_heat,
                            accepts_heat,
                        )
                        if broke and first_break[layout_index] == 0:
                            first_break[layout_index] = tick
                        if remainder <= 0:
                            hull -= drawn
                            # If the vent broke, Python mutates the detached old
                            # stack here; the new empty slot stays unchanged.
                            if not broke and codes[layout_index, position] != 0:
                                _unused, broke_self = alter_heat(
                                    codes,
                                    heat,
                                    damage,
                                    layout_index,
                                    position,
                                    -self_vent[code],
                                    kinds,
                                    max_heat,
                                    accepts_heat,
                                )
                                if broke_self and first_break[layout_index] == 0:
                                    first_break[layout_index] = tick

                elif kind == 3:  # exchanger
                    my_delta = 0
                    if exchange_side[code] > 0:
                        for direction in range(4):
                            neighbor = neighbor_at(position, direction, columns, slots)
                            if neighbor < 0 or not can_store(
                                codes, heat, layout_index, neighbor, kinds, max_heat, accepts_heat
                            ):
                                continue
                            neighbor_code = codes[layout_index, neighbor]
                            amount = exchange_amount(
                                heat[layout_index, position],
                                max_heat[code],
                                heat[layout_index, neighbor],
                                max_heat[neighbor_code],
                                exchange_side[code],
                                False,
                                -1,
                            )
                            my_delta -= amount
                            remainder, broke = alter_heat(
                                codes,
                                heat,
                                damage,
                                layout_index,
                                neighbor,
                                amount,
                                kinds,
                                max_heat,
                                accepts_heat,
                            )
                            my_delta += remainder
                            if broke and first_break[layout_index] == 0:
                                first_break[layout_index] = tick
                    if exchange_hull[code] > 0:
                        amount = exchange_amount(
                            heat[layout_index, position],
                            max_heat[code],
                            hull,
                            current_max_hull,
                            exchange_hull[code],
                            True,
                            exchange_side[code],
                        )
                        my_delta -= amount
                        hull += amount
                    _remainder, broke = alter_heat(
                        codes,
                        heat,
                        damage,
                        layout_index,
                        position,
                        my_delta,
                        kinds,
                        max_heat,
                        accepts_heat,
                    )
                    if broke and first_break[layout_index] == 0:
                        first_break[layout_index] = tick
                elif kind == 6:  # plating
                    current_max_hull += hull_bonus[code]

            # Energy pass. Fuel is automatically replaced in place on depletion.
            power = 0
            for position in range(slots):
                code = codes[layout_index, position]
                if code == 0 or kinds[code] != 1:
                    continue
                for _rod in range(rod_counts[code]):
                    pulses = internal_pulses[code]
                    for direction in range(4):
                        neighbor = neighbor_at(position, direction, columns, slots)
                        if neighbor < 0:
                            continue
                        neighbor_kind = kinds[codes[layout_index, neighbor]]
                        if neighbor_kind == 1 or neighbor_kind == 7:
                            pulses += 1
                    power += pulses * 5

            reactor_ticks[layout_index] = tick
            hull_values[layout_index] = hull
            max_hull_values[layout_index] = current_max_hull
            if hull > peak_values[layout_index]:
                peak_values[layout_index] = hull
            if first_critical[layout_index] == 0 and hull >= (current_max_hull * 85) // 100:
                first_critical[layout_index] = tick
            first_event = first_critical[layout_index]
            broken = first_break[layout_index]
            if first_event == 0 or (broken != 0 and broken < first_event):
                first_event = broken
            if first_event == 0 or tick <= first_event:
                safe_eu[layout_index] += power
                safe_cycles[layout_index] += 1

            if hull >= current_max_hull:
                meltdown_tick[layout_index] = tick
                status[layout_index] = 2
                return

            # Exact period-one test used by the CPU fast-forward path.
            if first_critical[layout_index] == 0 and first_break[layout_index] == 0:
                fixed = hull == previous_hull
                for position in range(slots):
                    code = codes[layout_index, position]
                    if code != previous_codes[layout_index, position]:
                        fixed = False
                    if heat[layout_index, position] != previous_heat[layout_index, position]:
                        fixed = False
                    if kinds[code] != 1 and damage[layout_index, position] != previous_damage[
                        layout_index, position
                    ]:
                        fixed = False
                if fixed:
                    # Match ReactorSimulator exactly: fixed states advance to
                    # the checkpoint after the next aligned boundary.
                    stable_tick = ((tick + 19_999) // 20_000 + 1) * 20_000
                    target_tick = stable_tick
                    if target_tick > max_reactor_ticks:
                        target_tick = max_reactor_ticks
                    remaining = target_tick - tick
                    safe_eu[layout_index] += power * remaining
                    safe_cycles[layout_index] += remaining
                    reactor_ticks[layout_index] = target_tick
                    status[layout_index] = 1 if target_tick == stable_tick else 3
                    return

            # Full-state periodic check at official 20,000-cycle boundaries.
            if tick % 20_000 == 0:
                digest = state_hash(codes, heat, damage, hull, layout_index, slots, kinds)
                count = history_count[layout_index]
                for checkpoint in range(count):
                    if history_hash[checkpoint, layout_index] != digest:
                        continue
                    if state_equals_history(
                        codes,
                        heat,
                        damage,
                        hull,
                        layout_index,
                        slots,
                        kinds,
                        history_codes,
                        history_heat,
                        history_damage,
                        history_hull,
                        checkpoint,
                    ):
                        status[layout_index] = 1
                        return
                if count < history_hash.shape[0]:
                    history_hash[count, layout_index] = digest
                    history_hull[count, layout_index] = hull
                    for position in range(slots):
                        history_codes[count, layout_index, position] = codes[layout_index, position]
                        history_heat[count, layout_index, position] = heat[layout_index, position]
                        history_damage[count, layout_index, position] = damage[layout_index, position]
                    history_count[layout_index] = count + 1

            if tick >= max_reactor_ticks:
                status[layout_index] = 3
                return

    _FULL_SIMULATION_KERNEL = simulate_chunk
    return simulate_chunk


class CudaFullSimulator:
    """Run one complete IC2 simulation sequence per CUDA thread."""

    THREADS_PER_BLOCK = 128

    def __init__(self, *, ticks_per_launch: int = 256) -> None:
        info = cuda_device_info()
        if not info.available:
            raise RuntimeError(info.reason or "CUDA 不可用")
        if ticks_per_launch < 1:
            raise ValueError("ticks_per_launch 必须大于 0")

        from numba import cuda

        self.info: CudaDeviceInfo = info
        self.ticks_per_launch = ticks_per_launch
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

        def device(values, dtype=np.int32):
            return cuda.to_device(np.asarray(values, dtype=dtype))

        self._kinds = device([kind_codes[spec.kind] for spec in specs])
        self._max_heat = device([spec.max_heat for spec in specs])
        self._max_damage = device([spec.max_damage for spec in specs])
        self._rod_counts = device([spec.rod_count for spec in specs])
        self._internal_pulses = device([spec.internal_pulses for spec in specs])
        self._accepts_heat = device([spec.accepts_heat for spec in specs])
        self._self_vent = device([spec.self_vent for spec in specs])
        self._hull_draw = device([spec.hull_draw for spec in specs])
        self._side_vent = device([spec.side_vent for spec in specs])
        self._exchange_side = device([spec.exchange_side for spec in specs])
        self._exchange_hull = device([spec.exchange_hull for spec in specs])
        self._hull_bonus = device([spec.hull_capacity_bonus for spec in specs])

    def simulate(
        self,
        layouts: Sequence[tuple[str, ...]],
        columns: int,
        max_reactor_ticks: int,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[CudaSimulationResult] | None:
        count = len(layouts)
        if count == 0:
            return []
        slots = columns * 6
        if any(len(layout) != slots for layout in layouts):
            raise ValueError("布局尺寸与反应堆列数不匹配")

        encoded = np.fromiter(
            (self._codes[item] for layout in layouts for item in layout),
            dtype=np.int16,
            count=count * slots,
        ).reshape(count, slots)
        uses_suc = [
            any(
                COMPONENTS[item].kind == "condensator"
                or (COMPONENTS[item].kind == "reflector" and COMPONENTS[item].max_damage > 0)
                for item in layout
            )
            for layout in layouts
        ]

        cuda = self._cuda
        codes = cuda.to_device(encoded)
        heat = cuda.to_device(np.zeros((count, slots), dtype=np.int32))
        damage = cuda.to_device(np.zeros((count, slots), dtype=np.int32))
        previous_codes = cuda.device_array((count, slots), dtype=np.int16)
        previous_heat = cuda.device_array((count, slots), dtype=np.int32)
        previous_damage = cuda.device_array((count, slots), dtype=np.int32)

        def zeros(dtype=np.int32):
            return cuda.to_device(np.zeros(count, dtype=dtype))

        hull_values = zeros()
        max_hull_values = cuda.to_device(np.full(count, 10_000, dtype=np.int32))
        peak_values = zeros()
        reactor_ticks = zeros()
        first_critical = zeros()
        first_break = zeros()
        meltdown_tick = zeros()
        safe_eu = zeros(np.int64)
        safe_cycles = zeros()
        status = zeros(np.int8)

        checkpoint_count = max(1, max_reactor_ticks // 20_000)
        history_codes = cuda.device_array((checkpoint_count, count, slots), dtype=np.int16)
        history_heat = cuda.device_array((checkpoint_count, count, slots), dtype=np.int32)
        history_damage = cuda.device_array((checkpoint_count, count, slots), dtype=np.int32)
        history_hull = cuda.device_array((checkpoint_count, count), dtype=np.int32)
        history_hash = cuda.device_array((checkpoint_count, count), dtype=np.uint64)
        history_count = zeros()

        blocks = (count + self.THREADS_PER_BLOCK - 1) // self.THREADS_PER_BLOCK
        kernel = _build_full_simulation_kernel()
        launches = (max_reactor_ticks + self.ticks_per_launch - 1) // self.ticks_per_launch
        host_status = np.zeros(count, dtype=np.int8)
        for _launch in range(launches):
            if cancel_check is not None and cancel_check():
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kernel[blocks, self.THREADS_PER_BLOCK](
                    codes,
                    heat,
                    damage,
                    previous_codes,
                    previous_heat,
                    previous_damage,
                    columns,
                    slots,
                    max_reactor_ticks,
                    self.ticks_per_launch,
                    self._kinds,
                    self._max_heat,
                    self._max_damage,
                    self._rod_counts,
                    self._internal_pulses,
                    self._accepts_heat,
                    self._self_vent,
                    self._hull_draw,
                    self._side_vent,
                    self._exchange_side,
                    self._exchange_hull,
                    self._hull_bonus,
                    hull_values,
                    max_hull_values,
                    peak_values,
                    reactor_ticks,
                    first_critical,
                    first_break,
                    meltdown_tick,
                    safe_eu,
                    safe_cycles,
                    status,
                    history_codes,
                    history_heat,
                    history_damage,
                    history_hull,
                    history_hash,
                    history_count,
                )
            status.copy_to_host(host_status)
            if np.all(host_status != 0):
                break

        host_ticks = reactor_ticks.copy_to_host()
        host_first_critical = first_critical.copy_to_host()
        host_first_break = first_break.copy_to_host()
        host_meltdown = meltdown_tick.copy_to_host()
        host_safe_eu = safe_eu.copy_to_host()
        host_safe_cycles = safe_cycles.copy_to_host()
        host_peak = peak_values.copy_to_host()
        host_max_hull = max_hull_values.copy_to_host()
        results: list[CudaSimulationResult] = []
        for index in range(count):
            critical = int(host_first_critical[index]) or None
            broken = int(host_first_break[index]) or None
            stable = int(host_status[index]) == 1
            mark = classify_mark(critical, broken, stable, uses_suc[index])
            intervention_values = [value for value in (critical, broken) if value is not None]
            safe_ticks = (
                min(intervention_values) if intervention_values else int(host_ticks[index])
            ) * 20
            cycles = int(host_safe_cycles[index])
            average = float(host_safe_eu[index]) / cycles if cycles else 0.0
            results.append(CudaSimulationResult(
                mark=mark,
                average_eu_per_tick=average,
                safe_game_ticks=safe_ticks,
                peak_hull_heat=int(host_peak[index]),
                max_hull_heat=int(host_max_hull[index]),
                reactor_ticks=int(host_ticks[index]),
                first_critical_tick=critical,
                first_component_break_tick=broken,
                meltdown_tick=int(host_meltdown[index]) or None,
                stable=stable,
            ))
        return results
