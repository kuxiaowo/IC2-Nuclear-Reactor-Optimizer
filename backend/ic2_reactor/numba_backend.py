from __future__ import annotations

import math

import numpy as np
from numba import config, njit, prange, set_num_threads

from .kernel_abi import (
    COMPONENT_CODE_BY_ID,
    COMPONENT_KERNEL_TABLE,
    KIND_CODE,
    MARK_FLAG_STABLE,
    MARK_FLAG_SUC,
    MARK_I,
    MARK_II,
    MARK_III,
    MARK_IV,
    MARK_UNCLASSIFIED,
    MARK_V,
    PackedEvaluationBatch,
    PackedLayoutBatch,
    packed_neighbor_table,
)


_EMPTY = COMPONENT_CODE_BY_ID["empty"]
_FUEL = KIND_CODE["fuel"]
_VENT = KIND_CODE["vent"]
_EXCHANGER = KIND_CODE["exchanger"]
_CONDENSATOR = KIND_CODE["condensator"]
_PLATING = KIND_CODE["plating"]
_REFLECTOR = KIND_CODE["reflector"]

_STOP_MELTDOWN = 1
_STOP_TICK_LIMIT = 2
_STOP_STABLE = 3

_FUEL_CYCLE = 20_000
_TEN_PERCENT_CYCLE = 2_000


@njit(cache=True, inline="always")
def _can_store_heat(codes, heat, index, kind, max_heat, accepts_heat):
    code = int(codes[index])
    if code == _EMPTY or accepts_heat[code] == 0:
        return False
    if kind[code] == _CONDENSATOR:
        return heat[index] < max_heat[code]
    return True


@njit(cache=True, inline="always")
def _remove_component(codes, heat, damage, index):
    codes[index] = _EMPTY
    heat[index] = 0
    damage[index] = 0


@njit(cache=True, inline="always")
def _alter_heat(codes, heat, damage, index, delta, kind, max_heat, accepts_heat):
    code = int(codes[index])
    if code == _EMPTY or accepts_heat[code] == 0:
        return delta, False

    if kind[code] == _CONDENSATOR:
        if delta < 0:
            return delta, False
        accepted = min(delta, int(max_heat[code]) - int(heat[index]))
        heat[index] += accepted
        return delta - accepted, False

    target = int(heat[index]) + delta
    if target > max_heat[code]:
        remainder = int(max_heat[code]) - target + 1
        _remove_component(codes, heat, damage, index)
        return remainder, True
    if target < 0:
        heat[index] = 0
        return target, False
    heat[index] = target
    return 0, False


@njit(cache=True, inline="always")
def _distribute_fuel_heat(
    codes,
    heat,
    damage,
    source,
    amount,
    hull_heat,
    reactor_tick,
    first_component_break,
    neighbors,
    kind,
    max_heat,
    accepts_heat,
):
    acceptors = np.empty(4, dtype=np.int16)
    count = 0
    for offset in range(4):
        neighbor = int(neighbors[source, offset])
        if neighbor < 0:
            break
        if _can_store_heat(codes, heat, neighbor, kind, max_heat, accepts_heat):
            acceptors[count] = neighbor
            count += 1

    remaining = amount
    cursor = 0
    while cursor < count and remaining > 0:
        active_count = count - cursor
        transfer = remaining // active_count
        remaining -= transfer
        target = int(acceptors[cursor])
        cursor += 1
        remainder, broke = _alter_heat(
            codes, heat, damage, target, transfer, kind, max_heat, accepts_heat
        )
        remaining += remainder
        if broke and first_component_break == 0:
            first_component_break = reactor_tick
    if remaining > 0:
        hull_heat = max(0, hull_heat + remaining)
    return hull_heat, first_component_break


@njit(cache=True, inline="always")
def _process_fuel_heat(
    codes,
    heat,
    damage,
    index,
    hull_heat,
    reactor_tick,
    first_component_break,
    neighbors,
    kind,
    max_heat,
    max_damage,
    rod_count,
    internal_pulses,
    accepts_heat,
):
    source_code = int(codes[index])
    for _ in range(int(rod_count[source_code])):
        pulses = int(internal_pulses[source_code])
        for offset in range(4):
            neighbor = int(neighbors[index, offset])
            if neighbor < 0:
                break
            target_code = int(codes[neighbor])
            if target_code == _EMPTY:
                continue
            target_kind = int(kind[target_code])
            if target_kind == _FUEL:
                pulses += 1
            elif target_kind == _REFLECTOR:
                if max_damage[target_code] > 0:
                    if damage[neighbor] + 1 >= max_damage[target_code]:
                        _remove_component(codes, heat, damage, neighbor)
                        if first_component_break == 0:
                            first_component_break = reactor_tick
                    else:
                        damage[neighbor] += 1
                # The pulse which removes a reflector is still accepted.
                pulses += 1
        generated = 2 * pulses * (pulses + 1)
        hull_heat, first_component_break = _distribute_fuel_heat(
            codes,
            heat,
            damage,
            index,
            generated,
            hull_heat,
            reactor_tick,
            first_component_break,
            neighbors,
            kind,
            max_heat,
            accepts_heat,
        )
    return hull_heat, first_component_break


@njit(cache=True, inline="always")
def _process_fuel_energy(
    codes,
    damage,
    index,
    neighbors,
    kind,
    max_damage,
    rod_count,
    internal_pulses,
):
    source_code = int(codes[index])
    output_pulses = 0
    for _ in range(int(rod_count[source_code])):
        output_pulses += int(internal_pulses[source_code])
        for offset in range(4):
            neighbor = int(neighbors[index, offset])
            if neighbor < 0:
                break
            target_code = int(codes[neighbor])
            if target_code == _EMPTY:
                continue
            target_kind = int(kind[target_code])
            if target_kind == _FUEL or target_kind == _REFLECTOR:
                output_pulses += 1

    if damage[index] >= max_damage[source_code] - 1:
        # Optimizer simulations always auto-refuel in place.
        damage[index] = 0
    else:
        damage[index] += 1
    return output_pulses


@njit(cache=True, inline="always")
def _process_vent(
    codes,
    heat,
    damage,
    index,
    hull_heat,
    reactor_tick,
    first_component_break,
    kind,
    max_heat,
    accepts_heat,
    self_vent,
    hull_draw,
):
    code = int(codes[index])
    if hull_draw[code] > 0:
        drawn = min(int(hull_draw[code]), hull_heat)
        remainder, broke = _alter_heat(
            codes, heat, damage, index, drawn, kind, max_heat, accepts_heat
        )
        if broke and first_component_break == 0:
            first_component_break = reactor_tick
        if remainder > 0:
            return hull_heat, first_component_break
        hull_heat -= drawn
    if codes[index] != _EMPTY and self_vent[code] > 0:
        _, broke = _alter_heat(
            codes,
            heat,
            damage,
            index,
            -int(self_vent[code]),
            kind,
            max_heat,
            accepts_heat,
        )
        if broke and first_component_break == 0:
            first_component_break = reactor_tick
    return hull_heat, first_component_break


@njit(cache=True, inline="always")
def _process_spread_vent(
    codes,
    heat,
    damage,
    index,
    reactor_tick,
    first_component_break,
    neighbors,
    kind,
    max_heat,
    accepts_heat,
    side_vent,
):
    code = int(codes[index])
    for offset in range(4):
        neighbor = int(neighbors[index, offset])
        if neighbor < 0:
            break
        if not _can_store_heat(codes, heat, neighbor, kind, max_heat, accepts_heat):
            continue
        _, broke = _alter_heat(
            codes,
            heat,
            damage,
            neighbor,
            -int(side_vent[code]),
            kind,
            max_heat,
            accepts_heat,
        )
        if broke and first_component_break == 0:
            first_component_break = reactor_tick
    return first_component_break


@njit(cache=True, inline="always")
def _exchange_amount(source_ratio, target_ratio, target_capacity, limit, rounded_base, low_range):
    combined = target_ratio + source_ratio / 2.0
    raw = target_capacity / 100.0 * combined
    if rounded_base:
        amount = math.floor(raw + 0.5)
    else:
        amount = int(raw)
    amount = min(amount, limit)
    if combined < 1.0:
        amount = low_range // 2
    if combined < 0.75:
        amount = low_range // 4
    if combined < 0.5:
        amount = low_range // 8
    if combined < 0.25:
        amount = 1
    source_tenth = math.floor(source_ratio * 10.0 + 0.5) / 10.0
    target_tenth = math.floor(target_ratio * 10.0 + 0.5) / 10.0
    if target_tenth > source_tenth:
        amount = -amount
    elif target_tenth == source_tenth:
        amount = 0
    return amount


@njit(cache=True, inline="always")
def _process_exchanger(
    codes,
    heat,
    damage,
    index,
    hull_heat,
    max_hull_heat,
    reactor_tick,
    first_component_break,
    neighbors,
    kind,
    max_heat,
    accepts_heat,
    exchange_side,
    exchange_hull,
):
    code = int(codes[index])
    source_heat = int(heat[index])
    source_capacity = int(max_heat[code])
    my_heat_delta = 0
    acceptors = np.empty(4, dtype=np.int16)
    count = 0
    for offset in range(4):
        neighbor = int(neighbors[index, offset])
        if neighbor < 0:
            break
        if _can_store_heat(codes, heat, neighbor, kind, max_heat, accepts_heat):
            acceptors[count] = neighbor
            count += 1

    if exchange_side[code] > 0:
        for offset in range(count):
            neighbor = int(acceptors[offset])
            neighbor_code = int(codes[neighbor])
            mine = source_heat * 100.0 / source_capacity
            theirs = heat[neighbor] * 100.0 / max_heat[neighbor_code]
            amount = _exchange_amount(
                mine,
                theirs,
                int(max_heat[neighbor_code]),
                int(exchange_side[code]),
                False,
                int(exchange_side[code]),
            )
            my_heat_delta -= amount
            remainder, broke = _alter_heat(
                codes, heat, damage, neighbor, amount, kind, max_heat, accepts_heat
            )
            my_heat_delta += remainder
            if broke and first_component_break == 0:
                first_component_break = reactor_tick

    if exchange_hull[code] > 0:
        mine = source_heat * 100.0 / source_capacity
        hull = hull_heat * 100.0 / max_hull_heat
        amount = _exchange_amount(
            mine,
            hull,
            max_hull_heat,
            int(exchange_hull[code]),
            True,
            int(exchange_side[code]),
        )
        my_heat_delta -= amount
        hull_heat += amount

    _, broke = _alter_heat(
        codes, heat, damage, index, my_heat_delta, kind, max_heat, accepts_heat
    )
    if broke and first_component_break == 0:
        first_component_break = reactor_tick
    return hull_heat, first_component_break


@njit(cache=True, inline="always")
def _fixed_state_matches(codes, heat, damage, prev_heat, prev_damage, hull_heat, prev_hull, kind):
    if hull_heat != prev_hull:
        return False
    for index in range(codes.shape[0]):
        if heat[index] != prev_heat[index]:
            return False
        code = int(codes[index])
        if code != _EMPTY and kind[code] != _FUEL and damage[index] != prev_damage[index]:
            return False
    return True


@njit(cache=True, inline="always")
def _copy_fixed_state(codes, heat, damage, prev_heat, prev_damage):
    for index in range(codes.shape[0]):
        prev_heat[index] = heat[index]
        prev_damage[index] = damage[index]


@njit(cache=True, inline="always")
def _checkpoint_matches(
    codes,
    heat,
    damage,
    hull_heat,
    checkpoint,
    checkpoint_codes,
    checkpoint_heat,
    checkpoint_damage,
    checkpoint_hull,
    kind,
):
    if hull_heat != checkpoint_hull[checkpoint]:
        return False
    for index in range(codes.shape[0]):
        if codes[index] != checkpoint_codes[checkpoint, index]:
            return False
        if heat[index] != checkpoint_heat[checkpoint, index]:
            return False
        code = int(codes[index])
        if code != _EMPTY and kind[code] != _FUEL:
            if damage[index] != checkpoint_damage[checkpoint, index]:
                return False
    return True


@njit(cache=True, inline="always")
def _store_checkpoint(
    codes,
    heat,
    damage,
    hull_heat,
    checkpoint,
    checkpoint_codes,
    checkpoint_heat,
    checkpoint_damage,
    checkpoint_hull,
):
    checkpoint_hull[checkpoint] = hull_heat
    for index in range(codes.shape[0]):
        checkpoint_codes[checkpoint, index] = codes[index]
        checkpoint_heat[checkpoint, index] = heat[index]
        checkpoint_damage[checkpoint, index] = damage[index]


@njit(cache=True)
def _simulate_one(
    initial_codes,
    initial_hull_heat,
    max_reactor_ticks,
    neighbors,
    kind,
    max_heat,
    max_damage,
    rod_count,
    internal_pulses,
    self_vent,
    hull_draw,
    side_vent,
    exchange_side,
    exchange_hull,
    hull_capacity_bonus,
    accepts_heat,
):
    slots = initial_codes.shape[0]
    codes = initial_codes.copy()
    heat = np.zeros(slots, dtype=np.int32)
    damage = np.zeros(slots, dtype=np.int32)
    hull_heat = int(initial_hull_heat)
    max_hull_heat = 10_000
    uses_suc = False
    for index in range(slots):
        code = int(codes[index])
        max_hull_heat += int(hull_capacity_bonus[code])
        if kind[code] == _CONDENSATOR or (kind[code] == _REFLECTOR and max_damage[code] > 0):
            uses_suc = True

    reactor_tick = 0
    peak_hull_heat = hull_heat
    first_critical = 0
    first_component_break = 0
    meltdown_tick = 0
    safe_pulses = 0
    safe_cycle_count = 0
    stable = False
    stop_reason = _STOP_TICK_LIMIT

    prev_heat = np.zeros(slots, dtype=np.int32)
    prev_damage = np.zeros(slots, dtype=np.int32)
    prev_hull = hull_heat

    max_checkpoints = max_reactor_ticks // _FUEL_CYCLE + 1
    checkpoint_codes = np.empty((max_checkpoints, slots), dtype=np.uint8)
    checkpoint_heat = np.empty((max_checkpoints, slots), dtype=np.int32)
    checkpoint_damage = np.empty((max_checkpoints, slots), dtype=np.int32)
    checkpoint_hull = np.empty(max_checkpoints, dtype=np.int64)
    checkpoint_count = 0

    for _ in range(max_reactor_ticks):
        reactor_tick += 1
        output_pulses = 0
        max_hull_heat = 10_000

        for index in range(slots):
            code = int(codes[index])
            if code == _EMPTY:
                continue
            component_kind = int(kind[code])
            if component_kind == _FUEL:
                hull_heat, first_component_break = _process_fuel_heat(
                    codes,
                    heat,
                    damage,
                    index,
                    hull_heat,
                    reactor_tick,
                    first_component_break,
                    neighbors,
                    kind,
                    max_heat,
                    max_damage,
                    rod_count,
                    internal_pulses,
                    accepts_heat,
                )
            elif component_kind == _VENT:
                if side_vent[code] > 0:
                    first_component_break = _process_spread_vent(
                        codes,
                        heat,
                        damage,
                        index,
                        reactor_tick,
                        first_component_break,
                        neighbors,
                        kind,
                        max_heat,
                        accepts_heat,
                        side_vent,
                    )
                else:
                    hull_heat, first_component_break = _process_vent(
                        codes,
                        heat,
                        damage,
                        index,
                        hull_heat,
                        reactor_tick,
                        first_component_break,
                        kind,
                        max_heat,
                        accepts_heat,
                        self_vent,
                        hull_draw,
                    )
            elif component_kind == _EXCHANGER:
                hull_heat, first_component_break = _process_exchanger(
                    codes,
                    heat,
                    damage,
                    index,
                    hull_heat,
                    max_hull_heat,
                    reactor_tick,
                    first_component_break,
                    neighbors,
                    kind,
                    max_heat,
                    accepts_heat,
                    exchange_side,
                    exchange_hull,
                )
            elif component_kind == _PLATING:
                max_hull_heat += int(hull_capacity_bonus[code])

        for index in range(slots):
            code = int(codes[index])
            if code != _EMPTY and kind[code] == _FUEL:
                output_pulses += _process_fuel_energy(
                    codes,
                    damage,
                    index,
                    neighbors,
                    kind,
                    max_damage,
                    rod_count,
                    internal_pulses,
                )

        peak_hull_heat = max(peak_hull_heat, hull_heat)
        critical_heat = math.floor(max_hull_heat * 0.85)
        if first_critical == 0 and hull_heat >= critical_heat:
            first_critical = reactor_tick
        if hull_heat >= max_hull_heat:
            meltdown_tick = reactor_tick

        if first_critical == 0:
            current_intervention = first_component_break
        elif first_component_break == 0:
            current_intervention = first_critical
        else:
            current_intervention = min(first_critical, first_component_break)
        if current_intervention == 0 or reactor_tick <= current_intervention:
            safe_pulses += output_pulses
            safe_cycle_count += 1

        if meltdown_tick > 0:
            stop_reason = _STOP_MELTDOWN
            break

        can_check_fixed = first_critical == 0 and first_component_break == 0
        if can_check_fixed and _fixed_state_matches(
            codes, heat, damage, prev_heat, prev_damage, hull_heat, prev_hull, kind
        ):
            stable_tick = ((reactor_tick + _FUEL_CYCLE - 1) // _FUEL_CYCLE + 1) * _FUEL_CYCLE
            target_tick = min(stable_tick, max_reactor_ticks)
            remaining = target_tick - reactor_tick
            safe_pulses += output_pulses * remaining
            safe_cycle_count += remaining
            reactor_tick = target_tick
            if target_tick == stable_tick:
                stable = True
                stop_reason = _STOP_STABLE
            else:
                stop_reason = _STOP_TICK_LIMIT
            break
        if can_check_fixed:
            _copy_fixed_state(codes, heat, damage, prev_heat, prev_damage)
            prev_hull = hull_heat

        if reactor_tick % _FUEL_CYCLE == 0:
            matched = False
            for checkpoint in range(checkpoint_count):
                if _checkpoint_matches(
                    codes,
                    heat,
                    damage,
                    hull_heat,
                    checkpoint,
                    checkpoint_codes,
                    checkpoint_heat,
                    checkpoint_damage,
                    checkpoint_hull,
                    kind,
                ):
                    matched = True
                    break
            if matched:
                stable = True
                stop_reason = _STOP_STABLE
                break
            _store_checkpoint(
                codes,
                heat,
                damage,
                hull_heat,
                checkpoint_count,
                checkpoint_codes,
                checkpoint_heat,
                checkpoint_damage,
                checkpoint_hull,
            )
            checkpoint_count += 1

    if safe_cycle_count > 0:
        average_eu = safe_pulses * 5.0 / safe_cycle_count
    else:
        average_eu = 0.0

    if first_critical == 0:
        intervention = first_component_break
    elif first_component_break == 0:
        intervention = first_critical
    else:
        intervention = min(first_critical, first_component_break)
    safe_game_ticks = (intervention if intervention > 0 else reactor_tick) * 20

    mark_family = MARK_UNCLASSIFIED
    mark_level = 0
    if stable and first_critical == 0 and first_component_break == 0:
        mark_family = MARK_I
    elif intervention > 0:
        if intervention >= _FUEL_CYCLE:
            mark_family = MARK_II
            cycles = intervention // _FUEL_CYCLE
            mark_level = 16 if cycles >= 16 else max(1, cycles)
        elif intervention < _TEN_PERCENT_CYCLE:
            mark_family = MARK_V
        else:
            broke_first = first_component_break > 0 and (
                first_critical == 0 or first_component_break <= first_critical
            )
            mark_family = MARK_IV if broke_first else MARK_III

    mark_flags = 0
    if mark_family != MARK_UNCLASSIFIED and uses_suc:
        mark_flags |= MARK_FLAG_SUC
    if stable:
        mark_flags |= MARK_FLAG_STABLE
    safety_margin = 1.0 - peak_hull_heat / max_hull_heat
    total_eu = average_eu * safe_game_ticks
    return (
        mark_family,
        mark_level,
        mark_flags,
        stop_reason,
        reactor_tick,
        safe_game_ticks,
        average_eu,
        total_eu,
        safety_margin,
    )


@njit(cache=True, parallel=True)
def _evaluate_batch_kernel(
    component_codes,
    initial_hull_heat,
    max_reactor_ticks,
    neighbors,
    kind,
    max_heat,
    max_damage,
    rod_count,
    internal_pulses,
    self_vent,
    hull_draw,
    side_vent,
    exchange_side,
    exchange_hull,
    hull_capacity_bonus,
    accepts_heat,
    out_mark_family,
    out_mark_level,
    out_mark_flags,
    out_stop_reason,
    out_reactor_ticks,
    out_safe_game_ticks,
    out_average_eu,
    out_total_eu,
    out_safety_margin,
):
    for row in prange(component_codes.shape[0]):
        result = _simulate_one(
            component_codes[row],
            initial_hull_heat[row],
            max_reactor_ticks,
            neighbors,
            kind,
            max_heat,
            max_damage,
            rod_count,
            internal_pulses,
            self_vent,
            hull_draw,
            side_vent,
            exchange_side,
            exchange_hull,
            hull_capacity_bonus,
            accepts_heat,
        )
        out_mark_family[row] = result[0]
        out_mark_level[row] = result[1]
        out_mark_flags[row] = result[2]
        out_stop_reason[row] = result[3]
        out_reactor_ticks[row] = result[4]
        out_safe_game_ticks[row] = result[5]
        out_average_eu[row] = result[6]
        out_total_eu[row] = result[7]
        out_safety_margin[row] = result[8]


class NumbaPackedEvaluator:
    """Parallel CPU implementation of the packed evaluator ABI."""

    def __init__(self, num_threads: int | None = None):
        self.num_threads = num_threads

    def evaluate(
        self,
        batch: PackedLayoutBatch,
        max_reactor_ticks: int,
    ) -> PackedEvaluationBatch:
        if self.num_threads is not None:
            # Numba only permits masks up to the thread pool created at import.
            set_num_threads(max(1, min(self.num_threads, config.NUMBA_NUM_THREADS)))
        size = batch.batch_size
        mark_family = np.zeros(size, dtype=np.uint8)
        mark_level = np.zeros(size, dtype=np.uint8)
        mark_flags = np.zeros(size, dtype=np.uint8)
        stop_reason = np.zeros(size, dtype=np.uint8)
        reactor_ticks = np.zeros(size, dtype=np.int64)
        safe_game_ticks = np.zeros(size, dtype=np.int64)
        average_eu_per_tick = np.zeros(size, dtype=np.float64)
        total_eu = np.zeros(size, dtype=np.float64)
        safety_margin = np.zeros(size, dtype=np.float64)
        table = COMPONENT_KERNEL_TABLE

        _evaluate_batch_kernel(
            batch.component_codes,
            batch.initial_hull_heat,
            max_reactor_ticks,
            packed_neighbor_table(batch.columns),
            table.kind,
            table.max_heat,
            table.max_damage,
            table.rod_count,
            table.internal_pulses,
            table.self_vent,
            table.hull_draw,
            table.side_vent,
            table.exchange_side,
            table.exchange_hull,
            table.hull_capacity_bonus,
            table.accepts_heat,
            mark_family,
            mark_level,
            mark_flags,
            stop_reason,
            reactor_ticks,
            safe_game_ticks,
            average_eu_per_tick,
            total_eu,
            safety_margin,
        )
        return PackedEvaluationBatch(
            mark_family=mark_family,
            mark_level=mark_level,
            mark_flags=mark_flags,
            stop_reason=stop_reason,
            reactor_ticks=reactor_ticks,
            safe_game_ticks=safe_game_ticks,
            average_eu_per_tick=average_eu_per_tick,
            total_eu=total_eu,
            safety_margin=safety_margin,
        )
