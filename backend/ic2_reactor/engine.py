from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Callable

from .components import COMPONENTS, ComponentSpec
from .mark import FUEL_CYCLE_REACTOR_TICKS, classify_mark
from .models import EventType, Layout, ReactorEvent, SimulationSummary, StopReason


@dataclass(slots=True)
class SlotState:
    component_id: str
    heat: int = 0
    damage: int = 0
    removed: bool = False
    spec: ComponentSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # A stack's component id never mutates; removed/replaced stacks get a
        # fresh SlotState. Cache the immutable specification for this hot path.
        self.spec = COMPONENTS[self.component_id]

    @property
    def broken(self) -> bool:
        """Whether this concrete component stack has been removed from the reactor."""
        return self.removed


@dataclass(slots=True)
class ReactorCycleRecord:
    reactor_tick: int
    hull_heat: int
    max_hull_heat: int
    eu_per_tick: float
    total_eu: float
    generated_heat: int
    vented_heat: int
    component_heat: tuple[int, ...]
    component_damage: tuple[int, ...]
    component_ids: tuple[str, ...]


@dataclass(slots=True)
class SimulationOptions:
    max_game_ticks: int = 400_000
    auto_refuel: bool = False
    stop_on_stable: bool = False
    record_components: bool = True
    record_history: bool = True
    stable_check_interval: int = FUEL_CYCLE_REACTOR_TICKS
    cancel_check: Callable[[], bool] | None = None


@dataclass(slots=True)
class SimulationRun:
    summary: SimulationSummary
    records: list[ReactorCycleRecord] = field(default_factory=list)


class ReactorSimulator:
    """IC2 Experimental 2.8.221 EU-reactor simulator.

    ``step`` is one IC2 reactor update (20 Minecraft game ticks).  The
    implementation follows the production bytecode's two row-major chamber
    passes and deliberately keeps all immediate inventory side effects.
    """

    BASE_HULL_HEAT = 10_000
    CRITICAL_RATIO = 0.85
    EU_PER_PULSE = 5.0
    OFFICIAL_NEIGHBOR_ORDER = ("left", "right", "up", "down")

    def __init__(self, layout: Layout):
        self.columns = layout.columns
        self.slots = [SlotState(component_id=item) for item in layout.slots]
        self._neighbor_indices = tuple(
            self._calculate_neighbors(index) for index in range(len(self.slots))
        )
        self.initial_component_ids = tuple(layout.slots)
        self.hull_heat = layout.initial_hull_heat
        self.max_hull_heat = self.BASE_HULL_HEAT + sum(
            slot.spec.hull_capacity_bonus for slot in self.slots
        )
        self.reactor_tick = 0
        self.total_eu = 0.0
        self.peak_hull_heat = self.hull_heat
        self.first_critical_tick: int | None = None
        self.first_component_break_tick: int | None = None
        self.meltdown_tick: int | None = None
        self.events: list[ReactorEvent] = []
        self._critical_reported = False
        self._output_pulses = 0
        self.uses_single_use_coolant = any(
            COMPONENTS[item].kind == "condensator"
            or (COMPONENTS[item].kind == "reflector" and COMPONENTS[item].max_damage > 0)
            for item in self.initial_component_ids
        )

    def _calculate_neighbors(self, index: int) -> tuple[int, ...]:
        row, col = divmod(index, self.columns)
        result: list[int] = []
        for direction in self.OFFICIAL_NEIGHBOR_ORDER:
            if direction == "left" and col > 0:
                result.append(index - 1)
            elif direction == "right" and col + 1 < self.columns:
                result.append(index + 1)
            elif direction == "up" and row > 0:
                result.append(index - self.columns)
            elif direction == "down" and row < 5:
                result.append(index + self.columns)
        return tuple(result)

    def _neighbors(self, index: int) -> tuple[int, ...]:
        return self._neighbor_indices[index]

    def _active(self, index: int) -> bool:
        slot = self.slots[index]
        return slot.component_id != "empty" and not slot.removed

    def _adjust_hull(self, delta: int) -> int:
        before = self.hull_heat
        self.hull_heat = max(0, self.hull_heat + delta)
        return self.hull_heat - before

    def _remove_component(self, index: int, slot: SlotState) -> None:
        """Remove the exact stack currently occupying ``index`` and report it."""
        if self.slots[index] is not slot or slot.removed:
            return
        component_id = slot.component_id
        spec = slot.spec
        slot.removed = True
        self.slots[index] = SlotState("empty")
        if spec.kind != "fuel" and self.first_component_break_tick is None:
            self.first_component_break_tick = self.reactor_tick
        self.events.append(ReactorEvent(
            reactor_tick=self.reactor_tick,
            game_tick=self.reactor_tick * 20,
            type=EventType.COMPONENT_BROKEN,
            slot=index,
            component_id=component_id,
            message=f"第 {index + 1} 格 {spec.name} 损坏",
        ))

    def _deplete_fuel(self, index: int, slot: SlotState, auto_refuel: bool) -> None:
        if self.slots[index] is not slot or slot.removed:
            return
        component_id = slot.component_id
        spec = slot.spec
        slot.removed = True
        if auto_refuel:
            self.slots[index] = SlotState(component_id)
            event_type = EventType.REFUEL
            message = f"第 {index + 1} 格燃料棒原位更换"
        else:
            # Depleted uranium is not an IReactorComponent.  Runtime traces use
            # an empty slot rather than exposing a non-placeable component id.
            self.slots[index] = SlotState("empty")
            event_type = EventType.FUEL_DEPLETED
            message = f"第 {index + 1} 格 {spec.name} 耗尽"
        self.events.append(ReactorEvent(
            reactor_tick=self.reactor_tick,
            game_tick=self.reactor_tick * 20,
            type=event_type,
            slot=index,
            component_id=component_id,
            message=message,
        ))

    def _can_store_heat(self, index: int) -> bool:
        if not self._active(index):
            return False
        slot = self.slots[index]
        spec = slot.spec
        if not spec.accepts_heat:
            return False
        if spec.kind == "condensator":
            return slot.heat < spec.max_heat
        # ItemReactorHeatStorage.canStoreHeat() stays true at exactly max heat.
        return True

    def _alter_heat(self, index: int, delta: int, slot: SlotState | None = None) -> int:
        """Apply IC2 ``alterHeat`` and return its signed remainder."""
        slot = self.slots[index] if slot is None else slot
        spec = slot.spec
        if not spec.accepts_heat:
            return delta

        if spec.kind == "condensator":
            if delta < 0:
                return delta
            accepted = min(delta, spec.max_heat - slot.heat)
            slot.heat += accepted
            return delta - accepted

        target = slot.heat + delta
        if target > spec.max_heat:
            remainder = spec.max_heat - target + 1
            self._remove_component(index, slot)
            return remainder
        if target < 0:
            slot.heat = 0
            return target
        slot.heat = target
        return 0

    def _accept_uranium_pulse(self, target: int, source: int, heat_run: bool) -> bool:
        if not self._active(target):
            return False
        slot = self.slots[target]
        kind = slot.spec.kind
        if kind == "fuel":
            if not heat_run:
                self._output_pulses += 1
            return True
        if kind != "reflector":
            return False

        if heat_run:
            if slot.spec.max_damage > 0:
                if slot.damage + 1 >= slot.spec.max_damage:
                    self._remove_component(target, slot)
                else:
                    slot.damage += 1
        else:
            # Reflectors forward the energy pulse back to the source fuel rod.
            if self._active(source) and self.slots[source].spec.kind == "fuel":
                self._output_pulses += 1
        return True

    def _distribute_fuel_heat(self, index: int, heat: int) -> None:
        acceptors = [
            neighbor
            for neighbor in self._neighbors(index)
            if self._can_store_heat(neighbor)
        ]
        remaining = heat
        while acceptors and remaining > 0:
            amount = remaining // len(acceptors)
            remaining -= amount
            target = acceptors.pop(0)
            target_slot = self.slots[target]
            remaining += self._alter_heat(target, amount, target_slot)
        if remaining > 0:
            self._adjust_hull(remaining)

    def _process_fuel(self, index: int, heat_run: bool, auto_refuel: bool = False) -> int:
        slot = self.slots[index]
        spec = slot.spec
        generated_heat = 0
        for _ in range(spec.rod_count):
            pulses = spec.internal_pulses
            if not heat_run:
                # Internal fuel pulses always target the still-active source
                # stack, so their energy contribution can be accumulated in
                # one operation without changing pulse ordering.
                self._output_pulses += spec.internal_pulses
            for neighbor in self._neighbors(index):
                if self._accept_uranium_pulse(neighbor, index, heat_run):
                    pulses += 1
            if heat_run:
                heat = 4 * (pulses * (pulses + 1) // 2)
                generated_heat += heat
                self._distribute_fuel_heat(index, heat)

        if not heat_run:
            if slot.damage >= spec.max_damage - 1:
                self._deplete_fuel(index, slot, auto_refuel)
            else:
                slot.damage += 1
        return generated_heat

    def _vent(self, index: int) -> int:
        slot = self.slots[index]
        spec = slot.spec
        vented = 0
        if spec.hull_draw:
            hull_after_draw = self.hull_heat - min(spec.hull_draw, self.hull_heat)
            drawn = self.hull_heat - hull_after_draw
            returned = self._alter_heat(index, drawn, slot)
            if returned > 0:
                return 0
            self.hull_heat = hull_after_draw
        if spec.self_vent:
            returned = self._alter_heat(index, -spec.self_vent, slot)
            if returned <= 0:
                vented += returned + spec.self_vent
        return vented

    def _spread_vent(self, index: int) -> int:
        spec = self.slots[index].spec
        vented = 0
        for neighbor in self._neighbors(index):
            if not self._can_store_heat(neighbor):
                continue
            neighbor_slot = self.slots[neighbor]
            returned = self._alter_heat(neighbor, -spec.side_vent, neighbor_slot)
            if returned <= 0:
                vented += returned + spec.side_vent
        return vented

    @staticmethod
    def _exchange_amount(
        source_ratio: float,
        target_ratio: float,
        target_capacity: int,
        limit: int,
        *,
        rounded_base: bool = False,
        low_range: int | None = None,
    ) -> int:
        combined = target_ratio + source_ratio / 2.0
        raw = target_capacity / 100.0 * combined
        amount = floor(raw + 0.5) if rounded_base else int(raw)
        amount = min(amount, limit)
        threshold_range = limit if low_range is None else low_range
        if combined < 1.0:
            amount = threshold_range // 2
        if combined < 0.75:
            amount = threshold_range // 4
        if combined < 0.5:
            amount = threshold_range // 8
        if combined < 0.25:
            amount = 1
        source_tenth = floor(source_ratio * 10.0 + 0.5) / 10.0
        target_tenth = floor(target_ratio * 10.0 + 0.5) / 10.0
        if target_tenth > source_tenth:
            amount = -amount
        elif target_tenth == source_tenth:
            amount = 0
        return amount

    def _exchange(self, index: int) -> None:
        slot = self.slots[index]
        spec = slot.spec
        my_heat_delta = 0
        neighbors = [
            (neighbor, self.slots[neighbor])
            for neighbor in self._neighbors(index)
            if self._can_store_heat(neighbor)
        ]
        if spec.exchange_side:
            for neighbor, neighbor_slot in neighbors:
                mine = slot.heat * 100.0 / spec.max_heat
                other_spec = neighbor_slot.spec
                theirs = neighbor_slot.heat * 100.0 / other_spec.max_heat
                amount = self._exchange_amount(mine, theirs, other_spec.max_heat, spec.exchange_side)
                my_heat_delta -= amount
                my_heat_delta += self._alter_heat(neighbor, amount, neighbor_slot)
        if spec.exchange_hull:
            mine = slot.heat * 100.0 / spec.max_heat
            hull = self.hull_heat * 100.0 / self.max_hull_heat
            amount = self._exchange_amount(
                mine,
                hull,
                self.max_hull_heat,
                spec.exchange_hull,
                rounded_base=True,
                low_range=spec.exchange_side,
            )
            my_heat_delta -= amount
            self.hull_heat += amount
        self._alter_heat(index, my_heat_delta, slot)

    def step(self, auto_refuel: bool = False) -> tuple[float, int, int]:
        self.reactor_tick += 1
        self._output_pulses = 0
        generated_heat = 0
        vented_heat = 0

        # IC2 resets these before every chamber pass.  Plating takes effect only
        # when its row-major slot is reached during the heat run.
        self.max_hull_heat = self.BASE_HULL_HEAT
        for index in range(len(self.slots)):
            slot = self.slots[index]
            if slot.component_id == "empty" or slot.removed:
                continue
            kind = slot.spec.kind
            if kind == "fuel":
                generated_heat += self._process_fuel(index, True)
            elif kind == "vent":
                if slot.spec.side_vent:
                    vented_heat += self._spread_vent(index)
                else:
                    vented_heat += self._vent(index)
            elif kind == "exchanger":
                self._exchange(index)
            elif kind == "plating":
                self.max_hull_heat += self.slots[index].spec.hull_capacity_bonus

        for index in range(len(self.slots)):
            slot = self.slots[index]
            if slot.component_id != "empty" and not slot.removed and slot.spec.kind == "fuel":
                self._process_fuel(index, False, auto_refuel)

        eu_per_tick = self._output_pulses * self.EU_PER_PULSE
        self.total_eu += eu_per_tick * 20
        self.peak_hull_heat = max(self.peak_hull_heat, self.hull_heat)
        if not self._critical_reported and self.hull_heat >= floor(self.max_hull_heat * self.CRITICAL_RATIO):
            self._critical_reported = True
            self.first_critical_tick = self.reactor_tick
            self.events.append(ReactorEvent(
                reactor_tick=self.reactor_tick,
                game_tick=self.reactor_tick * 20,
                type=EventType.CRITICAL,
                message="堆体达到 85% 临界热量",
            ))

        if self.hull_heat >= self.max_hull_heat:
            self.meltdown_tick = self.reactor_tick
            self.events.append(ReactorEvent(
                reactor_tick=self.reactor_tick,
                game_tick=self.reactor_tick * 20,
                type=EventType.MELTDOWN,
                message="反应堆融毁",
            ))
        return eu_per_tick, generated_heat, vented_heat

    def state_signature(self, include_fuel_damage: bool = False) -> tuple:
        values: list[int | str] = [self.hull_heat]
        for slot in self.slots:
            values.extend((slot.component_id, slot.heat))
            if include_fuel_damage or slot.spec.kind != "fuel":
                values.append(slot.damage)
        return tuple(values)

    def thermal_state_signature(self) -> tuple[int, ...]:
        """Minimal future-relevant key for a fixed auto-refuel layout.

        Component ids are fixed within one trajectory.  Zero-capacity heat,
        zero-durability damage and fuel damage cannot affect future thermal
        transitions, so repeating them in every cycle key only wastes hashing
        and memory.
        """

        values = [self.hull_heat]
        for slot in self.slots:
            if slot.spec.accepts_heat:
                values.append(slot.heat)
            if slot.spec.kind != "fuel" and slot.spec.max_damage > 0:
                values.append(slot.damage)
        return tuple(values)

    def _fast_forward_fixed_state(self, target_tick: int, eu_per_tick: float) -> None:
        """Advance a proven thermal fixed point without replaying every cycle.

        Fuel damage does not participate in heat or energy production. With
        automatic refuelling, depletion replaces a rod with a fresh identical
        stack, so only damage counters and refuel events need advancing.
        """
        start_tick = self.reactor_tick
        remaining = target_tick - start_tick
        refuels: list[tuple[int, int, str]] = []
        for index, slot in enumerate(self.slots):
            if slot.component_id == "empty" or slot.removed or slot.spec.kind != "fuel":
                continue
            first_refuel = slot.spec.max_damage - slot.damage
            for offset in range(first_refuel, remaining + 1, slot.spec.max_damage):
                refuels.append((start_tick + offset, index, slot.component_id))
            slot.damage = (slot.damage + remaining) % slot.spec.max_damage

        # Energy-pass refuels are emitted in tick and row-major slot order.
        for reactor_tick, index, component_id in sorted(refuels):
            self.events.append(ReactorEvent(
                reactor_tick=reactor_tick,
                game_tick=reactor_tick * 20,
                type=EventType.REFUEL,
                slot=index,
                component_id=component_id,
                message=f"第 {index + 1} 格燃料棒原位更换",
            ))
        self.total_eu += eu_per_tick * 20 * remaining
        self.reactor_tick = target_tick

    def simulate(self, options: SimulationOptions | None = None) -> SimulationRun:
        options = options or SimulationOptions()
        if options.max_game_ticks % 20:
            raise ValueError("max_game_ticks 必须是 20 的倍数")
        max_reactor_ticks = options.max_game_ticks // 20
        records: list[ReactorCycleRecord] = []
        stable = False
        signatures: dict[tuple, int] = {}
        last_eu = 0.0
        safe_eu = 0.0
        safe_cycle_count = 0
        can_fast_forward = options.auto_refuel and options.stop_on_stable and not options.record_history
        previous_thermal_state = self.state_signature(include_fuel_damage=False) if can_fast_forward else None

        for cycle in range(max_reactor_ticks):
            # Cross-process event checks are comparatively expensive. Polling
            # every 64 reactor cycles remains responsive without doing IPC on
            # every simulated second.
            if options.cancel_check and cycle % 64 == 0 and options.cancel_check():
                reason = StopReason.CANCELLED
                break
            last_eu, generated, vented = self.step(options.auto_refuel)
            if self.first_critical_tick is None:
                current_intervention = self.first_component_break_tick
            elif self.first_component_break_tick is None:
                current_intervention = self.first_critical_tick
            else:
                current_intervention = min(self.first_critical_tick, self.first_component_break_tick)
            if current_intervention is None or self.reactor_tick <= current_intervention:
                safe_eu += last_eu
                safe_cycle_count += 1
            if options.record_history:
                records.append(ReactorCycleRecord(
                    reactor_tick=self.reactor_tick,
                    hull_heat=self.hull_heat,
                    max_hull_heat=self.max_hull_heat,
                    eu_per_tick=last_eu,
                    total_eu=self.total_eu,
                    generated_heat=generated,
                    vented_heat=vented,
                    component_heat=tuple(slot.heat for slot in self.slots) if options.record_components else (),
                    component_damage=tuple(slot.damage for slot in self.slots) if options.record_components else (),
                    component_ids=tuple(slot.component_id for slot in self.slots) if options.record_components else (),
                ))
            if self.meltdown_tick is not None:
                reason = StopReason.MELTDOWN
                break

            if (
                can_fast_forward
                and self.first_critical_tick is None
                and self.first_component_break_tick is None
                and self.state_signature(include_fuel_damage=False) == previous_thermal_state
            ):
                interval = options.stable_check_interval
                stable_tick = ((self.reactor_tick + interval - 1) // interval + 1) * interval
                target_tick = min(stable_tick, max_reactor_ticks)
                remaining = target_tick - self.reactor_tick
                self._fast_forward_fixed_state(target_tick, last_eu)
                safe_eu += last_eu * remaining
                safe_cycle_count += remaining
                if target_tick == stable_tick:
                    stable = True
                    self.events.append(ReactorEvent(
                        reactor_tick=self.reactor_tick,
                        game_tick=self.reactor_tick * 20,
                        type=EventType.STABLE,
                        message=f"检测到与第 {self.reactor_tick - interval} 周期相同的完整状态",
                    ))
                    reason = StopReason.STABLE
                else:
                    reason = StopReason.TICK_LIMIT
                break
            if can_fast_forward:
                previous_thermal_state = self.state_signature(include_fuel_damage=False)
            if options.auto_refuel and self.reactor_tick % options.stable_check_interval == 0:
                signature = self.state_signature(include_fuel_damage=False)
                if signature in signatures:
                    stable = True
                    self.events.append(ReactorEvent(
                        reactor_tick=self.reactor_tick,
                        game_tick=self.reactor_tick * 20,
                        type=EventType.STABLE,
                        message=f"检测到与第 {signatures[signature]} 周期相同的完整状态",
                    ))
                    if options.stop_on_stable:
                        reason = StopReason.STABLE
                        break
                signatures[signature] = self.reactor_tick
        else:
            reason = StopReason.TICK_LIMIT

        intervention_ticks = [
            tick for tick in (self.first_critical_tick, self.first_component_break_tick) if tick is not None
        ]
        first_intervention = min(intervention_ticks) if intervention_ticks else None
        average = safe_eu / safe_cycle_count if safe_cycle_count else 0.0
        mark = classify_mark(
            self.first_critical_tick,
            self.first_component_break_tick,
            stable,
            self.uses_single_use_coolant,
        )
        summary = SimulationSummary(
            stop_reason=reason,
            reactor_ticks=self.reactor_tick,
            game_ticks=self.reactor_tick * 20,
            hull_heat=self.hull_heat,
            max_hull_heat=self.max_hull_heat,
            peak_hull_heat=self.peak_hull_heat,
            current_eu_per_tick=last_eu,
            average_eu_per_tick=average,
            total_eu=self.total_eu,
            first_intervention_tick=first_intervention * 20 if first_intervention else None,
            meltdown_tick=self.meltdown_tick * 20 if self.meltdown_tick else None,
            mark=mark,
            stable=stable,
            events=self.events,
        )
        return SimulationRun(summary=summary, records=records)
