from __future__ import annotations

FUEL_CYCLE_REACTOR_TICKS = 20_000
TEN_PERCENT_CYCLE = 2_000


def classify_mark(
    first_critical_tick: int | None,
    first_component_break_tick: int | None,
    stable: bool,
    uses_single_use_coolant: bool = False,
) -> str | None:
    suffix = "-SUC" if uses_single_use_coolant else ""
    if stable and first_critical_tick is None and first_component_break_tick is None:
        return f"Mark I-I{suffix}"

    events = [tick for tick in (first_critical_tick, first_component_break_tick) if tick is not None]
    if not events:
        return None
    intervention = min(events)

    if intervention >= FUEL_CYCLE_REACTOR_TICKS:
        cycles = intervention // FUEL_CYCLE_REACTOR_TICKS
        level = "E" if cycles >= 16 else str(max(1, cycles))
        return f"Mark II-{level}{suffix}"
    if intervention < TEN_PERCENT_CYCLE:
        return f"Mark V{suffix}"

    broke_first = first_component_break_tick is not None and (
        first_critical_tick is None or first_component_break_tick <= first_critical_tick
    )
    return f"Mark IV{suffix}" if broke_first else f"Mark III{suffix}"


def mark_family(mark: str | None) -> str | None:
    if not mark:
        return None
    # 必须先匹配长罗马数字；"Mark II" 同样以 "Mark I" 开头。
    for family in ("III", "II", "IV", "V", "I"):
        if mark.startswith(f"Mark {family}"):
            return family
    return None
