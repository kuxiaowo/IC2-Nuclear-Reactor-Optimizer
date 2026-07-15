"""Research prototype for the 54-slot, 25-rod optimization model.

This file deliberately models the problem independently from the optimizer's
search implementation.  The production simulator remains the final oracle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

from ortools.sat.python import cp_model


ROWS = 6
FUEL_TYPES = (1, 2, 3)


@dataclass(frozen=True)
class PowerType:
    symbol: str
    rods: int
    internal_pulses: int
    accepts_pulse: bool


POWER_TYPES = (
    PowerType(".", 0, 0, False),  # cooling-layer slot
    PowerType("S", 1, 1, True),
    PowerType("D", 2, 2, True),
    PowerType("Q", 4, 3, True),
    PowerType("R", 0, 0, True),  # iridium reflector
)


def neighbors(index: int, columns: int) -> tuple[int, ...]:
    row, column = divmod(index, columns)
    result = []
    if column:
        result.append(index - 1)
    if column + 1 < columns:
        result.append(index + 1)
    if row:
        result.append(index - columns)
    if row + 1 < ROWS:
        result.append(index + columns)
    return tuple(result)


def edges(columns: int) -> tuple[tuple[int, int], ...]:
    result = []
    for index in range(ROWS * columns):
        row, column = divmod(index, columns)
        if column + 1 < columns:
            result.append((index, index + 1))
        if row + 1 < ROWS:
            result.append((index, index + columns))
    return tuple(result)


def edge_power(first: int, second: int) -> int:
    a, b = POWER_TYPES[first], POWER_TYPES[second]
    pulses = a.rods * int(a.rods > 0 and b.accepts_pulse)
    pulses += b.rods * int(b.rods > 0 and a.accepts_pulse)
    return 5 * pulses


def build_power_heat_model(
    *,
    columns: int,
    rods: int,
    exact_rods: bool,
    aggregate_cooling_bound: bool,
    component_cap: int,
) -> tuple[cp_model.CpModel, list[list[cp_model.BoolVar]], cp_model.IntVar, cp_model.IntVar, list[cp_model.IntVar]]:
    """Build an exact power/heat master and an optional safe cooling relaxation."""
    slots = ROWS * columns
    model = cp_model.CpModel()
    labels = [
        [model.new_bool_var(f"is_{index}_{code}") for code in range(len(POWER_TYPES))]
        for index in range(slots)
    ]
    active_vars = [model.new_int_var(0, 1, f"active_{i}") for i in range(slots)]
    power_terms = []
    heat_terms = []

    for index in range(slots):
        model.add_exactly_one(labels[index])
        model.add(active_vars[index] == sum(labels[index][1:]))

    rod_expression = sum(
        POWER_TYPES[code].rods * labels[index][code]
        for index in range(slots)
        for code in FUEL_TYPES
    )
    if exact_rods:
        model.add(rod_expression == rods)
    else:
        model.add(rod_expression <= rods)

    for index in range(slots):
        degree = model.new_int_var(0, 4, f"pulse_degree_{index}")
        model.add(degree == sum(active_vars[other] for other in neighbors(index, columns)))
        for code in FUEL_TYPES:
            fuel_degree = []
            for adjacent in range(5):
                choice = model.new_bool_var(f"fuel_degree_{index}_{code}_{adjacent}")
                model.add(degree == adjacent).only_enforce_if(choice)
                fuel_degree.append(choice)
                item = POWER_TYPES[code]
                pulses = item.internal_pulses + adjacent
                power_terms.append(5 * item.rods * pulses * choice)
                heat_terms.append(2 * item.rods * pulses * (pulses + 1) * choice)
            model.add(sum(fuel_degree) == labels[index][code])

    power = model.new_int_var(0, 10_000, "total_power")
    heat = model.new_int_var(0, 30_000, "total_heat")
    model.add(power == sum(power_terms))
    model.add(heat == sum(heat_terms))

    if aggregate_cooling_bound:
        # Optimistic per-slot capacities: overclocked vent 20, component vent
        # at most 4*degree <= 16, advanced vent 12, basic vent 6, reactor vent
        # 5.  The component-vent value deliberately ignores whether four
        # compatible neighbours can really be supplied.  Sorting these values
        # gives a safe inventory-aware upper envelope V(c) for c cooling slots.
        cooling_values = sorted(
            [
                *([20] * min(component_cap, slots)),
                *([16] * min(component_cap, slots)),
                *([12] * min(component_cap, slots)),
                *([6] * min(component_cap, slots)),
                *([5] * min(component_cap, slots)),
                *([0] * slots),
            ],
            reverse=True,
        )[:slots]
        cooling_envelope = [sum(cooling_values[:count]) for count in range(slots + 1)]
        cooling_slots = model.new_int_var(0, slots, "cooling_slots")
        max_cooling = model.new_int_var(0, max(cooling_envelope), "max_cooling")
        model.add(cooling_slots == slots - sum(active_vars))
        model.add_element(cooling_slots, cooling_envelope, max_cooling)
        model.add(heat <= max_cooling)

    model.maximize(power)
    return model, labels, power, heat, active_vars


def solve(args: argparse.Namespace) -> None:
    model, labels, power, heat, active = build_power_heat_model(
        columns=args.columns,
        rods=args.rods,
        exact_rods=args.exact_rods,
        aggregate_cooling_bound=not args.no_cooling_bound,
        component_cap=args.component_cap,
    )
    if args.exact_power is not None:
        model.add(power == args.exact_power)
    if args.max_heat is not None:
        model.add(heat <= args.max_heat)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    started = perf_counter()
    status = solver.solve(model)
    elapsed = perf_counter() - started
    print(f"status={solver.status_name(status)} elapsed={elapsed:.3f}s")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    print(
        f"power={solver.value(power)} heat={solver.value(heat)} "
        f"active={sum(solver.value(item) for item in active)} "
        f"bound={solver.best_objective_bound}"
    )
    for row in range(ROWS):
        values = [
            POWER_TYPES[next(code for code, flag in enumerate(labels[row * args.columns + column]) if solver.value(flag))].symbol
            for column in range(args.columns)
        ]
        print(" ".join(values))
    print(f"conflicts={solver.num_conflicts} branches={solver.num_branches}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--columns", type=int, default=9)
    result.add_argument("--rods", type=int, default=25)
    result.add_argument("--exact-rods", action="store_true")
    result.add_argument("--no-cooling-bound", action="store_true")
    result.add_argument("--component-cap", type=int, default=8)
    result.add_argument("--seconds", type=float, default=60)
    result.add_argument("--workers", type=int, default=30)
    result.add_argument("--exact-power", type=int)
    result.add_argument("--max-heat", type=int)
    return result


if __name__ == "__main__":
    solve(parser().parse_args())
