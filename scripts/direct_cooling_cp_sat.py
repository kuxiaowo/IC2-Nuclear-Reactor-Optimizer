"""Joint 25-rod power and direct-cooling construction model.

The model is independent from the production optimizer.  Every fuel must send
all heat directly to adjacent overclocked vents.  Each such vent receives a
conservative (rounded-up) share and must be locally balanced by its own 20
venting plus four units for every adjacent component vent.  A returned layout
is still verified by the exact deterministic simulator.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter

from ortools.sat.python import cp_model


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.engine import ReactorSimulator  # noqa: E402
from ic2_reactor.models import Layout  # noqa: E402


ROWS = 6
EMPTY, SINGLE, DUAL, QUAD, REFLECTOR, OVERCLOCKED, COMPONENT_VENT = range(7)
FUEL_CODES = (SINGLE, DUAL, QUAD)
RODS = (0, 1, 2, 4, 0, 0, 0)
INTERNAL = (0, 1, 2, 3, 0, 0, 0)
SYMBOLS = (".", "S", "D", "Q", "R", "O", "C")
COMPONENT_IDS = (
    "empty",
    "uranium_single",
    "uranium_dual",
    "uranium_quad",
    "iridium_reflector",
    "overclocked_heat_vent",
    "component_heat_vent",
)


def neighbours(index: int, columns: int) -> tuple[int, ...]:
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


def build_model(columns: int, rods: int) -> tuple:
    slots = ROWS * columns
    model = cp_model.CpModel()
    labels = [model.new_int_var(0, len(SYMBOLS) - 1, f"label_{i}") for i in range(slots)]
    one_hot = [
        [model.new_bool_var(f"is_{i}_{code}") for code in range(len(SYMBOLS))]
        for i in range(slots)
    ]
    active = [model.new_bool_var(f"power_active_{i}") for i in range(slots)]
    is_overclocked = [model.new_bool_var(f"overclocked_{i}") for i in range(slots)]
    is_component_vent = [model.new_bool_var(f"component_vent_{i}") for i in range(slots)]
    power_degree = [model.new_int_var(0, 4, f"power_degree_{i}") for i in range(slots)]
    overclocked_degree = [model.new_int_var(0, 4, f"overclocked_degree_{i}") for i in range(slots)]
    power_terms = []
    heat_terms = []

    for index in range(slots):
        model.add_exactly_one(one_hot[index])
        model.add(labels[index] == sum(code * one_hot[index][code] for code in range(len(SYMBOLS))))
        model.add(active[index] == sum(one_hot[index][code] for code in (*FUEL_CODES, REFLECTOR)))
        model.add(is_overclocked[index] == one_hot[index][OVERCLOCKED])
        model.add(is_component_vent[index] == one_hot[index][COMPONENT_VENT])
        adjacent = neighbours(index, columns)
        model.add(power_degree[index] == sum(active[other] for other in adjacent))
        model.add(overclocked_degree[index] == sum(is_overclocked[other] for other in adjacent))
        for code in FUEL_CODES:
            model.add(overclocked_degree[index] >= 1).only_enforce_if(one_hot[index][code])

        for code in FUEL_CODES:
            choices = []
            for degree in range(5):
                choice = model.new_bool_var(f"fuel_degree_{index}_{code}_{degree}")
                model.add(power_degree[index] == degree).only_enforce_if(choice)
                choices.append(choice)
                pulses = INTERNAL[code] + degree
                power_terms.append(5 * RODS[code] * pulses * choice)
                heat_terms.append(2 * RODS[code] * pulses * (pulses + 1) * choice)
            model.add(sum(choices) == one_hot[index][code])

    model.add(sum(RODS[code] * one_hot[index][code] for index in range(slots) for code in FUEL_CODES) == rods)

    # Directed conservative heat shares from a potential fuel cell to each
    # adjacent potential overclocked vent.
    incoming: list[list[cp_model.IntVar]] = [[] for _ in range(slots)]
    tuples = []
    for code in range(len(SYMBOLS)):
        for power_neighbours in range(5):
            for vent_neighbours in range(5):
                for target_is_vent in range(2):
                    contribution = 0
                    if code in FUEL_CODES and target_is_vent and vent_neighbours:
                        pulses = INTERNAL[code] + power_neighbours
                        per_rod_heat = 2 * pulses * (pulses + 1)
                        contribution = RODS[code] * math.ceil(per_rod_heat / vent_neighbours)
                    tuples.append((code, power_neighbours, vent_neighbours, target_is_vent, contribution))

    for source in range(slots):
        for target in neighbours(source, columns):
            contribution = model.new_int_var(0, 336, f"heat_{source}_to_{target}")
            model.add_allowed_assignments(
                [labels[source], power_degree[source], overclocked_degree[source], is_overclocked[target], contribution],
                tuples,
            )
            incoming[target].append(contribution)

    for index in range(slots):
        component_vent_neighbours = sum(is_component_vent[other] for other in neighbours(index, columns))
        model.add(sum(incoming[index]) <= 20 + 4 * component_vent_neighbours).only_enforce_if(is_overclocked[index])

    power = model.new_int_var(0, 10_000, "power")
    generated_heat = model.new_int_var(0, 30_000, "generated_heat")
    model.add(power == sum(power_terms))
    model.add(generated_heat == sum(heat_terms))
    model.maximize(power)
    return model, labels, one_hot, power, generated_heat


def exact_cycle(layout: tuple[str, ...], horizon: int = 100_000) -> tuple[int, int] | None:
    simulator = ReactorSimulator(Layout(columns=9, slots=list(layout)))
    seen = {simulator.state_signature(include_fuel_damage=False): 0}
    for tick in range(1, horizon + 1):
        simulator.step(auto_refuel=True)
        if simulator.first_critical_tick is not None or simulator.first_component_break_tick is not None:
            return None
        signature = simulator.state_signature(include_fuel_damage=False)
        previous = seen.get(signature)
        if previous is not None:
            return previous, tick - previous
        seen[signature] = tick
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--rods", type=int, default=25)
    args = parser.parse_args()
    model, labels, _one_hot, power, generated_heat = build_model(9, args.rods)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    started = perf_counter()
    status = solver.solve(model)
    elapsed = perf_counter() - started
    print(
        f"status={solver.status_name(status)} elapsed={elapsed:.3f}s "
        f"bound={solver.best_objective_bound}"
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    codes = tuple(solver.value(label) for label in labels)
    layout = tuple(COMPONENT_IDS[code] for code in codes)
    cycle = exact_cycle(layout)
    print(f"power={solver.value(power)} heat={solver.value(generated_heat)} cycle={cycle}")
    for row in range(ROWS):
        print(" ".join(SYMBOLS[code] for code in codes[row * 9 : (row + 1) * 9]))
    print(repr(layout))


if __name__ == "__main__":
    main()
