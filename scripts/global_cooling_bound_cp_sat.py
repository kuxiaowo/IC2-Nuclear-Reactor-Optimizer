"""Rigorous sustainable-power upper bound from component dominance.

For an infinite safe period, storage and condensators cannot be net heat
sinks, and exchangers only move heat.  Every non-fuel/non-reflector slot is
therefore optimistically replaced by either an overclocked heat vent O or a
component vent C.  O dominates every self-venting real component (20 heat per
tick), while every C--O adjacency contributes four additional heat per tick.

The model deliberately allows heat to teleport between slots.  Consequently
``generated heat <= total optimistic cooling`` is necessary, not sufficient,
and the optimum is a rigorous global upper bound for every safe periodic
reactor under the unlimited-component interpretation.
"""

from __future__ import annotations

import argparse
from time import perf_counter

from ortools.sat.python import cp_model


ROWS, COLUMNS = 6, 9
SINGLE, DUAL, QUAD, REFLECTOR, OVERCLOCKED, COMPONENT_VENT = range(6)
FUEL = (SINGLE, DUAL, QUAD)
RODS = (1, 2, 4, 0, 0, 0)
INTERNAL = (1, 2, 3, 0, 0, 0)
SYMBOLS = ("S", "D", "Q", "R", "O", "C")


def neighbours(index: int) -> tuple[int, ...]:
    row, column = divmod(index, COLUMNS)
    result = []
    if column:
        result.append(index - 1)
    if column + 1 < COLUMNS:
        result.append(index + 1)
    if row:
        result.append(index - COLUMNS)
    if row + 1 < ROWS:
        result.append(index + COLUMNS)
    return tuple(result)


def edges() -> tuple[tuple[int, int], ...]:
    result = []
    for index in range(ROWS * COLUMNS):
        row, column = divmod(index, COLUMNS)
        if column + 1 < COLUMNS:
            result.append((index, index + 1))
        if row + 1 < ROWS:
            result.append((index, index + COLUMNS))
    return tuple(result)


def build_model(total_rods: int) -> tuple:
    model = cp_model.CpModel()
    slots = ROWS * COLUMNS
    one_hot = [[model.new_bool_var(f"x_{i}_{k}") for k in range(6)] for i in range(slots)]
    active = [model.new_bool_var(f"active_{i}") for i in range(slots)]
    degree = [model.new_int_var(0, 4, f"degree_{i}") for i in range(slots)]
    power_terms, heat_terms = [], []

    for index in range(slots):
        model.add_exactly_one(one_hot[index])
        model.add(active[index] == sum(one_hot[index][k] for k in (*FUEL, REFLECTOR)))
        model.add(degree[index] == sum(active[j] for j in neighbours(index)))
        for kind in FUEL:
            states = []
            for adjacent_active in range(5):
                state = model.new_bool_var(f"state_{index}_{kind}_{adjacent_active}")
                states.append(state)
                model.add(degree[index] == adjacent_active).only_enforce_if(state)
                pulses = INTERNAL[kind] + adjacent_active
                power_terms.append(5 * RODS[kind] * pulses * state)
                heat_terms.append(2 * RODS[kind] * pulses * (pulses + 1) * state)
            model.add(sum(states) == one_hot[index][kind])

    model.add(sum(RODS[k] * one_hot[i][k] for i in range(slots) for k in FUEL) == total_rods)

    component_edges = []
    for first, second in edges():
        for component, vent in ((first, second), (second, first)):
            adjacent = model.new_bool_var(f"co_{component}_{vent}")
            model.add(adjacent <= one_hot[component][COMPONENT_VENT])
            model.add(adjacent <= one_hot[vent][OVERCLOCKED])
            model.add(adjacent >= one_hot[component][COMPONENT_VENT] + one_hot[vent][OVERCLOCKED] - 1)
            component_edges.append(adjacent)

    power = model.new_int_var(0, 10_000, "power")
    heat = model.new_int_var(0, 30_000, "heat")
    cooling = model.new_int_var(0, 10_000, "optimistic_cooling")
    model.add(power == sum(power_terms))
    model.add(heat == sum(heat_terms))
    model.add(cooling == 20 * sum(one_hot[i][OVERCLOCKED] for i in range(slots)) + 4 * sum(component_edges))
    model.add(heat <= cooling)
    model.maximize(power)
    return model, one_hot, power, heat, cooling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rods", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()
    model, one_hot, power, heat, cooling = build_model(args.rods)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    started = perf_counter()
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} elapsed={perf_counter() - started:.3f}s "
        f"objective={solver.objective_value} bound={solver.best_objective_bound}"
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    codes = [max(range(6), key=lambda k: solver.value(one_hot[i][k])) for i in range(54)]
    print(f"power={solver.value(power)} heat={solver.value(heat)} cooling={solver.value(cooling)}")
    for row in range(ROWS):
        print(" ".join(SYMBOLS[k] for k in codes[row * COLUMNS : (row + 1) * COLUMNS]))


if __name__ == "__main__":
    main()
