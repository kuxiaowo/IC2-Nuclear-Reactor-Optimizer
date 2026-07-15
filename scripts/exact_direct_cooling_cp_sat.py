"""Exact ordered-split direct-cooling CP-SAT construction model.

Fuel heat must go to adjacent overclocked vents or 60k coolant cells.  Each
receiver is locally balanced by its own venting and adjacent component vents;
the hull and exchangers are unused.  Unlike ``direct_cooling_cp_sat.py``, heat
shares use IC2's exact ordered integer split rather than per-source ceilings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from ic2_reactor.engine import ReactorSimulator  # noqa: E402
from ic2_reactor.models import Layout  # noqa: E402

ROWS, COLUMNS = 6, 9
EMPTY, SINGLE, DUAL, QUAD, REFLECTOR, OVERCLOCKED, COMPONENT_VENT, BUFFER = range(8)
FUEL = (SINGLE, DUAL, QUAD)
RODS = (0, 1, 2, 4, 0, 0, 0, 0)
INTERNAL = (0, 1, 2, 3, 0, 0, 0, 0)
SYMBOLS = (".", "S", "D", "Q", "R", "O", "C", "B")
IDS = ("empty", "uranium_single", "uranium_dual", "uranium_quad", "iridium_reflector",
       "overclocked_heat_vent", "component_heat_vent", "coolant_60k")


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


def split(per_rod_heat: int, mask: int, size: int) -> tuple[list[int], int]:
    receivers = [position for position in range(size) if mask & (1 << position)]
    shares = [0] * size
    remaining, left = per_rod_heat, len(receivers)
    for position in receivers:
        shares[position] = remaining // left
        remaining -= shares[position]
        left -= 1
    return shares, remaining


def build(total_rods: int, min_power: int | None, exact_power: int | None,
          max_heat: int | None, allow_hull: bool,
          aggregate_counts: dict[tuple[int, int], int] | None = None,
          exact_hull_source: int | None = None) -> tuple:
    model = cp_model.CpModel()
    x = [[model.new_bool_var(f"x_{i}_{k}") for k in range(8)] for i in range(54)]
    label = [model.new_int_var(0, 7, f"label_{i}") for i in range(54)]
    active = [model.new_bool_var(f"active_{i}") for i in range(54)]
    accepts = [model.new_bool_var(f"accepts_{i}") for i in range(54)]
    degree = [model.new_int_var(0, 4, f"degree_{i}") for i in range(54)]
    mask = [model.new_int_var(0, 15, f"mask_{i}") for i in range(54)]
    powers, heats = [], []
    fuel_state: dict[tuple[int, int, int], cp_model.IntVar] = {}

    for index in range(54):
        model.add_exactly_one(x[index])
        model.add(label[index] == sum(k * x[index][k] for k in range(8)))
        model.add(active[index] == sum(x[index][k] for k in (*FUEL, REFLECTOR)))
        model.add(accepts[index] == x[index][OVERCLOCKED] + x[index][BUFFER])
        adjacent = neighbours(index)
        model.add(degree[index] == sum(active[j] for j in adjacent))
        model.add(mask[index] == sum((1 << p) * accepts[j] for p, j in enumerate(adjacent)))
        pvar = model.new_int_var(0, 140, f"power_{index}")
        hvar = model.new_int_var(0, 336, f"heat_{index}")
        rows = []
        for kind in range(8):
            for d in range(5):
                pulses = INTERNAL[kind] + d
                rows.append((kind, d, 5 * RODS[kind] * pulses,
                             2 * RODS[kind] * pulses * (pulses + 1)))
        model.add_allowed_assignments([label[index], degree[index], pvar, hvar], rows)
        powers.append(pvar); heats.append(hvar)
        for kind in FUEL:
            states = []
            for d in range(5):
                state = model.new_bool_var(f"fuel_state_{index}_{kind}_{d}")
                fuel_state[index, kind, d] = state
                model.add(state <= x[index][kind])
                model.add(degree[index] == d).only_enforce_if(state)
                states.append(state)
            model.add(sum(states) == x[index][kind])
        if not allow_hull:
            for kind in FUEL:
                model.add(mask[index] >= 1).only_enforce_if(x[index][kind])

    model.add(sum(RODS[k] * x[i][k] for i in range(54) for k in FUEL) == total_rods)
    if aggregate_counts is not None:
        for kind in FUEL:
            for d in range(5):
                model.add(sum(fuel_state[i, kind, d] for i in range(54))
                          == aggregate_counts.get((kind, d), 0))
    incoming = [[] for _ in range(54)]
    hull_source = []
    for source in range(54):
        adjacent = neighbours(source)
        tables = [[] for _ in adjacent]
        hull_table = []
        for kind in range(8):
            for d in range(5):
                per_rod = 2 * (INTERNAL[kind] + d) * (INTERNAL[kind] + d + 1)
                for m in range(1 << len(adjacent)):
                    shares, remainder = split(per_rod, m, len(adjacent))
                    for position in range(len(adjacent)):
                        tables[position].append((kind, d, m, RODS[kind] * shares[position]))
                    hull_table.append((kind, d, m, RODS[kind] * remainder))
        for position, target in enumerate(adjacent):
            flow = model.new_int_var(0, 336, f"heat_{source}_{target}")
            model.add_allowed_assignments([label[source], degree[source], mask[source], flow], tables[position])
            incoming[target].append(flow)
        source_heat = model.new_int_var(0, 336, f"hull_source_{source}")
        model.add_allowed_assignments([label[source], degree[source], mask[source], source_heat], hull_table)
        hull_source.append(source_heat)

    hull_draw = [0] * 54
    if allow_hull:
        hull = [model.new_int_var(0, 20_000, f"hull_{i}") for i in range(55)]
        for index in range(54):
            before_draw = model.new_int_var(0, 20_336, f"hull_before_draw_{index}")
            draw = model.new_int_var(0, 36, f"hull_draw_{index}")
            high = model.new_bool_var(f"hull_at_least_36_{index}")
            model.add(before_draw == hull[index] + hull_source[index])
            model.add(draw <= 36 * x[index][OVERCLOCKED])
            model.add(draw <= before_draw)
            model.add(high <= x[index][OVERCLOCKED])
            model.add(before_draw >= 36 * high)
            model.add(before_draw <= 35 + 20_301 * high + 20_301 * (1 - x[index][OVERCLOCKED]))
            model.add(draw >= before_draw - 20_336 * high - 20_336 * (1 - x[index][OVERCLOCKED]))
            model.add(draw >= 36 * high)
            model.add(hull[index + 1] == before_draw - draw)
            hull_draw[index] = draw
        model.add(hull[54] == hull[0])
        if exact_hull_source is not None:
            model.add(sum(hull_source) == exact_hull_source)
    else:
        model.add(sum(hull_source) == 0)

    for index in range(54):
        adjacent_c = sum(x[j][COMPONENT_VENT] for j in neighbours(index))
        model.add(sum(incoming[index]) + hull_draw[index]
                  <= 20 * x[index][OVERCLOCKED] + 4 * adjacent_c)

    power = model.new_int_var(0, 10_000, "total_power")
    heat = model.new_int_var(0, 30_000, "total_heat")
    model.add(power == sum(powers)); model.add(heat == sum(heats))
    if min_power is not None:
        model.add(power >= min_power)
    if exact_power is not None:
        model.add(power == exact_power)
    if max_heat is not None:
        model.add(heat <= max_heat)
    model.maximize(power)
    return model, x, power, heat


def cycle(layout: tuple[str, ...], horizon: int = 100_000) -> tuple[int, int] | None:
    simulator = ReactorSimulator(Layout(columns=9, slots=list(layout)))
    seen = {simulator.state_signature(include_fuel_damage=False): 0}
    for tick in range(1, horizon + 1):
        simulator.step(auto_refuel=True)
        if simulator.first_component_break_tick or simulator.first_critical_tick:
            return None
        signature = simulator.state_signature(include_fuel_damage=False)
        if signature in seen:
            return seen[signature], tick - seen[signature]
        seen[signature] = tick
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rods", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--min-power", type=int)
    parser.add_argument("--exact-power", type=int)
    parser.add_argument("--max-heat", type=int)
    parser.add_argument("--allow-hull", action="store_true")
    parser.add_argument("--exact-hull-source", type=int)
    args = parser.parse_args()
    model, x, power, heat = build(
        args.rods, args.min_power, args.exact_power, args.max_heat, args.allow_hull,
        exact_hull_source=args.exact_hull_source,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    started = perf_counter(); status = solver.solve(model)
    print(f"status={solver.status_name(status)} elapsed={perf_counter()-started:.3f}s "
          f"objective={solver.objective_value} bound={solver.best_objective_bound}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return
    codes = [max(range(8), key=lambda k: solver.value(x[i][k])) for i in range(54)]
    layout = tuple(IDS[k] for k in codes)
    print(f"power={solver.value(power)} heat={solver.value(heat)} cycle={cycle(layout)}")
    for row in range(ROWS):
        print(" ".join(SYMBOLS[k] for k in codes[row*9:(row+1)*9]))
    print(repr(layout))


if __name__ == "__main__": main()
