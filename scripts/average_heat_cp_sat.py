"""CP-SAT necessary average-flow model for globally safe periodic reactors.

Fuel topology and IC2's ordered heat split are exact.  Heat transport is a
relaxed capacitated network: exchanger percentages, row-major execution and
finite component temperatures are omitted.  Thus every infinite safe cycle
induces a feasible average flow, while a model solution still needs exact
simulation.  Integral supplies/capacities make the fixed-layout flow network
integral, so integer flow variables do not weaken the upper-bound argument.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

from ortools.sat.python import cp_model


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from ic2_reactor.engine import ReactorSimulator  # noqa: E402
from ic2_reactor.models import Layout  # noqa: E402

ROWS, COLUMNS = 6, 9


@dataclass(frozen=True)
class Kind:
    symbol: str
    component_id: str
    rods: int = 0
    internal: int = 0
    active: bool = False
    accepts: bool = False
    self_vent: int = 0
    side_exchange: int = 0
    hull_in: int = 0
    hull_out: int = 0


KINDS = (
    Kind(".", "empty"),
    Kind("S", "uranium_single", 1, 1, True),
    Kind("D", "uranium_dual", 2, 2, True),
    Kind("Q", "uranium_quad", 4, 3, True),
    Kind("R", "iridium_reflector", active=True),
    Kind("O", "overclocked_heat_vent", accepts=True, self_vent=20, hull_in=36),
    Kind("C", "component_heat_vent"),
    Kind("A", "advanced_heat_vent", accepts=True, self_vent=12),
    Kind("V", "reactor_heat_vent", accepts=True, self_vent=5, hull_in=5),
    Kind("X", "component_heat_exchanger", accepts=True, side_exchange=36),
    Kind("H", "reactor_heat_exchanger", accepts=True, hull_in=72, hull_out=72),
    # Optimistically allow the advanced exchanger 12 in either hull direction
    # (the real nominal hull limit is 8; a low-ratio branch can use 12).
    Kind("a", "advanced_heat_exchanger", accepts=True, side_exchange=24, hull_in=12, hull_out=12),
    Kind("B", "coolant_60k", accepts=True),
)
FUEL = (1, 2, 3)
REFLECTOR, COMPONENT_VENT = 4, 6
MAX_FLOW = 2_000
DIRECT_375 = "COODROCCOO.COOQOODDOCOCOCORDOOQO.ODSOCOOCOOCO.ODRODDO."


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


def split_heat(per_rod_heat: int, mask: int, degree: int) -> tuple[list[int], int]:
    receivers = [position for position in range(degree) if mask & (1 << position)]
    shares = [0] * degree
    remaining = per_rod_heat
    receivers_left = len(receivers)
    for position in receivers:
        amount = remaining // receivers_left
        remaining -= amount
        receivers_left -= 1
        shares[position] = amount
    return shares, remaining


def build_model(total_rods: int, min_power: int | None, exact_power: int | None,
                no_hull_source: bool) -> tuple:
    model = cp_model.CpModel()
    slots, kinds = ROWS * COLUMNS, len(KINDS)
    x = [[model.new_bool_var(f"x_{i}_{k}") for k in range(kinds)] for i in range(slots)]
    label = [model.new_int_var(0, kinds - 1, f"label_{i}") for i in range(slots)]
    active = [model.new_bool_var(f"active_{i}") for i in range(slots)]
    acceptor = [model.new_bool_var(f"acceptor_{i}") for i in range(slots)]
    degree = [model.new_int_var(0, 4, f"degree_{i}") for i in range(slots)]
    mask = [model.new_int_var(0, 15, f"mask_{i}") for i in range(slots)]
    cell_power, cell_heat = [], []

    for index in range(slots):
        model.add_exactly_one(x[index])
        model.add(label[index] == sum(k * x[index][k] for k in range(kinds)))
        model.add(active[index] == sum(x[index][k] for k in (*FUEL, REFLECTOR)))
        model.add(acceptor[index] == sum(x[index][k] for k, kind in enumerate(KINDS) if kind.accepts))
        adjacent = neighbours(index)
        model.add(degree[index] == sum(active[target] for target in adjacent))
        model.add(mask[index] == sum((1 << position) * acceptor[target] for position, target in enumerate(adjacent)))
        p = model.new_int_var(0, 140, f"power_{index}")
        h = model.new_int_var(0, 336, f"heat_{index}")
        table = []
        for kind, spec in enumerate(KINDS):
            for d in range(5):
                pulses = spec.internal + d
                table.append((kind, d, 5 * spec.rods * pulses, 2 * spec.rods * pulses * (pulses + 1)))
        model.add_allowed_assignments([label[index], degree[index], p, h], table)
        cell_power.append(p)
        cell_heat.append(h)

    model.add(sum(KINDS[k].rods * x[i][k] for i in range(slots) for k in FUEL) == total_rods)

    injection_terms: list[list] = [[] for _ in range(slots)]
    hull_source = []
    for source in range(slots):
        adjacent = neighbours(source)
        contribution_tables = [[] for _ in adjacent]
        hull_table = []
        for kind, spec in enumerate(KINDS):
            for d in range(5):
                pulses = spec.internal + d
                per_rod_heat = 2 * pulses * (pulses + 1)
                for m in range(1 << len(adjacent)):
                    shares, remainder = split_heat(per_rod_heat, m, len(adjacent))
                    for position in range(len(adjacent)):
                        contribution_tables[position].append((kind, d, m, spec.rods * shares[position]))
                    hull_table.append((kind, d, m, spec.rods * remainder))
        for position, target in enumerate(adjacent):
            contribution = model.new_int_var(0, 336, f"inject_{source}_{target}")
            model.add_allowed_assignments(
                [label[source], degree[source], mask[source], contribution],
                contribution_tables[position],
            )
            injection_terms[target].append(contribution)
        source_to_hull = model.new_int_var(0, 336, f"fuel_to_hull_{source}")
        model.add_allowed_assignments([label[source], degree[source], mask[source], source_to_hull], hull_table)
        hull_source.append(source_to_hull)

    side_in: list[list] = [[] for _ in range(slots)]
    side_out: list[list] = [[] for _ in range(slots)]
    for first, second in edges():
        forward = model.new_int_var(0, MAX_FLOW, f"side_{first}_{second}")
        backward = model.new_int_var(0, MAX_FLOW, f"side_{second}_{first}")
        capacity = sum(
            spec.side_exchange * (x[first][kind] + x[second][kind])
            for kind, spec in enumerate(KINDS)
            if spec.side_exchange
        )
        model.add(forward + backward <= capacity)
        model.add(forward <= MAX_FLOW * acceptor[first])
        model.add(forward <= MAX_FLOW * acceptor[second])
        model.add(backward <= MAX_FLOW * acceptor[first])
        model.add(backward <= MAX_FLOW * acceptor[second])
        side_out[first].append(forward)
        side_in[second].append(forward)
        side_out[second].append(backward)
        side_in[first].append(backward)

    hull_in, hull_out, self_sink = [], [], []
    for index in range(slots):
        incoming = model.new_int_var(0, 72, f"hull_to_{index}")
        outgoing = model.new_int_var(0, 72, f"{index}_to_hull")
        own_sink = model.new_int_var(0, 20, f"self_sink_{index}")
        model.add(incoming <= sum(spec.hull_in * x[index][kind] for kind, spec in enumerate(KINDS)))
        model.add(outgoing <= sum(spec.hull_out * x[index][kind] for kind, spec in enumerate(KINDS)))
        model.add(own_sink <= sum(spec.self_vent * x[index][kind] for kind, spec in enumerate(KINDS)))
        hull_in.append(incoming)
        hull_out.append(outgoing)
        self_sink.append(own_sink)

    component_sink: list[list] = [[] for _ in range(slots)]
    for vent in range(slots):
        for target in neighbours(vent):
            sink = model.new_int_var(0, 4, f"component_sink_{vent}_{target}")
            model.add(sink <= 4 * x[vent][COMPONENT_VENT])
            model.add(sink <= 4 * acceptor[target])
            component_sink[target].append(sink)

    for index in range(slots):
        model.add(
            sum(injection_terms[index]) + hull_in[index] + sum(side_in[index])
            == self_sink[index] + sum(component_sink[index]) + hull_out[index] + sum(side_out[index])
        )
    model.add(sum(hull_source) + sum(hull_out) == sum(hull_in))
    if no_hull_source:
        # In this branch "no hull" means no source and no use of the hull as
        # a free transshipment node.  The latter otherwise permits artificial
        # exchanger cycles even with zero fuel heat sent to the hull.
        model.add(sum(hull_in) == 0)

    power = model.new_int_var(0, 10_000, "total_power")
    heat = model.new_int_var(0, 30_000, "total_heat")
    model.add(power == sum(cell_power))
    model.add(heat == sum(cell_heat))
    if min_power is not None:
        model.add(power >= min_power)
    if exact_power is not None:
        model.add(power == exact_power)
    model.maximize(power)

    symbol_to_kind = {kind.symbol: code for code, kind in enumerate(KINDS)}
    for index, symbol in enumerate(DIRECT_375):
        for kind in range(kinds):
            model.add_hint(x[index][kind], int(kind == symbol_to_kind[symbol]))
    return model, x, power, heat


def exact_cycle(layout: tuple[str, ...], horizon: int = 100_000) -> tuple[int, int] | None:
    simulator = ReactorSimulator(Layout(columns=COLUMNS, slots=list(layout)))
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
    parser.add_argument("--rods", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--min-power", type=int)
    parser.add_argument("--exact-power", type=int)
    parser.add_argument("--no-hull-source", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    model, x, power, heat = build_model(
        args.rods, args.min_power, args.exact_power, args.no_hull_source
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    solver.parameters.log_search_progress = args.log
    started = perf_counter()
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} elapsed={perf_counter() - started:.3f}s "
        f"objective={solver.objective_value} bound={solver.best_objective_bound}"
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    codes = [max(range(len(KINDS)), key=lambda k: solver.value(x[i][k])) for i in range(54)]
    layout = tuple(KINDS[k].component_id for k in codes)
    print(f"power={solver.value(power)} heat={solver.value(heat)} cycle={exact_cycle(layout)}")
    for row in range(ROWS):
        print(" ".join(KINDS[k].symbol for k in codes[row * COLUMNS : (row + 1) * COLUMNS]))
    print(repr(layout))


if __name__ == "__main__":
    main()
