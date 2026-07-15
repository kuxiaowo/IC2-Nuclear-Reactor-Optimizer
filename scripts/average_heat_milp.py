"""Inventory/layout-aware average-heat MILP for 25-rod Mark-I upper bounds.

Every safe periodic run induces a time-averaged conserved heat flow.  This
model keeps exact fuel pulse topology and exact initial heat distribution to
the statically heat-accepting neighbours, while relaxing exchanger ratios,
row-major order, finite capacities and forced hull draws.  Therefore its
objective is a rigorous upper bound; a feasible solution is not a reactor
certificate and must be checked by the exact simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

from ortools.linear_solver import pywraplp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.engine import ReactorSimulator  # noqa: E402
from ic2_reactor.models import Layout  # noqa: E402


ROWS = 6


@dataclass(frozen=True)
class Kind:
    symbol: str
    component_id: str
    rods: int = 0
    internal: int = 0
    power_active: bool = False
    accepts_heat: bool = False
    self_vent: int = 0
    side_exchange: int = 0
    hull_in: int = 0
    hull_out: int = 0


KINDS = (
    Kind(".", "empty"),
    Kind("S", "uranium_single", 1, 1, True),
    Kind("D", "uranium_dual", 2, 2, True),
    Kind("Q", "uranium_quad", 4, 3, True),
    Kind("R", "iridium_reflector", power_active=True),
    Kind("O", "overclocked_heat_vent", accepts_heat=True, self_vent=20, hull_in=36),
    Kind("C", "component_heat_vent"),
    Kind("A", "advanced_heat_vent", accepts_heat=True, self_vent=12),
    Kind("V", "reactor_heat_vent", accepts_heat=True, self_vent=5, hull_in=5),
    Kind("X", "component_heat_exchanger", accepts_heat=True, side_exchange=36),
    Kind("H", "reactor_heat_exchanger", accepts_heat=True, hull_in=72, hull_out=72),
    # The low-percentage branch can use exchange_side/2=12, larger than the
    # nominal hull value 8, so 12 is the sound relaxed hull capacity.
    Kind("a", "advanced_heat_exchanger", accepts_heat=True, side_exchange=24, hull_in=12, hull_out=12),
    # Generic finite storage.  It covers coolant cells and a periodically
    # cooled condensator; plating/empty is covered by the non-accepting kind.
    Kind("B", "coolant_60k", accepts_heat=True),
)
FUEL_CODES = (1, 2, 3)
REFLECTOR = 4
OVERCLOCKED = 5
COMPONENT_VENT = 6

# Exact-simulator-verified 375 EU/t, period-one construction.  Besides being a
# useful lower bound, fixing this layout is a quick consistency check for the
# average-flow relaxation.
DIRECT_375 = (
    "COODROCCO"
    "O.COOQOOD"
    "DOCOCOCOR"
    "DOOQO.ODS"
    "OCOOCOOCO"
    ".ODRODDO."
)


def neighbours(index: int, columns: int) -> tuple[int, ...]:
    row, column = divmod(index, columns)
    result = []
    # This is the official left, right, up, down order used for fuel heat.
    if column:
        result.append(index - 1)
    if column + 1 < columns:
        result.append(index + 1)
    if row:
        result.append(index - columns)
    if row + 1 < ROWS:
        result.append(index + columns)
    return tuple(result)


def undirected_edges(columns: int) -> tuple[tuple[int, int], ...]:
    result = []
    for index in range(ROWS * columns):
        row, column = divmod(index, columns)
        if column + 1 < columns:
            result.append((index, index + 1))
        if row + 1 < ROWS:
            result.append((index, index + columns))
    return tuple(result)


def distribute_per_rod(heat: int, mask: int, degree: int) -> tuple[list[int], int]:
    """Exact no-overflow fuel distribution in official neighbour order."""
    receivers = [position for position in range(degree) if mask & (1 << position)]
    if not receivers:
        return [0] * degree, heat
    result = [0] * degree
    remaining = heat
    left = len(receivers)
    for position in receivers:
        amount = remaining // left
        remaining -= amount
        left -= 1
        result[position] = amount
    return result, remaining


def build_model(columns: int, total_rods: int, component_cap: int | None) -> tuple:
    slots = ROWS * columns
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP backend is unavailable")
    infinity = solver.infinity()
    x = [[solver.BoolVar(f"x_{i}_{t}") for t in range(len(KINDS))] for i in range(slots)]
    active = [solver.BoolVar(f"active_{i}") for i in range(slots)]
    acceptor = [solver.BoolVar(f"acceptor_{i}") for i in range(slots)]
    power_degree = [solver.IntVar(0, 4, f"power_degree_{i}") for i in range(slots)]
    z: dict[tuple[int, int, int], pywraplp.Variable] = {}
    patterns: dict[tuple[int, int], pywraplp.Variable] = {}
    joint: dict[tuple[int, int, int, int], pywraplp.Variable] = {}
    power_terms = []
    heat_terms = []

    for index in range(slots):
        solver.Add(sum(x[index]) == 1)
        solver.Add(active[index] == sum(x[index][code] for code in (*FUEL_CODES, REFLECTOR)))
        solver.Add(acceptor[index] == sum(x[index][code] for code, kind in enumerate(KINDS) if kind.accepts_heat))
        adjacent = neighbours(index, columns)
        solver.Add(power_degree[index] == sum(active[other] for other in adjacent))
        for code in FUEL_CODES:
            choices = []
            for degree in range(5):
                variable = solver.BoolVar(f"z_{index}_{code}_{degree}")
                z[index, code, degree] = variable
                choices.append(variable)
                solver.Add(power_degree[index] >= degree - 4 * (1 - variable))
                solver.Add(power_degree[index] <= degree + 4 * (1 - variable))
                pulses = KINDS[code].internal + degree
                power_terms.append(5 * KINDS[code].rods * pulses * variable)
                heat_terms.append(2 * KINDS[code].rods * pulses * (pulses + 1) * variable)
            solver.Add(sum(choices) == x[index][code])

        pattern_choices = []
        for mask in range(1 << len(adjacent)):
            variable = solver.BoolVar(f"pattern_{index}_{mask}")
            patterns[index, mask] = variable
            pattern_choices.append(variable)
        solver.Add(sum(pattern_choices) == 1)
        for local_position, other in enumerate(adjacent):
            solver.Add(
                sum(variable for (cell, mask), variable in patterns.items() if cell == index and mask & (1 << local_position))
                == acceptor[other]
            )

    solver.Add(sum(KINDS[code].rods * x[index][code] for index in range(slots) for code in FUEL_CODES) == total_rods)
    if component_cap is not None:
        # Each relaxed kind represents one strongest real component family.
        for code in range(OVERCLOCKED, len(KINDS)):
            solver.Add(sum(x[index][code] for index in range(slots)) <= component_cap)

    injection = [solver.NumVar(0, infinity, f"injection_{i}") for i in range(slots)]
    hull_source_terms = []
    injection_terms: list[list] = [[] for _ in range(slots)]
    for index in range(slots):
        adjacent = neighbours(index, columns)
        for code in FUEL_CODES:
            for degree in range(5):
                for mask in range(1 << len(adjacent)):
                    variable = solver.BoolVar(f"joint_{index}_{code}_{degree}_{mask}")
                    joint[index, code, degree, mask] = variable
                    solver.Add(variable <= z[index, code, degree])
                    solver.Add(variable <= patterns[index, mask])
                    solver.Add(variable >= z[index, code, degree] + patterns[index, mask] - 1)
                    pulses = KINDS[code].internal + degree
                    per_rod_heat = 2 * pulses * (pulses + 1)
                    shares, hull_remainder = distribute_per_rod(per_rod_heat, mask, len(adjacent))
                    rods = KINDS[code].rods
                    for local_position, target in enumerate(adjacent):
                        if shares[local_position]:
                            injection_terms[target].append(rods * shares[local_position] * variable)
                    if hull_remainder:
                        hull_source_terms.append(rods * hull_remainder * variable)

    for index in range(slots):
        solver.Add(injection[index] == sum(injection_terms[index]))

    side_in: list[list] = [[] for _ in range(slots)]
    side_out: list[list] = [[] for _ in range(slots)]
    for first, second in undirected_edges(columns):
        forward = solver.NumVar(0, infinity, f"side_{first}_{second}")
        backward = solver.NumVar(0, infinity, f"side_{second}_{first}")
        capacity = sum(
            kind.side_exchange * (x[first][code] + x[second][code])
            for code, kind in enumerate(KINDS)
            if kind.side_exchange
        )
        solver.Add(forward + backward <= capacity)
        # Heat can only reside at accepting endpoints.
        solver.Add(forward <= 1_000 * acceptor[first])
        solver.Add(forward <= 1_000 * acceptor[second])
        solver.Add(backward <= 1_000 * acceptor[first])
        solver.Add(backward <= 1_000 * acceptor[second])
        side_out[first].append(forward)
        side_in[second].append(forward)
        side_out[second].append(backward)
        side_in[first].append(backward)

    hull_in = [solver.NumVar(0, infinity, f"hull_to_{i}") for i in range(slots)]
    hull_out = [solver.NumVar(0, infinity, f"{i}_to_hull") for i in range(slots)]
    self_sink = [solver.NumVar(0, infinity, f"self_sink_{i}") for i in range(slots)]
    component_sink: list[list] = [[] for _ in range(slots)]
    for index in range(slots):
        solver.Add(hull_in[index] <= sum(kind.hull_in * x[index][code] for code, kind in enumerate(KINDS)))
        solver.Add(hull_out[index] <= sum(kind.hull_out * x[index][code] for code, kind in enumerate(KINDS)))
        solver.Add(self_sink[index] <= sum(kind.self_vent * x[index][code] for code, kind in enumerate(KINDS)))

    for vent in range(slots):
        for target in neighbours(vent, columns):
            sink = solver.NumVar(0, 4, f"component_sink_{vent}_{target}")
            solver.Add(sink <= 4 * x[vent][COMPONENT_VENT])
            solver.Add(sink <= 4 * acceptor[target])
            component_sink[target].append(sink)

    for index in range(slots):
        solver.Add(
            injection[index] + hull_in[index] + sum(side_in[index])
            == self_sink[index] + sum(component_sink[index]) + hull_out[index] + sum(side_out[index])
        )
    solver.Add(sum(hull_source_terms) + sum(hull_out) == sum(hull_in))

    power = solver.Sum(power_terms)
    generated_heat = solver.Sum(heat_terms)
    solver.Maximize(power)
    return solver, x, power, generated_heat


def exact_cycle(layout: tuple[str, ...], horizon: int = 20_000) -> tuple[int, int] | None:
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
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--component-cap", type=int)
    parser.add_argument("--fix-direct-375", action="store_true")
    parser.add_argument("--min-power", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    solver, x, power, generated_heat = build_model(9, 25, args.component_cap)
    symbol_to_code = {kind.symbol: code for code, kind in enumerate(KINDS)}
    if args.fix_direct_375:
        for index, symbol in enumerate(DIRECT_375):
            solver.Add(x[index][symbol_to_code[symbol]] == 1)
    else:
        # A partial MIP start is enough for SCIP to inherit the certified
        # 375-EU/t lower bound while all flow variables remain free.
        hint_vars = [x[index][code] for index in range(54) for code in range(len(KINDS))]
        hint_values = [
            1.0 if code == symbol_to_code[DIRECT_375[index]] else 0.0
            for index in range(54)
            for code in range(len(KINDS))
        ]
        solver.SetHint(hint_vars, hint_values)
    if args.min_power is not None:
        solver.Add(power >= args.min_power)
    solver.SetTimeLimit(args.seconds * 1000)
    solver.SetNumThreads(args.workers)
    if args.verbose:
        solver.EnableOutput()
    started = perf_counter()
    status = solver.Solve()
    elapsed = perf_counter() - started
    print(
        f"status={status} elapsed={elapsed:.3f}s objective={solver.Objective().Value():.3f} "
        f"bound={solver.Objective().BestBound():.3f} nodes={solver.nodes()}"
    )
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return
    codes = tuple(max(range(len(KINDS)), key=lambda code: x[index][code].solution_value()) for index in range(54))
    layout = tuple(KINDS[code].component_id for code in codes)
    print(f"power={power.solution_value():.3f} heat={generated_heat.solution_value():.3f} cycle={exact_cycle(layout)}")
    for row in range(ROWS):
        print(" ".join(KINDS[code].symbol for code in codes[row * 9 : (row + 1) * 9]))
    print(repr(layout))


if __name__ == "__main__":
    main()
