"""Closed-form CP-SAT model for the minimum-heat 380 EU/t layer.

Convexity gives Q >= 616 at 25 rods and 76 pulse-units.  Equality forces 24
rods to have exactly three pulses and one single rod to have four pulses.
Hence only S(deg=2), one S(deg=3), D(deg=1), and Q(deg=0) can occur.  Encoding
those four states directly avoids the generic type/degree/heat tables.
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
SYMBOLS = (".", "S", "D", "Q", "R", "O", "C", "B")
IDS = ("empty", "uranium_single", "uranium_dual", "uranium_quad", "iridium_reflector",
       "overclocked_heat_vent", "component_heat_vent", "coolant_60k")
STATE_RODS = (1, 1, 2, 4)  # S2, unique S3, D1, Q0.
STATE_PER_ROD_HEAT = (24, 40, 24, 24)


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


def split(heat: int, mask: int, size: int) -> tuple[list[int], int]:
    receivers = [position for position in range(size) if mask & (1 << position)]
    shares = [0] * size
    remaining, left = heat, len(receivers)
    for position in receivers:
        shares[position] = remaining // left
        remaining -= shares[position]
        left -= 1
    return shares, remaining


def build(exact_hull_source: int | None, cross_center: int | None = None,
          special_single: int | None = None,
          special_inactive_neighbor: int | None = None) -> tuple:
    model = cp_model.CpModel()
    x = [[model.new_bool_var(f"x_{i}_{kind}") for kind in range(8)] for i in range(54)]
    active = [model.new_bool_var(f"active_{i}") for i in range(54)]
    fuel = [model.new_bool_var(f"fuel_{i}") for i in range(54)]
    acceptor = [model.new_bool_var(f"acceptor_{i}") for i in range(54)]
    degree = [model.new_int_var(0, 4, f"degree_{i}") for i in range(54)]
    state = [[model.new_bool_var(f"state_{i}_{s}") for s in range(4)] for i in range(54)]
    patterns: list[list[cp_model.IntVar]] = []

    for index in range(54):
        model.add_exactly_one(x[index])
        model.add(fuel[index] == x[index][SINGLE] + x[index][DUAL] + x[index][QUAD])
        model.add(active[index] == fuel[index] + x[index][REFLECTOR])
        model.add(acceptor[index] == x[index][OVERCLOCKED] + x[index][BUFFER])
        adjacent = neighbours(index)
        model.add(degree[index] == sum(active[j] for j in adjacent))
        model.add(x[index][SINGLE] == state[index][0] + state[index][1])
        model.add(x[index][DUAL] == state[index][2])
        model.add(x[index][QUAD] == state[index][3])
        for s, required_degree in enumerate((2, 3, 1, 0)):
            model.add(degree[index] == required_degree).only_enforce_if(state[index][s])
        # A reflector not adjacent to fuel changes no power term and is
        # dominated by a cooling slot, so it can be removed without loss.
        model.add(x[index][REFLECTOR] <= sum(fuel[j] for j in adjacent))

        choices = [model.new_bool_var(f"pattern_{index}_{mask}") for mask in range(1 << len(adjacent))]
        model.add_exactly_one(choices)
        for position, target in enumerate(adjacent):
            model.add(sum(choices[mask] for mask in range(len(choices)) if mask & (1 << position))
                      == acceptor[target])
        patterns.append(choices)

    model.add(sum(state[i][1] for i in range(54)) == 1)
    if special_single is not None:
        model.add(state[special_single][1] == 1)
        if special_inactive_neighbor is not None:
            adjacent = neighbours(special_single)
            if len(adjacent) != 4 or special_inactive_neighbor not in adjacent:
                raise ValueError("special_inactive_neighbor must be one of four interior neighbours")
            for target in adjacent:
                model.add(active[target] == int(target != special_inactive_neighbor))
    model.add(sum(STATE_RODS[s] * state[i][s] for i in range(54) for s in range(4)) == 25)
    if cross_center is not None:
        if len(neighbours(cross_center)) != 4:
            raise ValueError("cross_center must be an interior grid cell")
        model.add(x[cross_center][OVERCLOCKED] == 1)
        for target in neighbours(cross_center):
            model.add(x[target][COMPONENT_VENT] == 1)

    incoming: list[list] = [[] for _ in range(54)]
    hull_source_terms = []
    for source in range(54):
        adjacent = neighbours(source)
        for s in range(4):
            rods = STATE_RODS[s]
            per_rod_heat = STATE_PER_ROD_HEAT[s]
            for mask, pattern in enumerate(patterns[source]):
                joint = model.new_bool_var(f"joint_{source}_{s}_{mask}")
                model.add(joint <= state[source][s])
                model.add(joint <= pattern)
                model.add(joint >= state[source][s] + pattern - 1)
                shares, remainder = split(per_rod_heat, mask, len(adjacent))
                for position, target in enumerate(adjacent):
                    if shares[position]:
                        incoming[target].append(rods * shares[position] * joint)
                if remainder:
                    hull_source_terms.append(rods * remainder * joint)

    hull_source = model.new_int_var(0, 616, "total_hull_source")
    model.add(hull_source == sum(hull_source_terms))
    model.add(hull_source >= 40)
    if exact_hull_source is not None:
        model.add(hull_source == exact_hull_source)

    hull = [model.new_int_var(0, 8_499, f"hull_{i}") for i in range(55)]
    hull_draw = []
    # Reconstruct each cell's source term separately for the row-major prefix.
    source_terms_by_cell: list[list] = [[] for _ in range(54)]
    for source in range(54):
        adjacent = neighbours(source)
        for s in range(4):
            rods = STATE_RODS[s]
            per_rod_heat = STATE_PER_ROD_HEAT[s]
            for mask, pattern in enumerate(patterns[source]):
                _shares, remainder = split(per_rod_heat, mask, len(adjacent))
                if not remainder:
                    continue
                joint = model.new_bool_var(f"hull_joint_{source}_{s}_{mask}")
                model.add(joint <= state[source][s])
                model.add(joint <= pattern)
                model.add(joint >= state[source][s] + pattern - 1)
                source_terms_by_cell[source].append(rods * remainder * joint)

    for index in range(54):
        cell_source = model.new_int_var(0, 160, f"cell_hull_source_{index}")
        model.add(cell_source == sum(source_terms_by_cell[index]))
        before = model.new_int_var(0, 8_659, f"hull_before_{index}")
        draw = model.new_int_var(0, 36, f"hull_draw_{index}")
        high = model.new_bool_var(f"hull_high_{index}")
        model.add(before == hull[index] + cell_source)
        model.add(draw <= 36 * x[index][OVERCLOCKED])
        model.add(draw <= before)
        model.add(high <= x[index][OVERCLOCKED])
        model.add(before >= 36 * high)
        model.add(before <= 35 + 8_624 * high + 8_624 * (1 - x[index][OVERCLOCKED]))
        model.add(draw >= before - 8_659 * high - 8_659 * (1 - x[index][OVERCLOCKED]))
        model.add(draw >= 36 * high)
        model.add(hull[index + 1] == before - draw)
        hull_draw.append(draw)
    model.add(hull[54] == hull[0])
    if cross_center is not None:
        model.add(hull_draw[cross_center] == 36)

    for index in range(54):
        adjacent_c = sum(x[j][COMPONENT_VENT] for j in neighbours(index))
        model.add(sum(incoming[index]) + hull_draw[index]
                  <= 20 * x[index][OVERCLOCKED] + 4 * adjacent_c)
    return model, x, hull_source


def exact_cycle(layout: tuple[str, ...], horizon: int = 100_000) -> tuple[int, int] | None:
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
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--exact-hull-source", type=int)
    parser.add_argument("--cross-center", type=int)
    parser.add_argument("--special-single", type=int)
    parser.add_argument("--special-inactive-neighbor", type=int)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    model, x, hull_source = build(
        args.exact_hull_source, args.cross_center, args.special_single,
        args.special_inactive_neighbor,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    solver.parameters.log_search_progress = args.log
    started = perf_counter(); status = solver.solve(model)
    print(f"status={solver.status_name(status)} elapsed={perf_counter()-started:.3f}s")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    codes = [max(range(8), key=lambda kind: solver.value(x[i][kind])) for i in range(54)]
    layout = tuple(IDS[kind] for kind in codes)
    print(f"power=380 heat=616 hull_source={solver.value(hull_source)} cycle={exact_cycle(layout)}")
    for row in range(ROWS):
        print(" ".join(SYMBOLS[k] for k in codes[row * 9:(row + 1) * 9]))
    print(repr(layout))


if __name__ == "__main__":
    main()
