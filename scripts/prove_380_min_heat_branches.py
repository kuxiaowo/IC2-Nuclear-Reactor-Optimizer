"""Parallel certificate search for the P=380, Q=616 non-exchanger layer.

The aggregate equations (25 rods, 76 pulse-units, 616 heat) have only 49
fuel-type/active-degree count vectors.  Each vector is an independent exact
ordered-hull CP-SAT branch.  Running one single-threaded solver per branch
avoids the large-model symmetry that defeated the monolithic portfolio.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from time import perf_counter

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from exact_direct_cooling_cp_sat import (  # noqa: E402
    IDS,
    INTERNAL,
    RODS,
    build,
    cycle,
)


def aggregate_patterns() -> list[tuple[int, ...]]:
    states = [(kind, degree) for kind in (1, 2, 3) for degree in range(5)]
    result: list[tuple[int, ...]] = []

    def visit(position: int, rods: int, pulse_units: int, heat: int, counts: list[int]) -> None:
        if position == len(states):
            if rods == 25 and pulse_units == 76 and heat == 616:
                result.append(tuple(counts))
            return
        kind, degree = states[position]
        pulses = INTERNAL[kind] + degree
        rod_cost = RODS[kind]
        pulse_cost = rod_cost * pulses
        heat_cost = 2 * rod_cost * pulses * (pulses + 1)
        cap = min(
            (25 - rods) // rod_cost,
            (76 - pulse_units) // pulse_cost if pulse_cost else 0,
            (616 - heat) // heat_cost if heat_cost else 25,
        )
        for count in range(max(0, cap) + 1):
            counts.append(count)
            visit(
                position + 1,
                rods + count * rod_cost,
                pulse_units + count * pulse_cost,
                heat + count * heat_cost,
                counts,
            )
            counts.pop()

    visit(0, 0, 0, 0, [])
    return result


def solve_branch(payload: tuple[int, tuple[int, ...], float]) -> dict:
    branch, pattern, seconds = payload
    states = [(kind, degree) for kind in (1, 2, 3) for degree in range(5)]
    counts = {state: count for state, count in zip(states, pattern)}
    model, x, power, heat = build(25, None, 380, 616, True, counts)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    started = perf_counter()
    status = solver.solve(model)
    result = {
        "branch": branch,
        "status": solver.status_name(status),
        "elapsed": perf_counter() - started,
        "pattern": pattern,
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        codes = tuple(max(range(len(IDS)), key=lambda kind: solver.value(x[i][kind])) for i in range(54))
        layout = tuple(IDS[kind] for kind in codes)
        result.update(power=solver.value(power), heat=solver.value(heat), cycle=cycle(layout), layout=layout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds-per-branch", type=float, default=180)
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()
    patterns = aggregate_patterns()
    print(f"branches={len(patterns)} workers={args.workers} seconds_per_branch={args.seconds_per_branch}", flush=True)
    started = perf_counter()
    results = []
    payloads = [(index, pattern, args.seconds_per_branch) for index, pattern in enumerate(patterns)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve_branch, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"branch={result['branch']:02d} status={result['status']} "
                f"elapsed={result['elapsed']:.2f}s",
                flush=True,
            )
            if result["status"] in {"OPTIMAL", "FEASIBLE"}:
                print(f"witness={result}", flush=True)

    counts = Counter(result["status"] for result in results)
    print(f"summary={dict(counts)} wall={perf_counter() - started:.2f}s", flush=True)
    if counts.get("OPTIMAL", 0) + counts.get("FEASIBLE", 0) == 0 and counts.get("UNKNOWN", 0) == 0:
        print("CERTIFICATE: all 49 P=380,Q=616 non-exchanger aggregate branches are infeasible", flush=True)
    elif counts.get("UNKNOWN", 0):
        unknown = sorted(result["branch"] for result in results if result["status"] == "UNKNOWN")
        print(f"UNCLOSED_BRANCHES={unknown}", flush=True)


if __name__ == "__main__":
    main()
