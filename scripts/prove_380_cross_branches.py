"""Parallel proof branches for the mandatory 36-heat O/C cross at P=380,Q=616."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prove_380_q616_structural import IDS, build, exact_cycle, neighbours  # noqa: E402


def interior_cells() -> list[int]:
    return [row * 9 + column for row in range(1, 5) for column in range(1, 8)]


def solve_branch(payload: tuple[int, int | None, int | None, float]) -> dict:
    center, special, inactive, seconds = payload
    model, x, hull_source = build(40, center, special, inactive)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    started = perf_counter(); status = solver.solve(model)
    result = {"center": center, "special": special, "inactive": inactive,
              "status": solver.status_name(status),
              "elapsed": perf_counter() - started}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        codes = tuple(max(range(8), key=lambda kind: solver.value(x[i][kind])) for i in range(54))
        layout = tuple(IDS[kind] for kind in codes)
        result.update(hull_source=solver.value(hull_source), cycle=exact_cycle(layout), layout=layout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds-per-branch", type=float, default=180)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--split-special", action="store_true")
    parser.add_argument("--split-special-star", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--results", default="artifacts/proof_380_cross_results.json")
    args = parser.parse_args()
    centers = interior_cells()
    retained = []
    if args.resume:
        previous = json.loads((ROOT / args.resume).read_text(encoding="utf-8"))
        retained = [item for item in previous if item["status"] != "UNKNOWN"]
        pairs = [(item["center"], item.get("special")) for item in previous if item["status"] == "UNKNOWN"]
    else:
        pairs = [
            (center, special)
            for center in centers
            for special in (range(54) if args.split_special else (None,))
        ]
    payloads = []
    for center, special in pairs:
        inactive_choices: tuple[int | None, ...] = (None,)
        if args.split_special_star and special is not None and len(neighbours(special)) == 4:
            inactive_choices = neighbours(special)
        for inactive in inactive_choices:
            payloads.append((center, special, inactive, args.seconds_per_branch))
    print(f"branches={len(payloads)} workers={args.workers}", flush=True)
    started = perf_counter(); results = list(retained)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve_branch, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result(); results.append(result)
            if result["status"] in {"OPTIMAL", "FEASIBLE"}:
                print(f"witness={result}", flush=True)
            elif completed % 100 == 0:
                partial = Counter(item["status"] for item in results if item not in retained)
                print(f"completed={completed}/{len(payloads)} partial={dict(partial)}", flush=True)
    statuses = Counter(result["status"] for result in results)
    output = ROOT / args.results
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sorted(results, key=lambda item: (
                                     item["center"], item.get("special") if item.get("special") is not None else -1,
                                     item.get("inactive") if item.get("inactive") is not None else -1)),
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={dict(statuses)} wall={perf_counter()-started:.2f}s", flush=True)
    print(f"results={output}", flush=True)
    if statuses.get("UNKNOWN", 0) == 0 and statuses.get("OPTIMAL", 0) + statuses.get("FEASIBLE", 0) == 0:
        print("CERTIFICATE: every mandatory-cross center is infeasible", flush=True)
    elif statuses.get("UNKNOWN", 0):
        unknown = sorted((r["center"], r.get("special"), r.get("inactive"))
                         for r in results if r["status"] == "UNKNOWN")
        print(f"UNCLOSED_COUNT={len(unknown)} UNCLOSED_SAMPLE={unknown[:100]}", flush=True)


if __name__ == "__main__":
    main()
