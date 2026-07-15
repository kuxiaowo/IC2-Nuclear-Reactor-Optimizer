"""Run the certified full-label Logic-Benders optimiser under a hard budget."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.ic2_global_proof import solve_ic2_global  # noqa: E402
from ic2_reactor.mathematical_model import ic2_mark_i_problem  # noqa: E402


MAX_WALL_TIME_SECONDS = 6 * 60 * 60


def _decode_layout(value: object, rows: int, columns: int) -> tuple[str, ...]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if len(value) != rows * columns:
            raise ValueError("component-id layout has the wrong number of slots")
        return tuple(value)
    if not isinstance(value, dict):
        raise ValueError("known layout must be a component-id list or symbolic rows")
    raw_rows = value.get("rows")
    symbols = value.get("symbols")
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != rows
        or not all(isinstance(row, str) and len(row) == columns for row in raw_rows)
        or not isinstance(symbols, dict)
    ):
        raise ValueError("invalid symbolic known layout")
    try:
        return tuple(symbols[symbol] for symbol in "".join(raw_rows))
    except KeyError as error:
        raise ValueError(f"missing symbol mapping: {error.args[0]}") from error


def _proof_json(proof) -> dict | None:
    if proof is None:
        return None
    return {
        "kind": proof.kind,
        "statement": proof.statement,
        "data": dict(proof.data),
    }


def _report_json(result) -> dict:
    report = result.report
    best = report.best_witness
    work = result.work_envelope
    return {
        "lower_bound": report.lower_bound,
        "upper_bound": report.upper_bound,
        "proven_global": report.proven_global,
        "stop_reason": report.stop_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "processed_nodes": report.processed_nodes,
        "initial_upper_bound": result.initial_upper_bound,
        "conditional_aggregate_pattern_counts": dict(
            result.conditional_aggregate_pattern_counts
        ),
        "input_work_envelope": {
            "vertices": work.vertices,
            "edges": work.edges,
            "maximum_degree": work.maximum_degree,
            "open_power_tiers": work.open_power_tiers,
            "aggregate_signature_types": work.aggregate_signature_types,
            "aggregate_count_vector_bound_per_tier": (
                work.aggregate_count_vector_bound_per_tier
            ),
            "rod_feasible_power_skeletons": work.rod_feasible_power_skeletons,
            "rod_feasible_full_layouts": work.rod_feasible_full_layouts,
            "frontier_width": work.frontier_width,
            "frontier_state_bound": work.frontier_state_bound,
            "frontier_transition_bound": work.frontier_transition_bound,
            "note": (
                "pre-search upper bounds only; a six-hour proof requires an "
                "actual remaining-unit ledger after all family cuts"
            ),
        },
        "analytical_proof": (
            None
            if result.analytical_proof is None
            else {
                "power_upper_bound": result.analytical_proof.power_upper_bound,
                "excluded_power_levels": result.analytical_proof.excluded_power_levels,
                "assumptions": result.analytical_proof.assumptions,
                "checks": dict(result.analytical_proof.checks),
                "derivation": result.analytical_proof.derivation,
            }
        ),
        "best_witness": (
            None
            if best is None
            else {
                "objective_value": best.objective_value,
                "layout": best.payload.layout,
                "cycle": {
                    "outcome": best.payload.cycle.outcome,
                    "transient_length": best.payload.cycle.transient_length,
                    "period_length": best.payload.cycle.period_length,
                    "checked_steps": best.payload.cycle.checked_steps,
                },
                "proof": _proof_json(best.proof),
            }
        ),
        "open_nodes": [
            {
                "node_id": node.node_id,
                "strict_upper_bound": node.strict_upper_bound,
                "iteration": node.payload.iteration,
                "cut_count": len(node.payload.cuts),
                "fixed_unknown_candidate": node.payload.fixed_candidate is not None,
            }
            for node in report.open_nodes
        ],
        "closed_nodes": [
            {
                "node_id": node.node_id,
                "upper_bound": node.upper_bound,
                "reason": node.reason,
                "proof": _proof_json(node.proof),
            }
            for node in report.closed_nodes
        ],
    }


def run(
    config_path: Path,
    output_path: Path,
    *,
    seconds_override: float | None = None,
    workers_override: int | None = None,
) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = int(config.get("rows", 6))
    columns = int(config.get("columns", 9))
    enabled_value = config.get("enabled_components", "all")
    enabled = None if enabled_value == "all" else tuple(enabled_value)
    component_limits = {
        str(item): (None if limit is None else int(limit))
        for item, limit in config.get("component_limits", {}).items()
    }
    problem = ic2_mark_i_problem(
        rows=rows,
        columns=columns,
        rod_budget=int(config.get("rod_budget", 25)),
        exact_rods=bool(config.get("exact_rods", True)),
        enabled_components=enabled,
        component_limits=component_limits,
    )
    seconds = float(
        config.get("time_limit_seconds", MAX_WALL_TIME_SECONDS)
        if seconds_override is None
        else seconds_override
    )
    if not 0 < seconds <= MAX_WALL_TIME_SECONDS:
        raise ValueError("time_limit_seconds must be in (0, 21600]")
    detected = os.cpu_count() or 1
    workers = int(
        config.get("workers", min(30, detected))
        if workers_override is None
        else workers_override
    )
    if not 1 <= workers <= detected:
        raise ValueError("workers must be between 1 and the detected CPU count")
    known = tuple(
        _decode_layout(item, rows, columns)
        for item in config.get("known_layouts", [])
    )
    result = solve_ic2_global(
        problem,
        time_limit_seconds=seconds,
        workers=workers,
        known_layouts=known,
        max_cycle_steps=int(config.get("thermal_horizon", 100_000)),
        master_fraction=float(config.get("master_fraction", 0.7)),
    )
    payload = _report_json(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"certified interval: {payload['lower_bound']} <= optimum <= "
        f"{payload['upper_bound']} EU/t; proven_global={payload['proven_global']}"
    )
    print(
        f"processed={payload['processed_nodes']} stop={payload['stop_reason']} "
        f"elapsed={payload['elapsed_seconds']:.3f}s"
    )
    print(f"report={output_path.resolve()}")
    return 0 if payload["best_witness"] is not None else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Certified full-label thermal Logic-Benders optimiser",
    )
    result.add_argument(
        "--config",
        type=Path,
        default=ROOT / "examples" / "math_25_rods_unlimited.json",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "global_benders_report.json",
    )
    result.add_argument("--seconds", type=float)
    result.add_argument("--workers", type=int)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(run(
        arguments.config,
        arguments.output,
        seconds_override=arguments.seconds,
        workers_override=arguments.workers,
    ))
