"""Run the parameterised anytime/proof algorithm from a JSON instance file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.anytime_math_optimizer import CertifiedAnytimeSolver  # noqa: E402
from ic2_reactor.mathematical_model import ic2_mark_i_problem  # noqa: E402


MAX_WALL_TIME_SECONDS = 6 * 60 * 60


def _decode_layout(value: object, rows: int, columns: int) -> tuple[str, ...]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if len(value) != rows * columns:
            raise ValueError("component-id layout has the wrong number of slots")
        return tuple(value)
    if not isinstance(value, dict):
        raise ValueError("known layout must be a component-id list or a rows/symbols object")
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


def run(
    config_path: Path,
    output_path: Path | None,
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
    requested_time = float(
        config.get("time_limit_seconds", MAX_WALL_TIME_SECONDS)
        if seconds_override is None
        else seconds_override
    )
    if not 0 < requested_time <= MAX_WALL_TIME_SECONDS:
        raise ValueError("time_limit_seconds must be in (0, 21600]")
    workers = int(
        config.get("workers", min(30, os.cpu_count() or 1))
        if workers_override is None
        else workers_override
    )
    if not 1 <= workers <= (os.cpu_count() or workers):
        raise ValueError("workers must be between 1 and the detected CPU count")
    known = tuple(
        _decode_layout(item, rows, columns)
        for item in config.get("known_layouts", [])
    )

    report = CertifiedAnytimeSolver(problem).solve(
        time_limit_seconds=requested_time,
        workers=workers,
        seed=int(config.get("seed", 221)),
        known_layouts=known,
        skeletons_per_tier=int(config.get("skeletons_per_tier", 8)),
        cooling_seconds_per_skeleton=float(
            config.get("cooling_seconds_per_skeleton", 60.0)
        ),
        thermal_horizon=int(config.get("thermal_horizon", 400)),
    )
    target = output_path or ROOT / "artifacts" / "mathematical_search_report.json"
    report.to_json(target)
    print(report.statement)
    print(
        f"closed_form_upper={report.closed_form_upper_bound} "
        f"static_master_upper={report.static_master_upper_bound} "
        f"analytical_cut_upper={report.analytical_cut_upper_bound} "
        f"elapsed={report.elapsed_seconds:.3f}s"
    )
    print(f"report={target.resolve()}")
    return 0 if report.best_cycle is not None else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Parameterised IC2 mathematical anytime/global-proof solver"
    )
    result.add_argument(
        "--config",
        type=Path,
        default=ROOT / "examples" / "math_25_rods_unlimited.json",
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--seconds", type=float, help="override JSON wall-time limit")
    result.add_argument("--workers", type=int, help="override JSON CPU worker count")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(run(
        arguments.config,
        arguments.output,
        seconds_override=arguments.seconds,
        workers_override=arguments.workers,
    ))
