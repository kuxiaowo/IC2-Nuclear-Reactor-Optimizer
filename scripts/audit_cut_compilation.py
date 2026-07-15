"""Compile representative proof cuts and stop before any layout search.

This is a capacity audit, not an optimiser.  Probe layouts only supply fixed
network partitions whose factor tables can be measured on the full enabled
component catalogue.  The script never invokes a cooling DP, LNS, cycle
simulation, GPU kernel, or objective search.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.factorized_cooling_master import (  # noqa: E402
    FactorizedCoolingCutMaster,
)
from ic2_reactor.ic2_thermal_catalog import (  # noqa: E402
    IC2_HEAT_FLOW_CATALOGUE,
    IC2_PERIODIC_PREFIX_CATALOGUE,
)
from ic2_reactor.mathematical_model import ic2_mark_i_problem  # noqa: E402
from ic2_reactor.periodic_prefix import periodic_prefix_flow_bound  # noqa: E402
from ic2_reactor.thermal_relaxation import layout_heat_flow_bound  # noqa: E402


def _decode_layout(value: object, rows: int, columns: int) -> tuple[str, ...]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if len(value) != rows * columns:
            raise ValueError("component-id probe layout has the wrong size")
        return tuple(value)
    if not isinstance(value, dict):
        raise ValueError("probe layout must be a component-id list or symbolic rows")
    raw_rows = value.get("rows")
    symbols = value.get("symbols")
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != rows
        or not all(
            isinstance(row, str) and len(row) == columns for row in raw_rows
        )
        or not isinstance(symbols, dict)
    ):
        raise ValueError("invalid symbolic probe layout")
    try:
        return tuple(symbols[symbol] for symbol in "".join(raw_rows))
    except KeyError as error:
        raise ValueError(f"missing probe symbol mapping: {error.args[0]}") from error


def run(
    config_path: Path,
    output_path: Path,
    *,
    seconds: float,
) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
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
    raw_probes = config.get("probe_layouts", config.get("known_layouts", []))
    probes = tuple(
        _decode_layout(item, rows, columns) for item in raw_probes
    )
    if not probes:
        raise ValueError("capacity audit requires at least one probe layout")

    average_results = tuple(
        layout_heat_flow_bound(problem, layout, IC2_HEAT_FLOW_CATALOGUE)
        for layout in probes
    )
    prefix_results = tuple(
        periodic_prefix_flow_bound(
            problem,
            layout,
            IC2_PERIODIC_PREFIX_CATALOGUE,
            base_hull_capacity=10_000,
        )
        for layout in probes
    )
    master = FactorizedCoolingCutMaster(
        problem,
        IC2_HEAT_FLOW_CATALOGUE,
        prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
        base_hull_capacity=10_000,
    )
    power_ids = {item.id for item in problem.power_components}
    audit_skeleton = tuple(
        label if label in power_ids else "empty"
        for label in probes[0]
    )
    compilation = master.compile_cuts_for_skeleton(
        audit_skeleton,
        average_cuts=tuple(item.cut_template for item in average_results),
        prefix_cuts=tuple(item.cut_template for item in prefix_results),
        time_limit_seconds=seconds,
    )
    payload = {
        "scope": (
            "fixed-skeleton partial evaluation and cut compilation only; no "
            "layout DP, objective search, cycle simulation, or GPU kernel"
        ),
        "problem": {
            "rows": rows,
            "columns": columns,
            "rod_budget": problem.rod_budget,
            "labels": len(master.labels),
            "probe_layouts": len(probes),
            "conditioned_probe_index": 0,
            "fixed_nonempty_power_cells": sum(
                label != "empty" for label in audit_skeleton
            ),
        },
        "probe_cuts": {
            "average_necessary_condition_satisfied": [
                item.necessary_condition_satisfied for item in average_results
            ],
            "prefix_feasible": [item.feasible for item in prefix_results],
        },
        "compilation": {
            "proven": compilation.proven,
            "elapsed_seconds": compilation.elapsed_seconds,
            "stop_reason": compilation.stop_reason,
            "selection": (
                None
                if compilation.selection is None
                else asdict(compilation.selection)
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if compilation.proven else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--config",
        type=Path,
        default=ROOT / "examples" / "global_benders_25_rods_unlimited.json",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "cut_compilation_audit.json",
    )
    result.add_argument("--seconds", type=float, default=60.0)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    raise SystemExit(run(args.config, args.output, seconds=args.seconds))
