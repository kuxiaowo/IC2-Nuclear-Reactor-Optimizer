"""Low-load microbenchmark for auditable proof work units (CPU only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.cycle_proof import DeterministicCycleVerifier, IC2TransitionSystem  # noqa: E402
from ic2_reactor.ic2_thermal_catalog import (  # noqa: E402
    IC2_HEAT_FLOW_CATALOGUE,
    IC2_PERIODIC_PREFIX_CATALOGUE,
)
from ic2_reactor.mathematical_model import (  # noqa: E402
    aggregate_overload_analysis,
    ic2_mark_i_problem,
    route_conditioned_overload_analysis,
)
from ic2_reactor.periodic_prefix import periodic_prefix_flow_bound  # noqa: E402
from ic2_reactor.proof_complexity import (  # noqa: E402
    proof_work_envelope,
    project_proof_budget,
)
from ic2_reactor.structural_master import AggregateDegreeEmbeddingMaster  # noqa: E402
from ic2_reactor.thermal_master import ThermalCutMaster  # noqa: E402


def _known_layout(config: dict, rows: int, columns: int) -> tuple[str, ...]:
    raw = config["known_layouts"][0]
    symbols = raw["symbols"]
    joined = "".join(raw["rows"])
    if len(joined) != rows * columns:
        raise ValueError("known witness has the wrong dimensions")
    return tuple(symbols[item] for item in joined)


def _samples(repeats: int, operation) -> list[float]:
    result = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        result.append(perf_counter() - started)
    return result


def _summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(samples),
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "maximum_seconds": max(samples),
    }


def run(config_path: Path, output_path: Path, repeats: int) -> None:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = int(config.get("rows", 6))
    columns = int(config.get("columns", 9))
    problem = ic2_mark_i_problem(
        rows=rows,
        columns=columns,
        rod_budget=int(config.get("rod_budget", 25)),
        exact_rods=bool(config.get("exact_rods", True)),
    )
    witness = _known_layout(config, rows, columns)
    envelope = proof_work_envelope(
        problem,
        incumbent_lower_bound=380,
        static_upper_bound=455,
    )

    prefix_samples = _samples(
        repeats,
        lambda: periodic_prefix_flow_bound(
            problem,
            witness,
            IC2_PERIODIC_PREFIX_CATALOGUE,
            base_hull_capacity=10_000,
        ),
    )
    cycle_samples = _samples(
        repeats,
        lambda: DeterministicCycleVerifier().verify(
            IC2TransitionSystem(columns),
            witness,
            max_steps=1_000,
            time_limit_seconds=5,
        ),
    )
    prefix_capacity = project_proof_budget(
        0,
        max(prefix_samples),
        workers=30,
        parallel_efficiency=0.70,
    )
    cycle_capacity = project_proof_budget(
        0,
        max(cycle_samples),
        workers=30,
        parallel_efficiency=0.70,
    )

    aggregate_started = perf_counter()
    aggregate = aggregate_overload_analysis(problem, 455)
    aggregate_seconds = perf_counter() - aggregate_started
    routing = tuple(
        route_conditioned_overload_analysis(problem, pattern)
        for pattern in aggregate.structurally_surviving_patterns
    )
    embedding_master = AggregateDegreeEmbeddingMaster(problem)
    embedding_samples = []
    for pattern in aggregate.structurally_surviving_patterns:
        started = perf_counter()
        result = embedding_master.solve(pattern, seconds=2, workers=1)
        embedding_samples.append(perf_counter() - started)
        if result.possible is not True:
            raise RuntimeError("the recorded 455 geometry benchmark changed status")

    deliberately_failed = tuple(
        ["uranium_quad"] * 6
        + ["uranium_single"]
        + ["empty"] * (problem.graph.size - 7)
    )
    failed = periodic_prefix_flow_bound(
        problem,
        deliberately_failed,
        IC2_PERIODIC_PREFIX_CATALOGUE,
        base_hull_capacity=10_000,
    )
    if failed.feasible:
        raise RuntimeError("the deliberately uncooled benchmark layout became feasible")

    build_results = {}
    for cut_count in (0, 1, 4):
        master = ThermalCutMaster(
            problem,
            IC2_HEAT_FLOW_CATALOGUE,
            prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
            base_hull_capacity=10_000,
        )
        started = perf_counter()
        model, _variables = master.build(
            prefix_cuts=(failed.cut_template,) * cut_count,
            exact_power=455,
        )
        elapsed = perf_counter() - started
        build_results[str(cut_count)] = {
            "seconds": elapsed,
            "variables": len(model.proto.variables),
            "constraints": len(model.proto.constraints),
        }
    periodic_master = ThermalCutMaster(
        problem,
        IC2_HEAT_FLOW_CATALOGUE,
        prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
        base_hull_capacity=10_000,
    )
    started = perf_counter()
    periodic_model, _variables = periodic_master.build(
        enforce_periodic_prefix_flow=True,
        exact_power=455,
    )
    embedded_periodic_build = {
        "seconds": perf_counter() - started,
        "variables": len(periodic_model.proto.variables),
        "constraints": len(periodic_model.proto.constraints),
    }

    payload = {
        "scope": "CPU-only low-load microbenchmark; no GPU and no full optimisation",
        "problem": {
            "rows": rows,
            "columns": columns,
            "rod_budget": problem.rod_budget,
            "incumbent_lower_bound": 380,
            "static_upper_bound": 455,
        },
        "input_envelope": {
            "open_power_tiers": envelope.open_power_tiers,
            "frontier_state_bound": envelope.frontier_state_bound,
            "frontier_transition_bound": envelope.frontier_transition_bound,
            "rod_feasible_power_skeletons": envelope.rod_feasible_power_skeletons,
            "rod_feasible_full_layouts": envelope.rod_feasible_full_layouts,
        },
        "safe_witness_periodic_prefix": _summary(prefix_samples),
        "safe_witness_exact_cycle": _summary(cycle_samples),
        "reference_capacity_at_30_workers_70pct_efficiency": {
            "periodic_prefix_units_in_six_hours": prefix_capacity.maximum_units_in_budget,
            "known_safe_cycle_units_in_six_hours": cycle_capacity.maximum_units_in_budget,
            "warning": (
                "reference throughput only; unresolved master branches and long "
                "UNKNOWN trajectories are not represented"
            ),
        },
        "aggregate_455": {
            "seconds": aggregate_seconds,
            "patterns": aggregate.pattern_count,
            "overload_survivors": len(aggregate.surviving_patterns),
            "structural_survivors": len(aggregate.structurally_surviving_patterns),
            "route_profile_counts": [item.profile_count for item in routing],
            "route_minimum_margins": [item.minimum_margin for item in routing],
            "route_conditioned_survivors": sum(not item.excluded for item in routing),
        },
        "geometry_455": _summary(embedding_samples),
        "prefix_cut_model_build": build_results,
        "embedded_periodic_model_build": embedded_periodic_build,
        "six_hour_projection": None,
        "six_hour_projection_reason": (
            "the post-cut remaining equivalence-class count is not yet measured; "
            "microbenchmark speed alone cannot prove a six-hour bound"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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
        default=ROOT / "artifacts" / "proof_unit_benchmark.json",
    )
    result.add_argument("--repeats", type=int, default=3)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.config, args.output, args.repeats)
