from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.gpu_acceleration import CudaFixedPointEvaluator  # noqa: E402
from ic2_reactor.gpu_full_simulation import CudaFullSimulator  # noqa: E402
from ic2_reactor.models import FuelConstraint, OptimizationRequest  # noqa: E402
from ic2_reactor.optimizer import (  # noqa: E402
    _fixed_point_certificate,
    _run_exhaustive_shard,
    estimate_exhaustive_space,
    prove_simple_fixed_point,
)


class QuietQueue:
    def put(self, _message) -> None:
        pass


class NeverCancelled:
    def is_set(self) -> bool:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较同一完整穷举空间的 CPU、CUDA 证书和 CUDA 完整模拟路径"
    )
    parser.add_argument("--vent-cap", type=int, default=3, choices=range(1, 9))
    parser.add_argument(
        "--component",
        default="reactor_heat_vent",
        choices=("reactor_heat_vent", "coolant_10k", "heat_exchanger"),
        help="用于构造穷举空间的非燃料组件；--vent-cap 同时作为其库存上限",
    )
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--ticks-per-launch", type=int, default=256)
    args = parser.parse_args()

    common = dict(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={args.component: args.vent_cap},
        marks=["I", "II", "III", "IV", "V"],
        solver="exhaustive",
        result_limit=3,
        cpu_workers=1,
        max_reactor_ticks=40_000,
        gpu_exhaustive_batch_size=args.batch_size,
        gpu_ticks_per_launch=args.ticks_per_launch,
    )
    # Compile before timing; a long-running server pays this cost only once.
    warmup = tuple(["uranium_single", args.component, *("empty" for _ in range(16))])
    evaluator = CudaFixedPointEvaluator()
    evaluator.certify([warmup], 3, 40_000)
    full_simulator = CudaFullSimulator(ticks_per_launch=args.ticks_per_launch)
    full_simulator.simulate([warmup], 3, 40_000)

    outputs = {}
    timings = {}
    for accelerator in ("cpu", "cuda", "cuda_full"):
        prove_simple_fixed_point.cache_clear()
        _fixed_point_certificate.cache_clear()
        request = OptimizationRequest(**common, accelerator=accelerator)
        started = time.perf_counter()
        outputs[accelerator] = _run_exhaustive_shard(
            request.model_dump(mode="json"),
            0,
            (),
            QuietQueue(),
            NeverCancelled(),
        )
        timings[accelerator] = time.perf_counter() - started

    cpu = outputs["cpu"]
    gpu = outputs["cuda"]
    gpu_full = outputs["cuda_full"]
    cpu_boards = {
        mark: [candidate.public_dict(3) for candidate in board]
        for mark, board in cpu["boards"].items()
    }
    gpu_boards = {
        mark: [candidate.public_dict(3) for candidate in board]
        for mark, board in gpu["boards"].items()
    }
    gpu_full_boards = {
        mark: [candidate.public_dict(3) for candidate in board]
        for mark, board in gpu_full["boards"].items()
    }
    request = OptimizationRequest(**common, accelerator="cuda")
    print(json.dumps({
        "device": evaluator.info.label,
        "component": args.component,
        "layouts": estimate_exhaustive_space(request),
        "cpu_seconds": round(timings["cpu"], 4),
        "gpu_hybrid_seconds": round(timings["cuda"], 4),
        "gpu_full_seconds": round(timings["cuda_full"], 4),
        "hybrid_speedup": round(timings["cpu"] / timings["cuda"], 2),
        "full_speedup": round(timings["cpu"] / timings["cuda_full"], 2),
        "gpu_certified": gpu["gpu_certified"],
        "cpu_fallback": gpu["gpu_fallback"],
        "gpu_full_simulated": gpu_full["gpu_full_simulated"],
        "same_checked_count": cpu["checked"] == gpu["checked"] == gpu_full["checked"],
        "same_leaderboards": cpu_boards == gpu_boards == gpu_full_boards,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
