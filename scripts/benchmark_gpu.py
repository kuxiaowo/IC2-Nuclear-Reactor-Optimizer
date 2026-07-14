from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.gpu_acceleration import CudaBatchScorer  # noqa: E402
from ic2_reactor.optimizer import skeleton_eu_per_tick, skeleton_heat_per_tick  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 GPU 批量静态评分与 Python CPU 基线")
    parser.add_argument("--layouts", type=int, default=65_536)
    parser.add_argument("--columns", type=int, choices=range(3, 10), default=3)
    parser.add_argument("--seed", type=int, default=221)
    args = parser.parse_args()
    if args.layouts < 1:
        parser.error("--layouts 必须大于 0")

    rng = random.Random(args.seed)
    values = (
        "empty",
        "uranium_single",
        "uranium_dual",
        "uranium_quad",
        "iridium_reflector",
        "advanced_heat_vent",
        "component_heat_vent",
        "coolant_10k",
    )
    slots = args.columns * 6
    layouts = [tuple(rng.choice(values) for _ in range(slots)) for _ in range(args.layouts)]

    scorer = CudaBatchScorer()
    scorer.score(layouts[: min(256, len(layouts))], args.columns)  # JIT warm-up
    started = time.perf_counter()
    gpu = scorer.score(layouts, args.columns)
    gpu_seconds = time.perf_counter() - started

    started = time.perf_counter()
    cpu = [
        (int(skeleton_eu_per_tick(layout, args.columns)), skeleton_heat_per_tick(layout, args.columns))
        for layout in layouts
    ]
    cpu_seconds = time.perf_counter() - started
    exact_match = all(
        power == int(gpu.power[index]) and heat == int(gpu.generated_heat[index])
        for index, (power, heat) in enumerate(cpu)
    )
    print(json.dumps({
        "device": scorer.info.label,
        "layouts": len(layouts),
        "gpu_end_to_end_seconds": round(gpu_seconds, 4),
        "cpu_python_seconds": round(cpu_seconds, 4),
        "speedup": round(cpu_seconds / gpu_seconds, 2),
        "gpu_layouts_per_second": round(len(layouts) / gpu_seconds),
        "exact_match": exact_match,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
