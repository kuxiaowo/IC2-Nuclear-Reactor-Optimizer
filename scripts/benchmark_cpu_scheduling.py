from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ic2_reactor.cpu_scheduling import (  # noqa: E402
    configure_current_process_cpu_sets,
    cpu_scheduling_plan,
    initialize_compute_worker,
    windows_cpu_sets,
)
from ic2_reactor.optimizer import _maximum_flow  # noqa: E402


def _pruning_kernel(iterations: int, seed: int) -> int:
    """Exercise the small dynamic graphs and branches used by heat pruning."""
    state = seed | 1
    checksum = 0
    columns = 6
    slots = 36
    source = slots + 1
    sink = slots + 2
    for _ in range(iterations):
        edges: list[tuple[int, int, int]] = []
        for index in range(slots):
            state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
            node = index + 1
            capacity = 1 + ((state >> 16) & 15)
            if state & 1:
                edges.append((source, node, capacity))
            if state & 2:
                edges.append((node, sink, 1 + capacity // 2))
            if index % columns + 1 < columns:
                edges.append((node, node + 1, capacity))
                edges.append((node + 1, node, capacity // 2 + 1))
            if index + columns < slots:
                edges.append((node, node + columns, capacity))
        checksum += _maximum_flow(slots + 3, edges, source, sink)
    return checksum


def _initialize_strict_worker(cpu_set_queue) -> None:
    cpu_set_id = cpu_set_queue.get()
    configure_current_process_cpu_sets((cpu_set_id,), high_performance=True)


def _one_cpu_set_per_physical_core(cpu_sets, allowed_ids: set[int]) -> tuple[int, ...]:
    cores: dict[tuple[int, int], list] = {}
    for value in cpu_sets:
        if value.id in allowed_ids:
            cores.setdefault((value.group, value.core_index), []).append(value)
    return tuple(
        min(core, key=lambda value: value.logical_processor_index).id
        for core in cores.values()
    )


def _run_mode(
    name: str,
    workers: int,
    iterations: int,
    cpu_set_ids: tuple[int, ...] | None,
    *,
    strict: bool = False,
) -> tuple[str, float]:
    context = multiprocessing.get_context("spawn")
    manager = None
    initializer = None
    initargs = ()
    if strict:
        manager = context.Manager()
        cpu_set_queue = manager.Queue()
        for cpu_set_id in cpu_set_ids or ():
            cpu_set_queue.put(cpu_set_id)
        initializer = _initialize_strict_worker
        initargs = (cpu_set_queue,)
    elif cpu_set_ids:
        initializer = initialize_compute_worker
        initargs = (cpu_set_ids,)

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            # Start and initialize every worker before measuring steady state.
            warmup = [executor.submit(_pruning_kernel, 20, 1000 + index) for index in range(workers)]
            for future in warmup:
                future.result()
            base, remainder = divmod(iterations, workers)
            started = time.perf_counter()
            futures = [
                executor.submit(_pruning_kernel, base + (index < remainder), 10_000 + index)
                for index in range(workers)
            ]
            checksum = sum(future.result() for future in futures)
            elapsed = time.perf_counter() - started
    finally:
        if manager is not None:
            manager.shutdown()
    throughput = iterations / elapsed
    print(
        f"{name:>18}: {workers:2d} workers, {elapsed:7.3f}s, "
        f"{throughput:,.0f} pruning kernels/s, checksum={checksum}",
        flush=True,
    )
    return name, throughput


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 Windows 混合 CPU 剪枝调度方案")
    parser.add_argument("--iterations", type=int, default=120_000)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    cpu_sets = windows_cpu_sets()
    plan = cpu_scheduling_plan(2, cpu_sets)
    if not plan.worker_cpu_set_ids:
        parser.error("当前系统不支持 Windows CPU Set 基准")
    physical_ids = _one_cpu_set_per_physical_core(
        cpu_sets,
        set(plan.worker_cpu_set_ids),
    )
    modes = (
        ("auto-30", min(30, len(plan.worker_cpu_set_ids)), None, False),
        ("highqos-pool-30", min(30, len(plan.worker_cpu_set_ids)), plan.worker_cpu_set_ids, False),
        ("strict-pin-30", min(30, len(plan.worker_cpu_set_ids)), plan.worker_cpu_set_ids, True),
        ("physical-only", len(physical_ids), physical_ids, False),
    )
    samples: dict[str, list[float]] = {name: [] for name, *_ in modes}
    print(
        f"detected {plan.available_physical_cores} physical / "
        f"{plan.available_logical_processors} logical; reserved sets "
        f"{plan.reserved_cpu_set_ids}",
        flush=True,
    )
    for round_index in range(args.rounds):
        ordered = modes if round_index % 2 == 0 else tuple(reversed(modes))
        print(f"round {round_index + 1}", flush=True)
        for name, workers, cpu_set_ids, strict in ordered:
            result_name, throughput = _run_mode(
                name,
                workers,
                args.iterations,
                cpu_set_ids,
                strict=strict,
            )
            samples[result_name].append(throughput)
    print("mean throughput", flush=True)
    for name, values in samples.items():
        print(f"{name:>18}: {sum(values) / len(values):,.0f} kernels/s", flush=True)


if __name__ == "__main__":
    main()
