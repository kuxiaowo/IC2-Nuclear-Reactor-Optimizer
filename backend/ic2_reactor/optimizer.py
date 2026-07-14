from __future__ import annotations

import math
import multiprocessing
import queue
import random
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field

from .components import COMPONENTS
from .engine import ReactorSimulator, SimulationOptions
from .mark import mark_family
from .models import Layout, OptimizationRequest


@dataclass(slots=True)
class CandidateResult:
    layout: tuple[str, ...]
    mark: str
    average_eu_per_tick: float
    total_eu: float
    safe_game_ticks: int
    safety_margin: float
    component_count: int
    canonical: str

    def score(self) -> tuple:
        return (
            self.average_eu_per_tick,
            self.total_eu,
            self.safe_game_ticks,
            self.safety_margin,
            -self.component_count,
        )

    def public_dict(self, columns: int) -> dict:
        return {
            "layout": {"ruleset": "ic2-experimental-2.8.221", "columns": columns, "initial_hull_heat": 0, "slots": list(self.layout)},
            "mark": self.mark,
            "average_eu_per_tick": self.average_eu_per_tick,
            "total_eu": self.total_eu,
            "safe_game_ticks": self.safe_game_ticks,
            "safety_margin": self.safety_margin,
            "component_count": self.component_count,
        }


def _transform(layout: tuple[str, ...], columns: int, flip_h: bool, flip_v: bool) -> tuple[str, ...]:
    rows = [list(layout[row * columns:(row + 1) * columns]) for row in range(6)]
    if flip_h:
        rows = [list(reversed(row)) for row in rows]
    if flip_v:
        rows = list(reversed(rows))
    return tuple(item for row in rows for item in row)


def canonical_layout(layout: tuple[str, ...], columns: int) -> str:
    return "|".join(canonical_tuple(layout, columns))


def canonical_tuple(layout: tuple[str, ...], columns: int) -> tuple[str, ...]:
    """Return a display-group key for mirrored layouts, not a simulation-equivalence key."""
    return min(_transform(layout, columns, h, v) for h in (False, True) for v in (False, True))


def _allowed_and_caps(request: OptimizationRequest) -> tuple[list[str], dict[str, int]]:
    slots = request.columns * 6
    if request.fuel.mode == "total_rods":
        fuels = ["uranium_single", "uranium_dual", "uranium_quad"] if request.fuel.total_rods else []
        caps = {item: slots for item in fuels}
    else:
        fuel_limits = {
            "uranium_single": request.fuel.single,
            "uranium_dual": request.fuel.dual,
            "uranium_quad": request.fuel.quad,
        }
        fuels = [item for item, limit in fuel_limits.items() if limit > 0]
        caps = {item: fuel_limits[item] for item in fuels}
    nonfuel = [item for item, limit in request.component_limits.items() if limit > 0]
    caps.update({item: request.component_limits[item] for item in nonfuel})
    return [*fuels, *nonfuel], caps


def _exhaustive_shards(request: OptimizationRequest) -> list[tuple[tuple[int, str], ...]]:
    """Split the search into disjoint assignments on two central cells.

    Central cells distribute the labelled search space more evenly than a
    top-left prefix does.
    """
    columns = request.columns
    positions = (2 * columns + columns // 2, 3 * columns + columns // 2)
    allowed, caps = _allowed_and_caps(request)
    values = ["empty", *allowed]
    shards: list[tuple[tuple[int, str], ...]] = []
    for first in values:
        for second in values:
            used: dict[str, int] = {}
            rods = 0
            valid = True
            for item in (first, second):
                if item == "empty":
                    continue
                used[item] = used.get(item, 0) + 1
                rods += COMPONENTS[item].rod_count
                if used[item] > caps[item]:
                    valid = False
            if request.fuel.mode == "total_rods" and rods > request.fuel.total_rods:
                valid = False
            if valid:
                shards.append(((positions[0], first), (positions[1], second)))
    # Prefer fuel-bearing assignments so leaderboards begin producing useful
    # results immediately. Every shard is still evaluated in full.
    shards.sort(key=lambda shard: (
        not any(COMPONENTS[item].rod_count for _, item in shard),
        shard,
    ))
    return shards


def estimate_exhaustive_space(request: OptimizationRequest) -> int:
    """Count inventory-valid labelled layouts (before symmetry reduction)."""
    slots = request.columns * 6
    types: list[tuple[int, int, bool]] = []  # cap, rod cost, is fuel
    if request.fuel.mode == "separate":
        types.extend((cap, 0, True) for cap in (request.fuel.single, request.fuel.dual, request.fuel.quad) if cap > 0)
    elif request.fuel.total_rods > 0:
        types.extend((request.fuel.total_rods // rods, rods, True) for rods in (1, 2, 4))
    types.extend((cap, 0, False) for cap in request.component_limits.values() if cap > 0)

    # dp[(occupied slots, used rods, has fuel)] = number of ways to choose labelled positions.
    dp: dict[tuple[int, int, bool], int] = {(0, 0, False): 1}
    for cap, rod_cost, is_fuel in types:
        next_dp: dict[tuple[int, int, bool], int] = {}
        for (used, rods, has_fuel), ways in dp.items():
            for count in range(min(cap, slots - used) + 1):
                next_rods = rods + count * rod_cost
                if request.fuel.mode == "total_rods" and next_rods > request.fuel.total_rods:
                    break
                key = (used + count, next_rods, has_fuel or (is_fuel and count > 0))
                next_dp[key] = next_dp.get(key, 0) + ways * math.comb(slots - used, count)
        dp = next_dp
    return sum(ways for (_, _, has_fuel), ways in dp.items() if has_fuel)


def evaluate_layout(
    layout: tuple[str, ...],
    columns: int,
    max_reactor_ticks: int,
    cancel_check=None,
) -> CandidateResult:
    """Process-safe full candidate evaluation used by heuristic worker processes."""
    simulator = ReactorSimulator(Layout(columns=columns, initial_hull_heat=0, slots=list(layout)))
    run = simulator.simulate(SimulationOptions(
        max_game_ticks=max_reactor_ticks * 20,
        auto_refuel=True,
        stop_on_stable=True,
        record_components=False,
        cancel_check=cancel_check,
    ))
    safe_ticks = run.summary.first_intervention_tick or run.summary.game_ticks
    return CandidateResult(
        layout=layout,
        mark=run.summary.mark or "未分类",
        average_eu_per_tick=run.summary.average_eu_per_tick,
        total_eu=run.summary.average_eu_per_tick * safe_ticks,
        safe_game_ticks=safe_ticks,
        safety_margin=1.0 - run.summary.peak_hull_heat / run.summary.max_hull_heat,
        component_count=sum(item != "empty" for item in layout),
        canonical=canonical_layout(layout, columns),
    )


def _rank_candidates(values: list[CandidateResult]) -> list[CandidateResult]:
    board: dict[str, CandidateResult] = {}
    for result in values:
        previous = board.get(result.canonical)
        if previous is None or result.score() > previous.score():
            board[result.canonical] = result
    ordered = sorted(board.values(), key=lambda item: item.canonical)
    ordered.sort(key=lambda item: item.score(), reverse=True)
    return ordered[:10]


def _run_exhaustive_shard(
    request_data: dict,
    shard_id: int,
    fixed_items: tuple[tuple[int, str], ...],
    progress_queue,
    cancel_event,
) -> dict:
    """Process worker for one disjoint exhaustive-search shard."""
    request = OptimizationRequest.model_validate(request_data)
    allowed, remaining = _allowed_and_caps(request)
    fixed = dict(fixed_items)
    slots = request.columns * 6
    layout = ["empty"] * slots
    rods = 0
    has_fuel = False
    for position, item in fixed_items:
        layout[position] = item
        if item != "empty":
            remaining[item] -= 1
            rods += COMPONENTS[item].rod_count
            has_fuel = has_fuel or COMPONENTS[item].rod_count > 0

    checked = 0
    pruned = 0
    evaluated = 0
    visits = 0
    last_report = time.monotonic()
    cancelled = False
    cancel_cache = False
    last_cancel_check = 0.0
    boards: dict[str, list[CandidateResult]] = {mark: [] for mark in request.marks}

    def cancellation_requested() -> bool:
        nonlocal cancel_cache, last_cancel_check
        now = time.monotonic()
        if now - last_cancel_check >= 0.1:
            cancel_cache = cancel_event.is_set()
            last_cancel_check = now
        return cancel_cache

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 0.25:
            progress_queue.put(("progress", shard_id, checked, pruned, evaluated))
            last_report = now

    def generate(position: int, current_rods: int, current_has_fuel: bool) -> None:
        nonlocal checked, pruned, evaluated, visits, cancelled
        visits += 1
        if visits % 4096 == 0 and cancel_event.is_set():
            cancelled = True
            return
        if position == slots:
            if not current_has_fuel:
                return
            checked += 1
            raw = tuple(layout)
            result = evaluate_layout(
                raw,
                request.columns,
                request.max_reactor_ticks,
                cancel_check=cancellation_requested,
            )
            if cancellation_requested():
                cancelled = True
                return
            evaluated += 1
            family = mark_family(result.mark)
            if family in boards:
                previous = boards[family]
                ranked = _rank_candidates([*previous, result])
                if [item.canonical for item in ranked] != [item.canonical for item in previous]:
                    boards[family] = ranked
                    progress_queue.put(("candidate", result))
            report()
            return
        if cancelled:
            return
        if position in fixed:
            generate(position + 1, current_rods, current_has_fuel)
            return

        layout[position] = "empty"
        generate(position + 1, current_rods, current_has_fuel)
        for item in allowed:
            if cancelled or remaining[item] <= 0:
                continue
            rod_cost = COMPONENTS[item].rod_count
            if request.fuel.mode == "total_rods" and current_rods + rod_cost > request.fuel.total_rods:
                continue
            remaining[item] -= 1
            layout[position] = item
            generate(position + 1, current_rods + rod_cost, current_has_fuel or rod_cost > 0)
            remaining[item] += 1
        layout[position] = "empty"

    generate(0, rods, has_fuel)
    report(force=True)
    return {
        "shard_id": shard_id,
        "checked": checked,
        "pruned": pruned,
        "evaluated": evaluated,
        "boards": boards,
        "cancelled": cancelled,
    }


class OptimizationJob:
    def __init__(self, request: OptimizationRequest):
        self.id = uuid.uuid4().hex
        self.request = request
        self.status = "queued"
        self.progress = 0.0
        self.evaluated = 0
        self.checked = 0
        self.pruned = 0
        self.generation = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.message = "等待开始"
        self.error: str | None = None
        self.proven_global = False
        self.exhaustive_estimate = estimate_exhaustive_space(request) if request.solver == "exhaustive" else None
        self.cancel_event = threading.Event()
        self.process_cancel_event = None
        self.leaderboards: dict[str, list[CandidateResult]] = {mark: [] for mark in request.marks}

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "evaluated": self.evaluated,
            "checked": self.checked,
            "pruned": self.pruned,
            "generation": self.generation,
            "message": self.message,
            "error": self.error,
            "proven_global": self.proven_global,
            "estimate": str(self.exhaustive_estimate) if self.exhaustive_estimate is not None else None,
            "cpu_workers": self.request.cpu_workers,
            "elapsed_seconds": (self.finished_at or time.time()) - self.started_at if self.started_at else 0,
            "leaderboards": {
                mark: [candidate.public_dict(self.request.columns) for candidate in values]
                for mark, values in self.leaderboards.items()
            },
        }

    def _fuel_allowed(self) -> list[str]:
        if self.request.fuel.mode == "total_rods":
            return ["uranium_single", "uranium_dual", "uranium_quad"] if self.request.fuel.total_rods > 0 else []
        result = []
        if self.request.fuel.single:
            result.append("uranium_single")
        if self.request.fuel.dual:
            result.append("uranium_dual")
        if self.request.fuel.quad:
            result.append("uranium_quad")
        return result

    def _within_limits(self, layout: tuple[str, ...]) -> bool:
        counts: dict[str, int] = {}
        for item in layout:
            counts[item] = counts.get(item, 0) + 1
        if self.request.fuel.mode == "total_rods":
            rods = sum(COMPONENTS[item].rod_count for item in layout)
            if rods > self.request.fuel.total_rods:
                return False
        else:
            if counts.get("uranium_single", 0) > self.request.fuel.single:
                return False
            if counts.get("uranium_dual", 0) > self.request.fuel.dual:
                return False
            if counts.get("uranium_quad", 0) > self.request.fuel.quad:
                return False
        return all(counts.get(item, 0) <= limit for item, limit in self.request.component_limits.items()) and all(
            item in {"empty", "uranium_single", "uranium_dual", "uranium_quad"} or item in self.request.component_limits
            for item in layout
        )

    def _random_layout(self, rng: random.Random) -> tuple[str, ...]:
        slots = self.request.columns * 6
        values = ["empty"]
        values.extend(self._fuel_allowed())
        values.extend(item for item, limit in self.request.component_limits.items() if limit > 0)
        if len(values) == 1:
            return tuple("empty" for _ in range(slots))
        for _ in range(500):
            layout = tuple(rng.choice(values) if rng.random() < 0.7 else "empty" for _ in range(slots))
            if any(COMPONENTS[item].kind == "fuel" for item in layout) and self._within_limits(layout):
                return layout
        result = ["empty"] * slots
        fuel = self._fuel_allowed()
        if fuel:
            result[rng.randrange(slots)] = fuel[0]
        return tuple(result)

    def _mutate(self, layout: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
        result = list(layout)
        mode = rng.randrange(3)
        if mode == 0:
            # 局部交换
            a, b = rng.sample(range(len(result)), 2)
            result[a], result[b] = result[b], result[a]
        elif mode == 1:
            # 将一个现有组件移动到空格。
            occupied = [index for index, item in enumerate(result) if item != "empty"]
            empty = [index for index, item in enumerate(result) if item == "empty"]
            if occupied and empty:
                source, target = rng.choice(occupied), rng.choice(empty)
                result[target], result[source] = result[source], "empty"
        else:
            # 在库存约束内替换组件。
            values = ["empty", *self._fuel_allowed(), *(item for item, limit in self.request.component_limits.items() if limit > 0)]
            result[rng.randrange(len(result))] = rng.choice(values)
        candidate = tuple(result)
        return candidate if self._within_limits(candidate) and any(COMPONENTS[x].kind == "fuel" for x in candidate) else layout

    def _evaluate(self, layout: tuple[str, ...]) -> CandidateResult | None:
        if self.cancel_event.is_set() or not self._within_limits(layout):
            return None
        result = evaluate_layout(layout, self.request.columns, self.request.max_reactor_ticks)
        family = mark_family(result.mark)
        if family not in self.request.marks:
            return None
        return result

    def _accept(self, result: CandidateResult | None, *, count_evaluation: bool = True) -> None:
        if count_evaluation:
            self.evaluated += 1
        if result is None:
            return
        family = mark_family(result.mark)
        if family is None or family not in self.leaderboards:
            return
        self.leaderboards[family] = _rank_candidates([*self.leaderboards[family], result])

    def _run_heuristic(self) -> None:
        rng = random.Random(self.request.seed + self.evaluated)
        island_count = min(self.request.cpu_workers, max(1, self.request.population // 10))
        base_size, remainder = divmod(self.request.population, island_count)
        islands = [
            [self._random_layout(rng) for _ in range(base_size + (island < remainder))]
            for island in range(island_count)
        ]
        deadline = time.time() + self.request.time_budget_seconds
        executor = ProcessPoolExecutor(max_workers=self.request.cpu_workers) if self.request.cpu_workers > 1 else None
        for generation in range(self.request.generations):
            if self.cancel_event.is_set() or time.time() >= deadline:
                break
            self.generation = generation + 1
            population = [layout for island in islands for layout in island]
            scored: list[tuple[tuple, tuple[str, ...]]] = []
            if executor is None:
                for layout in population:
                    if self.cancel_event.is_set() or time.time() >= deadline:
                        break
                    result = self._evaluate(layout)
                    self._accept(result)
                    scored.append((result.score() if result else (-1,), layout))
            else:
                valid = [layout for layout in population if self._within_limits(layout)]
                futures = {
                    executor.submit(evaluate_layout, layout, self.request.columns, self.request.max_reactor_ticks): layout
                    for layout in valid
                }
                try:
                    for future in as_completed(futures, timeout=max(0.01, deadline - time.time())):
                        if self.cancel_event.is_set() or time.time() >= deadline:
                            break
                        result = future.result()
                        if mark_family(result.mark) not in self.request.marks:
                            result = None
                        self._accept(result)
                        scored.append((result.score() if result else (-1,), futures[future]))
                except TimeoutError:
                    pass
                for future in futures:
                    future.cancel()
            score_by_layout = {layout: score for score, layout in scored}
            next_islands: list[list[tuple[str, ...]]] = []
            leaders: list[tuple[str, ...]] = []
            for island in islands:
                island_scored = sorted(
                    ((score_by_layout.get(layout, (-1,)), layout) for layout in island),
                    reverse=True,
                    key=lambda item: item[0],
                )
                elite_count = max(2, len(island) // 5)
                elites = [layout for _, layout in island_scored[:elite_count]]
                leaders.append(elites[0])
                next_island = list(elites)
                while len(next_island) < len(island):
                    next_island.append(self._mutate(rng.choice(elites), rng))
                next_islands.append(next_island)
            # 每五代做环形迁移：各岛最佳个体替换下一岛的最后一个个体。
            if island_count > 1 and (generation + 1) % 5 == 0:
                for island, leader in enumerate(leaders):
                    next_islands[(island + 1) % island_count][-1] = leader
            islands = next_islands
            self.progress = min(0.999, (generation + 1) / self.request.generations)
            self.message = f"第 {generation + 1} 代 · {island_count} 个岛 · 已评估 {self.evaluated} 个布局"
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_exhaustive(self) -> None:
        estimate = self.exhaustive_estimate or 0
        total = max(1, estimate)
        shards = _exhaustive_shards(self.request)
        worker_count = min(self.request.cpu_workers, len(shards))
        request_data = self.request.model_dump(mode="json")
        shard_progress: dict[int, tuple[int, int, int]] = {}

        def update_progress(shard_id: int, checked: int, pruned: int, evaluated: int) -> None:
            old_checked, old_pruned, old_evaluated = shard_progress.get(shard_id, (0, 0, 0))
            self.checked += checked - old_checked
            self.pruned += pruned - old_pruned
            self.evaluated += evaluated - old_evaluated
            shard_progress[shard_id] = (checked, pruned, evaluated)
            self.progress = min(0.999, self.checked / total)
            self.message = (
                f"{worker_count} 进程并行枚举 · 已检查 {self.checked:,} 个方案"
                f" · 模拟 {self.evaluated:,} 个有标签布局"
            )

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        self.process_cancel_event = manager.Event()
        executor = ProcessPoolExecutor(max_workers=worker_count)
        futures = {
            executor.submit(
                _run_exhaustive_shard,
                request_data,
                shard_id,
                shard,
                progress_queue,
                self.process_cancel_event,
            ): shard_id
            for shard_id, shard in enumerate(shards)
        }
        pending = set(futures)

        def handle_message(message: tuple) -> None:
            if message[0] == "progress":
                _, shard_id, checked, pruned, evaluated = message
                update_progress(shard_id, checked, pruned, evaluated)
            elif message[0] == "candidate":
                self._accept(message[1], count_evaluation=False)

        try:
            while pending:
                if self.cancel_event.is_set():
                    self.process_cancel_event.set()
                    for future in pending:
                        future.cancel()
                try:
                    handle_message(progress_queue.get(timeout=0.2))
                except queue.Empty:
                    pass

                finished = [future for future in pending if future.done()]
                for future in finished:
                    pending.remove(future)
                    if future.cancelled():
                        continue
                    result = future.result()
                    update_progress(
                        result["shard_id"], result["checked"], result["pruned"], result["evaluated"]
                    )
                    for values in result["boards"].values():
                        for candidate in values:
                            self._accept(candidate, count_evaluation=False)

            while True:
                try:
                    handle_message(progress_queue.get_nowait())
                except queue.Empty:
                    break
            if not self.cancel_event.is_set():
                self.proven_global = True
        finally:
            if self.process_cancel_event is not None:
                self.process_cancel_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.process_cancel_event = None
            manager.shutdown()

    def cancel(self) -> None:
        self.cancel_event.set()
        if self.process_cancel_event is not None:
            self.process_cancel_event.set()

    def run(self) -> None:
        self.status = "running"
        self.started_at = time.time()
        try:
            if self.request.solver == "exhaustive":
                self._run_exhaustive()
            else:
                self._run_heuristic()
            if self.cancel_event.is_set():
                self.status = "cancelled"
                self.message = "优化已取消，保留当前候选"
            else:
                self.status = "completed"
                self.progress = 1.0
                self.message = "优化完成"
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
            self.message = "优化失败"
        finally:
            self.finished_at = time.time()


class OptimizationManager:
    def __init__(self):
        self.jobs: dict[str, OptimizationJob] = {}
        self.lock = threading.Lock()

    def create(self, request: OptimizationRequest) -> OptimizationJob:
        job = OptimizationJob(request)
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=job.run, name=f"optimizer-{job.id[:8]}", daemon=True).start()
        return job

    def get(self, job_id: str) -> OptimizationJob:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError("优化任务不存在") from exc

    def latest(self) -> OptimizationJob:
        with self.lock:
            if not self.jobs:
                raise KeyError("暂无优化任务")
            return next(reversed(self.jobs.values()))

    def resume(self, job_id: str) -> OptimizationJob:
        job = self.get(job_id)
        if job.request.solver != "heuristic":
            raise ValueError("穷举任务不能续算；请新建任务并完整枚举")
        if job.status in {"queued", "running"}:
            raise ValueError("任务仍在运行")
        job.cancel_event.clear()
        job.status = "queued"
        job.progress = 0.0
        job.proven_global = False
        job.message = "准备继续改进当前候选"
        threading.Thread(target=job.run, name=f"optimizer-{job.id[:8]}-resume", daemon=True).start()
        return job
