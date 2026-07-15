"""Anytime search and proof ledger built on the independent mathematical model.

This is deliberately a small reference implementation of the algorithmic
contract, not another copy of the production optimiser.  It has two rules:

* a safe repeated state may raise the lower bound;
* only a mathematical contradiction or an exhausted finite branch may lower
  the upper bound.  A failed local search is always recorded as ``open``.

Consequently every interrupted run returns a valid interval ``[lower, upper]``
instead of silently turning a timeout into a false global-optimality claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from .components import COMPONENTS
from .ic2_thermal_catalog import IC2_HEAT_FLOW_CATALOGUE
from .engine import ReactorSimulator
from .mathematical_model import (
    AnalyticalCutProof,
    ClosedFormBound,
    CycleCertificate,
    IC2CycleOracle,
    MasterSolution,
    PowerHeatMaster,
    ReactorProblem,
    closed_form_upper_bound,
    derive_ic2_top_tier_cut,
    evaluate_power_skeleton,
)
from .models import Layout
from .thermal_relaxation import layout_heat_flow_bound


PERMANENT_LAYOUT_COMPONENTS: tuple[str, ...] = tuple(
    item
    for item, spec in COMPONENTS.items()
    if spec.kind not in {"empty", "fuel", "reflector"}
)


@dataclass(frozen=True, slots=True)
class ThermalFitness:
    periodic: bool
    survived_ticks: int
    tail_growth: int
    total_heat: int
    maximum_ratio_ppm: int
    broken_slot: int | None
    transient: int | None
    period: int | None

    def ordering_key(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.periodic),
            self.survived_ticks,
            -max(0, self.tail_growth),
            -self.maximum_ratio_ppm,
            -self.total_heat,
        )


def evaluate_thermal_fitness(
    layout: Sequence[str],
    *,
    columns: int,
    horizon: int,
    time_limit_seconds: float | None = None,
) -> ThermalFitness:
    """Integer transition screening with exact cycle detection.

    This function supplies a *ranking*, never an infeasibility proof.  The
    final candidate is checked again by :class:`IC2CycleOracle`.
    """

    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive or None")
    started = perf_counter()
    deadline = (
        None if time_limit_seconds is None else started + time_limit_seconds
    )
    simulator = ReactorSimulator(Layout(columns=columns, slots=list(layout)))
    seen = {simulator.thermal_state_signature(): 0}
    midpoint_heat = 0
    maximum_ratio = 0.0
    total = simulator.hull_heat + sum(slot.heat for slot in simulator.slots)
    checked_ticks = 0
    for tick in range(1, horizon + 1):
        if deadline is not None and perf_counter() >= deadline:
            break
        previous_total = simulator.hull_heat + sum(slot.heat for slot in simulator.slots)
        simulator.step(auto_refuel=True)
        checked_ticks = tick
        total = simulator.hull_heat + sum(slot.heat for slot in simulator.slots)
        ratios = [
            slot.heat / slot.spec.max_heat
            for slot in simulator.slots
            if slot.spec.max_heat > 0
        ]
        maximum_ratio = max(
            maximum_ratio,
            max(
                [simulator.hull_heat / simulator.max_hull_heat, *ratios],
                default=0.0,
            ),
        )
        if tick == max(1, horizon // 2):
            midpoint_heat = total
        if (
            simulator.first_critical_tick is not None
            or simulator.first_component_break_tick is not None
            or simulator.meltdown_tick is not None
        ):
            broken = next(
                (event.slot for event in reversed(simulator.events) if event.slot is not None),
                None,
            )
            return ThermalFitness(
                periodic=False,
                survived_ticks=tick,
                tail_growth=max(0, previous_total - midpoint_heat) if midpoint_heat else previous_total,
                total_heat=previous_total,
                maximum_ratio_ppm=round(maximum_ratio * 1_000_000),
                broken_slot=broken,
                transient=None,
                period=None,
            )
        signature = simulator.thermal_state_signature()
        previous = seen.get(signature)
        if previous is not None:
            return ThermalFitness(
                periodic=True,
                survived_ticks=horizon,
                tail_growth=0,
                total_heat=total,
                maximum_ratio_ppm=round(maximum_ratio * 1_000_000),
                broken_slot=None,
                transient=previous,
                period=tick - previous,
            )
        seen[signature] = tick
    return ThermalFitness(
        periodic=False,
        survived_ticks=checked_ticks,
        tail_growth=max(0, total - midpoint_heat),
        total_heat=total,
        maximum_ratio_ppm=round(maximum_ratio * 1_000_000),
        broken_slot=None,
        transient=None,
        period=None,
    )


@dataclass(frozen=True, slots=True)
class CoolingSearchResult:
    layout: tuple[str, ...]
    fitness: ThermalFitness
    certificate: CycleCertificate | None
    evaluated: int
    flow_pruned: int
    elapsed_seconds: float


class IC2CoolingLNS:
    """Graph-parameterised large-neighbourhood search for one power skeleton."""

    def __init__(
        self,
        problem: ReactorProblem,
        *,
        allowed_components: Sequence[str] | None = None,
    ) -> None:
        graph = problem.graph
        if graph.rows != 6 or graph.columns is None:
            raise ValueError("the IC2 exact transition oracle requires a six-row rectangular graph")
        selected_components = (
            problem.layout_components
            if allowed_components is None
            else tuple(allowed_components)
        )
        unknown = set(selected_components) - COMPONENTS.keys()
        if unknown:
            raise ValueError(f"unknown layout components: {sorted(unknown)}")
        self.problem = problem
        self.allowed_components = tuple(selected_components)
        self.component_limits = dict(problem.component_limits)

    def _within_limits(self, layout: Sequence[str]) -> bool:
        if not self.component_limits:
            return True
        counts: dict[str, int] = {}
        for item in layout:
            counts[item] = counts.get(item, 0) + 1
        return all(
            limit is None or counts.get(item, 0) <= limit
            for item, limit in self.component_limits.items()
        )

    def _structured_seeds(self, skeleton: Sequence[str]) -> list[tuple[str, ...]]:
        free = [index for index, item in enumerate(skeleton) if item == "empty"]
        if not free:
            return [tuple(skeleton)]
        result: list[tuple[str, ...]] = []

        def make(selector) -> tuple[str, ...]:
            values = list(skeleton)
            for index in free:
                values[index] = selector(index)
            return tuple(values)

        if "overclocked_heat_vent" in self.allowed_components:
            result.append(make(lambda _index: "overclocked_heat_vent"))
        if {
            "overclocked_heat_vent",
            "component_heat_vent",
        }.issubset(self.allowed_components):
            # Two complementary parity patterns cover both sides of every
            # grid edge.  They are deterministic and work on any rectangle.
            columns = self.problem.graph.columns or 1
            for parity in (0, 1):
                result.append(make(
                    lambda index, parity=parity: (
                        "component_heat_vent"
                        if (sum(divmod(index, columns)) & 1) == parity
                        else "overclocked_heat_vent"
                    )
                ))
        if not result:
            first = self.allowed_components[0] if self.allowed_components else "empty"
            result.append(make(lambda _index: first))
        valid = [item for item in dict.fromkeys(result) if self._within_limits(item)]
        if valid:
            return valid

        # Inventory-aware deterministic fallback.  It favours persistent
        # cooling, then fills with any remaining enabled component/empty slot.
        priority = sorted(
            self.allowed_components,
            key=lambda item: (
                COMPONENTS[item].self_vent
                + COMPONENTS[item].side_vent * self.problem.graph.maximum_degree,
                COMPONENTS[item].max_heat,
            ),
            reverse=True,
        )
        values = list(skeleton)
        used: dict[str, int] = {}
        for index in free:
            selected = next(
                (
                    item
                    for item in priority
                    if self.component_limits.get(item) is None
                    or used.get(item, 0) < int(self.component_limits[item])
                ),
                "empty",
            )
            values[index] = selected
            used[selected] = used.get(selected, 0) + 1
        fallback = tuple(values)
        return [fallback] if self._within_limits(fallback) else []

    def _mutate(
        self,
        layout: tuple[str, ...],
        free: tuple[int, ...],
        score: ThermalFitness,
        rng: random.Random,
    ) -> tuple[str, ...]:
        values = list(layout)
        graph = self.problem.graph
        if score.broken_slot in free and rng.random() < 0.8:
            broken = int(score.broken_slot)
            adjacent_free = [value for value in graph.neighbours[broken] if value in free]
            if adjacent_free and "component_heat_vent" in self.allowed_components:
                values[rng.choice(adjacent_free)] = "component_heat_vent"
            if "overclocked_heat_vent" in self.allowed_components and rng.random() < 0.4:
                values[broken] = "overclocked_heat_vent"

        moves = 1 + int(rng.random() < 0.75) + rng.randrange(4) * int(rng.random() < 0.2)
        for _ in range(moves):
            if len(free) >= 2 and rng.random() < 0.55:
                first, second = rng.sample(free, 2)
                values[first], values[second] = values[second], values[first]
            elif free and self.allowed_components:
                values[rng.choice(free)] = rng.choice(self.allowed_components)
        candidate = tuple(values)
        return candidate if self._within_limits(candidate) else layout

    def search(
        self,
        skeleton: Sequence[str],
        *,
        seconds: float,
        horizon: int = 400,
        population: int = 64,
        seed: int = 221,
        initial_layouts: Iterable[Sequence[str]] = (),
    ) -> CoolingSearchResult:
        if seconds <= 0 or horizon <= 0 or population <= 0:
            raise ValueError("search limits must be positive")
        started = perf_counter()
        deadline = started + seconds
        metrics = evaluate_power_skeleton(self.problem, skeleton)
        if self.problem.exact_rods and metrics.rods != self.problem.rod_budget:
            raise ValueError("skeleton violates exact rod budget")
        free = tuple(index for index, item in enumerate(skeleton) if item == "empty")
        candidates = self._structured_seeds(skeleton)
        for raw in initial_layouts:
            layout = tuple(raw)
            if len(layout) != self.problem.graph.size:
                continue
            if all(
                skeleton[index] == "empty" or layout[index] == skeleton[index]
                for index in self.problem.graph.vertices
            ):
                if self._within_limits(layout):
                    candidates.append(layout)
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise ValueError("no cooling seed satisfies the enabled-component inventories")
        rng = random.Random(seed)
        cache: dict[tuple[str, ...], ThermalFitness] = {}
        flow_pruned = 0
        best_layout = candidates[0]
        best_score: ThermalFitness | None = None

        while perf_counter() < deadline:
            unseen = [item for item in candidates if item not in cache]
            if not unseen:
                unseen = [self._mutate(best_layout, free, best_score, rng)] if best_score else []
            for layout in unseen:
                if perf_counter() >= deadline:
                    break
                flow_bound = layout_heat_flow_bound(
                    self.problem,
                    layout,
                    IC2_HEAT_FLOW_CATALOGUE,
                )
                if not flow_bound.necessary_condition_satisfied:
                    flow_pruned += 1
                    score = ThermalFitness(
                        periodic=False,
                        survived_ticks=0,
                        tail_growth=flow_bound.deficit,
                        total_heat=flow_bound.generated_heat,
                        maximum_ratio_ppm=1_000_000,
                        broken_slot=None,
                        transient=None,
                        period=None,
                    )
                else:
                    remaining = deadline - perf_counter()
                    if remaining <= 0:
                        break
                    score = evaluate_thermal_fitness(
                        layout,
                        columns=self.problem.graph.columns or 0,
                        horizon=horizon,
                        time_limit_seconds=remaining,
                    )
                cache[layout] = score
                if best_score is None or score.ordering_key() > best_score.ordering_key():
                    best_layout, best_score = layout, score
                if score.periodic:
                    remaining = deadline - perf_counter()
                    if remaining <= 0:
                        break
                    certificate = IC2CycleOracle().verify(
                        layout,
                        columns=self.problem.graph.columns or 0,
                        max_ticks=max(100_000, horizon),
                        time_limit_seconds=remaining,
                    )
                    if certificate.safe:
                        return CoolingSearchResult(
                            layout=layout,
                            fitness=score,
                            certificate=certificate,
                            evaluated=len(cache),
                            flow_pruned=flow_pruned,
                            elapsed_seconds=perf_counter() - started,
                        )
            ranked = sorted(cache, key=lambda item: cache[item].ordering_key(), reverse=True)
            elites = ranked[: max(1, min(len(ranked), population // 8))]
            candidates = list(elites)
            while elites and len(candidates) < population:
                parent = rng.choice(elites)
                candidates.append(self._mutate(parent, free, cache[parent], rng))

        if best_score is None:
            best_score = ThermalFitness(
                periodic=False,
                survived_ticks=0,
                tail_growth=0,
                total_heat=0,
                maximum_ratio_ppm=0,
                broken_slot=None,
                transient=None,
                period=None,
            )
        return CoolingSearchResult(
            layout=best_layout,
            fitness=best_score,
            certificate=None,
            evaluated=len(cache),
            flow_pruned=flow_pruned,
            elapsed_seconds=perf_counter() - started,
        )


@dataclass(frozen=True, slots=True)
class PowerTierRecord:
    power: int
    static_status: str
    sampled_skeletons: int
    thermal_evaluations: int
    disposition: str
    reason: str


@dataclass(frozen=True, slots=True)
class CertifiedSearchReport:
    lower_bound: int
    upper_bound: int
    proven_global: bool
    best_layout: tuple[str, ...] | None
    best_cycle: CycleCertificate | None
    closed_form_upper_bound: int
    static_master_upper_bound: int
    analytical_cut_upper_bound: int
    analytical_cuts: tuple[int, ...]
    closed_form_proof: ClosedFormBound
    analytical_proof: AnalyticalCutProof | None
    static_master_status: str
    static_master_proven_optimal: bool
    elapsed_seconds: float
    tiers: tuple[PowerTierRecord, ...]
    open_power_tiers: tuple[int, ...]
    statement: str

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _extract_master_solution(problem: ReactorProblem, solver, variables, status_code) -> MasterSolution:
    from math import ceil
    from ortools.sat.python import cp_model

    status = solver.status_name(status_code)
    feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not feasible:
        return MasterSolution(
            status=status,
            feasible=False,
            proven_optimal=status_code == cp_model.OPTIMAL,
            power=None,
            generated_heat=None,
            active_cells=None,
            skeleton=None,
            strict_power_upper_bound=max(0, ceil(solver.best_objective_bound - 1e-9)),
            elapsed_seconds=0.0,
            conflicts=solver.num_conflicts,
            branches=solver.num_branches,
        )
    one_hot = variables["one_hot"]
    skeleton = tuple(
        problem.power_components[next(
            code
            for code, flag in enumerate(one_hot[vertex])
            if solver.value(flag)
        )].id
        for vertex in problem.graph.vertices
    )
    return MasterSolution(
        status=status,
        feasible=True,
        proven_optimal=status_code == cp_model.OPTIMAL,
        power=solver.value(variables["power"]),
        generated_heat=solver.value(variables["heat"]),
        active_cells=solver.value(variables["active_count"]),
        skeleton=skeleton,
        strict_power_upper_bound=ceil(solver.best_objective_bound - 1e-9),
        elapsed_seconds=0.0,
        conflicts=solver.num_conflicts,
        branches=solver.num_branches,
    )


def sample_skeletons_at_power(
    problem: ReactorProblem,
    *,
    power: int,
    limit: int,
    seconds: float,
    workers: int,
    seed: int,
) -> tuple[list[MasterSolution], str]:
    """Sample distinct exact-power skeletons; UNSAT is a proof, timeout is not."""

    from ortools.sat.python import cp_model

    if limit <= 0:
        return [], "sample_limit"
    master = PowerHeatMaster(problem)
    model, variables = master.build(exact_power=power, use_cooling_envelope=True)
    started = perf_counter()
    results: list[MasterSolution] = []
    final_status = "sample_limit"
    while len(results) < limit:
        remaining = seconds - (perf_counter() - started)
        if remaining <= 0:
            final_status = "timeout"
            break
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = seed + len(results)
        status_code = solver.solve(model)
        solution = _extract_master_solution(problem, solver, variables, status_code)
        if not solution.feasible:
            final_status = "exhausted" if status_code == cp_model.INFEASIBLE else "timeout"
            break
        results.append(solution)
        chosen = [
            variables["one_hot"][vertex][
                next(
                    code
                    for code, item in enumerate(problem.power_components)
                    if item.id == solution.skeleton[vertex]
                )
            ]
            for vertex in problem.graph.vertices
        ]
        model.add(sum(chosen) <= problem.graph.size - 1)
    return results, final_status


class CertifiedAnytimeSolver:
    """Time-bounded two-level solver with an auditable optimality gap."""

    def __init__(self, problem: ReactorProblem) -> None:
        self.problem = problem

    def solve(
        self,
        *,
        time_limit_seconds: float,
        workers: int = 1,
        seed: int = 221,
        known_layouts: Iterable[Sequence[str]] = (),
        skeletons_per_tier: int = 4,
        cooling_seconds_per_skeleton: float = 2.0,
        thermal_horizon: int = 400,
    ) -> CertifiedSearchReport:
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        graph = self.problem.graph
        if graph.rows != 6 or graph.columns is None:
            raise ValueError("the bundled exact IC2 oracle supports six-row rectangles")
        started = perf_counter()
        deadline = started + time_limit_seconds
        lower = 0
        best_layout: tuple[str, ...] | None = None
        best_cycle: CycleCertificate | None = None
        exact_oracle = IC2CycleOracle()

        for raw in known_layouts:
            if perf_counter() >= deadline:
                break
            layout = tuple(raw)
            if len(layout) != graph.size:
                continue
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            certificate = exact_oracle.verify(
                layout,
                columns=graph.columns,
                max_ticks=100_000,
                time_limit_seconds=remaining,
            )
            rods = sum(COMPONENTS[item].rod_count for item in layout)
            if certificate.safe and (
                rods == self.problem.rod_budget
                if self.problem.exact_rods
                else rods <= self.problem.rod_budget
            ) and certificate.power > lower:
                lower = certificate.power
                best_layout = layout
                best_cycle = certificate

        closed = closed_form_upper_bound(self.problem)
        top_tier_cut = derive_ic2_top_tier_cut(self.problem)
        remaining = deadline - perf_counter()
        if remaining > 0:
            root = PowerHeatMaster(self.problem).solve(
                seconds=min(60.0, remaining),
                workers=workers,
                random_seed=seed,
            )
            root_upper = (
                root.strict_power_upper_bound
                if root.feasible or root.status in {"OPTIMAL", "INFEASIBLE"}
                else closed.power_upper_bound
            )
            root_status = root.status
            root_proven_optimal = root.proven_optimal
        else:
            root_upper = closed.power_upper_bound
            root_status = "SKIPPED_TIME_LIMIT"
            root_proven_optimal = False
        static_upper = min(closed.power_upper_bound, root_upper)
        analytical_upper = min(
            static_upper,
            top_tier_cut.power_upper_bound if top_tier_cut is not None else static_upper,
        )
        upper = analytical_upper
        records: list[PowerTierRecord] = []
        open_tiers: list[int] = []
        cooling = IC2CoolingLNS(self.problem)

        quantum = self.problem.eu_per_pulse
        first_tier = upper - (upper % quantum)
        for power in range(first_tier, lower, -quantum):
            if power <= lower:
                break
            if perf_counter() >= deadline:
                open_tiers.extend(range(power, lower, -quantum))
                break
            remaining = deadline - perf_counter()
            samples, status = sample_skeletons_at_power(
                self.problem,
                power=power,
                limit=skeletons_per_tier,
                seconds=min(remaining, max(0.05, remaining * 0.2)),
                workers=workers,
                seed=seed + power,
            )
            thermal_count = 0
            witness_found = False
            for sample_index, sample in enumerate(samples):
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break
                result = cooling.search(
                    sample.skeleton or (),
                    seconds=min(cooling_seconds_per_skeleton, remaining),
                    horizon=thermal_horizon,
                    seed=seed + power * 31 + sample_index,
                )
                thermal_count += result.evaluated
                if result.certificate is not None and result.certificate.power > lower:
                    lower = result.certificate.power
                    best_layout = result.layout
                    best_cycle = result.certificate
                    witness_found = True
                    break
            if status == "exhausted" and not samples:
                disposition = "closed"
                reason = "static power/heat master proved UNSAT"
            else:
                disposition = "witness" if witness_found else "open"
                reason = (
                    "reachable safe cycle found"
                    if witness_found
                    else "sampled cooling search is not an infeasibility proof"
                )
                if power > lower:
                    open_tiers.append(power)
            records.append(PowerTierRecord(
                power=power,
                static_status=status,
                sampled_skeletons=len(samples),
                thermal_evaluations=thermal_count,
                disposition=disposition,
                reason=reason,
            ))

        unresolved = tuple(sorted(set(value for value in open_tiers if value > lower), reverse=True))
        if unresolved:
            upper = max(unresolved)
        else:
            upper = lower
        proven_global = upper == lower and best_cycle is not None
        statement = (
            f"global optimum certified at {lower} EU/t"
            if proven_global
            else f"certified interval: {lower} <= optimum <= {upper} EU/t; "
                 "open tiers are not claimed infeasible"
        )
        return CertifiedSearchReport(
            lower_bound=lower,
            upper_bound=upper,
            proven_global=proven_global,
            best_layout=best_layout,
            best_cycle=best_cycle,
            closed_form_upper_bound=closed.power_upper_bound,
            static_master_upper_bound=static_upper,
            analytical_cut_upper_bound=analytical_upper,
            analytical_cuts=(
                top_tier_cut.excluded_power_levels if top_tier_cut is not None else ()
            ),
            closed_form_proof=closed,
            analytical_proof=top_tier_cut,
            static_master_status=root_status,
            static_master_proven_optimal=root_proven_optimal,
            elapsed_seconds=perf_counter() - started,
            tiers=tuple(records),
            open_power_tiers=unresolved,
            statement=statement,
        )
