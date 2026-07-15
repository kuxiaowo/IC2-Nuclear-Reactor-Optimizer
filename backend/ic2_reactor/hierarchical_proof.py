"""Hierarchical skeleton/cooling/dynamics proof search with honest open units."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from .factorized_cooling_master import FactorizedCoolingCutMaster
from .ic2_thermal_catalog import (
    IC2_HEAT_FLOW_CATALOGUE,
    IC2_PERIODIC_PREFIX_CATALOGUE,
)
from .logic_benders import SubproblemStatus
from .mathematical_model import AggregatePattern, ReactorProblem
from .periodic_prefix import PeriodicPrefixCutTemplate
from .skeleton_thermal_master import IdealSkeletonThermalMaster
from .thermal_benders import (
    CertifiedThermalLayout,
    IC2ThermalSubproblem,
    ThermalNoGood,
)
from .thermal_master import PowerSkeletonNoGood, ThermalCutMaster
from .thermal_relaxation import ThermalCutTemplate


@dataclass(frozen=True, slots=True)
class HierarchicalPatternReport:
    power: int
    status: str
    proven_closed: bool
    witness: CertifiedThermalLayout | None
    skeleton_candidates: int
    skeletons_closed: int
    layouts_checked: int
    average_flow_cuts: int
    periodic_prefix_cuts: int
    exact_layout_no_goods: int
    power_skeleton_cores: int
    power_skeleton_core_sizes: tuple[int, ...]
    open_skeletons: tuple[tuple[str, ...], ...]
    open_layouts: tuple[tuple[str, ...], ...]
    elapsed_seconds: float
    stop_reason: str


class IC2HierarchicalPatternProver:
    """Resolve one aggregate signature without enumerating cooling completions.

    The ideal skeleton master is a superset of every real completion.  Each
    returned skeleton is checked by a fixed-skeleton actual-label master; an
    actual candidate then enters the deterministic thermal subproblem.  Every
    UNKNOWN singleton is recorded and excluded only from the residual branch,
    so search can continue without ever treating a timeout as infeasibility.
    """

    def __init__(
        self,
        problem: ReactorProblem,
        *,
        workers: int = 1,
        max_cycle_steps: int = 100_000,
        random_seed: int = 221,
        cooling_master_backend: Literal["cp_sat", "factorized"] = "cp_sat",
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        if max_cycle_steps <= 0:
            raise ValueError("max_cycle_steps must be positive")
        if cooling_master_backend not in {"cp_sat", "factorized"}:
            raise ValueError("unknown cooling master backend")
        self.problem = problem
        self.workers = workers
        self.random_seed = random_seed
        self.cooling_master_backend = cooling_master_backend
        self.ideal = IdealSkeletonThermalMaster(
            problem,
            IC2_HEAT_FLOW_CATALOGUE,
            prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
            base_hull_capacity=10_000,
        )
        self.actual = ThermalCutMaster(
            problem,
            IC2_HEAT_FLOW_CATALOGUE,
            prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
            base_hull_capacity=10_000,
        )
        self.factorized_actual = FactorizedCoolingCutMaster(
            problem,
            IC2_HEAT_FLOW_CATALOGUE,
            prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
            base_hull_capacity=10_000,
        )
        self.subproblem = IC2ThermalSubproblem(
            problem,
            max_steps=max_cycle_steps,
        )

    def prove(
        self,
        pattern: AggregatePattern,
        *,
        power: int,
        time_limit_seconds: float,
        master_unit_seconds: float = 2.0,
        subproblem_unit_seconds: float = 2.0,
        core_extraction_seconds: float = 1.0,
        max_skeletons: int | None = None,
        max_layouts_per_skeleton: int | None = None,
        project_prefix_cuts_to_ideal: bool = False,
    ) -> HierarchicalPatternReport:
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if master_unit_seconds <= 0 or subproblem_unit_seconds <= 0:
            raise ValueError("unit time limits must be positive")
        if core_extraction_seconds < 0:
            raise ValueError("core_extraction_seconds must be non-negative")
        if max_skeletons is not None and max_skeletons <= 0:
            raise ValueError("max_skeletons must be positive or None")
        if max_layouts_per_skeleton is not None and max_layouts_per_skeleton <= 0:
            raise ValueError("max_layouts_per_skeleton must be positive or None")

        started = perf_counter()
        deadline = started + time_limit_seconds
        counts = {
            (item, degree): count
            for item, degree, count in pattern.fuel_degree_counts
        }
        excluded_skeletons: list[tuple[str, ...]] = []
        skeleton_cores: list[PowerSkeletonNoGood] = []
        average_cuts: list[ThermalCutTemplate] = []
        prefix_cuts: list[PeriodicPrefixCutTemplate] = []
        excluded_layouts: list[tuple[str, ...]] = []
        open_skeletons: list[tuple[str, ...]] = []
        open_layouts: list[tuple[str, ...]] = []
        skeleton_candidates = skeletons_closed = layouts_checked = 0
        exact_no_goods = 0
        stop_reason = "ideal_domain_exhausted"

        def remaining() -> float:
            return deadline - perf_counter()

        while remaining() > 0:
            if max_skeletons is not None and skeleton_candidates >= max_skeletons:
                stop_reason = "skeleton_limit"
                break
            skeleton_answer = self.ideal.solve(
                excluded_skeletons=excluded_skeletons,
                excluded_power_cores=skeleton_cores,
                prefix_cuts=(prefix_cuts if project_prefix_cuts_to_ideal else ()),
                exact_power=power,
                aggregate_fuel_degree_counts=counts,
                exact_active_cells=pattern.active_cells,
                seconds=min(master_unit_seconds, remaining()),
                workers=self.workers,
                random_seed=self.random_seed + skeleton_candidates,
            )
            if skeleton_answer.status == "INFEASIBLE":
                proven_closed = not open_skeletons and not open_layouts
                return HierarchicalPatternReport(
                    power=power,
                    status="closed" if proven_closed else "unknown",
                    proven_closed=proven_closed,
                    witness=None,
                    skeleton_candidates=skeleton_candidates,
                    skeletons_closed=skeletons_closed,
                    layouts_checked=layouts_checked,
                    average_flow_cuts=len(average_cuts),
                    periodic_prefix_cuts=len(prefix_cuts),
                    exact_layout_no_goods=exact_no_goods,
                    power_skeleton_cores=len(skeleton_cores),
                    power_skeleton_core_sizes=tuple(
                        len(core.assignments) for core in skeleton_cores
                    ),
                    open_skeletons=tuple(open_skeletons),
                    open_layouts=tuple(open_layouts),
                    elapsed_seconds=perf_counter() - started,
                    stop_reason=(
                        "ideal_domain_exhausted"
                        if proven_closed
                        else "residual_exhausted_with_unknown_singletons"
                    ),
                )
            if not skeleton_answer.feasible or skeleton_answer.layout is None:
                stop_reason = "ideal_master_unknown"
                break
            skeleton = skeleton_answer.layout
            skeleton_candidates += 1
            layouts_for_skeleton = 0
            skeleton_has_unknown = False

            while remaining() > 0:
                if (
                    max_layouts_per_skeleton is not None
                    and layouts_for_skeleton >= max_layouts_per_skeleton
                ):
                    skeleton_has_unknown = True
                    stop_reason = "layout_limit"
                    break
                actual_answer = None
                if self.cooling_master_backend == "factorized":
                    factorized_answer = self.factorized_actual.solve(
                        skeleton,
                        average_cuts=average_cuts,
                        prefix_cuts=prefix_cuts,
                        excluded_layouts=excluded_layouts,
                        time_limit_seconds=min(master_unit_seconds, remaining()),
                    )
                    actual_status = (
                        "FEASIBLE"
                        if factorized_answer.proven and factorized_answer.feasible
                        else (
                            "INFEASIBLE"
                            if factorized_answer.proven
                            else "UNKNOWN"
                        )
                    )
                    actual_layout = factorized_answer.layout
                else:
                    actual_answer = self.actual.solve(
                        cuts=average_cuts,
                        prefix_cuts=prefix_cuts,
                        excluded_layouts=excluded_layouts,
                        fixed_power_skeleton=skeleton,
                        excluded_power_cores=skeleton_cores,
                        exact_power=power,
                        aggregate_fuel_degree_counts=counts,
                        exact_active_cells=pattern.active_cells,
                        # The compact free-source flow represents every ordinary
                        # average min-cut at once.  Exact ordered fuel injection,
                        # finite prefix capacity and dynamics remain separated so
                        # their failures return reusable Hoffman cuts.
                        enforce_full_flow=True,
                        seconds=min(master_unit_seconds, remaining()),
                        workers=self.workers,
                        random_seed=self.random_seed + layouts_checked,
                    )
                    if (
                        not actual_answer.feasible
                        and actual_answer.status != "INFEASIBLE"
                        and prefix_cuts
                        and remaining() > 0
                    ):
                        # A large explicit Hoffman inequality can propagate less
                        # effectively than the compact ordered-flow extension.
                        actual_answer = self.actual.solve(
                            cuts=average_cuts,
                            excluded_layouts=excluded_layouts,
                            fixed_power_skeleton=skeleton,
                            excluded_power_cores=skeleton_cores,
                            exact_power=power,
                            aggregate_fuel_degree_counts=counts,
                            exact_active_cells=pattern.active_cells,
                            enforce_ordered_distribution_flow=True,
                            seconds=min(master_unit_seconds, remaining()),
                            workers=self.workers,
                            random_seed=self.random_seed + layouts_checked,
                        )
                    actual_status = actual_answer.status
                    actual_layout = actual_answer.layout

                if actual_status == "INFEASIBLE":
                    if skeleton_has_unknown:
                        open_skeletons.append(skeleton)
                    else:
                        if (
                            self.cooling_master_backend == "cp_sat"
                            and core_extraction_seconds > 0
                            and remaining() > 0
                        ):
                            core_answer = self.actual.solve(
                                cuts=average_cuts,
                                prefix_cuts=prefix_cuts,
                                excluded_layouts=excluded_layouts,
                                fixed_power_skeleton=skeleton,
                                excluded_power_cores=skeleton_cores,
                                extract_fixed_skeleton_core=True,
                                infer_empty_skeleton_from_active_count=True,
                                exact_power=power,
                                aggregate_fuel_degree_counts=counts,
                                exact_active_cells=pattern.active_cells,
                                enforce_full_flow=True,
                                seconds=min(core_extraction_seconds, remaining()),
                                workers=self.workers,
                                random_seed=self.random_seed + layouts_checked,
                            )
                            if core_answer.status == "INFEASIBLE":
                                skeleton_cores.append(PowerSkeletonNoGood(
                                    core_answer.fixed_skeleton_core
                                ))
                        skeletons_closed += 1
                    break
                if actual_status != "FEASIBLE" or actual_layout is None:
                    skeleton_has_unknown = True
                    open_skeletons.append(skeleton)
                    stop_reason = "actual_master_unknown"
                    break

                layout = actual_layout
                layouts_checked += 1
                layouts_for_skeleton += 1
                if remaining() <= 0:
                    open_layouts.append(layout)
                    excluded_layouts.append(layout)
                    skeleton_has_unknown = True
                    break
                answer = self.subproblem.check(
                    layout,
                    min(subproblem_unit_seconds, remaining()),
                )
                if answer.status == SubproblemStatus.FEASIBLE:
                    assert answer.witness_payload is not None
                    return HierarchicalPatternReport(
                        power=power,
                        status="witness",
                        proven_closed=False,
                        witness=answer.witness_payload,
                        skeleton_candidates=skeleton_candidates,
                        skeletons_closed=skeletons_closed,
                        layouts_checked=layouts_checked,
                        average_flow_cuts=len(average_cuts),
                        periodic_prefix_cuts=len(prefix_cuts),
                        exact_layout_no_goods=exact_no_goods,
                        power_skeleton_cores=len(skeleton_cores),
                        power_skeleton_core_sizes=tuple(
                            len(core.assignments) for core in skeleton_cores
                        ),
                        open_skeletons=tuple(open_skeletons),
                        open_layouts=tuple(open_layouts),
                        elapsed_seconds=perf_counter() - started,
                        stop_reason="certified_witness",
                    )
                if answer.status == SubproblemStatus.UNKNOWN:
                    open_layouts.append(layout)
                    excluded_layouts.append(layout)
                    skeleton_has_unknown = True
                    continue
                cut = answer.generalized_cut
                if cut is None:
                    raise ValueError("infeasible thermal subproblem omitted its cut")
                if isinstance(cut.payload, ThermalCutTemplate):
                    if cut.payload not in average_cuts:
                        average_cuts.append(cut.payload)
                elif isinstance(cut.payload, PeriodicPrefixCutTemplate):
                    if cut.payload not in prefix_cuts:
                        prefix_cuts.append(cut.payload)
                elif isinstance(cut.payload, ThermalNoGood):
                    excluded_layouts.append(cut.payload.layout)
                    exact_no_goods += 1
                else:  # pragma: no cover - public union exhaustiveness
                    raise TypeError(f"unsupported hierarchical cut: {type(cut.payload)!r}")

            if (skeleton_has_unknown or remaining() <= 0) and skeleton not in open_skeletons:
                open_skeletons.append(skeleton)
            excluded_skeletons.append(skeleton)
            if stop_reason in {"actual_master_unknown", "layout_limit"}:
                # The unresolved singleton is retained above; continue with
                # the exact residual ideal-skeleton domain while time remains.
                stop_reason = "continuing_after_unknown_singleton"

        if remaining() <= 0:
            stop_reason = "time_limit"
        return HierarchicalPatternReport(
            power=power,
            status="unknown",
            proven_closed=False,
            witness=None,
            skeleton_candidates=skeleton_candidates,
            skeletons_closed=skeletons_closed,
            layouts_checked=layouts_checked,
            average_flow_cuts=len(average_cuts),
            periodic_prefix_cuts=len(prefix_cuts),
            exact_layout_no_goods=exact_no_goods,
            power_skeleton_cores=len(skeleton_cores),
            power_skeleton_core_sizes=tuple(
                len(core.assignments) for core in skeleton_cores
            ),
            open_skeletons=tuple(open_skeletons),
            open_layouts=tuple(open_layouts),
            elapsed_seconds=perf_counter() - started,
            stop_reason=stop_reason,
        )
