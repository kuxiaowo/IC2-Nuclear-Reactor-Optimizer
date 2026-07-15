"""Concrete Benders adapters for the full-label thermal master."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from time import perf_counter
from .cycle_proof import (
    DeterministicCycleSession,
    IC2TransitionState,
    IC2TransitionSystem,
    ReachableCycleProof,
)
from .factorized_cooling_master import FactorizedCoolingCutMaster
from .ic2_thermal_catalog import (
    IC2_HEAT_FLOW_CATALOGUE,
    IC2_PERIODIC_PREFIX_CATALOGUE,
)
from .logic_benders import (
    BendersCut,
    BendersDomain,
    MasterAnswer,
    MasterCandidate,
    MasterStatus,
    SubproblemAnswer,
    SubproblemStatus,
)
from .mathematical_model import AggregatePattern, ReactorProblem
from .proof_coordinator import ProofArtifact
from .periodic_prefix import PeriodicPrefixCutTemplate, periodic_prefix_flow_bound
from .thermal_master import ThermalCutMaster
from .thermal_relaxation import ThermalCutTemplate, layout_heat_flow_bound


@dataclass(frozen=True, slots=True)
class ThermalNoGood:
    layout: tuple[str, ...]


ThermalMasterCut = ThermalCutTemplate | PeriodicPrefixCutTemplate | ThermalNoGood


@dataclass(frozen=True, slots=True)
class CertifiedThermalLayout:
    layout: tuple[str, ...]
    cycle: ReachableCycleProof


def _layout_digest(prefix: str, layout: tuple[str, ...]) -> str:
    digest = sha256("\0".join(layout).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class ThermalLayoutMasterAdapter:
    """Expose :class:`ThermalCutMaster` through the generic Benders protocol."""

    def __init__(
        self,
        master: ThermalCutMaster,
        *,
        workers: int = 1,
        random_seed: int = 221,
        power_upper_bound: int | None = None,
        enforce_full_flow: bool = True,
        enforce_ordered_distribution_flow: bool = False,
        conditional_aggregate_patterns: dict[
            int, tuple[AggregatePattern, ...]
        ] | None = None,
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self.master = master
        self.workers = workers
        self.random_seed = random_seed
        self.power_upper_bound = power_upper_bound
        self.enforce_full_flow = enforce_full_flow
        self.enforce_ordered_distribution_flow = enforce_ordered_distribution_flow
        self.conditional_aggregate_patterns = (
            {} if conditional_aggregate_patterns is None
            else dict(conditional_aggregate_patterns)
        )

    def solve(
        self,
        domain: BendersDomain[object, ThermalMasterCut],
        incumbent_lower_bound: int,
        time_limit_seconds: float,
    ) -> MasterAnswer[tuple[str, ...], ThermalMasterCut]:
        thermal_cuts = []
        prefix_cuts = []
        excluded_layouts = []
        for cut in domain.cuts:
            if isinstance(cut.payload, ThermalCutTemplate):
                thermal_cuts.append(cut.payload)
            elif isinstance(cut.payload, PeriodicPrefixCutTemplate):
                prefix_cuts.append(cut.payload)
            elif isinstance(cut.payload, ThermalNoGood):
                excluded_layouts.append(cut.payload.layout)
            else:  # pragma: no cover - guarded by the public union
                raise TypeError(f"unsupported thermal master cut: {type(cut.payload)!r}")
        result = self.master.solve(
            cuts=thermal_cuts,
            prefix_cuts=prefix_cuts,
            excluded_layouts=excluded_layouts,
            enforce_full_flow=self.enforce_full_flow,
            enforce_ordered_distribution_flow=self.enforce_ordered_distribution_flow,
            seconds=max(1e-6, time_limit_seconds),
            workers=self.workers,
            random_seed=self.random_seed + domain.iteration,
            minimum_power=incumbent_lower_bound + 1,
            maximum_power_limit=self.power_upper_bound,
            conditional_aggregate_patterns=self.conditional_aggregate_patterns,
        )
        if result.status == "INFEASIBLE":
            return MasterAnswer(
                MasterStatus.INFEASIBLE,
                proof=ProofArtifact(
                    kind="cp_sat_domain_bound",
                    statement=(
                        "the exact full-label master proves that this domain "
                        "contains no candidate above the certified incumbent"
                    ),
                    data=(
                        ("incumbent", incumbent_lower_bound),
                        ("thermal_cuts", len(thermal_cuts)),
                        ("prefix_cuts", len(prefix_cuts)),
                        ("no_goods", len(excluded_layouts)),
                    ),
                ),
            )
        if not result.feasible or result.layout is None or result.power is None:
            return MasterAnswer(
                MasterStatus.UNKNOWN,
                proof=ProofArtifact(
                    kind="cp_sat_partial_bound",
                    statement="the full-label master timed out with a rigorous dual bound",
                    data=(
                        ("status", result.status),
                        ("dual_upper_bound", result.strict_power_upper_bound),
                    ),
                ),
                strict_upper_bound=result.strict_power_upper_bound,
            )
        layout = result.layout
        no_good_proof = ProofArtifact(
            kind="exact_layout_no_good",
            statement="exclude exactly the checked full component layout",
            data=(("layout_digest", _layout_digest("layout", layout)),),
        )
        no_good = BendersCut[ThermalMasterCut](
            cut_id=_layout_digest("nogood", layout),
            payload=ThermalNoGood(layout),
            proof=no_good_proof,
        )
        return MasterAnswer(
            MasterStatus.CANDIDATE,
            candidate=MasterCandidate(
                objective_value=result.power,
                payload=layout,
                exact_no_good=no_good,
                residual_upper_bound=result.strict_power_upper_bound,
            ),
        )


class FutureQuotientLayoutMasterAdapter:
    """Expose the one-level full-label future quotient as a Benders master."""

    def __init__(
        self,
        master: FactorizedCoolingCutMaster,
        *,
        power_upper_bound: int | None = None,
    ) -> None:
        self.master = master
        self.power_upper_bound = power_upper_bound

    def solve(
        self,
        domain: BendersDomain[object, ThermalMasterCut],
        incumbent_lower_bound: int,
        time_limit_seconds: float,
    ) -> MasterAnswer[tuple[str, ...], ThermalMasterCut]:
        thermal_cuts = []
        prefix_cuts = []
        excluded_layouts = []
        for cut in domain.cuts:
            if isinstance(cut.payload, ThermalCutTemplate):
                thermal_cuts.append(cut.payload)
            elif isinstance(cut.payload, PeriodicPrefixCutTemplate):
                prefix_cuts.append(cut.payload)
            elif isinstance(cut.payload, ThermalNoGood):
                excluded_layouts.append(cut.payload.layout)
            else:  # pragma: no cover - guarded by the public union
                raise TypeError(f"unsupported quotient master cut: {type(cut.payload)!r}")

        result = self.master.solve_joint_layouts(
            average_cuts=thermal_cuts,
            prefix_cuts=prefix_cuts,
            excluded_layouts=excluded_layouts,
            incumbent_lower_bound=incumbent_lower_bound,
            time_limit_seconds=max(1e-6, time_limit_seconds),
        )
        if not result.proven:
            return MasterAnswer(
                MasterStatus.UNKNOWN,
                proof=ProofArtifact(
                    kind="future_quotient_partial_bound",
                    statement=(
                        "the exact full-label future quotient did not finish; "
                        "no unfinished state is treated as infeasible"
                    ),
                    data=(
                        ("stop_reason", result.stop_reason),
                        ("completed_layers", max(0, len(result.layer_statistics) - 1)),
                        ("raw_transitions", result.raw_transitions),
                    ),
                ),
                strict_upper_bound=self.power_upper_bound,
            )
        if not result.frontier:
            return MasterAnswer(
                MasterStatus.INFEASIBLE,
                proof=ProofArtifact(
                    kind="future_quotient_domain_bound",
                    statement=(
                        "the exact full-label quotient contains no layout above "
                        "the certified incumbent satisfying every submitted cut"
                    ),
                    data=(
                        ("incumbent", incumbent_lower_bound),
                        ("thermal_cuts", len(thermal_cuts)),
                        ("prefix_cuts", len(prefix_cuts)),
                        ("no_goods", len(excluded_layouts)),
                        ("raw_transitions", result.raw_transitions),
                    ),
                ),
            )

        representative = min(
            result.frontier,
            key=lambda point: (-point.power, point.generated_heat, point.skeleton),
        )
        layout = representative.skeleton
        no_good_proof = ProofArtifact(
            kind="exact_layout_no_good",
            statement="exclude exactly the checked full component layout",
            data=(("layout_digest", _layout_digest("layout", layout)),),
        )
        no_good = BendersCut[ThermalMasterCut](
            cut_id=_layout_digest("nogood", layout),
            payload=ThermalNoGood(layout),
            proof=no_good_proof,
        )
        return MasterAnswer(
            MasterStatus.CANDIDATE,
            candidate=MasterCandidate(
                objective_value=representative.power,
                payload=layout,
                exact_no_good=no_good,
                # A completed exact master proves that no residual layout can
                # exceed its maximum returned power.  Equal-power alternatives
                # remain open and appear after the no-good rebuild.
                residual_upper_bound=representative.power,
            ),
        )


class IC2ThermalSubproblem:
    """Use a flow cut first, then an exact reachable-cycle check for IC2."""

    def __init__(
        self,
        problem: ReactorProblem,
        *,
        max_steps: int = 100_000,
        max_open_sessions: int = 8,
    ) -> None:
        if problem.graph.rows != 6 or problem.graph.columns is None:
            raise ValueError("the locked IC2 transition adapter requires a six-row grid")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_open_sessions <= 0:
            raise ValueError("max_open_sessions must be positive")
        self.problem = problem
        self.max_steps = max_steps
        self.max_open_sessions = max_open_sessions
        self._conclusive_cache: dict[
            tuple[str, ...],
            SubproblemAnswer[ThermalMasterCut, CertifiedThermalLayout],
        ] = {}
        self._cache_lock = Lock()
        self.cache_hits = 0
        self.session_evictions = 0
        self._cycle_sessions: OrderedDict[
            tuple[str, ...],
            DeterministicCycleSession[tuple[str, ...], IC2TransitionState],
        ] = OrderedDict()

    @property
    def cache_entries(self) -> int:
        with self._cache_lock:
            return len(self._conclusive_cache)

    @property
    def open_cycle_sessions(self) -> int:
        with self._cache_lock:
            return len(self._cycle_sessions)

    def cycle_session_checked_steps(
        self,
        candidate: tuple[str, ...],
    ) -> int | None:
        """Return retained exact-prefix work, or ``None`` after eviction."""

        with self._cache_lock:
            session = self._cycle_sessions.get(candidate)
        return None if session is None else session.progress_steps

    def _cached(
        self,
        candidate: tuple[str, ...],
    ) -> SubproblemAnswer[ThermalMasterCut, CertifiedThermalLayout] | None:
        with self._cache_lock:
            answer = self._conclusive_cache.get(candidate)
            if answer is not None:
                self.cache_hits += 1
            return answer

    def _remember(
        self,
        candidate: tuple[str, ...],
        answer: SubproblemAnswer[ThermalMasterCut, CertifiedThermalLayout],
    ) -> SubproblemAnswer[ThermalMasterCut, CertifiedThermalLayout]:
        if answer.status == SubproblemStatus.UNKNOWN:
            return answer
        with self._cache_lock:
            self._conclusive_cache.setdefault(candidate, answer)
            self._cycle_sessions.pop(candidate, None)
            return self._conclusive_cache[candidate]

    def check(
        self,
        candidate: tuple[str, ...],
        time_limit_seconds: float,
    ) -> SubproblemAnswer[ThermalMasterCut, CertifiedThermalLayout]:
        if cached := self._cached(candidate):
            return cached
        started = perf_counter()
        with self._cache_lock:
            session = self._cycle_sessions.get(candidate)
            if session is not None:
                self._cycle_sessions.move_to_end(candidate)
        if session is None:
            flow = layout_heat_flow_bound(
                self.problem,
                candidate,
                IC2_HEAT_FLOW_CATALOGUE,
            )
            if not flow.necessary_condition_satisfied:
                proof = ProofArtifact(
                    kind="optimistic_thermal_min_cut",
                    statement=(
                        "even the optimistic heat-flow network cannot carry all "
                        "generated heat across its minimum cut"
                    ),
                    data=(
                        ("generated_heat", flow.generated_heat),
                        ("maximum_flow", flow.maximum_removable_heat),
                        ("deficit", flow.deficit),
                    ),
                )
                cut = BendersCut[ThermalMasterCut](
                    cut_id=_layout_digest("thermal_cut", candidate),
                    payload=flow.cut_template,
                    proof=proof,
                )
                return self._remember(candidate, SubproblemAnswer(
                    SubproblemStatus.INFEASIBLE,
                    proof=proof,
                    generalized_cut=cut,
                ))

            prefix = periodic_prefix_flow_bound(
                self.problem,
                candidate,
                IC2_PERIODIC_PREFIX_CATALOGUE,
                base_hull_capacity=10_000,
            )
            if not prefix.feasible:
                proof = ProofArtifact(
                    kind="periodic_prefix_min_cut",
                    statement=(
                        "no integer cyclic flow can satisfy ordered-event storage "
                        "capacities and the exact constant fuel-heat injections; "
                        "therefore every true periodic trajectory has positive "
                        "drift or a prefix overflow"
                    ),
                    data=(
                        ("generated_heat", prefix.generated_heat),
                        ("required_circulation", prefix.required_circulation),
                        ("routed_circulation", prefix.routed_circulation),
                        ("deficit", prefix.deficit),
                        ("time_expanded_nodes", prefix.node_count),
                        ("time_expanded_edges", prefix.edge_count),
                    ),
                )
                generalized = BendersCut[ThermalMasterCut](
                    cut_id=_layout_digest("periodic_prefix_cut", candidate),
                    payload=prefix.cut_template,
                    proof=proof,
                )
                return self._remember(candidate, SubproblemAnswer(
                    SubproblemStatus.INFEASIBLE,
                    proof=proof,
                    generalized_cut=generalized,
                ))
            # A retained session is also the certificate that both necessary
            # filters have already passed.  Creation is double-checked because
            # another worker may have checked the same layout concurrently.
            with self._cache_lock:
                session = self._cycle_sessions.get(candidate)
                if session is None:
                    session = DeterministicCycleSession(
                        IC2TransitionSystem(self.problem.graph.columns),
                        candidate,
                    )
                    self._cycle_sessions[candidate] = session
                    if len(self._cycle_sessions) > self.max_open_sessions:
                        self._cycle_sessions.popitem(last=False)
                        self.session_evictions += 1
                else:
                    self._cycle_sessions.move_to_end(candidate)

        remaining_time = time_limit_seconds - (perf_counter() - started)
        if remaining_time <= 0:
            return SubproblemAnswer(SubproblemStatus.UNKNOWN)
        assert session is not None
        cycle = session.advance(
            self.max_steps,
            time_limit_seconds=max(1e-6, remaining_time),
        )
        if cycle.safe:
            proof = ProofArtifact(
                kind="reachable_safe_cycle",
                statement=(
                    "an exact integer state reachable from the prescribed initial "
                    "state repeats without failure"
                ),
                data=(
                    ("transient", cycle.transient_length),
                    ("period", cycle.period_length),
                    ("checked_steps", cycle.checked_steps),
                ),
            )
            return self._remember(candidate, SubproblemAnswer(
                SubproblemStatus.FEASIBLE,
                witness_payload=CertifiedThermalLayout(candidate, cycle),
                proof=proof,
            ))
        if cycle.outcome == "failed":
            metrics = dict(cycle.last_metrics)
            proof = ProofArtifact(
                kind="exact_failed_trajectory",
                statement="the exact trajectory reaches a forbidden state",
                data=(
                    ("failure_step", cycle.failure_step),
                    ("failure_reason", cycle.failure_reason),
                    ("failure_component", metrics.get("failure_component")),
                    ("peak_hull_heat", metrics.get("peak_hull_heat")),
                    ("peak_component_heat", metrics.get("peak_component_heat")),
                ),
            )
            no_good = BendersCut[ThermalMasterCut](
                cut_id=_layout_digest("failed_layout", candidate),
                payload=ThermalNoGood(candidate),
                proof=proof,
            )
            return self._remember(candidate, SubproblemAnswer(
                SubproblemStatus.INFEASIBLE,
                proof=proof,
                generalized_cut=no_good,
            ))
        return SubproblemAnswer(SubproblemStatus.UNKNOWN)
