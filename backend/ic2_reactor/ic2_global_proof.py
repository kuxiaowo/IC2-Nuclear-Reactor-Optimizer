"""End-to-end certified IC2 optimisation built from the generic proof layers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Sequence

from .factorized_cooling_master import FactorizedCoolingCutMaster
from .ic2_thermal_catalog import (
    IC2_HEAT_FLOW_CATALOGUE,
    IC2_PERIODIC_PREFIX_CATALOGUE,
)
from .logic_benders import BendersDomain, LogicBendersProcessor
from .mathematical_model import (
    AggregatePattern,
    AnalyticalCutProof,
    ReactorProblem,
    aggregate_overload_analysis,
    closed_form_upper_bound,
    derive_ic2_top_tier_cut,
    evaluate_power_skeleton,
    route_conditioned_overload_analysis,
)
from .proof_coordinator import (
    CertifiedWitness,
    ProofCoordinator,
    ProofSearchReport,
    SearchNode,
)
from .proof_complexity import ProofWorkEnvelope, proof_work_envelope
from .thermal_benders import (
    CertifiedThermalLayout,
    IC2ThermalSubproblem,
    FutureQuotientLayoutMasterAdapter,
    ThermalLayoutMasterAdapter,
    ThermalMasterCut,
)
from .thermal_master import ThermalCutMaster


@dataclass(frozen=True, slots=True)
class IC2GlobalProofResult:
    report: ProofSearchReport[
        BendersDomain[str, ThermalMasterCut],
        CertifiedThermalLayout,
    ]
    analytical_proof: AnalyticalCutProof | None
    initial_upper_bound: int
    work_envelope: ProofWorkEnvelope
    conditional_aggregate_pattern_counts: tuple[tuple[int, int], ...]
    elapsed_seconds: float


def solve_ic2_global(
    problem: ReactorProblem,
    *,
    time_limit_seconds: float,
    workers: int = 1,
    known_layouts: Sequence[Sequence[str]] = (),
    max_cycle_steps: int = 100_000,
    master_fraction: float = 0.7,
    master_backend: Literal["cp_sat", "future_quotient"] = "cp_sat",
) -> IC2GlobalProofResult:
    """Run the certified full-label Benders algorithm under one wall deadline."""

    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if master_backend not in {"cp_sat", "future_quotient"}:
        raise ValueError("unknown global proof master backend")
    started = perf_counter()
    deadline = started + time_limit_seconds
    subproblem = IC2ThermalSubproblem(problem, max_steps=max_cycle_steps)
    best_known: CertifiedWitness[CertifiedThermalLayout] | None = None
    power_ids = {item.id for item in problem.power_components}

    for raw_layout in known_layouts:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        layout = tuple(raw_layout)
        answer = subproblem.check(layout, remaining)
        if answer.witness_payload is None or answer.proof is None:
            continue
        skeleton = tuple(label if label in power_ids else "empty" for label in layout)
        objective = evaluate_power_skeleton(problem, skeleton).power
        witness = CertifiedWitness(
            objective_value=objective,
            payload=answer.witness_payload,
            proof=answer.proof,
        )
        if best_known is None or witness.objective_value > best_known.objective_value:
            best_known = witness

    # The aggregate top-tier derivation is cheap relative to a six-hour run,
    # but not relative to a seconds-long API call.  Preserve the wall budget
    # and the certified incumbent: when less than ten seconds remain, keep the
    # weaker closed-form upper bound and skip optional preprocessing.
    analytical = (
        derive_ic2_top_tier_cut(problem)
        if deadline - perf_counter() >= 10.0
        else None
    )
    closed_form_upper = closed_form_upper_bound(problem).power_upper_bound
    # The CP-SAT master receives the instance-specific analytical tier cuts.
    # The future-quotient backend currently compiles only catalogue-local
    # Benders factors, so its root bound remains the sound closed-form static
    # envelope until those analytical tier exclusions are compiled as an
    # automaton as well.
    initial_upper = (
        analytical.power_upper_bound
        if analytical is not None and master_backend == "cp_sat"
        else closed_form_upper
    )
    conditional_patterns: dict[int, tuple[AggregatePattern, ...]] = {}
    if (
        analytical is not None
        and analytical.power_upper_bound == 455
        and 460 in analytical.excluded_power_levels
        and deadline - perf_counter() >= 5.0
    ):
        aggregate_455 = aggregate_overload_analysis(problem, 455)
        conditional_patterns[455] = tuple(
            pattern
            for pattern in aggregate_455.structurally_surviving_patterns
            if not route_conditioned_overload_analysis(problem, pattern).excluded
        )

    if master_backend == "future_quotient":
        master = FutureQuotientLayoutMasterAdapter(
            FactorizedCoolingCutMaster(
                problem,
                IC2_HEAT_FLOW_CATALOGUE,
                prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
                base_hull_capacity=10_000,
            ),
            power_upper_bound=initial_upper,
        )
    else:
        master = ThermalLayoutMasterAdapter(
            ThermalCutMaster(
                problem,
                IC2_HEAT_FLOW_CATALOGUE,
                prefix_catalogue=IC2_PERIODIC_PREFIX_CATALOGUE,
                base_hull_capacity=10_000,
            ),
            workers=workers,
            power_upper_bound=initial_upper,
            enforce_full_flow=False,
            enforce_ordered_distribution_flow=True,
            conditional_aggregate_patterns=conditional_patterns,
        )
    processor = LogicBendersProcessor(
        master,
        subproblem,
        master_fraction=master_fraction,
    )
    remaining = max(1e-6, deadline - perf_counter())
    report = ProofCoordinator().run(
        SearchNode(
            node_id="ic2_root",
            strict_upper_bound=initial_upper,
            payload=BendersDomain(problem.ruleset),
        ),
        processor,
        time_limit_seconds=remaining,
        known_witness=best_known,
    )
    return IC2GlobalProofResult(
        report=report,
        analytical_proof=analytical,
        initial_upper_bound=initial_upper,
        work_envelope=proof_work_envelope(
            problem,
            incumbent_lower_bound=report.lower_bound,
            static_upper_bound=initial_upper,
        ),
        conditional_aggregate_pattern_counts=tuple(
            sorted((power, len(patterns)) for power, patterns in conditional_patterns.items())
        ),
        elapsed_seconds=perf_counter() - started,
    )
