from __future__ import annotations

from ic2_reactor.logic_benders import (
    BendersCut,
    BendersDomain,
    LogicBendersProcessor,
    MasterStatus,
    SubproblemAnswer,
    SubproblemStatus,
)
from ic2_reactor.factorized_cooling_master import FactorizedCoolingCutMaster
from ic2_reactor.mathematical_model import (
    Graph,
    PowerComponent,
    ReactorProblem,
    ic2_mark_i_problem,
)
from ic2_reactor.proof_coordinator import ProofArtifact, ProofCoordinator, SearchNode
from ic2_reactor.thermal_benders import (
    FutureQuotientLayoutMasterAdapter,
    IC2ThermalSubproblem,
    ThermalLayoutMasterAdapter,
    ThermalMasterCut,
)
from ic2_reactor.thermal_master import ThermalCutMaster
from ic2_reactor.thermal_relaxation import (
    HeatFlowComponent,
    layout_heat_flow_bound,
)


class FlowIsTheSafetyRule:
    """Toy ruleset where satisfying the optimistic flow is exact feasibility."""

    def __init__(self, problem, catalogue):
        self.problem = problem
        self.catalogue = catalogue
        self.generated_cuts = 0

    def check(self, candidate, time_limit_seconds):
        flow = layout_heat_flow_bound(self.problem, candidate, self.catalogue)
        if flow.necessary_condition_satisfied:
            return SubproblemAnswer(
                SubproblemStatus.FEASIBLE,
                witness_payload=candidate,
                proof=ProofArtifact("toy_flow_witness", "flow is the toy safety rule"),
            )
        proof = ProofArtifact(
            "toy_min_cut",
            "the toy layout violates its exact flow safety rule",
            (("deficit", flow.deficit),),
        )
        cut = BendersCut[ThermalMasterCut](
            f"flow_{candidate}",
            flow.cut_template,
            proof,
        )
        self.generated_cuts += 1
        return SubproblemAnswer(
            SubproblemStatus.INFEASIBLE,
            proof=proof,
            generalized_cut=cut,
        )


def test_concrete_thermal_benders_full_flow_skips_impossible_high_power_family() -> None:
    problem = ReactorProblem(
        graph=Graph.from_edges(2, ((0, 1),)),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
            PowerComponent("mirror", 0, 0, True),
        ),
        cooling_components=(),
        layout_components=("sink",),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "mirror": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    master = ThermalLayoutMasterAdapter(
        ThermalCutMaster(problem, catalogue),
        workers=1,
    )
    subproblem = FlowIsTheSafetyRule(problem, catalogue)
    processor = LogicBendersProcessor(
        master,
        subproblem,
        master_fraction=0.8,
    )
    report = ProofCoordinator().run(
        SearchNode("root", 2, BendersDomain(object())),
        processor,
        time_limit_seconds=5,
    )
    assert report.proven_global
    assert report.lower_bound == report.upper_bound == 1
    assert report.best_witness is not None
    assert "sink" in report.best_witness.payload
    # The compact flow extension already represents every min-cut, so the
    # subproblem never receives the impossible high-power reflector family.
    assert subproblem.generated_cuts == 0
    assert any(
        record.proof.kind == "logic_benders_partition"
        for record in report.closed_nodes
    )


def test_future_quotient_master_is_a_complete_benders_master_on_small_grid() -> None:
    problem = ReactorProblem(
        graph=Graph.rectangular(1, 2),
        rod_budget=1,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            PowerComponent("fuel", 1, 1, True),
            PowerComponent("mirror", 0, 0, True),
        ),
        cooling_components=(),
        layout_components=("sink",),
        eu_per_pulse=1,
        heat_scale=1,
    )
    catalogue = {
        "empty": HeatFlowComponent(),
        "fuel": HeatFlowComponent(),
        "mirror": HeatFlowComponent(),
        "sink": HeatFlowComponent(accepts_heat=True, self_vent=2),
    }
    master = FutureQuotientLayoutMasterAdapter(
        FactorizedCoolingCutMaster(problem, catalogue),
        power_upper_bound=2,
    )
    first = master.solve(BendersDomain(object()), 0, 2)
    assert first.status == MasterStatus.CANDIDATE
    assert first.candidate is not None
    assert first.candidate.objective_value == 2
    assert first.candidate.residual_upper_bound == 2

    following = master.solve(
        BendersDomain(object(), cuts=(first.candidate.exact_no_good,)),
        0,
        2,
    )
    assert following.status == MasterStatus.CANDIDATE
    assert following.candidate is not None
    assert following.candidate.payload != first.candidate.payload

    closed = master.solve(BendersDomain(object()), 2, 2)
    assert closed.status == MasterStatus.INFEASIBLE
    assert closed.proof is not None
    assert closed.proof.kind == "future_quotient_domain_bound"


WITNESS_380 = tuple(
    {
        ".": "empty",
        "S": "uranium_single",
        "Q": "uranium_quad",
        "R": "iridium_reflector",
        "O": "overclocked_heat_vent",
        "C": "component_heat_vent",
        "X": "component_heat_exchanger",
        "P": "reactor_plating",
    }[symbol]
    for symbol in "".join((
        "QCOXOOCRP",
        ".COOCORSR",
        "POQOOQOOX",
        "COOCOOCOO",
        "OQOOQOOQO",
        "COCPOCPOC",
    ))
)


def test_ic2_subproblem_uses_exact_cycle_after_flow_filter() -> None:
    subproblem = IC2ThermalSubproblem(
        ic2_mark_i_problem(),
        max_steps=1_000,
    )
    answer = subproblem.check(WITNESS_380, 2.0)
    assert answer.status == SubproblemStatus.FEASIBLE
    assert answer.witness_payload is not None
    assert answer.witness_payload.cycle.transient_length == 380
    assert answer.witness_payload.cycle.period_length == 18
    assert subproblem.check(WITNESS_380, 2.0) is answer
    assert subproblem.cache_entries == 1
    assert subproblem.cache_hits == 1


def test_ic2_subproblem_never_caches_an_unknown_horizon() -> None:
    subproblem = IC2ThermalSubproblem(
        ic2_mark_i_problem(),
        max_steps=1,
    )
    assert subproblem.check(WITNESS_380, 2.0).status == SubproblemStatus.UNKNOWN
    assert subproblem.cycle_session_checked_steps(WITNESS_380) == 1
    assert subproblem.check(WITNESS_380, 2.0).status == SubproblemStatus.UNKNOWN
    assert subproblem.cycle_session_checked_steps(WITNESS_380) == 2
    assert subproblem.open_cycle_sessions == 1
    assert subproblem.cache_entries == 0
    assert subproblem.cache_hits == 0
