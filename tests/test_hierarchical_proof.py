from __future__ import annotations

import pytest

from ic2_reactor.hierarchical_proof import IC2HierarchicalPatternProver
from ic2_reactor.mathematical_model import AggregatePattern, ic2_mark_i_problem


def test_hierarchy_closes_a_whole_pattern_at_the_ideal_skeleton_layer() -> None:
    problem = ic2_mark_i_problem(
        rows=6,
        columns=1,
        rod_budget=1,
        enabled_components={"uranium_single"},
    )
    pattern = AggregatePattern(
        active_cells=1,
        generated_heat=4,
        slack=0,
        required_relief=0,
        maximum_available_relief=0,
        margin=0,
        fuel_degree_counts=(("uranium_single", 0, 1),),
    )
    report = IC2HierarchicalPatternProver(
        problem,
        workers=1,
        max_cycle_steps=100,
    ).prove(
        pattern,
        power=5,
        time_limit_seconds=2,
        master_unit_seconds=1,
        subproblem_unit_seconds=1,
    )
    assert report.proven_closed
    assert report.status == "closed"
    assert report.skeleton_candidates == 0
    assert not report.open_skeletons


def test_hierarchy_rejects_unknown_cooling_master_backend() -> None:
    problem = ic2_mark_i_problem(
        rows=1,
        columns=1,
        rod_budget=1,
        enabled_components={"uranium_single"},
    )
    with pytest.raises(ValueError, match="unknown cooling master backend"):
        IC2HierarchicalPatternProver(
            problem,
            cooling_master_backend="unknown",  # type: ignore[arg-type]
        )
