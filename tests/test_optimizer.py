from ic2_reactor.models import FuelConstraint, OptimizationRequest
from ic2_reactor.optimizer import (
    CandidateResult,
    OptimizationJob,
    _rank_candidates,
    canonical_layout,
    canonical_tuple,
    estimate_exhaustive_space,
)


def test_canonical_layout_removes_horizontal_vertical_and_180_symmetry():
    original = tuple(["uranium_single", *(["empty"] * 17)])
    horizontal = tuple(["empty", "empty", "uranium_single", *(["empty"] * 15)])
    vertical = tuple([*(["empty"] * 15), "uranium_single", "empty", "empty"])
    assert canonical_layout(original, 3) == canonical_layout(horizontal, 3) == canonical_layout(vertical, 3)
    assert canonical_tuple(original, 3) == canonical_tuple(horizontal, 3) == canonical_tuple(vertical, 3)


def test_leaderboard_keeps_the_best_scoring_direction_in_a_mirror_group():
    original = tuple(["uranium_single", "heat_vent", *("empty" for _ in range(16))])
    mirrored = tuple(["empty", "heat_vent", "uranium_single", *("empty" for _ in range(15))])
    canonical = canonical_layout(original, 3)
    assert canonical == canonical_layout(mirrored, 3)

    lower = CandidateResult(original, "I", 5.0, 100.0, 20, 1.0, 2, canonical)
    higher = CandidateResult(mirrored, "I", 6.0, 120.0, 20, 1.0, 2, canonical)

    assert _rank_candidates([lower, higher]) == [higher]


def test_exhaustive_estimate_respects_inventory_instead_of_alphabet_power():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"heat_vent": 1},
    )
    # 18 个仅燃料布局 + 18×17 个燃料和散热片布局。
    assert estimate_exhaustive_space(request) == 324


def test_exhaustive_estimate_has_no_artificial_safety_cap():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=3, dual=2, quad=1),
        component_limits={"heat_vent": 4, "advanced_heat_vent": 4},
        solver="exhaustive",
    )
    assert estimate_exhaustive_space(request) > 2_000_000


def test_total_rod_limit_accepts_mixed_packages_without_exceeding_budget():
    request = OptimizationRequest(columns=3, fuel=FuelConstraint(mode="total_rods", total_rods=4))
    job = OptimizationJob(request)
    valid = tuple(["uranium_dual", "uranium_single", "uranium_single", *(["empty"] * 15)])
    invalid = tuple(["uranium_quad", "uranium_single", *(["empty"] * 16)])
    assert job._within_limits(valid)
    assert not job._within_limits(invalid)


def test_fixed_seed_random_population_is_reproducible():
    import random

    request = OptimizationRequest(columns=3, population=10, cpu_workers=1, seed=221)
    a, b = OptimizationJob(request), OptimizationJob(request)
    assert [a._random_layout(random.Random(221)) for _ in range(2)] == [b._random_layout(random.Random(221)) for _ in range(2)]


def test_multicore_exhaustive_search_checks_each_layout_once_and_proves_optimum():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={},
        marks=["I", "II", "III", "IV", "V"],
        solver="exhaustive",
        cpu_workers=2,
        max_reactor_ticks=2_000,
    )
    job = OptimizationJob(request)
    job.run()

    assert job.status == "completed"
    assert job.proven_global
    assert job.checked == 18
    assert job.evaluated == 18
    assert job.pruned == 0
