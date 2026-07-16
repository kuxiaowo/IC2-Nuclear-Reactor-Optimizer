import json
import random
import threading
import time

from ic2_reactor.models import FuelConstraint, OptimizationRequest
from ic2_reactor.mark import mark_family
from ic2_reactor.optimizer import (
    CandidateResult,
    OptimizationJob,
    _fixed_point_certificate,
    _evaluate_search_batch,
    _partial_mark_i_heat_infeasible,
    _partial_skeleton_heat_increment,
    _rank_candidates,
    _run_exhaustive_shard,
    _wait_for_worker_control,
    canonical_layout,
    canonical_tuple,
    count_cooling_completions,
    estimate_exhaustive_space,
    evaluate_layout,
    evaluate_layout_batch,
    has_degrading_power_component,
    power_skeleton,
    skeleton_heat_per_tick,
    sustainable_heat_flow_upper_bound,
    sustainable_vent_upper_bound,
    theoretical_eu_per_tick,
)


def test_worker_pause_blocks_at_safe_point_until_resumed():
    cancel_event = threading.Event()
    pause_event = threading.Event()
    pause_event.set()
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            _wait_for_worker_control(cancel_event, pause_event)
        )
    )

    worker.start()
    time.sleep(0.1)
    assert worker.is_alive()
    pause_event.clear()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result == [False]


def test_job_pause_persists_checkpoint_and_resumes_in_place(monkeypatch, tmp_path):
    monkeypatch.setattr("ic2_reactor.optimizer.CHECKPOINT_DIRECTORY", tmp_path)
    job = OptimizationJob(OptimizationRequest())
    job.started_at = time.time() - 5
    job.status = "running"
    job.process_pause_event = threading.Event()

    job.pause()
    paused_elapsed = job.snapshot()["elapsed_seconds"]

    assert job.status == "paused"
    assert job.pause_event.is_set()
    assert job.process_pause_event.is_set()
    assert job.checkpoint_path.exists()
    payload = json.loads(job.checkpoint_path.read_text(encoding="utf-8"))
    assert payload["paused"] is True
    assert payload["snapshot"]["checked"] == 0
    assert payload["restart_resumable"] is False
    time.sleep(0.05)
    assert job.snapshot()["elapsed_seconds"] == paused_elapsed

    job.resume_in_place()

    assert job.status == "running"
    assert not job.pause_event.is_set()
    assert not job.process_pause_event.is_set()
    payload = json.loads(job.checkpoint_path.read_text(encoding="utf-8"))
    assert payload["paused"] is False


def test_incremental_partial_heat_matches_full_skeleton_calculation():
    rng = random.Random(221)
    values = (
        "uranium_single",
        "uranium_dual",
        "uranium_quad",
        "iridium_reflector",
        "empty",
    )
    for _ in range(20):
        skeleton = ["empty"] * 18
        heat = 0
        positions = list(range(18))
        rng.shuffle(positions)
        for position in positions:
            item = rng.choice(values)
            if item == "empty":
                continue
            heat += _partial_skeleton_heat_increment(
                skeleton,
                position,
                item,
                3,
            )
            skeleton[position] = item
            assert heat == skeleton_heat_per_tick(tuple(skeleton), 3)


def test_partial_heat_bound_keeps_a_thermally_possible_root():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(
            mode="separate", usage="exact", single=1, dual=0, quad=0
        ),
        component_limits={"overclocked_heat_vent": 17},
        marks=["I"],
        solver="exhaustive",
    )

    assert not _partial_mark_i_heat_infeasible(
        request,
        {"uranium_single": 1},
        (("overclocked_heat_vent", 17),),
        current_rods=0,
        current_power_slots=0,
        available_slots=18,
        current_heat=0,
    )


def test_partial_heat_bound_closes_an_impossible_root_combinatorially():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(
            mode="separate", usage="exact", single=0, dual=0, quad=4
        ),
        component_limits={"overclocked_heat_vent": 18},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=1,
        max_reactor_ticks=2_000,
    )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    result = _run_exhaustive_shard(
        request.model_dump(mode="json"), 0, (), Queue(), Event()
    )

    assert result["checked"] == estimate_exhaustive_space(request) == 50_135_040
    assert result["pruned"] == result["checked"]
    assert result["evaluated"] == 0


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


def test_leaderboard_respects_requested_result_limit():
    layouts = [
        tuple(["uranium_single", *("empty" for _ in range(16)), item])
        for item in ("heat_vent", "reactor_heat_vent", "overclocked_heat_vent")
    ]
    candidates = [
        CandidateResult(
            layout,
            "Mark I-I",
            float(score),
            float(score * 40_000),
            40_000,
            1.0,
            2,
            canonical_layout(layout, 3),
        )
        for score, layout in enumerate(layouts, start=1)
    ]

    assert _rank_candidates(candidates, result_limit=1) == [candidates[-1]]


def test_candidate_score_is_strictly_generation_power_only():
    layout_a = tuple(["uranium_single", *("empty" for _ in range(17))])
    layout_b = tuple(["empty", "uranium_single", *("empty" for _ in range(16))])
    a = CandidateResult(layout_a, "I", 5.0, 100.0, 20, 0.99, 1, canonical_layout(layout_a, 3))
    b = CandidateResult(layout_b, "I", 5.0, 1_000.0, 200, 1.0, 2, canonical_layout(layout_b, 3))
    assert a.score() == b.score() == (5.0,)


def test_power_skeleton_calculates_exact_static_eu_output():
    isolated_quad = tuple(["uranium_quad", "reactor_heat_vent", *("empty" for _ in range(16))])
    reflected_quad = tuple(["iridium_reflector", "uranium_quad", *("empty" for _ in range(16))])
    adjacent_singles = tuple(["uranium_single", "uranium_single", *("empty" for _ in range(16))])

    assert power_skeleton(isolated_quad)[1] == "empty"
    assert theoretical_eu_per_tick(isolated_quad, 3) == 60.0
    assert theoretical_eu_per_tick(reflected_quad, 3) == 80.0
    assert theoretical_eu_per_tick(adjacent_singles, 3) == 20.0


def test_power_skeleton_calculates_exact_static_heat_output():
    isolated_single = tuple(["uranium_single", *("empty" for _ in range(17))])
    reflected_single = tuple(["uranium_single", "iridium_reflector", *("empty" for _ in range(16))])
    adjacent_singles = tuple(["uranium_single", "uranium_single", *("empty" for _ in range(16))])

    assert skeleton_heat_per_tick(isolated_single, 3) == 4
    assert skeleton_heat_per_tick(reflected_single, 3) == 12
    assert skeleton_heat_per_tick(adjacent_singles, 3) == 24


def test_sustainable_vent_bound_is_optimistic_but_respects_slots_and_inventory():
    skeleton = tuple(["uranium_single", "iridium_reflector", *("empty" for _ in range(16))])
    assert sustainable_vent_upper_bound(skeleton, 3, (("heat_vent", 1),)) == 6
    assert sustainable_vent_upper_bound(
        skeleton,
        3,
        (("heat_vent", 1), ("advanced_heat_vent", 1), ("coolant_60k", 10)),
    ) == 18


def test_sustainable_heat_flow_bound_tracks_heat_delivery_paths():
    adjacent_vent = tuple(["uranium_single", "heat_vent", *("empty" for _ in range(16))])
    remote_vent = tuple(["uranium_single", "empty", "heat_vent", *("empty" for _ in range(15))])
    hull_vent = tuple(["uranium_single", "empty", "reactor_heat_vent", *("empty" for _ in range(15))])
    reflected = tuple([
        "uranium_single", "iridium_reflector", "empty", "heat_vent", *("empty" for _ in range(14))
    ])

    assert sustainable_heat_flow_upper_bound(adjacent_vent, 3) == 4
    assert sustainable_heat_flow_upper_bound(remote_vent, 3) == 0
    assert sustainable_heat_flow_upper_bound(hull_vent, 3) == 4
    assert sustainable_heat_flow_upper_bound(reflected, 3) == 6


def test_active_finite_reflector_cannot_form_mark_i_fixed_state():
    active = tuple(["neutron_reflector", "uranium_single", *("empty" for _ in range(16))])
    inactive = tuple(["neutron_reflector", "empty", "uranium_single", *("empty" for _ in range(15))])
    assert has_degrading_power_component(active, 3)
    assert not has_degrading_power_component(inactive, 3)


def test_fixed_point_certificate_reuses_identical_layout_result():
    _fixed_point_certificate.cache_clear()
    stable = tuple(["uranium_single", "reactor_heat_vent", *("empty" for _ in range(16))])
    first = evaluate_layout(stable, 3, 40_000)
    second = evaluate_layout(stable, 3, 40_000)
    info = _fixed_point_certificate.cache_info()
    assert first == second
    assert (info.misses, info.hits) == (1, 1)


def test_scalar_batch_boundary_matches_individual_evaluation():
    stable = tuple(["uranium_single", "reactor_heat_vent", *(["empty"] * 16)])
    unsafe = tuple(["uranium_quad", *(["empty"] * 17)])

    batch = evaluate_layout_batch((stable, unsafe), 3, 40_000)
    scalar = [
        evaluate_layout(layout, 3, 40_000, use_certificate=False)
        for layout in (stable, unsafe)
    ]

    assert batch == scalar


def test_numba_batch_boundary_matches_scalar_candidates():
    stable = tuple(["uranium_single", "reactor_heat_vent", *(["empty"] * 16)])
    unsafe = tuple(["uranium_quad", *(["empty"] * 17)])
    layouts = (stable, unsafe)

    scalar = evaluate_layout_batch(layouts, 3, 40_000)
    accelerated = evaluate_layout_batch(
        layouts,
        3,
        40_000,
        False,
        None,
        "numba_cpu",
        2,
    )

    assert accelerated == scalar


def test_late_mark_i_layout_is_not_treated_as_permanently_unclassified():
    layout = ["empty"] * 18
    layout[4] = "uranium_single"
    for position in (1, 3, 5, 7):
        layout[position] = "lzh_condensator"
    layout[17] = "reactor_heat_vent"

    short = evaluate_layout(tuple(layout), 3, 40_000, use_certificate=False)
    long = evaluate_layout(tuple(layout), 3, 140_000, use_certificate=False)

    assert mark_family(short.mark) is None
    assert long.mark == "Mark I-I-SUC"


def test_competitive_unclassified_layout_is_extended_to_requested_horizon():
    layout = ["empty"] * 18
    layout[4] = "uranium_single"
    for position in (1, 3, 5, 7):
        layout[position] = "lzh_condensator"
    layout[17] = "reactor_heat_vent"
    request = OptimizationRequest(
        columns=3,
        marks=["I"],
        max_reactor_ticks=40_000,
        unresolved_max_reactor_ticks=140_000,
    )

    results = _evaluate_search_batch(
        (tuple(layout),),
        request,
        {"I": []},
        lambda: False,
    )

    assert results[0].mark == "Mark I-I-SUC"


def test_noncompetitive_unclassified_layout_skips_extension(monkeypatch):
    layout = tuple(["uranium_single", *(["empty"] * 17)])
    calls = []

    def fake_batch(layouts, columns, max_reactor_ticks, *args):
        calls.append(max_reactor_ticks)
        return [
            CandidateResult(
                candidate,
                "未分类",
                5.0,
                200_000.0,
                max_reactor_ticks * 20,
                1.0,
                1,
                canonical_layout(candidate, columns),
            )
            for candidate in layouts
        ]

    board = [
        CandidateResult(
            tuple(["uranium_single", *(["empty"] * 16), str(index)]),
            "Mark I-I",
            10.0,
            400_000.0,
            40_000,
            1.0,
            1,
            f"candidate-{index}",
        )
        for index in range(10)
    ]
    request = OptimizationRequest(
        columns=3,
        marks=["I"],
        max_reactor_ticks=40_000,
        unresolved_max_reactor_ticks=140_000,
    )
    monkeypatch.setattr("ic2_reactor.optimizer.evaluate_layout_batch", fake_batch)

    results = _evaluate_search_batch((layout,), request, {"I": board}, lambda: False)

    assert results[0].mark == "未分类"
    assert calls == [40_000]


def test_unclassified_layouts_cannot_block_completed_mark_v_proof_after_threshold():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={},
        marks=["V"],
        solver="exhaustive",
        cpu_workers=1,
        max_reactor_ticks=2_000,
    )
    job = OptimizationJob(request)

    job.run()

    assert job.unresolved == 18
    assert job.proven_global


def test_mark_i_exhaustive_power_bound_skips_only_noncompetitive_layouts(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", usage="maximum", single=1, dual=1, quad=0),
        component_limits={},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=1,
        max_reactor_ticks=2_000,
    )

    def fake_evaluate(layout, columns, max_reactor_ticks, cancel_check=None):
        power = theoretical_eu_per_tick(layout, columns)
        return CandidateResult(
            layout, "Mark I-I", power, power * 40_000, 40_000, 1.0,
            sum(item != "empty" for item in layout), canonical_layout(layout, columns)
        )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    monkeypatch.setattr("ic2_reactor.optimizer.evaluate_layout", fake_evaluate)
    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_vent_upper_bound", lambda *_args: 10**9)
    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_heat_flow_upper_bound", lambda *_args: 10**9)
    monkeypatch.setattr("ic2_reactor.optimizer._partial_mark_i_heat_infeasible", lambda *_args: False)
    result = _run_exhaustive_shard(request.model_dump(mode="json"), 0, (), Queue(), Event())

    assert result["checked"] == estimate_exhaustive_space(request) == 342
    assert result["pruned"] > 0
    assert result["evaluated"] + result["pruned"] == result["checked"]

    exact_best = 0.0
    for first in range(18):
        for first_item in ("uranium_single", "uranium_dual"):
            layout = ["empty"] * 18
            layout[first] = first_item
            exact_best = max(exact_best, theoretical_eu_per_tick(tuple(layout), 3))
        for second in range(18):
            if first == second:
                continue
            layout = ["empty"] * 18
            layout[first] = "uranium_single"
            layout[second] = "uranium_dual"
            exact_best = max(exact_best, theoretical_eu_per_tick(tuple(layout), 3))
    assert result["boards"]["I"][0].average_eu_per_tick == exact_best


def test_mark_i_exact_fuel_generator_counts_only_complete_inventory(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(
            mode="separate", usage="exact", single=1, dual=1, quad=0
        ),
        component_limits={},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=1,
        max_reactor_ticks=2_000,
    )

    def fake_evaluate(layout, columns, max_reactor_ticks, cancel_check=None):
        power = theoretical_eu_per_tick(layout, columns)
        return CandidateResult(
            layout, "Mark I-I", power, power * 40_000, 40_000, 1.0,
            sum(item != "empty" for item in layout), canonical_layout(layout, columns)
        )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    monkeypatch.setattr("ic2_reactor.optimizer.evaluate_layout", fake_evaluate)
    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_vent_upper_bound", lambda *_args: 10**9)
    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_heat_flow_upper_bound", lambda *_args: 10**9)
    result = _run_exhaustive_shard(
        request.model_dump(mode="json"), 0, (), Queue(), Event()
    )

    assert result["checked"] == estimate_exhaustive_space(request) == 306
    assert result["evaluated"] + result["pruned"] == 306


def test_mark_i_cooling_search_visits_full_layout_before_empty_variants(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(
            mode="separate", usage="exact", single=16, dual=0, quad=0
        ),
        component_limits={"component_heat_vent": 2},
        marks=["I"],
        solver="exhaustive",
        result_limit=1,
        cpu_workers=1,
        max_reactor_ticks=2_000,
    )
    evaluated_layouts = []

    def fake_evaluate(layout, columns, max_reactor_ticks, cancel_check=None):
        evaluated_layouts.append(layout)
        power = theoretical_eu_per_tick(layout, columns)
        return CandidateResult(
            layout,
            "Mark I-I",
            power,
            power * 40_000,
            40_000,
            1.0,
            sum(item != "empty" for item in layout),
            canonical_layout(layout, columns),
        )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    monkeypatch.setattr("ic2_reactor.optimizer.evaluate_layout", fake_evaluate)
    monkeypatch.setattr(
        "ic2_reactor.optimizer.sustainable_vent_upper_bound",
        lambda *_args: 10**9,
    )
    monkeypatch.setattr(
        "ic2_reactor.optimizer.sustainable_heat_flow_upper_bound",
        lambda *_args: 10**9,
    )
    monkeypatch.setattr(
        "ic2_reactor.optimizer._partial_mark_i_heat_infeasible",
        lambda *_args: False,
    )
    fixed = tuple((position, "uranium_single") for position in range(16))

    result = _run_exhaustive_shard(
        request.model_dump(mode="json"), 0, fixed, Queue(), Event()
    )

    assert evaluated_layouts[0].count("empty") == 0
    assert result["checked"] == result["evaluated"] == 4
    assert len(result["boards"]["I"]) == 1


def test_mark_i_heat_conservation_prunes_an_impossible_cooling_subtree():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"iridium_reflector": 1, "heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=1,
        max_reactor_ticks=40_000,
    )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    result = _run_exhaustive_shard(
        request.model_dump(mode="json"),
        0,
        ((0, "uranium_single"), (1, "iridium_reflector")),
        Queue(),
        Event(),
    )

    # Empty cooling plus one heat vent in any of 16 free slots: all have
    # generated heat 12 > optimistic sustainable venting 6.
    assert result["checked"] == result["pruned"] == 17
    assert result["evaluated"] == 0


def test_mark_i_heat_flow_prunes_completions_without_a_sustainable_path(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=1,
        max_reactor_ticks=40_000,
    )

    def fake_evaluate(layout, columns, max_reactor_ticks, cancel_check=None):
        power = theoretical_eu_per_tick(layout, columns)
        return CandidateResult(
            layout, "Mark I-I", power, power * 40_000, 40_000, 1.0,
            sum(item != "empty" for item in layout), canonical_layout(layout, columns)
        )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    monkeypatch.setattr("ic2_reactor.optimizer.evaluate_layout", fake_evaluate)
    result = _run_exhaustive_shard(
        request.model_dump(mode="json"),
        0,
        ((0, "uranium_single"),),
        Queue(),
        Event(),
    )

    assert result["checked"] == 18
    assert result["evaluated"] == 2
    assert result["pruned"] == 16


def test_exhaustive_estimate_respects_inventory_instead_of_alphabet_power():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"heat_vent": 1},
    )
    # 18 个仅燃料布局 + 18×17 个燃料和散热片布局。
    assert estimate_exhaustive_space(request) == 324


def test_exact_separate_fuel_counts_remove_partial_inventory_layouts():
    exact = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(
            mode="separate", usage="exact", single=1, dual=1, quad=0
        ),
    )
    maximum = exact.model_copy(update={
        "fuel": exact.fuel.model_copy(update={"usage": "maximum"})
    })

    assert estimate_exhaustive_space(exact) == 18 * 17 == 306
    assert estimate_exhaustive_space(maximum) == 306 + 18 + 18 == 342


def test_exact_total_rods_counts_only_package_combinations_with_target_sum():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(
            mode="total_rods", usage="exact", total_rods=4
        ),
    )

    # 4 singles; 2 singles + 1 dual; 2 duals; or 1 quad.
    assert estimate_exhaustive_space(request) == 3060 + 2448 + 153 + 18 == 5679


def test_two_level_generator_counts_entire_cooling_subtree_combinatorially():
    # 三个空位，两种冷却组件各最多一个：1 + 3 + 3 + 3×2。
    assert count_cooling_completions(3, (1, 1)) == 13


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
    incomplete = tuple(["uranium_dual", *(["empty"] * 17)])
    assert job._within_limits(valid)
    assert not job._within_limits(invalid)
    assert not job._within_limits(incomplete)

    maximum = OptimizationJob(request.model_copy(update={
        "fuel": request.fuel.model_copy(update={"usage": "maximum"})
    }))
    assert maximum._within_limits(incomplete)


def test_exact_heuristic_population_always_uses_requested_fuel_counts():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(
            mode="separate", usage="exact", single=2, dual=1, quad=1
        ),
        component_limits={"heat_vent": 3},
    )
    job = OptimizationJob(request)

    for seed in range(10):
        layout = job._random_layout(random.Random(seed))
        assert layout.count("uranium_single") == 2
        assert layout.count("uranium_dual") == 1
        assert layout.count("uranium_quad") == 1
        assert job._within_limits(layout)


def test_exact_random_layout_fills_all_slots_when_cooling_inventory_allows_it():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="total_rods", usage="exact", total_rods=4),
        component_limits={
            "overclocked_heat_vent": 18,
            "component_heat_vent": 18,
        },
    )
    job = OptimizationJob(request)

    for seed in range(10):
        layout = job._random_layout(random.Random(seed))
        assert "empty" not in layout
        assert job._within_limits(layout)


def test_fixed_seed_random_population_is_reproducible():
    import random

    request = OptimizationRequest(columns=3, population=10, cpu_workers=1, seed=221)
    a, b = OptimizationJob(request), OptimizationJob(request)
    assert [a._random_layout(random.Random(221)) for _ in range(2)] == [b._random_layout(random.Random(221)) for _ in range(2)]


def test_heuristic_simulates_duplicate_layout_only_once():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={},
        marks=["I", "II", "III", "IV", "V"],
        solver="heuristic",
        time_budget_seconds=30,
        generations=3,
        population=10,
        cpu_workers=1,
        max_reactor_ticks=2_000,
    )
    job = OptimizationJob(request)
    repeated = tuple(["uranium_single", *("empty" for _ in range(17))])
    job._random_layout = lambda _rng: repeated
    job._mutate = lambda _layout, _rng: repeated

    job.run()

    assert job.status == "completed"
    assert job.evaluated == 1
    assert len(job._heuristic_cache) == 1


def test_multicore_heuristic_uses_batch_worker_boundary():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I", "II", "III", "IV", "V"],
        solver="heuristic",
        time_budget_seconds=30,
        generations=1,
        population=10,
        cpu_workers=2,
        max_reactor_ticks=2_000,
    )

    job = OptimizationJob(request)
    job.run()

    assert job.status == "completed"
    assert 0 < job.evaluated <= request.population


def test_multicore_exhaustive_search_closes_horizon_without_hiding_unresolved_layouts():
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
    assert job.enumeration_processes == 2
    assert job.simulation_processes == 0
    assert job.proven_within_horizon
    assert not job.proven_global
    assert job.checked == 18
    assert job.evaluated == 18
    assert job.pruned == 0
    assert job.unresolved == 18


def test_numba_exhaustive_backend_preserves_search_counts_and_proof_state():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={},
        marks=["I", "II", "III", "IV", "V"],
        solver="exhaustive",
        cpu_workers=2,
        compute_backend="numba_cpu",
        max_reactor_ticks=2_000,
    )
    job = OptimizationJob(request)

    job.run()

    assert job.status == "completed"
    assert job.enumeration_processes == 2
    assert job.simulation_processes == 0
    assert job.proven_within_horizon
    assert not job.proven_global
    assert (job.checked, job.evaluated, job.pruned, job.unresolved) == (18, 18, 0, 18)


def test_multicore_mark_i_two_level_generator_preserves_complete_counts():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=2,
        max_reactor_ticks=40_000,
    )
    job = OptimizationJob(request)
    job.run()

    assert job.status == "completed"
    assert job.proven_global
    assert job.checked == estimate_exhaustive_space(request) == 324
    assert job.evaluated + job.pruned == job.checked
    assert job.leaderboards["I"]


def test_mark_i_partial_bound_counting_respects_total_rod_packages(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="total_rods", total_rods=3),
        component_limits={"reactor_heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=1,
        max_reactor_ticks=2_000,
    )

    def fake_evaluate(layout, columns, max_reactor_ticks, cancel_check=None):
        power = theoretical_eu_per_tick(layout, columns)
        return CandidateResult(
            layout, "Mark I-I", power, power * 40_000, 40_000, 1.0,
            sum(item != "empty" for item in layout), canonical_layout(layout, columns)
        )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    monkeypatch.setattr("ic2_reactor.optimizer.evaluate_layout", fake_evaluate)
    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_vent_upper_bound", lambda *_args: 10**9)
    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_heat_flow_upper_bound", lambda *_args: 10**9)
    result = _run_exhaustive_shard(
        request.model_dump(mode="json"), 0, (), Queue(), Event()
    )

    assert result["checked"] == estimate_exhaustive_space(request)
    assert result["evaluated"] + result["pruned"] == result["checked"]
    assert result["pruned"] > 0
