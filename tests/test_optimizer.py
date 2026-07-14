from ic2_reactor.engine import ReactorSimulator, SimulationOptions
from ic2_reactor.models import FuelConstraint, Layout, OptimizationRequest
from ic2_reactor.optimizer import (
    CandidateResult,
    OptimizationJob,
    _fixed_point_certificate,
    _allowed_and_caps,
    _exhaustive_shards,
    _rank_candidates,
    _run_exhaustive_shard,
    canonical_layout,
    canonical_tuple,
    construct_simple_cooling_candidates,
    count_cooling_completions,
    estimate_exhaustive_space,
    evaluate_layout,
    has_degrading_power_component,
    power_skeleton,
    prove_simple_fixed_point,
    skeleton_heat_per_tick,
    sustainable_heat_flow_upper_bound,
    sustainable_vent_upper_bound,
    theoretical_eu_per_tick,
)
from ic2_reactor.skeleton_table import POWER_EMPTY


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
    assert _rank_candidates([lower, higher], limit=1) == [higher]


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


def test_simple_fixed_point_proof_accepts_reachable_period_one_state():
    vent_after_fuel = tuple(["uranium_single", "heat_vent", *("empty" for _ in range(16))])
    vent_before_fuel = tuple(["heat_vent", "uranium_single", *("empty" for _ in range(16))])

    after = prove_simple_fixed_point(vent_after_fuel, 3, 40_000)
    before = prove_simple_fixed_point(vent_before_fuel, 3, 40_000)

    assert after is not None and before is not None
    assert after.mark == before.mark == "Mark I-I"
    assert after.average_eu_per_tick == before.average_eu_per_tick == 5.0
    assert after.safe_game_ticks == before.safe_game_ticks == 800_000


def test_simple_fixed_point_proof_matches_the_full_simulator_result():
    layout = tuple(["uranium_single", "reactor_heat_vent", *("empty" for _ in range(16))])
    certificate = prove_simple_fixed_point(layout, 3, 40_000)
    run = ReactorSimulator(Layout(columns=3, slots=list(layout))).simulate(SimulationOptions(
        max_game_ticks=800_000,
        auto_refuel=True,
        stop_on_stable=True,
        record_components=False,
        record_history=False,
    ))

    assert certificate is not None
    assert certificate.mark == run.summary.mark
    assert certificate.average_eu_per_tick == run.summary.average_eu_per_tick
    assert certificate.safe_game_ticks == run.summary.game_ticks
    assert certificate.safety_margin == 1.0 - run.summary.peak_hull_heat / run.summary.max_hull_heat


def test_simple_fixed_point_proof_does_not_confuse_nominal_capacity_with_a_heat_path():
    remote_vent = tuple(["uranium_single", "empty", "heat_vent", *("empty" for _ in range(15))])

    assert prove_simple_fixed_point(remote_vent, 3, 40_000) is None


def test_cooling_constructor_returns_only_a_proved_inventory_valid_witness():
    skeleton = tuple(["uranium_single", *("empty" for _ in range(17))])
    candidates = construct_simple_cooling_candidates(
        skeleton,
        3,
        tuple(range(1, 18)),
        {"heat_vent": 1},
        generated_heat=4,
        max_reactor_ticks=40_000,
        target_count=1,
    )

    assert len(candidates) == 1
    assert candidates[0].mark == "Mark I-I"
    assert candidates[0].layout.count("heat_vent") == 1
    assert prove_simple_fixed_point(candidates[0].layout, 3, 40_000) == candidates[0]


def test_mark_i_exhaustive_power_bound_skips_only_noncompetitive_layouts(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=1, quad=0),
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
    result = _run_exhaustive_shard(request.model_dump(mode="json"), 0, (), Queue(), Event())

    assert result["checked"] == estimate_exhaustive_space(request) == 342
    assert result["pruned"] > 0
    assert result["evaluated"] + result["pruned"] == result["checked"]


def test_top_one_activates_the_power_floor_earlier_than_top_ten(monkeypatch):
    base = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=1, quad=0),
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
    top_one = _run_exhaustive_shard(
        base.model_copy(update={"result_limit": 1}).model_dump(mode="json"),
        0,
        (),
        Queue(),
        Event(),
    )
    top_ten = _run_exhaustive_shard(
        base.model_copy(update={"result_limit": 10}).model_dump(mode="json"),
        0,
        (),
        Queue(),
        Event(),
    )

    expected = estimate_exhaustive_space(base)
    assert top_one["checked"] == top_ten["checked"] == expected
    assert top_one["evaluated"] + top_one["pruned"] == expected
    assert top_ten["evaluated"] + top_ten["pruned"] == expected
    assert top_one["boards"]["I"][0].average_eu_per_tick == top_ten["boards"]["I"][0].average_eu_per_tick
    assert top_one["evaluated"] < top_ten["evaluated"]


def test_top_one_constructed_witness_closes_the_remaining_cooling_subtree(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        result_limit=1,
        cpu_workers=1,
        max_reactor_ticks=40_000,
    )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    def full_simulation_must_not_run(*_args, **_kwargs):
        raise AssertionError("a proved constructed witness should avoid the simulator fallback")

    monkeypatch.setattr("ic2_reactor.optimizer.evaluate_layout", full_simulation_must_not_run)
    result = _run_exhaustive_shard(
        request.model_dump(mode="json"), 0, (), Queue(), Event()
    )

    assert result["checked"] == estimate_exhaustive_space(request) == 324
    assert result["evaluated"] == 1
    assert result["pruned"] == 323
    assert result["evaluated"] + result["pruned"] == result["checked"]
    assert result["boards"]["I"][0].average_eu_per_tick == 5.0


def test_constructor_failure_falls_back_without_changing_the_global_optimum(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        result_limit=1,
        cpu_workers=1,
        max_reactor_ticks=40_000,
    )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    fast = _run_exhaustive_shard(
        request.model_dump(mode="json"), 0, (), Queue(), Event()
    )
    monkeypatch.setattr(
        "ic2_reactor.optimizer.construct_simple_cooling_candidates",
        lambda *_args, **_kwargs: [],
    )
    fallback = _run_exhaustive_shard(
        request.model_dump(mode="json"), 0, (), Queue(), Event()
    )

    assert fast["checked"] == fallback["checked"] == estimate_exhaustive_space(request)
    assert fast["boards"]["I"][0].average_eu_per_tick == fallback["boards"]["I"][0].average_eu_per_tick
    assert fallback["evaluated"] + fallback["pruned"] == fallback["checked"]


def test_mark_i_heat_conservation_prunes_an_impossible_cooling_subtree():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"iridium_reflector": 1, "heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        result_limit=3,
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
        result_limit=3,
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


def test_oversized_nonfuel_caps_are_normalized_to_the_physical_slot_limit():
    saturated = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"heat_vent": 17},
    )
    oversized = saturated.model_copy(update={"component_limits": {"heat_vent": 54}})

    assert _allowed_and_caps(saturated) == _allowed_and_caps(oversized)
    assert estimate_exhaustive_space(saturated) == estimate_exhaustive_space(oversized)


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
    assert job._within_limits(valid)
    assert not job._within_limits(invalid)


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
        result_limit=3,
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


def test_adaptive_mark_i_shards_fill_requested_process_pool():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=4, dual=0, quad=0),
        component_limits={"heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=31,
    )

    shards = _exhaustive_shards(
        request,
        power_only=True,
        target_shards=request.cpu_workers * 4,
    )

    assert len(shards) == 163
    assert min(request.cpu_workers, len(shards)) == 31
    assert {len(shard) for shard in shards} == {8}
    assert all(
        item in {POWER_EMPTY, "uranium_single"}
        for shard in shards
        for _, item in shard
    )


def test_power_only_shards_keep_cooling_cells_and_cover_the_full_space(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        result_limit=1,
        cpu_workers=31,
        max_reactor_ticks=2_000,
    )

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_vent_upper_bound", lambda *_args: 10**9)
    monkeypatch.setattr("ic2_reactor.optimizer.sustainable_heat_flow_upper_bound", lambda *_args: 10**9)
    shards = _exhaustive_shards(
        request,
        power_only=True,
        target_shards=request.cpu_workers * 4,
    )
    results = [
        _run_exhaustive_shard(
            request.model_dump(mode="json"),
            shard_id,
            shard,
            Queue(),
            Event(),
        )
        for shard_id, shard in enumerate(shards)
    ]

    assert len(shards) == 19
    assert {len(shard) for shard in shards} == {18}
    assert sum(result["checked"] for result in results) == estimate_exhaustive_space(request) == 324
    assert sum(result["evaluated"] + result["pruned"] for result in results) == 324


def test_mark_i_fixed_cooling_prefix_preserves_exact_subspace_count(monkeypatch):
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        result_limit=3,
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
        request.model_dump(mode="json"),
        0,
        ((7, "heat_vent"), (10, "empty")),
        Queue(),
        Event(),
    )

    # Two fixed full-layout cells leave 16 possible positions for the one fuel.
    assert result["checked"] == result["evaluated"] == 16
    assert result["pruned"] == 0
