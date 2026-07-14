import random

import numpy as np
import pytest

from ic2_reactor.gpu_acceleration import (
    CudaBatchScorer,
    CudaBatchScores,
    CudaFixedPointEvaluator,
    cuda_device_info,
    select_screened_layouts,
)
from ic2_reactor.models import FuelConstraint, OptimizationRequest
from ic2_reactor.optimizer import (
    OptimizationJob,
    _run_exhaustive_shard,
    prove_simple_fixed_point,
    skeleton_eu_per_tick,
    skeleton_heat_per_tick,
)


def test_gpu_screening_selection_preserves_power_and_thermal_frontiers():
    layouts = [(str(index),) for index in range(6)]
    scores = CudaBatchScores(
        power=np.asarray([10, 100, 90, 80, 70, 60], dtype=np.int32),
        generated_heat=np.asarray([1, 100, 90, 80, 70, 60], dtype=np.int32),
        cooling_proxy=np.asarray([1, 0, 0, 100, 80, 0], dtype=np.int32),
    )

    selected = select_screened_layouts(layouts, scores, 4, mark_i_only=True)

    assert layouts[1] in selected  # theoretical-power frontier
    assert layouts[3] in selected  # plausible Mark-I frontier
    assert len(selected) == 4
    assert len(set(selected)) == 4


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_cuda_batch_scores_match_exact_cpu_static_metrics():
    layouts = [
        tuple(["uranium_single", *("empty" for _ in range(17))]),
        tuple([
            "uranium_single", "iridium_reflector", "reactor_heat_vent",
            *("empty" for _ in range(15)),
        ]),
        tuple([
            "uranium_quad", "uranium_dual", "advanced_heat_vent",
            "component_heat_vent", "coolant_10k", *("empty" for _ in range(13)),
        ]),
    ]

    scores = CudaBatchScorer().score(layouts, columns=3)

    assert scores.power.tolist() == [
        int(skeleton_eu_per_tick(layout, 3)) for layout in layouts
    ]
    assert scores.generated_heat.tolist() == [
        skeleton_heat_per_tick(layout, 3) for layout in layouts
    ]
    assert np.all(scores.cooling_proxy >= 0)


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_cuda_fixed_point_certificates_match_cpu_proofs():
    layouts = [
        tuple(["uranium_single", "reactor_heat_vent", *("empty" for _ in range(16))]),
        tuple(["reactor_heat_vent", "uranium_single", *("empty" for _ in range(16))]),
        tuple([
            "uranium_single", "reactor_heat_vent", "heat_capacity_plating",
            *("empty" for _ in range(15)),
        ]),
    ]

    certificates = CudaFixedPointEvaluator().certify(layouts, 3, 40_000)

    for layout, certificate in zip(layouts, certificates, strict=True):
        expected = prove_simple_fixed_point(layout, 3, 40_000)
        assert expected is not None
        assert certificate is not None
        assert certificate.average_eu_per_tick == expected.average_eu_per_tick
        assert certificate.safe_game_ticks == expected.safe_game_ticks
        assert certificate.peak_hull_heat == round((1.0 - expected.safety_margin) * certificate.max_hull_heat)


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_cuda_random_fixed_point_certificates_have_no_cpu_false_positives():
    rng = random.Random(221)
    values = [
        *("empty" for _ in range(12)),
        "uranium_single",
        "reactor_heat_vent",
        "reactor_heat_vent",
        "advanced_heat_vent",
        "heat_capacity_plating",
        "iridium_reflector",
    ]
    layouts = []
    for _ in range(256):
        layout = [rng.choice(values) for _ in range(18)]
        layout[rng.randrange(18)] = "uranium_single"
        layouts.append(tuple(layout))

    certificates = CudaFixedPointEvaluator().certify(layouts, 3, 40_000)
    certified = 0
    for layout, certificate in zip(layouts, certificates, strict=True):
        if certificate is None:
            continue
        certified += 1
        expected = prove_simple_fixed_point(layout, 3, 40_000)
        assert expected is not None
        assert certificate.average_eu_per_tick == expected.average_eu_per_tick
        assert certificate.safe_game_ticks == expected.safe_game_ticks
        assert certificate.peak_hull_heat == round((1.0 - expected.safety_margin) * certificate.max_hull_heat)
    assert certified > 20


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_cuda_exhaustive_matches_cpu_space_and_leaderboards():
    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    common = dict(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I", "II", "III", "IV", "V"],
        solver="exhaustive",
        result_limit=3,
        cpu_workers=1,
        max_reactor_ticks=40_000,
        gpu_exhaustive_batch_size=256,
    )
    cpu_request = OptimizationRequest(**common, accelerator="cpu")
    cuda_request = OptimizationRequest(**common, accelerator="cuda")

    cpu = _run_exhaustive_shard(cpu_request.model_dump(mode="json"), 0, (), Queue(), Event())
    gpu = _run_exhaustive_shard(cuda_request.model_dump(mode="json"), 0, (), Queue(), Event())

    assert cpu["checked"] == gpu["checked"] == 324
    assert cpu["evaluated"] == gpu["evaluated"] == 324
    assert gpu["gpu_certified"] == 306
    assert gpu["gpu_fallback"] == 18
    assert {
        mark: [candidate.public_dict(3) for candidate in board]
        for mark, board in cpu["boards"].items()
    } == {
        mark: [candidate.public_dict(3) for candidate in board]
        for mark, board in gpu["boards"].items()
    }


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_cuda_mark_i_two_level_exhaustive_uses_gpu_without_changing_proof(tmp_path, monkeypatch):
    monkeypatch.setenv("IC2_SKELETON_TABLE_DB", str(tmp_path / "skeletons.sqlite3"))

    class Queue:
        def put(self, _message):
            pass

    class Event:
        def is_set(self):
            return False

    common = dict(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        result_limit=3,
        cpu_workers=1,
        max_reactor_ticks=40_000,
        gpu_exhaustive_batch_size=256,
    )
    cpu_request = OptimizationRequest(**common, accelerator="cpu")
    cuda_request = OptimizationRequest(**common, accelerator="cuda")

    cpu = _run_exhaustive_shard(cpu_request.model_dump(mode="json"), 0, (), Queue(), Event())
    gpu = _run_exhaustive_shard(cuda_request.model_dump(mode="json"), 0, (), Queue(), Event())

    assert cpu["checked"] == gpu["checked"] == 324
    assert cpu["evaluated"] + cpu["pruned"] == 324
    assert gpu["evaluated"] + gpu["pruned"] == 324
    assert gpu["gpu_certified"] > 0
    assert [candidate.public_dict(3) for candidate in cpu["boards"]["I"]] == [
        candidate.public_dict(3) for candidate in gpu["boards"]["I"]
    ]


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_cuda_exhaustive_job_reports_certificates_and_global_proof():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I", "II", "III", "IV", "V"],
        solver="exhaustive",
        result_limit=1,
        cpu_workers=4,
        max_reactor_ticks=40_000,
        accelerator="cuda",
        gpu_exhaustive_batch_size=256,
    )

    job = OptimizationJob(request)
    job.run()

    assert job.status == "completed"
    assert job.proven_global
    assert job.accelerator == "cuda"
    assert job.checked == 324
    assert job.gpu_exhaustive_certified == 306
    assert job.gpu_exhaustive_fallback == 18
