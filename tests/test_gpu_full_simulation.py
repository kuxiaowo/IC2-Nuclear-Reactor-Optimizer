import random

import pytest

from ic2_reactor.components import COMPONENT_IDS
from ic2_reactor.gpu_acceleration import cuda_device_info
from ic2_reactor.gpu_full_simulation import CudaFullSimulator
from ic2_reactor.models import FuelConstraint, OptimizationRequest
from ic2_reactor.optimizer import OptimizationJob, _run_exhaustive_shard, evaluate_layout


def make_layout(items: dict[int, str], columns: int = 3) -> tuple[str, ...]:
    layout = ["empty"] * (columns * 6)
    for position, item in items.items():
        layout[position] = item
    return tuple(layout)


class SilentQueue:
    def put(self, _message):
        pass


class NeverCancelled:
    def is_set(self):
        return False


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_full_cuda_simulation_matches_cpu_across_component_families():
    layouts = [
        make_layout({0: "uranium_single", 1: "reactor_heat_vent"}),
        make_layout({0: "uranium_single"}),
        make_layout({0: "uranium_quad"}),
        make_layout({0: "uranium_single", 1: "coolant_10k"}),
        make_layout({4: "uranium_single", 1: "coolant_60k", 7: "coolant_60k"}),
        make_layout({0: "uranium_single", 1: "rsh_condensator"}),
        make_layout({0: "neutron_reflector", 1: "uranium_single"}),
        make_layout({0: "uranium_single", 1: "heat_exchanger"}),
        make_layout({0: "uranium_single", 1: "reactor_heat_exchanger"}),
        make_layout({0: "uranium_single", 1: "overclocked_heat_vent"}),
        make_layout({0: "uranium_single", 1: "heat_capacity_plating"}),
        make_layout({4: "uranium_quad", 3: "iridium_reflector", 5: "coolant_10k"}),
    ]

    actual = CudaFullSimulator(ticks_per_launch=256).simulate(layouts, 3, 40_000)

    assert actual is not None
    for layout, gpu in zip(layouts, actual, strict=True):
        cpu = evaluate_layout(layout, 3, 40_000, use_certificate=False)
        assert (gpu.mark or "未分类") == cpu.mark, layout
        assert gpu.average_eu_per_tick == pytest.approx(cpu.average_eu_per_tick), layout
        assert gpu.safe_game_ticks == cpu.safe_game_ticks, layout
        gpu_margin = 1.0 - gpu.peak_hull_heat / gpu.max_hull_heat
        assert gpu_margin == pytest.approx(cpu.safety_margin), layout


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_full_cuda_random_mixed_layouts_match_cpu():
    rng = random.Random(221)
    placeable = [item for item in COMPONENT_IDS if item != "empty"]
    layouts = []
    for _ in range(96):
        layout = [
            rng.choice(placeable) if rng.random() < 0.32 else "empty"
            for _ in range(18)
        ]
        if not any(item.startswith("uranium_") for item in layout):
            layout[rng.randrange(18)] = rng.choice(
                ["uranium_single", "uranium_dual", "uranium_quad"]
            )
        layouts.append(tuple(layout))

    actual = CudaFullSimulator(ticks_per_launch=128).simulate(layouts, 3, 2_000)

    assert actual is not None
    for layout, gpu in zip(layouts, actual, strict=True):
        cpu = evaluate_layout(layout, 3, 2_000, use_certificate=False)
        assert (gpu.mark or "未分类") == cpu.mark, layout
        assert gpu.average_eu_per_tick == pytest.approx(cpu.average_eu_per_tick), layout
        assert gpu.safe_game_ticks == cpu.safe_game_ticks, layout
        gpu_margin = 1.0 - gpu.peak_hull_heat / gpu.max_hull_heat
        assert gpu_margin == pytest.approx(cpu.safety_margin), layout


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_full_cuda_long_random_sequences_match_cpu():
    rng = random.Random(8221)
    placeable = [item for item in COMPONENT_IDS if item != "empty"]
    layouts = []
    for _ in range(32):
        layout = [
            rng.choice(placeable) if rng.random() < 0.24 else "empty"
            for _ in range(18)
        ]
        layout[rng.randrange(18)] = rng.choice(
            ["uranium_single", "uranium_dual", "uranium_quad"]
        )
        layouts.append(tuple(layout))

    actual = CudaFullSimulator(ticks_per_launch=256).simulate(layouts, 3, 40_000)

    assert actual is not None
    for layout, gpu in zip(layouts, actual, strict=True):
        cpu = evaluate_layout(layout, 3, 40_000, use_certificate=False)
        assert (gpu.mark or "未分类") == cpu.mark, layout
        assert gpu.average_eu_per_tick == pytest.approx(cpu.average_eu_per_tick), layout
        assert gpu.safe_game_ticks == cpu.safe_game_ticks, layout
        gpu_margin = 1.0 - gpu.peak_hull_heat / gpu.max_hull_heat
        assert gpu_margin == pytest.approx(cpu.safety_margin), layout


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_full_cuda_six_by_nine_layouts_match_cpu():
    rng = random.Random(9054)
    placeable = [item for item in COMPONENT_IDS if item != "empty"]
    layouts = []
    for _ in range(24):
        layout = [
            rng.choice(placeable) if rng.random() < 0.18 else "empty"
            for _ in range(54)
        ]
        layout[rng.randrange(54)] = rng.choice(
            ["uranium_single", "uranium_dual", "uranium_quad"]
        )
        layouts.append(tuple(layout))

    actual = CudaFullSimulator(ticks_per_launch=128).simulate(layouts, 9, 2_000)

    assert actual is not None
    for layout, gpu in zip(layouts, actual, strict=True):
        cpu = evaluate_layout(layout, 9, 2_000, use_certificate=False)
        assert (gpu.mark or "未分类") == cpu.mark, layout
        assert gpu.average_eu_per_tick == pytest.approx(cpu.average_eu_per_tick), layout
        assert gpu.safe_game_ticks == cpu.safe_game_ticks, layout
        gpu_margin = 1.0 - gpu.peak_hull_heat / gpu.max_hull_heat
        assert gpu_margin == pytest.approx(cpu.safety_margin), layout


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
@pytest.mark.parametrize("marks", [["I", "II", "III", "IV", "V"], ["I"]])
def test_full_cuda_exhaustive_matches_cpu_space_and_leaderboards(
    marks, tmp_path, monkeypatch
):
    monkeypatch.setenv("IC2_SKELETON_TABLE_DB", str(tmp_path / "skeletons.sqlite3"))
    common = dict(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=marks,
        solver="exhaustive",
        result_limit=3,
        cpu_workers=1,
        max_reactor_ticks=40_000,
        gpu_exhaustive_batch_size=256,
    )
    cpu_request = OptimizationRequest(**common, accelerator="cpu")
    gpu_request = OptimizationRequest(**common, accelerator="cuda_full")

    cpu = _run_exhaustive_shard(
        cpu_request.model_dump(mode="json"), 0, (), SilentQueue(), NeverCancelled()
    )
    gpu = _run_exhaustive_shard(
        gpu_request.model_dump(mode="json"), 0, (), SilentQueue(), NeverCancelled()
    )

    assert cpu["checked"] == gpu["checked"] == 324
    assert gpu["gpu_full_simulated"] > 0
    assert gpu["gpu_certified"] == gpu["gpu_fallback"] == 0
    assert {
        mark: [candidate.public_dict(3) for candidate in board]
        for mark, board in cpu["boards"].items()
    } == {
        mark: [candidate.public_dict(3) for candidate in board]
        for mark, board in gpu["boards"].items()
    }


@pytest.mark.skipif(not cuda_device_info().available, reason="CUDA runtime unavailable")
def test_full_cuda_exhaustive_job_reports_global_proof():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I", "II", "III", "IV", "V"],
        solver="exhaustive",
        result_limit=1,
        cpu_workers=4,
        max_reactor_ticks=40_000,
        accelerator="cuda_full",
        gpu_exhaustive_batch_size=256,
    )

    job = OptimizationJob(request)
    job.run()

    assert job.status == "completed"
    assert job.proven_global
    assert job.accelerator == "cuda_full"
    assert job.checked == 324
    assert job.gpu_full_simulated == 324
    assert job.gpu_exhaustive_certified == job.gpu_exhaustive_fallback == 0
