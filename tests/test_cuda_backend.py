import random

import numpy as np
import pytest

from ic2_reactor.components import COMPONENT_IDS
from ic2_reactor.compute_backend import ScalarPackedEvaluator
from ic2_reactor.cuda_backend import CudaPackedEvaluator, cuda_available
from ic2_reactor.kernel_abi import pack_layouts
from ic2_reactor.models import FuelConstraint, OptimizationRequest
from ic2_reactor.optimizer import OptimizationJob, evaluate_layout_batch


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")


def assert_cuda_equal(layouts, columns, max_reactor_ticks, initial_hull_heat=0):
    batch = pack_layouts(layouts, columns, initial_hull_heat)
    expected = ScalarPackedEvaluator().evaluate(batch, max_reactor_ticks)
    actual = CudaPackedEvaluator().evaluate(batch, max_reactor_ticks)
    for field in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_cuda_backend_matches_scalar_on_semantic_boundaries():
    empty = ["empty"] * 18
    layouts = []

    stable = empty.copy()
    stable[:2] = ["uranium_single", "reactor_heat_vent"]
    layouts.append(tuple(stable))

    meltdown = empty.copy()
    meltdown[4] = "uranium_quad"
    layouts.append(tuple(meltdown))

    exchanger = empty.copy()
    exchanger[:4] = [
        "reactor_heat_exchanger",
        "heat_capacity_plating",
        "uranium_dual",
        "coolant_10k",
    ]
    layouts.append(tuple(exchanger))

    reflector = empty.copy()
    reflector[3:6] = [
        "uranium_single",
        "neutron_reflector",
        "component_heat_vent",
    ]
    layouts.append(tuple(reflector))

    condensator = empty.copy()
    condensator[4] = "uranium_single"
    for position in (1, 3, 5, 7):
        condensator[position] = "lzh_condensator"
    condensator[17] = "reactor_heat_vent"
    layouts.append(tuple(condensator))

    assert_cuda_equal(layouts, 3, 40_000, [0, 0, 25, 0, 0])


def test_cuda_backend_matches_scalar_on_seeded_random_layouts():
    rng = random.Random(221)
    components = list(COMPONENT_IDS)
    layouts = [
        tuple(rng.choice(components) if rng.random() < 0.55 else "empty" for _ in range(18))
        for _ in range(64)
    ]
    initial_heat = [rng.randrange(0, 8_000) for _ in layouts]
    assert_cuda_equal(layouts, 3, 4_000, initial_heat)


def test_cuda_backend_matches_late_stable_suc_layout():
    layout = ["empty"] * 18
    layout[4] = "uranium_single"
    for position in (1, 3, 5, 7):
        layout[position] = "lzh_condensator"
    layout[17] = "reactor_heat_vent"
    assert_cuda_equal([tuple(layout)], 3, 140_000)


def test_cuda_backend_matches_scalar_at_maximum_width():
    rng = random.Random(8221)
    components = list(COMPONENT_IDS)
    layouts = [
        tuple(rng.choice(components) if rng.random() < 0.35 else "empty" for _ in range(54))
        for _ in range(16)
    ]
    assert_cuda_equal(layouts, 9, 2_500)


def test_cuda_search_boundary_returns_candidate_results(monkeypatch):
    layouts = (
        tuple(["uranium_single", "reactor_heat_vent", *(["empty"] * 16)]),
        tuple(["uranium_quad", *(["empty"] * 17)]),
    )
    monkeypatch.setattr("ic2_reactor.optimizer.CUDA_MIN_BATCH_SIZE", 1)
    expected = evaluate_layout_batch(layouts, 3, 40_000)
    actual = evaluate_layout_batch(layouts, 3, 40_000, False, None, "cuda", 2)
    assert actual == expected


def test_cuda_failure_recomputes_batch_on_numba(monkeypatch):
    layouts = (tuple(["uranium_single", *(["empty"] * 17)]),)

    def fail(_self, _batch, _max_reactor_ticks):
        raise RuntimeError("synthetic CUDA failure")

    monkeypatch.setattr("ic2_reactor.optimizer.CUDA_MIN_BATCH_SIZE", 1)
    monkeypatch.setattr(CudaPackedEvaluator, "evaluate", fail)
    with pytest.warns(RuntimeWarning, match="falling back"):
        actual = evaluate_layout_batch(layouts, 3, 2_000, False, None, "cuda", 2)
    expected = evaluate_layout_batch(layouts, 3, 2_000)
    assert actual == expected


def test_cuda_exhaustive_worker_preserves_counts_and_proof_state():
    request = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={},
        marks=["I", "II", "III", "IV", "V"],
        solver="exhaustive",
        cpu_workers=8,
        compute_backend="cuda",
        max_reactor_ticks=2_000,
    )
    job = OptimizationJob(request)
    job.run()

    assert job.status == "completed"
    assert (job.enumeration_processes, job.simulation_processes) == (8, 1)
    assert job.proven_within_horizon
    assert not job.proven_global
    assert (job.checked, job.evaluated, job.pruned, job.unresolved) == (18, 18, 0, 18)


def test_cuda_parallel_pruning_matches_scalar_mark_i_leaderboard():
    base = OptimizationRequest(
        columns=3,
        fuel=FuelConstraint(mode="separate", single=1, dual=0, quad=0),
        component_limits={"reactor_heat_vent": 1},
        marks=["I"],
        solver="exhaustive",
        cpu_workers=4,
        max_reactor_ticks=40_000,
    )
    scalar = OptimizationJob(base)
    cuda = OptimizationJob(base.model_copy(update={"compute_backend": "cuda"}))

    scalar.run()
    cuda.run()

    assert scalar.status == cuda.status == "completed"
    assert scalar.proven_global and cuda.proven_global
    assert scalar.checked == cuda.checked == 324
    assert scalar.evaluated + scalar.pruned == scalar.checked
    assert cuda.evaluated + cuda.pruned == cuda.checked
    assert (cuda.enumeration_processes, cuda.simulation_processes) == (4, 1)
    assert [
        (result.canonical, result.average_eu_per_tick)
        for result in cuda.leaderboards["I"]
    ] == [
        (result.canonical, result.average_eu_per_tick)
        for result in scalar.leaderboards["I"]
    ]
