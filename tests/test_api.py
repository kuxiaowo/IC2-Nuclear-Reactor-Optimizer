import asyncio

import httpx

from ic2_reactor.api import app, optimization_manager
from ic2_reactor.models import OptimizationRequest
from ic2_reactor.optimizer import OptimizationJob


def test_exhaustive_estimate_endpoint_returns_decimal_string_without_starting_job():
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/optimizations/estimate", json={
                "columns": 3,
                "fuel": {"mode": "separate", "single": 1, "dual": 0, "quad": 0, "total_rods": 1},
                "component_limits": {"heat_vent": 1},
                "marks": ["I"],
                "solver": "exhaustive",
            })

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"estimate": "324"}


def test_exhaustive_estimate_endpoint_distinguishes_exact_and_maximum_fuel():
    async def estimate(usage: str):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/optimizations/estimate", json={
                "columns": 3,
                "fuel": {
                    "mode": "separate",
                    "usage": usage,
                    "single": 1,
                    "dual": 1,
                    "quad": 0,
                    "total_rods": 1,
                },
                "component_limits": {},
                "marks": ["I"],
                "solver": "exhaustive",
            })

    exact = asyncio.run(estimate("exact"))
    maximum = asyncio.run(estimate("maximum"))

    assert exact.json() == {"estimate": "306"}
    assert maximum.json() == {"estimate": "342"}


def test_simulation_rejects_partial_reactor_cycle_tick_limit():
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/simulations", json={
                "layout": {"columns": 3, "slots": ["uranium_single", *(["empty"] * 17)]},
                "max_game_ticks": 21,
            })

    response = asyncio.run(request())
    assert response.status_code == 422
    assert "20 的倍数" in response.text


def test_optimization_pause_and_resume_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr("ic2_reactor.optimizer.CHECKPOINT_DIRECTORY", tmp_path)
    job = OptimizationJob(OptimizationRequest())
    job.status = "running"
    with optimization_manager.lock:
        optimization_manager.jobs[job.id] = job

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            paused = await client.post(f"/api/optimizations/{job.id}/pause")
            resumed = await client.post(f"/api/optimizations/{job.id}/resume")
            return paused, resumed

    try:
        paused, resumed = asyncio.run(request())
    finally:
        with optimization_manager.lock:
            optimization_manager.jobs.pop(job.id, None)

    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "running"
