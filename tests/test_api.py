import asyncio

import httpx

from ic2_reactor.api import app


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
