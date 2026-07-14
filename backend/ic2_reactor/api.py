from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .components import COMPONENTS, RULESET_VERSION
from .engine import ReactorSimulator, SimulationOptions
from .models import OptimizationRequest, SimulationCreated, SimulationRequest
from .optimizer import OptimizationManager, estimate_exhaustive_space
from .trace_store import TraceStore

app = FastAPI(title="IC2 核反应堆模拟与优化器", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

trace_store = TraceStore()
optimization_manager = OptimizationManager()
simulation_summaries: dict[str, dict] = {}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "ruleset": RULESET_VERSION}


@app.get("/api/components")
def components() -> dict:
    return {"ruleset": RULESET_VERSION, "components": [spec.public_dict() for spec in COMPONENTS.values()]}


@app.post("/api/simulations", response_model=SimulationCreated)
def create_simulation(request: SimulationRequest) -> SimulationCreated:
    simulation_id = uuid.uuid4().hex
    run = ReactorSimulator(request.layout).simulate(SimulationOptions(
        max_game_ticks=request.max_game_ticks,
        auto_refuel=request.auto_refuel,
        stop_on_stable=request.stop_on_stable,
        record_components=request.record_components,
    ))
    trace_store.write(simulation_id, run)
    simulation_summaries[simulation_id] = run.summary.model_dump(mode="json")
    return SimulationCreated(id=simulation_id, summary=run.summary)


def _require_simulation(simulation_id: str) -> None:
    if simulation_id not in simulation_summaries or not trace_store.path_for(simulation_id).exists():
        raise HTTPException(status_code=404, detail="模拟结果不存在")


@app.get("/api/simulations/{simulation_id}")
def get_simulation(simulation_id: str) -> dict:
    _require_simulation(simulation_id)
    return {"id": simulation_id, "summary": simulation_summaries[simulation_id]}


@app.get("/api/simulations/{simulation_id}/ticks")
def get_ticks(simulation_id: str, offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=10_000)) -> dict:
    _require_simulation(simulation_id)
    return trace_store.page(simulation_id, offset, limit)


@app.get("/api/simulations/{simulation_id}/components")
def get_component_state(simulation_id: str, game_tick: int = Query(..., ge=0)) -> dict:
    _require_simulation(simulation_id)
    return trace_store.components_at(simulation_id, game_tick)


@app.get("/api/simulations/{simulation_id}/chart")
def get_chart(simulation_id: str, points: int = Query(1200, ge=10, le=5000)) -> dict:
    _require_simulation(simulation_id)
    return trace_store.chart(simulation_id, points)


@app.get("/api/simulations/{simulation_id}/export.csv")
def export_csv(simulation_id: str, components: bool = False) -> StreamingResponse:
    _require_simulation(simulation_id)
    filename = f"ic2-simulation-{simulation_id[:8]}{'-components' if components else ''}.csv"
    return StreamingResponse(
        trace_store.csv_rows(simulation_id, include_components=components),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/optimizations")
def create_optimization(request: OptimizationRequest) -> dict:
    job = optimization_manager.create(request)
    return {"id": job.id, "estimate": str(job.exhaustive_estimate) if job.exhaustive_estimate is not None else None}


@app.post("/api/optimizations/estimate")
def estimate_optimization(request: OptimizationRequest) -> dict:
    """Return an exact inventory-valid enumeration count without starting a job."""
    if request.solver != "exhaustive":
        return {"estimate": None}
    return {"estimate": str(estimate_exhaustive_space(request))}


def _get_job(job_id: str):
    try:
        return optimization_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/optimizations/latest")
def get_latest_optimization() -> dict:
    try:
        return optimization_manager.latest().snapshot()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/optimizations/{job_id}")
def get_optimization(job_id: str) -> dict:
    return _get_job(job_id).snapshot()


@app.post("/api/optimizations/{job_id}/candidates/{mark}/{rank}/simulation", response_model=SimulationCreated)
def simulate_optimization_candidate(job_id: str, mark: str, rank: int) -> SimulationCreated:
    job = _get_job(job_id)
    board = job.leaderboards.get(mark)
    if board is None or rank < 0 or rank >= len(board):
        raise HTTPException(status_code=404, detail="优化候选不存在")
    candidate = board[rank]
    return create_simulation(SimulationRequest(
        layout={"columns": job.request.columns, "initial_hull_heat": 0, "slots": list(candidate.layout)},
        max_game_ticks=job.request.max_reactor_ticks * 20,
        auto_refuel=True,
        stop_on_stable=True,
        record_components=True,
    ))


@app.post("/api/optimizations/{job_id}/cancel")
def cancel_optimization(job_id: str) -> dict:
    job = _get_job(job_id)
    job.cancel()
    return {"status": "cancelling"}


@app.post("/api/optimizations/{job_id}/resume")
def resume_optimization(job_id: str) -> dict:
    try:
        job = optimization_manager.resume(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.snapshot()


@app.get("/api/optimizations/{job_id}/events")
async def optimization_events(job_id: str) -> StreamingResponse:
    job = _get_job(job_id)

    async def stream():
        while True:
            snapshot = job.snapshot()
            yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
            if snapshot["status"] in {"completed", "cancelled", "failed"}:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


static_candidates = [Path("dist/client"), Path(".output/public"), Path("dist")]
static_root = next((path for path in static_candidates if path.exists() and (path / "index.html").exists()), None)
if static_root:
    app.mount("/", StaticFiles(directory=static_root, html=True), name="frontend")
