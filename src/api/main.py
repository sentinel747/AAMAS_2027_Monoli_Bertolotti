from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import routes_agents, routes_events, routes_runs, routes_simulation, routes_state, routes_world
from .websocket import simulation_ws

app = FastAPI(title="Mars Artificial Society Simulator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in [
    routes_state.router,
    routes_world.router,
    routes_agents.router,
    routes_events.router,
    routes_simulation.router,
    routes_runs.router,
]:
    app.include_router(router)

app.websocket("/ws/simulation")(simulation_ws)


@app.get("/")
def root():
    return {"name": "Mars Artificial Society Simulator", "status": "ok"}


# Production UI for long runs: the React development build (Vite dev server on
# :5173) logs a performance.measure per component render and the accumulated
# clones exhaust renderer memory after ~13-15 minutes of continuous stepping,
# wedging the page. The production bundle has no such instrumentation: build it
# once (cd frontend && npm run build) and open http://127.0.0.1:8000/ui/.
_ui_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _ui_dist.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_ui_dist), html=True), name="ui")
