from fastapi import APIRouter
import time

from src.experiments.scenarios import list_standard_scenarios
from src.llm.provider_registry import load_provider_registry

from .state_store import controller

router = APIRouter()


@router.post("/api/simulation/start")
def simulation_start():
    controller.start()
    return controller.current_state()


@router.post("/api/simulation/pause")
def simulation_pause():
    controller.pause()
    return controller.current_state()


@router.post("/api/simulation/stop")
def simulation_stop():
    controller.stop()
    return controller.current_state()


@router.post("/api/simulation/step")
def simulation_step(count: int = 1):
    controller.step(count)
    return controller.current_state()


@router.post("/api/simulation/reset")
def simulation_reset():
    controller.reset()
    return controller.current_state()


@router.post("/api/scenarios/load")
def scenario_load(config: dict):
    started = time.perf_counter()
    print(f"[api] scenario_load start name={config.get('name', 'gui_run')}", flush=True)
    controller.reset(config)
    state = controller.current_state()
    print(f"[api] scenario_load done seconds={time.perf_counter() - started:.3f}", flush=True)
    return state


@router.get("/api/scenarios/standard")
def standard_scenarios():
    return {"scenarios": list_standard_scenarios()}


@router.get("/api/llm/providers")
def llm_providers():
    return load_provider_registry()


@router.post("/api/experiments/run")
def experiment_run(config: dict | None = None):
    controller.reset(config or controller.config)
    controller.step(int((config or {}).get("days", 30)))
    return controller.current_state()
