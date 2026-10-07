from fastapi import APIRouter

from .state_store import controller

router = APIRouter()


@router.get("/api/state/current")
def current_state():
    return controller.current_state()


@router.get("/api/metrics/timeseries")
def metrics_timeseries():
    if not controller.initialized:
        return [{"day": 0, **controller.current_state()["metrics"]}]
    return [{"day": controller.world.day, **controller.current_state()["metrics"]}]
