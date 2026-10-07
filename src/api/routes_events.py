from fastapi import APIRouter

from .state_store import controller

router = APIRouter()


@router.get("/api/events/recent")
def events_recent(limit: int = 50):
    if not controller.initialized:
        return []
    return controller.world.events[-limit:]


@router.get("/api/conversations/recent")
def conversations_recent(limit: int = 50):
    if not controller.initialized:
        return []
    return [e for e in controller.world.events if e.get("type") == "conversation"][-limit:]


@router.get("/api/thoughts/recent")
def thoughts_recent(limit: int = 50):
    if not controller.initialized:
        return []
    return [{"day": e.get("day"), "agent_id": e.get("data", {}).get("agent_id"), "thought_summary": e.get("message")} for e in controller.world.events[-limit:]]


@router.get("/api/social-network")
def social_network():
    if not controller.initialized:
        return {"nodes": [], "edges": []}
    return controller.social.to_dict()
