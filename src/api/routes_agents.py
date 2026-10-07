from fastapi import APIRouter, HTTPException

from .state_store import controller

router = APIRouter()


@router.get("/api/agents")
def agents():
    return [agent.to_dict() for agent in controller.agents.values()]


@router.get("/api/agents/{agent_id}")
def agent(agent_id: str):
    item = controller.agents.get(agent_id)
    if not item:
        raise HTTPException(404, "agent not found")
    return item.to_dict()
