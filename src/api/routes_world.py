from fastapi import APIRouter, HTTPException

from .state_store import controller

router = APIRouter()


@router.get("/api/world/chunk")
def world_chunk(x: int = 0, y: int = 0, zoom: int = 1, size: int = 2000):
    return controller.read_world().compact_chunk(x=x, y=y, zoom=zoom, size=size)


@router.get("/api/world/render_map")
def world_render_map(include_static: int = 0):
    return controller.read_world().render_map_payload(include_static=bool(include_static))


@router.get("/api/world/cell/{x}/{y}")
def world_cell(x: int, y: int):
    # **Coordinate fuori dai limiti rispondono 404, non sollevano (2026-08-30).**
    # `get_cell` alza `IndexError`, che qui diventava un 500: una richiesta
    # sbagliata del client faceva sembrare rotto il server. Misurato con
    # `/api/world/cell/0/80` su un mondo piu' piccolo.
    mondo = controller.read_world()
    if not mondo.in_bounds(x, y):
        raise HTTPException(
            status_code=404,
            detail=f"cella ({x}, {y}) fuori dal mondo {mondo.width}x{mondo.height}",
        )
    return mondo.get_cell(x, y).to_public_dict()
