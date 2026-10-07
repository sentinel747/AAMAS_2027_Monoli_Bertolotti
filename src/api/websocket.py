from __future__ import annotations

import asyncio
import time

from fastapi import WebSocket, WebSocketDisconnect

from .state_store import controller

# A frozen/suspended browser tab keeps its TCP connection alive (the browser's
# network process still answers protocol pings) but stops draining messages:
# without a timeout send_json blocks forever and the stream silently dies
# mid-run while the socket still looks OPEN on both sides.
SEND_TIMEOUT_S = 15.0
# When the simulation is idle (scenario load, pause, finished run) the state
# stops changing and no frames would flow at all. The heartbeat keeps the
# client's liveness clock fresh (no stale-detector reconnect churn) and makes
# a dead client raise on send within seconds instead of leaking a handler
# task that polls current_state() forever.
HEARTBEAT_EVERY_S = 3.0
POLL_INTERVAL_S = 1.0


async def simulation_ws(websocket: WebSocket):
    await websocket.accept()
    if controller.initialized:
        controller.log_runtime_event("ws_connected", "GUI realtime stream connected")
    last_step = -1
    last_day = -1.0
    last_sent = time.monotonic()
    try:
        while True:
            # current_state() takes the simulation step lock and can block for a
            # full step; keep it off the event loop so the API stays responsive.
            current = await asyncio.to_thread(controller.current_state)
            step = current.get("step", 0)
            day = current.get("day", 0.0)

            # Send if something changed or first time
            if step != last_step or day != last_day or last_step == -1:
                await asyncio.wait_for(websocket.send_json(current), timeout=SEND_TIMEOUT_S)
                last_step = step
                last_day = day
                last_sent = time.monotonic()
            elif time.monotonic() - last_sent >= HEARTBEAT_EVERY_S:
                heartbeat = {
                    "type": "ws_heartbeat",
                    "step": step,
                    "day": day,
                    "running": bool(current.get("running", False)),
                }
                await asyncio.wait_for(websocket.send_json(heartbeat), timeout=SEND_TIMEOUT_S)
                last_sent = time.monotonic()

            await asyncio.sleep(POLL_INTERVAL_S)
    except (asyncio.TimeoutError, TimeoutError):
        # Zombie client: drop the connection so the tab reconnects cleanly (and
        # immediately receives the current state) once the browser wakes it up.
        if controller.initialized:
            controller.log_runtime_event(
                "ws_send_timeout",
                "GUI realtime stream stalled (client not draining); closing zombie connection",
            )
        try:
            await asyncio.wait_for(websocket.close(), timeout=5.0)
        except Exception:
            pass
    except WebSocketDisconnect:
        if controller.initialized:
            controller.log_runtime_event("ws_disconnected", "GUI realtime stream disconnected")
    except Exception as e:
        if controller.initialized:
            controller.log_runtime_event("ws_error", f"WebSocket error: {str(e)}")
