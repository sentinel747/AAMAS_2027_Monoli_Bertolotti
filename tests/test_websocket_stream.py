"""Il loop websocket non deve mai restare bloccato su un client che non legge.

Un tab del browser congelato (Chrome Memory Saver / tab freezing) mantiene la
connessione TCP viva ma smette di consumare i messaggi: senza timeout il
send_json del server resta sospeso per sempre e lo stream muore in silenzio
mentre il socket risulta OPEN da entrambe le parti (freeze GUI a meta' run).
"""
from __future__ import annotations

import asyncio

from fastapi import WebSocketDisconnect

from src.api import websocket as ws_mod


class StubController:
    initialized = True

    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self._step = 0

    def log_runtime_event(self, event_type: str, message: str, **data) -> None:
        self.events.append((event_type, message))

    def current_state(self) -> dict:
        self._step += 1
        return {"step": self._step, "day": float(self._step)}


class StuckWebSocket:
    """Accetta la connessione ma non consuma mai i messaggi inviati."""

    def __init__(self):
        self.closed = False

    async def accept(self):
        return None

    async def send_json(self, payload):
        await asyncio.sleep(3600)

    async def close(self, code: int = 1000, reason: str | None = None):
        self.closed = True


class DisconnectingWebSocket:
    """Consegna il primo stato, poi il client si disconnette normalmente."""

    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self):
        return None

    async def send_json(self, payload):
        if self.sent:
            raise WebSocketDisconnect(code=1001)
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str | None = None):
        self.closed = True


def test_zombie_client_send_times_out_and_connection_is_closed(monkeypatch):
    stub = StubController()
    monkeypatch.setattr(ws_mod, "controller", stub)
    monkeypatch.setattr(ws_mod, "SEND_TIMEOUT_S", 0.2)
    sock = StuckWebSocket()

    # Senza il timeout questa coroutine non terminerebbe mai (wait_for esterno
    # come cintura di sicurezza del test).
    asyncio.run(asyncio.wait_for(ws_mod.simulation_ws(sock), timeout=5.0))

    assert sock.closed
    assert any(kind == "ws_send_timeout" for kind, _ in stub.events)


def test_normal_disconnect_is_logged_and_loop_exits(monkeypatch):
    stub = StubController()
    monkeypatch.setattr(ws_mod, "controller", stub)
    sock = DisconnectingWebSocket()

    asyncio.run(asyncio.wait_for(ws_mod.simulation_ws(sock), timeout=10.0))

    assert len(sock.sent) == 1
    assert sock.sent[0]["step"] == 1
    assert any(kind == "ws_disconnected" for kind, _ in stub.events)


class IdleController(StubController):
    """Simulazione ferma: lo stato non cambia mai (pausa, fine run, load)."""

    def current_state(self) -> dict:
        return {"step": 42, "day": 42.0, "running": False}


class CountingWebSocket:
    """Consegna N frame poi il client si disconnette."""

    def __init__(self, max_frames: int):
        self.sent: list[dict] = []
        self.max_frames = max_frames

    async def accept(self):
        return None

    async def send_json(self, payload):
        if len(self.sent) >= self.max_frames:
            raise WebSocketDisconnect(code=1001)
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str | None = None):
        return None


def test_idle_stream_sends_heartbeats_and_detects_dead_client(monkeypatch):
    stub = IdleController()
    monkeypatch.setattr(ws_mod, "controller", stub)
    monkeypatch.setattr(ws_mod, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(ws_mod, "HEARTBEAT_EVERY_S", 0.02)
    sock = CountingWebSocket(max_frames=3)

    # Senza heartbeat questo loop non invierebbe mai piu' nulla dopo il primo
    # stato (step/day immutati) e non noterebbe MAI la disconnessione: il test
    # andrebbe in timeout con un handler che gira a vuoto per sempre.
    asyncio.run(asyncio.wait_for(ws_mod.simulation_ws(sock), timeout=5.0))

    assert sock.sent[0]["step"] == 42  # primo frame: stato pieno
    heartbeats = [f for f in sock.sent[1:] if f.get("type") == "ws_heartbeat"]
    assert len(heartbeats) == 2  # poi solo battiti, stato invariato
    assert all(f["step"] == 42 and f["running"] is False for f in heartbeats)
    assert any(kind == "ws_disconnected" for kind, _ in stub.events)
