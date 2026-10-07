"""Telemetria JSONL minimale e redatta per la governance semantica."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .schemas import SemanticDecision


_SECRET_MARKERS = ("api_key", "apikey", "authorization", "password", "secret", "token")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): ("[REDACTED]" if any(marker in str(key).lower() for marker in _SECRET_MARKERS)
                       else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


class InMemorySemanticTelemetry:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, decision: SemanticDecision) -> None:
        with self._lock:
            self.rows.append(redact_secrets(decision.to_dict()))


class JsonlSemanticTelemetry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, decision: SemanticDecision) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    redact_secrets(decision.to_dict()), ensure_ascii=False, allow_nan=False
                ) + "\n")
