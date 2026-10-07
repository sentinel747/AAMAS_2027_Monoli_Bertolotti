"""Replay deterministico di decisioni SemIf, senza rete."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schemas import SemanticDecision, SemanticDecisionRequest


class ReplaySemanticDecisionProvider:
    def __init__(self, decisions: Iterable[SemanticDecision], telemetry=None) -> None:
        self._decisions: dict[tuple[str, int, str, str, str], SemanticDecision] = {}
        self._telemetry = telemetry
        for decision in decisions:
            key = decision.replay_key()
            previous = self._decisions.get(key)
            if previous is not None and previous != decision:
                raise ValueError(f"conflicting replay decisions for {key!r}")
            self._decisions[key] = decision

    @classmethod
    def from_jsonl(cls, path: str | Path, telemetry=None):
        decisions = []
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                decisions.append(SemanticDecision.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid semantic replay line {line_number}") from error
        return cls(decisions, telemetry=telemetry)

    def decide(self, request: SemanticDecisionRequest) -> SemanticDecision:
        recorded = self._decisions.get(request.replay_key())
        if recorded is None:
            raise KeyError(
                "semantic replay has no exact match for run, step, actor, "
                "question and input hash"
            )
        replayed = SemanticDecision.from_dict({
            **recorded.to_dict(),
            "route": "replay",
            "rationale": {**recorded.rationale, "replayed_route": recorded.route},
        })
        if self._telemetry is not None:
            self._telemetry.record(replayed)
        return replayed
