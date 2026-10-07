"""Schemi versionati per decisioni SemIf bounded.

Il wire format di JEV resta quello del runtime separato. I campi di provenienza
e riproducibilita' vivono nel contratto interno e non vengono inviati al
motore, salvo stato, domanda e alternative.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"
FALLBACK_OPTIONS = frozenset({"unknown", "none_of_the_above"})
MAX_OPTIONS = 16
#: Limite del provider ufficiale TypeSafe (`Choice` fino a 255 opzioni).
#: `MAX_OPTIONS` resta il default: nasce dagli slot `A`-`P` del gateway locale.
MAX_OPTIONS_TYPESAFE = 255
VALID_ROUTES = frozenset({"lookup", "filter", "deep", "fallback", "replay"})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class SemanticOption:
    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip() or len(self.id) > 128:
            raise ValueError("option id must contain 1..128 characters")
        if not self.description.strip() or len(self.description) > 2_000:
            raise ValueError("option description must contain 1..2000 characters")

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "description": self.description}


@dataclass(frozen=True)
class SemanticDecisionRequest:
    run_id: str
    step: int
    actor: str
    question_id: str
    state: str | dict[str, Any] | list[Any]
    question: str
    options: tuple[SemanticOption, ...]
    schema_version: str = SCHEMA_VERSION
    input_hash: str = ""
    #: Limite di opzioni applicabile al provider di destinazione. Non entra
    #: nel wire payload ne' nell'hash: cambia la validazione, non la domanda.
    max_options: int = MAX_OPTIONS

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if not self.run_id.strip() or not self.actor.strip() or not self.question_id.strip():
            raise ValueError("run_id, actor and question_id must not be empty")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.question.strip() or len(self.question) > 4_000:
            raise ValueError("question must contain 1..4000 characters")
        if not 2 <= int(self.max_options) <= MAX_OPTIONS_TYPESAFE:
            raise ValueError(
                f"max_options must be between 2 and {MAX_OPTIONS_TYPESAFE}"
            )
        if not 2 <= len(self.options) <= int(self.max_options):
            raise ValueError(f"options must contain 2..{int(self.max_options)} items")
        option_ids = [option.id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("option ids must be unique")
        if FALLBACK_OPTIONS.isdisjoint(option_ids):
            raise ValueError("options must include unknown or none_of_the_above")
        if isinstance(self.state, str):
            if not self.state.strip():
                raise ValueError("state must not be empty")
        elif isinstance(self.state, (dict, list)):
            if not self.state:
                raise ValueError("structured state must not be empty")
        else:
            raise TypeError("state must be a string, mapping or list")

        digest = hashlib.sha256(_canonical_json(self.wire_payload()).encode("utf-8")).hexdigest()
        if self.input_hash and self.input_hash != digest:
            raise ValueError("input_hash does not match the canonical request")
        object.__setattr__(self, "input_hash", digest)

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(option.id for option in self.options)

    def wire_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "question": self.question,
            "options": [option.to_dict() for option in self.options],
        }

    def replay_key(self) -> tuple[str, int, str, str, str]:
        return self.run_id, self.step, self.actor, self.question_id, self.input_hash


@dataclass(frozen=True)
class SemanticDecision:
    run_id: str
    step: int
    actor: str
    question_id: str
    input_hash: str
    selected_option: str
    probabilities: Mapping[str, float]
    confidence: float
    route: str
    schema_version: str = SCHEMA_VERSION
    rationale: Mapping[str, Any] = field(default_factory=dict)
    runtime: str = ""
    model: str = ""
    latency_ms: float = 0.0
    fallback_reason: str = ""
    request_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if not self.run_id.strip() or not self.actor.strip() or not self.question_id.strip():
            raise ValueError("run_id, actor and question_id must not be empty")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.input_hash or not self.selected_option:
            raise ValueError("input_hash and selected_option must not be empty")
        if self.route not in VALID_ROUTES:
            raise ValueError(f"unsupported route {self.route!r}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("latency_ms must be finite and non-negative")
        probabilities = {str(key): float(value) for key, value in self.probabilities.items()}
        if self.selected_option not in probabilities:
            raise ValueError("selected_option must be present in probabilities")
        if not probabilities or any(
            not math.isfinite(value) or value < 0.0 for value in probabilities.values()
        ):
            raise ValueError("probabilities must be finite and non-negative")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "rationale", dict(self.rationale))

    def replay_key(self) -> tuple[str, int, str, str, str]:
        return self.run_id, self.step, self.actor, self.question_id, self.input_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "step": self.step,
            "actor": self.actor,
            "question_id": self.question_id,
            "input_hash": self.input_hash,
            "selected_option": self.selected_option,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "rationale": dict(self.rationale),
            "route": self.route,
            "runtime": self.runtime,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "fallback_reason": self.fallback_reason,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SemanticDecision":
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            run_id=str(raw.get("run_id", "")),
            step=int(raw.get("step", -1)),
            actor=str(raw.get("actor", "")),
            question_id=str(raw.get("question_id", "")),
            input_hash=str(raw.get("input_hash", "")),
            selected_option=str(raw.get("selected_option", "")),
            probabilities=dict(raw.get("probabilities") or {}),
            confidence=float(raw.get("confidence", -1.0)),
            rationale=dict(raw.get("rationale") or {}),
            route=str(raw.get("route", "")),
            runtime=str(raw.get("runtime", "")),
            model=str(raw.get("model", "")),
            latency_ms=float(raw.get("latency_ms", 0.0)),
            fallback_reason=str(raw.get("fallback_reason", "")),
            request_id=str(raw.get("request_id", "")),
        )
