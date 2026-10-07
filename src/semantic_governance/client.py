"""Provider SemIf: protocollo, fake deterministico e client REST fail-closed."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .schemas import (
    FALLBACK_OPTIONS,
    SemanticDecision,
    SemanticDecisionRequest,
)


class SemanticDecisionProvider(Protocol):
    def decide(self, request: SemanticDecisionRequest) -> SemanticDecision:
        ...


DecisionFactory = Callable[[SemanticDecisionRequest], SemanticDecision]
Transport = Callable[[str, bytes, Mapping[str, str], float], tuple[int, bytes, Mapping[str, str]]]


def fallback_decision(
    request: SemanticDecisionRequest,
    reason: str,
    *,
    latency_ms: float = 0.0,
    runtime: str = "jev-semif-rest",
) -> SemanticDecision:
    fallback = next(option for option in request.option_ids if option in FALLBACK_OPTIONS)
    probabilities = {option: 0.0 for option in request.option_ids}
    probabilities[fallback] = 1.0
    return SemanticDecision(
        run_id=request.run_id,
        step=request.step,
        actor=request.actor,
        question_id=request.question_id,
        input_hash=request.input_hash,
        selected_option=fallback,
        probabilities=probabilities,
        confidence=0.0,
        route="fallback",
        rationale={"status": "fallback"},
        runtime=runtime,
        latency_ms=max(0.0, latency_ms),
        fallback_reason=reason,
    )


class FakeSemanticDecisionProvider:
    """Provider senza rete, configurabile per chiave o con una factory pura."""

    def __init__(
        self,
        decisions: Mapping[str, str] | None = None,
        factory: DecisionFactory | None = None,
    ) -> None:
        self._decisions = dict(decisions or {})
        self._factory = factory
        self.requests: list[SemanticDecisionRequest] = []

    def decide(self, request: SemanticDecisionRequest) -> SemanticDecision:
        self.requests.append(request)
        if self._factory is not None:
            decision = self._factory(request)
            if decision.replay_key() != request.replay_key():
                raise ValueError("fake decision does not match the request identity")
            return decision
        selected = self._decisions.get(request.question_id, request.option_ids[0])
        if selected not in request.option_ids:
            return fallback_decision(request, "fake_option_not_available", runtime="fake")
        probabilities = {option: float(option == selected) for option in request.option_ids}
        route = "fallback" if selected in FALLBACK_OPTIONS else (
            "deep" if selected == "deep_reasoning" else "filter"
        )
        return SemanticDecision(
            run_id=request.run_id,
            step=request.step,
            actor=request.actor,
            question_id=request.question_id,
            input_hash=request.input_hash,
            selected_option=selected,
            probabilities=probabilities,
            confidence=1.0,
            route=route,
            rationale={"source": "fake"},
            runtime="fake",
        )


class RestSemanticDecisionProvider:
    """Client REST JEV con retry limitato e circuit breaker in memoria.

    Non registra URL, header, payload o testo delle eccezioni. La chiave viene
    risolta esclusivamente dall'environment al momento della chiamata.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key_env: str = "",
        timeout_seconds: float = 10.0,
        max_retries: int = 1,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 30.0,
        min_confidence: float = 0.65,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
        telemetry=None,
        temperature=None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must use http or https")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and retries non-negative")
        if circuit_failure_threshold < 1 or circuit_cooldown_seconds < 0:
            raise ValueError("invalid circuit breaker settings")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self._endpoint = endpoint.rstrip("/") + "/v1/decide"
        self._api_key_env = api_key_env
        self._timeout = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._failure_threshold = int(circuit_failure_threshold)
        self._cooldown = float(circuit_cooldown_seconds)
        self._min_confidence = float(min_confidence)
        self._transport = transport or _urlopen_transport
        self._clock = clock
        self._telemetry = telemetry
        #: Calibrazione a temperatura (src/semantic_governance/calibration.py):
        #: `None` o 1 lasciano la distribuzione del gateway intatta.
        self._temperature = temperature
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        # Lo strato degli agenti chiama `decide` da piu' thread: lo stato del
        # circuit breaker si legge e si scrive sotto lock. La rete resta fuori.
        self._breaker_lock = threading.Lock()

    def decide(self, request: SemanticDecisionRequest) -> SemanticDecision:
        started = self._clock()
        with self._breaker_lock:
            aperto = False
            if self._opened_at is not None:
                if self._clock() - self._opened_at < self._cooldown:
                    aperto = True
                else:
                    self._opened_at = None
                    self._consecutive_failures = 0
        if aperto:
            return self._emit(fallback_decision(request, "circuit_open"))

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key_env:
            secret = os.getenv(self._api_key_env, "")
            if not secret:
                return self._emit(fallback_decision(request, "missing_api_key_env"))
            headers["Authorization"] = f"Bearer {secret}"
        payload = json.dumps(
            request.wire_payload(), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")

        reason = "transport_error"
        for _attempt in range(self._max_retries + 1):
            try:
                status, body, response_headers = self._transport(
                    self._endpoint, payload, headers, self._timeout
                )
                if status < 200 or status >= 300:
                    reason = f"http_status_{status}"
                    continue
                decision = self._parse_response(
                    request,
                    body,
                    response_headers,
                    latency_ms=(self._clock() - started) * 1_000.0,
                )
            except (TimeoutError, socket.timeout):
                reason = "timeout"
                continue
            except urllib.error.HTTPError as error:
                # urllib trasforma gli status non-2xx in eccezioni prima che
                # il trasporto possa restituirli. Conservare il codice rende
                # il fallback diagnosticabile senza copiare body o URL.
                reason = f"http_status_{int(error.code)}"
                continue
            except (OSError, urllib.error.URLError):
                reason = "transport_error"
                continue
            except (KeyError, TypeError, ValueError, UnicodeDecodeError):
                reason = "invalid_response"
                break
            except Exception:  # noqa: BLE001 - un trasporto iniettato non ferma la run
                reason = "transport_error"
                continue
            with self._breaker_lock:
                self._consecutive_failures = 0
                self._opened_at = None
            if decision.confidence < self._min_confidence:
                return self._emit(fallback_decision(
                    request,
                    "low_confidence",
                    latency_ms=decision.latency_ms,
                ))
            return self._emit(decision)

        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._opened_at = self._clock()
        return self._emit(fallback_decision(
            request,
            reason,
            latency_ms=(self._clock() - started) * 1_000.0,
        ))

    def _parse_response(
        self,
        request: SemanticDecisionRequest,
        body: bytes,
        headers: Mapping[str, str],
        *,
        latency_ms: float,
    ) -> SemanticDecision:
        raw = json.loads(body.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("response is not an object")
        selected = str(raw["selected_option"])
        probabilities = {str(k): float(v) for k, v in dict(raw["probabilities"]).items()}
        if set(probabilities) != set(request.option_ids) or selected not in request.option_ids:
            raise ValueError("response options differ from request options")
        confidence = float(raw["confidence"])
        inference_ms = float(raw["inference_time_ms"])
        if self._temperature is not None:
            from .calibration import normalized_confidence, temper, temperature_for

            t = temperature_for(request.question_id, self._temperature)
            if t != 1.0:
                # La preferenza non cambia, cambia quanto e' netta.
                probabilities = temper(probabilities, t)
                confidence = normalized_confidence(probabilities)
        route = "fallback" if selected in FALLBACK_OPTIONS else (
            "deep" if selected == "deep_reasoning" else "filter"
        )
        request_id = str(headers.get("X-Request-ID", headers.get("x-request-id", "")))
        return SemanticDecision(
            run_id=request.run_id,
            step=request.step,
            actor=request.actor,
            question_id=request.question_id,
            input_hash=request.input_hash,
            selected_option=selected,
            probabilities=probabilities,
            confidence=confidence,
            route=route,
            rationale={"source": "jev"},
            runtime="jev-semif-rest",
            latency_ms=max(latency_ms, inference_ms),
            fallback_reason="selected_fallback" if selected in FALLBACK_OPTIONS else "",
            request_id=request_id,
        )

    def _emit(self, decision: SemanticDecision) -> SemanticDecision:
        if self._telemetry is not None:
            self._telemetry.record(decision)
        return decision


def _urlopen_transport(
    url: str,
    payload: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(url, data=payload, headers=dict(headers), method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return int(response.status), response.read(), dict(response.headers.items())
