"""Provider Jev ufficiale (TypeSafe), fail-closed e con registro di spesa.

Mai una richiesta senza chiave, mai una richiesta oltre il tetto. Piu' decisioni
viaggiano nello stesso corpo, una domanda per decisione: TypeSafe le valuta in
parallelo. Chiavi e testo delle eccezioni non entrano mai nelle decisioni.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlparse

from .budget import BudgetExceeded, SpendingLedger
from .client import FALLBACK_OPTIONS, Transport, _urlopen_transport, fallback_decision
from .schemas import SemanticDecision, SemanticDecisionRequest

RUNTIME = "jev-typesafe"
#: TypeSafe: stato + domanda piu' lunga <= ~32k token. Misura del 2026-09-23 su
#: stati di amministratore impacchettati: 27 casi (43.617 token reali) passano,
#: 32 ricevono 400 `max_tokens_exceeded`; blocchi a 30k stimati stavano sul
#: bordo. 24k stimati sono circa 22k reali: margine ampio.
MAX_STATE_TOKENS = 24_000
#: Host per cui `http://` e' ammesso (proxy locali come LiteLLM): altrove la
#: chiave viaggerebbe in chiaro.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
MAX_RETRY_AFTER_SECONDS = 30.0


class TypeSafeSemanticDecisionProvider:
    def __init__(
        self,
        *,
        ledger: SpendingLedger | None,
        endpoint: str = "https://api.typesafe.ai",
        api_key_env: str = "TYPESAFE_AI",
        model: str = "jev-latest",
        question_type: str = "choice",
        timeout_seconds: float = 15.0,
        min_confidence: float = 0.65,
        max_prompt_tokens: int = 50_000,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("endpoint must be an absolute http(s) URL")
        if (
            parsed.scheme == "http"
            and parsed.hostname not in _LOCAL_HOSTS
            and api_key_env.strip()
        ):
            # Una chiave non viaggia mai in chiaro fuori da localhost. Senza
            # chiave (server Laya su una macchina privata) http e' ammesso.
            raise ValueError(
                "endpoint must use https (http only for localhost): the API key "
                "would otherwise travel in cleartext"
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.ledger = ledger
        self._runtime = RUNTIME
        self._url = endpoint.rstrip("/") + "/v1/systemone"
        self._api_key_env = api_key_env
        self._model = model
        self._question_type = question_type
        self._timeout = float(timeout_seconds)
        self._min_confidence = float(min_confidence)
        self._max_prompt_tokens = int(max_prompt_tokens)
        self._transport = transport or _urlopen_transport
        self._sleep = sleep
        self._clock = clock

    # -- interfaccia ---------------------------------------------------------
    def decide(self, request: SemanticDecisionRequest) -> SemanticDecision:
        return self.decide_many([request])[0]

    def decide_many(
        self, requests: Sequence[SemanticDecisionRequest]
    ) -> list[SemanticDecision]:
        results: list[SemanticDecision] = []
        for chunk in self._chunks(list(requests)):
            results.extend(self._decide_chunk(chunk))
        return results

    # -- impacchettamento ----------------------------------------------------
    @staticmethod
    def _estimate_tokens(body: Mapping) -> int:
        # Caratteri/2: sugli stati numerici i token reali sono ~1,3 volte
        # caratteri/3 (misura del 2026-09-23); sottostimare fa superare il
        # limite di 64k e il server risponde 400.
        return len(json.dumps(body, ensure_ascii=False)) // 2 + 1

    def _body(self, requests: list[SemanticDecisionRequest]):
        single = len(requests) == 1
        state: object = requests[0].state if single else {}
        questions: dict[str, dict] = {}
        maps: list[dict[str, str]] = []
        for index, request in enumerate(requests):
            key = f"q{index}"
            mapping = {f"o{i}": option.id for i, option in enumerate(request.options)}
            maps.append(mapping)
            criteria = {f"o{i}": option.description for i, option in enumerate(request.options)}
            instructions = request.question if single else (
                f"Answer only about state.{key}. {request.question}"
            )
            if not single:
                state[key] = request.state  # type: ignore[index]
            questions[key] = {
                "type": self._question_type,
                "instructions": instructions,
                "criteria": criteria,
            }
        return {"model": self._model, "state": state, "questions": questions}, maps

    def _too_big(self, body) -> bool:
        """Oltre uno dei due limiti: totale, oppure stato (piu' ogni domanda)."""
        if self._estimate_tokens(body) > self._max_prompt_tokens:
            return True
        longest = max(
            (self._estimate_tokens(q) for q in body["questions"].values()), default=0
        )
        return self._estimate_tokens({"state": body["state"]}) + longest > MAX_STATE_TOKENS

    def _chunks(self, requests):
        chunk: list[SemanticDecisionRequest] = []
        for request in requests:
            candidate = chunk + [request]
            body, _ = self._body(candidate)
            if chunk and self._too_big(body):
                yield chunk
                chunk = [request]
            else:
                chunk = candidate
        if chunk:
            yield chunk

    # -- chiamata ------------------------------------------------------------
    def _fallback_all(self, requests, reason: str, latency_ms: float = 0.0):
        return [
            fallback_decision(r, reason, latency_ms=latency_ms, runtime=self._runtime)
            for r in requests
        ]

    def _decide_chunk(self, requests: list[SemanticDecisionRequest]):
        secret = os.getenv(self._api_key_env, "") if self._api_key_env else ""
        if self._api_key_env and not secret:
            return self._fallback_all(requests, "missing_api_key_env")
        body, maps = self._body(requests)
        estimate = self._estimate_tokens(body)
        try:
            if self.ledger is not None:
                self.ledger.authorize(estimate)
        except BudgetExceeded:
            return self._fallback_all(requests, "budget_exceeded")
        except (OSError, ValueError):
            # Registro illeggibile: il tetto non e' verificabile, niente chiamata.
            return self._fallback_all(requests, "ledger_error")
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        started = self._clock()
        status, raw, response_headers, reason = 0, b"", {}, "transport_error"
        for attempt in range(2):
            try:
                status, raw, response_headers = self._transport(
                    self._url, payload, headers, self._timeout
                )
            except urllib.error.HTTPError as error:
                status, raw = int(error.code), b""
                response_headers = dict(error.headers.items()) if error.headers else {}
            except (TimeoutError, socket.timeout):
                # Il server puo' aver fatturato: si addebita la stima.
                return self._charged_fallback(requests, estimate, "timeout", started)
            except Exception:  # noqa: BLE001 - il testo puo' contenere segreti
                return self._charged_fallback(
                    requests, estimate, "transport_error", started
                )
            if status == 429 and attempt == 0:
                wait = self._retry_after(response_headers)
                self._sleep(wait)
                continue
            break
        latency = self._elapsed(started)
        if not 200 <= status < 300:
            return self._charged_fallback(
                requests, estimate, f"http_status_{status}", started
            )
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            data = None
        usage = data.get("usage") if isinstance(data, dict) else None
        tokens = estimate
        if isinstance(usage, dict):
            try:
                tokens = int(usage.get("input_tokens", estimate))
            except (TypeError, ValueError):
                tokens = estimate
        # Un solo addebito per richiesta, prima di leggere le risposte.
        if not self._charge(tokens):
            return self._fallback_all(requests, "ledger_error", latency)
        if not isinstance(data, dict):
            return self._fallback_all(requests, "invalid_response", latency)
        try:
            return [
                self._decision(request, data["answers"][f"q{index}"], maps[index], latency)
                for index, request in enumerate(requests)
            ]
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._fallback_all(requests, "invalid_response", latency)

    def _charge(self, tokens: int) -> bool:
        """Registra la spesa; `False` se il registro non si puo' scrivere."""
        if self.ledger is None:
            return True
        try:
            self.ledger.record(tokens)
        except (OSError, ValueError):
            return False
        return True

    def _charged_fallback(self, requests, estimate: int, reason: str, started: float):
        latency = self._elapsed(started)
        if not self._charge(estimate):
            return self._fallback_all(requests, "ledger_error", latency)
        return self._fallback_all(requests, reason, latency)

    def _elapsed(self, started: float) -> float:
        return max(0.0, (self._clock() - started) * 1_000.0)

    @staticmethod
    def _retry_after(headers: Mapping[str, str]) -> float:
        for name in ("Retry-After", "retry-after"):
            if name in headers:
                try:
                    return min(MAX_RETRY_AFTER_SECONDS, max(0.0, float(headers[name])))
                except ValueError:
                    break
        return 1.0

    def _decision(self, request, answer: Mapping, mapping: Mapping[str, str], latency_ms):
        raw_probabilities = {str(k): float(v) for k, v in dict(answer["probabilities"]).items()}
        if set(raw_probabilities) != set(mapping):
            raise ValueError("response options differ from request options")
        total = sum(raw_probabilities.values())
        if total <= 0.0:
            raise ValueError("empty distribution")
        probabilities = {mapping[k]: v / total for k, v in raw_probabilities.items()}
        selected = mapping[str(answer["choice"])]
        confidence = float(answer.get("confidence", 0.0))
        if confidence < self._min_confidence:
            return fallback_decision(
                request, "low_confidence", latency_ms=latency_ms, runtime=self._runtime
            )
        return SemanticDecision(
            run_id=request.run_id,
            step=request.step,
            actor=request.actor,
            question_id=request.question_id,
            input_hash=request.input_hash,
            selected_option=selected,
            probabilities=probabilities,
            confidence=max(0.0, min(1.0, confidence)),
            route="fallback" if selected in FALLBACK_OPTIONS else "filter",
            rationale={"source": self._runtime},
            runtime=self._runtime,
            model=self._model,
            latency_ms=latency_ms,
            fallback_reason="selected_fallback" if selected in FALLBACK_OPTIONS else "",
        )


class LayaSemanticDecisionProvider(TypeSafeSemanticDecisionProvider):
    """Laya (encoder System One open-weights) servito da `laya.serve`.

    Stesso protocollo `/v1/systemone` di Jev, ma: nessun registro di spesa
    (gira su hardware proprio), chiave facoltativa, e **una domanda per
    richiesta**. Il contesto di Laya e' di 512 token: impacchettare piu' casi in
    uno stato li troncherebbe. Le richieste corrono in parallelo nel pool di
    `decide_many`, che vede `decide_many = None` e usa `decide` per ciascuna.
    """

    decide_many = None  # type: ignore[assignment]

    def __init__(self, *, endpoint: str, api_key_env: str = "", model: str = "laya", **kwargs) -> None:
        super().__init__(ledger=None, endpoint=endpoint, api_key_env=api_key_env,
                         model=model, **kwargs)
        self._runtime = "laya"

    def decide(self, request: SemanticDecisionRequest) -> SemanticDecision:
        return self._decide_chunk([request])[0]
