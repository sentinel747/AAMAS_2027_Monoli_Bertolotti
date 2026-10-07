"""Molte decisioni in una volta, nell'ordine d'ingresso, senza eccezioni in uscita."""

from __future__ import annotations

import concurrent.futures as cf
from collections.abc import Sequence

from .client import fallback_decision
from .replay import ReplaySemanticDecisionProvider
from .schemas import SemanticDecision, SemanticDecisionRequest


def decide_many(
    provider,
    requests: Sequence[SemanticDecisionRequest],
    *,
    max_workers: int = 32,
) -> list[SemanticDecision]:
    requests = list(requests)
    if not requests:
        return []
    native = getattr(provider, "decide_many", None)
    if callable(native):
        # Nessuna eccezione del provider arriva alla simulazione, e una risposta
        # di lunghezza sbagliata non si puo' riallineare: fallback per tutti.
        try:
            decisions = list(native(requests))
        except Exception:  # noqa: BLE001 - il testo puo' contenere segreti
            decisions = []
        if len(decisions) != len(requests):
            return [
                fallback_decision(r, "provider_exception", runtime="batch")
                for r in requests
            ]
        return decisions
    # Un replay che non trova la decisione registrata e' una run DIVERSA che si
    # spaccia per replay: l'errore deve fermarla, non diventare un fattore neutro.
    strict = isinstance(provider, ReplaySemanticDecisionProvider)

    def one(request: SemanticDecisionRequest) -> SemanticDecision:
        if strict:
            return provider.decide(request)
        try:
            return provider.decide(request)
        except Exception:  # noqa: BLE001 - il testo puo' contenere segreti
            return fallback_decision(request, "provider_exception", runtime="batch")

    workers = max(1, min(int(max_workers), len(requests)))
    if workers == 1:
        return [one(r) for r in requests]
    with cf.ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, requests))
