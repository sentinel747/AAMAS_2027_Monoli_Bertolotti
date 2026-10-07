"""Calibrazione a temperatura delle distribuzioni di un provider.

La lettura dei logprob di un LLM generico (Qwen via gateway) sceglie spesso
l'opzione giusta ma con distribuzioni troppo piatte: la regola di decisione
v2h, che guarda quanto e' netta la preferenza, non scatta mai. Una temperatura
per tipo di domanda, stimata su casi etichettati da un riferimento, rende la
distribuzione piu' o meno netta senza cambiare quale opzione e' preferita.

    q_i proporzionale a p_i ** (1 / T)     T < 1 affila, T > 1 appiattisce

Non modifica il modello e non addestra nulla: e' un parametro per tipo di
domanda, dichiarato nel manifest della run.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

_EPS = 1e-12


def temper(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Distribuzione temperata; `temperature == 1` restituisce la stessa."""
    p = {str(k): float(v) for k, v in probabilities.items()}
    if temperature == 1.0:
        return p
    if not temperature > 0.0:
        raise ValueError("temperature must be positive")
    powered = {k: (v ** (1.0 / temperature) if v > 0.0 else 0.0) for k, v in p.items()}
    total = sum(powered.values())
    if total <= 0.0:
        return p
    return {k: v / total for k, v in powered.items()}


def fit_temperature(
    rows: Iterable[tuple[Mapping[str, float], str]],
    *,
    low: float = 0.05,
    high: float = 10.0,
    points: int = 241,
) -> float:
    """Temperatura che massimizza la verosimiglianza delle etichette.

    `rows` sono coppie (distribuzione dello studente, opzione di riferimento).
    Ricerca su griglia logaritmica: un solo parametro, nessuna dipendenza.
    """
    data = [(dict(p), str(target)) for p, target in rows]
    if not data:
        return 1.0
    best_t, best_nll = 1.0, math.inf
    for i in range(points):
        t = low * (high / low) ** (i / (points - 1))
        nll = 0.0
        for p, target in data:
            nll -= math.log(temper(p, t).get(target, 0.0) + _EPS)
        nll /= len(data)
        if nll < best_nll - 1e-12:
            best_t, best_nll = t, nll
    return best_t


def calibration_key(question_id: str) -> str:
    """Chiave della tabella per una domanda: profilo e livello, senza l'area.

    `jev-semif-governor-v2h:L2:build` -> `governor-v2h:L2`.
    """
    key = str(question_id)
    if key.startswith("jev-semif-"):
        key = key[len("jev-semif-"):]
    parts = key.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else key


def temperature_for(question_id: str, table: Mapping[str, float] | float | None) -> float:
    """Temperatura da applicare: scalare per tutti, o per chiave di domanda."""
    if table is None:
        return 1.0
    if isinstance(table, (int, float)):
        return float(table)
    return float(table.get(calibration_key(question_id), 1.0))


def normalized_confidence(probabilities: Mapping[str, float]) -> float:
    """Concentrazione: 0 uniforme, 1 certezza (come il gateway semif-runtime)."""
    values = [float(v) for v in probabilities.values()]
    if len(values) < 2:
        return 1.0
    entropy = -sum(v * math.log(v) for v in values if v > 0.0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))
