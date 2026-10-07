from __future__ import annotations

"""Digest deterministico di ``CoreState``, oracolo della migrazione Rust.

Il piano di migrazione (``docs/superpowers/plans/2026-08-02-rust-kernel-migration.md``)
richiede di localizzare la *prima* divergenza fra due backend a livello di singolo
campo, non di confrontare solo le metriche finali. Questo modulo produce quindi
due cose insieme:

- un digest **per campo**, cosi' il comparator puo' dire *quale* buffer diverge;
- un digest **complessivo**, ordinato in modo stabile, per il confronto rapido.

Il confronto e' volutamente *bit-exact* sui float: gli array sono float64
(``src/core/arrays.py`` documenta perche' non float32) e la strategia di
equivalenza del piano richiede uguaglianza esatta su ID, codici azione, claim e
coordinate. Due valori numericamente uguali ma con pattern di bit diversi
(``0.0`` contro ``-0.0``) vengono segnalati come divergenti: e' un falso positivo
accettabile, perche' indica comunque un percorso di calcolo diverso.

Il modulo non importa nulla dal kernel per restare utilizzabile anche dai test di
parita' che girano prima che un backend alternativo esista.
"""

from hashlib import blake2b
from typing import Any

import numpy as np

# Lunghezza del digest per campo: 16 byte bastano per un confronto di uguaglianza
# fra due run della stessa suite e tengono i file di parita' leggibili.
_DIGEST_BYTES = 16

SCHEMA_VERSION = 2

# Chiavi di evento che dipendono dall'orologio di parete e non dalla
# simulazione. Vanno escluse dal digest: due run identiche avviate a un secondo
# di distanza produrrebbero altrimenti digest diversi, e il passo 0.0 del piano
# segnalerebbe come non-determinismo cio' che e' solo un timestamp. `event_id`
# NON e' in questo insieme: e' un contatore progressivo, quindi deterministico e
# significativo per l'ordine degli eventi, che il piano richiede di confrontare.
_VOLATILE_EVENT_KEYS = frozenset({"wall_time"})

# Campi cold di `AgentSideState` che influenzano il comportamento simulativo e
# che quindi devono entrare nel digest anche se non sono colonne SoA.
_SIDE_FIELDS = (
    "actions_taken",
    "founder_kit_reserved",
    "local_x_m",
    "local_y_m",
    "movement_distance_m_per_step",
    "perception_radius",
    "perception_radius_m",
    "survival_priority",
)

# SCHEMA 2 - due strutture cold MUTABILI che la simulazione legge davvero, e che
# fino allo schema 1 restavano fuori dal digest:
#
#   `recent_actions`  -> `src/agents/vitals.py` decide idratazione e sazieta' di
#                        ogni step guardando `recent_actions[-1]`, e
#                        `rule_based_agent` ci legge la cadenza di
#                        coordinamento e l'ultima azione produttiva.
#   `memory`          -> `rule_based_agent` legge `memory.recent_failures` per
#                        non ripetere azioni fallite e `memory.task_queue` per
#                        riprendere un lavoro interrotto.
#
# Erano coperte solo in modo INDIRETTO e in ritardo: una divergenza qui cambia
# una decisione, e la decisione compare in `outcome.actions` -- ma solo allo step
# in cui cambia davvero, quindi una divergenza latente poteva restare invisibile
# per l'intero orizzonte misurato. Dichiarare "bit-exact" su quella base era piu'
# forte di quanto la misura sostenesse.
#
# I campi puramente descrittivi di `AgentMemory` (`long_term_summary`,
# `conversations`, `social_memory`) restano fuori di proposito: nessun percorso
# di decisione li legge, e includerli renderebbe il digest sensibile alla
# formattazione di stringhe di log invece che allo stato.
_MEMORY_FIELDS = (
    "active_alerts",
    "discovered_resources",
    "explored_cells",
    "past_decisions",
    "recent_events",
    "recent_failures",
    "task_queue",
)


def _new_hasher() -> Any:
    return blake2b(digest_size=_DIGEST_BYTES)


def _hash_array(array: np.ndarray) -> str:
    """Digest di un ndarray, dtype e shape inclusi.

    dtype e shape entrano nell'hash perche' un cambio di tipo o di layout e' una
    divergenza di schema, non un dettaglio di rappresentazione: senza di essi un
    array int16 e uno int32 con gli stessi valori darebbero digest diversi solo
    per caso (byte diversi) o uguali per caso (stessi byte), e in nessuno dei due
    casi il risultato sarebbe interpretabile.
    """
    hasher = _new_hasher()
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(b"|")
    hasher.update(repr(array.shape).encode("utf-8"))
    hasher.update(b"|")
    # `ascontiguousarray` e' necessario: `tobytes()` su una vista non contigua
    # materializza comunque, ma su una vista *contigua di un buffer piu' grande*
    # (come `agents.x[:n]`) restituirebbe i byte giusti solo per fortuna.
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _hash_text(text: str) -> str:
    hasher = _new_hasher()
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def _stable_repr(value: Any) -> str:
    """Serializza una struttura cold **conservandone l'ordine**.

    Deliberatamente diverso dalle mappe sparse di ``digest_cells``, che vengono
    ordinate. Qui l'ordine e' contenuto, non storia: ``AgentMemory.remember_cell``
    sfratta la cella piu' vecchia con ``next(iter(self.explored_cells))`` quando
    si supera il limite, e ``recent_failures``/``task_queue`` sono code lette
    dalla coda o dalla testa. Ordinare nasconderebbe una divergenza di sfratto,
    che e' una divergenza di comportamento a tutti gli effetti.

    Conservare l'ordine non reintroduce dipendenza da ``PYTHONHASHSEED``: dal
    3.7 l'iterazione di un dict segue l'inserimento, non l'hash. Nessuno di
    questi campi e' un ``set``, che sarebbe invece l'unico caso a rischio.
    """
    return repr(value)


def _array_fields(container: Any) -> list[str]:
    """Nomi ordinati degli attributi ndarray pubblici di un contenitore SoA.

    L'ordinamento e' esplicito e non dipende dall'ordine di inserimento negli
    attributi: e' proprio il tipo di dipendenza implicita che il passo 0.0 del
    piano (determinismo cross-processo) deve escludere.
    """
    return sorted(
        name
        for name, value in vars(container).items()
        if not name.startswith("_") and isinstance(value, np.ndarray)
    )


def digest_agents(agents: Any) -> dict[str, str]:
    """Digest per campo di ``AgentArrays``, limitato alle righe realmente usate.

    Solo il prefisso ``[:n]`` viene considerato: oltre ``n`` gli array contengono
    valori di riempimento mai letti dalla simulazione, e includerli renderebbe il
    digest sensibile alla capacita' allocata invece che allo stato.
    """
    live = int(getattr(agents, "n", 0))
    out: dict[str, str] = {}
    for name in _array_fields(agents):
        out[f"agents.{name}"] = _hash_array(getattr(agents, name)[:live])
    out["agents.n"] = _hash_text(str(live))
    out["agents.ids"] = _hash_text("\x1f".join(agents.ids[:live]))
    return out


def digest_cells(cells: Any) -> dict[str, str]:
    """Digest per campo di ``CellArrays``, incluse le mappe sparse di stato.

    ``_site_keys``, ``_site_extra`` e ``_structure_instances`` sono dizionari, non
    array, ma fanno parte dello stato autorevole (cantieri aperti e posizione
    sub-cella delle strutture). Vengono serializzati con chiavi ordinate: l'ordine
    di inserimento di un dict e' deterministico dentro un processo ma non e'
    garantito che due backend lo producano nello stesso ordine, e la parita' deve
    misurare il contenuto, non la storia degli inserimenti.
    """
    out: dict[str, str] = {}
    for name in _array_fields(cells):
        out[f"cells.{name}"] = _hash_array(getattr(cells, name))
    out["cells.shape"] = _hash_text(f"{int(cells.H)}x{int(cells.W)}")

    site_keys = getattr(cells, "_site_keys", {}) or {}
    out["cells.site_keys"] = _hash_text(
        "\x1e".join(f"{k!r}={v!r}" for k, v in sorted(site_keys.items(), key=repr))
    )
    site_extra = getattr(cells, "_site_extra", {}) or {}
    out["cells.site_extra"] = _hash_text(
        "\x1e".join(f"{k!r}={v!r}" for k, v in sorted(site_extra.items(), key=repr))
    )
    instances = getattr(cells, "_structure_instances", {}) or {}
    out["cells.structure_instances"] = _hash_text(
        "\x1e".join(
            f"{key!r}={value!r}"
            for key, value in sorted(instances.items(), key=repr)
        )
    )
    return out


def digest_side(side: Any, ids: list[str]) -> dict[str, str]:
    """Digest dei campi cold per-agente, in ordine di ``ids``.

    L'ordine segue le righe SoA e non l'iterazione del dict ``side``: e' la stessa
    ragione per cui le mappe sparse vengono ordinate sopra.
    """
    out: dict[str, str] = {}
    entries = [
        (side.get(agent_id) if hasattr(side, "get") else None) for agent_id in ids
    ]
    for field in _SIDE_FIELDS:
        out[f"side.{field}"] = _hash_text(
            "\x1f".join(repr(getattr(entry, field, None)) for entry in entries)
        )

    out["side.recent_actions"] = _hash_text(
        "\x1f".join(
            "\x1d".join(map(str, getattr(entry, "recent_actions", ()) or ()))
            for entry in entries
        )
    )
    for field in _MEMORY_FIELDS:
        out[f"side.memory.{field}"] = _hash_text(
            "\x1f".join(
                _stable_repr(getattr(getattr(entry, "memory", None), field, None))
                for entry in entries
            )
        )
    return out


def digest_state(state: Any) -> dict[str, str]:
    """Digest per campo dell'intero ``CoreState``."""
    agents = state.agents
    live = int(getattr(agents, "n", 0))
    fields: dict[str, str] = {}
    fields.update(digest_agents(agents))
    fields.update(digest_cells(state.cells))
    fields.update(digest_side(state.side, agents.ids[:live]))
    return fields


def digest_outcome(outcome: Any) -> dict[str, str]:
    """Digest della decisione e dell'esecuzione prodotte da uno step.

    Lo stato finale da solo non basta: due backend possono convergere sullo stesso
    stato passando per azioni diverse (una accettata al posto di un'altra
    equivalente), e il piano richiede parita' anche su azioni validate/respinte e
    ordine degli eventi.
    """
    records = list(getattr(outcome, "agent_step_records", ()) or ())
    action_rows = []
    for record in records:
        request = record.request
        action = getattr(request, "action", None)
        target = getattr(request, "target", None)
        action_rows.append(
            f"{record.agent_id}|{getattr(action, 'value', action)}|"
            f"{bool(record.accepted)}|{target!r}|{record.x},{record.y}"
        )
    out = {
        "outcome.actions": _hash_text("\x1e".join(action_rows)),
        "outcome.accepted_count": _hash_text(
            str(sum(1 for r in records if r.accepted))
        ),
        "outcome.rejected_count": _hash_text(
            str(sum(1 for r in records if not r.accepted))
        ),
        "outcome.deaths": _hash_text(
            "\x1e".join(f"{a}|{c}" for a, c in getattr(outcome, "deaths_by_agent", ()))
        ),
        "outcome.events": _hash_text(
            "\x1e".join(
                repr(
                    sorted(
                        (k, v)
                        for k, v in event.items()
                        if k not in _VOLATILE_EVENT_KEYS
                    )
                )
                for event in getattr(outcome, "events", ()) or ()
            )
        ),
        "outcome.knowledge_gain": _hash_text(
            repr(float(getattr(outcome, "knowledge_gain", 0.0)))
        ),
        "outcome.flows": _hash_text(
            "\x1e".join(repr(flow) for flow in getattr(outcome, "flows", ()) or ())
        ),
    }
    return out


def combine(fields: dict[str, str]) -> str:
    """Digest complessivo, stabile rispetto all'ordine di inserimento."""
    hasher = _new_hasher()
    hasher.update(f"v{SCHEMA_VERSION}|".encode("utf-8"))
    for name in sorted(fields):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"=")
        hasher.update(fields[name].encode("utf-8"))
        hasher.update(b";")
    return hasher.hexdigest()


def step_digest(state: Any, outcome: Any | None = None) -> dict[str, Any]:
    """Record di parita' per un singolo step: campi + digest complessivo."""
    fields = digest_state(state)
    if outcome is not None:
        fields.update(digest_outcome(outcome))
    return {"overall": combine(fields), "fields": fields}


def first_divergence(
    left: dict[str, str], right: dict[str, str]
) -> tuple[str, str | None, str | None] | None:
    """Primo campo divergente in ordine stabile, o ``None`` se identici.

    Restituisce ``(campo, valore_sinistro, valore_destro)``; un valore ``None``
    segnala un campo presente da un solo lato, che e' una divergenza di schema e
    va distinta da una divergenza di valore.
    """
    for name in sorted(set(left) | set(right)):
        lhs = left.get(name)
        rhs = right.get(name)
        if lhs != rhs:
            return name, lhs, rhs
    return None
