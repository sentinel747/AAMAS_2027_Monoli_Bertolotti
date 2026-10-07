from __future__ import annotations

"""Test dell'oracolo di parita' (Task 0 del piano di migrazione Rust).

Questi test proteggono tre proprieta' da cui dipende l'intera migrazione:

1. il digest e' **sensibile**: una mutazione dello stato viene rilevata e
   localizzata sul campo giusto (un digest che non fallisce mai e' inutile);
2. il digest e' **insensibile** a cio' che non e' stato simulativo (orologio di
   parete), altrimenti segnala falsi non-determinismi;
3. la simulazione e' **deterministica fra processi distinti**, il passo bloccante
   0.0 del piano: senza questa proprieta' non esiste oracolo con cui confrontare
   un futuro backend Rust.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core import state_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "scripts" / "parity_harness.py"


class _FakeAgents:
    def __init__(self) -> None:
        self.x = np.array([1, 2, 3, 0], dtype=np.int16)
        self.y = np.array([4, 5, 6, 0], dtype=np.int16)
        self.health = np.array([1.0, 0.5, 0.25, 0.0], dtype=np.float64)
        self.n = 3
        self.ids = ["a", "b", "c"]
        self.index = {"a": 0, "b": 1, "c": 2}


class _FakeCells:
    def __init__(self) -> None:
        self.H, self.W = 2, 2
        self.explored = np.zeros((2, 2), dtype=np.bool_)
        self.water_ice = np.zeros((2, 2), dtype=np.float64)
        self._site_keys: dict = {}
        self._site_extra: dict = {}
        self._structure_instances: dict = {}


class _FakeState:
    def __init__(self) -> None:
        self.agents = _FakeAgents()
        self.cells = _FakeCells()
        self.side: dict = {}


def test_digest_ignores_rows_beyond_live_count() -> None:
    """Le righe oltre ``n`` sono riempimento mai letto: non devono contare.

    Se contassero, il digest dipenderebbe dalla capacita' allocata degli array e
    due backend con margini di crescita diversi divergerebbero senza che lo stato
    simulativo differisca.
    """
    state = _FakeState()
    before = state_digest.digest_state(state)
    state.agents.x[3] = 999
    state.agents.health[3] = 0.77
    assert state_digest.digest_state(state) == before


def test_digest_detects_and_localizes_a_mutation() -> None:
    state = _FakeState()
    before = state_digest.digest_state(state)
    state.agents.health[1] = 0.6

    after = state_digest.digest_state(state)
    assert state_digest.combine(after) != state_digest.combine(before)

    divergence = state_digest.first_divergence(before, after)
    assert divergence is not None
    field, lhs, rhs = divergence
    assert field == "agents.health"
    assert lhs != rhs


def test_digest_detects_cell_mutation() -> None:
    state = _FakeState()
    before = state_digest.digest_state(state)
    state.cells.water_ice[1, 1] = 3.5

    divergence = state_digest.first_divergence(before, state_digest.digest_state(state))
    assert divergence is not None
    assert divergence[0] == "cells.water_ice"


def test_first_divergence_returns_none_when_equal() -> None:
    state = _FakeState()
    fields = state_digest.digest_state(state)
    assert state_digest.first_divergence(fields, dict(fields)) is None


def test_first_divergence_flags_schema_mismatch() -> None:
    """Un campo presente da un solo lato e' una divergenza di schema."""
    left = {"agents.x": "aa", "agents.y": "bb"}
    right = {"agents.x": "aa"}
    divergence = state_digest.first_divergence(left, right)
    assert divergence == ("agents.y", "bb", None)


class _FakeOutcome:
    def __init__(self, wall_time: str) -> None:
        self.agent_step_records: list = []
        self.deaths_by_agent: list = []
        self.flows: list = []
        self.knowledge_gain = 0.0
        self.events = [
            {
                "type": "structure_built",
                "step": 1,
                "wall_time": wall_time,
                "data": {"structure": "habitat"},
            }
        ]


def test_digest_ignores_wall_clock_in_events() -> None:
    """Due run identiche avviate a un secondo di distanza devono coincidere.

    `wall_time` e' l'orologio di parete, non stato simulativo. Includerlo faceva
    fallire il passo 0.0 con un falso positivo.
    """
    early = state_digest.digest_outcome(_FakeOutcome("2026-08-04T16:33:15"))
    late = state_digest.digest_outcome(_FakeOutcome("2026-08-04T16:33:16"))
    assert early == late


def test_digest_still_detects_a_real_event_change() -> None:
    """L'esclusione di `wall_time` non deve rendere cieco il digest sugli eventi."""
    baseline = _FakeOutcome("2026-08-04T16:33:15")
    changed = _FakeOutcome("2026-08-04T16:33:15")
    changed.events[0]["data"] = {"structure": "greenhouse"}
    assert (
        state_digest.digest_outcome(baseline)["outcome.events"]
        != state_digest.digest_outcome(changed)["outcome.events"]
    )


@pytest.mark.timeout(600)
def test_kernel_is_deterministic_across_processes() -> None:
    """Passo bloccante 0.0: stesso seed, processi e PYTHONHASHSEED diversi.

    Gira in sottoprocessi di proposito: il non-determinismo che questo test cerca
    (ordine di iterazione dipendente da `PYTHONHASHSEED`, identita' di oggetti nei
    tie-break) non e' osservabile dentro un singolo interprete.
    """
    from scripts.parity_harness import read_digests

    with tempfile.TemporaryDirectory() as tmp:
        outputs = []
        for replica, hash_seed in (("a", "0"), ("b", "12345")):
            out = Path(tmp) / f"{replica}.jsonl"
            proc = subprocess.run(
                [
                    sys.executable, str(HARNESS),
                    "--seed", "0", "--steps", "8", "--agents", "40",
                    "--out", str(out), "--quiet",
                ],
                cwd=str(REPO_ROOT),
                env=os.environ | {"PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr[-3000:]
            outputs.append(out)

        _, left = read_digests(outputs[0])
        _, right = read_digests(outputs[1])

    assert len(left) == len(right) == 8
    for lhs, rhs in zip(left, right):
        divergence = state_digest.first_divergence(lhs["fields"], rhs["fields"])
        assert divergence is None, (
            f"step {lhs['step']}: campo {divergence[0]} diverge fra processi "
            f"({divergence[1]} contro {divergence[2]})"
        )


def _state_with_side():
    """Stato con lo stato cold reale, non un finto: il digest deve leggerlo."""
    from src.core.views import AgentSideState

    state = _FakeState()
    state.side = {agent_id: AgentSideState() for agent_id in state.agents.ids}
    return state


def test_digest_detects_a_change_in_recent_actions() -> None:
    """`vitals.py` decide idratazione e sazieta' leggendo `recent_actions[-1]`.

    Fino allo schema 1 quella lista era fuori dal digest, quindi una divergenza
    di comportamento poteva restare invisibile fino allo step in cui cambiava
    davvero un'azione. Qui si verifica che ora venga vista subito.
    """
    state = _state_with_side()
    before = state_digest.digest_state(state)

    state.side["b"].recent_actions.append("drink_water")

    after = state_digest.digest_state(state)
    assert after != before
    divergence = state_digest.first_divergence(before, after)
    assert divergence is not None and divergence[0] == "side.recent_actions"


def test_digest_detects_a_change_in_memory_read_by_decisions() -> None:
    """`rule_based_agent` non ripete un'azione presente in `recent_failures`."""
    state = _state_with_side()
    before = state_digest.digest_state(state)

    state.side["a"].memory.add_failure("build_habitat", None, step=3)

    after = state_digest.digest_state(state)
    divergence = state_digest.first_divergence(before, after)
    assert divergence is not None and divergence[0] == "side.memory.recent_failures"


def test_digest_detects_a_different_eviction_order_in_explored_cells() -> None:
    """L'ordine di `explored_cells` e' contenuto, non storia degli inserimenti.

    `remember_cell` sfratta la cella piu' vecchia con `next(iter(...))`: due
    ordini diversi portano a sfrattare celle diverse, cioe' a una divergenza di
    comportamento. Ordinare le chiavi -- come si fa invece per le mappe sparse
    delle celle -- la nasconderebbe.
    """
    left = _state_with_side()
    right = _state_with_side()

    left.side["a"].memory.remember_cell(1, 1)
    left.side["a"].memory.remember_cell(2, 2)
    right.side["a"].memory.remember_cell(2, 2)
    right.side["a"].memory.remember_cell(1, 1)

    divergence = state_digest.first_divergence(
        state_digest.digest_state(left), state_digest.digest_state(right)
    )
    assert divergence is not None and divergence[0] == "side.memory.explored_cells"


def test_digest_ignores_purely_descriptive_memory() -> None:
    """Un riassunto di log non e' stato simulativo: includerlo darebbe falsi allarmi."""
    state = _state_with_side()
    before = state_digest.digest_state(state)

    state.side["c"].memory.long_term_summary = "una frase qualsiasi"
    state.side["c"].memory.social_memory["b"] = "collaborativo"

    assert state_digest.digest_state(state) == before
