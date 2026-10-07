from __future__ import annotations

"""Test dell'indice per-cella degli agenti vivi (Task 0b, migrazione Rust).

`CellView.agents_present` non scandisce piu' gli array a ogni lettura: consulta un
indice invalidato a epoca. Il rischio dell'invalidazione manuale e' che una
scrittura futura di ``x``, ``y`` o ``alive`` dimentichi ``touch_positions()`` e
faccia leggere posizioni obsolete - un errore silenzioso che cambierebbe il
comportamento della simulazione senza sollevare eccezioni.

Questi test bloccano i tre percorsi che mutano quelle colonne (spostamento,
comparsa, morte) e verificano che l'indice resti d'accordo con la scansione
diretta, che resta la definizione di riferimento.
"""

import numpy as np
import pytest

from src.core.arrays import AgentArrays
from src.core import constants as C


def _scan_reference(agents: AgentArrays, x: int, y: int) -> list[str]:
    """Definizione di riferimento: la scansione che l'indice sostituisce."""
    rows = agents.alive_rows()
    if rows.size == 0:
        return []
    mask = (agents.x[rows] == x) & (agents.y[rows] == y)
    return [agents.ids[int(i)] for i in rows[mask]]


def _agents_with(positions: list[tuple[int, int]]) -> AgentArrays:
    arrays = AgentArrays(max(8, len(positions) * 2))
    for index, (x, y) in enumerate(positions):
        arrays.spawn(f"agent_{index:03d}", x, y, np.zeros(C.NR, dtype=np.float64))
    return arrays


def test_index_matches_the_scan_it_replaces() -> None:
    agents = _agents_with([(0, 0), (1, 2), (0, 0), (3, 3)])
    index = agents.positions_index()
    for x, y in {(0, 0), (1, 2), (3, 3), (5, 5)}:
        assert index.get((x, y), []) == _scan_reference(agents, x, y)


def test_index_preserves_ascending_row_order() -> None:
    """L'ordine degli id dentro una cella e' osservabile e deve restare stabile.

    `action_space.py` costruisce l'insieme dei vicini iterando questa lista; un
    ordine diverso non cambierebbe l'insieme, ma cambierebbe l'ordine di
    inserimento e quindi ogni struttura che ne dipende a valle.
    """
    agents = _agents_with([(2, 2), (9, 9), (2, 2), (2, 2)])
    assert agents.positions_index()[(2, 2)] == ["agent_000", "agent_002", "agent_003"]


def test_spawn_invalidates_the_index() -> None:
    agents = _agents_with([(4, 4)])
    assert agents.positions_index()[(4, 4)] == ["agent_000"]

    agents.spawn("newcomer", 4, 4, np.zeros(C.NR, dtype=np.float64))
    assert agents.positions_index()[(4, 4)] == ["agent_000", "newcomer"]


def test_death_removes_the_agent_from_its_cell() -> None:
    agents = _agents_with([(6, 1), (6, 1)])
    assert len(agents.positions_index()[(6, 1)]) == 2

    agents.alive[1] = False
    agents.touch_positions()
    assert agents.positions_index()[(6, 1)] == ["agent_000"]
    assert agents.positions_index()[(6, 1)] == _scan_reference(agents, 6, 1)


def test_movement_through_agent_view_invalidates_the_index() -> None:
    """Il percorso reale di spostamento passa dai setter di `AgentView`."""
    from src.core.views import AgentSideState, AgentView

    agents = _agents_with([(0, 5)])
    side = {"agent_000": AgentSideState()}
    view = AgentView(agents, 0, side["agent_000"])

    assert agents.positions_index()[(0, 5)] == ["agent_000"]
    view.x = 7
    view.y = 8
    assert agents.positions_index().get((0, 5), []) == []
    assert agents.positions_index()[(7, 8)] == ["agent_000"]


def test_verification_mode_detects_a_missing_invalidation() -> None:
    """Il guardrail deve accorgersi di una scrittura senza `touch_positions`.

    Se questo test fallisce, la modalita' `MARSABM_VERIFY_INDEX` non protegge
    piu' nulla e l'invarianza dell'indice torna affidata alla sola disciplina di
    chi modifica il codice.
    """
    import src.core.arrays as arrays_module

    agents = _agents_with([(1, 1)])
    agents.positions_index()  # popola la cache all'epoca corrente

    agents.x[0] = 2  # scrittura DELIBERATAMENTE senza touch_positions()

    original = arrays_module._VERIFY_POSITION_INDEX
    arrays_module._VERIFY_POSITION_INDEX = True
    try:
        with pytest.raises(AssertionError, match="non invalidato"):
            agents.positions_index()
    finally:
        arrays_module._VERIFY_POSITION_INDEX = original
