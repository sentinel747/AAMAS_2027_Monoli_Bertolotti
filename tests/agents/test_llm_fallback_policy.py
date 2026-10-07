"""Un agente LLM che salta la chiamata deve decidere come i suoi pari.

Difetto annotato nella spec delle preferenze (`2026-07-24-preference-decision-design.md`)
e mai risolto: `LLMAgent.decide` chiama `super().decide()`, cioe' la policy AD
ALBERO. In modalita' preferenze i suoi pari usano la policy A PREFERENZE, quindi
un agente LLM che ricade sul fallback decideva con una politica diversa dalla loro
e sporcava il confronto fra le due popolazioni.
"""

import inspect

from src.agents.llm_agent import LLMAgent


def test_the_fallback_goes_through_the_configured_policy_not_always_the_tree():
    calls = {"preference": 0, "tree": 0}

    class _Agent(LLMAgent):
        def _preference_decide(self, observation, world):
            calls["preference"] += 1
            return "preference-request"

        def _tree_decide(self, observation, world):
            calls["tree"] += 1
            return "tree-request"

    agent = _Agent.__new__(_Agent)
    agent.decision_mode = "preferences"
    assert agent._fallback_decide(None, None) == "preference-request"
    assert calls == {"preference": 1, "tree": 0}

    agent.decision_mode = "tree"
    assert agent._fallback_decide(None, None) == "tree-request"
    assert calls == {"preference": 1, "tree": 1}


def test_an_unset_decision_mode_keeps_the_historical_behaviour():
    """Nessuna run esistente deve cambiare comportamento per questo fix."""

    class _Agent(LLMAgent):
        def _tree_decide(self, observation, world):
            return "tree-request"

    agent = _Agent.__new__(_Agent)
    assert agent._fallback_decide(None, None) == "tree-request"


# --- Aggiunte al piano ------------------------------------------------------
# Tre pericoli che il piano non copre: la scelta della griglia delle celle nei
# due tipi di mondo, e il numero di ritorni di ripiego effettivamente agganciati.


class _CellStub:
    """La sola parte di `CellView`/`Cell` che serve per indicizzare le maschere."""

    def __init__(self, x: int, y: int):
        self.x, self.y = x, y


class _ArrayWorldStub:
    """Un mondo con stato ad array, come `WorldView` del kernel."""

    def __init__(self, cells):
        self._cells = cells
        # `WorldView.cells` costruisce una `_LazyCellGrid`, che non ha ne' `H`
        # ne' le colonne che `compute_cell_masks` legge: se il fallback la
        # scegliesse, la chiamata non potrebbe funzionare.
        self.cells = "griglia-pigra-di-facciata"
        self.step = 7

    def get_cell(self, x, y):
        return _CellStub(x, y)


class _ObjectWorldStub:
    """Un mondo del motore a oggetti, come `GridWorld`: nessuno stato ad array."""

    def __init__(self):
        self.cells = [[_CellStub(0, 0)]]
        self.step = 7

    def get_cell(self, x, y):
        return _CellStub(x, y)


class _MasksStub:
    def __getitem__(self, key):
        return ("mask-row", key)


def test_the_preference_fallback_reads_the_array_cells_even_when_they_look_empty(
    monkeypatch,
):
    """Il ramo va scelto sull'esistenza dello stato ad array, non sulla sua verita'.

    `getattr(world, "_cells", None) or world.cells` scavalca in silenzio uno
    stato ad array che valuta falso. Oggi `CellArrays` non definisce ne'
    `__len__` ne' `__bool__` e quindi e' sempre vero, ma la garanzia poggia
    sull'assenza di un metodo, non su una scelta dichiarata. Il ramo scavalcato
    porta a `WorldView.cells`, cioe' una `_LazyCellGrid` che `compute_cell_masks`
    non puo' leggere: la decisione del fallback dipenderebbe da un dettaglio di
    implementazione di una classe che non c'entra.
    """
    import src.agents.preference_agent as preference_agent
    import src.core.cell_action_mask as cell_action_mask

    class _FalsyCells:
        def __len__(self):
            return 0

    array_cells = _FalsyCells()
    world = _ArrayWorldStub(array_cells)
    seen = {}

    def _fake_compute_cell_masks(cells):
        seen["cells"] = cells
        return _MasksStub()

    def _fake_decide_preferences(agent, world_arg, mask_row, u01, sampling, claims, step):
        seen["mask_row"] = mask_row
        seen["step"] = step
        return "preference-request"

    monkeypatch.setattr(cell_action_mask, "compute_cell_masks", _fake_compute_cell_masks)
    monkeypatch.setattr(preference_agent, "decide_preferences", _fake_decide_preferences)

    agent = LLMAgent.__new__(LLMAgent)
    agent.x, agent.y = 3, 5
    assert agent._preference_decide(None, world) == "preference-request"
    assert seen["cells"] is array_cells
    assert seen["mask_row"] == ("mask-row", (5, 3))
    assert seen["step"] == 7


def test_a_world_without_array_cells_falls_back_to_the_tree_instead_of_raising():
    """Nel motore a oggetti i pari decidono ad albero, quindi il fallback anche.

    `GridWorld.cells` e' una lista di liste di `Cell`: passarla a
    `compute_cell_masks` solleva `AttributeError` a meta' di un passo, cioe' un
    agente LLM fermerebbe la run. E ricostruire le maschere da quella griglia
    sarebbe comunque la risposta sbagliata: quella shell non esegue affatto la
    policy a preferenze, quindi la coerenza con i pari -- l'unica cosa che
    questo fix difende -- si ottiene tornando all'albero.
    """

    class _Agent(LLMAgent):
        def _tree_decide(self, observation, world):
            return "tree-request"

    agent = _Agent.__new__(_Agent)
    agent.x, agent.y = 0, 0
    agent.decision_mode = "preferences"
    assert agent._fallback_decide(None, _ObjectWorldStub()) == "tree-request"


def test_every_fallback_return_in_both_paths_goes_through_the_hook():
    """I sei ritorni di ripiego, non solo alcuni, devono passare dal gancio.

    `decide` e `async_decide` hanno tre ritorni di ripiego ciascuno -- seconda
    chiamata nello stesso passo, budget esaurito, risposta rifiutata -- per sei
    in tutto. Uno lasciato a `super().decide` sarebbe un buco silenzioso: la run
    continuerebbe e solo quel ramo deciderebbe con la policy sbagliata, che e'
    esattamente il difetto che questo fix chiude.
    """
    source = inspect.getsource(LLMAgent.decide) + inspect.getsource(
        LLMAgent.async_decide
    )
    assert "super().decide(" not in source
    assert source.count("self._fallback_decide(observation, world)") == 6
