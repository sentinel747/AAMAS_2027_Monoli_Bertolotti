"""La policy inclina il menu, e senza policy non tocca niente.

L'applicazione deve funzionare su DUE forme: la griglia intera
(`MARSABM_CELL_SUBSET=0`) e il sottoinsieme compatto `1 x K`. Non c'e' doppia
indicizzazione da sbagliare — gli indicatori si calcolano dagli array di
`cells`, che hanno la stessa forma della `priority` — ma la promessa che le due
vie diano lo stesso numero va comunque fissata da un test.

Le semantiche da presidiare sono tre: prima regola che scatta vince (per
cella), la regola incondizionata chiude la catena, e le celle vuote non vengono
toccate — il governo agisce su chi c'e', non sul terreno.
"""

import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionType
from src.agents.pillars import ACTION_INDEX, N_ACTIONS
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.cell_subset import CellSubset
from src.governors.apply import apply_policy, indicator_values
from src.governors.policy import Condition, Policy, Rule

FORAGE = ACTION_INDEX[ActionType.FORAGE]
COLLECT_MINERALS = ACTION_INDEX[ActionType.COLLECT_MINERALS]
GREENHOUSE = ACTION_INDEX[ActionType.BUILD_GREENHOUSE]


def _cells(height=4, width=5):
    cells = CellArrays(height, width)
    return cells


def _priority(height=4, width=5, value=2.0):
    return np.full((height, width, N_ACTIONS), value, dtype=np.float64)


def _sustenance_rule(threshold=1.0, weight=3.0):
    return Rule(
        Condition("food_per_occupant", "<", threshold),
        {pillars.P_SUSTENANCE: weight},
    )


def test_an_empty_policy_touches_nothing():
    cells = _cells()
    cells.occupancy[2, 3] = 4
    priority = _priority()
    before = priority.copy()
    apply_policy(priority, Policy(), cells)
    assert np.array_equal(priority, before)


def test_a_matching_rule_weighs_every_action_of_the_pillar_only_there():
    cells = _cells()
    cells.occupancy[2, 3] = 4          # cibo 0 / 4 occupanti -> sotto soglia
    cells.occupancy[1, 1] = 2
    cells.cell_res[1, 1, C.R["food"]] = 10.0   # 5 a testa -> sopra soglia
    priority = _priority()

    apply_policy(priority, Policy((_sustenance_rule(),)), cells)

    for action in pillars.PILLAR_ACTIONS[pillars.P_SUSTENANCE]:
        index = ACTION_INDEX[action]
        assert priority[2, 3, index] == 6.0, "la cella affamata va pesata"
        assert priority[1, 1, index] == 2.0, "la cella sazia no"
    assert priority[2, 3, COLLECT_MINERALS] == 2.0, "gli altri pilastri restano"


def test_empty_cells_are_never_touched():
    cells = _cells()          # tutte a occupazione zero, tutte "sotto soglia"
    priority = _priority()
    before = priority.copy()
    apply_policy(priority, Policy((_sustenance_rule(),)), cells)
    assert np.array_equal(priority, before), (
        "il governo agisce sui popolani, non sul terreno vuoto"
    )


def test_first_matching_rule_wins_per_cell():
    """E' un if/elif/else, non una somma di pesi."""
    cells = _cells()
    cells.occupancy[0, 0] = 2   # cibo a zero: scattano ENTRAMBE le condizioni
    priority = _priority()

    policy = Policy((
        _sustenance_rule(threshold=1.0, weight=3.0),
        Rule(Condition("food_per_occupant", "<", 2.0), {pillars.P_SUSTENANCE: 0.5}),
    ))
    apply_policy(priority, policy, cells)
    assert priority[0, 0, FORAGE] == 6.0, (
        "deve valere la prima regola (x3), non la seconda ne' il prodotto"
    )


def test_the_unconditional_rule_closes_the_chain():
    cells = _cells()
    cells.occupancy[0, 0] = 2
    cells.cell_res[0, 0, C.R["food"]] = 10.0   # non affamata
    priority = _priority()

    policy = Policy((
        Rule(None, {pillars.P_BUILD: 2.0}),
        _sustenance_rule(threshold=100.0, weight=4.0),  # scatterebbe ovunque
    ))
    apply_policy(priority, policy, cells)
    assert priority[0, 0, GREENHOUSE] == 4.0, "la regola incondizionata agisce"
    assert priority[0, 0, FORAGE] == 2.0, (
        "dopo l'incondizionata nessuna regola successiva puo' piu' scattare"
    )


def test_the_greater_than_operator_works_too():
    cells = _cells()
    cells.occupancy[3, 3] = 200
    cells.occupancy[0, 0] = 3
    priority = _priority()
    policy = Policy((
        Rule(Condition("occupants", ">", 100.0), {pillars.P_EXPLORE: 4.0}),
    ))
    apply_policy(priority, policy, cells)
    explore = ACTION_INDEX[ActionType.EXPLORE]
    assert priority[3, 3, explore] == 8.0
    assert priority[0, 0, explore] == 2.0


def test_grid_and_subset_produce_identical_numbers():
    """Le due forme sono lo stesso dato: la policy deve dare gli stessi pesi."""
    cells = _cells()
    cells.occupancy[2, 3] = 4
    cells.occupancy[0, 1] = 2
    cells.cell_res[0, 1, C.R["food"]] = 6.0    # 3 a testa: solo (2,3) e' affamata

    agents = AgentArrays(2)
    for row, (y, x) in enumerate(((2, 3), (0, 1))):
        agents.ids.append(f"a{row}")
        agents.index[f"a{row}"] = row
        agents.alive[row] = True
        agents.y[row], agents.x[row] = y, x
    agents.n = 2
    subset = CellSubset.at_agent_cells(cells, agents, np.array([0, 1], dtype=np.int64))

    policy = Policy((_sustenance_rule(),))
    full = _priority()
    apply_policy(full, policy, cells)
    compact = np.full((1, subset.keys.size, N_ACTIONS), 2.0, dtype=np.float64)
    apply_policy(compact, policy, subset.cells)

    for position in range(subset.keys.size):
        key = int(subset.keys[position])
        y, x = divmod(key, subset.width)
        assert np.array_equal(compact[0, position], full[y, x]), (
            f"cella ({y},{x}): il sottoinsieme e la griglia devono coincidere"
        )


def test_ice_per_occupant_sums_deposit_and_surface():
    cells = _cells()
    cells.occupancy[1, 2] = 2
    cells.cell_res[1, 2, C.R["ice"]] = 1.0
    cells.water_ice[1, 2] = 3.0
    values = indicator_values("ice_per_occupant", cells)
    assert values[1, 2] == 2.0, "(1 + 3) / 2: e' la somma che decide collect_ice"


def test_hit_counters_partition_the_occupied_cells():
    """Prima-regola-vince rende gli insiemi disgiunti: regole + else devono
    ripartire ESATTAMENTE le celle occupate, o il conteggio mente."""
    from src.governors.apply import ELSE_KEY
    from src.governors.policy import rule_text

    cells = _cells()
    cells.occupancy[0, 0] = 2                                  # affamata
    cells.occupancy[1, 1] = 3
    cells.cell_res[1, 1, C.R["food"]] = 30.0                   # sazia -> else
    cells.occupancy[2, 2] = 1
    cells.cell_res[2, 2, C.R["food"]] = 10.0                   # sazia -> else
    priority = _priority()

    rule = _sustenance_rule(threshold=1.0)
    hits: dict = {}
    apply_policy(priority, Policy((rule,)), cells, hits=hits)
    apply_policy(_priority(), Policy((rule,)), cells, hits=hits)

    assert hits[rule_text(rule)] == 2, "una cella affamata, due passi"
    assert hits[ELSE_KEY] == 4, "due celle sazie, due passi"
    assert sum(hits.values()) == 6, "3 celle occupate x 2 passi, ne' piu' ne' meno"


def test_hit_counting_does_not_change_the_arrays():
    """Il conteggio e' pura osservazione: stessi numeri con e senza."""
    import numpy as np

    cells = _cells()
    cells.occupancy[2, 3] = 4
    policy = Policy((_sustenance_rule(),))
    counted = _priority()
    plain = _priority()
    apply_policy(counted, policy, cells, hits={})
    apply_policy(plain, policy, cells)
    assert np.array_equal(counted, plain)


def test_an_empty_policy_counts_everything_as_else_when_asked():
    from src.governors.apply import ELSE_KEY

    cells = _cells()
    cells.occupancy[0, 0] = 5
    hits: dict = {}
    apply_policy(_priority(), Policy(), cells, hits=hits)
    assert hits == {ELSE_KEY: 1}


def test_a_weight_cannot_resurrect_a_masked_action():
    """La cella resta il filtro: su priorita' zero un peso e' un non-intervento."""
    cells = _cells()
    cells.occupancy[2, 3] = 4
    priority = _priority(value=0.0)
    apply_policy(priority, Policy((_sustenance_rule(weight=4.0),)), cells)
    assert not priority.any(), "0 x 4 = 0: il peso inclina, non crea"
