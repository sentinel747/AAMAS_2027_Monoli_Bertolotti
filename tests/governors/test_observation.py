"""Il quadro aggregato: statistiche giuste, sulle celle giuste, digest stabile.

Il governatore non vede piu' celle ma distribuzioni: se una media fosse
calcolata sulle celle sbagliate (tutte invece delle occupate) o con gli
occupanti del kernel invece che dalle posizioni degli agenti vivi, la policy
verrebbe scritta su un mondo che non esiste — e nessun errore lo direbbe.
"""

import numpy as np

from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.governors.observation import build_picture, picture_digest
from src.governors.policy import INDICATORS


def _colony():
    """Due celle occupate (3 e 1 coloni), una piena di cibo e una no."""
    cells = CellArrays(4, 5)
    cells.cell_res[2, 3, C.R["food"]] = 6.0
    cells.cell_res[0, 1, C.R["construction_material"]] = 5.0
    cells.water_ice[0, 1] = 2.0
    cells.struct_count[2, 3, C.S["greenhouse"]] = 2
    agents = AgentArrays(4)
    positions = ((2, 3), (2, 3), (2, 3), (0, 1))
    for row, (y, x) in enumerate(positions):
        agents.ids.append(f"a{row}")
        agents.index[f"a{row}"] = row
        agents.alive[row] = True
        agents.y[row], agents.x[row] = y, x
        agents.health[row] = 0.5 + 0.1 * row
    agents.n = 4
    return cells, agents, np.arange(4, dtype=np.int64)


def test_the_picture_aggregates_over_occupied_cells_only():
    cells, agents, rows = _colony()
    picture = build_picture(7, {"food_margin": 1.2}, cells, agents, rows)

    assert picture.step == 7
    assert picture.population == 4
    assert picture.n_cells == 2
    food = picture.indicators["food_per_occupant"]
    # (2,3): 6/3 = 2; (0,1): 0/1 = 0 -> media 1, min 0, max 2.
    assert food["mean"] == 1.0 and food["min"] == 0.0 and food["max"] == 2.0
    ice = picture.indicators["ice_per_occupant"]
    assert ice["max"] == 2.0, "il ghiaccio di superficie conta"
    assert picture.structures == {"greenhouse": 2}
    assert picture.metrics == {"food_margin": 1.2}


def test_every_vocabulary_indicator_appears_in_the_picture():
    """La policy si scrive su questi nomi: il governatore deve vederli tutti."""
    cells, agents, rows = _colony()
    picture = build_picture(1, {}, cells, agents, rows)
    assert set(picture.indicators) == set(INDICATORS)
    for stats in picture.indicators.values():
        assert set(stats) == {"mean", "std", "min", "max"}


def test_population_stats_average_the_living_rows():
    cells, agents, rows = _colony()
    picture = build_picture(1, {}, cells, agents, rows)
    assert picture.population_stats["health_mean"] == round((0.5 + 0.6 + 0.7 + 0.8) / 4, 3)


def test_dead_rows_do_not_count():
    cells, agents, _ = _colony()
    picture = build_picture(1, {}, cells, agents, np.array([0, 3], dtype=np.int64))
    assert picture.population == 2
    assert picture.n_cells == 2
    occupants = picture.indicators["occupants"]
    assert occupants["max"] == 1.0, "la cella (2,3) ha UN vivo fra quelli passati"


def test_an_empty_colony_yields_an_empty_but_valid_picture():
    cells, agents, _ = _colony()
    picture = build_picture(1, {}, cells, agents, np.array([], dtype=np.int64))
    assert picture.population == 0
    assert picture.indicators == {}
    assert picture.population_stats == {}
    assert picture_digest(picture), "anche il quadro vuoto deve avere un digest"


def test_the_digest_is_stable_and_sensitive():
    cells, agents, rows = _colony()
    first = picture_digest(build_picture(7, {"a": 1.0}, cells, agents, rows))
    second = picture_digest(build_picture(7, {"a": 1.0}, cells, agents, rows))
    assert first == second, "stesso quadro, stesso digest"
    moved = picture_digest(build_picture(8, {"a": 1.0}, cells, agents, rows))
    assert moved != first, "un passo diverso e' un quadro diverso"
