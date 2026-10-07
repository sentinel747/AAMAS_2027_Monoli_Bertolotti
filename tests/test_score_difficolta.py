# -*- coding: utf-8 -*-
"""Il punteggio di difficolta' deve essere esogeno, ordinato e riproducibile.

- la cella madre e' quella del centro dichiarato nei metadati;
- la finestra si chiude in longitudine (la griglia e' un cilindro);
- un mondo senza pericoli, ricco come il riferimento, con sito perfetto e
  dotazione piena ha D = 0; uno all'opposto ha D = 1;
- togliere risorse alza `scarsita'` e lascia le altre componenti ferme;
- Spearman: +1 su ranghi concordi, -1 su discordi, con i pareggi mediati.
"""

from scripts.score_difficolta import (
    RIFERIMENTO,
    cella_madre,
    componenti,
    finestra,
    spearman,
)


def _base(width=20, height=10, lat=0.0, lon=0.0, **campi):
    default = {
        "radiation": 0.0, "dust": 0.0, "terrain_base_risk": 0.0, "traversal_risk": 0.0,
        "colony_site_score": 1.0, "water_ice": RIFERIMENTO["water_ice"],
        "resources": {"minerals": RIFERIMENTO["minerals"], "construction_material": RIFERIMENTO["construction_material"]},
    }
    default.update(campi)
    return {
        "width": width, "height": height,
        "metadata": {"center_latitude_deg": lat, "center_longitude_deg": lon},
        "cells": [dict(default, x=x, y=y) for y in range(height) for x in range(width)],
    }


def test_mother_cell_comes_from_the_declared_centre():
    base = _base(width=360, height=180, lat=-4.5, lon=29.5)
    assert cella_madre(base) == (94, 209)   # verificato contro il registro degli amministratori


def test_the_window_wraps_around_in_longitude():
    base = _base(width=20, height=10, lat=0.0, lon=-180.0)  # madre a x = 0
    assert cella_madre(base)[1] == 0
    W = finestra(base, 2)
    assert len(W) == 5 * 5
    assert any(c["x"] == 19 for c in W) and any(c["x"] == 1 for c in W)


def test_easiest_and_hardest_worlds_span_zero_to_one():
    facile = componenti(_base(), dotazione=1.0, raggio=2)
    assert facile["D"] == 0.0 and facile["celle_finestra"] == 25
    difficile = componenti(
        _base(radiation=2.0, dust=1.0, terrain_base_risk=1.0, traversal_risk=1.0, colony_site_score=0.0,
              water_ice=0.0, resources={"minerals": 0.0, "construction_material": 0.0}),
        dotazione=0.0, raggio=2,
    )
    assert difficile["D"] == 1.0


def test_halving_resources_moves_only_scarcity():
    ricco = componenti(_base(), dotazione=0.6, raggio=2)
    povero = componenti(
        _base(water_ice=RIFERIMENTO["water_ice"] / 2,
              resources={"minerals": RIFERIMENTO["minerals"] / 2, "construction_material": RIFERIMENTO["construction_material"] / 2}),
        dotazione=0.6, raggio=2,
    )
    assert ricco["scarsita"] == 0.0 and povero["scarsita"] == 0.5
    assert ricco["pericolo"] == povero["pericolo"] == 0.0
    assert ricco["dotazione"] == povero["dotazione"] == 0.4
    assert povero["D"] > ricco["D"]


def test_spearman_signs_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert spearman([1, 1, 2], [1, 2, 3]) == 0.866
    assert spearman([1, 2], [1, 2]) is None
