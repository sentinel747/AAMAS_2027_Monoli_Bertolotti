"""Gli eventi catastrofici devono lasciare un segno, e il segno giusto.

Il pericolo che questi test coprono e' stato misurato prima di essere
scritto: su 400 passi con 300 coloni, diciotto eventi arrivati e settantuno
passi sotto evento, la salute totale persa era **0,03 punti** e le morti
zero. Un evento c'era, si vedeva nei log e nella GUI, e non faceva nulla —
il modo peggiore di sbagliare, perche' un apparato visibile e inerte si
legge come "il modello prevede le tempeste" senza che ne prevenga alcun
effetto.

Le due cause erano una riga ciascuna: il riparo era una proprieta' della
cella (una struttura qualsiasi proteggeva tutti i trecento occupanti, e il
99,9% degli agenti colpiti risultava protetto), e l'evento non toccava
nessuna variabile fisica del mondo, quindi restava fuori dalle catene che
il simulatore gia' percorre.
"""

import numpy as np
import pytest

from src.simulation.extreme_events import (
    DUST_STORM_OPACITY_SURGE,
    SHELTER_CAPACITY,
    SOLAR_FLARE_RADIATION_SURGE,
    ExtremeEventEngine,
)
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType


def _mondo(width=6, height=5):
    # `GridWorld` e' un dataclass che vuole la griglia gia' costruita: stesso
    # modo di tests/core/test_kernel_biology_equivalence.py.
    celle = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)]
             for y in range(height)]
    world = GridWorld(width=width, height=height, cells=celle)
    for row in world.cells:
        for cell in row:
            cell.dust_level = 0.20
            cell.radiation_level = 0.60
    return world


def _evento(tipo, severity=1.0):
    return {"type": tipo, "severity": severity, "started_step": 0,
            "duration_steps": 3, "cells": [], "trail_cells": []}


def _celle(*coordinate):
    return [{"x": x, "y": y} for x, y in coordinate]


class _Agente:
    """Il minimo che `_apply_event_impacts` tocca di un colono."""

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.health = 1.0
        self.fatigue = 0.0
        self.memory = type("M", (), {"add_event": lambda self, *a: None})()


def test_a_dust_storm_raises_the_dust_of_the_cells_it_crosses():
    """E' il canale che nessun riparo annulla: da `dust` la catena prosegue
    verso l'abitabilita' (`recompute_habitability` sottrae `dust * 0,02`) e
    verso l'usura (`dust * 0,010` per anno)."""
    world = _mondo()
    motore = ExtremeEventEngine()
    prima = world.get_cell(1, 1).dust_level

    motore._apply_event_impacts(world, {}, _evento("Global Dust Storm"), _celle((1, 1)))

    dopo = world.get_cell(1, 1).dust_level
    assert dopo > prima, "una tempesta deve alzare la polvere della cella"
    assert dopo == pytest.approx(min(1.0, prima + DUST_STORM_OPACITY_SURGE))
    assert world.get_cell(4, 4).dust_level == prima, "le celle fuori dal fronte restano"


def test_a_solar_flare_raises_the_radiation_of_the_cells_it_crosses():
    world = _mondo()
    motore = ExtremeEventEngine()
    prima = world.get_cell(2, 2).radiation_level

    motore._apply_event_impacts(world, {}, _evento("Solar Flare"), _celle((2, 2)))

    assert world.get_cell(2, 2).radiation_level == pytest.approx(
        min(1.8, prima + SOLAR_FLARE_RADIATION_SURGE)
    )
    assert world.get_cell(2, 2).dust_level == 0.20, "un brillamento non alza la polvere"


def test_the_front_leaves_the_cells_it_has_passed_as_it_found_them():
    """Il fronte si muove: senza ripristino la mappa accumulerebbe polvere a
    ogni passaggio e la tempesta diventerebbe permanente."""
    world = _mondo()
    motore = ExtremeEventEngine()
    evento = _evento("Global Dust Storm")

    motore._apply_event_impacts(world, {}, evento, _celle((0, 0)))
    assert world.get_cell(0, 0).dust_level > 0.20
    motore._apply_event_impacts(world, {}, evento, _celle((1, 0)))

    assert world.get_cell(0, 0).dust_level == pytest.approx(0.20), (
        "la cella attraversata al passo prima deve tornare al suo clima"
    )
    assert world.get_cell(1, 0).dust_level > 0.20


def test_the_environment_returns_to_normal_when_the_event_ends():
    world = _mondo()
    motore = ExtremeEventEngine()
    motore._apply_event_impacts(world, {}, _evento("Solar Flare"), _celle((3, 3)))
    assert world.get_cell(3, 3).radiation_level > 0.60

    motore._restore_environment(world)
    assert world.get_cell(3, 3).radiation_level == pytest.approx(0.60)


def test_shelter_has_a_capacity_and_the_overflow_stays_exposed():
    """La correzione della riga che rendeva inerte l'intero apparato: prima
    bastava UNA struttura per dichiarare al riparo qualunque folla."""
    world = _mondo()
    cell = world.get_cell(1, 1)
    cell.structures.append(Structure(type=StructureType.HABITAT, x=1, y=1))  # 5 posti
    agenti = {f"a{i}": _Agente(1, 1) for i in range(8)}

    ExtremeEventEngine()._apply_event_impacts(
        world, agenti, _evento("Solar Flare", severity=1.0), _celle((1, 1))
    )

    salute = sorted(a.health for a in agenti.values())
    esposti = [h for h in salute if h < 1.0]
    assert len(esposti) == 3, (
        f"un habitat ripara {SHELTER_CAPACITY[StructureType.HABITAT]} coloni su otto: "
        f"tre restano fuori, invece ne sono stati colpiti {len(esposti)}"
    )
    assert all(h == pytest.approx(0.90) for h in esposti)


def test_a_ruined_shelter_does_not_shelter():
    world = _mondo()
    cell = world.get_cell(1, 1)
    cell.structures.append(Structure(type=StructureType.HABITAT, x=1, y=1, integrity=0.2))
    agenti = {"a": _Agente(1, 1)}

    ExtremeEventEngine()._apply_event_impacts(
        world, agenti, _evento("Solar Flare"), _celle((1, 1))
    )
    assert agenti["a"].health < 1.0, "sotto il 30% di integrita' non ripara piu'"


def test_solar_yield_follows_the_dust_and_both_engines_agree():
    """Il canale per cui una tempesta di polvere e' pericolosa DAVVERO.

    `struct_fx` nasce da una matrice di effetti costanti, quindi senza questo
    fattore un pannello renderebbe uguale sotto un cielo limpido e dentro una
    tempesta globale. La profondita' e' quella misurata dal MCD: 57,8 W/m2
    contro 527 in climatologia, cioe' l'11%.

    **La parita' fra i due motori e' la meta' importante del test.** Il
    fattore e' stato aggiunto prima al solo kernel, e nessun test se n'e'
    accorto perche' i percorsi che confrontano i motori usano polvere bassa,
    dove il fattore vale uno: misurato a mano, 0,075 contro 0,0439 a dust
    0,60. Qui si guarda dove diverge.
    """
    from src.core import constants as C
    from src.core.arrays import AgentArrays, CellArrays
    from src.core.kernel_biology import SOLAR_DUST_LOSS, TYPICAL_DUST, update_cells
    from src.simulation.step_effects import apply_structure_effects, solar_yield

    assert solar_yield(TYPICAL_DUST) == pytest.approx(1.0)
    assert solar_yield(0.0) == pytest.approx(1.0), "sotto il tipico non si guadagna"
    assert solar_yield(1.0) == pytest.approx(1.0 - SOLAR_DUST_LOSS)

    stato = {"mean_temperature_c": -63.0, "pressure_pa": 600.0,
             "liquid_water_stability": 0.0}
    for dust in (0.20, 0.60, 1.00):
        world = _mondo(1, 1)
        world.add_structure(Structure(type=StructureType.SOLAR_ARRAY, x=0, y=0))
        world.get_cell(0, 0).dust_level = dust
        apply_structure_effects(world)
        oggetti = world.get_cell(0, 0).resources.energy

        world2 = _mondo(1, 1)
        world2.add_structure(Structure(type=StructureType.SOLAR_ARRAY, x=0, y=0))
        world2.get_cell(0, 0).dust_level = dust
        celle = CellArrays.from_world(world2)
        celle.recompute_struct_fx()
        update_cells(celle, AgentArrays.from_agents({}), stato, 7.0,
                     cell_degradation=True, full_grid=True)
        kernel = float(celle.cell_res[0, 0, C.R["energy"]])

        assert oggetti == pytest.approx(kernel), (
            f"a dust {dust} i due motori divergono sull'energia solare: "
            f"oggetti={oggetti}, kernel={kernel}"
        )
    assert oggetti < 0.02, "a polvere satura la resa deve essere crollata"


def test_a_well_equipped_capital_still_protects_its_own():
    """La capienza non deve punire chi si e' attrezzato: dodici habitat
    riparano sessanta coloni, e trenta ci stanno comodamente."""
    world = _mondo()
    cell = world.get_cell(2, 2)
    for _ in range(12):
        cell.structures.append(Structure(type=StructureType.HABITAT, x=2, y=2))
    agenti = {f"a{i}": _Agente(2, 2) for i in range(30)}

    ExtremeEventEngine()._apply_event_impacts(
        world, agenti, _evento("Solar Flare"), _celle((2, 2))
    )
    assert all(a.health == 1.0 for a in agenti.values())
    # ...ma l'ambiente li raggiunge lo stesso, ed e' il punto del ridisegno.
    assert world.get_cell(2, 2).radiation_level > 0.60
