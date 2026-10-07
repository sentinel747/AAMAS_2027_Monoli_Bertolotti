# -*- coding: utf-8 -*-
"""Il pozzo eroga la razione delle persone che serve, e l'acqua conta.

**Perche' esiste (2026-09-01).** `local_life_support_capacity` dichiarava quante
persone una cella sostiene guardando alloggi, cibo, energia e ossigeno — e mai
se poteva DISSETARLE. Un avamposto risultava percio' operativo mentre i suoi
abitanti morivano di sete: misurato all'istante del decesso, quattordici casi su
diciannove avevano il magazzino idrico sotto 0,1 con cibo e ossigeno abbondanti
(voce C9 della checklist), e l'unica fonte d'acqua di struttura era la serra, che
per le sue sette persone ne produce 0,423 mentre loro ne bevono 0,7.

La relazione che questa prova sorveglia attraversa **tre strati** — la resa sta
in `src/world`, la razione in `src/agents`, il coefficiente di produzione in
`src/simulation` — e non puo' essere scritta in un posto solo senza invertire le
dipendenze. E' percio' un test a tenerla ferma, come per le linee di base dello
strato psicosociale.
"""
from __future__ import annotations

from src.agents.build_policy import (
    COLONISTS_PER_STRUCTURE,
    local_life_support_capacity,
    posti_di_cantiere,
    termini_di_capienza,
)
from src.agents.vitals import RAZIONE_ACQUA
from src.simulation.step_effects import RESA_ACQUA
from src.world.cell import Cell
from src.world.structures import (
    STRUTTURE_DI_AVVIAMENTO,
    Structure,
    StructureType,
)
from src.world.terrain import TerrainType


def _cella(*tipi: StructureType) -> Cell:
    c = Cell(x=0, y=0, terrain=TerrainType.REGOLITH_PLAIN)
    for tipo in tipi:
        c.structures.append(Structure(type=tipo, x=0, y=0, integrity=1.0))
    return c


def test_la_resa_del_pozzo_e_la_razione_di_chi_serve():
    """`resa x RESA_ACQUA` deve valere `persone servite x RAZIONE_ACQUA`.

    E' l'equazione che ricava `local_effect["water"]` del pozzo. Se qualcuno
    cambia la razione, il coefficiente di produzione o il numero di persone
    servite senza toccare la resa, il pozzo smette di coprire il fabbisogno che
    dichiara di coprire — e lo fa in silenzio, perche' la colonia continua a
    costruirlo.
    """
    pozzo = Structure(type=StructureType.WATER_EXTRACTOR, x=0, y=0, integrity=1.0)
    resa = float(pozzo.local_effect["water"]) * RESA_ACQUA
    servite = COLONISTS_PER_STRUCTURE[StructureType.WATER_EXTRACTOR]
    assert abs(resa - servite * RAZIONE_ACQUA) < 1.0e-3, (
        f"il pozzo produce {resa:.4f} per {servite} persone che ne bevono "
        f"{servite * RAZIONE_ACQUA:.4f}"
    )


def test_la_serra_da_sola_non_copre_il_fabbisogno_idrico():
    """La ragione per cui il pozzo serve, scritta come misura.

    Non e' un difetto della serra: produce acqua come sottoprodotto della
    coltivazione. E' un difetto del modello dichiarare «operativa» una cella che
    ha solo quella.
    """
    serra = Structure(type=StructureType.GREENHOUSE, x=0, y=0, integrity=1.0)
    resa = float(serra.local_effect["water"]) * RESA_ACQUA
    servite = COLONISTS_PER_STRUCTURE[StructureType.GREENHOUSE]
    assert resa < servite * RAZIONE_ACQUA


def test_senza_pozzo_una_cella_non_ha_capienza_vitale():
    """L'acqua e' un requisito come gli altri tre, non un di piu'."""
    senza = _cella(
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # NIENTE pozzo: e' il punto di questa prova.
    )
    assert local_life_support_capacity(senza) == 0
    assert posti_di_cantiere(senza) == 1  # manca solo l'acqua

    con = _cella(*STRUTTURE_DI_AVVIAMENTO)
    assert local_life_support_capacity(con) > 0
    assert posti_di_cantiere(con) == 0


def test_i_termini_di_capienza_sono_quelli_dell_avviamento():
    """Una cella vergine ha da coprire tanti requisiti quante sono le strutture.

    E' la relazione che lega `posti_di_cantiere` (la squadra di fondazione) a
    `STRUTTURE_DI_AVVIAMENTO` (il kit del fondatore) e a `REGOLITE_LAVORABILE`
    (il regolito che ogni cella contiene): se una delle tre si muove da sola, la
    squadra arriva con il kit sbagliato o la cella non ha di che pagarlo.
    """
    vergine = _cella()
    assert len(termini_di_capienza(vergine)) == len(STRUTTURE_DI_AVVIAMENTO)
    assert posti_di_cantiere(vergine) == len(STRUTTURE_DI_AVVIAMENTO)
