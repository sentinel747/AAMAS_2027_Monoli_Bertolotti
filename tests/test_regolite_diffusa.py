# -*- coding: utf-8 -*-
"""Marte non e' roccia morta: ogni cella ha del regolito lavorabile.

Fino al 2026-09-01 le uniche celle con qualcosa da estrarre erano quelle vicine
ai ventiquattro centri minerari: il 6,4-7,1% del pianeta, e nella fascia
equatoriale -- dove le colonie vivono -- il 92-94% delle celle era
COMPLETAMENTE vuoto. Un avamposto fondato fuori da una vena non poteva
raccogliere nulla ne' costruire nulla (voce C13 della checklist).

Queste prove sorvegliano le tre cose che rendono la dotazione una dotazione
ricavata e non una manopola: che il fondo valga quanto serve ad avviare un
avamposto, che la miscela sia quella che la colonia consuma, e che il profilo
di mappa continui a mordere.
"""
from __future__ import annotations

from src.world.structures import (
    BUILD_COSTS,
    COSTO_MATERIALE_AVVIAMENTO,
    DOMANDA_MINERALI_SU_MATERIALE,
    STRUTTURE_DI_AVVIAMENTO,
)
from src.world.world_generator import REGOLITE_LAVORABILE, WorldGenerator

W, H = 120, 60


def _celle(seme: int, profilo: str = "balanced"):
    mondo = WorldGenerator(seed=seme, map_profile=profilo).generate(W, H)
    return [c for riga in mondo.cells for c in riga]


def test_il_fondo_e_il_costo_di_avviare_un_avamposto():
    """Il fondo non e' scelto: e' letto da `BUILD_COSTS`.

    Le quattro strutture che sbloccano la capienza vitale costano dodici unita'
    di materiale. Se un costo cambia, il pianeta lo segue da solo.
    """
    atteso = sum(
        float(BUILD_COSTS[tipo].construction_material) for tipo in STRUTTURE_DI_AVVIAMENTO
    )
    assert REGOLITE_LAVORABILE == COSTO_MATERIALE_AVVIAMENTO == atteso


def test_la_fascia_equatoriale_non_ha_celle_vuote():
    """Dove le colonie vivono, ogni cella offre qualcosa.

    E' la proprieta' che C13 chiedeva: cio' che un colono puo' fare nella cella
    madre deve poterlo fare in qualunque cella colonizzi.
    """
    for seme in (0, 9, 21):
        vuote = [
            c
            for c in _celle(seme)
            if abs(c.y - H // 2) <= 8
            and float(c.resources.construction_material) <= 0.0
        ]
        assert not vuote, f"seme {seme}: {len(vuote)} celle equatoriali senza materiale"


def test_la_cella_mediana_puo_avviare_un_avamposto():
    """Meta' delle celle ha almeno cio' che serve a fondarci sopra.

    Il fattore casuale e' simmetrico intorno a uno, quindi la mediana della
    fascia non polare deve valere il fondo. Senza questa proprieta' il numero
    sarebbe una taratura invece che una derivazione.
    """
    for seme in (0, 9, 21):
        materiale = sorted(
            float(c.resources.construction_material)
            for c in _celle(seme)
            if float(getattr(c, "polar_severity", 0.0)) <= 0.0
        )
        mediana = materiale[len(materiale) // 2]
        assert mediana >= COSTO_MATERIALE_AVVIAMENTO * 0.9, (
            f"seme {seme}: mediana {mediana:.2f} contro un fondo di "
            f"{COSTO_MATERIALE_AVVIAMENTO}"
        )


def test_la_miscela_e_quella_che_la_colonia_consuma():
    """Minerali e materiale stanno nel rapporto che `BUILD_COSTS` chiede.

    Prima il materiale valeva `minerali x 0,35`, cioe' il pianeta offriva 2,86
    minerali per unita' di materiale mentre il catalogo ne chiede 0,58: cinque
    volte piu' minerale del necessario, e materiale col contagocce. Nessuno dei
    due beni deve poter strozzare l'altro per un disallineamento fra tabelle.
    """
    for c in _celle(0):
        materiale = float(c.resources.construction_material)
        if materiale <= 0.0:
            continue
        atteso = materiale * DOMANDA_MINERALI_SU_MATERIALE
        assert abs(float(c.resources.minerals) - atteso) <= 0.002


def test_le_calotte_polari_sono_ghiaccio_non_regolito():
    """Dove la calotta e' piena il FONDO si spegne, e non e' una costante nuova.

    `polar_severity` esiste gia' e vale 1 sulla calotta solida: il regolito
    lavorabile e' quello che il ghiaccio non copre. Le VENE restano dove il
    generatore le ha messe, calotta compresa — un giacimento sotto il ghiaccio
    e' comunque un giacimento — quindi la proprieta' da sorvegliare e' che la
    calotta non abbia il fondo, non che sia sterile: la sua cella mediana vale
    zero, mentre quella non polare vale il fondo intero.
    """
    celle = _celle(0)
    polari = sorted(
        float(c.resources.construction_material)
        for c in celle
        if float(getattr(c, "polar_severity", 0.0)) >= 1.0
    )
    assert polari, "la mappa di prova deve avere calotte"
    assert polari[len(polari) // 2] <= 0.001
    non_polari = sorted(
        float(c.resources.construction_material)
        for c in celle
        if float(getattr(c, "polar_severity", 0.0)) <= 0.0
    )
    assert non_polari[len(non_polari) // 2] >= COSTO_MATERIALE_AVVIAMENTO * 0.9


def test_il_profilo_di_mappa_morde_ancora():
    """La scarsita' deve restare una leva, non un'etichetta.

    Il fondo segue il profilo con la stessa forza delle vene: `_forza_del_profilo`
    e' `centri x (raggio / raggio_di_riferimento)^3`, cioe' la geometria con cui
    il totale di una mappa scala. Senza questo il fondo avrebbe smussato la leva
    da un ordine di grandezza a meno della meta'.
    """
    def totale(profilo: str) -> float:
        return sum(float(c.resources.minerals) for c in _celle(0, profilo))

    assert totale("scarce_resources") < totale("balanced") / 5
    assert totale("mineral_rich") > totale("balanced") * 3
