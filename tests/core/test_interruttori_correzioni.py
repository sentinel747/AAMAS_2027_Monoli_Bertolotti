# -*- coding: utf-8 -*-
"""Gli interruttori delle correzioni al motore: spenti, il motore e' quello d'archivio.

**Perche' questo test esiste.** Due difetti trovati il 14 settembre 2026 sono
stati messi dietro un interruttore invece di essere corretti in blocco: correggere
e basta avrebbe cambiato il simulatore sotto una campagna in corso, e reso i suoi
numeri non confrontabili con le 191 esecuzioni gia' in archivio. La condizione
perche' quella scelta regga e' una sola, e va verificata invece che sperata: **a
interruttori spenti non deve cambiare niente**.

Il test verifica anche il contrario --- che accesi cambino qualcosa --- perche' un
interruttore che non fa niente e' peggio di nessun interruttore: si misurerebbe
la stessa cosa due volte credendo di avere un confronto.
"""

import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionType
from src.core import constants as C
from src.core.arrays import CellArrays
from src.core.cell_action_mask import compute_cell_masks
from src.core.cell_proposals import TASK_ACTIONS, compute_cell_proposals
from src.core.needs import RedistributionConfig, compute_needs
from src.world.structures import StructureType
from src.world.terrain import TerrainType
from src.world.world_generator import WorldGenerator

#: Il valore con cui sono state misurate tutte le campagne in archivio.
PAVIMENTO_DI_ARCHIVIO = 1.1

#: Cinque tipi di struttura in cantiere: con il taglio a cinque del menu di
#: cella bastano questi perche' il pavimento espella ogni altra attivita'.
CANTIERI = (
    StructureType.SHELTER,
    StructureType.HABITAT,
    StructureType.GREENHOUSE,
    StructureType.SOLAR_ARRAY,
    StructureType.OXYGEN_PLANT,
)


def _cella_matura():
    """Una cella che ha di che mangiare, di che bere e di che riparare."""
    world = WorldGenerator(seed=71).generate(32, 20)
    world.get_cell(8, 7).terrain = TerrainType.REGOLITH_PLAIN
    cells = CellArrays.from_world(world)
    x, y = 8, 7
    cells.occupancy[y, x] = 40
    for tipo, quante in ((StructureType.SHELTER, 20),
                         (StructureType.GREENHOUSE, 6),
                         (StructureType.SOLAR_ARRAY, 6),
                         (StructureType.OXYGEN_PLANT, 4)):
        cells.struct_count[y, x, C.S[tipo]] = quante
        cells.struct_integrity[y, x, C.S[tipo]] = 0.7 * quante
    cells.water_ice[y, x] = 200.0
    cells.cell_res[y, x, C.R["minerals"]] = 200.0
    cells.cell_res[y, x, C.R["construction_material"]] = 200.0
    cells.vegetation[y, x] = 4.0
    cells.proto_soil[y, x] = 2.0
    return cells, (x, y)


def _menu(cells, pavimento=None):
    needs = compute_needs(cells, RedistributionConfig())
    extra = {} if pavimento is None else {"pavimento_cantieri": pavimento}
    return compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=5, **extra
    )


def _attivita_di_lavoro(proposals, x, y):
    return {
        azione for azione in TASK_ACTIONS
        if proposals.mask[y, x, pillars.ACTION_INDEX[azione]]
    }


def test_il_pavimento_predefinito_e_quello_di_archivio():
    """Il valore di sempre deve restare il predefinito, o l'archivio si spezza."""
    cells, (x, y) = _cella_matura()
    esplicito = _menu(cells, PAVIMENTO_DI_ARCHIVIO)
    predefinito = _menu(cells)
    assert np.array_equal(predefinito.mask, esplicito.mask)
    assert np.allclose(predefinito.priority, esplicito.priority)
    assert np.array_equal(predefinito.quota, esplicito.quota)


def test_con_cinque_cantieri_il_pavimento_di_archivio_svuota_il_menu():
    """Il difetto, messo per iscritto: e' il comportamento da confrontare.

    Cinque cantieri aperti e un menu tagliato a cinque: le attivita' che restano
    sono tutte costruzione, e chi riempiva il magazzino o riparava le strutture
    non compare piu'.
    """
    cells, (x, y) = _cella_matura()
    prima = _attivita_di_lavoro(_menu(cells), x, y)
    for tipo in CANTIERI:
        cells.site_progress[y, x, C.S[tipo]] = 0.5
    dopo = _attivita_di_lavoro(_menu(cells), x, y)

    assert prima - dopo, "il pavimento non toglie niente: la prova non prova nulla"
    costruzioni = {a for a in dopo if a.value.startswith("build_")}
    assert dopo == costruzioni, f"restano anche attivita' non di costruzione: {dopo - costruzioni}"


def test_abbassare_il_pavimento_restituisce_il_menu():
    """Acceso l'interruttore, le attivita' di sostentamento tornano nel menu."""
    cells, (x, y) = _cella_matura()
    for tipo in CANTIERI:
        cells.site_progress[y, x, C.S[tipo]] = 0.5
    archivio = _attivita_di_lavoro(_menu(cells, PAVIMENTO_DI_ARCHIVIO), x, y)
    corretto = _attivita_di_lavoro(_menu(cells, 0.85), x, y)
    tornate = corretto - archivio
    assert tornate, "abbassare il pavimento non cambia il menu: l'interruttore e' inerte"
