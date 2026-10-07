"""La massa si conserva: guardia contro il difetto piu' difficile da vedere.

Due difetti gravi di questo simulatore non si sono visti guardando gli esiti —
la colonia sopravviveva, le metriche erano plausibili, i test passavano. Il
primo era un'unita' di misura sbagliata, il secondo una legge di conservazione
violata: `REFILL_WATER` aggiungeva all'inventario senza sottrarre da nulla, e
la colonia accumulava 199.774 unita' d'acqua in ottocento passi.

`scripts/audit_conservazione.py` fa il bilancio completo su una run vera; questi
test fissano le stesse proprieta' in forma puntuale, dove un errore futuro
verrebbe visto subito invece che alla prossima campagna.

La proprieta' generale e': **un'azione di TRASPORTO non cambia la massa
totale.** Raccogliere, rifornirsi, attingere spostano; solo produzione e
consumo la cambiano, e devono dichiararsi.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.action_space import ActionRequest, ActionType, execute_action
from src.agents.base_agent import BaseAgent
from src.agents.vitals import RAZIONE_ACQUA, RAZIONE_CIBO, tick_agent_vitals
from src.simulation.step_effects import (
    CARICO_ELETTRICO,
    RESA_ENERGIA_PER_CARICO,
    apply_colony_resource_feedback,
    apply_structure_effects,
)
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.kernel_vitals import tick_vitals
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.resources import ResourceBundle
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType

#: I serbatoi ambientali partecipano al bilancio: raccogliere ghiaccio sposta
#: massa da qui a un inventario, e ometterli farebbe apparire ogni estrazione
#: come una creazione dal nulla.
SERBATOI = {"water_ice": "ice", "liquid_water": "water", "vegetation_biomass": "biomass"}


def _mondo(**risorse):
    celle = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(3)]
             for y in range(3)]
    world = GridWorld(width=3, height=3, cells=celle)
    cell = world.get_cell(1, 1)
    for nome, valore in risorse.items():
        if hasattr(cell.resources, nome):
            setattr(cell.resources, nome, valore)
        else:
            setattr(cell, nome, valore)
    return world, cell


def _massa(world: GridWorld, agenti: dict) -> dict:
    tot: dict = {}
    for riga in world.cells:
        for c in riga:
            for nome, valore in c.resources.to_dict().items():
                tot[nome] = tot.get(nome, 0.0) + valore
            for attr, risorsa in SERBATOI.items():
                tot[risorsa] = tot.get(risorsa, 0.0) + float(getattr(c, attr, 0.0))
    for a in agenti.values():
        for nome, valore in a.inventory.to_dict().items():
            tot[nome] = tot.get(nome, 0.0) + valore
    return tot


def _agente(**kw) -> BaseAgent:
    return BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1, **kw)


# ---------------------------------------------------------------------------
# Le azioni di trasporto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "azione,risorse,inventario",
    [
        (ActionType.COLLECT_ICE, {"water_ice": 5.0}, {}),
        (ActionType.COLLECT_MINERALS, {"minerals": 5.0}, {}),
        (ActionType.COLLECT_MATERIALS, {"construction_material": 5.0}, {}),
        # Con l'impianto idrico: e' il ramo che creava acqua dal nulla.
        (ActionType.REFILL_WATER, {"water": 5.0}, {"water": 0.5}),
        (ActionType.REFILL_WATER, {"liquid_water": 5.0}, {"water": 0.5}),
        (ActionType.REFILL_WATER, {"water_ice": 5.0}, {"water": 0.5}),
        # La serra: raccogliere e' prelevare dal magazzino della cella.
        (ActionType.FORAGE, {"food": 5.0}, {}),
    ],
)
def test_le_azioni_di_trasporto_non_creano_massa(azione, risorse, inventario):
    struttura = {
        ActionType.REFILL_WATER: StructureType.GREENHOUSE,
        ActionType.FORAGE: StructureType.GREENHOUSE,
    }.get(azione)
    world, cell = _mondo(**risorse)
    if struttura is not None:
        world.add_structure(Structure(type=struttura, x=1, y=1, integrity=1.0))
    agent = _agente(inventory=ResourceBundle(**inventario))
    agenti = {agent.agent_id: agent}

    prima = _massa(world, agenti)
    esito = execute_action(agent, agenti, world, ActionRequest("a0", azione))
    dopo = _massa(world, agenti)

    assert esito.accepted, esito.message
    for risorsa in set(prima) | set(dopo):
        assert dopo.get(risorsa, 0.0) == pytest.approx(prima.get(risorsa, 0.0), abs=1e-9), (
            f"{azione.value} ha cambiato la massa di {risorsa}: "
            f"{prima.get(risorsa, 0.0)} -> {dopo.get(risorsa, 0.0)}"
        )


def test_bere_da_un_impianto_preleva_dalla_cella():
    """Gemello del difetto di REFILL_WATER: il ramo idratava senza prelevare."""
    world, cell = _mondo(water=5.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    agent = _agente()
    agent.hydration = 0.2
    agenti = {agent.agent_id: agent}

    esito = execute_action(agent, agenti, world, ActionRequest("a0", ActionType.DRINK_WATER))

    assert esito.accepted
    assert esito.data["source"] == "life support structure"
    assert cell.resources.water == pytest.approx(4.9)
    assert agent.hydration > 0.2


def test_la_serra_senza_raccolto_non_produce_cibo_dal_nulla():
    world, cell = _mondo(food=0.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    agent = _agente(inventory=ResourceBundle())
    agenti = {agent.agent_id: agent}

    esito = execute_action(agent, agenti, world, ActionRequest("a0", ActionType.FORAGE))

    assert not esito.accepted
    assert agent.inventory.food == 0.0


# ---------------------------------------------------------------------------
# La razione vitale
# ---------------------------------------------------------------------------

def test_la_razione_vitale_preleva_dalla_giacenza_della_cella():
    world, cell = _mondo(food=5.0, water=5.0, oxygen=5.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    world.add_structure(Structure(type=StructureType.OXYGEN_PLANT, x=1, y=1, integrity=1.0))
    agent = _agente()
    cell.agents_present = [agent.agent_id]

    tick_agent_vitals(agent, cell, 7.0)

    assert cell.resources.food == pytest.approx(5.0 - RAZIONE_CIBO)
    assert cell.resources.water == pytest.approx(5.0 - RAZIONE_ACQUA)
    assert agent.steps_without_food == 0


def test_senza_giacenza_il_contatore_di_privazione_riparte():
    """La proprieta' che mancava: la struttura ESISTE ma non ha nulla da dare.

    Prima il contatore si azzerava al solo esistere di un effetto alimentare
    positivo, e una serra sopravvissuta nutriva chiunque per sempre. E' la
    ragione misurata per cui distruggere l'80% delle strutture non produceva
    ne' morti ne' sofferenza.
    """
    world, cell = _mondo(food=0.0, water=0.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    # Inventario vuoto: altrimenti la razione di scorta personale
    # (`consume_food_ration`) azzererebbe il contatore, ed e' giusto che lo
    # faccia — qui si verifica il caso in cui non c'e' nessuna delle due.
    agent = _agente(inventory=ResourceBundle())
    cell.agents_present = [agent.agent_id]

    tick_agent_vitals(agent, cell, 7.0)

    assert agent.steps_without_food == 1
    assert agent.steps_without_water == 1


def test_una_sola_razione_non_ne_sfama_due():
    """Il prelievo e' sequenziale: due coloni non attingono alla stessa unita'."""
    world, cell = _mondo(food=RAZIONE_CIBO, water=10.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    primo = BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1,
                      inventory=ResourceBundle())
    secondo = BaseAgent(agent_id="a1", name="a1", role="colonist", x=1, y=1,
                        inventory=ResourceBundle())
    cell.agents_present = [primo.agent_id, secondo.agent_id]

    tick_agent_vitals(primo, cell, 7.0)
    tick_agent_vitals(secondo, cell, 7.0)

    # Il magazzino ne aveva UNA: il primo la riceve, il secondo no. Prima della
    # correzione la stessa unita' li sfamava entrambi — e anche cento.
    assert primo.steps_without_food == 0
    assert secondo.steps_without_food == 1
    assert cell.resources.food == pytest.approx(0.0)


def test_il_kernel_preleva_come_il_motore_a_oggetti():
    """Stessa razione, stesso ordine, stesso residuo: e' la parita' che conta.

    Il rischio di una trascrizione vettoriale e' proprio questo: una lettura
    unica della giacenza servirebbe la stessa unita' a tutti gli occupanti
    della cella e fabbricherebbe massa dal nulla.
    """
    world, cell = _mondo(food=RAZIONE_CIBO, water=10.0, oxygen=10.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    agenti = {
        aid: BaseAgent(agent_id=aid, name=aid, role="colonist", x=1, y=1,
                       inventory=ResourceBundle())
        for aid in ("a0", "a1")
    }
    cell.agents_present = list(agenti)

    aa = AgentArrays.from_agents(agenti)
    ca = CellArrays.from_world(world)
    ca.occupancy[1, 1] = 2

    tick_vitals(
        aa, ca, 7.0, False, 0.0,
        np.zeros(aa.n, dtype=bool), np.zeros(aa.n, dtype=bool),
    )
    for a in agenti.values():
        tick_agent_vitals(a, cell, 7.0)

    assert ca.cell_res[1, 1, C.R["food"]] == pytest.approx(cell.resources.food)
    for aid, a in agenti.items():
        riga = aa.index[aid]
        assert int(aa.steps_without_food[riga]) == int(a.steps_without_food)


# ---------------------------------------------------------------------------
# Il carico elettrico
# ---------------------------------------------------------------------------

def test_gli_impianti_consumano_energia():
    """Il carico era DICHIARATO in `power_margin` e non lo pagava nessuno."""
    world, cell = _mondo(energy=10.0)
    world.add_structure(Structure(type=StructureType.OXYGEN_PLANT, x=1, y=1, integrity=1.0))
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))

    apply_structure_effects(world)

    atteso_carico = (
        CARICO_ELETTRICO[StructureType.OXYGEN_PLANT]
        + CARICO_ELETTRICO[StructureType.GREENHOUSE]
    ) * RESA_ENERGIA_PER_CARICO
    # La cella ha anche prodotto energia? Nessun pannello qui: solo consumo.
    assert cell.resources.energy == pytest.approx(10.0 - atteso_carico)


def test_senza_corrente_un_impianto_non_produce():
    """La copertura scala la produzione: e' l'anello che rende utili i pannelli."""
    world, cell = _mondo(energy=0.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    world.add_structure(Structure(type=StructureType.OXYGEN_PLANT, x=1, y=1, integrity=1.0))

    apply_structure_effects(world)
    apply_colony_resource_feedback(world, {}, 7.0)

    assert cell.resources.oxygen == pytest.approx(0.0)
    assert cell.resources.food == pytest.approx(0.0)
    assert cell.resources.water == pytest.approx(0.0)


def test_con_corrente_lo_stesso_impianto_produce():
    """Il controllo che rende il test precedente una misura e non un caso."""
    world, cell = _mondo(energy=10.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    world.add_structure(Structure(type=StructureType.OXYGEN_PLANT, x=1, y=1, integrity=1.0))

    apply_structure_effects(world)
    apply_colony_resource_feedback(world, {}, 7.0)

    assert cell.resources.oxygen > 0.0
    assert cell.resources.food > 0.0
    assert cell.resources.water > 0.0


def test_le_infermerie_producono_kit_medici():
    """Prima nessuno li produceva: l'unica sorgente erano i nuovi nati."""
    from src.simulation.step_effects import RESA_KIT_MEDICI

    world, cell = _mondo(energy=10.0, med_kits=0.0)
    world.add_structure(Structure(type=StructureType.INFIRMARY, x=1, y=1, integrity=1.0))

    apply_structure_effects(world)
    apply_colony_resource_feedback(world, {}, 7.0)

    # L'effetto curativo dichiarato dall'infermeria e' 0,55.
    assert cell.resources.med_kits == pytest.approx(0.55 * RESA_KIT_MEDICI)


def test_senza_infermeria_i_kit_non_compaiono():
    world, cell = _mondo(energy=10.0, med_kits=0.0)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))

    apply_structure_effects(world)
    apply_colony_resource_feedback(world, {}, 7.0)

    assert cell.resources.med_kits == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# I termini ambientali lenti, tarati su ancore fisiche
# ---------------------------------------------------------------------------

def test_la_radiazione_consuma_la_salute_in_quattro_anni_marziani():
    """L'ancora e' la fisica, non il modello.

    La superficie marziana consegna ~230 mSv/anno contro un limite di carriera
    dell'ordine del sievert: quattro anni marziani di esposizione piena. Prima
    era `0,014 x dt` con `dt = dt_days/3650`, cioe' 0,003 di salute in cento
    passi — il pericolo che definisce Marte era assente dal modello.
    """
    from src.agents.vitals import PASSI_ANNO_MARZIANO, USURA_RADIAZIONE_PER_PASSO

    world, cell = _mondo()
    cell.radiation_level = 1.0
    agent = _agente(inventory=ResourceBundle(water=50.0, food=50.0, oxygen=50.0))
    cell.agents_present = [agent.agent_id]
    salute_iniziale = agent.health

    passi = 20
    for _ in range(passi):
        tick_agent_vitals(agent, cell, 7.0)

    # Nessuna schermatura (nessuna struttura) e nessun'altra pressione: la
    # perdita deve essere quella dichiarata dalla costante.
    persa = salute_iniziale - agent.health
    assert persa == pytest.approx(USURA_RADIAZIONE_PER_PASSO * passi, rel=0.05)
    # ...e la costante deve valere quattro anni marziani.
    assert 1.0 / USURA_RADIAZIONE_PER_PASSO == pytest.approx(
        4.0 * PASSI_ANNO_MARZIANO, rel=1e-9
    )


def test_la_schermatura_riduce_la_radiazione_come_dichiarato():
    """Habitat e rifugi valgono qualcosa: e' il senso della taratura."""
    world, cell = _mondo()
    cell.radiation_level = 1.0
    nudo = _agente(inventory=ResourceBundle(water=50.0, food=50.0, oxygen=50.0))
    cell.agents_present = [nudo.agent_id]
    for _ in range(10):
        tick_agent_vitals(nudo, cell, 7.0)
    persa_nudo = 1.0 - nudo.health

    world2, cell2 = _mondo(water=50.0, food=50.0, oxygen=50.0)
    cell2.radiation_level = 1.0
    for _ in range(3):
        world2.add_structure(Structure(type=StructureType.HABITAT, x=1, y=1, integrity=1.0))
    riparato = _agente(inventory=ResourceBundle(water=50.0, food=50.0, oxygen=50.0))
    cell2.agents_present = [riparato.agent_id]
    for _ in range(10):
        tick_agent_vitals(riparato, cell2, 7.0)

    assert (1.0 - riparato.health) < persa_nudo


def test_il_freddo_polare_uccide_alla_finestra_dichiarata():
    """L'esposizione polare e' ZERO alle celle della colonia (misurato).

    Questo termine governa quindi le sole spedizioni, e senza un test non
    avrebbe alcuna verifica: le run non lo attraversano mai.
    """
    from src.agents.vitals import PASSI_LETALI_ESPOSIZIONE_POLARE

    world, cell = _mondo()
    cell.polar_severity = 1.0
    cell.radiation_level = 0.0
    agent = _agente(inventory=ResourceBundle(water=50.0, food=50.0, oxygen=50.0))
    cell.agents_present = [agent.agent_id]

    for _ in range(int(PASSI_LETALI_ESPOSIZIONE_POLARE) - 1):
        tick_agent_vitals(agent, cell, 7.0)
        assert agent.health > 0.0

    tick_agent_vitals(agent, cell, 7.0)
    assert agent.health == pytest.approx(0.0, abs=1e-9)
