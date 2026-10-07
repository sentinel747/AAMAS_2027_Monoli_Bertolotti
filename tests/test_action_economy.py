"""Economia delle azioni (audit run 120726_testlogic5_1000steps).

Nella run da 1000 step il 66% delle azioni era drink_water: l'executor
consumava la borraccia personale 0.1 alla volta e attingeva alla struttura
(con ricarica +1.0) solo a inventario vuoto, quindi il rifornimento pre
spedizione richiedeva ~50 azioni invece di 2-3. Alla colonia si beve dal
"pozzo" (struttura con supporto idrico) prima che dalla borraccia.

Il 24% era communicate, gonfiato anche dal need-signal invertito
`inventory.food >= 3.0` (cibo abbondante spingeva a comunicare).
"""

import pytest
from types import SimpleNamespace

from src.agents.action_space import ActionRequest, ActionType, execute_action
from src.agents.population import spawn_initial_agents
from src.world.structures import BUILD_COSTS, Structure, StructureType
from src.world.world_generator import WorldGenerator


def _world_with_agent(seed=9, with_greenhouse=False):
    world = WorldGenerator(seed=seed).generate(8, 8)
    agents = spawn_initial_agents(1, world, seed=seed)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    # Nessuna fonte libera nella cella: il test controlla solo borraccia/pozzo.
    cell.liquid_water = 0.0
    cell.water_ice = 0.0
    cell.resources.ice = 0.0
    if with_greenhouse:
        world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))
        # **La serra ha un magazzino, e va riempito (2026-08-25).** Il
        # rifornimento da impianto prelevava da una sorgente infinita: bastava
        # la struttura perche' la borraccia si riempisse, e la massa non si
        # conservava. Ora preleva dal magazzino della cella, che nel mondo vero
        # l'impianto alimenta passo dopo passo. Questi test misurano l'economia
        # dell'AZIONE — quante ne servono per raggiungere il target — non la
        # provenienza dell'acqua, quindi il magazzino si riempie qui una volta
        # con quanto basta a non essere il vincolo.
        world.get_cell(agent.x, agent.y).resources.water = 100.0
    return world, agents, agent


def test_drink_from_structure_hydrates_without_changing_canteen():
    world, agents, agent = _world_with_agent(with_greenhouse=True)
    agent.inventory.water = 5.0
    agent.hydration = 0.5

    result = execute_action(agent, agents, world, ActionRequest(agent.agent_id, ActionType.DRINK_WATER))

    assert result.accepted
    assert agent.inventory.water == 5.0
    assert agent.hydration > 0.5


def test_drink_away_from_structures_still_drains_canteen():
    world, agents, agent = _world_with_agent(with_greenhouse=False)
    agent.inventory.water = 5.0
    agent.inventory.ice = 0.0
    agent.hydration = 0.5

    result = execute_action(agent, agents, world, ActionRequest(agent.agent_id, ActionType.DRINK_WATER))

    assert result.accepted
    assert abs(agent.inventory.water - 4.9) < 1e-9


def test_scout_refill_reaches_target_in_two_actions():
    world, agents, agent = _world_with_agent(with_greenhouse=True)
    agent.inventory.water = 5.0
    agent.inventory.ice = 0.0

    for _ in range(2):
        execute_action(agent, agents, world, ActionRequest(agent.agent_id, ActionType.REFILL_WATER))

    assert agent.inventory.water + agent.inventory.ice >= 7.0  # SCOUT_TARGET_WATER_RESERVE


def test_expedition_refill_target_fills_canteen_in_one_weekly_action():
    world, agents, agent = _world_with_agent(with_greenhouse=True)
    agent.inventory.water = 4.5
    agent.inventory.ice = 0.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(
            agent.agent_id,
            ActionType.REFILL_WATER,
            target={"reserve_target": 8.0},
        ),
    )

    assert result.accepted
    assert agent.inventory.water == 8.0
    assert result.data["amount"] == 3.5


def test_expedition_food_target_collects_missing_rations_in_one_action():
    world, agents, agent = _world_with_agent(with_greenhouse=True)
    agent.inventory.food = 2.0
    # Il raccolto della serra sta nella giacenza della cella (2026-08-25): il
    # test verifica che UNA azione basti a coprire il bersaglio, non che il
    # cibo si materializzi.
    world.get_cell(agent.x, agent.y).resources.food = 10.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(
            agent.agent_id,
            ActionType.FORAGE,
            target={"food_target": 4.0},
        ),
    )

    assert result.accepted
    assert agent.inventory.food == 4.0
    assert result.data["amount"] == 2.0


def test_refill_from_local_ice_increases_reserve_without_draining_canteen():
    world, agents, agent = _world_with_agent(with_greenhouse=False)
    cell = world.get_cell(agent.x, agent.y)
    cell.water_ice = 2.0
    agent.inventory.water = 0.5
    agent.inventory.ice = 0.0
    agent.hydration = 1.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.REFILL_WATER),
    )

    assert result.accepted
    assert agent.inventory.water == 0.5
    assert agent.inventory.ice == 1.0
    assert cell.water_ice == 1.0
    assert agent.hydration == 1.0


def _observation(nearby):
    return SimpleNamespace(nearby_agents=nearby, nearby_agents_details=[], local_danger=0.0, local_habitability=0.0, agent_count=len(nearby) + 1)


def test_abundant_food_is_not_a_communication_need_signal():
    world, agents, agent = _world_with_agent(with_greenhouse=True)
    agent.cooperation = 0.9
    agent.actions_taken = 7  # cadenza non allineata
    agent.recent_actions = ["move", "move", "move"]
    agent.inventory.food = 10.0

    assert agent._should_coordinate(_observation(["other"])) is False

    agent.inventory.food = 1.0  # cibo davvero scarso: segnale legittimo
    assert agent._should_coordinate(_observation(["other"])) is True


def _costo_manutenzione(strutture_danneggiate: int, mancante: float) -> float:
    """Il costo atteso, ricavato dalle costanti invece che scritto a mano."""
    from src.simulation.step_effects import (
        COSTO_MANUTENZIONE_PER_INTEGRITA,
        RIPARAZIONE_MANUTENZIONE,
    )

    ripristino = strutture_danneggiate * min(RIPARAZIONE_MANUTENZIONE, mancante)
    return ripristino * COSTO_MANUTENZIONE_PER_INTEGRITA


def test_maintenance_draws_missing_material_from_cell_warehouse():
    world, agents, agent = _world_with_agent()
    cell = world.get_cell(agent.x, agent.y)
    # Dal 2026-08-25 la manutenzione ripara solo cio' che e' danneggiato: una
    # struttura nuova non ne ha bisogno, e l'azione viene respinta.
    world.add_structure(Structure(StructureType.SHELTER, agent.x, agent.y, integrity=0.5))
    costo = _costo_manutenzione(1, 0.5)
    agent.inventory.construction_material = costo / 4.0
    cell.resources.construction_material = costo - costo / 4.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.MAINTAIN_STRUCTURE),
    )

    assert result.accepted
    assert agent.inventory.construction_material == pytest.approx(0.0)
    assert cell.resources.construction_material == pytest.approx(0.0)


def test_maintenance_charges_personal_material_only_once():
    world, agents, agent = _world_with_agent()
    world.add_structure(Structure(StructureType.SHELTER, agent.x, agent.y, integrity=0.5))
    agent.inventory.construction_material = 2.0
    costo = _costo_manutenzione(1, 0.5)

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.MAINTAIN_STRUCTURE),
    )

    assert result.accepted
    assert agent.inventory.construction_material == pytest.approx(2.0 - costo)


def test_maintenance_su_struttura_intatta_non_costa_nulla():
    """Il difetto piu' caro non era il prezzo ma il lavoro inutile.

    Con l'usura ferma a 2e-5 per passo la manutenzione riparava un degrado che
    non avveniva, e costava un quarto di tutto il materiale della colonia.
    """
    world, agents, agent = _world_with_agent()
    world.add_structure(Structure(StructureType.SHELTER, agent.x, agent.y, integrity=1.0))
    agent.inventory.construction_material = 5.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.MAINTAIN_STRUCTURE),
    )

    assert not result.accepted
    assert agent.inventory.construction_material == 5.0


def test_il_costo_della_manutenzione_cresce_col_parco_riparato():
    """Prima un'unita' fissa comprava ventiquattro unita' di integrita'."""
    world, agents, agent = _world_with_agent()
    for _ in range(4):
        world.add_structure(
            Structure(StructureType.SHELTER, agent.x, agent.y, integrity=0.5)
        )
    agent.inventory.construction_material = 10.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.MAINTAIN_STRUCTURE),
    )

    assert result.accepted
    assert result.data["restored"] == pytest.approx(4 * 0.4)
    assert agent.inventory.construction_material == pytest.approx(
        10.0 - _costo_manutenzione(4, 0.5)
    )


def test_new_build_draws_project_inputs_from_agent_and_cell_warehouse():
    world, agents, agent = _world_with_agent()
    cell = world.get_cell(agent.x, agent.y)
    cost = BUILD_COSTS[StructureType.SOLAR_ARRAY]
    agent.inventory.construction_material = 0.5
    agent.inventory.minerals = 0.5
    cell.resources.construction_material = cost.construction_material - 0.5
    cell.resources.minerals = cost.minerals - 0.5

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.BUILD_SOLAR_ARRAY),
    )

    assert result.accepted
    assert agent.inventory.construction_material == 0.0
    assert agent.inventory.minerals == 0.0
    assert cell.resources.construction_material == 0.0
    assert cell.resources.minerals == 0.0


def test_un_colono_puo_permettersi_la_manutenzione_di_un_parco_grande():
    """Il difetto che ha estinto sei run su sette il 26-27 agosto.

    Dal 2026-08-25 il costo seguiva cio' che si riparava, ma cio' che si
    riparava era il parco INTERO della cella mentre a pagare era un solo
    colono. In una cella matura (235 strutture al 77%) faceva 13,7 unita' di
    materiale contro le 1,9 che un colono porta addosso: l'azione veniva
    proposta e respinta a ogni passo, l'integrita' scendeva, e con essa la
    produzione d'acqua, fino alla morte per sete dell'intera colonia.

    Una manutenzione vale ora il lavoro di un passo-uomo: al piu'
    `CAPACITA_MANUTENZIONE` unita' di integrita', spalmate su tutto il parco.
    """
    world, agents, agent = _world_with_agent()
    for _ in range(20):
        world.add_structure(
            Structure(StructureType.SHELTER, agent.x, agent.y, integrity=0.5)
        )
    # Quanto la redistribuzione tiene nella sacca di ciascuno.
    agent.inventory.construction_material = 2.0
    cell = world.get_cell(agent.x, agent.y)
    cell.resources.construction_material = 0.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.MAINTAIN_STRUCTURE),
    )

    # Arretrato 20 x 0,4 = 8,0, oltre la capienza: se ne ripara 4,0 e si paga 1,0.
    assert result.accepted, result.reason
    assert result.data["restored"] == pytest.approx(4.0)
    assert agent.inventory.construction_material == pytest.approx(1.0)
    # La passata solleva TUTTE le strutture in proporzione, non alcune sole.
    integrita = [s.integrity for s in cell.structures]
    assert integrita == pytest.approx([0.7] * 20)
