from src.agents.vitals import (
    LETHAL_STEPS_WITHOUT_FOOD,
    LETHAL_STEPS_WITHOUT_WATER,
)

#: Il passo reale delle run: settimanale.
PASSO_GIORNI = 7.0
from src.agents.population import spawn_initial_agents
from src.agents.action_space import ActionRequest, ActionType, execute_action
from src.agents.vitals import tick_agent_vitals
from src.agents.vitals import tick_all_agents
from src.simulation.colony_dynamics import (
    compute_colony_dynamics_metrics,
    compute_earth_input_package_metrics,
    compute_inter_colony_cooperation_metrics,
)
from src.social_network.network import SocialNetwork
from src.world.colony_site import colony_site_score
from src.world.resources import ResourceBundle
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def test_v2_default_vitals_do_not_update_psychosocial_state():
    world = WorldGenerator(seed=101).generate(12, 12)
    agents = spawn_initial_agents(1, world, seed=101)
    agent = next(iter(agents.values()))
    agent.oxygen_level = 0.42
    agent.hydration = 0.45
    agent.fatigue = 0.75
    before = agent.stress_index

    tick_all_agents([agent], world, 3650, config={"social": {"earth_mars_delay_minutes": 22}})

    assert agent.stress_index == before
    assert 0.0 <= agent.morale <= 1.0
    assert "stress_index" in agent.to_dict()


def test_legacy_psychosocial_vitals_can_be_enabled_explicitly():
    world = WorldGenerator(seed=101).generate(12, 12)
    agents = spawn_initial_agents(1, world, seed=101)
    agent = next(iter(agents.values()))
    agent.oxygen_level = 0.42
    agent.hydration = 0.45
    agent.fatigue = 0.75
    before = agent.stress_index

    tick_all_agents(
        [agent],
        world,
        3650,
        config={"model": {"psychosocial_enabled": True}, "social": {"earth_mars_delay_minutes": 22}},
    )

    assert agent.stress_index > before
    assert 0.0 <= agent.morale <= 1.0
    assert "stress_index" in agent.to_dict()


def test_il_colono_muore_alla_finestra_dichiarata_senza_bere():
    """La finestra e' letta dalla costante, non scritta nel nome del test.

    Valeva sette passi — 49 giorni a passo settimanale — e ora due, che sono
    quattordici. Il test si ancora a `LETHAL_STEPS_WITHOUT_WATER` cosi' una
    ritaratura futura lo aggiorna invece di romperlo.
    """
    world = WorldGenerator(seed=111).generate(6, 6)
    agents = spawn_initial_agents(1, world, seed=111)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.radiation_level = 0.0
    agent.inventory.water = 0.0
    agent.inventory.ice = 0.0
    agent.recent_actions = [ActionType.OBSERVE.value]

    for _ in range(LETHAL_STEPS_WITHOUT_WATER - 1):
        tick_agent_vitals(agent, cell, PASSO_GIORNI)
        assert agent.health > 0

    tick_agent_vitals(agent, cell, PASSO_GIORNI)

    assert agent.steps_without_water == LETHAL_STEPS_WITHOUT_WATER
    assert agent.health == 0.0


def test_il_colono_muore_alla_finestra_dichiarata_senza_mangiare():
    world = WorldGenerator(seed=112).generate(6, 6)
    agents = spawn_initial_agents(1, world, seed=112)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.radiation_level = 0.0
    agent.inventory.food = 0.0
    # **Acqua garantita dall'impianto**, non solo dalla borraccia: l'orologio
    # della sete e' piu' corto e vincerebbe la gara della causa di morte. Si usa
    # un'infermeria perche' e' l'unica struttura con effetto idrico e senza
    # effetto alimentare — una serra riforniva anche di cibo e annullava il test.
    world.add_structure(Structure(StructureType.INFIRMARY, agent.x, agent.y))
    cell = world.get_cell(agent.x, agent.y)
    cell.resources.water = 100.0
    cell.resources.oxygen = 100.0
    agent.inventory.water = 50.0
    agent.recent_actions = [ActionType.DRINK_WATER.value]

    for _ in range(LETHAL_STEPS_WITHOUT_FOOD - 1):
        tick_agent_vitals(agent, cell, PASSO_GIORNI)
        assert agent.health > 0

    tick_agent_vitals(agent, cell, PASSO_GIORNI)

    assert agent.steps_without_food == LETHAL_STEPS_WITHOUT_FOOD
    assert agent.health == 0.0


def test_agent_can_drink_from_inventory_ice():
    world = WorldGenerator(seed=113).generate(6, 6)
    agents = spawn_initial_agents(1, world, seed=113)
    agent = next(iter(agents.values()))
    agent.hydration = 0.35
    agent.inventory.water = 0.0
    agent.inventory.ice = 1.0

    result = execute_action(agent, agents, world, ActionRequest(agent.agent_id, ActionType.DRINK_WATER))

    assert result.accepted
    assert agent.inventory.ice < 1.0
    assert agent.hydration > 0.35


def test_le_scorte_iniziali_coprono_la_dotazione_dichiarata():
    """Quanti passi copre la dotazione personale, prima che si produca nulla.

    La dotazione e' 4,0 d'acqua e 5,0 di cibo, e una razione ne consuma 0,1:
    quaranta e cinquanta passi. Il test ne verifica trenta, che restano dentro
    entrambi. Girava su `dt=3650` — un passo da dieci anni — perche' prima i
    vitali non si muovevano; ora il passo e' quello vero.
    """
    world = WorldGenerator(seed=114).generate(6, 6)
    agents = spawn_initial_agents(1, world, seed=114)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.radiation_level = 0.0
    # **La cella di questo seme e' polare** (severita' 0,75, modificatore
    # termico -24,7): dal 2026-08-25 il freddo uccide in una decina di passi a
    # esposizione piena, e il colono moriva assiderato al diciottesimo prima di
    # arrivare a esaurire le scorte. Finche' il termine polare era congelato la
    # cosa non si vedeva. Qui si azzera, perche' il soggetto del test sono le
    # scorte.
    cell.polar_severity = 0.0
    cell.local_temperature_modifier = 0.0
    # **L'ossigeno viene dall'impianto, non dalla bombola.** La dotazione
    # personale e' 2,0 unita', cioe' venti razioni: su una cella spoglia il
    # colono muore d'ipossia al ventesimo passo, ed e' corretto. Il soggetto di
    # questo test sono pero' le scorte di CIBO e ACQUA, e serve un impianto per
    # isolarle.
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, agent.x, agent.y))
    cell.resources.oxygen = 100.0
    agent.recent_actions = [ActionType.OBSERVE.value]

    for _ in range(30):
        tick_agent_vitals(agent, cell, PASSO_GIORNI)

    assert agent.health > 0.0
    assert agent.steps_without_water == 0
    assert agent.steps_without_food == 0
    assert agent.inventory.water < 4.0
    assert agent.inventory.food < 5.0


def test_fractional_last_water_ration_counts_despite_float_rounding():
    world = WorldGenerator(seed=115).generate(6, 6)
    agents = spawn_initial_agents(1, world, seed=115)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.radiation_level = 0.0
    agent.inventory.water = 0.09999999999999759
    agent.inventory.ice = 0.0
    agent.recent_actions = [ActionType.OBSERVE.value]

    tick_agent_vitals(agent, cell, 3650)

    assert agent.steps_without_water == 0
    assert agent.inventory.water == 0.0


def test_colony_dynamics_metrics_reward_life_support_and_governance():
    world = WorldGenerator(seed=102).generate(12, 12)
    agents = spawn_initial_agents(4, world, seed=102)
    social = SocialNetwork(set(agents.keys()))
    ids = list(agents)
    social.update(ids[0], ids[1], "communicate")
    social.update(ids[1], ids[2], "cooperate")
    social.update(ids[2], ids[3], "share_resource")
    x, y = world.width // 2, world.height // 2
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, x, y))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))
    world.add_structure(Structure(StructureType.GREENHOUSE, x, y))
    world.add_structure(Structure(StructureType.HABITAT, x, y))

    metrics = compute_colony_dynamics_metrics(agents, world, {"social": {"earth_mars_delay_minutes": 12}}, social)

    assert metrics["life_support_reliability"] > 0.4
    assert metrics["colony_area_m2"] == metrics["colony_built_area_m2"]
    assert 0 < metrics["colony_built_area_m2"] < metrics["colony_claimed_area_m2"]
    assert metrics["mission_operational_readiness"] > 0.2
    assert metrics["cohesion_index"] > 0.0
    assert metrics["earth_mars_delay_minutes"] == 12
    # Espansione in celle: una cella con strutture, piu' quelle dove stanno i
    # coloni (spawn casuale: almeno la cella delle strutture, al massimo 1 + 4).
    assert metrics["structure_cells"] == 1.0
    assert 1.0 <= metrics["occupied_cells"] <= 5.0
    assert metrics["occupied_cells"] == len({(a.x, a.y) for a in agents.values()} | {(x, y)})


def test_automatic_logistics_contributes_to_canonical_cooperation_metric():
    world = WorldGenerator(seed=121).generate(12, 12)
    agents = spawn_initial_agents(4, world, seed=121)
    world.metadata.update(
        {
            "_automatic_logistics_enabled": True,
            "_automatic_logistics_deposited_total": 20.0,
            "_automatic_logistics_withdrawn_total": 8.0,
            "_automatic_logistics_flow_total": 4.0,
            "_automatic_logistics_flow_events": 3,
            "_automatic_logistics_steps": 2,
            "_automatic_logistics_active_steps": 2,
            "_automatic_logistics_agent_steps": 8,
        }
    )

    metrics = compute_colony_dynamics_metrics(
        agents, world, {}, SocialNetwork(set(agents))
    )

    assert metrics["automatic_resource_deposited_total"] == 20.0
    assert metrics["automatic_resource_withdrawn_total"] == 8.0
    assert metrics["automatic_intercell_flow_total"] == 4.0
    assert metrics["automatic_intercell_flow_events"] == 3.0
    assert metrics["logistics_cooperation_index"] == 0.75
    assert metrics["cooperation_index"] == 0.75


def test_vector_world_metrics_use_structure_columns_and_invalidate_position_cache(
    monkeypatch,
):
    from src.core.shell_common import build_core_state
    from src.core.views import CellView

    world = WorldGenerator(seed=120).generate(12, 12)
    agents = spawn_initial_agents(3, world, seed=120)
    social = SocialNetwork(set(agents))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, 6, 6, integrity=0.8))
    world.add_structure(Structure(StructureType.GREENHOUSE, 6, 6, integrity=0.6))
    expected = compute_colony_dynamics_metrics(agents, world, {}, social)
    core, view = build_core_state(world, agents)
    assert core.cells.structure_positions() == ((6, 6),)

    def forbidden_structures(_self):
        raise AssertionError("vector metrics must not materialize StructureView lists")

    monkeypatch.setattr(CellView, "structures", property(forbidden_structures))
    actual = compute_colony_dynamics_metrics(agents, view, {}, social)

    assert actual["colony_built_area_m2"] == expected["colony_built_area_m2"]
    assert actual["power_margin"] == expected["power_margin"]
    assert actual["food_margin"] == expected["food_margin"]

    view.add_structure(Structure(StructureType.HABITAT, 7, 6))
    assert set(core.cells.structure_positions()) == {(6, 6), (7, 6)}


def test_cell_density_pollution_is_area_scaled_not_a_hard_capacity_limit():
    world = WorldGenerator(seed=103).generate(12, 12)
    cell = world.get_cell(6, 6)
    for index in range(40):
        cell.agents_present.append(f"a{index}")
    for _ in range(40):
        world.add_structure(Structure(StructureType.STORAGE_DEPOT, 6, 6))

    cell.recompute_habitability(world.planetary_state)

    assert 0.0 < cell.pollution_risk < 0.08
    assert len(cell.structures) == 40
    assert len(cell.agents_present) == 40


def test_colony_metrics_report_capacity_based_overcrowding():
    world = WorldGenerator(seed=122).generate(12, 12)
    agents = spawn_initial_agents(12, world, seed=122, start_x=6, start_y=6)
    social = SocialNetwork(set(agents))

    unsupported = compute_colony_dynamics_metrics(agents, world, {}, social)
    assert unsupported["max_cell_occupancy"] == 12.0
    assert unsupported["max_cell_occupancy_capacity"] == 10.0
    assert unsupported["max_cell_overcrowding_excess"] == 2.0
    assert unsupported["overcrowded_cell_count"] == 1.0

    for _ in range(12):
        world.add_structure(Structure(StructureType.SHELTER, 6, 6))
    supported = compute_colony_dynamics_metrics(agents, world, {}, social)
    assert supported["max_cell_occupancy_capacity"] == 12.0
    assert supported["max_cell_overcrowding_excess"] == 0.0
    assert supported["overcrowded_cell_count"] == 0.0


def test_colony_site_score_rewards_resources_and_safety():
    world = WorldGenerator(seed=104).generate(12, 12)
    poor = world.get_cell(1, 1)
    rich = world.get_cell(6, 6)
    poor.water_ice = 0.0
    poor.resources = ResourceBundle()
    poor.radiation_level = 1.4
    poor.dust_level = 0.8
    rich.water_ice = 8.0
    rich.resources = ResourceBundle(ice=4.0, minerals=8.0, construction_material=6.0, energy=1.0)
    rich.radiation_level = 0.25
    rich.dust_level = 0.05
    poor.recompute_habitability(world.planetary_state)
    rich.recompute_habitability(world.planetary_state)

    assert colony_site_score(rich) > colony_site_score(poor)


def test_colony_dynamics_exposes_site_selection_metrics():
    world = WorldGenerator(seed=105).generate(12, 12)
    agents = spawn_initial_agents(2, world, seed=105)
    social = SocialNetwork(set(agents.keys()))
    best = world.get_cell(2, 2)
    best.water_ice = 9.0
    best.resources = ResourceBundle(ice=5.0, minerals=9.0, construction_material=7.0, energy=1.0)
    best.radiation_level = 0.2
    best.dust_level = 0.03
    best.recompute_habitability(world.planetary_state)

    metrics = compute_colony_dynamics_metrics(agents, world, {}, social)

    assert 0.0 <= metrics["colony_site_score"] <= 1.0
    assert metrics["best_colony_site_score"] >= metrics["colony_site_score"]
    assert metrics["best_colony_site_x"] == 2.0
    assert metrics["best_colony_site_y"] == 2.0
    assert metrics["colony_site_regret"] >= 0.0


def test_earth_input_package_metrics_reward_initial_support():
    poor = {
        "agents": {"initial_inventory": {"food": 1, "water": 1, "construction_material": 0, "tools": 0}},
        "colony": {"initial_structures": {}},
    }
    rich = {
        "agents": {
            "initial_inventory": {
                "food": 8,
                "water": 8,
                "construction_material": 8,
                "tools": 3,
                "oxygen": 2,
                "energy": 2,
                "minerals": 4,
                "med_kits": 2,
            }
        },
        "colony": {"initial_structures": {"habitat": 2, "greenhouse": 2, "solar_array": 2, "oxygen_plant": 1}},
    }

    poor_metrics = compute_earth_input_package_metrics(poor, population=12)
    rich_metrics = compute_earth_input_package_metrics(rich, population=12)

    assert rich_metrics["earth_input_package_units"] > poor_metrics["earth_input_package_units"]
    assert rich_metrics["earth_input_mass_proxy_kg"] > poor_metrics["earth_input_mass_proxy_kg"]
    assert rich_metrics["earth_input_sufficiency_index"] > poor_metrics["earth_input_sufficiency_index"]
    assert rich_metrics["earth_input_deficit_index"] < poor_metrics["earth_input_deficit_index"]


def test_colony_dynamics_exposes_earth_input_metrics():
    world = WorldGenerator(seed=106).generate(12, 12)
    agents = spawn_initial_agents(3, world, seed=106)
    social = SocialNetwork(set(agents.keys()))
    config = {
        "agents": {"initial_inventory": {"food": 6, "water": 6, "construction_material": 4, "tools": 1}},
        "colony": {"initial_structures": {"habitat": 1, "greenhouse": 1, "solar_array": 1}},
    }

    metrics = compute_colony_dynamics_metrics(agents, world, config, social)

    assert metrics["earth_input_package_units"] > 0.0
    assert metrics["earth_input_mass_proxy_kg"] > 0.0
    assert 0.0 <= metrics["earth_input_sufficiency_index"] <= 1.0
    assert metrics["earth_input_deficit_index"] == 1.0 - metrics["earth_input_sufficiency_index"]


def test_inter_colony_cooperation_metrics_are_neutral_for_single_colony():
    world = WorldGenerator(seed=107).generate(12, 12)
    agents = spawn_initial_agents(3, world, seed=107)
    social = SocialNetwork(set(agents.keys()))

    metrics = compute_inter_colony_cooperation_metrics(agents, social, {})

    assert metrics["inter_colony_count"] == 1.0
    assert metrics["inter_colony_cooperation_need_index"] == 0.0
    assert metrics["inter_colony_cooperation_gap_index"] == 0.0


def test_inter_colony_cooperation_metrics_measure_need_and_delivery():
    world = WorldGenerator(seed=108).generate(12, 12)
    agents = spawn_initial_agents(4, world, seed=108)
    ids = list(agents)
    for agent_id in ids[:2]:
        setattr(agents[agent_id], "colony_id", "alpha")
        agents[agent_id].inventory.food = 10.0
        agents[agent_id].inventory.water = 10.0
    for agent_id in ids[2:]:
        setattr(agents[agent_id], "colony_id", "beta")
        agents[agent_id].inventory.food = 1.0
        agents[agent_id].inventory.water = 1.0
    social = SocialNetwork(set(agents.keys()))

    isolated = compute_inter_colony_cooperation_metrics(agents, social, {"social": {"earth_mars_delay_minutes": 22}})
    social.update(ids[0], ids[2], "communicate")
    social.update(ids[0], ids[2], "share_resource")
    social.update(ids[1], ids[3], "cooperate")
    cooperative = compute_inter_colony_cooperation_metrics(agents, social, {"social": {"earth_mars_delay_minutes": 22}})

    assert isolated["inter_colony_count"] == 2.0
    assert isolated["inter_colony_cooperation_need_index"] > 0.0
    assert cooperative["inter_colony_cooperation_index"] > isolated["inter_colony_cooperation_index"]
    assert cooperative["inter_colony_cooperation_gap_index"] < isolated["inter_colony_cooperation_gap_index"]


def test_le_scorte_pubblicate_contano_anche_i_magazzini():
    """`food_stock` e compagni misurano la colonia, non le sole sacche.

    Il difetto corretto il 2026-08-30: le quattro scorte pubblicate — che
    entrano nei margini, nella prosperita' e nell'indice composito — sommavano
    solo gli inventari dei coloni. Su una colonia reale il magazzino tiene la
    quota maggiore, quindi la metrica ne riportava una frazione.
    """
    world = WorldGenerator(seed=404).generate(12, 12)
    agents = spawn_initial_agents(3, world, seed=404)
    x, y = world.width // 2, world.height // 2
    world.add_structure(Structure(StructureType.STORAGE_DEPOT, x, y))
    deposito = world.get_cell(x, y)
    deposito.resources.food = 90.0
    deposito.resources.water = 70.0
    deposito.resources.construction_material = 50.0
    deposito.resources.tools = 30.0
    for agent in agents.values():
        agent.inventory.food = 1.0
        agent.inventory.water = 2.0
        agent.inventory.construction_material = 3.0
        agent.inventory.tools = 4.0

    metrics = compute_colony_dynamics_metrics(agents, world, {}, SocialNetwork())

    assert metrics["food_stock"] == 90.0 + 3.0
    assert metrics["water_stock"] == 70.0 + 6.0
    assert metrics["material_stock"] == 50.0 + 9.0
    assert metrics["tool_stock"] == 30.0 + 12.0


def test_impronta_della_colonia_unica_fra_metriche_e_natalita():
    """Le metriche e la natalita' devono guardare le stesse celle.

    Due impronte diverse sarebbero la famiglia 10 di questo progetto: due
    formule per la stessa grandezza che smettono di coincidere in silenzio.
    """
    from src.agents.population import colony_cells
    from src.simulation.colony_dynamics import _structure_snapshot

    world = WorldGenerator(seed=405).generate(14, 14)
    for dx in (0, 1, 3):
        world.add_structure(
            Structure(StructureType.SHELTER, world.width // 2 + dx, world.height // 2)
        )

    dalle_metriche = {(cell.x, cell.y) for cell in _structure_snapshot(world)[0]}
    dalla_natalita = {(cell.x, cell.y) for cell in colony_cells(world)}

    assert dalle_metriche == dalla_natalita
    assert len(dalle_metriche) == 3


def test_il_rimpianto_del_sito_e_zero_quando_la_colonia_tiene_la_cella_migliore():
    """`colony_site_regret` confrontava un MASSIMO con una MEDIA.

    `best_colony_site_score` e' il massimo di `colony_site_score` su tutta la
    mappa; `colony_site_score` pubblicato e' la MEDIA sulle celle occupate. La
    differenza fra i due era percio' strutturalmente positiva anche quando la
    colonia occupava la cella migliore del pianeta — misurato sulla run
    `verifica1000_seed9`: il sito migliore era (0,80), cioe' la cella madre, e
    il rimpianto pubblicato valeva 0,319. Ora si confrontano due massimi: la
    miglior cella che la colonia tiene contro la miglior cella che esiste.
    """
    world = WorldGenerator(seed=105).generate(12, 12)
    agents = spawn_initial_agents(2, world, seed=105)
    social = SocialNetwork(set(agents.keys()))
    # Rendo la cella dove i coloni gia' stanno la migliore della mappa.
    primo = next(iter(agents.values()))
    tenuta = world.get_cell(primo.x, primo.y)
    tenuta.water_ice = 14.0
    tenuta.resources = ResourceBundle(ice=8.0, minerals=14.0, construction_material=12.0, energy=2.0)
    tenuta.radiation_level = 0.1
    tenuta.dust_level = 0.0
    tenuta.recompute_habitability(world.planetary_state)

    metrics = compute_colony_dynamics_metrics(agents, world, {}, social)

    assert metrics["best_colony_site_x"] == float(primo.x), "sanita': la cella migliore e' quella occupata"
    assert metrics["best_colony_site_y"] == float(primo.y)
    assert abs(metrics["best_held_site_score"] - metrics["best_colony_site_score"]) < 1e-9
    assert metrics["colony_site_regret"] < 1e-9, (
        f"la colonia tiene la cella migliore ma il rimpianto vale {metrics['colony_site_regret']}"
    )


def test_il_canale_di_cooperazione_non_e_piu_agganciato_alla_redistribuzione():
    """Correzione di un errore introdotto il 2026-08-30, e misurato il 31.

    Il canale di cella era stato agganciato a `logistics_cooperation_index`,
    cioe' alla redistribuzione automatica, mentre nella STESSA campagna la
    redistribuzione veniva portata a spenta di default. Le due modifiche si
    annullavano: `social_cooperation_channel_index` restava piatto a 0,000 per
    tutti i 1000 passi della run `verifica1000_seed9`, esattamente come il
    grafo a coppie che doveva sostituire.

    Il canale che c'e' sempre e' il magazzino condiviso, e questo test lo
    verifica dove morde: redistribuzione spenta, nessun arco sociale, scorte
    reali nel magazzino della cella.
    """
    world = WorldGenerator(seed=77).generate(10, 10)
    agents = spawn_initial_agents(3, world, seed=77)
    social = SocialNetwork(set(agents.keys()))
    assert not getattr(social, "edges", None), "sanita': nessun arco sociale"

    primo = next(iter(agents.values()))
    cella = world.get_cell(primo.x, primo.y)
    world.add_structure(Structure(type=StructureType.STORAGE_DEPOT, x=primo.x, y=primo.y))
    cella.resources = ResourceBundle(
        food=90.0, water=90.0, oxygen=90.0,
        construction_material=90.0, minerals=90.0, energy=90.0,
    )
    for agente in agents.values():
        agente.inventory = ResourceBundle(
            food=10.0, water=10.0, oxygen=10.0,
            construction_material=10.0, minerals=10.0, energy=10.0,
        )

    metrics = compute_colony_dynamics_metrics(agents, world, {}, social)

    # 90 in comune contro 30 nelle sacche = 0,75 per ogni risorsa del ciclo.
    assert abs(metrics["resource_pooling_index"] - 0.75) < 1e-9, (
        f"messa in comune = {metrics['resource_pooling_index']}, attesa 0,75")
    assert metrics["social_cooperation_channel_index"] > 0.0, (
        "il canale di cooperazione e' tornato a zero")
    assert metrics["social_graph_active"] == 0.0, "sanita': il grafo e' inattivo"


def test_la_messa_in_comune_e_zero_se_tutto_sta_nelle_sacche():
    """L'estremo opposto deve dare zero, altrimenti l'indice non discrimina."""
    world = WorldGenerator(seed=78).generate(10, 10)
    agents = spawn_initial_agents(3, world, seed=78)
    social = SocialNetwork(set(agents.keys()))
    primo = next(iter(agents.values()))
    world.add_structure(Structure(type=StructureType.SHELTER, x=primo.x, y=primo.y))
    for _x, _y, cella in [(c.x, c.y, c) for row in world.cells for c in row]:
        cella.resources = ResourceBundle()
    for agente in agents.values():
        agente.inventory = ResourceBundle(food=5.0, water=5.0, oxygen=5.0,
                                          construction_material=5.0, minerals=5.0, energy=5.0)

    metrics = compute_colony_dynamics_metrics(agents, world, {}, social)
    assert metrics["resource_pooling_index"] == 0.0


def test_la_carestia_locale_si_vede_anche_con_la_colonia_piena():
    """161 decessi, 132 per fame, con `life_support_reliability` a 1,0.

    Ogni metrica di supporto vitale misurava la CAPIENZA — quante strutture
    ci sono — e nessuna la SCORTA dove la gente sta. Misurato sulla run
    `verifica3000_seed9`, passo 1400: la cella madre aveva dodici serre,
    tredici occupanti e **zero cibo in magazzino**, mentre la colonia ne
    teneva oltre mille altrove. Con la redistribuzione spenta (il default)
    nulla lo sposta.

    Questo test costruisce esattamente quella configurazione: due celle, una
    ricca e vuota, una con le serre, la gente e la dispensa a zero.
    """
    world = WorldGenerator(seed=131).generate(8, 8)
    agents = spawn_initial_agents(4, world, seed=131)
    social = SocialNetwork(set(agents.keys()))
    primo = next(iter(agents.values()))
    affamata = world.get_cell(primo.x, primo.y)
    for agente in agents.values():
        agente.x, agente.y = primo.x, primo.y
    affamata.agents_present = [a.agent_id for a in agents.values()]
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=primo.x, y=primo.y))
    affamata.resources = ResourceBundle(food=0.0, water=0.0)

    ricca = world.get_cell((primo.x + 3) % world.width, primo.y)
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=ricca.x, y=ricca.y))
    ricca.resources = ResourceBundle(food=5000.0, water=5000.0)

    metrics = compute_colony_dynamics_metrics(agents, world, {}, social)

    assert metrics["food_stock"] > 1000.0, "sanita': la colonia e' ricchissima di cibo"
    assert metrics["local_food_shortfall_index"] == 1.0, (
        "tutti i coloni stanno in una cella senza dispensa e la metrica non lo dice: "
        f"{metrics['local_food_shortfall_index']}")
    assert metrics["local_water_shortfall_index"] == 1.0
    assert metrics["underfed_population_now"] == 4.0


def test_nessuna_carestia_locale_quando_la_dispensa_e_piena():
    """L'estremo opposto, altrimenti l'indice non discrimina."""
    world = WorldGenerator(seed=132).generate(8, 8)
    agents = spawn_initial_agents(4, world, seed=132)
    social = SocialNetwork(set(agents.keys()))
    primo = next(iter(agents.values()))
    cella = world.get_cell(primo.x, primo.y)
    for agente in agents.values():
        agente.x, agente.y = primo.x, primo.y
    cella.agents_present = [a.agent_id for a in agents.values()]
    world.add_structure(Structure(type=StructureType.GREENHOUSE, x=primo.x, y=primo.y))
    world.add_structure(Structure(type=StructureType.HABITAT, x=primo.x, y=primo.y))
    cella.resources = ResourceBundle(food=500.0, water=500.0)

    metrics = compute_colony_dynamics_metrics(agents, world, {}, social)
    assert metrics["local_food_shortfall_index"] == 0.0
    assert metrics["local_water_shortfall_index"] == 0.0
    assert metrics["underfed_population_now"] == 0.0


def test_il_sito_automatico_cambia_con_il_seme():
    """La regola precedente restituiva sempre la stessa cella.

    Misurato su sei semi e sei profili di mappa, trentasei combinazioni: **y
    valeva 80 in tutte** e x valeva 0 in ventinove, perche' la scansione
    partiva dal bordo della fascia e si fermava al primo `regolith_plain`, che
    copre circa meta' di ogni riga. Il sito, che la tesi dichiara controllo
    sperimentale esplicito, era una costante: stessa cella, zero minerali, su
    ogni run.
    """
    from src.world.colony_site import FASCIA_EQUATORIALE_RIGHE, colony_site_score, find_safe_start

    siti = {}
    for seme in (0, 9, 21, 33):
        world = WorldGenerator(seed=seme).generate(120, 60)
        siti[seme] = find_safe_start(world)

    assert len(set(siti.values())) > 1, f"il sito non cambia con il seme: {siti}"

    # E resta dentro la fascia equatoriale, che e' il vincolo di missione.
    world = WorldGenerator(seed=9).generate(120, 60)
    x, y = find_safe_start(world)
    centro = world.height // 2
    assert abs(y - centro) <= FASCIA_EQUATORIALE_RIGHE

    # Ed e' davvero il massimo del punteggio dentro la fascia.
    migliore = max(
        (
            colony_site_score(world.get_cell(xx, yy))
            for yy in range(centro - FASCIA_EQUATORIALE_RIGHE, centro + FASCIA_EQUATORIALE_RIGHE + 1)
            for xx in range(world.width)
        )
    )
    assert abs(colony_site_score(world.get_cell(x, y)) - migliore) < 1e-12


def test_il_sito_automatico_e_riproducibile():
    """Stesso seme, stessa cella: e' un controllo sperimentale, non un sorteggio."""
    from src.world.colony_site import find_safe_start

    a = find_safe_start(WorldGenerator(seed=42).generate(120, 60))
    b = find_safe_start(WorldGenerator(seed=42).generate(120, 60))
    assert a == b


def _colonia_con_habitat(integrita: float = 1.0):
    """Una colonia con impianti d'ossigeno, habitat e infermeria."""
    world = WorldGenerator(seed=404).generate(12, 12)
    agents = spawn_initial_agents(6, world, seed=404)
    x, y = 5, 5
    for tipo, quante in ((StructureType.OXYGEN_PLANT, 4),
                         (StructureType.HABITAT, 8),
                         (StructureType.INFIRMARY, 2),
                         (StructureType.GREENHOUSE, 4),
                         (StructureType.SOLAR_ARRAY, 6)):
        for _ in range(quante):
            world.add_structure(Structure(tipo, x, y, integrity=integrita))
    return agents, world, SocialNetwork(set(agents))


def test_interruttore_capienze_spento_e_il_calcolo_di_archivio():
    """Spento, deve dare esattamente i numeri con cui l'archivio e' stato misurato.

    Il termine `habitat x 0,35` dentro l'ossigeno non esiste nella fisica (0,35 e'
    l'effetto CIBO dell'habitat) ma e' il calcolo con cui sono state misurate
    tutte le campagne: toglierlo senza interruttore renderebbe i numeri nuovi non
    confrontabili con quelli vecchi.
    """
    agents, world, social = _colonia_con_habitat()
    senza = compute_colony_dynamics_metrics(agents, world, {}, social)
    spento = compute_colony_dynamics_metrics(
        agents, world, {"engine_fixes": {"capacity": False}}, social
    )
    assert senza["eclss_margin"] == spento["eclss_margin"]
    assert senza["power_margin"] == spento["power_margin"]


def test_interruttore_capienze_acceso_toglie_l_ossigeno_fantasma():
    """Acceso, l'ossigeno cala: gli habitat non ne producono."""
    agents, world, social = _colonia_con_habitat()
    spento = compute_colony_dynamics_metrics(agents, world, {}, social)
    acceso = compute_colony_dynamics_metrics(
        agents, world, {"engine_fixes": {"capacity": True}}, social
    )
    assert acceso["eclss_margin"] <= spento["eclss_margin"]


def test_interruttore_capienze_acceso_vede_l_integrita():
    """Acceso, un parco di strutture consumate non dichiara la capacita' del nuovo."""
    _, world_nuovo, _ = _colonia_con_habitat(integrita=1.0)
    agents, world_vecchio, social = _colonia_con_habitat(integrita=0.4)
    agents_nuovo = spawn_initial_agents(6, world_nuovo, seed=404)
    social_nuovo = SocialNetwork(set(agents_nuovo))

    cfg = {"engine_fixes": {"capacity": True}}
    nuovo = compute_colony_dynamics_metrics(agents_nuovo, world_nuovo, cfg, social_nuovo)
    vecchio = compute_colony_dynamics_metrics(agents, world_vecchio, cfg, social)
    assert vecchio["power_margin"] < nuovo["power_margin"]

    spento = {"engine_fixes": {"capacity": False}}
    nuovo_spento = compute_colony_dynamics_metrics(
        agents_nuovo, world_nuovo, spento, social_nuovo)
    vecchio_spento = compute_colony_dynamics_metrics(
        agents, world_vecchio, spento, social)
    assert vecchio_spento["power_margin"] == nuovo_spento["power_margin"], (
        "a interruttore spento l'integrita' non deve contare: e' il difetto")
