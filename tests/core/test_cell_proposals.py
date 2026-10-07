import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionType
from src.agents.preference_agent import decide_preferences
from src.agents.population import spawn_initial_agents
from src.core import constants as C
from src.core.arrays import CellArrays
from src.core.cell_action_mask import compute_cell_masks
from src.core.cell_proposals import (
    ALWAYS_CELL_ACTIONS,
    CELL_ONLY_ACTIONS,
    PERSONAL_ACTIONS,
    TASK_ACTIONS,
    compute_cell_proposals,
)
from src.core.needs import RedistributionConfig, compute_needs
import pytest
from src.world.resources import ResourceBundle
from src.world.terrain import TerrainType
from src.world.structures import BUILD_COSTS, Structure, StructureType
from src.world.world_generator import WorldGenerator


def _proposal_fixture():
    world = WorldGenerator(seed=71).generate(32, 20)
    # Terreno edificabile per costruzione (2026-09-01): su montagna e canyon
    # il modello vieta ora i cantieri, e una prova sul menu di costruzione
    # non deve dipendere da quale terreno il seme abbia messo li'.
    world.get_cell(8, 7).terrain = TerrainType.REGOLITH_PLAIN
    cells = CellArrays.from_world(world)
    x, y = 8, 7
    cells.occupancy[y, x] = 12
    # Mark the cell as settled while leaving the key infrastructure absent.
    cells.struct_count[y, x, C.S[StructureType.SHELTER]] = 1
    cells.struct_integrity[y, x, C.S[StructureType.SHELTER]] = 1.0
    cells.water_ice[y, x] = 20.0
    cells.cell_res[y, x, C.R["minerals"]] = 20.0
    cells.cell_res[y, x, C.R["construction_material"]] = 20.0
    cells.vegetation[y, x] = 1.0
    cells.proto_soil[y, x] = 1.0
    needs = compute_needs(cells, RedistributionConfig())
    proposals = compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=5
    )
    return cells, x, y, needs, proposals


def test_cell_publishes_bounded_menu_and_keeps_personal_safety_actions():
    _cells, x, y, _needs, proposals = _proposal_fixture()
    row = proposals.mask[y, x]

    assert all(row[pillars.ACTION_INDEX[action]] for action in PERSONAL_ACTIONS)
    assert all(row[pillars.ACTION_INDEX[action]] for action in ALWAYS_CELL_ACTIONS)
    assert not any(row[pillars.ACTION_INDEX[action]] for action in CELL_ONLY_ACTIONS)
    assert sum(row[pillars.ACTION_INDEX[action]] for action in TASK_ACTIONS) == 5
    assert not row[pillars.ACTION_INDEX[ActionType.COMMUNICATE]]
    assert not row[pillars.ACTION_INDEX[ActionType.SHARE_RESOURCE]]


def test_cell_only_mode_replaces_personal_actions_with_recovery_everywhere():
    world = WorldGenerator(seed=75).generate(32, 20)
    cells = CellArrays.from_world(world)
    needs = compute_needs(cells, RedistributionConfig())
    proposals = compute_cell_proposals(
        cells,
        compute_cell_masks(cells),
        needs,
        top_k=3,
        individual_survival_priority_enabled=False,
    )
    recovery = pillars.ACTION_INDEX[ActionType.PHYSIOLOGICAL_RECOVERY]

    assert np.all(proposals.mask[:, :, recovery])
    assert np.all(proposals.priority[:, :, recovery] == 1.0)
    assert np.all(proposals.quota[:, :, recovery] == -1)
    for action in PERSONAL_ACTIONS:
        action_i = pillars.ACTION_INDEX[action]
        assert not np.any(proposals.mask[:, :, action_i])
        assert not np.any(proposals.priority[:, :, action_i])
        assert not np.any(proposals.quota[:, :, action_i])


def test_cell_priorities_and_quotas_follow_current_needs():
    cells, x, y, needs, proposals = _proposal_fixture()
    assert needs[y, x, C.ND["water"]] > 0.0
    assert needs[y, x, C.ND["build_solar"]] == 2.0
    assert proposals.priority[y, x, pillars.ACTION_INDEX[ActionType.COLLECT_ICE]] > 0.5
    assert proposals.priority[y, x, pillars.ACTION_INDEX[ActionType.BUILD_SOLAR_ARRAY]] == 1.0
    assert 1 <= proposals.quota[y, x, pillars.ACTION_INDEX[ActionType.COLLECT_ICE]] <= cells.occupancy[y, x]
    # **La quota dei cantieri segue il deficit (2026-08-25).** Era la costante 1
    # per ogni tipo, e con due pannelli mancanti la cella poteva aprire un solo
    # cantiere per passo: e' il difetto che rendeva muto il peso del
    # governatore, perche' la priorita' decide CHI costruisce e la quota decide
    # QUANTI. Qui ne mancano due, quindi la quota e' due.
    assert needs[y, x, C.ND["build_solar"]] == 2.0
    assert proposals.quota[y, x, pillars.ACTION_INDEX[ActionType.BUILD_SOLAR_ARRAY]] == 2
    assert (
        proposals.quota[y, x, pillars.ACTION_INDEX[ActionType.BUILD_SOLAR_ARRAY]]
        <= cells.occupancy[y, x]
    )


def test_second_solar_and_oxygen_units_remain_top_priority_until_coverage():
    cells, x, y, _needs, _proposals = _proposal_fixture()
    cells.occupancy[y, x] = 20
    cells.struct_count[y, x, C.S[StructureType.SOLAR_ARRAY]] = 1
    cells.struct_count[y, x, C.S[StructureType.OXYGEN_PLANT]] = 1
    # L'integrita' va impostata insieme al conteggio: `GridWorld.add_structure`
    # crea sempre una struttura intatta, e una con conteggio 1 e integrita' 0
    # e' uno stato che il motore non produce mai. Senza questa riga la cella
    # risulta con un arretrato di manutenzione da parco distrutto, e il freno
    # sui cantieri, che e' proprio la regola qui sotto esaminata, spegne le
    # costruzioni per una ragione che non c'entra con la copertura.
    cells.struct_integrity[y, x, C.S[StructureType.SOLAR_ARRAY]] = 1.0
    cells.struct_integrity[y, x, C.S[StructureType.OXYGEN_PLANT]] = 1.0

    needs = compute_needs(cells, RedistributionConfig())
    proposals = compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=5
    )

    for need_name, action, missing in (
        ("build_solar", ActionType.BUILD_SOLAR_ARRAY, 2.0),
        ("build_oxygen_plant", ActionType.BUILD_OXYGEN_PLANT, 1.0),
    ):
        action_i = pillars.ACTION_INDEX[action]
        assert needs[y, x, C.ND[need_name]] == missing
        assert proposals.mask[y, x, action_i]
        assert proposals.priority[y, x, action_i] == 1.0


def test_collection_quotas_are_capped_by_real_stock_and_yield_slots():
    world = WorldGenerator(seed=74).generate(32, 20)
    # Terreno edificabile per costruzione (2026-09-01): su montagna e canyon
    # il modello vieta ora i cantieri, e una prova sul menu di costruzione
    # non deve dipendere da quale terreno il seme abbia messo li'.
    world.get_cell(8, 7).terrain = TerrainType.REGOLITH_PLAIN
    cells = CellArrays.from_world(world)
    x, y = 8, 7
    cells.occupancy[y, x] = 12
    cells.water_ice[y, x] = 2.1
    # **La giacenza va azzerata, non solo impostata (2026-09-01).** Da quando
    # ogni cella ha un fondo idrico sotterraneo, `resources.ice` non e' piu'
    # zero di default e la scorta visibile sarebbe 2,1 PIU' il fondo: la prova
    # misurerebbe il generatore invece della regola sulle quote.
    cells.cell_res[y, x, C.R["ice"]] = 0.0
    cells.cell_res[y, x, C.R["construction_material"]] = 0.25
    cells.cell_res[y, x, C.R["minerals"]] = 1.2
    cells.struct_count[y, x, C.S[StructureType.GREENHOUSE]] = 1
    needs = compute_needs(cells, RedistributionConfig())

    proposals = compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=len(TASK_ACTIONS)
    )

    quota = proposals.quota[y, x]
    assert quota[pillars.ACTION_INDEX[ActionType.COLLECT_ICE]] == 2
    assert quota[pillars.ACTION_INDEX[ActionType.COLLECT_MATERIALS]] == 1
    assert quota[pillars.ACTION_INDEX[ActionType.COLLECT_MINERALS]] == 2
    assert quota[pillars.ACTION_INDEX[ActionType.FORAGE]] == 1


def test_mature_greenhouses_publish_one_routine_cultivation_job_each():
    world = WorldGenerator(seed=79).generate(32, 20)
    # Terreno edificabile per costruzione (2026-09-01): su montagna e canyon
    # il modello vieta ora i cantieri, e una prova sul menu di costruzione
    # non deve dipendere da quale terreno il seme abbia messo li'.
    world.get_cell(8, 7).terrain = TerrainType.REGOLITH_PLAIN
    cells = CellArrays.from_world(world)
    x, y = 8, 7
    cells.occupancy[y, x] = 12
    cells.struct_count[y, x, C.S[StructureType.GREENHOUSE]] = 6
    cells.struct_integrity[y, x, C.S[StructureType.GREENHOUSE]] = 1.0
    # Tredici e non dodici: dal 2026-08-30 una cella e' "matura" quando copre i
    # presenti **piu' la riserva di crescita** (un posto ogni cinquanta coloni,
    # ricavata dalla regola di natalita'). A copertura esatta la cella ha ancora
    # un alloggio da costruire, e questo test smetterebbe di parlare di serre
    # mature per parlare di un cantiere aperto.
    cells.struct_count[y, x, C.S[StructureType.SHELTER]] = 13
    cells.struct_count[y, x, C.S[StructureType.SOLAR_ARRAY]] = 2
    cells.struct_count[y, x, C.S[StructureType.OXYGEN_PLANT]] = 2
    # **E i pozzi (2026-09-01).** L'acqua e' entrata fra i requisiti di
    # capienza: senza, questa cella non e' matura ma un cantiere, il menu
    # pubblica `build_water_extractor` a priorita' piena e la raccolta di
    # routine esce dal taglio `top_k`. Il difetto sarebbe della fixture, non
    # del menu.
    # Ample stock means food need is zero: the jobs represent operation of
    # productive infrastructure, not an emergency response.
    cells.struct_count[y, x, C.S[StructureType.WATER_EXTRACTOR]] = 2
    cells.cell_res[y, x, C.R["food"]] = 1_000.0
    # **E l'acqua (2026-09-01).** La cella e' «matura e rifornita»: con la sola
    # dispensa piena ma il magazzino idrico a zero, `collect_ice` -- ora
    # ammissibile ovunque, perche' il ghiaccio e' sotto ogni cella -- sale a
    # priorita' piena e occupa il quinto posto del menu. Sarebbe il menu a fare
    # la cosa giusta e la fixture a descrivere una cella assetata.
    cells.cell_res[y, x, C.R["water"]] = 1_000.0
    needs = compute_needs(cells, RedistributionConfig())

    proposals = compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=5
    )
    forage = pillars.ACTION_INDEX[ActionType.FORAGE]

    assert needs[y, x, C.ND["food"]] == 0.0
    assert proposals.mask[y, x, forage]
    assert proposals.priority[y, x, forage] == 0.22
    assert proposals.quota[y, x, forage] == 6


def test_shared_exploration_frontier_wraps_across_longitude():
    world = WorldGenerator(seed=77).generate(12, 8)
    cells = CellArrays.from_world(world)
    cells.explored[:, :] = True
    cells.explored[4, 0] = False
    needs = np.zeros((cells.H, cells.W, len(C.ND)), dtype=np.float64)

    proposals = compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=len(TASK_ACTIONS)
    )
    explore = pillars.ACTION_INDEX[ActionType.EXPLORE]

    assert proposals.priority[4, cells.W - 1, explore] > proposals.priority[4, 6, explore]


def test_occupied_virgin_cell_keeps_homestead_greenhouse_in_top_k():
    world = WorldGenerator(seed=78).generate(12, 8)
    # Terreno edificabile per costruzione (2026-09-01): su montagna e canyon
    # il modello vieta ora i cantieri, e una prova sul menu di costruzione
    # non deve dipendere da quale terreno il seme abbia messo li'.
    world.get_cell(6, 4).terrain = TerrainType.REGOLITH_PLAIN
    cells = CellArrays.from_world(world)
    x, y = 6, 4
    cells.occupancy[y, x] = 1
    needs = compute_needs(cells, RedistributionConfig())

    # Virgin cells are deliberately outside the ordinary needs matrix. The
    # homestead bridge must nevertheless survive the same top-k=5 used by GUI
    # and CLI runs, while an empty virgin cell remains untouched.
    assert not np.any(needs[y, x])
    proposals = compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=5
    )
    greenhouse = pillars.ACTION_INDEX[ActionType.BUILD_GREENHOUSE]

    assert proposals.mask[y, x, greenhouse]
    assert proposals.priority[y, x, greenhouse] == 1.0
    assert proposals.quota[y, x, greenhouse] == 1
    assert not proposals.mask[0, 0, greenhouse]


def test_open_construction_site_outranks_fresh_jobs_in_bounded_menu():
    cells, x, y, needs, _proposals = _proposal_fixture()
    greenhouse_structure = C.S[StructureType.GREENHOUSE]
    greenhouse_action = pillars.ACTION_INDEX[ActionType.BUILD_GREENHOUSE]
    cells.site_progress[y, x, greenhouse_structure] = 50.0

    proposals = compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=1
    )

    assert proposals.mask[y, x, greenhouse_action]
    assert proposals.priority[y, x, greenhouse_action] == 1.1
    assert sum(
        proposals.mask[y, x, pillars.ACTION_INDEX[action]]
        for action in TASK_ACTIONS
    ) == 1


def test_agent_binder_uses_cell_priority_instead_of_rigid_action_order():
    world = WorldGenerator(seed=72).generate(32, 20)
    agents = spawn_initial_agents(1, world, seed=72)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.resources.construction_material = 10.0
    cell.resources.minerals = 10.0
    agent.pillar_preferences = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    for action, value in (
        (ActionType.COLLECT_MATERIALS, 0.2),
        (ActionType.COLLECT_MINERALS, 0.9),
    ):
        mask[pillars.ACTION_INDEX[action]] = True
        priority[pillars.ACTION_INDEX[action]] = value

    request = decide_preferences(
        agent,
        world,
        mask,
        0.0,
        "greedy",
        {},
        1,
        action_priority_row=priority,
    )
    assert request.action == ActionType.COLLECT_MINERALS


def test_cell_only_decision_keeps_preferences_but_ignores_critical_needs():
    world = WorldGenerator(seed=76).generate(32, 20)
    agent = next(iter(spawn_initial_agents(1, world, seed=76).values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.resources.minerals = 10.0
    agent.hydration = 0.05
    agent.health = 0.05
    agent.pillar_preferences = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    mask[pillars.ACTION_INDEX[ActionType.PHYSIOLOGICAL_RECOVERY]] = True
    mask[pillars.ACTION_INDEX[ActionType.COLLECT_MINERALS]] = True
    priority[pillars.ACTION_INDEX[ActionType.PHYSIOLOGICAL_RECOVERY]] = 0.2
    priority[pillars.ACTION_INDEX[ActionType.COLLECT_MINERALS]] = 0.9

    request = decide_preferences(
        agent,
        world,
        mask,
        0.0,
        "greedy",
        {},
        1,
        action_priority_row=priority,
        individual_survival_priority_enabled=False,
    )

    assert request.action == ActionType.PHYSIOLOGICAL_RECOVERY


def test_cell_quota_prevents_duplicate_seven_day_job():
    world = WorldGenerator(seed=73).generate(32, 20)
    agents = spawn_initial_agents(1, world, seed=73)
    agent = next(iter(agents.values()))
    world.get_cell(agent.x, agent.y).resources.construction_material = 10.0
    agent.pillar_preferences = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    quota = np.zeros(pillars.N_ACTIONS, dtype=np.int32)
    action_i = pillars.ACTION_INDEX[ActionType.COLLECT_MATERIALS]
    mask[action_i] = True
    priority[action_i] = 1.0
    quota[action_i] = 1
    claims = {}

    first = decide_preferences(
        agent, world, mask, 0.0, "greedy", claims, 1,
        action_priority_row=priority, quota_row=quota,
    )
    second = decide_preferences(
        agent, world, mask, 0.0, "greedy", claims, 1,
        action_priority_row=priority, quota_row=quota,
    )
    assert first.action == ActionType.COLLECT_MATERIALS
    assert second.action != ActionType.COLLECT_MATERIALS


def test_cross_action_claims_do_not_double_spend_cell_materials():
    world = WorldGenerator(seed=80).generate(32, 20)
    collector, builder = list(spawn_initial_agents(2, world, seed=80).values())
    cell = world.get_cell(collector.x, collector.y)
    cost = BUILD_COSTS[StructureType.SOLAR_ARRAY]
    for resource in cost.__dataclass_fields__:
        setattr(cell.resources, resource, float(getattr(cost, resource)))
        setattr(collector.inventory, resource, 0.0)
        setattr(builder.inventory, resource, 0.0)
    collector.pillar_preferences = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    builder.pillar_preferences = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    quota = np.zeros(pillars.N_ACTIONS, dtype=np.int32)
    material_i = pillars.ACTION_INDEX[ActionType.COLLECT_MATERIALS]
    build_i = pillars.ACTION_INDEX[ActionType.BUILD_SOLAR_ARRAY]
    fallback_i = pillars.ACTION_INDEX[ActionType.DO_NOTHING]
    mask[[material_i, build_i, fallback_i]] = True
    priority[[material_i, build_i]] = 1.0
    quota[material_i] = 1
    quota[build_i] = 1
    quota[fallback_i] = -1
    claims = {}

    first = decide_preferences(
        collector, world, mask, 0.0, "greedy", claims, 1,
        action_priority_row=priority, quota_row=quota,
    )
    second = decide_preferences(
        builder, world, mask, 0.0, "greedy", claims, 1,
        action_priority_row=priority, quota_row=quota,
    )

    assert first.action == ActionType.COLLECT_MATERIALS
    assert second.action != ActionType.BUILD_SOLAR_ARRAY
    assert claims[("resource", cell.y, cell.x, "construction_material")] == 1.0

    reverse_claims = {}
    first = decide_preferences(
        builder, world, mask, 0.0, "greedy", reverse_claims, 1,
        action_priority_row=priority, quota_row=quota,
    )
    second = decide_preferences(
        collector, world, mask, 0.0, "greedy", reverse_claims, 1,
        action_priority_row=priority, quota_row=quota,
    )

    assert first.action == ActionType.BUILD_SOLAR_ARRAY
    assert second.action != ActionType.COLLECT_MATERIALS
    assert reverse_claims[("resource", cell.y, cell.x, "construction_material")] == float(
        cost.construction_material
    )


def test_locally_blocked_builder_does_not_consume_shared_build_quota():
    from src.world.structures import Structure, StructureType

    world = WorldGenerator(seed=74).generate(32, 20)
    agents = spawn_initial_agents(2, world, seed=74)
    first, second = list(agents.values())
    cell = world.get_cell(first.x, first.y)
    first.local_x_m, first.local_y_m = 1_000.0, 1_000.0
    second.local_x_m, second.local_y_m = 20_000.0, 20_000.0
    world.add_structure(
        Structure(
            StructureType.GREENHOUSE,
            first.x,
            first.y,
            local_x_m=first.local_x_m,
            local_y_m=first.local_y_m,
            owner=first.agent_id,
        )
    )
    cell.resources.construction_material = 100.0
    cell.resources.minerals = 100.0
    cell.resources.energy = 100.0
    cell.resources.water = 100.0
    for agent in (first, second):
        agent.pillar_preferences = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    quota = np.zeros(pillars.N_ACTIONS, dtype=np.int32)
    action_i = pillars.ACTION_INDEX[ActionType.BUILD_GREENHOUSE]
    mask[action_i] = True
    priority[action_i] = 1.0
    quota[action_i] = 1
    claims = {}

    blocked = decide_preferences(
        first,
        world,
        mask,
        0.0,
        "greedy",
        claims,
        1,
        action_priority_row=priority,
        quota_row=quota,
    )
    eligible = decide_preferences(
        second,
        world,
        mask,
        0.0,
        "greedy",
        claims,
        1,
        action_priority_row=priority,
        quota_row=quota,
    )

    assert blocked.action != ActionType.BUILD_GREENHOUSE
    assert eligible.action == ActionType.BUILD_GREENHOUSE
    assert claims[(cell.y, cell.x, action_i)] == 1


def _cella_con_parco(occupanti: int, integrita: float):
    """Una cella MATURA e usurata: nessun deficit edilizio, parco tutto a `integrita`.

    La maturita' non e' un dettaglio della fixture, e' la condizione in cui il
    difetto viveva: finche' mancano strutture i cantieri riempiono il menu e la
    manutenzione non compete con nessuno. Le sette run del 26-27 agosto avevano
    tutti i deficit a zero dal passo dieci in poi.
    """
    world = WorldGenerator(seed=71).generate(32, 20)
    # Terreno edificabile per costruzione (2026-09-01): su montagna e canyon
    # il modello vieta ora i cantieri, e una prova sul menu di costruzione
    # non deve dipendere da quale terreno il seme abbia messo li'.
    world.get_cell(8, 7).terrain = TerrainType.REGOLITH_PLAIN
    cells = CellArrays.from_world(world)
    x, y = 8, 7
    cells.occupancy[y, x] = occupanti
    for tipo in (
        StructureType.SHELTER,
        StructureType.HABITAT,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        StructureType.INFIRMARY,
        StructureType.RESEARCH_LAB,
        StructureType.STORAGE_DEPOT,
        StructureType.WEATHER_STATION,
        StructureType.HEATER,
    ):
        indice = C.S[tipo]
        cells.struct_count[y, x, indice] = occupanti
        # `struct_integrity` e' la SOMMA sulle istanze del tipo.
        cells.struct_integrity[y, x, indice] = integrita * occupanti
    cells.cell_res[y, x, C.R["construction_material"]] = 5000.0
    needs = compute_needs(cells, RedistributionConfig())
    proposals = compute_cell_proposals(cells, compute_cell_masks(cells), needs, top_k=5)
    return x, y, proposals


def test_maintenance_priority_does_not_depend_on_how_many_people_live_there():
    """Il secondo difetto delle run del 26-27 agosto.

    La priorita' era `bisogno / occupanti`: con duecento coloni valeva 0,09,
    sotto `observe` che non fa nulla e vale 0,20. La sola run sopravvissuta
    riprese a manutenere solo dopo essere crollata a UN abitante, perche' li'
    lo stesso rapporto saturava. Le strutture da tenere in piedi non calano
    quando la colonia cresce: la priorita' non puo' dipendere dalla popolazione.
    """
    indice = pillars.ACTION_INDEX[ActionType.MAINTAIN_STRUCTURE]
    priorita = [
        _cella_con_parco(n, 0.90)[2].priority[7, 8, indice] for n in (1, 12, 200)
    ]
    assert np.allclose(priorita, priorita[0], atol=1e-9), priorita


def test_maintenance_outranks_the_no_op_while_the_cell_is_still_self_sufficient():
    """La manutenzione deve entrare nel menu PRIMA del punto di non ritorno.

    La resa di una struttura e' la sua integrita', e l'autosufficienza misurata
    si perdeva gia' a 0,82. Se a quel punto la manutenzione vale meno di
    `observe` (0,20), il taglio top-k la esclude e la cella si spegne lentamente
    senza che nessuno se ne accorga: e' esattamente cio' che e' successo.
    """
    indice = pillars.ACTION_INDEX[ActionType.MAINTAIN_STRUCTURE]
    osservare = pillars.ACTION_INDEX[ActionType.OBSERVE]
    _, _, proposals = _cella_con_parco(200, 0.90)
    assert proposals.priority[7, 8, indice] > proposals.priority[7, 8, osservare]

    # A impianti nuovi resta invece l'ultima delle priorita': non c'e' nulla da fare.
    _, _, nuove = _cella_con_parco(200, 1.0)
    assert nuove.priority[7, 8, indice] < nuove.priority[7, 8, osservare]


def test_maintenance_quota_counts_maintainers_and_caps_on_occupancy():
    """Una manutenzione vale il lavoro di un colono, non il rifacimento del parco.

    Prima il costo era l'arretrato INTERO della cella pagato da una persona
    sola: 13,7 unita' di materiale contro le 1,9 che un colono porta addosso,
    quindi l'azione veniva proposta e respinta a ogni passo. Ora l'arretrato si
    smaltisce con piu' manutentori, ed e' la quota a dire quanti ne servono.
    """
    indice = pillars.ACTION_INDEX[ActionType.MAINTAIN_STRUCTURE]
    # 200 occupanti x 10 tipi = 2000 strutture; arretrato = min(0,4, 1-i) x 2000,
    # diviso la capienza di un colono (4,0 unita' di integrita' per passo).
    assert _cella_con_parco(200, 0.90)[2].quota[7, 8, indice] == 50   # 200 / 4
    assert _cella_con_parco(200, 0.70)[2].quota[7, 8, indice] == 151  # 600 / 4, ceil
    # L'arretrato satura a 0,4 per struttura e la quota non supera gli abitanti.
    assert _cella_con_parco(200, 0.50)[2].quota[7, 8, indice] == 200


def test_la_tempesta_scoraggia_il_viaggio_ma_non_se_la_cella_non_nutre():
    """Il livello decisionale non sapeva che ci fosse una tempesta.

    Fino al 2026-08-31 `cell_proposals`, `cell_action_mask`, `preference_agent`
    e `needs` non avevano un solo riferimento agli eventi estremi: una colonia
    sotto una tempesta di polvere globale componeva lo stesso identico menu di
    una giornata serena. Gli eventi agivano sulla sola fisica della cella.

    La severita' si ricava dall'eccesso di polvere o radiazione sopra la linea
    di base, normalizzato per l'ampiezza che il motore degli eventi applica.
    Si scoraggia il VIAGGIO e non il lavoro, perche' in questo modello un
    colono non e' dentro o fuori rispetto alla propria cella. E il fattore si
    annulla dove la cella non puo' erogare ne' cibo ne' acqua: li' restare
    fermi non e' prudenza ma condanna.
    """
    from src.agents.action_space import ActionType
    from src.simulation.extreme_events import DUST_STORM_OPACITY_SURGE

    def priorita(polvere: float, cibo: float) -> dict[str, float]:
        world = WorldGenerator(seed=404).generate(4, 4)
        cell = world.get_cell(1, 1)
        cell.structures.clear()
        # Il pozzo c'e' (2026-09-01): senza, la cella e' un cantiere, il menu
        # pubblica `build_water_extractor` a priorita' piena e `explore` esce
        # dal taglio `top_k`. La prova parlerebbe di acqua invece che di
        # tempesta.
        for tipo in ([StructureType.SHELTER] * 6 + [StructureType.GREENHOUSE] * 2
                     + [StructureType.WATER_EXTRACTOR]):
            world.add_structure(Structure(type=tipo, x=1, y=1, integrity=0.9))
        cell.resources = ResourceBundle(construction_material=50.0, food=cibo, water=cibo)
        cell.baseline_dust_level = 0.10
        cell.dust_level = polvere
        cell.baseline_radiation_level = 0.60
        cell.radiation_level = 0.60
        cell.agents_present = [f"a{i}" for i in range(6)]
        ca = CellArrays.from_world(world)
        proposte = compute_cell_proposals(
            ca, compute_cell_masks(ca), compute_needs(ca, RedistributionConfig()), top_k=5
        )
        return {
            azione.value: float(proposte.priority[1, 1, pillars.ACTION_INDEX[azione]])
            for azione in (ActionType.MOVE, ActionType.EXPLORE, ActionType.OBSERVE)
        }

    tempesta = 0.10 + DUST_STORM_OPACITY_SURGE
    sereno_pieno = priorita(0.10, 30.0)
    tempesta_pieno = priorita(tempesta, 30.0)
    tempesta_vuoto = priorita(tempesta, 0.0)

    assert sereno_pieno["move"] > 0.0 and sereno_pieno["explore"] > 0.0
    assert tempesta_pieno["move"] == 0.0, "con la tempesta e la dispensa piena non si viaggia"
    assert tempesta_pieno["explore"] == 0.0
    assert tempesta_vuoto["move"] == pytest.approx(sereno_pieno["move"]), (
        "se la cella non nutre, il viaggio torna possibile: restare e' condanna"
    )
    # `explore` non si asserisce a dispensa vuota: e' un'azione di lavoro e
    # soggiace al taglio top-k, che con la fame in corso la fa uscire dal menu
    # per ragioni che non hanno a che vedere con la tempesta. `move` invece e'
    # sempre proposta, quindi e' su di essa che il tilt si legge pulito.
    # `observe` non e' toccata: restare fermi e' cio' che si deve fare.
    assert tempesta_pieno["observe"] == pytest.approx(sereno_pieno["observe"])


def test_una_linea_di_base_assente_non_e_una_tempesta():
    """Senza storia, il valore corrente E' la linea di base.

    Trattarla come zero farebbe apparire come tempesta permanente la polvere
    ordinaria di qualunque cella priva di linea di base registrata.
    """
    from src.agents.action_space import ActionType

    world = WorldGenerator(seed=405).generate(4, 4)
    cell = world.get_cell(1, 1)
    cell.structures.clear()
    world.add_structure(Structure(type=StructureType.SHELTER, x=1, y=1, integrity=0.9))
    cell.resources = ResourceBundle(construction_material=10.0, food=30.0, water=30.0)
    cell.dust_level = 0.90
    cell.baseline_dust_level = None
    cell.radiation_level = 0.90
    cell.baseline_radiation_level = None
    cell.agents_present = ["a0"]
    ca = CellArrays.from_world(world)
    proposte = compute_cell_proposals(
        ca, compute_cell_masks(ca), compute_needs(ca, RedistributionConfig()), top_k=5
    )
    assert float(proposte.priority[1, 1, pillars.ACTION_INDEX[ActionType.MOVE]]) > 0.0
