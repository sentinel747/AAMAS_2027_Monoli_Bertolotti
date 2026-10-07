"""Espansione multi-cella: spedizioni di fondazione (audit run testlogic5).

Nella run da 1000 step il 97% degli agenti viveva ancora nella cella di
partenza dopo 19 anni: nessun meccanismo portava la colonia a insediarsi
altrove. Quando la cella madre ha TUTTA la copertura vitale soddisfatta ed e'
affollata, i coloni curiosi e ben riforniti partono verso una cella vergine
vicina e fondano un avamposto costruendo per prima la serra (acqua+cibo).
"""

from src.agents.action_space import ActionType, execute_action
from src.agents.build_policy import local_life_support_capacity
from src.agents.population import spawn_initial_agents
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _saturated_colony(seed=21, colonists=20, greenhouses=3, oxygen_plants=2, habitats=10):
    world = WorldGenerator(seed=seed).generate(8, 8)
    world.step = 80
    agents = spawn_initial_agents(1, world, seed=seed)
    agent = next(iter(agents.values()))
    x, y = agent.x, agent.y
    cell = world.get_cell(x, y)
    cell.agents_present.extend(f"ghost_{i:02d}" for i in range(colonists - 1))
    for _ in range(greenhouses):
        world.add_structure(Structure(StructureType.GREENHOUSE, x, y))
    for _ in range(oxygen_plants):
        world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))
        # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
        world.add_structure(Structure(StructureType.WATER_EXTRACTOR, x, y))
    for _ in range(habitats):
        world.add_structure(Structure(StructureType.HABITAT, x, y))
    # Vicini attraversabili e non polari, cosi' un bersaglio esiste sempre.
    for neighbor in world.neighbors(x, y, 1):
        neighbor.terrain = cell.terrain
        neighbor.polar_severity = 0.0
    _provision(agent)
    agent.curiosity = 0.8
    agent.settle_next_at = 0
    agent.actions_taken = 100
    return world, agent, cell


def _provision(agent):
    # Dal 2026-08-30 il kit riserva anche l'acqua della griglia di avviamento
    # (una unita', quella della serra) oltre a quella da viaggio: senza,
    # l'avamposto nasce senza supporto vitale. Vedi `_founder_kit_targets`.
    # **Le quantita' si leggono dal kit (2026-09-01).** L'elenco delle
    # strutture di avviamento e' cresciuto — il pozzo idrico — e un fondatore
    # equipaggiato a mano con i vecchi totali arriva sul posto senza di che
    # finire la griglia: la prova parlerebbe di scorte invece che della
    # macchina a stati.
    for risorsa, quanto in agent._founder_kit_targets().items():
        setattr(agent.inventory, risorsa, quanto)
    agent.inventory.ice = 0.0
    agent.inventory.energy = max(2.0, float(agent.inventory.energy))
    agent.health = 1.0
    agent.fatigue = 0.0
    agent.oxygen_level = 1.0
    agent.hydration = 1.0


def test_mature_saturated_colony_launches_settlement_expedition():
    world, agent, cell = _saturated_colony()

    request = agent._settlement_action(world, cell)

    assert request is not None
    assert request.action in {ActionType.EXPLORE, ActionType.MOVE}
    assert agent.settle_phase == "out"
    tx, ty = agent.settle_target
    assert (tx, ty) != (agent.x, agent.y)
    assert not world.get_cell(tx, ty).structures  # terreno vergine


def test_founder_kit_cannot_withdraw_resources_reserved_by_cell_actions():
    world = WorldGenerator(seed=22).generate(8, 8)
    agent = next(iter(spawn_initial_agents(1, world, seed=22).values()))
    cell = world.get_cell(agent.x, agent.y)
    targets = agent._founder_kit_targets()
    for resource, target in targets.items():
        setattr(agent.inventory, resource, 0.0)
        setattr(cell.resources, resource, float(target))
    claims = {
        ("resource", cell.y, cell.x, "construction_material"): 1.0,
    }

    assert not agent._equip_founder_kit_from_warehouse(cell, claims)
    assert agent.inventory.construction_material == 0.0
    assert cell.resources.construction_material == targets["construction_material"]

    cell.resources.construction_material += 1.0
    assert agent._equip_founder_kit_from_warehouse(cell, claims)
    assert agent.inventory.construction_material == targets["construction_material"]
    assert cell.resources.construction_material == 1.0


def test_settler_founds_greenhouse_on_arrival():
    world, agent, cell = _saturated_colony()
    agent._settlement_action(world, cell)
    tx, ty = agent.settle_target
    world.move_agent(agent.agent_id, agent.x, agent.y, tx, ty, agent.local_x_m, agent.local_y_m)
    agent.x, agent.y = tx, ty
    target = world.get_cell(tx, ty)
    personal_before = {
        resource: float(getattr(agent.inventory, resource))
        for resource in ("construction_material", "minerals", "energy")
    }
    warehouse_before = {
        resource: float(getattr(target.resources, resource))
        for resource in personal_before
    }

    request = agent._settlement_action(world, target)

    assert request is not None
    assert request.action == ActionType.BUILD_GREENHOUSE
    assert agent.settle_phase == "out"
    assert {
        resource: float(getattr(agent.inventory, resource))
        for resource in personal_before
    } == personal_before
    assert {
        resource: float(getattr(target.resources, resource))
        for resource in warehouse_before
    } == warehouse_before


def test_founder_stays_until_complete_bootstrap_grid_is_operational():
    world, agent, cell = _saturated_colony()
    agent._settlement_action(world, cell)
    tx, ty = agent.settle_target
    world.move_agent(
        agent.agent_id,
        agent.x,
        agent.y,
        tx,
        ty,
        agent.local_x_m,
        agent.local_y_m,
    )
    agent.x, agent.y = tx, ty
    agents = {agent.agent_id: agent}

    # Cinque strutture invece di quattro: il ciclo deve avere corda a
    # sufficienza perche' la griglia si completi.
    for _ in range(30):
        request = agent._settlement_action(world, world.get_cell(tx, ty))
        assert request is not None
        result = execute_action(agent, agents, world, request)
        assert result.accepted
        # Isolate the state-machine contract from long-horizon fatigue, which
        # has its own tests and can insert REST between construction weeks.
        agent.fatigue = 0.0
        agent.health = 1.0
        if agent.settle_phase is None:
            break

    target = world.get_cell(tx, ty)
    assert agent.settle_phase is None
    assert local_life_support_capacity(target) >= 1
    assert {structure.type for structure in target.structures}.issuperset(
        {
            StructureType.GREENHOUSE,
            StructureType.SHELTER,
            StructureType.SOLAR_ARRAY,
            StructureType.OXYGEN_PLANT,
        }
    )


def test_exact_founder_kit_covers_shelter_before_oxygen_plant():
    """The exact reserved kit must survive the real bootstrap build order.

    The preference run builds greenhouse, shelter and solar before O2.  The
    old kit omitted the shelter cost, so its construction consumed the oxygen
    inputs and stranded the pioneer indefinitely.
    """
    world, agent, cell = _saturated_colony()
    targets = agent._founder_kit_targets()
    for resource, target in targets.items():
        setattr(agent.inventory, resource, float(target))
    agent._settlement_action(world, cell)
    tx, ty = agent.settle_target
    world.move_agent(
        agent.agent_id,
        agent.x,
        agent.y,
        tx,
        ty,
        agent.local_x_m,
        agent.local_y_m,
    )
    agent.x, agent.y = tx, ty
    agents = {agent.agent_id: agent}

    # Cinque strutture invece di quattro: il ciclo deve avere corda a
    # sufficienza perche' la griglia si completi.
    for _ in range(30):
        request = agent._settlement_action(world, world.get_cell(tx, ty))
        assert request is not None
        result = execute_action(agent, agents, world, request)
        assert result.accepted
        agent.fatigue = 0.0
        agent.health = 1.0
        if agent.settle_phase is None:
            break

    target = world.get_cell(tx, ty)
    assert agent.settle_phase is None
    assert local_life_support_capacity(target) >= 1
    assert StructureType.OXYGEN_PLANT in {s.type for s in target.structures}


def test_founder_commitment_outlives_decision_timeout_at_bootstrap_site():
    """Preference interleaving must not orphan an operational build site."""
    world, agent, cell = _saturated_colony()
    agent._settlement_action(world, cell)
    tx, ty = agent.settle_target
    world.move_agent(
        agent.agent_id,
        agent.x,
        agent.y,
        tx,
        ty,
        agent.local_x_m,
        agent.local_y_m,
    )
    agent.x, agent.y = tx, ty
    # The old branch ended the mission on the next EXPLORE selection even
    # though the founder had arrived with the complete reserved kit.
    agent.settle_steps = 40
    agents = {agent.agent_id: agent}

    # Cinque strutture invece di quattro: il ciclo deve avere corda a
    # sufficienza perche' la griglia si completi.
    for _ in range(30):
        request = agent._settlement_action(world, world.get_cell(tx, ty))
        assert request is not None
        result = execute_action(agent, agents, world, request)
        assert result.accepted
        agent.fatigue = 0.0
        agent.health = 1.0
        if agent.settle_phase is None:
            break

    target = world.get_cell(tx, ty)
    assert agent.settle_phase is None
    assert agent.founder_kit_reserved is False
    assert local_life_support_capacity(target) >= 1
    assert {structure.type for structure in target.structures}.issuperset(
        {
            StructureType.GREENHOUSE,
            StructureType.SHELTER,
            StructureType.SOLAR_ARRAY,
            StructureType.OXYGEN_PLANT,
        }
    )


def test_founder_refills_from_completed_greenhouse_without_ending_mission():
    world, agent, cell = _saturated_colony()
    agent._settlement_action(world, cell)
    tx, ty = agent.settle_target
    world.move_agent(
        agent.agent_id,
        agent.x,
        agent.y,
        tx,
        ty,
        agent.local_x_m,
        agent.local_y_m,
    )
    agent.x, agent.y = tx, ty
    target = world.get_cell(tx, ty)
    world.add_structure(Structure(StructureType.GREENHOUSE, tx, ty))
    agent.inventory.water = 0.5
    agent.inventory.ice = 0.0
    agent.settle_steps = 80

    request = agent._settlement_action(world, target)

    assert request is not None
    assert request.action == ActionType.REFILL_WATER
    assert request.target == {"reserve_target": 7.0}
    assert agent.settle_phase == "out"
    assert agent.founder_kit_reserved is True


def test_underserved_colony_keeps_building_instead_of_settling():
    # Copertura serre sotto fabbisogno: prima si consolida, poi si espande.
    world, agent, cell = _saturated_colony(greenhouses=1)

    request = agent._settlement_action(world, cell)

    assert request is None
    assert agent.settle_phase is None


def test_expedition_aborts_when_provisions_run_out():
    world, agent, cell = _saturated_colony()
    agent._settlement_action(world, cell)
    # Sotto il margine di rientro (attraversare una cella costa ~1.2 acqua di
    # razioni): la missione abortisce PRIMA che il ritorno diventi impossibile.
    agent.inventory.water = 3.4

    request = agent._settlement_action(world, world.get_cell(agent.x, agent.y))

    assert request is None
    assert agent.settle_phase is None  # la sicurezza idrica ordinaria lo riporta a casa


def test_expedition_mode_suppresses_security_builds_en_route():
    world, agent, cell = _saturated_colony()
    agent._settlement_action(world, cell)
    assert agent.settle_phase == "out"
    agent.inventory.food = 3.0  # sotto la soglia che prima faceva aprire serre in viaggio

    wilderness = world.get_cell(*agent.settle_target)
    build = agent._food_security_action(world, wilderness, None)

    assert build is None or build.action not in {ActionType.BUILD_GREENHOUSE, ActionType.BUILD_HABITAT}
