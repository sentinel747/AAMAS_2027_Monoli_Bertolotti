import numpy as np
import pytest

from src.agents import pillars
from src.agents.action_space import ActionType
from src.agents.preference_agent import (
    CRITICAL_HYDRATION,
    agent_action_mask,
    compute_urgencies,
    pillar_availability,
)
from src.agents.rule_based_agent import RuleBasedAgent
from src.world.resources import ResourceBundle


def _agent(**kwargs) -> RuleBasedAgent:
    return RuleBasedAgent(
        agent_id="a0", name="A", role="colonist", x=0, y=0, **kwargs
    )


def _open_mask() -> np.ndarray:
    return np.ones(pillars.N_ACTIONS, dtype=np.bool_)


@pytest.fixture
def world_small():
    from src.agents.population import spawn_initial_agents
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=3).generate(32, 20)
    agents = spawn_initial_agents(1, world, seed=3)
    return world, next(iter(agents.values()))


def test_dehydration_dominates_urgencies():
    agent = _agent(hydration=0.1, satiety=1.0)
    urgencies = compute_urgencies(agent, _open_mask())
    assert urgencies[pillars.P_SUSTENANCE] == max(urgencies)
    assert 0.0 <= urgencies.min() and urgencies.max() <= 1.0


def test_mission_raises_explore_urgency():
    calm = compute_urgencies(_agent(), _open_mask())
    scouting = _agent()
    scouting.scout_phase = "out"
    active = compute_urgencies(scouting, _open_mask())
    assert active[pillars.P_EXPLORE] > calm[pillars.P_EXPLORE]


def test_healthy_life_preference_falls_through_instead_of_wasting_week_on_rest(world_small):
    from src.agents.preference_agent import decide_preferences

    world, agent = world_small
    agent.health = 1.0
    agent.fatigue = 0.0
    agent.stress_index = 0.0
    agent.pillar_preferences = (0.0, 0.0, 0.0, 0.9, 0.0, 0.1)
    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    mask[pillars.ACTION_INDEX[ActionType.REST]] = True
    mask[pillars.ACTION_INDEX[ActionType.EXPLORE]] = True
    mask[pillars.ACTION_INDEX[ActionType.OBSERVE]] = True
    priority = mask.astype(np.float64)

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

    assert request.action == ActionType.OBSERVE


def test_physiological_guard_restricts_to_survival():
    agent = _agent(hydration=CRITICAL_HYDRATION - 0.05)
    mask = agent_action_mask(agent, _open_mask())
    allowed = {
        action
        for action in pillars.ACTION_ORDER
        if mask[pillars.ACTION_INDEX[action]]
    }
    assert allowed <= {
        ActionType.DRINK_WATER,
        ActionType.REFILL_WATER,
        ActionType.EAT_FOOD,
        ActionType.COLLECT_ICE,
        ActionType.FORAGE,
        ActionType.USE_MED_KIT,
        ActionType.REST,
        ActionType.OBSERVE,
        ActionType.DO_NOTHING,
        ActionType.MOVE,
    }
    assert not mask[pillars.ACTION_INDEX[ActionType.BUILD_GREENHOUSE]]


def test_agent_mask_mirrors_personal_conditions():
    poor = _agent(
        inventory=ResourceBundle(
            energy=0,
            oxygen=0,
            water=0,
            food=0,
            construction_material=0,
            minerals=0,
            med_kits=0,
        )
    )
    mask = agent_action_mask(poor, _open_mask())
    assert not mask[pillars.ACTION_INDEX[ActionType.USE_MED_KIT]]
    assert not mask[pillars.ACTION_INDEX[ActionType.EAT_FOOD]]
    assert not mask[pillars.ACTION_INDEX[ActionType.BUILD_HABITAT]]
    assert not mask[pillars.ACTION_INDEX[ActionType.COMMUNICATE]]
    assert not mask[pillars.ACTION_INDEX[ActionType.SHARE_RESOURCE]]


def test_agent_mask_accepts_build_and_maintenance_backed_by_cell_stock():
    poor = _agent(
        inventory=ResourceBundle(
            energy=0,
            oxygen=0,
            water=0,
            food=0,
            construction_material=0,
            minerals=0,
            med_kits=0,
        )
    )
    warehouse = ResourceBundle(
        construction_material=10,
        minerals=10,
        energy=10,
        oxygen=10,
        water=10,
    )

    mask = agent_action_mask(poor, _open_mask(), cell_resources=warehouse)

    assert mask[pillars.ACTION_INDEX[ActionType.BUILD_HABITAT]]
    assert mask[pillars.ACTION_INDEX[ActionType.MAINTAIN_STRUCTURE]]


def test_pillar_availability_follows_mask():
    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    mask[pillars.ACTION_INDEX[ActionType.COLLECT_MINERALS]] = True
    available = pillar_availability(mask)
    assert available[pillars.P_RESOURCES]
    assert not available[pillars.P_SUSTENANCE]


def test_score_pillars_is_pref_times_urgency_normalized():
    from src.agents.preference_agent import score_pillars

    preferences = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.2])
    urgencies = np.array([0.5, 1.0, 0.0, 0.2, 0.2, 0.4])
    available = np.array([True, True, False, True, True, True])
    scores = score_pillars(preferences, urgencies, available)
    assert abs(scores.sum() - 1.0) < 1e-12
    assert scores[2] == 0.0
    raw = preferences * urgencies * available
    assert np.allclose(scores, raw / raw.sum())


def test_role_skill_multiplier_changes_the_normalized_choice_weight():
    from src.agents.preference_agent import score_pillars

    preferences = np.full(pillars.N_PILLARS, 1.0 / pillars.N_PILLARS)
    urgencies = np.ones(pillars.N_PILLARS)
    available = np.ones(pillars.N_PILLARS, dtype=np.bool_)
    skills = np.ones(pillars.N_PILLARS)
    skills[pillars.P_BUILD] = 1.5
    scores = score_pillars(
        preferences, urgencies, available, np.ones(pillars.N_PILLARS), skills
    )
    assert scores[pillars.P_BUILD] == scores.max()
    assert scores[pillars.P_BUILD] > scores[pillars.P_RESOURCES]


def test_choose_pillar_greedy_ignores_uniform_softmax_uses_it():
    from src.agents.preference_agent import choose_pillar

    scores = np.array([0.1, 0.5, 0.0, 0.2, 0.1, 0.1])
    assert choose_pillar(scores, "greedy", 0.999) == 1
    assert choose_pillar(scores, "softmax", 0.05) == 0
    assert choose_pillar(scores, "softmax", 0.55) == 1
    assert choose_pillar(scores, "softmax", 0.999) == 5


def test_decide_preferences_dehydrated_drinks(world_small):
    from src.agents.preference_agent import decide_preferences

    world, agent = world_small
    agent.hydration = 0.2
    agent.pillar_preferences = tuple(np.full(6, 1.0 / 6.0))
    request = decide_preferences(
        agent,
        world,
        _open_mask(),
        u01=0.0,
        sampling="greedy",
        claims={},
        step=1,
    )
    assert request.action == ActionType.DRINK_WATER


def test_claims_stop_second_maintainer(world_small):
    from src.agents.preference_agent import decide_preferences

    world, agent = world_small
    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    mask[pillars.ACTION_INDEX[ActionType.MAINTAIN_STRUCTURE]] = True
    mask[pillars.ACTION_INDEX[ActionType.OBSERVE]] = True
    mask[pillars.ACTION_INDEX[ActionType.DO_NOTHING]] = True
    agent.pillar_preferences = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    claims = {}
    first = decide_preferences(agent, world, mask, 0.0, "greedy", claims, 1)
    second = decide_preferences(agent, world, mask, 0.0, "greedy", claims, 1)
    assert first.action == ActionType.MAINTAIN_STRUCTURE
    assert second.action != ActionType.MAINTAIN_STRUCTURE


def test_physiological_override_returns_toward_water_before_preferences(world_small):
    from src.agents.preference_agent import decide_preferences
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    target_x = min(world.width - 1, agent.x + 2)
    world.add_structure(
        Structure(type=StructureType.GREENHOUSE, x=target_x, y=agent.y)
    )
    agent.inventory.water = 0.0
    agent.inventory.ice = 0.0
    # The vitals clock is the binding constraint: after two missed water
    # actions the return starts even while the continuous hydration scalar is
    # still high.
    agent.hydration = 0.9
    agent.steps_without_water = 2
    # **La cella di partenza va prosciugata (2026-09-01).** Da quando ogni cella
    # ha un fondo idrico sotterraneo, un colono assetato beve o scava dove si
    # trova e non ha alcun motivo di viaggiare: la prova misurerebbe il fondo
    # idrico invece del rientro d'emergenza che dichiara di misurare.
    _qui = world.get_cell(agent.x, agent.y)
    _qui.water_ice = 0.0
    _qui.resources.ice = 0.0
    _qui.liquid_water = 0.0
    agent.pillar_preferences = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    request = decide_preferences(
        agent, world, _open_mask(), 0.9, "softmax", {}, 10
    )
    assert request.action == ActionType.MOVE
    assert "dehydration emergency" in (request.message or "")


def test_dehydrated_agent_enters_full_supported_outpost_and_drinks(world_small):
    from src.agents.action_space import execute_action
    from src.agents.preference_agent import decide_preferences
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    target = world.neighbors(agent.x, agent.y, 1)[0]
    for structure_type in (
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, target.x, target.y))
    target.agents_present.append("resident")  # nominal support capacity is one
    agent.inventory.water = 0.0
    agent.inventory.ice = 0.0
    agent.hydration = 0.1
    agent.steps_without_water = 7
    agent.satiety = 1.0
    # **La cella di partenza va prosciugata (2026-09-01).** Da quando ogni cella
    # ha un fondo idrico sotterraneo, un colono assetato beve o scava dove si
    # trova e non ha alcun motivo di viaggiare: la prova misurerebbe il fondo
    # idrico invece del rientro d'emergenza che dichiara di misurare.
    # Il prosciugamento vale per tutta la ROTTA: un macro-passo di cella si
    # percorre in piu' mosse settimanali, e basta una cella intermedia con del
    # ghiaccio perche' il colono si fermi a bere per strada.
    for _c in [world.get_cell(agent.x, agent.y), *world.neighbors(agent.x, agent.y, 2)]:
        if (_c.x, _c.y) == (target.x, target.y):
            continue
        _c.water_ice = 0.0
        _c.resources.ice = 0.0
        _c.liquid_water = 0.0
    agent.pillar_preferences = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    agents = {agent.agent_id: agent}

    for step in range(10, 30):
        world.step = step
        move = decide_preferences(
            agent, world, _open_mask(), 0.9, "softmax", {}, step
        )
        assert move.action == ActionType.MOVE
        assert move.target == {
            "x": target.x,
            "y": target.y,
            "survival_return": True,
        }
        moved = execute_action(agent, agents, world, move)
        assert moved.accepted
        if (agent.x, agent.y) == (target.x, target.y):
            break

    assert (agent.x, agent.y) == (target.x, target.y)
    # Un avamposto rifornito ha acqua in magazzino: dal 2026-08-25 bere
    # dall'impianto la preleva invece di crearla, e senza scorta l'avamposto
    # non e' rifornito ma solo costruito.
    target.resources.water = 5.0
    before_drink = agent.hydration
    world.step += 1

    drink = decide_preferences(
        agent, world, _open_mask(), 0.9, "softmax", {}, world.step
    )
    assert drink.action == ActionType.DRINK_WATER
    hydrated = execute_action(agent, agents, world, drink)
    assert hydrated.accepted
    assert agent.hydration > before_drink


def test_low_water_reserve_returns_before_dehydration(world_small):
    from src.agents.preference_agent import decide_preferences
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    target_x = min(world.width - 1, agent.x + 2)
    world.add_structure(
        Structure(type=StructureType.HABITAT, x=target_x, y=agent.y)
    )
    agent.hydration = 1.0
    agent.steps_without_water = 0
    agent.inventory.water = 3.0
    agent.inventory.ice = 0.0
    request = decide_preferences(
        agent, world, _open_mask(), 0.99, "softmax", {}, 4
    )

    assert request.action == ActionType.MOVE
    assert "low water reserve" in (request.message or "")


def test_starvation_emergency_preempts_preventive_water_refill(world_small):
    from src.agents.preference_agent import decide_preferences

    world, agent = world_small
    agent.hydration = 1.0
    agent.steps_without_water = 0
    agent.inventory.water = 0.0
    agent.inventory.ice = 0.0
    agent.satiety = 0.2
    agent.steps_without_food = 6
    agent.inventory.food = 0.2

    request = decide_preferences(
        agent, world, _open_mask(), 0.99, "softmax", {}, 4
    )

    assert request.action == ActionType.EAT_FOOD
    assert "starvation emergency" in (request.message or "")


def test_starvation_clock_preempts_repeated_drinking_when_water_clock_is_safe(
    world_small,
):
    from src.agents.preference_agent import decide_preferences

    world, agent = world_small
    agent.hydration = 0.2
    agent.steps_without_water = 0
    agent.inventory.water = 3.0
    agent.satiety = 0.2
    agent.steps_without_food = 6
    agent.inventory.food = 0.2

    request = decide_preferences(
        agent, world, _open_mask(), 0.99, "softmax", {}, 4
    )

    assert request.action == ActionType.EAT_FOOD
    assert "starvation emergency" in (request.message or "")


def test_starvation_return_can_escape_cell_menu_without_move_proposal(world_small):
    from src.agents.preference_agent import decide_preferences
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    target_x = min(world.width - 1, agent.x + 2)
    world.add_structure(
        Structure(type=StructureType.GREENHOUSE, x=target_x, y=agent.y)
    )
    agent.hydration = 0.2
    agent.steps_without_water = 0
    agent.inventory.water = 3.0
    agent.satiety = 0.2
    agent.steps_without_food = 6
    agent.inventory.food = 0.0
    cell_menu = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    cell_menu[pillars.ACTION_INDEX[ActionType.DRINK_WATER]] = True
    cell_menu[pillars.ACTION_INDEX[ActionType.DO_NOTHING]] = True

    request = decide_preferences(
        agent, world, cell_menu, 0.99, "softmax", {}, 4
    )

    assert request.action == ActionType.MOVE
    assert request.target.get("survival_return") is True
    assert "starvation emergency" in (request.message or "")


def test_returning_scout_is_not_starved_by_repeated_transit_claim(world_small):
    from src.agents.preference_agent import decide_preferences
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    home_x = min(world.width - 1, agent.x + 2)
    world.add_structure(
        Structure(type=StructureType.GREENHOUSE, x=home_x, y=agent.y)
    )
    intermediate_x = agent.x + 1
    intermediate = world.get_cell(intermediate_x, agent.y)
    intermediate.agents_present.extend(["resident_a", "resident_b"])
    agent.scout_phase = "back"
    agent.scout_home = (home_x, agent.y)
    agent.hydration = 0.2
    agent.steps_without_water = 0
    agent.inventory.water = 3.0
    agent.satiety = 0.2
    agent.steps_without_food = 6
    agent.inventory.food = 0.0
    cell_menu = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    cell_menu[pillars.ACTION_INDEX[ActionType.DRINK_WATER]] = True
    cell_menu[pillars.ACTION_INDEX[ActionType.DO_NOTHING]] = True
    claims = {("arrival", intermediate_x, agent.y): 1}

    request = decide_preferences(
        agent, world, cell_menu, 0.99, "softmax", claims, 4
    )

    assert request.action == ActionType.MOVE
    assert request.target["x"] == intermediate_x
    assert "starvation emergency" in (request.message or "")


def test_structure_return_lookup_is_shared_within_step(world_small):
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    target_x = min(world.width - 1, agent.x + 2)
    world.add_structure(
        Structure(type=StructureType.HABITAT, x=target_x, y=agent.y)
    )
    world.step = 11

    first = agent._move_toward_structure(world, {StructureType.HABITAT}, "home")
    second = agent._move_toward_structure(world, {StructureType.HABITAT}, "home")

    assert first is not None and second is not None
    assert first.target == second.target
    cache_step, cached = world._structure_target_cache
    assert cache_step == 11
    assert tuple((cell.x, cell.y) for cell in cached[("habitat",)]) == (
        (target_x, agent.y),
    )


def test_structure_target_lookup_uses_compact_position_index(world_small):
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    target_x = (agent.x + 2) % world.width
    world.add_structure(Structure(StructureType.GREENHOUSE, target_x, agent.y))

    original_cells = world.cells

    class FullGridScanForbidden:
        def __iter__(self):
            raise AssertionError("full world.cells scan must not be used")

        def __getitem__(self, index):
            return original_cells[index]

    world.cells = FullGridScanForbidden()
    try:
        targets = agent._structure_targets(world, {StructureType.GREENHOUSE})
    finally:
        world.cells = original_cells

    assert [(cell.x, cell.y) for cell in targets] == [(target_x, agent.y)]


def test_support_distance_is_cached_per_cell_and_step(world_small, monkeypatch):
    """Cache per (tipo, origine) del ramo `scalar`, verificata con un sentinella.

    Il test e' ancorato esplicitamente al backend `scalar` perche' ispeziona la
    sua struttura di memorizzazione. I backend a campo memorizzano un array per
    step invece di un valore per origine, quindi non hanno questo attributo: la
    proprieta' che li accomuna tutti e' verificata da
    ``test_support_distance_cache_is_invalidated_by_step_on_every_backend``.
    """
    from src.core import reachability_batch
    from src.world.structures import Structure, StructureType

    monkeypatch.setattr(reachability_batch, "_RESOLVED_BACKEND", "scalar")

    world, agent = world_small
    target_x = (agent.x + 2) % world.width
    world.add_structure(Structure(StructureType.GREENHOUSE, target_x, agent.y))
    world.step = 7

    first = agent._support_distance_cells(world, {StructureType.GREENHOUSE})
    cache_step, cached = world._support_distance_cache
    cached[(("greenhouse",), agent.x, agent.y)] = 99
    second = agent._support_distance_cells(world, {StructureType.GREENHOUSE})

    assert first == 2
    assert second == 99
    assert cache_step == 7

    world.step = 8
    assert agent._support_distance_cells(world, {StructureType.GREENHOUSE}) == 2


@pytest.mark.parametrize("backend", ["scalar", "numpy", "rust"])
def test_support_distance_cache_is_invalidated_by_step_on_every_backend(
    world_small, monkeypatch, backend
):
    """Ogni backend deve dare la stessa distanza e rispettare la validita' per step.

    E' l'intento del test sopra, espresso senza dipendere da come il valore viene
    memorizzato: i tre backend calcolano la stessa distanza, la memorizzano per lo
    step corrente e la ricalcolano quando lo step cambia. Se un backend divergesse
    qui, divergerebbero anche le decisioni di rientro degli agenti.
    """
    from src.core import native, reachability_batch
    from src.world.structures import Structure, StructureType

    if backend == "rust" and not native.is_available():
        pytest.skip(f"kernel nativo non disponibile: {native.unavailable_reason()}")
    monkeypatch.setattr(reachability_batch, "_RESOLVED_BACKEND", backend)

    world, agent = world_small
    target_x = (agent.x + 2) % world.width
    world.add_structure(Structure(StructureType.GREENHOUSE, target_x, agent.y))
    world.step = 7

    assert agent._support_distance_cells(world, {StructureType.GREENHOUSE}) == 2
    # Seconda lettura nello stesso step: stesso valore, servito dalla memoria.
    assert agent._support_distance_cells(world, {StructureType.GREENHOUSE}) == 2

    world.step = 8
    assert agent._support_distance_cells(world, {StructureType.GREENHOUSE}) == 2


@pytest.mark.parametrize("backend", ["scalar", "numpy", "rust"])
def test_support_distance_is_none_without_targets_on_every_backend(
    world_small, monkeypatch, backend
):
    """Senza strutture di supporto la distanza e' ``None``, non zero ne' infinito.

    E' il ramo che distingue "mondo senza insediamento" da "supporto lontano": il
    campo precomputato usa un sentinella intero, e una traduzione sbagliata del
    sentinella renderebbe raggiungibile qualunque cella.
    """
    from src.core import native, reachability_batch
    from src.world.structures import StructureType

    if backend == "rust" and not native.is_available():
        pytest.skip(f"kernel nativo non disponibile: {native.unavailable_reason()}")
    monkeypatch.setattr(reachability_batch, "_RESOLVED_BACKEND", backend)

    world, agent = world_small
    world.step = 3
    assert agent._support_distance_cells(world, {StructureType.INFIRMARY}) is None


def test_structure_return_uses_shortest_wrapped_longitude_route():
    from src.agents.population import spawn_initial_agents
    from src.world.structures import Structure, StructureType
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=91).generate(360, 20)
    agent = next(iter(spawn_initial_agents(1, world, seed=91).values()))
    origin = world.get_cell(agent.x, agent.y)
    origin.agents_present.remove(agent.agent_id)
    agent.x = 348
    world.get_cell(agent.x, agent.y).agents_present.append(agent.agent_id)
    world.add_structure(Structure(StructureType.GREENHOUSE, 0, agent.y))

    request = agent._move_toward_structure(
        world, {StructureType.GREENHOUSE}, "returning home"
    )

    assert request is not None
    assert request.target["x"] == 349


def test_move_toward_cell_crosses_antimeridian_in_one_step():
    from src.agents.population import spawn_initial_agents
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=92).generate(360, 20)
    agent = next(iter(spawn_initial_agents(1, world, seed=92).values()))
    origin = world.get_cell(agent.x, agent.y)
    origin.agents_present.remove(agent.agent_id)
    agent.x = 359
    world.get_cell(359, agent.y).agents_present.append(agent.agent_id)
    # **Un viaggiatore, non un residente qualunque (2026-09-01).** Dal momento
    # in cui `_move_toward_cell` sceglie il passo fra le celle che accetteranno
    # davvero chi arriva, un colono senza spedizione in corso non ha alcun
    # permesso di entrare in terreno vergine -- ed e' la regola del modello, non
    # un effetto di questa prova, che verifica la GEOMETRIA della cucitura a 360
    # gradi. I cinque chiamanti veri di questo metodo sono tutti spedizioni:
    # qui se ne dichiara una.
    agent.scout_phase = "out"

    request = agent._move_toward_cell(world, 0, agent.y, "crossing seam")

    assert request is not None
    assert request.target == {"x": 0, "y": agent.y}


def test_return_budget_preempts_preferences_before_food_is_critical():
    from src.agents.preference_agent import decide_preferences
    from src.agents.population import spawn_initial_agents
    from src.world.structures import Structure, StructureType
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=93).generate(360, 20)
    agent = next(iter(spawn_initial_agents(1, world, seed=93).values()))
    origin = world.get_cell(agent.x, agent.y)
    origin.agents_present.remove(agent.agent_id)
    agent.x = 348
    world.get_cell(agent.x, agent.y).agents_present.append(agent.agent_id)
    world.add_structure(Structure(StructureType.GREENHOUSE, 0, agent.y))
    agent.inventory.food = 2.5
    agent.inventory.water = 8.0
    agent.inventory.ice = 0.0
    agent.satiety = 1.0
    agent.hydration = 1.0

    request = decide_preferences(
        agent, world, _open_mask(), 0.99, "softmax", {}, 20
    )

    assert request.action == ActionType.MOVE
    assert request.target["x"] == 349
    assert "food return budget" in (request.message or "")


def test_shared_frontier_is_not_extended_without_round_trip_rations():
    from src.agents.preference_agent import _bind_explore
    from src.agents.population import spawn_initial_agents
    from src.world.structures import Structure, StructureType
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=94).generate(360, 20)
    agent = next(iter(spawn_initial_agents(1, world, seed=94).values()))
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))
    agent.inventory.food = 0.2
    agent.inventory.water = 0.2
    priorities = np.ones(pillars.N_ACTIONS, dtype=np.float64)

    request = _bind_explore(agent, world, world.get_cell(agent.x, agent.y), _open_mask(), priorities)

    assert request is not None
    assert request.action == ActionType.OBSERVE


def test_supported_cell_yields_the_turn_instead_of_wandering_over_mapped_ground(
    world_small, monkeypatch
):
    """Senza missione il pilastro CEDE il turno, e non vagabonda (2026-09-01).

    L'intento della prova non cambia: una preferenza di esplorazione non deve
    degradare in vagabondaggio fra celle gia' mappate. Cambia cio' che il
    binder restituisce quando non ha una missione. Prima era `OBSERVE`, e una
    richiesta restituita CHIUDE la decisione: quella riga sola produceva il
    48,3% di tutte le decisioni della colonia, e i coloni non provavano piu'
    nessun altro pilastro. Ora restituisce `None`, cosi' il ciclo passa a
    risorse, costruzione e vita; se nessuno di quelli ha lavoro, il ripiego
    finale di `decide_preferences_precomputed` restituisce comunque `OBSERVE`.

    Misurato: con il fondo regolitico questa differenza vale la popolazione
    81/95/97 contro 99/100/100, perche' il turno ceduto finisce in
    manutenzione. Vedi la voce 34 della checklist.
    """
    from src.agents.preference_agent import _bind_explore
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    current = world.get_cell(agent.x, agent.y)
    for structure_type in (
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, agent.x, agent.y))
    for cell in [current, *world.neighbors(agent.x, agent.y, 3)]:
        world.mark_explored(cell.x, cell.y)
    agent.inventory.water = 8.0
    agent.inventory.food = 4.0
    agent.movement_distance_m_per_step = 1.0e9
    agent.scout_next_at = 0
    agent.actions_taken = 100
    monkeypatch.setattr(agent, "_settlement_action", lambda *_args: None)

    request = _bind_explore(
        agent,
        world,
        current,
        _open_mask(),
        np.ones(pillars.N_ACTIONS, dtype=np.float64),
    )

    # cede il turno: niente movimento su terreno gia' mappato...
    assert request is None

    # ...e la decisione completa non finisce mai in `do_nothing`: il ripiego
    # finale restituisce `OBSERVE` con il messaggio che dice perche'. Sono
    # entrambi senza effetto, ma solo uno dei due conserva la causa.
    from src.agents.preference_agent import decide_preferences_precomputed

    finale = decide_preferences_precomputed(
        agent,
        world,
        _open_mask(),
        0.5,
        "greedy",
        {},
        0,
        action_priority_row=np.ones(pillars.N_ACTIONS, dtype=np.float64),
        individual_survival_priority_enabled=False,
    )
    assert finale.action != ActionType.DO_NOTHING
    if finale.action == ActionType.OBSERVE:
        assert (finale.message or "").strip(), "un `observe` di ripiego deve dire perche'"


def test_explore_preference_can_start_settlement_when_phase_is_inactive(
    world_small, monkeypatch
):
    from src.agents.action_space import ActionRequest
    from src.agents.preference_agent import _bind_explore

    world, agent = world_small
    assert agent.settle_phase is None
    calls = []
    target = world.neighbors(agent.x, agent.y, 1)[0]

    def start_or_continue(world_arg, cell_arg):
        calls.append((world_arg, cell_arg))
        agent.settle_phase = "out"
        agent.settle_target = (target.x, target.y)
        return ActionRequest(
            agent.agent_id,
            ActionType.EXPLORE,
            target={"x": target.x, "y": target.y},
            message="settlement expedition: founding an outpost",
        )

    monkeypatch.setattr(agent, "_settlement_action", start_or_continue)
    priorities = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    priorities[pillars.ACTION_INDEX[ActionType.EXPLORE]] = 1.0

    request = _bind_explore(
        agent,
        world,
        world.get_cell(agent.x, agent.y),
        _open_mask(),
        priorities,
    )

    assert len(calls) == 1
    assert request is not None and request.action == ActionType.EXPLORE
    assert request.target == {"x": target.x, "y": target.y}
    assert request.message.startswith("[pref]")


def test_founder_waits_in_nominating_cell_between_preparation_retries(
    world_small, monkeypatch
):
    from src.agents.preference_agent import _bind_explore

    world, agent = world_small
    agent.founder_kit_reserved = True
    monkeypatch.setattr(agent, "_settlement_action", lambda *_args: None)

    request = _bind_explore(
        agent,
        world,
        world.get_cell(agent.x, agent.y),
        _open_mask(),
        np.ones(pillars.N_ACTIONS, dtype=np.float64),
    )

    assert request is not None
    assert request.action == ActionType.OBSERVE
    assert "founder kit" in (request.message or "")


def test_founder_does_not_spend_reserved_kit_on_ordinary_build():
    from src.agents.preference_agent import _bind_build

    agent = _agent()
    agent.founder_kit_reserved = True
    priorities = np.ones(pillars.N_ACTIONS, dtype=np.float64)

    request = _bind_build(
        agent,
        type("CellStub", (), {"agents_present": [], "construction_sites": {}})(),
        _open_mask(),
        priorities,
    )

    assert request is None


def test_arrived_founder_build_pillar_is_bound_to_bootstrap(monkeypatch):
    from src.agents.action_space import ActionRequest
    from src.agents.preference_agent import _bind_build

    agent = _agent()
    agent.founder_kit_reserved = True
    agent.settle_phase = "out"
    agent.settle_target = (agent.x, agent.y)
    monkeypatch.setattr(
        agent,
        "_bootstrap_build_action",
        lambda _cell: ActionRequest(agent.agent_id, ActionType.BUILD_OXYGEN_PLANT),
    )
    cell = type(
        "CellStub",
        (),
        {"agents_present": [agent.agent_id], "construction_sites": {}},
    )()

    request = _bind_build(
        agent,
        cell,
        _open_mask(),
        np.ones(pillars.N_ACTIONS, dtype=np.float64),
    )

    assert request is not None
    assert request.action == ActionType.BUILD_OXYGEN_PLANT
    assert request.target == {"coverage_population": 1}


def test_move_does_not_overfill_outpost_without_survival_trio(world_small, monkeypatch):
    from src.agents.preference_agent import _bind_explore
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    current = world.get_cell(agent.x, agent.y)
    neighbors = world.neighbors(agent.x, agent.y, 1)
    target = neighbors[0]
    target.structures.append(Structure(StructureType.GREENHOUSE, target.x, target.y))
    target.agents_present.append("founder")
    for other in neighbors[1:]:
        other.agents_present.append("pioneer")  # virgin capacity is one
    monkeypatch.setattr(agent, "_settlement_action", lambda *_args: None)
    monkeypatch.setattr(agent, "_can_visit_cell_and_return", lambda *_args: True)
    priorities = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    priorities[pillars.ACTION_INDEX[ActionType.MOVE]] = 1.0

    request = _bind_explore(agent, world, current, _open_mask(), priorities)

    assert request is not None
    assert request.action == ActionType.OBSERVE


def test_move_admits_residents_up_to_complete_local_coverage(world_small, monkeypatch):
    from src.agents.preference_agent import _bind_explore
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    current = world.get_cell(agent.x, agent.y)
    neighbors = world.neighbors(agent.x, agent.y, 1)
    target = neighbors[0]
    for structure_type in (
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        target.structures.append(Structure(structure_type, target.x, target.y))
    for other in neighbors[1:]:
        other.agents_present.append("pioneer")
    monkeypatch.setattr(agent, "_settlement_action", lambda *_args: None)
    monkeypatch.setattr(agent, "_can_visit_cell_and_return", lambda *_args: True)
    priorities = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    priorities[pillars.ACTION_INDEX[ActionType.MOVE]] = 1.0

    request = _bind_explore(agent, world, current, _open_mask(), priorities)

    assert request is not None
    assert request.action == ActionType.MOVE
    assert request.target == {"x": target.x, "y": target.y}


def test_explore_does_not_overfill_under_supported_cell(world_small, monkeypatch):
    from src.agents.preference_agent import _bind_explore

    world, agent = world_small
    current = world.get_cell(agent.x, agent.y)
    for target in world.neighbors(agent.x, agent.y, 1):
        target.agents_present.append("pioneer")
    monkeypatch.setattr(agent, "_settlement_action", lambda *_args: None)
    monkeypatch.setattr(agent, "_can_visit_cell_and_return", lambda *_args: True)
    priorities = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    priorities[pillars.ACTION_INDEX[ActionType.EXPLORE]] = 1.0

    request = _bind_explore(agent, world, current, _open_mask(), priorities)

    assert request is not None
    assert request.action == ActionType.OBSERVE


def test_batch_arrival_claim_reserves_the_founding_party_and_no_more(world_small):
    """Il cantiere ammette la squadra che gli serve, e il registro la limita.

    **Fissava «un pioniere solo» (2026-09-01).** Era la regola di allora:
    `popolazione == 0`. Ma perche' una cella diventi abitabile servono quattro
    strutture -- alloggio, serra, pannelli, impianto d'ossigeno -- e finche' ne
    manca una la capienza resta zero, quindi la cella rifiuta chiunque:
    l'avamposto si chiudeva alle spalle del primo arrivato, che da solo non le
    finiva. Misurato a 400 passi su tre semi, 32 avamposti fermi a capienza
    zero, ventidue gia' abbandonati.

    Ora i posti sono `posti_di_cantiere`, cioe' quanti dei quattro requisiti
    mancano. La PROPRIETA' che questa prova difende non cambia: il registro
    delle prenotazioni deve limitare la squadra dentro lo stesso passo, cosi'
    che una folla non si riversi su una sola cella.
    """
    from src.agents.action_space import ActionRequest
    from src.agents.build_policy import posti_di_cantiere
    from src.world.structures import STRUTTURE_DI_AVVIAMENTO
    from src.agents.preference_agent import _claim_arrival_if_available

    world, agent = world_small
    origin = world.get_cell(agent.x, agent.y)
    target = world.neighbors(agent.x, agent.y, 1)[0]
    request = ActionRequest(
        agent.agent_id,
        ActionType.EXPLORE,
        {"x": target.x, "y": target.y},
    )
    claims = {}
    agent.settle_phase = "out"

    squadra = posti_di_cantiere(target)
    assert squadra == len(STRUTTURE_DI_AVVIAMENTO), (
        "una cella vergine ha da coprire tutti i requisiti dell'avviamento"
    )
    for _ in range(squadra):
        assert _claim_arrival_if_available(agent, request, origin, world, claims)
    assert not _claim_arrival_if_available(agent, request, origin, world, claims)
    assert claims[("arrival", target.x, target.y)] == squadra


def test_survival_return_reserves_only_one_emergency_overflow_slot(world_small):
    from src.agents.action_space import ActionRequest
    from src.agents.preference_agent import _claim_arrival_if_available
    from src.world.structures import Structure, StructureType

    world, agent = world_small
    origin = world.get_cell(agent.x, agent.y)
    target = world.neighbors(agent.x, agent.y, 1)[0]
    for structure_type in (
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        target.structures.append(Structure(structure_type, target.x, target.y))
    target.agents_present.append("resident")
    request = ActionRequest(
        agent.agent_id,
        ActionType.MOVE,
        {"x": target.x, "y": target.y, "survival_return": True},
    )
    claims = {}

    assert _claim_arrival_if_available(agent, request, origin, world, claims)
    assert not _claim_arrival_if_available(agent, request, origin, world, claims)
    assert claims[("arrival", target.x, target.y)] == 1

    ordinary = ActionRequest(
        agent.agent_id,
        ActionType.MOVE,
        {"x": target.x, "y": target.y},
    )
    assert not _claim_arrival_if_available(agent, ordinary, origin, world, {})

    unsupported = world.neighbors(agent.x, agent.y, 1)[1]
    unsupported.agents_present.append("stranded")
    unsupported_return = ActionRequest(
        agent.agent_id,
        ActionType.MOVE,
        {
            "x": unsupported.x,
            "y": unsupported.y,
            "survival_return": True,
        },
    )
    assert not _claim_arrival_if_available(
        agent, unsupported_return, origin, world, {}
    )
