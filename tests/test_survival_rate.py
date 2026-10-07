"""survival_rate = sopravvissuti della coorte fondatrice / coorte fondatrice.

Regressione della run 120726_testlogic5_1000steps: con le nascite interne la
popolazione (207) supera il conteggio iniziale (100) e la vecchia formula
``vivi / iniziali`` riportava 207%. Il tasso deve partire da 1.0, scendere solo
quando muore un fondatore e restare sempre in [0, 1] su entrambi i motori.
"""

from src.agents.population import maybe_spawn_agent, spawn_initial_agents
from src.agents.rule_based_agent import RuleBasedAgent
from src.api.state_store import SimulationController
from src.experiments.metrics import agent_metrics
from src.simulation.agent_coupled_runner import AgentCoupledRunner, _aggregate_metrics
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


BASE_CONFIG = {
    "name": "survival_unit",
    "seed": 3,
    "days": 5,
    "world": {"width": 8, "height": 8},
    "agents": {"count": 2, "llm_count": 0},
    "simulation": {"days_per_step": 1, "max_days": 10},
    "population": {"enabled": False},
    "climate": {"enabled": False},
    "environmental_layer": {"enabled": False},
}

GROWTH_CONFIG = {
    "population": {
        "enabled": True,
        "max_agents": 10,
        "daily_spawn_probability": 1.0,
        "prosperity_growth_scale": 1.0,
        "min_habitability_for_growth": 0.0,
    }
}


def _force_birth(engine, seed):
    """Nascita deterministica: prosperita' e habitability sopra ogni soglia.

    Takes the ENGINE (not a bare `agents`/`world` pair) since Task 12's
    core-backed `SimulationController` needs an extra commit step
    `AgentCoupledRunner` (still the object engine, untouched) does not:
    `maybe_spawn_agent` inserts a real, fully-formed `RuleBasedAgent` object
    straight into whatever `agents` dict it is given - for the object engine
    that dict IS the live population, done; for the core engine `sim.agents`
    is a view rebuilt from `AgentArrays` every step, so the newborn needs a
    real array row too (`SimulationController._commit_spawned_agent`,
    the exact bridge `step()` itself calls on a real in-run birth) or it
    would vanish the moment anything rebuilds that dict.
    """
    agents = engine.agents
    world = engine.world
    for agent in agents.values():
        agent.inventory.food = 10.0
        agent.inventory.water = 10.0
        agent.inventory.construction_material = 10.0
        agent.inventory.tools = 4.0
    first = next(iter(agents.values()))
    world.add_structure(Structure(StructureType.GREENHOUSE, first.x, first.y))
    world.add_structure(Structure(StructureType.HABITAT, first.x, first.y))
    world.add_structure(Structure(StructureType.HABITAT, first.x, first.y))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, first.x, first.y))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, first.x, first.y))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, first.x, first.y))
    world.get_cell(first.x, first.y).habitability_score = 0.5
    # Il corredo del neonato lo paga la CELLA (2026-08-30): senza scorte in
    # magazzino la colonia non puo' mantenerlo e la nascita e' vietata.
    _magazzino = world.get_cell(first.x, first.y).resources
    _magazzino.food = max(_magazzino.food, 50.0)
    _magazzino.water = max(_magazzino.water, 40.0)
    _magazzino.oxygen = max(_magazzino.oxygen, 20.0)
    config = {**GROWTH_CONFIG, "colony": {"start_x": first.x, "start_y": first.y}}
    spawned = maybe_spawn_agent(agents, world, config, day=1, seed=seed)
    assert spawned is not None, "spawn forzato fallito: il fixture non e' piu' valido"
    commit = getattr(engine, "_commit_spawned_agent", None)
    if commit is not None:
        commit(spawned)
    return spawned


def _kill(engine, agent_id):
    agent = engine.agents[agent_id]
    if hasattr(engine.world, "remove_agent"):
        engine.world.remove_agent(agent_id, agent.x, agent.y)
        del engine.agents[agent_id]
    else:
        # Core-backed SimulationController (Task 12): there is no per-cell
        # agents_present list to update (CellView.agents_present derives
        # membership live from AgentArrays.x/y) - "death" is
        # AgentArrays.alive flipping False, mirroring what kernel_vitals's
        # tick_vitals does for a real death.
        core_agents = engine.core.agents
        core_agents.alive[core_agents.index[agent_id]] = False
        del engine.agents[agent_id]


def test_gui_engine_survival_rate_stays_in_unit_range_with_births():
    sim = SimulationController({**BASE_CONFIG, "name": "survival_gui"})
    founders = sorted(sim.agents.keys())
    spawned = _force_birth(sim, seed=3)

    metrics = sim._current_metrics(1, 1.0)
    assert metrics["population"] == 3
    assert metrics["survival_rate"] == 1.0

    _kill(sim, founders[0])
    assert sim._current_metrics(2, 2.0)["survival_rate"] == 0.5

    # La morte di un nato dopo non tocca la sopravvivenza della coorte.
    _kill(sim, spawned.agent_id)
    metrics = sim._current_metrics(3, 3.0)
    assert metrics["survival_rate"] == 0.5
    assert 0.0 <= metrics["survival_rate"] <= 1.0


def test_cli_engine_survival_rate_stays_in_unit_range_with_births():
    runner = AgentCoupledRunner({**BASE_CONFIG, "name": "survival_cli"})
    founders = sorted(runner.agents.keys())
    _force_birth(runner, seed=3)

    metrics = runner._current_metrics(1, 1.0)
    assert metrics["population"] == 3
    assert metrics["survival_rate"] == 1.0

    _kill(runner, founders[0])
    assert runner._current_metrics(2, 2.0)["survival_rate"] == 0.5


def test_aggregate_metrics_survival_rate_is_clamped():
    state = {
        "population": 20.0,
        "deaths": 0.0,
        "food": 1.0,
        "water": 1.0,
        "construction_material": 1.0,
        "tools": 1.0,
        "occupied_m2": 1.0,
    }
    assert _aggregate_metrics(1, 1.0, state, {}, 10)["survival_rate"] == 1.0

    state["deaths"] = 4.0
    assert _aggregate_metrics(2, 2.0, state, {}, 10)["survival_rate"] == 0.6

    state["deaths"] = 25.0
    assert _aggregate_metrics(3, 3.0, state, {}, 10)["survival_rate"] == 0.0


def test_experiment_agent_metrics_survival_uses_cohort_and_caps_fallback():
    world = WorldGenerator(seed=6).generate(6, 6)
    agents = spawn_initial_agents(3, world, seed=6)
    founders = set(agents.keys())
    newborn = RuleBasedAgent(
        agent_id="agent_099",
        name="Settler 99",
        role="colonist",
        x=world.width // 2,
        y=world.height // 2,
        perception_radius=1,
        perception_radius_m=5_000.0,
        local_x_m=0.0,
        local_y_m=0.0,
    )
    agents[newborn.agent_id] = newborn

    grown = agent_metrics(agents, [], initial_agent_count=3, initial_agent_ids=founders)
    assert grown["survival_rate"] == 1.0

    del agents[sorted(founders)[0]]
    after_death = agent_metrics(agents, [], initial_agent_count=3, initial_agent_ids=founders)
    assert abs(after_death["survival_rate"] - 2.0 / 3.0) < 1e-9

    # Fallback senza coorte (config storiche): mai sopra 1.0.
    fallback = agent_metrics(agents, [], initial_agent_count=2)
    assert fallback["survival_rate"] == 1.0
