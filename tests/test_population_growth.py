from src.agents.population import maybe_spawn_agent, spawn_initial_agents
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def test_population_growth_respects_disabled_flag():
    world = WorldGenerator(seed=4).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=4)
    spawned = maybe_spawn_agent(agents, world, {"population": {"enabled": False}}, day=1)
    assert spawned is None
    assert len(agents) == 1


def test_population_growth_is_internal_and_prosperity_gated():
    world = WorldGenerator(seed=5).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=5)
    agent = next(iter(agents.values()))
    agent.inventory.food = 10.0
    agent.inventory.water = 10.0
    agent.inventory.construction_material = 10.0
    agent.inventory.tools = 4.0
    x, y = world.width // 2, world.height // 2
    world.add_structure(Structure(StructureType.GREENHOUSE, x, y))
    world.add_structure(Structure(StructureType.HABITAT, x, y))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, x, y))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, x, y))
    world.get_cell(x, y).habitability_score = 0.5
    # Il corredo del neonato lo paga la CELLA (2026-08-30): senza scorte in
    # magazzino la colonia non puo' mantenerlo e la nascita e' vietata.
    _magazzino = world.get_cell(x, y).resources
    _magazzino.food = max(_magazzino.food, 50.0)
    _magazzino.water = max(_magazzino.water, 40.0)
    _magazzino.oxygen = max(_magazzino.oxygen, 20.0)

    spawned = maybe_spawn_agent(
        agents,
        world,
        {
            "population": {
                "enabled": True,
                "max_agents": 3,
                "daily_spawn_probability": 1.0,
                "prosperity_growth_scale": 1.0,
                "min_habitability_for_growth": 0.0,
            }
        },
        day=1,
        seed=5,
    )

    assert spawned is not None
    assert spawned.role == "colonist"
    assert len(agents) == 2
    assert world.events[-1]["type"] == "population_grew"


def test_population_growth_waits_for_housing_and_local_life_support(monkeypatch):
    from src.agents import population

    world = WorldGenerator(seed=6).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=6)
    founder = next(iter(agents.values()))
    monkeypatch.setattr(population, "_colony_growth_probability", lambda *_: 1.0)
    config = {
        "population": {
            "enabled": True,
            "max_agents": 3,
            "daily_spawn_probability": 1.0,
            "min_habitability_for_growth": -1.0,
        }
    }

    for structure_type in (
        StructureType.HABITAT,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
    ):
        world.add_structure(Structure(structure_type, founder.x, founder.y))
    # Il magazzino e' pieno: il rifiuto che questo test verifica deve venire dal
    # supporto vitale mancante, non dal corredo impagabile.
    _magazzino = world.get_cell(founder.x, founder.y).resources
    _magazzino.food, _magazzino.water, _magazzino.oxygen = 50.0, 40.0, 20.0
    assert maybe_spawn_agent(agents, world, config, day=1, seed=6) is None

    # **Il requisito che completa la griglia e' ora il pozzo (2026-09-01).**
    # L'impianto d'ossigeno da solo non basta piu': l'acqua e' entrata fra i
    # termini della capienza vitale, e una cella che non disseta non sostiene
    # nessuno. La prova conserva la sua forma — si nega la nascita finche' la
    # griglia e' incompleta, la si concede appena e' completa — e cambia
    # soltanto quale sia l'ultimo mattone.
    world.add_structure(
        Structure(StructureType.OXYGEN_PLANT, founder.x, founder.y)
    )
    assert maybe_spawn_agent(agents, world, config, day=1, seed=6) is None

    world.add_structure(
        Structure(StructureType.WATER_EXTRACTOR, founder.x, founder.y)
    )
    assert maybe_spawn_agent(agents, world, config, day=1, seed=6) is not None


def test_population_growth_uses_supported_outpost_spare_capacity(monkeypatch):
    from src.agents import population

    world = WorldGenerator(seed=8).generate(10, 10)
    agents = spawn_initial_agents(2, world, seed=8)
    first, second = list(agents.values())
    start = world.get_cell(first.x, first.y)
    target = world.get_cell((first.x + 1) % world.width, first.y)
    world.move_agent(second.agent_id, second.x, second.y, target.x, target.y)
    second.x, second.y = target.x, target.y

    for structure_type in (
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, start.x, start.y))
    for structure_type in (
        StructureType.HABITAT,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, target.x, target.y))

    # Il corredo del neonato lo paga la CELLA (2026-08-30). Entrambe le celle
    # hanno scorte: cosi' la scelta fra le due dipende dalla capienza libera,
    # che e' cio' che questo test verifica, e non da chi ha il magazzino pieno.
    for _cella in (start, target):
        _cella.resources.food = 50.0
        _cella.resources.water = 40.0
        _cella.resources.oxygen = 20.0

    monkeypatch.setattr(population, "_colony_growth_probability", lambda *_: 1.0)
    spawned = maybe_spawn_agent(
        agents,
        world,
        {
            "population": {
                "enabled": True,
                "max_agents": 3,
                "daily_spawn_probability": 1.0,
                "min_habitability_for_growth": -1.0,
            }
        },
        day=1,
        seed=8,
    )

    assert spawned is not None
    assert (spawned.x, spawned.y) == (target.x, target.y)
    assert len(start.agents_present) == 1
    assert len(target.agents_present) == 2


def _colonia_attrezzata(seed: int, coloni: int):
    """Una colonia con capienza abbondante: qui interessano prosperita' e dadi."""
    world = WorldGenerator(seed=seed).generate(10, 10)
    agents = spawn_initial_agents(coloni, world, seed=seed)
    x, y = next(iter(agents.values())).x, next(iter(agents.values())).y
    for _ in range(coloni):
        world.add_structure(Structure(StructureType.HABITAT, x, y))
        world.add_structure(Structure(StructureType.GREENHOUSE, x, y))
        world.add_structure(Structure(StructureType.SOLAR_ARRAY, x, y))
        world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))
        # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
        world.add_structure(Structure(StructureType.WATER_EXTRACTOR, x, y))
    cella = world.get_cell(x, y)
    cella.habitability_score = 1.0
    # Il corredo del neonato lo paga la CELLA (2026-08-30): una colonia
    # attrezzata ha anche di che mantenere i figli che mette al mondo.
    cella.resources.food = max(cella.resources.food, 20.0 * max(1, coloni))
    cella.resources.water = max(cella.resources.water, 16.0 * max(1, coloni))
    cella.resources.oxygen = max(cella.resources.oxygen, 8.0 * max(1, coloni))
    return world, agents, cella


def test_prosperity_counts_the_warehouse_and_not_only_the_knapsacks():
    """La prosperita' misurava una costante di configurazione, non la ricchezza.

    Con la redistribuzione attiva ogni sacca viene riempita fino a un bersaglio
    fisso, quindi `cibo_nelle_sacche / (popolazione x 3)` valeva sempre lo
    stesso numero: misurato 0,666 su una colonia stabile di centoventi coloni,
    identico che il magazzino fosse pieno o vuoto.
    """
    from src.agents.population import _colony_growth_probability

    cfg = {"prosperity_growth_scale": 1.0}
    world, agents, cella = _colonia_attrezzata(11, 4)
    for agente in agents.values():
        agente.inventory.food = 0.0
        agente.inventory.water = 0.0
        agente.inventory.construction_material = 0.0
        agente.inventory.tools = 0.0
    cella.resources.food = 0.0
    cella.resources.water = 0.0
    affamata = _colony_growth_probability(agents, world, cfg)

    # Stessa colonia, stesse sacche vuote, ma il magazzino e' pieno.
    cella.resources.food = 100.0
    cella.resources.water = 100.0
    cella.resources.construction_material = 100.0
    cella.resources.tools = 100.0
    ricca = _colony_growth_probability(agents, world, cfg)

    assert affamata == 0.0
    assert ricca > affamata


def test_births_scale_with_population_instead_of_one_per_step():
    """Prima la colonia guadagnava al piu' una persona a passo, a ogni scala.

    Erano 52 nascite all'anno terrestre con cento abitanti come con centomila,
    cioe' una crescita lineare invece che proporzionale. Ora si tira un dado
    ogni `POPOLAZIONE_PER_DADO_DI_NATALITA` coloni.
    """
    from src.agents.population import POPOLAZIONE_PER_DADO_DI_NATALITA, maybe_spawn_agents

    cfg = {
        "population": {
            "enabled": True,
            "max_agents": 10_000,
            "daily_spawn_probability": 1.0,
            "prosperity_growth_scale": 1.0,
            "min_habitability_for_growth": 0.0,
        },
        "simulation": {"days_per_step": 1},
    }
    grande = 4 * POPOLAZIONE_PER_DADO_DI_NATALITA
    world, agents, cella = _colonia_attrezzata(12, grande)
    for agente in agents.values():
        agente.inventory.food = 50.0
        agente.inventory.water = 50.0
        agente.inventory.construction_material = 50.0
        agente.inventory.tools = 50.0

    nati = maybe_spawn_agents(agents, world, cfg, day=1, seed=3)
    assert len(nati) > 1, "una colonia grande non puo' fermarsi a una nascita per passo"

    # Alla popolazione di riferimento il comportamento resta quello di prima.
    world2, agents2, _ = _colonia_attrezzata(12, POPOLAZIONE_PER_DADO_DI_NATALITA)
    for agente in agents2.values():
        agente.inventory.food = 50.0
        agente.inventory.water = 50.0
        agente.inventory.construction_material = 50.0
        agente.inventory.tools = 50.0
    assert len(maybe_spawn_agents(agents2, world2, cfg, day=1, seed=3)) <= 1
