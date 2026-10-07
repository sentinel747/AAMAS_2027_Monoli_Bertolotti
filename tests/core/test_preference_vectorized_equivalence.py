import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionRequest, ActionType, BUILD_ACTIONS
from src.agents.population import spawn_initial_agents
from src.agents.preference_agent import (
    URGENCY_VERSION,
    _claim_arrival_if_available,
    agent_action_mask,
    choose_pillar,
    compute_urgencies,
    pillar_availability,
    pillar_priority,
    score_pillars,
)
from src.core import constants as C, kernel
from src.core.cell_action_mask import compute_cell_masks
from src.core.kernel import _PreferenceAgentOnViews
from src.core.kernel_decision import (
    URGENCY_VERSION as BATCH_URGENCY_VERSION,
    agent_action_mask_batch,
    choose_pillar_batch,
    compute_urgencies_batch,
    pillar_availability_batch,
    pillar_priority_batch,
    score_pillars_batch,
    decide_batch,
)
from src.core.shell_common import build_core_state, rebuild_agent_views
from src.world.world_generator import WorldGenerator
from src.world.structures import Structure, StructureType


def _dirty_state():
    world = WorldGenerator(seed=31).generate(32, 20)
    objects = spawn_initial_agents(20, world, seed=31)
    core, _ = build_core_state(world, objects)
    config = {
        "seed": 31,
        "agents": {
            "decision_mode": "preferences",
            "decision_sampling": "softmax",
        },
        "headless": {"fast_observation": True, "rule_based_observation": True},
    }
    for step_index in range(1, 6):
        kernel.step(core, step_index, 7.0, config, None)
    return core


def test_batch_primitives_are_bit_exact_with_reference():
    core = _dirty_state()
    rows = core.agents.alive_rows()
    views = rebuild_agent_views(core)
    cell_masks = compute_cell_masks(core.cells)
    at_agent = cell_masks[core.agents.y[rows], core.agents.x[rows]]

    batch_masks = agent_action_mask_batch(core.agents, rows, at_agent)
    reference_masks = np.stack(
        [
            agent_action_mask(views[core.agents.ids[int(row)]], at_agent[index])
            for index, row in enumerate(rows)
        ]
    )
    assert np.array_equal(batch_masks, reference_masks)

    batch_urgencies = compute_urgencies_batch(core.agents, rows, batch_masks)
    reference_urgencies = np.stack(
        [
            compute_urgencies(views[core.agents.ids[int(row)]], batch_masks[index])
            for index, row in enumerate(rows)
        ]
    )
    assert URGENCY_VERSION == BATCH_URGENCY_VERSION
    # **Un ULP di tolleranza, e non e' un allentamento di comodo.** Le urgenze
    # elevano al quadrato, e il percorso scalare lo fa con `x ** 2` su un float
    # Python — cioe' attraverso `pow()` della libreria di sistema — mentre il
    # percorso vettoriale passa da NumPy. Le due strade divergono all'ultimo bit
    # per circa 556 valori su un milione nel dominio vero, come il docstring di
    # `src/core/decision_scoring.py` documenta da tempo, e per quei valori
    # l'uguaglianza esatta NON e' ottenibile: misurato su
    # `base = 0.00012085331841926727`, Python da' 1.4605524572948808e-08 e
    # nessuna forma NumPy (`a**2`, `np.power(a, 2.0)`, `a*a`) restituisce
    # qualcosa di diverso da 1.4605524572948806e-08.
    #
    # Il test passava finche' lo stato dopo cinque passi non centrava uno di
    # quei valori; una ritaratura dei vitali lo ha centrato. La tolleranza e'
    # stretta apposta — `rtol=1e-15` accetta un paio di ULP e nient'altro —
    # cosi' una divergenza vera resta rossa.
    assert np.allclose(batch_urgencies, reference_urgencies, rtol=1e-15, atol=0.0), (
        "le urgenze divergono oltre l'errore di arrotondamento: "
        f"scarto massimo {np.abs(batch_urgencies - reference_urgencies).max():.3e}"
    )

    batch_available = pillar_availability_batch(batch_masks)
    reference_available = np.stack(
        [pillar_availability(mask) for mask in batch_masks]
    )
    assert np.array_equal(batch_available, reference_available)

    action_priorities = batch_masks.astype(np.float64) * np.linspace(
        0.1, 1.0, pillars.N_ACTIONS
    )
    batch_priority = pillar_priority_batch(batch_masks, action_priorities)
    reference_priority = np.stack(
        [
            pillar_priority(batch_masks[index], action_priorities[index])
            for index in range(rows.size)
        ]
    )
    assert np.array_equal(batch_priority, reference_priority)

    batch_scores = score_pillars_batch(
        core.agents.pref[rows],
        batch_urgencies,
        batch_available,
        batch_priority,
        core.agents.skill[rows],
    )
    reference_scores = np.stack(
        [
            score_pillars(
                core.agents.pref[int(row)],
                reference_urgencies[index],
                reference_available[index],
                reference_priority[index],
                views[core.agents.ids[int(row)]].pillar_skills,
            )
            for index, row in enumerate(rows)
        ]
    )
    # Stessa tolleranza delle urgenze, e per la stessa ragione: i punteggi le
    # moltiplicano, quindi l'ULP di differenza sull'elevamento al quadrato si
    # propaga qui. La scelta del pilastro, che segue, resta invece un confronto
    # ESATTO: un ULP sui punteggi non deve mai cambiare quale pilastro vince, e
    # se lo cambiasse vorremmo saperlo.
    assert np.allclose(batch_scores, reference_scores, rtol=1e-15, atol=0.0), (
        "i punteggi divergono oltre l'errore di arrotondamento: "
        f"scarto massimo {np.abs(batch_scores - reference_scores).max():.3e}"
    )

    uniforms = np.random.default_rng(91).random(rows.size)
    for sampling in ("greedy", "softmax"):
        batch_choice = choose_pillar_batch(batch_scores, sampling, uniforms)
        reference_choice = np.array(
            [
                choose_pillar(score, sampling, uniforms[index])
                for index, score in enumerate(reference_scores)
            ],
            dtype=np.int64,
        )
        assert np.array_equal(batch_choice, reference_choice)


def test_paid_construction_site_remains_feasible_without_full_cost_stock():
    world = WorldGenerator(seed=39).generate(32, 20)
    objects = spawn_initial_agents(1, world, seed=39)
    agent = next(iter(objects.values()))
    cell = world.get_cell(agent.x, agent.y)
    for resource in agent.inventory.__dataclass_fields__:
        setattr(agent.inventory, resource, 0.0)
        setattr(cell.resources, resource, 0.0)
    cell.construction_sites[
        f"{StructureType.OXYGEN_PLANT.value}@1000:1000"
    ] = 100.0 / 3.0

    core, _ = build_core_state(world, objects)
    rows = core.agents.alive_rows()
    view = rebuild_agent_views(core)[agent.agent_id]
    open_mask = np.ones(pillars.N_ACTIONS, dtype=np.bool_)

    scalar = agent_action_mask(
        view,
        open_mask,
        cell_resources=cell.resources,
        cell=cell,
    )
    batch = agent_action_mask_batch(
        core.agents,
        rows,
        open_mask[np.newaxis, :],
        cells=core.cells,
    )[0]

    oxygen_i = pillars.ACTION_INDEX[ActionType.BUILD_OXYGEN_PLANT]
    solar_i = pillars.ACTION_INDEX[ActionType.BUILD_SOLAR_ARRAY]
    assert scalar[oxygen_i]
    assert batch[oxygen_i]
    assert not scalar[solar_i]
    assert not batch[solar_i]


def test_paid_founder_site_survives_claim_reservation_and_completes():
    """The claim ledger must not charge an already-paid site a second time."""
    world = WorldGenerator(seed=40).generate(32, 20)
    objects = spawn_initial_agents(1, world, seed=40)
    agent = next(iter(objects.values()))
    cell = world.get_cell(agent.x, agent.y)
    for structure_type in (StructureType.GREENHOUSE, StructureType.SHELTER):
        world.add_structure(
            Structure(
                structure_type,
                agent.x,
                agent.y,
                local_x_m=agent.local_x_m,
                local_y_m=agent.local_y_m,
            )
        )
    for resource in agent.inventory.__dataclass_fields__:
        setattr(agent.inventory, resource, 0.0)
        setattr(cell.resources, resource, 0.0)
    agent.inventory.water = 6.0
    agent.inventory.food = 3.0
    cell.construction_sites[
        f"{StructureType.SOLAR_ARRAY.value}@1000:1000"
    ] = 50.0
    agent.pillar_preferences = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    core, _ = build_core_state(world, objects)
    row = core.agents.index[agent.agent_id]
    view = _PreferenceAgentOnViews(core.agents, row, core.side[agent.agent_id])
    view.settle_phase = "out"
    view.settle_target = (view.x, view.y)
    view.founder_kit_reserved = True
    view.health = view.hydration = view.satiety = view.oxygen_level = 1.0
    view.fatigue = 0.0

    config = {
        "seed": 40,
        "agents": {
            "decision_mode": "preferences",
            "decision_sampling": "greedy",
            "decision_engine": "vectorized",
            "cell_proposal_top_k": 5,
        },
        "headless": {"fast_observation": True},
    }
    outcome = kernel.step(core, 1, 7.0, config, None)

    assert outcome.action_results[0]["action"] == ActionType.BUILD_SOLAR_ARRAY.value
    assert core.cells.site_progress[
        view.y, view.x, C.S[StructureType.SOLAR_ARRAY]
    ] < 0.0
    assert core.cells.struct_count[
        view.y, view.x, C.S[StructureType.SOLAR_ARRAY]
    ] == 1


def test_preference_explore_pillar_can_start_bounded_scout_mission():
    world = WorldGenerator(seed=44).generate(32, 20)
    objects = spawn_initial_agents(1, world, seed=44)
    agent = next(iter(objects.values()))
    origin = world.get_cell(agent.x, agent.y)
    for structure_type in (
        StructureType.GREENHOUSE,
        StructureType.HABITAT,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # **E il pozzo (2026-09-01).** L'acqua e' entrata fra i requisiti della
        # capienza vitale: senza, la cella non sostiene nessuno, non e' una
        # «casa sostenuta» e nessuna ricognizione puo' partirne. La prova
        # misurerebbe la mancanza d'acqua invece della partenza della missione.
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, agent.x, agent.y))
    for candidate in world.neighbors(agent.x, agent.y, 3):
        candidate.terrain = origin.terrain
        candidate.polar_severity = 0.0
    agent.pillar_preferences = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    agent.inventory.water = 8.0
    agent.inventory.food = 5.0

    core, vector_world = build_core_state(world, objects)
    rows = core.agents.alive_rows()
    row = int(rows[0])
    view = _PreferenceAgentOnViews(core.agents, row, core.side[agent.agent_id])
    view.actions_taken = 100
    view.settle_next_at = 10_000
    view.scout_next_at = 0
    views = {agent.agent_id: view}
    masks = np.ones(
        (core.cells.H, core.cells.W, pillars.N_ACTIONS), dtype=np.bool_
    )
    priorities = masks.astype(np.float64)
    quotas = np.full(masks.shape, -1, dtype=np.int32)

    requests = decide_batch(
        core.agents,
        core.cells,
        core.side,
        rows,
        masks,
        priorities,
        quotas,
        np.zeros(1, dtype=np.float64),
        "greedy",
        1,
        {(view.x, view.y): [agent.agent_id]},
        world=vector_world,
        agent_views=views,
    )

    request = requests[agent.agent_id]
    assert request.action in {ActionType.MOVE, ActionType.EXPLORE}
    assert request.target is not None
    assert (request.target["x"], request.target["y"]) != (view.x, view.y)
    assert view.scout_phase == "out"


def test_scout_can_return_to_its_saturated_home_cell():
    world = WorldGenerator(seed=45).generate(8, 8)
    objects = spawn_initial_agents(1, world, seed=45)
    agent = next(iter(objects.values()))
    home = (agent.x, agent.y)
    target = world.neighbors(agent.x, agent.y, 1)[0]
    world.move_agent(
        agent.agent_id,
        agent.x,
        agent.y,
        target.x,
        target.y,
        agent.local_x_m,
        agent.local_y_m,
    )
    agent.x, agent.y = target.x, target.y
    agent.scout_phase = "back"
    agent.scout_home = home
    crowded_home = world.get_cell(*home)
    crowded_home.agents_present.extend(f"resident_{index}" for index in range(20))
    request = ActionRequest(
        agent.agent_id,
        ActionType.MOVE,
        target={"x": home[0], "y": home[1]},
    )

    assert _claim_arrival_if_available(agent, request, target, world, {})


def test_founder_cannot_spend_bootstrap_kit_on_transit_build():
    world = WorldGenerator(seed=46).generate(16, 10)
    objects = spawn_initial_agents(1, world, seed=46)
    agent = next(iter(objects.values()))
    core, vector_world = build_core_state(world, objects)
    rows = core.agents.alive_rows()
    row = int(rows[0])
    view = _PreferenceAgentOnViews(core.agents, row, core.side[agent.agent_id])
    view.settle_phase = "out"
    view.settle_target = ((view.x + 2) % core.cells.W, view.y)
    view.founder_kit_reserved = True
    core.agents.pref[row] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    view.inventory.construction_material = 20.0
    view.inventory.minerals = 20.0
    view.inventory.energy = 20.0
    before = view.inventory.to_dict()
    masks = np.ones(
        (core.cells.H, core.cells.W, pillars.N_ACTIONS), dtype=np.bool_
    )
    priorities = masks.astype(np.float64)
    quotas = np.full(masks.shape, -1, dtype=np.int32)

    requests = decide_batch(
        core.agents,
        core.cells,
        core.side,
        rows,
        masks,
        priorities,
        quotas,
        np.zeros(1, dtype=np.float64),
        "greedy",
        1,
        {(view.x, view.y): [agent.agent_id]},
        world=vector_world,
        agent_views={agent.agent_id: view},
    )

    assert requests[agent.agent_id].action not in BUILD_ACTIONS
    assert view.inventory.to_dict() == before


def _engine_state(seed: int = 41):
    world = WorldGenerator(seed=seed).generate(32, 20)
    objects = spawn_initial_agents(30, world, seed=seed)
    return build_core_state(world, objects)[0]


def test_reference_and_vectorized_engines_are_end_to_end_identical():
    for sampling in ("softmax", "greedy"):
        reference = _engine_state()
        vectorized = _engine_state()
        base_agents = {
            "decision_mode": "preferences",
            "decision_sampling": sampling,
        }
        reference_config = {
            "seed": 41,
            "agents": {**base_agents, "decision_engine": "reference"},
            "headless": {"fast_observation": True, "rule_based_observation": True},
        }
        vectorized_config = {
            **reference_config,
            "agents": {**base_agents, "decision_engine": "vectorized"},
        }

        for step_index in range(1, 31):
            kernel.step(reference, step_index, 7.0, reference_config, None)
            kernel.step(vectorized, step_index, 7.0, vectorized_config, None)

        n = reference.agents.n
        assert np.array_equal(reference.agents.x[:n], vectorized.agents.x[:n])
        assert np.array_equal(reference.agents.y[:n], vectorized.agents.y[:n])
        assert np.array_equal(reference.agents.alive[:n], vectorized.agents.alive[:n])
        assert np.array_equal(reference.cells.struct_count, vectorized.cells.struct_count)
        assert np.allclose(
            reference.agents.health[:n], vectorized.agents.health[:n], atol=0, rtol=0
        )
        assert np.allclose(
            reference.agents.inv[:n], vectorized.agents.inv[:n], atol=0, rtol=0
        )


def test_cell_only_reference_and_vectorized_engines_are_identical():
    reference = _engine_state(seed=42)
    vectorized = _engine_state(seed=42)
    base_agents = {
        "decision_mode": "preferences",
        "decision_sampling": "softmax",
        "individual_survival_priority_enabled": False,
    }
    reference_config = {
        "seed": 42,
        "agents": {**base_agents, "decision_engine": "reference"},
        "headless": {"fast_observation": True, "rule_based_observation": True},
    }
    vectorized_config = {
        **reference_config,
        "agents": {**base_agents, "decision_engine": "vectorized"},
    }

    for step_index in range(1, 16):
        kernel.step(reference, step_index, 7.0, reference_config, None)
        kernel.step(vectorized, step_index, 7.0, vectorized_config, None)

    n = reference.agents.n
    assert np.array_equal(reference.agents.x[:n], vectorized.agents.x[:n])
    assert np.array_equal(reference.agents.y[:n], vectorized.agents.y[:n])
    assert np.array_equal(reference.agents.alive[:n], vectorized.agents.alive[:n])
    assert np.array_equal(reference.agents.health[:n], vectorized.agents.health[:n])
    assert np.array_equal(reference.agents.inv[:n], vectorized.agents.inv[:n])


def test_vectorized_binder_skips_local_duplicate_and_assigns_next_builder():
    world = WorldGenerator(seed=43).generate(32, 20)
    objects = spawn_initial_agents(2, world, seed=43)
    first, second = list(objects.values())
    first.local_x_m, first.local_y_m = 1_000.0, 1_000.0
    second.local_x_m, second.local_y_m = 20_000.0, 20_000.0
    cell = world.get_cell(first.x, first.y)
    for resource in ("construction_material", "minerals", "energy", "water"):
        setattr(cell.resources, resource, 100.0)
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
    for agent in objects.values():
        agent.pillar_preferences = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    core, vector_world = build_core_state(world, objects)
    rows = core.agents.alive_rows()
    masks = np.zeros(
        (core.cells.H, core.cells.W, pillars.N_ACTIONS), dtype=np.bool_
    )
    priorities = np.zeros_like(masks, dtype=np.float64)
    quotas = np.zeros_like(masks, dtype=np.int32)
    action_i = pillars.ACTION_INDEX[ActionType.BUILD_GREENHOUSE]
    masks[first.y, first.x, action_i] = True
    priorities[first.y, first.x, action_i] = 1.0
    quotas[first.y, first.x, action_i] = 1
    views = {
        core.agents.ids[int(row)]: _PreferenceAgentOnViews(
            core.agents,
            int(row),
            core.side[core.agents.ids[int(row)]],
        )
        for row in rows
    }

    requests = decide_batch(
        core.agents,
        core.cells,
        core.side,
        rows,
        masks,
        priorities,
        quotas,
        np.zeros(rows.size, dtype=np.float64),
        "greedy",
        1,
        {(first.x, first.y): [first.agent_id, second.agent_id]},
        world=vector_world,
        agent_views=views,
    )

    assert requests[first.agent_id].action != ActionType.BUILD_GREENHOUSE
    assert requests[second.agent_id].action == ActionType.BUILD_GREENHOUSE
