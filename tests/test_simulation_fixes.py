"""Regression tests for the six behavior/model fixes driven by the
110726_testlogics run analysis (vitals, construction, scouting, MCD, ODE,
population gates)."""

from src.agents.vitals import LETHAL_STEPS_WITHOUT_WATER
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.action_space import ActionRequest, ActionType, execute_action, validate_action
from src.agents.population import colony_local_habitability, maybe_spawn_agent, spawn_initial_agents
from src.agents.vitals import tick_agent_vitals
from src.model.scientific_overlay import _carbon_cycle_flux
from src.simulation.colony_dynamics import compute_colony_dynamics_metrics
from src.simulation.planetary_coupling import PlanetaryCoupler
from src.social_network.network import SocialNetwork
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _dry_cell(world, x, y):
    cell = world.get_cell(x, y)
    cell.water_ice = 0.0
    cell.liquid_water = 0.0
    cell.resources.ice = 0.0
    return cell


def _strip_water(agent):
    agent.inventory.water = 0.0
    agent.inventory.ice = 0.0


# --- Fix 1: vitals tied to deprivation counters -----------------------------

def test_hydration_tracks_deprivation_and_death_at_zero():
    world = WorldGenerator(seed=1).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=1, start_x=5, start_y=5)
    agent = next(iter(agents.values()))
    _strip_water(agent)
    cell = _dry_cell(world, agent.x, agent.y)

    # La finestra e' letta dalla costante: valeva sette passi (49 giorni a
    # passo settimanale), ora due (quattordici).
    seen = []
    for _ in range(LETHAL_STEPS_WITHOUT_WATER - 1):
        tick_agent_vitals(agent, cell, dt_days=7)
        seen.append(agent.hydration)
    assert agent.steps_without_water == LETHAL_STEPS_WITHOUT_WATER - 1
    assert seen == sorted(seen, reverse=True)
    # Dopo N-1 razioni mancate ne resta 1/N: il clamp del contatore e il
    # consumo per passo ora dicono la stessa cosa.
    atteso = 1.0 / LETHAL_STEPS_WITHOUT_WATER
    assert agent.hydration == pytest.approx(atteso, abs=1e-6)
    assert agent.health > 0.0

    tick_agent_vitals(agent, cell, dt_days=7)
    assert agent.hydration == 0.0
    assert agent.health == 0.0
    assert any("Died from dehydration" in event for event in agent.memory.recent_events)


def test_declining_hydration_makes_sharing_reachable():
    world = WorldGenerator(seed=2).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=2, start_x=5, start_y=5)
    helper = next(iter(agents.values()))
    helper.inventory.water = 2.0
    observation = SimpleNamespace(
        nearby_agents_details=[
            {"agent_id": "agent_999", "name": "Thirsty", "health": 0.9, "hydration": 0.28, "satiety": 0.9}
        ]
    )
    request = helper._nearby_agent_to_help(observation)
    assert request is not None
    assert request.action == ActionType.SHARE_RESOURCE
    assert request.target["resources"] == {"water": 1.0}


def test_water_return_rule_heads_to_water_support():
    world = WorldGenerator(seed=3).generate(12, 12)
    world.add_structure(Structure(StructureType.GREENHOUSE, 5, 5))
    agents = spawn_initial_agents(1, world, seed=3, start_x=8, start_y=5)
    agent = next(iter(agents.values()))
    _strip_water(agent)
    agent.steps_without_water = 4
    cell = _dry_cell(world, agent.x, agent.y)

    request = agent._water_return_action(cell, world)
    assert request is not None
    assert request.action in {ActionType.MOVE, ActionType.EXPLORE}
    assert "dehydration emergency" in (request.message or "")
    assert abs(request.target["x"] - 5) < abs(agent.x - 5)


# --- Fix 2: shared construction sites + rejection cooldown -------------------

def test_construction_site_shared_across_cell():
    world = WorldGenerator(seed=4).generate(10, 10)
    agents = spawn_initial_agents(2, world, seed=4, start_x=5, start_y=5)
    builder, helper = list(agents.values())
    builder.inventory.construction_material = 10.0
    builder.inventory.minerals = 10.0
    builder.inventory.oxygen = 5.0
    builder.inventory.energy = 5.0
    builder.local_x_m, builder.local_y_m = 100.0, 100.0
    helper.local_x_m, helper.local_y_m = 30_000.0, 30_000.0
    helper.inventory.construction_material = 0.0
    helper.inventory.minerals = 0.0

    start = ActionRequest(builder.agent_id, ActionType.BUILD_INFIRMARY)
    assert execute_action(builder, agents, world, start).accepted

    join = ActionRequest(helper.agent_id, ActionType.BUILD_INFIRMARY)
    validation = validate_action(helper, world, join)
    assert validation.accepted  # previously: "insufficient resources for infirmary"

    result = execute_action(helper, agents, world, join)
    assert result.accepted
    final = execute_action(helper, agents, world, join)
    assert final.accepted
    built = [s for s in world.get_cell(5, 5).structures if s.type == StructureType.INFIRMARY]
    assert built and built[0].local_x_m == 100.0  # stands where the site was opened


def test_build_rejection_cooldown_abandons_looping_plan():
    world = WorldGenerator(seed=5).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=5, start_x=5, start_y=5)
    agent = next(iter(agents.values()))
    agent.inventory.construction_material = 0.0
    agent.inventory.minerals = 0.0
    observation = SimpleNamespace(construction_sites={"infirmary@10:10": 33.0}, local_danger=0.0, nearby_agents=[])

    proposal = agent._critical_infrastructure(set(), observation, world)
    assert proposal is not None and proposal.action == ActionType.BUILD_INFIRMARY

    agent.memory.add_failure(ActionType.BUILD_INFIRMARY.value, None, step=int(world.step))
    agent.memory.add_failure(ActionType.BUILD_INFIRMARY.value, None, step=int(world.step))
    assert agent._critical_infrastructure(set(), observation, world) is None


# --- Fix 3: scouting expeditions ---------------------------------------------

def test_scout_mission_starts_when_due_and_stocked():
    world = WorldGenerator(seed=6).generate(12, 12)
    agents = spawn_initial_agents(1, world, seed=6, start_x=6, start_y=6)
    agent = next(iter(agents.values()))
    agent.inventory.water = 5.0
    agent.inventory.food = 5.0
    agent.fatigue = 0.0
    agent.scout_next_at = 0
    agent.actions_taken = 10
    agent.movement_distance_m_per_step = 1.0e9

    cell = world.get_cell(agent.x, agent.y)
    request = agent._scouting_action(world, cell)
    assert request is not None
    assert request.action == ActionType.EXPLORE
    assert agent.scout_phase == "out"
    assert agent.scout_target is not None
    assert agent.scout_home == (6, 6)


def test_scout_mission_aborts_toward_home_when_water_low():
    world = WorldGenerator(seed=7).generate(12, 12)
    agents = spawn_initial_agents(1, world, seed=7, start_x=6, start_y=6)
    agent = next(iter(agents.values()))
    agent.scout_phase = "out"
    agent.scout_target = (9, 9)
    agent.scout_home = (6, 6)
    agent.x, agent.y = 8, 8
    agent.inventory.water = 1.0
    agent.inventory.ice = 0.0

    request = agent._scouting_action(world, world.get_cell(8, 8))
    assert agent.scout_phase == "back"
    assert request is not None
    assert "returning to colony" in (request.message or "")


# --- Tuning: infirmary per-capita cap -----------------------------------------

def test_infirmary_build_capped_per_capita():
    world = WorldGenerator(seed=11).generate(10, 10)
    world.step = 10  # oltre la finestra "early infrastructure"
    agents = spawn_initial_agents(1, world, seed=11, start_x=5, start_y=5)
    agent = next(iter(agents.values()))
    agent.inventory.oxygen = 5.0
    agent.inventory.energy = 5.0
    cell = world.get_cell(5, 5)
    world.add_structure(Structure(StructureType.INFIRMARY, 5, 5, local_x_m=50_000.0, local_y_m=50_000.0))
    cell.agents_present.extend(f"ghost_{i}" for i in range(10))  # ~11 colonists, 1 infirmary: saturated

    assert agent._first_buildable([StructureType.INFIRMARY], set(), world=world, cell=cell) is None

    cell.agents_present.extend(f"extra_{i}" for i in range(40))  # ~51 colonists: cap rises to 3
    proposal = agent._first_buildable([StructureType.INFIRMARY], set(), world=world, cell=cell)
    assert proposal is not None and proposal.action == ActionType.BUILD_INFIRMARY


def test_executor_refuses_new_site_of_saturated_type():
    world = WorldGenerator(seed=15).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=15, start_x=5, start_y=5)
    agent = next(iter(agents.values()))
    agent.inventory.oxygen = 5.0
    agent.inventory.energy = 5.0
    cell = world.get_cell(5, 5)
    world.add_structure(Structure(StructureType.INFIRMARY, 5, 5, local_x_m=50_000.0, local_y_m=50_000.0))
    cell.agents_present.extend(f"ghost_{i}" for i in range(10))  # cap = 1, gia' coperto

    result = validate_action(agent, world, ActionRequest(agent.agent_id, ActionType.BUILD_INFIRMARY))
    assert not result.accepted
    assert "coverage already sufficient" in result.message


def test_preference_build_uses_proposal_time_population_for_coverage():
    """Movers executed earlier in the step must not invalidate a cell menu."""
    world = WorldGenerator(seed=122).generate(12, 8)
    agents = spawn_initial_agents(1, world, seed=122, start_x=6, start_y=4)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.agents_present.extend(f"proposal_peer_{i}" for i in range(7))
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))
    agent.inventory.construction_material = 10.0
    agent.inventory.minerals = 10.0
    agent.inventory.water = 10.0
    # The proposal saw eight residents (two greenhouses required).  Before
    # validation, seven movers leave and the live count falls to one.
    proposal_population = len(cell.agents_present)
    cell.agents_present[:] = [agent.agent_id]

    result = validate_action(
        agent,
        world,
        ActionRequest(
            agent.agent_id,
            ActionType.BUILD_GREENHOUSE,
            target={"coverage_population": proposal_population},
        ),
    )

    assert result.accepted

    # Un cantiere gia' aperto resta continuabile anche a tipo saturo.
    cell.construction_sites["infirmary@10:10"] = 33.0
    result = validate_action(agent, world, ActionRequest(agent.agent_id, ActionType.BUILD_INFIRMARY))
    assert result.accepted


def test_weather_station_capped_per_capita():
    world = WorldGenerator(seed=16).generate(10, 10)
    world.step = 10
    agents = spawn_initial_agents(1, world, seed=16, start_x=5, start_y=5)
    agent = next(iter(agents.values()))
    agent.inventory.energy = 5.0
    cell = world.get_cell(5, 5)
    world.add_structure(Structure(StructureType.WEATHER_STATION, 5, 5, local_x_m=50_000.0, local_y_m=50_000.0))
    cell.agents_present.extend(f"ghost_{i}" for i in range(10))  # ~11 coloni: cap 1, saturo

    assert agent._first_buildable([StructureType.WEATHER_STATION], set(), world=world, cell=cell) is None


# --- Tuning: scout return logistics --------------------------------------------

def test_scout_turns_back_with_only_return_rations():
    world = WorldGenerator(seed=12).generate(14, 14)
    agents = spawn_initial_agents(1, world, seed=12, start_x=6, start_y=6)
    agent = next(iter(agents.values()))
    agent.scout_phase = "out"
    agent.scout_target = (11, 6)
    agent.scout_home = (6, 6)
    agent.x, agent.y = 9, 6  # 3 celle da casa: servono (3+1)*12*0.1 = 4.8 razioni
    agent.inventory.water = 3.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 5.0

    request = agent._scouting_action(world, world.get_cell(9, 6))
    assert agent.scout_phase == "back"
    assert request is not None and "returning to colony" in (request.message or "")


def test_scout_target_range_limited_by_carried_water():
    world = WorldGenerator(seed=13).generate(14, 14)
    agents = spawn_initial_agents(1, world, seed=13, start_x=6, start_y=6)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell_diagonal = math.hypot(
        float(cell.geometry["width_m"]),
        float(cell.geometry["height_m"]),
    )
    agent.movement_distance_m_per_step = cell_diagonal / 12.0
    agent.inventory.water = 4.0  # (4-1)/2.4 -> raggio 1
    agent.inventory.ice = 0.0
    target = agent._pick_scout_target(world)
    assert target is not None
    assert max(abs(target.x - 6), abs(target.y - 6)) <= 1


def _seed_operational_scout_base(world, x: int, y: int, resident: str | None = None):
    if resident is not None:
        world.place_agent(resident, x, y)
    for structure_type in (
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, x, y))
    return world.get_cell(x, y)


def test_scout_relay_hands_frontier_mission_to_operational_outpost():
    world = WorldGenerator(seed=130).generate(40, 24)
    agents = spawn_initial_agents(
        1,
        world,
        seed=130,
        start_x=15,
        start_y=12,
    )
    agent = next(iter(agents.values()))
    current = _seed_operational_scout_base(world, agent.x, agent.y)
    outpost = _seed_operational_scout_base(
        world,
        agent.x + 2,
        agent.y,
        "relay_resident",
    )
    for cell in [current, *world.neighbors(current.x, current.y, 3)]:
        world.mark_explored(cell.x, cell.y)
    agent.inventory.water = 8.0
    agent.inventory.food = 4.0
    agent.movement_distance_m_per_step = 1.0e9
    agent.scout_next_at = 0
    agent.actions_taken = 10
    agent.fatigue = 0.0

    request = agent._scouting_action(world, current)

    assert request is not None and request.action == ActionType.EXPLORE
    assert agent.scout_phase == "out"
    assert agent.scout_home == (outpost.x, outpost.y)
    target = world.get_cell(*agent.scout_target)
    assert not target.explored
    assert any(
        neighbor.explored
        for neighbor in world.neighbors(target.x, target.y, 1)
    )
    assert "relay scouting" in (request.message or "")

    # Execute the bounded state machine: the scout surveys the target and
    # finishes at the forward outpost, making it the base for later relays.
    for step in range(1, 40):
        world.step = step
        result = execute_action(agent, agents, world, request)
        assert result.accepted
        request = agent._scouting_action(
            world,
            world.get_cell(agent.x, agent.y),
        )
        if request is None and agent.scout_phase is None:
            break
    assert agent.scout_phase is None
    assert (agent.x, agent.y) == (outpost.x, outpost.y)


def test_scout_does_not_revisit_when_shared_frontier_is_exhausted():
    world = WorldGenerator(seed=131).generate(24, 20)
    agents = spawn_initial_agents(
        1,
        world,
        seed=131,
        start_x=12,
        start_y=10,
    )
    agent = next(iter(agents.values()))
    current = _seed_operational_scout_base(world, agent.x, agent.y)
    for cell in [current, *world.neighbors(current.x, current.y, 3)]:
        world.mark_explored(cell.x, cell.y)
    agent.inventory.water = 8.0
    agent.inventory.food = 4.0
    agent.movement_distance_m_per_step = 1.0e9
    agent.scout_next_at = 0
    agent.actions_taken = 10
    agent.fatigue = 0.0

    request = agent._scouting_action(world, current)

    assert request is None
    assert agent.scout_phase is None
    assert agent.scout_target is None


def test_scout_preparation_tops_up_water_and_food():
    world = WorldGenerator(seed=14).generate(10, 10)
    world.add_structure(Structure(StructureType.GREENHOUSE, 5, 5))
    agents = spawn_initial_agents(1, world, seed=14, start_x=5, start_y=5)
    agent = next(iter(agents.values()))
    agent.scout_next_at = 0
    agent.actions_taken = 10
    agent.fatigue = 0.0
    agent.movement_distance_m_per_step = 1.0e9
    cell = world.get_cell(5, 5)

    agent.inventory.water = 5.0  # sotto il target 7.0 con supporto idrico presente
    request = agent._scouting_action(world, cell)
    assert request is not None and request.action == ActionType.REFILL_WATER

    agent.inventory.water = 8.0
    agent.inventory.food = 2.0  # sotto il target cibo 4.0 con serra presente
    request = agent._scouting_action(world, cell)
    assert request is not None and request.action == ActionType.FORAGE

    agent.inventory.food = 5.0
    request = agent._scouting_action(world, cell)
    assert request is not None and request.action == ActionType.EXPLORE
    assert agent.scout_phase == "out"


def test_scout_does_not_launch_when_subcell_range_cannot_cover_round_trip():
    world = WorldGenerator(seed=132).generate(360, 180)
    agents = spawn_initial_agents(
        1,
        world,
        seed=132,
        start_x=180,
        start_y=90,
        operational_range_m=100.0,
    )
    agent = next(iter(agents.values()))
    current = _seed_operational_scout_base(world, agent.x, agent.y)
    assert agent.perception_radius_m == agent.movement_distance_m_per_step == 100.0
    agent.inventory.water = 8.0
    agent.inventory.food = 4.0
    agent.scout_next_at = 0
    agent.actions_taken = 10
    agent.fatigue = 0.0

    request = agent._scouting_action(world, current)

    assert request is None
    assert agent.scout_phase is None
    assert agent._movement_steps_per_cell(world) > 500


def test_scout_progress_scales_with_ten_km_operational_range():
    world = WorldGenerator(seed=133).generate(360, 180)
    agents = spawn_initial_agents(
        1,
        world,
        seed=133,
        start_x=180,
        start_y=90,
        operational_range_m=10_000.0,
    )
    agent = next(iter(agents.values()))
    current = _seed_operational_scout_base(world, agent.x, agent.y)
    assert (
        agent.perception_radius_m
        == agent.movement_distance_m_per_step
        == 10_000.0
    )
    outpost = _seed_operational_scout_base(
        world,
        agent.x + 2,
        agent.y,
        "ten_km_relay_resident",
    )
    for cell in [current, *world.neighbors(current.x, current.y, 3)]:
        world.mark_explored(cell.x, cell.y)
    agent.inventory.water = 20.0
    agent.inventory.food = 20.0
    agent.scout_next_at = 0
    agent.actions_taken = 10
    agent.fatigue = 0.0

    request = agent._scouting_action(world, current)
    assert agent.scout_home == (outpost.x, outpost.y)
    assert "relay scouting" in (request.message or "")
    distances = []
    visited_other_cell = False
    for step in range(1, 140):
        assert request is not None
        world.step = step
        result = execute_action(agent, agents, world, request)
        assert result.accepted
        if "distance_m" in result.data:
            distances.append(float(result.data["distance_m"]))
        visited_other_cell |= (agent.x, agent.y) != (current.x, current.y)
        request = agent._scouting_action(
            world,
            world.get_cell(agent.x, agent.y),
        )
        if request is None and agent.scout_phase is None:
            break

    assert visited_other_cell
    assert agent.scout_phase is None
    assert (agent.x, agent.y) == (outpost.x, outpost.y)
    assert distances
    assert max(distances) <= 10_000.0 + 1e-6
    assert len(distances) > 2


# --- Fix 5: carbon cycle and baseline atmosphere ------------------------------

def test_no_respiration_without_living_biosphere():
    assert _carbon_cycle_flux(co2=100.0, o2=5.0, vegetation_cover=0.6, suitability=0.0, timestep=7.0) == 0.0


def test_respiration_capped_to_fraction_of_o2():
    flux = _carbon_cycle_flux(co2=0.0, o2=4.0, vegetation_cover=1.0, suitability=1.0, timestep=1000.0)
    assert flux == pytest.approx(-1.0)  # never more than 25% of the O2 pool


def test_planetary_coupler_seeds_baseline_atmosphere():
    coupler = PlanetaryCoupler({"climate": {"enabled": False}, "simulation": {"days_per_step": 7}})
    values = coupler.state.values
    assert values["PressioneAtmosferica"] > 500.0
    assert values["CO2"] > 400.0
    assert values["O2"] > 0.0


# --- Fix 4: solar power follows the climate flux -------------------------------

def _world_with_power_colony(seed=8):
    world = WorldGenerator(seed=seed).generate(10, 10)
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, 5, 5))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, 5, 5))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, 5, 5))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, 5, 5))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, 5, 5))
    agents = spawn_initial_agents(2, world, seed=seed, start_x=5, start_y=5)
    return world, agents


def test_power_margin_segue_la_produzione_e_non_una_seconda_formula():
    """Il margine e' il rapporto FISICO fra produzione e carico.

    Fino al 2026-08-25 questo test verificava che `power_margin` scalasse col
    flusso solare del layer climatico. Nessuna energia della simulazione
    dipende pero' da quel flusso: la produzione e'
    `struct_fx[energy] x RESA_ENERGIA x solar_yield(dust)`, e il flusso tocca
    soltanto la radiazione. Il test fissava una proprieta' di una grandezza di
    sola visualizzazione, ed era proprio il punto in cui metrica e fisica
    divergevano.

    Ora la metrica usa la stessa `solar_yield` della produzione, e il test lo
    verifica dove conta: alzando la polvere delle celle che ospitano i
    pannelli, il margine deve scendere esattamente come scende la resa.
    """
    from src.simulation.step_effects import solar_yield

    world, agents = _world_with_power_colony()
    social = SocialNetwork()
    world.planetary_state = {}
    # Carico abbastanza alto da tenere il margine sotto il tetto di 1,5 in
    # entrambe le misure: contro il tetto il rapporto non sarebbe piu' leggibile.
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, 5, 5))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, 5, 5))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, 5, 5))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, 5, 5))

    cella = world.get_cell(5, 5)
    cella.dust_level = 0.35  # la polvere tipica: resa piena
    pieno = compute_colony_dynamics_metrics(agents, world, {}, social)
    assert solar_yield(cella.dust_level) == pytest.approx(1.0)

    cella.dust_level = 0.80  # tempesta: la resa crolla
    offuscato = compute_colony_dynamics_metrics(agents, world, {}, social)

    atteso = solar_yield(0.80) / solar_yield(0.35)
    assert offuscato["power_margin"] == pytest.approx(pieno["power_margin"] * atteso)
    assert offuscato["power_margin"] < pieno["power_margin"]


def test_power_margin_scende_sotto_uno_quando_il_carico_supera_i_pannelli():
    """Un margine che non puo' scendere sotto 1 non e' un margine.

    La formula precedente normalizzava su `max(1.0, carico)`: un carico minimo
    di 1 che non esiste in nessun luogo del modello, e che rendeva impossibile
    riportare un deficit su una colonia piccola.
    """
    world, agents = _world_with_power_colony()
    social = SocialNetwork()
    world.planetary_state = {}
    con_pannello = compute_colony_dynamics_metrics(agents, world, {}, social)

    senza = WorldGenerator(seed=8).generate(10, 10)
    senza.add_structure(Structure(StructureType.OXYGEN_PLANT, 5, 5))
    senza.add_structure(Structure(StructureType.OXYGEN_PLANT, 5, 5))
    agenti2 = spawn_initial_agents(2, senza, seed=8, start_x=5, start_y=5)
    senza.planetary_state = {}
    esito = compute_colony_dynamics_metrics(agenti2, senza, {}, SocialNetwork())

    assert esito["power_margin"] == pytest.approx(0.0)
    assert con_pannello["power_margin"] > esito["power_margin"]


# --- Fix 4: MCD netcdf sampling ------------------------------------------------

def test_netcdf_diurnal_solar_flux_is_positive():
    from src.data.mars_climate import _find_netcdf_root, _sample_netcdf

    root = _find_netcdf_root(Path("data/mcd_runtime"), "climatology")
    if root is None:
        pytest.skip("MCD netcdf archive not available")
    sample = _sample_netcdf(root, "climatology", 152.0, -4.6, 137.4, 12.0)
    assert sample is not None
    assert sample.solar_flux_w_m2 > 10.0  # was exactly 0.0 for the whole run
    assert sample.pressure_pa > 0.0


# --- Fix 6: growth gated on colony-local habitability --------------------------

def test_colony_local_habitability_uses_structure_cells():
    world = WorldGenerator(seed=9).generate(10, 10)
    world.add_structure(Structure(StructureType.HABITAT, 5, 5))
    world.get_cell(5, 5).habitability_score = 0.5
    assert colony_local_habitability(world) == pytest.approx(0.5)


def test_population_growth_possible_with_low_planetary_average():
    world = WorldGenerator(seed=10).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=10, start_x=5, start_y=5)
    agent = next(iter(agents.values()))
    agent.inventory.food = 10.0
    agent.inventory.water = 10.0
    agent.inventory.construction_material = 10.0
    agent.inventory.tools = 4.0
    world.add_structure(Structure(StructureType.GREENHOUSE, 5, 5))
    world.add_structure(Structure(StructureType.HABITAT, 5, 5))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, 5, 5))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, 5, 5))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, 5, 5))
    world.get_cell(5, 5).habitability_score = 0.5
    # Il corredo del neonato lo paga la CELLA (2026-08-30): senza scorte in
    # magazzino la colonia non potrebbe mantenerlo e la nascita sarebbe vietata.
    _mag = world.get_cell(5, 5).resources
    _mag.food, _mag.water, _mag.oxygen = 50.0, 40.0, 20.0
    # Planet-wide mean is ~0 (one good cell out of 100), but the colony cell
    # is habitable: growth must be possible.
    config = {
        "population": {
            "enabled": True,
            "max_agents": 3,
            "daily_spawn_probability": 1.0,
            "prosperity_growth_scale": 1.0,
            "min_habitability_for_growth": 0.02,
        },
        "simulation": {"days_per_step": 7},
    }
    spawned = maybe_spawn_agent(agents, world, config, day=1, seed=10)
    assert spawned is not None
