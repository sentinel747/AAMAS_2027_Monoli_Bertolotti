"""Fix dell'audit della run 140726_testlogic5_1500steps (300 fondatori, 960 step).

La run supera tutte le verifiche del terzo audit (0 morti, chat 15.7%,
396 destinatari, power margin 1.5, avamposti autoalimentati) ma espone:

1) ESPANSIONE SOLO PER DISPERSIONE: zero spedizioni di fondazione in 960
   step, perche' il trigger richiedeva anche la saturazione O2 (1/10,
   vincolata ai minerali) che a scala non arriva mai -> valvola di pressione:
   cella 3x oltre la soglia di folla con serre a copertura fonda comunque.
   E l'homestead a soglia 5.0 respingeva i pionieri: il pieno al pozzo (6.0)
   meno la traversata (~1.2) li fa arrivare a ~4.8 -> soglia 4.5.
2) SYSTEM ALERT PERMANENTI IN MEMORIA: gli alert di sovraffollamento/
   inquinamento venivano accodati a recent_events a ogni move e restavano
   anche dopo che la situazione era rientrata o l'agente era altrove ->
   ora sono stato transiente (memory.active_alerts) rinfrescato ogni step
   sulla cella CORRENTE.
3) TOGGLE `world.cell_degradation` (default on): off = niente inquinamento
   da occupazione ne' SYSTEM ALERT.
"""

from src.agents.action_space import ActionType
from src.agents.population import spawn_initial_agents
from src.agents.vitals import refresh_cell_alerts, tick_agent_vitals
from src.experiments.manual_config import ManualConfigOptions, build_manual_config, options_from_scenario_config
from src.world.cell import cell_degradation_enabled, set_cell_degradation
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator

from tests.test_run3_kpi_fixes import _observation


def _colony_agent(seed):
    world = WorldGenerator(seed=seed).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=seed)
    agent = next(iter(agents.values()))
    return world, agent


# ------------------------------------------------- espansione (valvola + homestead)


def _crowded_mother(world, agent, greenhouses):
    cell = world.get_cell(agent.x, agent.y)
    for _ in range(greenhouses):
        world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))
    while len(cell.agents_present) < 45:
        cell.agents_present.append(f"crowd_{len(cell.agents_present)}")
    agent.settle_next_at = 5
    agent.actions_taken = 10
    agent.curiosity = 0.9
    agent.health = 1.0
    agent.fatigue = 0.0
    agent.oxygen_level = 1.0
    agent.inventory.water = 8.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 5.0
    # Exact operational kit covers greenhouse, shelter, solar and O2.
    agent.inventory.construction_material = 13.0
    agent.inventory.minerals = 8.0
    agent.inventory.energy = 2.0
    return cell


def test_pressure_valve_launches_expedition_without_o2_coverage():
    world, agent = _colony_agent(seed=51)
    _crowded_mother(world, agent, greenhouses=7)  # ceil(45/7)=7: serre a copertura

    request = agent._maybe_start_settlement(world, world.get_cell(agent.x, agent.y))

    # Prima del fix: core_covered falso (zero impianti O2) -> mai spedizioni.
    assert request is not None


def test_pressure_valve_still_requires_food_coverage():
    world, agent = _colony_agent(seed=52)
    _crowded_mother(world, agent, greenhouses=2)  # serre NON a copertura

    request = agent._maybe_start_settlement(world, world.get_cell(agent.x, agent.y))

    assert request is None


def test_kit_preparation_gathers_materials_with_short_retry():
    """Il fallimento del kit per i MATERIALI era silenzioso e rimandava di
    ~60 azioni, durante le quali il loop dei cantieri spendeva di nuovo tutto:
    la sonda sul run da 300 fondatori contava 139 fallimenti-kit e zero
    spedizioni. In preparazione-kit ora si raccolgono materiali dedicati e si
    ritenta dopo pochi step."""
    world, agent = _colony_agent(seed=58)
    cell = _crowded_mother(world, agent, greenhouses=7)
    # Non basta a completare atomicamente il kit: resta valido il fallback
    # esplicito di raccolta, con retry breve.
    cell.resources.construction_material = 1.05
    cell.resources.minerals = 10.0
    agent.inventory.construction_material = 0.5  # kit serra non pagabile
    agent.inventory.minerals = 0.5
    before = agent.actions_taken

    request = agent._maybe_start_settlement(world, cell)

    assert request is not None
    assert request.action == ActionType.COLLECT_MATERIALS
    assert "founder kit" in (request.message or "")
    assert agent.settle_next_at <= before + 4  # retry breve, non ~60 azioni
    assert agent.founder_kit_reserved is True


def test_cell_equips_complete_founder_kit_without_weekly_share_actions():
    world, agent = _colony_agent(seed=60)
    cell = _crowded_mother(world, agent, greenhouses=7)
    # **Le quantita' si leggono dal kit, non si scrivono a mano (2026-09-01).**
    # Il kit copre le strutture di avviamento, e quell'elenco e' cresciuto (il
    # pozzo). Una prova che ricopia i totali smette di parlare del travaso e
    # comincia a parlare di quante strutture ci sono, cambiando ogni volta.
    _kit = agent._founder_kit_targets()
    agent.inventory.water = _kit["water"] / 2.0
    agent.inventory.ice = 0.0
    agent.inventory.food = _kit["food"] / 2.0
    agent.inventory.construction_material = 2.0
    agent.inventory.minerals = 1.0
    cell.resources.water = _kit["water"] - agent.inventory.water
    cell.resources.food = _kit["food"] - agent.inventory.food
    cell.resources.construction_material = (
        _kit["construction_material"] - agent.inventory.construction_material
    )
    cell.resources.minerals = _kit["minerals"] - agent.inventory.minerals
    water_before = agent.inventory.water + cell.resources.water
    food_before = agent.inventory.food + cell.resources.food
    materials_before = (
        agent.inventory.construction_material
        + cell.resources.construction_material
    )

    request = agent._maybe_start_settlement(world, cell)

    assert request is not None
    assert "settlement expedition" in (request.message or "")
    assert agent.inventory.water == _kit["water"]
    assert agent.inventory.food == _kit["food"]
    assert agent.inventory.construction_material == _kit["construction_material"]
    assert agent.inventory.minerals == _kit["minerals"]
    assert agent.inventory.energy >= 1.0
    assert agent.inventory.water + cell.resources.water == water_before
    assert agent.inventory.food + cell.resources.food == food_before
    assert (
        agent.inventory.construction_material
        + cell.resources.construction_material
        == materials_before
    )


def test_founder_kit_survives_weekly_material_draw_and_opens_outpost():
    """La sonda end-to-end arrivava sul target con 2.9998 materiali: il
    consumo biologico settimanale aveva eroso il kit esatto da 3.0 e la
    missione terminava con OBSERVE invece di avviare la serra."""
    world, agent = _colony_agent(seed=63)
    origin = _crowded_mother(world, agent, greenhouses=7)
    # Il kit se lo porta gia' addosso: il magazzino vuoto e' cio' che la prova
    # vuole, ma la dotazione personale deve coprire l'avviamento COMPLETO, che
    # dal 2026-09-01 comprende il pozzo.
    for _risorsa, _quanto in agent._founder_kit_targets().items():
        setattr(agent.inventory, _risorsa, max(float(getattr(agent.inventory, _risorsa, 0.0)), _quanto))
    origin.resources.construction_material = 0.0
    origin.resources.minerals = 0.0

    departure = agent._maybe_start_settlement(world, origin)

    assert departure is not None
    assert agent.settle_target is not None
    target_x, target_y = agent.settle_target
    agent.x, agent.y = target_x, target_y
    # Same order of magnitude as kernel_biology's weekly passive draw.
    agent.inventory.construction_material -= 0.01 * 7.0 / 365.0

    founding = agent._continue_settle_out(world)

    assert founding is not None
    assert founding.action == ActionType.BUILD_GREENHOUSE
    assert "founding new settlement" in (founding.message or "")
    assert agent.settle_phase == "out"
    assert agent.founder_kit_reserved is True


def test_founder_reservation_is_released_when_mission_ends():
    world, agent = _colony_agent(seed=59)
    _crowded_mother(world, agent, greenhouses=7)

    request = agent._maybe_start_settlement(
        world, world.get_cell(agent.x, agent.y)
    )

    assert request is not None
    assert agent.founder_kit_reserved is True
    agent._end_settlement()
    assert agent.founder_kit_reserved is False


def test_committed_founder_uses_abort_threshold_not_initial_health_gate():
    world, agent = _colony_agent(seed=61)
    cell = _crowded_mother(world, agent, greenhouses=7)
    # Force a real preparation request: the warehouse cannot provide an
    # atomic kit, while the greenhouse still supports a refill action.
    cell.resources.water = 0.0
    cell.resources.food = 0.0
    cell.resources.construction_material = 0.0
    cell.resources.minerals = 0.0
    agent.inventory.water = 5.0

    preparation = agent._maybe_start_settlement(world, cell)

    assert preparation is not None
    assert preparation.action == ActionType.REFILL_WATER
    assert agent.founder_kit_reserved is True

    # A moderate decline no longer cancels a founder already selected; it is
    # still above the same 0.55 health abort threshold used in transit.
    agent.actions_taken = agent.settle_next_at
    agent.health = 0.65
    # Il kit si legge dal modello: l'elenco delle strutture di avviamento e'
    # cresciuto col pozzo (2026-09-01) e i totali scritti a mano invecchiano.
    for _risorsa, _quanto in agent._founder_kit_targets().items():
        setattr(agent.inventory, _risorsa, _quanto)
    agent.inventory.energy = max(2.0, agent.inventory.energy)
    departure = agent._maybe_start_settlement(world, cell)

    assert departure is not None
    assert "settlement expedition" in (departure.message or "")
    assert agent.settle_phase == "out"


def test_committed_founder_is_released_below_mission_abort_health():
    world, agent = _colony_agent(seed=62)
    cell = _crowded_mother(world, agent, greenhouses=7)
    agent.founder_kit_reserved = True
    agent.health = 0.54

    request = agent._maybe_start_settlement(world, cell)

    assert request is None
    assert agent.founder_kit_reserved is False


def test_pressure_valve_opens_at_three_quarters_coverage():
    # 45 presenti -> fabbisogno 7 serre; 6 serre = 86% >= 75%: la copertura
    # piena e' un bersaglio mobile nella cella che cresce (sonda: gap medio
    # ~10 serre), quindi la valvola chiede i 3/4.
    world, agent = _colony_agent(seed=57)
    _crowded_mother(world, agent, greenhouses=6)

    request = agent._maybe_start_settlement(world, world.get_cell(agent.x, agent.y))

    assert request is not None


def test_homestead_gate_matches_well_topped_canteen():
    world, agent = _colony_agent(seed=53)
    agent.inventory.water = 4.6  # pieno al pozzo (6.0) meno una traversata (~1.2)
    agent.inventory.ice = 0.0
    agent.inventory.food = 5.0   # sopra 4.2: nessuna proposta need-driven
    agent.oxygen_level = 1.0
    agent.inventory.construction_material = 4.0
    agent.inventory.minerals = 2.0
    cell = world.get_cell(agent.x, agent.y)

    request = agent._planned_infrastructure({s.type for s in cell.structures}, _observation(), world)

    # Prima del fix la soglia 5.0 respingeva il pioniere appena arrivato.
    assert request is not None
    assert request.action == ActionType.BUILD_GREENHOUSE


def test_kit_ready_founder_departs_instead_of_building_locally():
    """Inversione di priorita': con la spedizione solo nel fallback movimento,
    chi aveva il kit veniva intercettato dal costruttore a domanda che
    spendeva i materiali nell'ennesimo cantiere locale — tre validazioni da
    400 step con valvola aperta e ZERO partenze. Fondare batte il cantiere
    marginale quando la cella e' 3x sovraffollata."""
    world, agent = _colony_agent(seed=59)
    cell = _crowded_mother(world, agent, greenhouses=7)
    cell.resources.minerals = 0.0
    cell.resources.construction_material = 0.0
    cell.water_ice = 0.0
    cell.resources.ice = 0.0
    agent.satiety = 1.0
    agent.hydration = 1.0
    agent.steps_without_water = 0
    agent.steps_without_food = 0
    # Il kit dell'avviamento e' cresciuto (il pozzo, 2026-09-01): il fondatore
    # se lo porta addosso, cosi' la prova misura l'inversione di priorita' e non
    # la disponibilita' di materiale.
    for _risorsa, _quanto in agent._founder_kit_targets().items():
        setattr(agent.inventory, _risorsa,
                max(float(getattr(agent.inventory, _risorsa, 0.0)), _quanto))

    # agent_count sotto la soglia della previsione idrica (4): il ramo
    # forecast e' un intercettore legittimo e autolimitante (satura a 1/20);
    # qui si testa l'inversione di priorita' col costruttore a domanda.
    # La valvola usa la folla REALE della cella (45), non l'osservazione.
    request = agent.decide(_observation(agent_count=3), world)

    assert "settlement expedition" in (request.message or "") or "founder kit" in (request.message or "")


# ------------------------------------------------------------- SYSTEM ALERT


def test_alert_appears_and_clears_with_the_condition():
    world, agent = _colony_agent(seed=54)
    cell = world.get_cell(agent.x, agent.y)
    while len(cell.agents_present) < 12:
        cell.agents_present.append(f"crowd_{len(cell.agents_present)}")

    tick_agent_vitals(agent, cell, 7)
    assert "overcrowding" in agent.memory.active_alerts
    assert not any("SYSTEM ALERT" in e for e in agent.memory.recent_events)
    assert "SYSTEM ALERT" in agent.memory.summarize()

    del cell.agents_present[1:]  # la folla si disperde
    tick_agent_vitals(agent, cell, 7)
    assert "overcrowding" not in agent.memory.active_alerts
    assert "SYSTEM ALERT" not in agent.memory.summarize()


def test_housing_capacity_clears_false_overcrowding_alert_and_reduces_pollution():
    world, agent = _colony_agent(seed=57)
    housed = world.get_cell(agent.x, agent.y)
    while len(housed.agents_present) < 12:
        housed.agents_present.append(f"crowd_{len(housed.agents_present)}")
    for _ in range(12):
        world.add_structure(
            Structure(StructureType.SHELTER, housed.x, housed.y)
        )

    refresh_cell_alerts(agent, housed)
    assert housed.occupancy_capacity() == 12.0
    assert housed.overcrowding_excess() == 0.0
    assert "overcrowding" not in agent.memory.active_alerts

    unhoused = world.get_cell((housed.x + 1) % world.width, housed.y)
    unhoused.agents_present.extend(f"unhoused_{index}" for index in range(12))
    housed.pollution_risk = 0.0
    unhoused.pollution_risk = 0.0
    housed.update_biology({}, 365.0)
    unhoused.update_biology({}, 365.0)

    assert unhoused.overcrowding_excess() == 2.0
    assert unhoused.pollution_risk > housed.pollution_risk


def test_toggle_off_disables_alerts_and_pollution():
    world, agent = _colony_agent(seed=55)
    cell = world.get_cell(agent.x, agent.y)
    while len(cell.agents_present) < 12:
        cell.agents_present.append(f"crowd_{len(cell.agents_present)}")
    set_cell_degradation(False)
    try:
        assert cell_degradation_enabled() is False
        refresh_cell_alerts(agent, cell)
        assert agent.memory.active_alerts == {}

        cell.pollution_risk = 0.3
        cell.update_biology({}, 365.0)
        assert cell.pollution_risk == 0.0
    finally:
        set_cell_degradation(True)


def test_pollution_grows_by_default_under_occupancy():
    world, agent = _colony_agent(seed=56)
    cell = world.get_cell(agent.x, agent.y)
    while len(cell.agents_present) < 12:
        cell.agents_present.append(f"crowd_{len(cell.agents_present)}")
    cell.pollution_risk = 0.0

    cell.update_biology({}, 365.0)

    assert cell.pollution_risk > 0.0


# ------------------------------------------------------------------- config


def test_cell_degradation_config_round_trip():
    options = ManualConfigOptions(run_name="toggletest", cell_degradation_enabled=False)
    config = build_manual_config(options)
    assert config["world"]["cell_degradation"] is False

    back = options_from_scenario_config(config)
    assert back.cell_degradation_enabled is False

    default_config = build_manual_config(ManualConfigOptions(run_name="toggletest"))
    assert default_config["world"]["cell_degradation"] is True
