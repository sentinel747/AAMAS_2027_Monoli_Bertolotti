from __future__ import annotations

import copy
import random

import numpy as np

from src.agents.action_space import execute_action
from src.agents.rule_based_agent import RuleBasedAgent
from src.agents.vitals import tick_all_agents
from src.api.state_store import _full_biology_update_enabled
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.kernel import CoreState, step
from src.core.views import AgentSideState
from src.simulation.biology_update import update_biology_cells
from src.simulation.extreme_events import ExtremeEventEngine
from src.simulation.step_effects import (
    apply_colony_resource_feedback,
    apply_structure_effects,
    apply_structure_wear,
)
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.mars_geometry import cell_geometry
from src.world.perception import observe, observe_fast, observe_rule_based
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType

VITAL_COLS = ("health", "oxygen", "hydration", "satiety", "fatigue",
              "stress", "morale", "cooperation", "compliance")
OBJ_ATTR = {"oxygen": "oxygen_level", "stress": "stress_index", "compliance": "protocol_compliance"}


def _grid(width: int, height: int) -> GridWorld:
    # Task 12: CellView.geometry now computes the same real geometry
    # WorldGenerator populates in production (a pure function of
    # x/y/width/height, src.world.mars_geometry.cell_geometry - see
    # src/core/views.py's docstring), instead of only ever carrying
    # `area_m2`. Setting the same real geometry on these hand-built object
    # cells keeps this module's object-vs-vector equivalence checks (MOVE/
    # EXPLORE geometry-dependent resolution in action_space.py:182-256)
    # meaningful, instead of relying on "both sides leave geometry empty"
    # being incidentally true.
    cells = []
    for y in range(height):
        row = []
        for x in range(width):
            cell = Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN)
            geom = cell_geometry(x, y, width, height)
            cell.geometry = {
                "center_lat_deg": geom.center_lat_deg,
                "center_lon_deg": geom.center_lon_deg,
                "width_m": geom.width_m,
                "height_m": geom.height_m,
                "area_m2": geom.area_m2,
                "area_km2": geom.area_km2,
            }
            row.append(cell)
        cells.append(row)
    return GridWorld(width=width, height=height, cells=cells)


def _agent_specs():
    # Fixed, deterministic per-agent parameters (no RNG at construction time,
    # so the two builds below - object side and array side - start byte-for-byte
    # identical without any deep-copy of mutable AgentMemory instances).
    return [
        {"agent_id": "a0", "name": "a0", "role": "colonist", "x": 2, "y": 2,
         "perception_radius": 2, "survival_priority": 0.80},
        {"agent_id": "a1", "name": "a1", "role": "colonist", "x": 2, "y": 2,
         "perception_radius": 2, "survival_priority": 0.75},
        {"agent_id": "a2", "name": "a2", "role": "colonist", "x": 1, "y": 1,
         "perception_radius": 1, "survival_priority": 0.85},
        {"agent_id": "a3", "name": "a3", "role": "colonist", "x": 4, "y": 3,
         "perception_radius": 3, "survival_priority": 0.60},
    ]


def _build_world() -> GridWorld:
    w = _grid(6, 5)
    home = w.get_cell(2, 2)
    for st in (StructureType.HABITAT, StructureType.GREENHOUSE, StructureType.SOLAR_ARRAY, StructureType.OXYGEN_PLANT):
        w.add_structure(Structure(type=st, x=2, y=2, integrity=1.0))
    home.water_ice, home.liquid_water = 4.0, 0.6
    home.nutrients.update({"N": 5.0, "P": 1.0, "C": 3.0})
    w.get_cell(1, 1).water_ice = 1.5
    w.get_cell(4, 3).radiation_level = 0.5
    w.get_cell(4, 3).dust_level = 0.2
    w.planetary_state.update({
        "mean_temperature_c": -55.0,
        "pressure_pa": 650.0,
        "liquid_water_stability": 0.2,
    })
    return w


def _build_object_scenario():
    w = _build_world()
    agents: dict[str, RuleBasedAgent] = {}
    for spec in _agent_specs():
        a = RuleBasedAgent(**spec)
        agents[a.agent_id] = a
        w.place_agent(a.agent_id, a.x, a.y)
    return w, agents


def _build_array_scenario():
    # A SEPARATE set of RuleBasedAgent instances (own AgentMemory objects) fed
    # only into AgentArrays.from_agents/AgentSideState.from_agent - never
    # touched again, so there is no aliasing with the object-side agents.
    w = _build_world()
    agents_src: dict[str, RuleBasedAgent] = {}
    for spec in _agent_specs():
        a = RuleBasedAgent(**spec)
        agents_src[a.agent_id] = a
    aa = AgentArrays.from_agents(agents_src)
    ca = CellArrays.from_world(w)
    side = {aid: AgentSideState.from_agent(a) for aid, a in agents_src.items()}
    ps = dict(w.planetary_state)
    return aa, ca, side, ps


BASE_CONFIG = {
    "headless": {"fast_observation": True, "rule_based_observation": True},
    "model": {"psychosocial_enabled": True},
    "social": {"earth_mars_delay_minutes": 12.0},
    "world": {"cell_degradation": True},
    "seed": 123,
}


def _object_step(world, agents, config, step_index, dt_days, events_engine):
    world.step = step_index
    world.metadata["active_event"] = events_engine.active_event
    world.metadata["upcoming_event"] = events_engine.upcoming_event

    use_fast = bool(config.get("headless", {}).get("fast_observation", True))
    use_rule = bool(config.get("headless", {}).get("rule_based_observation", True))

    agents_list = list(agents.values())
    observations = []
    for agent in agents_list:
        if use_fast:
            observer = observe_fast
        elif use_rule and getattr(agent, "mode", "rule_based") != "llm":
            observer = observe_rule_based
        else:
            observer = observe
        observations.append(observer(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents))

    # NOTE (documented simplification, see test module docstring): with an
    # all-rule_based population, `_decide_with_llm_governor` reduces exactly
    # to a direct `agent.decide(obs, world)` call per agent (verified against
    # agent_coupled_runner.py:324-347/state_store.py's twin) - calling it
    # directly here avoids standing up asyncio plumbing that changes nothing
    # for this population.
    requests = [agent.decide(obs, world) for agent, obs in zip(agents_list, observations)]

    for agent, request in zip(agents_list, requests):
        execute_action(agent, agents, world, request)

    tick_all_agents(agents_list, world, dt_days, config=config)

    dead_ids = [aid for aid, a in agents.items() if a.health <= 0]
    for dead_id in dead_ids:
        agent = agents[dead_id]
        # **Le scorte del morto restano nella cella (2026-08-28).** Specchio di
        # `kernel_vitals._lascia_le_scorte`: prima l'inventario spariva con
        # l'agente, ed essendo la stessa omissione su entrambi i lati la parita'
        # non poteva accorgersene. E' il limite di un oracolo differenziale: due
        # motori d'accordo su una massa che evapora restano d'accordo.
        cella_morte = world.get_cell(agent.x, agent.y)
        for nome_risorsa in C.R:
            portato = float(getattr(agent.inventory, nome_risorsa, 0.0) or 0.0)
            if portato:
                setattr(
                    cella_morte.resources,
                    nome_risorsa,
                    float(getattr(cella_morte.resources, nome_risorsa, 0.0)) + portato,
                )
                setattr(agent.inventory, nome_risorsa, 0.0)
        world.remove_agent(dead_id, agent.x, agent.y)
        del agents[dead_id]

    # Important 3 (review round 3): both real engines stop HERE, before
    # spawn/biology/structure-effects/events, the step the population hits
    # zero (agent_coupled_runner.py:240-244 `break`s the run loop;
    # state_store.py:431-437 `return`s from `step()`). This reference helper
    # used to omit that guard entirely - it would happily keep running
    # biology/wear/events on an empty `agents` dict, which is harmless code
    # -wise (nothing in that tail actually requires a nonempty `agents`) but
    # NOT what either engine does, so it could never expose kernel.py's
    # missing extinction guard (Important 3): a "does the kernel's cell state
    # freeze at the same point as the object model's?" test needs a
    # reference that itself freezes there.
    if not agents:
        return False

    planetary_metrics = dict(world.planetary_state)
    # Important 1 (review round 3): this used to hardcode full_grid=True,
    # which is NOT what either engine does - both call
    # `_full_biology_update_enabled(config)` here (agent_coupled_runner.py:254,
    # state_store.py:447), which defaults to the "active" scope
    # (`full_grid=False`) unless `config["headless"]["biology_update_scope"]`
    # is explicitly "full". Hardcoding True made this reference function
    # justify itself to kernel_biology.update_cells's own (then-unconditional)
    # whole-grid behavior instead of to the real engines, hiding the exact
    # divergence kernel.py's `full_grid_biology` now closes (a bare cell's
    # habitability recomputes to a nonzero value under full_grid=True and
    # stays frozen at 0 under the "active" scope both engines actually use
    # by default).
    update_biology_cells(world, planetary_metrics, dt_days, full_grid=_full_biology_update_enabled(config))
    apply_structure_effects(world)
    apply_colony_resource_feedback(world, agents, dt_days)
    apply_structure_wear(world, dt_days)

    events_engine.advance(world, agents, config, step_index)
    return True


def _strip_life_support(world, agents, doomed_ids: frozenset[str]) -> None:
    """Toglie dalle celle dei condannati tutto cio' che li terrebbe in vita.

    **Perche' serve (2026-08-25).** Il test presupponeva che idratazione zero
    e riserve vuote bastassero a garantire la morte. Non e' piu' cosi', e la
    ragione e' una correzione del modello e non un difetto: dopo la
    normalizzazione del supporto per occupante e la ritaratura dei
    coefficienti, una cella provvista di impianti reidrata chi vi arriva a
    secco — cosa che una colonia attrezzata deve saper fare. Misurato: a0 e a1
    stanno su (2,2), che offre 0,125 di abitabilita' e 0,065 di acqua per
    occupante, e con quelle sopravvivono.

    La premessa del test torna vera togliendo il supporto, non abbassando
    l'asticella: chi e' condannato deve trovarsi davvero senza nulla.
    """
    for aid in doomed_ids:
        agente = agents.get(aid)
        if agente is None:
            continue
        cella = world.get_cell(agente.x, agente.y)
        cella.structures.clear()
        cella.liquid_water = 0.0
        cella.water_ice = 0.0


def _make_lethal_agents(doomed_ids: frozenset[str]) -> dict[str, RuleBasedAgent]:
    """Same 4-agent population as `_agent_specs`, but every id in `doomed_ids`
    starts at hydration=0.0 with no water/ice reserves. Le loro celle vengono
    poi spogliate da `_strip_life_support`, senza il quale un impianto idrico
    li rimetterebbe in piedi."""
    agents: dict[str, RuleBasedAgent] = {}
    for spec in _agent_specs():
        a = RuleBasedAgent(**spec)
        if spec["agent_id"] in doomed_ids:
            a.hydration = 0.0
            a.inventory.water = 0.0
            a.inventory.ice = 0.0
        agents[a.agent_id] = a
    return agents


def _build_lethal_object_scenario(doomed_ids: frozenset[str]):
    w = _build_world()
    agents = _make_lethal_agents(doomed_ids)
    _strip_life_support(w, agents, doomed_ids)
    for a in agents.values():
        w.place_agent(a.agent_id, a.x, a.y)
    return w, agents


def _build_lethal_array_scenario(doomed_ids: frozenset[str]):
    w = _build_world()
    agents_src = _make_lethal_agents(doomed_ids)
    _strip_life_support(w, agents_src, doomed_ids)
    aa = AgentArrays.from_agents(agents_src)
    ca = CellArrays.from_world(w)
    side = {aid: AgentSideState.from_agent(a) for aid, a in agents_src.items()}
    ps = dict(w.planetary_state)
    return aa, ca, side, ps


def _assert_agents_and_cells_match(agents_obj, w_obj, aa, ca, atol, context=""):
    """Shared comparison core, factored out of what used to be a single
    end-of-run block in test_step_matches_object_engine_short_horizon so
    every equivalence test in this module (including the per-step loop
    below) can call it after each step with a step-tagged `context` -
    localizing a failure to the step it first appeared on instead of only
    ever reporting the final state."""
    prefix = f"{context}: " if context else ""
    for aid, a in agents_obj.items():
        row = aa.index[aid]
        assert aa.alive[row], f"{prefix}{aid} should be alive"
        assert int(aa.x[row]) == a.x, f"{prefix}{aid}.x: obj={a.x} vec={aa.x[row]}"
        assert int(aa.y[row]) == a.y, f"{prefix}{aid}.y: obj={a.y} vec={aa.y[row]}"
        obj_inv = a.inventory.to_dict()
        for res, idx in C.R.items():
            assert np.isclose(aa.inv[row, idx], obj_inv[res], atol=atol), (
                f"{prefix}{aid}.inv[{res}]: obj={obj_inv[res]} vec={aa.inv[row, idx]}")
        for col in VITAL_COLS:
            obj_val = getattr(a, OBJ_ATTR.get(col, col))
            assert np.isclose(getattr(aa, col)[row], obj_val, atol=atol), (
                f"{prefix}{aid}.{col}: obj={obj_val} vec={getattr(aa, col)[row]}")

    for y, row_cells in enumerate(w_obj.cells):
        for x, cell in enumerate(row_cells):
            obj_res = cell.resources.to_dict()
            for res, idx in C.R.items():
                assert np.isclose(ca.cell_res[y, x, idx], obj_res[res], atol=atol), (
                    f"{prefix}cell({x},{y}).resources.{res}: obj={obj_res[res]} vec={ca.cell_res[y, x, idx]}")
            assert np.isclose(ca.habitability[y, x], cell.habitability_score, atol=atol), (
                f"{prefix}cell({x},{y}).habitability: obj={cell.habitability_score} vec={ca.habitability[y, x]}")
            counts = {}
            integrities = {}
            for s in cell.structures:
                counts[s.type] = counts.get(s.type, 0) + 1
                integrities[s.type] = integrities.get(s.type, 0.0) + s.integrity
            for st, si in C.S.items():
                assert int(ca.struct_count[y, x, si]) == counts.get(st, 0), (
                    f"{prefix}cell({x},{y}).struct_count[{st}]: obj={counts.get(st, 0)} vec={ca.struct_count[y, x, si]}")
                assert np.isclose(ca.struct_integrity[y, x, si], integrities.get(st, 0.0), atol=atol), (
                    f"{prefix}cell({x},{y}).struct_integrity[{st}]: obj={integrities.get(st, 0.0)} vec={ca.struct_integrity[y, x, si]}")


def test_step_smoke_no_crash_no_deaths_state_evolves():
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    rng = np.random.default_rng(0)
    random.seed(7)

    x0 = aa.x[: aa.n].copy()
    y0 = aa.y[: aa.n].copy()
    inv0 = aa.inv[: aa.n].copy()

    for i in range(1, 21):
        out = step(state, i, 3.0, BASE_CONFIG, rng)
        assert out.deaths == []

    assert aa.alive_rows().size == aa.n
    # state actually evolved: inventories or positions moved somewhere over 20
    # steps of real decide()/execute_action()/vitals/biology activity.
    assert not (np.array_equal(x0, aa.x[: aa.n]) and np.array_equal(y0, aa.y[: aa.n])
                and np.allclose(inv0, aa.inv[: aa.n]))


def test_step_determinism_same_seed_same_trajectory():
    def _run():
        aa, ca, side, ps = _build_array_scenario()
        state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
        rng = np.random.default_rng(0)
        random.seed(42)
        for i in range(1, 16):
            step(state, i, 3.0, BASE_CONFIG, rng)
        return aa, ca

    aa1, ca1 = _run()
    aa2, ca2 = _run()

    assert np.array_equal(aa1.x[: aa1.n], aa2.x[: aa2.n])
    assert np.array_equal(aa1.y[: aa1.n], aa2.y[: aa2.n])
    assert np.array_equal(aa1.alive[: aa1.n], aa2.alive[: aa2.n])
    assert np.allclose(aa1.inv[: aa1.n], aa2.inv[: aa2.n])
    assert np.allclose(aa1.health[: aa1.n], aa2.health[: aa2.n])
    assert np.allclose(ca1.cell_res, ca2.cell_res)
    assert np.array_equal(ca1.struct_count, ca2.struct_count)
    assert np.allclose(ca1.struct_integrity, ca2.struct_integrity)


def test_redistribution_disabled_yields_no_flows():
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    rng = np.random.default_rng(0)
    random.seed(1)
    config = dict(BASE_CONFIG)
    config["redistribution"] = {"enabled": False}
    for i in range(1, 6):
        out = step(state, i, 3.0, config, rng)
        assert out.flows == []


def test_med_kits_non_calano_senza_uso_ne_dono():
    # med_kits e' consumato solo quando la salute/ossigeno/idratazione di un
    # agente scende sotto soglia (uso personale, action_space.py) o quando un
    # vicino sta sotto 0,42 (dono, rule_based_agent.py). Lo scenario tiene
    # tutti comodamente riforniti, quindi nessuno dei due percorsi scatta.
    #
    # **Dal 2026-08-25 il totale puo' CRESCERE**: le infermerie producono kit
    # (`RESA_KIT_MEDICI`), e in questo scenario gli agenti ne costruiscono una
    # durante gli otto passi. Prima nessuno li produceva e la scorta poteva
    # solo calare — il che rendeva il fabbisogno di 1,0 kit per abitante
    # dichiarato dal piano impossibile da soddisfare. Il test verifica quindi
    # cio' che resta vero: senza uso ne' dono il totale non DIMINUISCE, e
    # cresce solo dove esistono strutture che curano.
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    rng = np.random.default_rng(0)
    random.seed(3)

    total_before = float(aa.inv[: aa.n, C.R["med_kits"]].sum() + ca.cell_res[..., C.R["med_kits"]].sum())
    assert total_before > 0.0  # sanity: agents actually start with med_kits

    for i in range(1, 9):
        step(state, i, 3.0, BASE_CONFIG, rng)
        for aid in aa.ids[: aa.n]:
            row = aa.index[aid]
            assert aa.health[row] > 0.6, "scenario must stay well above med-kit thresholds"

    total_after = float(aa.inv[: aa.n, C.R["med_kits"]].sum() + ca.cell_res[..., C.R["med_kits"]].sum())
    healing = float(ca.struct_fx[..., C.E["healing_bonus"]].sum())
    assert total_after >= total_before - 1e-6, "nessun uso ne' dono: non puo' calare"
    if healing <= 0.0:
        assert np.isclose(total_before, total_after, atol=1e-6)
    else:
        assert total_after > total_before, (
            "con infermerie in piedi la produzione deve vedersi"
        )


def test_habitability_refreshes_every_step_not_just_at_construction():
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    rng = np.random.default_rng(0)
    random.seed(5)

    hab_after_construction = ca.habitability.copy()
    step(state, 1, 3.0, BASE_CONFIG, rng)
    hab_after_step1 = ca.habitability.copy()
    for _ in range(4):
        step(state, 2, 3.0, BASE_CONFIG, rng)
    hab_after_more_steps = ca.habitability.copy()

    # update_cells (kernel_biology) must actually be wired into step() and run
    # every call - not just leave habitability frozen at its construction-time
    # value (Task 8 carry-over, see kernel.py's module docstring).
    assert not np.allclose(hab_after_construction, hab_after_step1)
    assert not np.array_equal(hab_after_step1, hab_after_more_steps)


def test_step_matches_object_engine_short_horizon():
    """The decisive equivalence test.

    COVERS: 4 agents (2 sharing a structured home cell, 1 on a bare cell, 1 on
    a harsher cell with ambient radiation/dust), a home cell with 4 structure
    types (habitat/greenhouse/solar array/oxygen plant), psychosocial ON, 3
    steps, compared AFTER EVERY STEP (not just once at the end, so a
    divergence introduced at step 2 is reported at step 2, not only
    discoverable once it has propagated through step 3): agent x/y, every
    inventory resource, every vitals+psychosocial column, alive mask,
    per-cell resources, and per-cell struct_count/struct_integrity/
    habitability across the whole grid.

    SIGNAL, not just a longer horizon (review round 3, Important 4):
    `dt_days=365.0` (up from the original 3.0) so the continuous vitals/
    psychosocial channels actually move enough over 3 steps for every
    column's assertion to be non-vacuous. At dt_days=3.0, tick_vitals's own
    dt normalizes to ~8.2e-4 per step, and 6 of the 9 VITAL_COLS moved LESS
    than this test's atol=1e-4 over the whole 3-step run (oxygen 7.0e-5,
    hydration 4.4e-5, satiety 3.7e-5, morale 8.6e-5, cooperation 5.0e-6,
    compliance 7.7e-6) - replacing any of those 6 columns with its INITIAL
    value would still have passed. At dt_days=365.0 every column moves well
    past atol=1e-4 (smallest margin: cooperation at ~6.7e-4, ~6.7x atol).
    Verified by mutation: forcing `psychosocial_enabled=False` inside
    kernel.py is now caught with real margin on multiple columns at once
    (previously only `stress`, at a ~1.5x margin over atol, caught it);
    deleting `events_engine.advance` from `step()` is NOT caught here (this
    scenario never makes an event fire - see DOES NOT COVER) but IS caught by
    test_step_matches_object_engine_with_events_enabled below.

    DOES NOT COVER:
      - more than 3 steps: AgentArrays stores several vitals/psychosocial
        columns as float32 versus the object model's Python float64: over
        enough steps a threshold comparison inside decide() (e.g.
        `health < 0.55`) could in principle land on opposite sides of a
        branch for the two engines from nothing but float32/float64 rounding,
        which would then cascade into a genuine action divergence unrelated
        to any kernel bug. Keeping the horizon short (matching the existing
        precedent in test_kernel_vitals_equivalence.py's `for _round in
        range(3)`) stays safely inside float tolerance without masking a real
        divergence - `dt_days` (the SIGNAL knob above), not step count, is
        what changed.
      - the LLM-governor branch (out of scope for this task; see kernel.py's
        module docstring) and death/spawn handling: this scenario is kept
        deliberately survivable so no agent dies and no spawn logic runs -
        the death path now has its OWN dedicated equivalence coverage,
        test_step_matches_object_engine_lethal_scenario (partial deaths) and
        test_extinction_short_circuit_freezes_cell_state (population wipe)
        below.
      - extreme events actually firing: config leaves them at the default
        off here, so `events_engine.advance` only exercises its early-return
        path on both sides in THIS test -
        test_step_matches_object_engine_with_events_enabled below drives the
        firing path.
      - GUI/API-only observation memory-logging (`agent.memory.remember_cell`
        called by the *runner*, not by decide()/execute_action() - confirmed
        by grep that rule_based_agent.py never reads `explored_cells`, so
        omitting it from step() cannot affect any array-observable outcome
        checked here).
    """
    w_obj, agents_obj = _build_object_scenario()
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})

    config = dict(BASE_CONFIG)
    obj_events = ExtremeEventEngine()
    rng = np.random.default_rng(0)
    # **Passo settimanale, non annuale (2026-08-25).** Questi confronti usavano
    # `dt_days=365.0` per una ragione dichiarata: "so the continuous vitals /
    # psychosocial channels actually move enough over 3 steps". Era vero, ed era
    # il difetto: su `dt = dt_days/3650` la sazieta' calava di 2,9e-5 per passo
    # settimanale e per farla muovere serviva un passo di un anno. Ora i vitali
    # di privazione sono per passo e a sette giorni si muovono eccome — mentre
    # a 365 saturano, i coloni muoiono, e cio' che si confronta e' il
    # trattamento dei morti invece dell'aritmetica.
    dt_days = 7.0

    for i in range(1, 4):
        # Reseeded per step (rather than once before each engine's whole
        # 3-step run, as the original version did) so the two engines can be
        # driven and compared ONE STEP AT A TIME: both sides still see the
        # identical draw sequence for step i (that is all determinism
        # requires here), just no longer coupled to "run all of engine A,
        # then all of engine B".
        random.seed(999 + i)
        _object_step(w_obj, agents_obj, config, i, dt_days, obj_events)
        random.seed(999 + i)
        step(state, i, dt_days, config, rng)

        assert set(agents_obj.keys()) == set(aa.ids[: aa.n])  # nobody died on either side
        _assert_agents_and_cells_match(agents_obj, w_obj, aa, ca, atol=1e-4, context=f"step {i}")


def test_step_matches_object_engine_with_events_enabled():
    """Important 4 (review round 3): the previous equivalence test left
    `events_engine.advance` exercising only its early-return path (extreme
    events off by default) - deleting the call from `step()` entirely was
    NOT caught by anything committed. Force a deterministic event
    (`chance_per_step=1.0`) so `advance` actually runs its impact-application
    branch on both sides, and compare after every step. Same
    `dt_days=365.0`/atol=1e-4 signal rationale as
    test_step_matches_object_engine_short_horizon.

    COVERS the agent-impact branch too, not just structure wear (review
    round 3, Minor 3): `a2` (bare cell, no HABITAT/SHELTER, so never
    sheltered) sits at (1,1), which the deterministic Solar Flare footprint
    for this scenario's seed/config reaches on step 2 - `a2.health` drops by
    ~0.098 that step (`_apply_event_impacts`'s `agent.health -= 0.10 *
    severity` branch, extreme_events.py:154-167), well past atol. Verified
    by mutation: replacing `step()`'s `events_engine.advance(wv,
    live_agent_views, config, step_index)` call with
    `events_engine.advance(wv, {}, ...)` (passing the wrong - empty - agent
    set) is caught HERE (`a2.health` mismatch at "events step 2"), on top of
    the structure-wear divergence a deleted `advance` call already produced.
    a0/a1 (HABITAT-sheltered) and a3 (outside this scenario's footprint at
    every step) do NOT exercise this branch on their own - a2 is what closes
    the gap."""
    w_obj, agents_obj = _build_object_scenario()
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})

    config = dict(BASE_CONFIG)
    config["extreme_events_enabled"] = True
    config["extreme_events"] = {"chance_per_step": 1.0}
    obj_events = ExtremeEventEngine()
    rng = np.random.default_rng(0)
    # **Passo settimanale, non annuale (2026-08-25).** Questi confronti usavano
    # `dt_days=365.0` per una ragione dichiarata: "so the continuous vitals /
    # psychosocial channels actually move enough over 3 steps". Era vero, ed era
    # il difetto: su `dt = dt_days/3650` la sazieta' calava di 2,9e-5 per passo
    # settimanale e per farla muovere serviva un passo di un anno. Ora i vitali
    # di privazione sono per passo e a sette giorni si muovono eccome — mentre
    # a 365 saturano, i coloni muoiono, e cio' che si confronta e' il
    # trattamento dei morti invece dell'aritmetica.
    dt_days = 7.0

    event_fired = False
    for i in range(1, 4):
        random.seed(777 + i)
        _object_step(w_obj, agents_obj, config, i, dt_days, obj_events)
        random.seed(777 + i)
        step(state, i, dt_days, config, rng)

        if obj_events.active_event is not None or obj_events.upcoming_event is not None:
            event_fired = True
        assert state.metadata.get("active_event") == obj_events.active_event, (
            f"step {i}: active_event obj={obj_events.active_event} kernel={state.metadata.get('active_event')}")

        assert set(agents_obj.keys()) == set(aa.ids[: aa.n])  # this scenario stays survivable
        _assert_agents_and_cells_match(agents_obj, w_obj, aa, ca, atol=1e-4, context=f"events step {i}")

    # Sanity: the forced config actually exercised advance()'s firing branch,
    # not just its early-return - otherwise this test would silently degrade
    # back into test_step_matches_object_engine_short_horizon's coverage gap.
    assert event_fired, "scenario must actually trigger an extreme event"


def test_step_matches_object_engine_lethal_scenario():
    """Important 4 (review round 3): the death path through `step()` had NO
    committed equivalence coverage - the reviewer had to hand-write this
    scenario to confirm it works. `a3` is doomed (hydration=0.0, no water/ice
    reserves, on a bare cell with none nearby - see `_make_lethal_agents`)
    while a0/a1/a2 stay healthy. Both engines must kill a3 on step 1 with
    cause "dehydration" and keep matching everywhere else (survivor vitals/
    positions/inventories and full cell state - the population does NOT go
    extinct here, so biology/structure effects/events continue normally on
    both sides, unlike test_extinction_short_circuit_freezes_cell_state
    below, which is the population-wipe case)."""
    doomed = frozenset({"a3"})
    w_obj, agents_obj = _build_lethal_object_scenario(doomed)
    aa, ca, side, ps = _build_lethal_array_scenario(doomed)
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})

    config = dict(BASE_CONFIG)
    obj_events = ExtremeEventEngine()
    rng = np.random.default_rng(0)
    # **Passo settimanale, non annuale (2026-08-25).** Questi confronti usavano
    # `dt_days=365.0` per una ragione dichiarata: "so the continuous vitals /
    # psychosocial channels actually move enough over 3 steps". Era vero, ed era
    # il difetto: su `dt = dt_days/3650` la sazieta' calava di 2,9e-5 per passo
    # settimanale e per farla muovere serviva un passo di un anno. Ora i vitali
    # di privazione sono per passo e a sette giorni si muovono eccome — mentre
    # a 365 saturano, i coloni muoiono, e cio' che si confronta e' il
    # trattamento dei morti invece dell'aritmetica.
    dt_days = 7.0
    death_row = aa.index["a3"]

    for i in range(1, 3):
        random.seed(555 + i)
        _object_step(w_obj, agents_obj, config, i, dt_days, obj_events)
        random.seed(555 + i)
        out = step(state, i, dt_days, config, rng)

        if i == 1:
            assert "a3" not in agents_obj, "a3 must be dead on the object side by step 1"
            assert not aa.alive[death_row], "a3 must be dead on the vector side by step 1"
            assert out.deaths == [(death_row, "dehydration")]
            assert out.deaths_by_agent == [("a3", "dehydration")]
        else:
            assert out.deaths == []  # a3 was the only casualty, and only once

        alive_ids = {aid for aid in aa.ids[: aa.n] if aa.alive[aa.index[aid]]}
        assert set(agents_obj.keys()) == alive_ids
        _assert_agents_and_cells_match(agents_obj, w_obj, aa, ca, atol=1e-4, context=f"lethal step {i}")


def test_extinction_short_circuit_freezes_cell_state():
    """Important 3 (review round 3): both object engines stop BEFORE spawn/
    biology/structure-effects/events on the step the population hits zero
    (agent_coupled_runner.py:240-244 `break`s; state_store.py:431-437
    `return`s) - `step()` had no such guard, so on the step the last agent
    died it would advance cell state one pass further than either engine.

    All 4 agents are doomed (see `_make_lethal_agents`), so the population
    goes fully extinct on step 1. Two independent checks:

      1. `step()`'s own cell arrays must be BYTE-IDENTICAL, before vs. after
         the extinction step, on every column biology/wear could touch
         (habitability/vegetation/struct_integrity/cell_res) - the only
         cell-array change allowed is `occupancy` dropping to 0, from the
         (unconditional, pre-extinction-check) post-death recompute.
      2. The (now also extinction-guarded, see `_object_step`) object
         reference must match `step()` exactly - same deaths, same frozen
         cells - proving `step()` stops at the same point the real engines
         do, not just that it stops "early enough" by some other measure.
    """
    doomed = frozenset({"a0", "a1", "a2", "a3"})
    w_obj, agents_obj = _build_lethal_object_scenario(doomed)
    aa, ca, side, ps = _build_lethal_array_scenario(doomed)
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})

    config = dict(BASE_CONFIG)
    obj_events = ExtremeEventEngine()
    rng = np.random.default_rng(0)
    # **Passo settimanale, non annuale (2026-08-25).** Questi confronti usavano
    # `dt_days=365.0` per una ragione dichiarata: "so the continuous vitals /
    # psychosocial channels actually move enough over 3 steps". Era vero, ed era
    # il difetto: su `dt = dt_days/3650` la sazieta' calava di 2,9e-5 per passo
    # settimanale e per farla muovere serviva un passo di un anno. Ora i vitali
    # di privazione sono per passo e a sette giorni si muovono eccome — mentre
    # a 365 saturano, i coloni muoiono, e cio' che si confronta e' il
    # trattamento dei morti invece dell'aritmetica.
    dt_days = 7.0

    hab_before = ca.habitability.copy()
    veg_before = ca.vegetation.copy()
    struct_before = ca.struct_integrity.copy()
    cell_res_before = ca.cell_res.copy()

    random.seed(11)
    object_advanced = _object_step(w_obj, agents_obj, config, 1, dt_days, obj_events)
    random.seed(11)
    out = step(state, 1, dt_days, config, rng)

    assert object_advanced is False, "sanity: the object reference must ALSO hit its extinction guard"
    assert agents_obj == {}
    assert aa.alive_rows().size == 0
    assert {aid for aid, _ in out.deaths_by_agent} == {"a0", "a1", "a2", "a3"}
    assert all(cause == "dehydration" for _, cause in out.deaths_by_agent)

    # Check 1: step()'s own cells did not advance past the death sweep.
    assert np.array_equal(hab_before, ca.habitability), "habitability must not advance past extinction"
    assert np.array_equal(veg_before, ca.vegetation), "vegetation must not advance past extinction"
    assert np.array_equal(struct_before, ca.struct_integrity), "struct_integrity must not advance past extinction"
    # **`cell_res` non e' piu' congelato, e non poteva restarlo (2026-08-28).**
    # Il decesso deposita nella cella cio' che il colono portava, invece di
    # cancellarlo. Ricostruire l'atteso dagli inventari di INIZIO passo sarebbe
    # sbagliato — i condannati consumano razioni nel tick prima di morire — e
    # una tolleranza larga toglierebbe forza al test. L'asserzione giusta e'
    # quella del Check 2 esteso alle risorse: `cell_res` deve coincidere con la
    # referenza a oggetti, che fa lo stesso travaso. Che biologia e usura non
    # siano girate resta provato da habitability/vegetation/struct_integrity,
    # che il travaso non tocca.
    assert ca.cell_res.sum() > cell_res_before.sum(), (
        "le scorte dei morti devono essere finite da qualche parte"
    )
    assert int(ca.occupancy.sum()) == 0

    # Check 2: matches the object engine's own frozen state, cell for cell.
    for y, row_cells in enumerate(w_obj.cells):
        for x, cell in enumerate(row_cells):
            assert np.isclose(ca.habitability[y, x], cell.habitability_score, atol=1e-4), (
                f"cell({x},{y}).habitability: obj={cell.habitability_score} vec={ca.habitability[y, x]}")
            for nome_risorsa, indice in C.R.items():
                atteso = float(getattr(cell.resources, nome_risorsa, 0.0) or 0.0)
                assert np.isclose(ca.cell_res[y, x, indice], atteso, atol=1e-6), (
                    f"cell({x},{y}).{nome_risorsa}: obj={atteso} vec={ca.cell_res[y, x, indice]}")

    # No biology/events output leaked into StepOutcome on this path either.
    assert out.knowledge_gain == 0.0
    assert out.flows == []
    assert all(e["type"] == "agent_died" for e in out.events)


def test_refresh_cell_alerts_wired_into_kernel_step():
    """Task 14 fix (equivalence audit, scripts/compare_v1_v2.py): `refresh_
    cell_alerts` (vitals.py:75-97 - called from inside `tick_agent_vitals` at
    vitals.py:196, i.e. squarely inside the "1:1 transcription" range
    kernel_vitals.py's own module docstring claims) was entirely ABSENT from
    `kernel.step()` before this fix: agent memory on the kernel path never
    received the overcrowding/pollution SYSTEM ALERT entries the object
    engine's `tick_all_agents` always writes every step. `step()` now calls
    the real, unmodified `refresh_cell_alerts` directly on the views, right
    after `tick_vitals`, for every agent in the pre-action alive set - see
    kernel.py's own comment at that call site.

    This scenario packs 12 agents onto one cell (agents_present > 10, the
    overcrowding threshold, vitals.py:90) with that cell's `pollution_risk`
    pre-set above the 0.04 pollution threshold (vitals.py:94), so BOTH
    alerts must be active after one step.

    Verifies by INDEPENDENT reconstruction, not a second engine run: reads
    back the exact post-step `agents_present` count and `pollution_risk` for
    the cell `kernel.step()` actually used, feeds those into a fresh, plain
    object `Cell` + `AgentMemory` through the SAME unmodified `refresh_cell_
    alerts` function, and asserts the two `active_alerts` dicts are equal.
    This sidesteps decide()/execute()'s own random exploration roll (see
    scripts/compare_v1_v2.py's worker docstring - src/agents/rule_based_
    agent.py reads the process-global `random` module for one branch, so a
    real two-process comparison needs its own seeding discipline this test
    does not need)."""
    from src.agents.memory import AgentMemory
    from src.agents.vitals import refresh_cell_alerts
    from src.world.cell import Cell as ObjectCell
    from src.world.cell import set_cell_degradation

    set_cell_degradation(True)
    try:
        w = _grid(3, 3)
        home = w.get_cell(1, 1)
        home.pollution_risk = 0.10  # > vitals.py's 0.04 pollution threshold

        agents_src: dict[str, RuleBasedAgent] = {}
        for i in range(12):  # > vitals.py's 10-agent overcrowding threshold
            aid = f"a{i}"
            agents_src[aid] = RuleBasedAgent(
                agent_id=aid, name=aid, role="colonist", x=1, y=1,
                perception_radius=1, survival_priority=0.7,
            )
        aa = AgentArrays.from_agents(agents_src)
        ca = CellArrays.from_world(w)
        side = {aid: AgentSideState.from_agent(a) for aid, a in agents_src.items()}
        ps = dict(w.planetary_state)
        state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})

        config = dict(BASE_CONFIG)
        config["extreme_events"] = {"enabled": False}
        dt_days = 7.0

        step(state, 1, dt_days, config, None)

        target_aid = "a0"
        row = aa.index[target_aid]
        x, y = int(aa.x[row]), int(aa.y[row])
        agents_present_count = int(ca.occupancy[y, x])
        pollution = float(ca.pollution[y, x])
        assert agents_present_count > 10, "sanity: overcrowding condition must actually hold"
        assert pollution > 0.04, "sanity: pollution condition must actually hold"

        shadow_cell = ObjectCell(x=x, y=y, terrain=home.terrain)
        shadow_cell.agents_present = [f"shadow{i}" for i in range(agents_present_count)]
        shadow_cell.pollution_risk = pollution

        class _ShadowAgent:
            def __init__(self):
                self.memory = AgentMemory()

        shadow_agent = _ShadowAgent()
        refresh_cell_alerts(shadow_agent, shadow_cell)

        kernel_alerts = state.side[target_aid].memory.active_alerts
        assert kernel_alerts == shadow_agent.memory.active_alerts, (
            f"kernel alerts={kernel_alerts} expected={shadow_agent.memory.active_alerts}"
        )
        assert any("Overcrowding" in v for v in kernel_alerts.values()), kernel_alerts
        assert any("degrading oxygen quality" in v for v in kernel_alerts.values()), kernel_alerts

        # And the SAME summarize() text a snapshot's memory_summary would show.
        assert AgentSideState.__init__  # sanity import stays used
        expected_summary = shadow_agent.memory.summarize()
        assert state.side[target_aid].memory.summarize() == expected_summary
    finally:
        set_cell_degradation(True)
