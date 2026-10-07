"""Task 12, Part 1: equivalence test for `kernel.py`'s `StepOutcome.agent_step_records`
enrichment (`AgentStepRecord`).

Context (see `.superpowers/sdd/task-12-report.md`): migrating
`SimulationController` to run on `kernel.step()` requires the shell to
reconstruct per-agent bookkeeping (action log rows, replay events,
conversations, `_record_social_effect` inputs, "thoughts") that the object
engines build INSIDE their execute loop, right after each agent's own
`execute_action` call and BEFORE `tick_all_agents`/`tick_vitals` mutates
health/satiety/oxygen/hydration/fatigue/stress/morale in place. The first
attempt at this task found `StepOutcome` insufficient for that (only a
flattened summary dict, no `Observation`/full `ActionRequest`, no pre-vitals
scalar snapshot) and stopped rather than duplicate the decide/execute loop in
the shell or touch a pinned logic module.

The fix: `kernel.step()`'s OWN execute loop now builds each agent's full
bookkeeping record (`AgentStepRecord`) at that exact pre-vitals point, via
the SAME shared formatters the shells already import
(`build_action_log_row`, `build_replay_event`) - not reimplemented. This test
proves that enrichment is faithful: an independent reproduction of the
object engine's own full step - not just its decide/execute prefix, the
WHOLE step (vitals/death/biology/events, mirroring test_kernel_step.py's own
`_object_step`) so the reference world/agents stay synchronized with
`kernel.step()`'s persistent `CoreState` call over call - with bookkeeping
captured at the same pre-vitals point, must produce the SAME rows, run in
lockstep against `kernel.step()`, agent-by-agent, step-by-step.
"""

from __future__ import annotations

import random

from src.agents.action_space import execute_action
from src.agents.rule_based_agent import RuleBasedAgent
from src.agents.vitals import tick_all_agents
from src.api.state_store import _full_biology_update_enabled
from src.core.arrays import AgentArrays, CellArrays
from src.core.kernel import CoreState, step
from src.core.views import AgentSideState
from src.simulation.action_logging import build_action_log_row
from src.simulation.biology_update import update_biology_cells
from src.simulation.extreme_events import ExtremeEventEngine
from src.simulation.run_artifacts import build_replay_event
from src.simulation.step_effects import (
    apply_colony_resource_feedback,
    apply_structure_effects,
    apply_structure_wear,
)
from src.world.perception import observe_fast
from src.world.resources import ResourceBundle

from src.agents.vitals import PASSI_SINTOMO_SENZA_CIBO
from tests.core.test_kernel_step import BASE_CONFIG, _agent_specs, _build_world


def _bookkeeping_agent_specs() -> list[dict]:
    """Stessa popolazione di quattro di `_agent_specs`, con un ritocco voluto.

    A questo test serve almeno un `memory.add_event` deterministico ("thought")
    entro tre passi, dentro lo stesso orizzonte breve che gli altri confronti
    hanno gia' stabilito come sicuro rispetto alla divergenza float32/float64.

    **Si usa l'orologio del CIBO, non quello dell'acqua (2026-08-25).** Prima
    a3 partiva da `steps_without_water=3`, uno sotto la soglia di sintomo di
    allora (4) e ben sotto la morte (7). Con le finestre nuove — sintomo a 1,
    morte a 2 — fra le due non c'e' piu' spazio per tre passi. La finestra del
    cibo (sintomo a 4, morte a 6) lo lascia: partendo due passi sotto la
    soglia, a3 la attraversa al secondo passo e non arriva alla morte entro il
    terzo. Le soglie sono lette dalle costanti, cosi' una futura ritaratura
    non rimette il test fuori scala in silenzio.
    """
    specs = []
    for spec in _agent_specs():
        spec = dict(spec)
        if spec["agent_id"] == "a3":
            spec["steps_without_food"] = max(0, PASSI_SINTOMO_SENZA_CIBO - 2)
            # L'inventario di default di BaseAgent porta food=5.0: la razione
            # personale lo rifornirebbe ogni passo azzerando il contatore, e il
            # preset qui sopra non servirebbe a nulla.
            spec["inventory"] = ResourceBundle(energy=2, oxygen=2, water=4.0, food=0.0, construction_material=5, minerals=3, med_kits=2.0)
        specs.append(spec)
    return specs


def _build_object_scenario():
    w = _build_world()
    agents: dict[str, RuleBasedAgent] = {}
    for spec in _bookkeeping_agent_specs():
        a = RuleBasedAgent(**spec)
        agents[a.agent_id] = a
        w.place_agent(a.agent_id, a.x, a.y)
    return w, agents


def _build_array_scenario():
    w = _build_world()
    agents_src: dict[str, RuleBasedAgent] = {}
    for spec in _bookkeeping_agent_specs():
        agents_src[spec["agent_id"]] = RuleBasedAgent(**spec)
    aa = AgentArrays.from_agents(agents_src)
    ca = CellArrays.from_world(w)
    side = {aid: AgentSideState.from_agent(a) for aid, a in agents_src.items()}
    ps = dict(w.planetary_state)
    return aa, ca, side, ps


def _object_step_with_bookkeeping(world, agents, config, step_index, dt_days, events_engine):
    """The FULL object-engine step (observe -> decide -> execute -> vitals ->
    death -> biology -> events), matching test_kernel_step.py's own
    `_object_step` closely (same imports, same order) so `world`/`agents`
    stay synchronized with `kernel.step()`'s `CoreState` from one call to the
    next - PLUS the per-agent bookkeeping capture (state_store.py:376-405),
    taken at the same pre-vitals point `_object_step` doesn't otherwise need.
    Returns the list of bookkeeping dicts for this step."""
    world.step = step_index
    world.metadata["active_event"] = events_engine.active_event
    world.metadata["upcoming_event"] = events_engine.upcoming_event

    agents_list = list(agents.values())
    observations = [
        observe_fast(world, a.agent_id, a.x, a.y, a.perception_radius, agents)
        for a in agents_list
    ]
    requests = [a.decide(obs, world) for a, obs in zip(agents_list, observations)]

    day_start = world.day
    records = []
    for agent, obs, req in zip(agents_list, observations, requests):
        result = execute_action(agent, agents, world, req)
        records.append({
            "agent_id": agent.agent_id,
            "request": req,
            "accepted": bool(result.accepted),
            "result_data": dict(result.data) if isinstance(result.data, dict) else {},
            "nearby_agents": list(obs.nearby_agents),
            "action_log_row": build_action_log_row(step_index, day_start, agent, req, result, obs),
            "replay_event": build_replay_event(step_index, day_start, agent, req, result),
            "thought": agent.memory.recent_events[-1] if agent.memory.recent_events else None,
            "x": agent.x,
            "y": agent.y,
        })

    tick_all_agents(agents_list, world, dt_days, config=config)

    dead_ids = [aid for aid, a in agents.items() if a.health <= 0]
    for dead_id in dead_ids:
        agent = agents[dead_id]
        world.remove_agent(dead_id, agent.x, agent.y)
        del agents[dead_id]

    if not agents:
        return records

    planetary_metrics = dict(world.planetary_state)
    update_biology_cells(world, planetary_metrics, dt_days, full_grid=_full_biology_update_enabled(config))
    apply_structure_effects(world)
    apply_colony_resource_feedback(world, agents, dt_days)
    apply_structure_wear(world, dt_days)

    events_engine.advance(world, agents, config, step_index)
    world.day = day_start + dt_days
    return records


def _assert_close(a, b, atol: float, path: str = "") -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys()), f"{path}: keys differ {sorted(a.keys())} vs {sorted(b.keys())}"
        for key in a:
            _assert_close(a[key], b[key], atol, f"{path}.{key}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"{path}: length differs ({a!r} vs {b!r})"
        for index, (av, bv) in enumerate(zip(a, b)):
            _assert_close(av, bv, atol, f"{path}[{index}]")
    elif isinstance(a, bool) or isinstance(b, bool):
        assert a == b, f"{path}: {a!r} vs {b!r}"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert abs(float(a) - float(b)) <= atol, f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def test_agent_step_records_match_object_engine_bookkeeping():
    w_obj, agents_obj = _build_object_scenario()
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})

    config = dict(BASE_CONFIG)
    obj_events = ExtremeEventEngine()
    # dt_days/step-count kept inside the same safe horizon
    # test_kernel_step.py's own equivalence tests already established (see
    # its test_step_matches_object_engine_short_horizon docstring, "DOES NOT
    # COVER: more than 3 steps"): AgentArrays stores vitals/psychosocial
    # columns as float32 vs. the object model's float64, so a threshold
    # comparison inside decide() can land on opposite sides of a branch from
    # nothing but rounding once enough steps compound - a genuine action
    # divergence unrelated to this test's actual target (the bookkeeping
    # rows built from whatever action WAS taken). 3 steps at dt_days=365.0
    # (matching test_kernel_step.py's own signal choice) stays inside that
    # margin while still producing enough signal (multiple action types, at
    # least one thought, at least one action with real result data -
    # asserted below).
    # **Passo settimanale, non annuale (2026-08-25).** Questi confronti usavano
    # `dt_days=365.0` per una ragione dichiarata: "so the continuous vitals /
    # psychosocial channels actually move enough over 3 steps". Era vero, ed era
    # il difetto: su `dt = dt_days/3650` la sazieta' calava di 2,9e-5 per passo
    # settimanale e per farla muovere serviva un passo di un anno. Ora i vitali
    # di privazione sono per passo e a sette giorni si muovono eccome — mentre
    # a 365 saturano, i coloni muoiono, e cio' che si confronta e' il
    # trattamento dei morti invece dell'aritmetica.
    dt_days = 7.0
    atol = 1e-3

    seen_thought = False
    seen_result_data = False
    seen_action_types: set[str] = set()

    for i in range(1, 4):
        random.seed(4200 + i)
        obj_records = _object_step_with_bookkeeping(w_obj, agents_obj, config, i, dt_days, obj_events)
        random.seed(4200 + i)
        out = step(state, i, dt_days, config, None)

        assert len(out.agent_step_records) == len(obj_records), f"step {i}: record count differs"
        for rec, expected in zip(out.agent_step_records, obj_records):
            ctx = f"step {i} agent {rec.agent_id}"
            assert rec.agent_id == expected["agent_id"], ctx
            assert rec.accepted == expected["accepted"], ctx
            assert rec.nearby_agents == expected["nearby_agents"], ctx
            assert rec.x == expected["x"], ctx
            assert rec.y == expected["y"], ctx
            assert rec.request.action == expected["request"].action, ctx
            assert rec.request.message == expected["request"].message, ctx
            _assert_close(rec.result_data, expected["result_data"], atol, f"{ctx} result_data")
            _assert_close(rec.action_log_row, expected["action_log_row"], atol, f"{ctx} action_log_row")
            _assert_close(rec.replay_event, expected["replay_event"], atol, f"{ctx} replay_event")
            assert rec.thought == expected["thought"], f"{ctx} thought"

            seen_action_types.add(rec.request.action.value)
            if expected["thought"]:
                seen_thought = True
            if expected["result_data"]:
                seen_result_data = True

        # Both engines must also still agree on population/positions (same
        # equivalence discipline as test_kernel_step.py) - a divergence here
        # would make the per-agent comparison above meaningless (comparing
        # bookkeeping for agents that are no longer at the same position/
        # state on both sides).
        assert set(agents_obj.keys()) == {aid for aid in aa.ids[: aa.n] if aa.alive[aa.index[aid]]}

    assert seen_thought, "scenario must exercise at least one real memory event this run"
    assert seen_result_data, "scenario must exercise at least one action with real result data"
    assert len(seen_action_types) > 1, "scenario must exercise more than one action type"
