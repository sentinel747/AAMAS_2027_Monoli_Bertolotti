from __future__ import annotations

"""Task 13: orchestration shared by BOTH shells - `SimulationController`
(src/api/state_store.py, GUI) and `AgentCoupledRunner`
(src/simulation/agent_coupled_runner.py, CLI). Both build a `CoreState` from
an object-model world+agents at construction time (the UNMODIFIED
`WorldGenerator`/`spawn_initial_agents`/`seed_initial_colony_support`), then
drive every step through `src.core.kernel.step`, draining the returned
`StepOutcome.agent_step_records` into their own artifact buffers. This is
the >30-line orchestration the Task 13 brief calls out to hoist here rather
than copy-paste a second time: every function below is called by BOTH
shells, byte-identically, so they cannot silently drift apart on how they
build state, commit a birth, or drain a step's bookkeeping - the exact
regression class `tests/core/test_cross_engine_determinism.py` (Task 13)
guards against.

Also home to `_observer_for_agent`/`_full_biology_update_enabled`: both
shells' config-driven "which observer"/"which biology scope" choice, and
`src/core/kernel.py`'s own internal copy of the same choice (kernel.py reads
the identical config keys and must pick the identical observer/scope - see
kernel.py's module docstring, stage 2 and stage 8). These lived on
`src/simulation/agent_coupled_runner.py` pre-Task-13 (kernel.py imported them
from there); moved here so kernel.py does not have to import EITHER shell -
`AgentCoupledRunner` needing `src.core.kernel.step` (Task 13) would otherwise
close a `kernel -> agent_coupled_runner -> kernel` cycle. `SimulationController`
(`src/api/state_store.py`) keeps its own byte-identical copy for now (its own
docstring explains why - the two must never drift, verified line-for-line);
this module is the one `kernel.py` itself imports from.

Deliberately NOT imported at module level: `src.core.kernel` (would recreate
the same cycle this module exists to avoid, since `kernel.py` imports FROM
here - `CoreState`/`StepOutcome`/`AgentStepRecord` annotations below are
inert strings only, courtesy of `from __future__ import annotations`, so
they need no real import to appear in a signature). `build_core_state` is
the one function that actually NEEDS the real `CoreState` class (to
construct one) - it imports it locally, at call time, when every module has
long finished loading and the cycle risk is gone.
"""

import os
from collections import Counter

import numpy as np

from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.views import AgentSideState, AgentView, WorldView
from src.world.perception import observe, observe_fast, observe_rule_based


def build_core_state(object_world, object_agents: dict) -> tuple["CoreState", WorldView]:
    """Convert an object-model `GridWorld` + `dict[str, BaseAgent]` (built by
    the unmodified `WorldGenerator`/`spawn_initial_agents`/
    `seed_initial_colony_support`) into a vectorized `CoreState` plus a
    persistent `WorldView` over it - the exact conversion both
    `SimulationController._build_core_state` and `AgentCoupledRunner.__init__`
    need. `object_world`/`object_agents` are local to the caller and
    discarded once this returns - nothing after this call ever touches them
    again (see either shell's own docstring)."""
    from src.core.kernel import CoreState  # local: see module docstring (import-cycle avoidance)

    cell_arrays = CellArrays.from_world(object_world)
    agent_arrays = AgentArrays.from_agents(object_agents)
    side = {aid: AgentSideState.from_agent(agent) for aid, agent in object_agents.items()}
    metadata = dict(object_world.metadata)
    planetary_state = dict(object_world.planetary_state)
    core = CoreState(agents=agent_arrays, cells=cell_arrays, side=side, planetary_state=planetary_state, metadata=metadata)
    world = WorldView(
        core.cells,
        core.agents,
        core.planetary_state,
        core.metadata,
        core.side,
    )
    world.events.extend(object_world.events)
    return core, world


def rebuild_agent_views(core: CoreState) -> dict[str, AgentView]:
    """`dict[str, AgentView]` over the CURRENT alive set - mirrors the object
    engines' own `dict[str, BaseAgent]` (only alive agents present) closely
    enough that every existing consumer keeps working unmodified. Rebuilt
    after every `step()` (deaths/spawn change the alive set) and once at
    construction time."""
    core_agents = core.agents
    return {
        core_agents.ids[int(r)]: AgentView(core_agents, int(r), core.side[core_agents.ids[int(r)]])
        for r in core_agents.alive_rows()
    }


def commit_spawned_agent(core: CoreState, agents: dict, spawned) -> None:
    """Bridges `src.agents.population.maybe_spawn_agent`'s output (a real,
    fully-formed `RuleBasedAgent` - random psychological profile included)
    into the vectorized `CoreState`: allocates a new `AgentArrays` row
    (`AgentArrays.spawn`) and a new `AgentSideState`, both populated FROM the
    spawned object's own values - not re-derived. `maybe_spawn_agent` itself
    runs completely unmodified against the caller's own `agents`/`world`;
    this only commits its result to the array representation `kernel.step()`
    actually iterates. Mutates `core`/`agents` in place (adds the new row and
    the new `AgentView` under `spawned.agent_id`)."""
    core_agents = core.agents
    inv_row = np.zeros(C.NR, dtype=np.float64)
    for name, idx in C.R.items():
        inv_row[idx] = float(getattr(spawned.inventory, name, 0.0))
    row = core_agents.spawn(spawned.agent_id, spawned.x, spawned.y, inv_row)
    core_agents.health[row] = float(spawned.health)
    core_agents.satiety[row] = float(spawned.satiety)
    core_agents.oxygen[row] = float(spawned.oxygen_level)
    core_agents.hydration[row] = float(spawned.hydration)
    core_agents.fatigue[row] = float(spawned.fatigue)
    core_agents.stress[row] = float(spawned.stress_index)
    core_agents.morale[row] = float(spawned.morale)
    core_agents.cooperation[row] = float(spawned.cooperation)
    core_agents.compliance[row] = float(spawned.protocol_compliance)
    core_agents.autonomy[row] = float(spawned.autonomy_preference)
    core_agents.risk_tolerance[row] = float(spawned.risk_tolerance)
    core_agents.curiosity[row] = float(spawned.curiosity)
    preferences = getattr(spawned, "pillar_preferences", None)
    if preferences is not None:
        core_agents.pref[row] = np.asarray(preferences, dtype=np.float64)
    skills = getattr(spawned, "pillar_skills", None)
    if skills is not None:
        core_agents.skill[row] = np.asarray(skills, dtype=np.float64)
    core_agents.home_x[row], core_agents.home_y[row] = int(spawned.x), int(spawned.y)
    core.side[spawned.agent_id] = AgentSideState.from_agent(spawned)
    agents[spawned.agent_id] = AgentView(core_agents, row, core.side[spawned.agent_id])
    # Task 14 fix (equivalence audit): mirror `WorldView.place_agent`'s object
    # -engine counterpart (`GridWorld.place_agent`, called by
    # `maybe_spawn_agent`/`spawn_initial_agents`) keeping `cells.occupancy`
    # live for the newborn's cell - `WorldView.place_agent` itself is a
    # deliberate no-op here (the row does not exist yet when population.py
    # calls it, see its own docstring); this is the point right after the row
    # actually exists.
    core.cells.occupancy[int(spawned.y), int(spawned.x)] += 1


def _observer_for_agent(agent, use_fast_observation: bool, use_rule_observation: bool):
    """Which observer function a given agent's observation this step comes
    from - byte-identical to the copy `SimulationController`
    (`src/api/state_store.py`) still keeps for itself, and the canonical
    source `src/core/kernel.py` imports from (see this module's own
    docstring for why kernel.py imports from here rather than from either
    shell)."""
    if use_fast_observation:
        return observe_fast
    if use_rule_observation and getattr(agent, "mode", "rule_based") != "llm":
        return observe_rule_based
    return observe


def _full_biology_update_enabled(config: dict) -> bool:
    """Whether `headless.biology_update_scope` requests the "full" (whole-
    grid) biology scope instead of the default "active" (occupied/structured/
    already-changing cells only) scope - see `kernel_biology.py`'s own
    docstring for how `kernel.step()`'s vectorized `update_cells` reproduces
    this exact scoping."""
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    return str(headless.get("biology_update_scope", "active")).strip().lower() == "full"


def _action_log_row_may_be_read(config: dict) -> bool:
    """Se qualcuno potrebbe leggere la riga di log COMPLETA di un'azione accettata.

    `build_action_log_row` non ha effetti collaterali: costruisce un dizionario di
    ~45 campi, due dei quali passano da `json.dumps`. Misurato su configurazione
    reale: **12.000 righe costruite in 40 passi, zero lette** -- il 7,61% della run
    -- perche' `headless.store_memory_logs` vale `False` nella config che GUI,
    wizard e CLI producono, cioe' anche nelle run di tesi. La riga di un'azione
    **rifiutata** serve comunque, perche' `drain_agent_step_records` ne legge il
    messaggio per contare i motivi di rifiuto: quella si costruisce sempre.

    **Il default e' `True` di proposito, ed e' la parte che conta.** Le due shell
    decidono `store_memory_logs` in modo leggermente diverso quando la chiave
    manca (`agent_coupled_runner`: `len(agents) <= 10_000`; `state_store`: `True`),
    e il kernel non sa quale delle due lo chiamera'. Rispondendo `True` in caso di
    dubbio il kernel non puo' mai costruire MENO di quanto un consumatore
    leggera': al peggio costruisce una riga che nessuno usa, che e' esattamente il
    comportamento storico. La direzione opposta sarebbe un dato mancante.
    """
    if os.environ.get("MARSABM_LOG_ROWS", "lazy").strip().lower() == "always":
        # Percorso storico, conservato per poter rimisurare il guadagno con un
        # confronto A/B interlacciato invece di doverlo dichiarare. Non e' una
        # scelta di modellazione: i due percorsi producono lo stesso stato e gli
        # stessi artefatti, verificato in `tests/core/test_action_log_rows.py`.
        return True
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = headless.get("store_memory_logs", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(raw)


def invalidate_structure_cache(world: WorldView) -> None:
    """`src/simulation/colony_dynamics.py`'s `_structure_cells` and
    `src/agents/population.py`'s `colony_local_habitability` both cache
    `world._structure_positions` as a plain mutable attribute the first time
    they see a nonempty set, then trust it forever - correct against a
    persistent `GridWorld` (whose own `add_structure` keeps that same
    attribute updated incrementally), wrong against a shell-level `WorldView`
    (`kernel.step()` builds its own TRANSIENT `WorldView` every call -
    anything it sets on that instance is discarded the moment `step()`
    returns, so nothing keeps the shell-level `WorldView`'s own cached
    attribute in sync as new structures get built). Clearing it every step/
    metrics call instead forces a fresh rebuild: `CellArrays.struct_count` is
    a live vectorized array, so a full rescan is cheap and never wrong."""
    world.__dict__.pop("_structure_positions", None)


def record_social_effect(social, agent_id: str, request, accepted: bool, nearby_agents: list[str], step: int) -> None:
    """Both shells' `_record_social_effect` bodies were byte-identical
    already (see git history of `state_store.py`/`agent_coupled_runner.py`
    pre-Task-13) - hoisted here verbatim so a future edit cannot apply to
    only one of them by accident. `step` is only used for the deterministic
    targetless-communicate fallback rotation below."""
    if not accepted:
        return
    target = ""
    if isinstance(request.target, str):
        target = request.target
    elif isinstance(request.target, dict):
        target = request.target.get("agent_id") or request.target.get("target_agent_id")
    if request.action.value == "communicate":
        if not target and nearby_agents:
            # Targetless fallback rotates like the rule policy: a fixed
            # [0] on the sorted neighbor list funnels every edge onto the
            # lexicographically first agent (the run3 call-center hub).
            target = nearby_agents[(sum(map(ord, agent_id)) + int(step)) % len(nearby_agents)]
        if target:
            social.update(agent_id, target, "communicate")
            social.update(agent_id, target, "cooperate")
    elif request.action.value == "share_resource" and target:
        social.update(agent_id, target, "share_resource")


def drain_agent_step_records(
    outcome,
    step: int,
    day: float,
    *,
    world,
    social,
    validated_actions: list,
    rejected_actions: list,
    replay_events: list,
    conversations: list,
    thoughts: list,
    action_counts: Counter,
    rejection_counts: Counter,
    max_action_log_rows: int,
    max_replay_rows: int,
    max_observe_replay_rows_per_step: int = 10,
    action_log_truncated: bool,
    replay_log_truncated: bool = False,
    replay_compactions: int = 0,
    store_memory_logs: bool = True,
    cap_thoughts: bool = True,
    include_nearby_agent_ids: bool = True,
) -> dict:
    """Drains one `kernel.step()` call's `StepOutcome.agent_step_records`
    into the shell's own artifact buffers - the per-step bookkeeping loop
    both `SimulationController.step` and `AgentCoupledRunner.run_async` run
    against the SAME records, byte-identical apart from two flags neither
    shell can just drop without changing pre-existing, tested behavior:

    - `store_memory_logs`: shared GUI/CLI gate (`headless.store_memory_logs`)
      that skips action-log rows and thoughts on long or large runs.
    - `include_nearby_agent_ids`: keeps the compact nearby-agent count while
      allowing the usually redundant full ID list to be omitted.
    - `cap_thoughts`: the GUI caps `thoughts` at `max_action_log_rows` (its
      pre-Task-13 `step()` did `len(self.thoughts) < max_action_log_rows`);
      the CLI's pre-Task-13 `run_async` never capped it (only gated by
      `store_memory_logs`) - both behaviors are preserved via this flag
      rather than silently unifying two already-shipped, already-tested
      artifact shapes.

    Mutates `validated_actions`/`rejected_actions`/`replay_events`/
    `conversations`/`thoughts`/`action_counts`/`rejection_counts` in place
    (shell-owned lists/Counter). Rejection counters are unconditional, so
    they remain complete even when row logs are disabled or capped. Returns
    the attempted/accepted/rejected deltas and the truncation/compaction
    state, since ints/bools are immutable and the caller owns those counters
    directly.
    """
    attempted = 0
    accepted = 0
    rejected = 0
    passive_replay_actions = ("observe", "do_nothing")
    passive_accepted = Counter()
    passive_replayed = Counter()

    def append_replay_event(event: dict) -> None:
        nonlocal replay_log_truncated, replay_compactions
        if len(replay_events) >= max_replay_rows:
            replay_log_truncated = True
            replay_compactions += 1
            if max_replay_rows <= 1:
                replay_events.clear()
            else:
                # Deterministic temporal compaction: retain a sparse history
                # and always leave room for the newest accepted action.
                replay_events[:] = replay_events[::2]
                if len(replay_events) >= max_replay_rows:
                    del replay_events[max_replay_rows - 1 :]
            if replay_compactions == 1:
                world.log_event(
                    "replay_log_compacted",
                    f"replay cap reached ({max_replay_rows} rows): keeping a representative temporal sample",
                    max_replay_rows=max_replay_rows,
                )
        replay_events.append(event)

    for record in outcome.agent_step_records:
        record_social_effect(social, record.agent_id, record.request, record.accepted, record.nearby_agents, step)
        attempted += 1
        action_counts[record.request.action.value] += 1
        if record.accepted:
            accepted += 1
        else:
            rejected += 1
            if rejection_counts is not None:
                reason = str(record.action_log_row.get("message") or "unspecified")
                rejection_counts[(record.request.action.value, reason)] += 1
        if store_memory_logs and (len(validated_actions) + len(rejected_actions)) < max_action_log_rows:
            action_log_row = record.action_log_row
            if not include_nearby_agent_ids and "nearby_agents" in action_log_row:
                action_log_row = dict(action_log_row)
                action_log_row.pop("nearby_agents", None)
            (validated_actions if record.accepted else rejected_actions).append(action_log_row)
        elif store_memory_logs and not action_log_truncated:
            action_log_truncated = True
            world.log_event(
                "action_log_truncated",
                f"action log cap reached ({max_action_log_rows} rows): later actions are counted in metrics but not logged",
                max_action_log_rows=max_action_log_rows,
            )
        if record.accepted:
            action_value = record.request.action.value
            if action_value in passive_replay_actions:
                passive_accepted[action_value] += 1
                if passive_replayed[action_value] < max(0, max_observe_replay_rows_per_step):
                    append_replay_event(record.replay_event)
                    passive_replayed[action_value] += 1
            else:
                append_replay_event(record.replay_event)
        if record.request.action.value == "communicate":
            conversations.append({
                "step": step,
                "day": day,
                "agent_id": record.agent_id,
                "target": record.result_data.get("target"),
                "message": record.request.message,
                "x": record.x,
                "y": record.y,
            })
        if store_memory_logs and record.thought is not None and (not cap_thoughts or len(thoughts) < max_action_log_rows):
            thoughts.append({"step": step, "day": day, "agent_id": record.agent_id, "thought_summary": record.thought})
    omitted_by_action = {
        action: passive_accepted[action] - passive_replayed[action]
        for action in passive_replay_actions
    }
    for action, omitted in omitted_by_action.items():
        if omitted <= 0:
            continue
        accepted_count = passive_accepted[action]
        sampled_count = passive_replayed[action]
        if action == "observe":
            label = "Observation summary"
            result_data = {
                "accepted_observations": accepted_count,
                "sampled_observations": sampled_count,
                "omitted_observations": omitted,
            }
        else:
            label = "Idle-action summary"
            result_data = {
                "accepted_idle_actions": accepted_count,
                "sampled_idle_actions": sampled_count,
                "omitted_idle_actions": omitted,
            }
        append_replay_event(
            {
                "step": step,
                "day": day,
                # Empty id keeps aggregate rows out of replay agent-position
                # maps while the live/action feed can still render the label.
                "agent_id": "",
                "agent_name": label,
                "mode": "aggregate",
                "action": f"{action}_summary",
                "x": None,
                "y": None,
                "local_x_m": 0.0,
                "local_y_m": 0.0,
                "target": {},
                "result_data": result_data,
                "message": (
                    f"{accepted_count} accepted {action} actions; "
                    f"{sampled_count} agent rows retained"
                ),
            }
        )
    return {
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "action_log_truncated": action_log_truncated,
        "replay_log_truncated": replay_log_truncated,
        "replay_compactions": replay_compactions,
        "replay_observations_omitted": omitted_by_action["observe"],
        "replay_passive_rows_omitted": sum(omitted_by_action.values()),
    }


def morti_per_cella(dead_agents: list, gia_visti: int) -> tuple[dict, int]:
    """I decessi NUOVI dall'ultima chiamata, raggruppati per cella e per causa.

    Sta qui e non nelle due shell perche' le due shell devono dare lo stesso
    quadro allo strato amministrativo: due copie di questo conteggio
    divergerebbero in silenzio, e la voce che divergerebbe e' l'unico ESITO che
    un amministratore riceve --- tutto il resto del suo quadro sono giacenze e
    medie. In questo mondo si muore di fame mentre il margine alimentare di
    colonia sta al tetto, quindi senza questa riga un amministratore non puo'
    sapere che il proprio distretto sta morendo.

    Restituisce il conteggio e il nuovo cursore. Solo i nuovi: il cumulato
    racconterebbe a ogni tornata la stessa storia sempre piu' grande, e la
    differenza fra "sta succedendo ora" e "e' successo una volta" e' quella che
    rende la riga azionabile.
    """
    per_cella: dict[tuple[int, int], dict[str, int]] = {}
    for morto in dead_agents[int(gia_visti):]:
        try:
            chiave = (int(morto["y"]), int(morto["x"]))
        except (KeyError, TypeError, ValueError):
            continue
        causa = str(morto.get("cause") or "sconosciuta")
        voce = per_cella.setdefault(chiave, {})
        voce[causa] = voce.get(causa, 0) + 1
    return per_cella, len(dead_agents)


def snapshot_compatto_richiesto(config: dict) -> bool:
    """`headless.snapshot_compact`: le sole celle che raccontano qualcosa."""
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    return bool(headless.get("snapshot_compact", False))


def compatta_snapshot(snapshot: dict) -> dict:
    """Tiene le celle occupate, con strutture o esplorate, e lo dichiara.

    **Il numero che lo giustifica.** Sulla griglia della campagna (360x180) uno
    snapshot completo pesa 73 MB perche' elenca 64.800 celle anche dove non c'e'
    mai stato nessuno; quaranta per run sono tre gigabyte, e la richiesta e' di
    rivedere OGNI run nell'analysis frontend, distretti compresi. Il renderer
    disegna le sole celle elencate su fondo scuro; `compact` dice a chi legge
    che le assenti sono vuote e non mancanti; larghezza e altezza restano per
    la scala. Il resto del payload (agenti, metriche, amministratori) non
    cambia: e' cio' che si vuole rivedere.
    """
    celle = snapshot.get("cells") or []
    compatto = dict(snapshot)
    compatto["cells"] = [
        c for c in celle
        if c.get("agents_present") or c.get("structures") or c.get("explored")
    ]
    compatto["compact"] = True
    return compatto
