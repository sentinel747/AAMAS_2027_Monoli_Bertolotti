from __future__ import annotations

"""Task 9: the single per-step orchestrator that assembles Tasks 6-8's pieces
(AgentArrays/CellArrays, the read-write views, kernel_vitals.tick_vitals,
kernel_biology.update_cells) into one `step()` call that reproduces - not
reimplements - what BOTH object-model engines do every step.

GOVERNING CONSTRAINT: simulation logic must not change. `step()` calls the
untouched logic modules (src/agents/rule_based_agent.py's `decide`,
src/agents/action_space.py's `execute_action`, src/world/perception.py's
observers, src/simulation/extreme_events.py's `ExtremeEventEngine`) through
the read-write views (src/core/views.py) exactly as the object engines call
them on `BaseAgent`/`GridWorld` - only the container changes.

STEP ORDER, derived by reading both engines line-for-line (identical in
both - see task-9-report.md for the full derivation):
  - src/simulation/agent_coupled_runner.py `run_async`, lines 162-291
  - src/api/state_store.py `step`, lines 338-484

Per step, both engines do, in this exact order:
  1. world.step = step_number; sync world.metadata["active_event"/"upcoming_event"]
     from the persistent ExtremeEventEngine (agent_coupled_runner.py:166-173,
     state_store.py:349-351).
  2. For EVERY agent (in dict-insertion/ascending-row order): build an
     observation via `_observer_for_agent(agent, use_fast_observation,
     use_rule_observation)` (agent_coupled_runner.py:177-179,
     state_store.py:358-360, `_observer_for_agent` at
     agent_coupled_runner.py:478-483) - observations for ALL agents are
     gathered in this SAME pass, before any agent has acted this step.
  3. For EVERY agent, using the observations from step 2: `agent.decide(obs,
     world)` for rule-based agents (`RuleBasedAgent.decide`, the LLM branch is
     out of scope here - see the module docstring on `_RuleBasedAgentOnViews`
     below) - again a full batch pass before any action executes
     (agent_coupled_runner.py:186, state_store.py:366-371).
  4. For EVERY agent, SEQUENTIALLY in the same order: `execute_action(agent,
     agents_dict, world, request)` (agent_coupled_runner.py:193,
     state_store.py:380) - this is where agents actually move/build/collect/etc.
  5. `tick_all_agents` == vectorized `tick_vitals` (kernel_vitals.py) for
     every agent that was alive at the START of this step
     (agent_coupled_runner.py:218, state_store.py:407).
  6. Death sweep: `agent.health <= 0` -> log + remove
     (agent_coupled_runner.py:220-238, state_store.py:410-429). Vectorized:
     `tick_vitals` already flips `agents.alive` in place; `step()` only logs
     and (kernel_vitals.py's documented precondition) recomputes
     `cells.occupancy` from the POST-death alive set, since
     `tick_vitals` only refreeshes it from the PRE-death set and only when
     psychosocial is enabled.
  6a. EXTINCTION SHORT-CIRCUIT (Task 13 refined the condition - see
      task-13-report.md): if the PRE-action alive set was nonempty and the
      post-death alive set is empty - a TRANSITION into extinction this step -
      stop here: no spawn, no biology/structure-effects, no
      events_engine.advance (state_store.py's own `step()` `return`s
      unconditionally on `if not self.agents` for this same case).
      `world.day` stays at its START-of-step value on this path in both
      engines. A step that STARTS with zero agents (`pre_rows.size == 0`,
      e.g. `AgentCoupledRunner`'s own `agents.count: 0` "planetary background
      only" scenario) is NOT this case - see 6a's in-code comment (right
      above the `if` in `step()`) for why it now falls through to stage 8
      onward instead, matching what both object engines' own loops always
      did for a permanently-empty population (their per-agent loops were
      simply no-ops, never a special-cased early return).

      NOT identical to agent_coupled_runner.py's own `if not self.agents and
      agents_list:` in the OTHER direction either: that engine's guard only
      breaks the RUN LOOP (stops the whole multi-step `run_async`, not just
      one `step()` call) when the CURRENT step started non-empty, so a step
      that starts already extinct (mid-run, after a previous step's
      extinction) falls through to spawn/biology/events on THAT engine
      (harmless in practice there: `maybe_spawn_agent` may repopulate before
      biology runs). This kernel has no spawn stage (see 7. below) and
      therefore cannot repopulate mid-step, so short-circuiting on a
      real (nonempty -> empty) extinction transition - matching
      state_store.py, not agent_coupled_runner.py's `agents_list` guard - is
      the deliberate, correct choice for this kernel.
  7. (population spawn - NOT part of this task's scope; the brief's own
     "what to build" list omits it, and it is not one of the Task 6-8 pieces
     step() is assembling.)
  8. `update_biology_cells(..., full_grid=_full_biology_update_enabled(config))`
     + `apply_structure_effects` + `apply_colony_resource_feedback` +
     `apply_structure_wear` == vectorized `update_cells` (kernel_biology.py),
     which is a 1:1 transcription of that exact sub-sequence, in that exact
     order, INCLUDING the `full_grid`/"active" scope (see
     kernel_biology.py's own module docstring and `_active_biology_mask`)
     (agent_coupled_runner.py:254-267, state_store.py:447-458,
     `_full_biology_update_enabled` at agent_coupled_runner.py:486-488/
     state_store.py:844-846 - default "active" scope, `full_grid=False`).
  9. `events_engine.advance(world, agents, config, step_index)` -
     unmodified `ExtremeEventEngine.advance`, called on the POST-death alive
     agent set (agent_coupled_runner.py:268, state_store.py:459).
  10. Redistribution (Task 10): automatic for preference runs unless an
      explicit `config["redistribution"]["enabled"]` overrides it -
      `src.core.needs.compute_needs` +
      `src.core.redistribution.redistribute`, run on the POST-death alive
      agent set, right after biology/wear/events (step 9). This is the ONE
      deliberate model addition on top of the pure representation refactor:
      disabled by default so the baseline stays byte-equivalent to the
      pre-refactor engines (see test_disabled_redistribution_is_a_no_op,
      tests/core/test_kernel_redistribution_wiring.py). Purely a resource
      mechanism - the needs matrix is computed and consumed entirely inside
      this stage and never attached to `StepOutcome` or any view, so no
      decision module can ever see it (binding modification #2,
      .superpowers/sdd/task-10-brief.md). `flows` stays `[]` whenever the
      stage does not run (disabled, or no flow occurred).
  11. Collect events/log: `StepOutcome` bundles deaths, memory events, action
      results, the accumulated event log (WorldView.events +
      kernel_biology's structure_warning events), and knowledge_gain.

Explicitly OUTSIDE `step()`, exactly as in both engines: planetary/climate
advance (`self.planetary.advance(...)` / `sync_static_mars_environment`),
`maybe_spawn_agent`, and any snapshot/metrics/persistence bookkeeping - the
shell calls those around `step()`, unchanged.
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from src.agents.action_space import ActionRequest, ActionType, execute_action
from src.agents.pillars import decision_rng
from src.agents.preference_agent import PreferenceAgent, decide_preferences
from src.agents.rule_based_agent import RuleBasedAgent
from src.agents.vitals import refresh_cell_alerts
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.cell_action_mask import compute_cell_masks
from src.core.cell_proposals import compute_cell_proposals, explore_frontier
from src.core.cell_subset import CellSubset
from src.core.kernel_biology import update_cells
from src.core.kernel_decision import decide_batch
from src.core.kernel_vitals import tick_vitals
from src.core.needs import RedistributionConfig, compute_needs, settled_mask
from src.core.redistribution import redistribute
from src.core.shell_common import (
    _action_log_row_may_be_read,
    _full_biology_update_enabled,
    _observer_for_agent,
)
from src.core.views import AgentSideState, AgentView, WorldView
from src.governors.apply import apply_layered, apply_policy
from src.governors.policy import Policy
from src.simulation.action_logging import build_action_log_row
from src.simulation.extreme_events import ExtremeEventEngine
from src.simulation.run_artifacts import build_replay_event
from src.world.perception import observe, observe_fast, observe_rule_based

# Toggle di sola misura, come MARSABM_FILTER_ORDER e MARSABM_PRECOMPUTE: con
# `0` il menu di cella torna a essere pubblicato su tutta la griglia. I due
# percorsi producono lo stesso stato bit-per-bit (verificato con
# `scripts/parity_harness.py` sulla config reale), quindi non e' un'opzione di
# modellazione ma lo strumento che permette di rimisurare il guadagno con un
# confronto A/B interlacciato invece di doverlo prendere sulla fiducia.
_CELL_SUBSET_ENABLED = os.environ.get("MARSABM_CELL_SUBSET", "1") != "0"

# Restrizione di `compute_needs` alle sole celle insediate. Interruttore proprio
# e non condiviso con quello sopra: i due interventi restringono cose diverse su
# insiemi diversi, e un solo toggle per entrambi renderebbe impossibile
# rimisurare il guadagno dell'uno senza spegnere anche l'altro.
_NEEDS_SUBSET_ENABLED = os.environ.get("MARSABM_NEEDS_SUBSET", "1") != "0"


def _shared_preference_observations(
    world: WorldView,
    agent_views: dict[str, AgentView],
    by_cell: dict[tuple[int, int], list[str]],
    *,
    fast: bool,
) -> dict[str, SimpleNamespace]:
    """Observe once per occupied cell and share the visibility digest."""
    observations: dict[str, SimpleNamespace] = {}
    for (x, y), occupant_ids in by_cell.items():
        representative = agent_views[occupant_ids[0]]
        if fast:
            shared = observe_fast(
                world,
                representative.agent_id,
                x,
                y,
                representative.perception_radius,
                agent_views,
            )
        else:
            cell = world.get_cell(x, y)
            probe_id = f"__cell_vision__:{x}:{y}"
            probe = SimpleNamespace(
                role="cell",
                perception_radius_m=representative.perception_radius_m,
                local_x_m=float(cell.geometry.get("width_m", 0.0)) * 0.5,
                local_y_m=float(cell.geometry.get("height_m", 0.0)) * 0.5,
            )
            # Solo le celle visibili: qui l'osservazione serve al conteggio
            # del registro e a `mark_explored`, non alla decisione.
            shared = observe(
                world,
                probe_id,
                x,
                y,
                representative.perception_radius,
                {probe_id: probe},
                solo_visibili=True,
            )

        nearby_ids: list[str] = []
        for nearby_y in range(y - 1, y + 2):
            for nearby_x in range(x - 1, x + 2):
                nearby_ids.extend(by_cell.get((nearby_x, nearby_y), ()))
        for agent_id in occupant_ids:
            observations[agent_id] = SimpleNamespace(
                nearby_agents=[aid for aid in nearby_ids if aid != agent_id],
                visible_cells=shared.visible_cells,
                observation_mode=f"preferences_shared_{shared.observation_mode}",
            )
    return observations


class _RuleBasedAgentOnViews(AgentView, RuleBasedAgent):
    """`AgentView` + `RuleBasedAgent`'s methods, combined via Python's normal
    multiple-inheritance MRO - NOT a rewrite of either.

    `RuleBasedAgent.decide` (and the ~50 private `_foo` helpers it calls -
    `_settlement_action`, `_scouting_action`, `_water_security_action`, ...)
    read/write `self.x`, `self.health`, `self.inventory`, `self.memory`,
    `self.scout_target`, etc. throughout. Calling the unbound
    `RuleBasedAgent.decide` with a plain `AgentView` as `self` fails the
    moment `decide` reaches one of those `self._foo()` calls: Python resolves
    `self._foo` through `type(self).__mro__`, and a bare `AgentView` has none
    of RuleBasedAgent's private methods (this is exactly the gap
    `tests/core/test_views_writeback.py`'s `_ScoutSettleView` patches
    around, by hand, for exactly two of them, to test
    `_maybe_start_scout`/`_maybe_start_settlement` in isolation).

    This class generalizes that same trick to the whole class instead of two
    methods: MRO is `[_RuleBasedAgentOnViews, AgentView, RuleBasedAgent,
    BaseAgent, object]` (verified: a valid C3 linearization - AgentView's
    only base is `object`, so there is no conflict with RuleBasedAgent's
    `BaseAgent` base). `AgentView.__init__` is first in the MRO and is what
    actually runs on construction (`BaseAgent.__init__`, the dataclass
    `__init__`, never runs) - so the instance dict only ever holds `_a`,
    `row`, `_side`. Every attribute AgentView defines as a property (`x`,
    `y`, `health`, `inventory`, `memory`, `scout_target`, `perception_radius`,
    `survival_priority`, ...) is a DATA descriptor, so it is found and used
    before Python would ever fall back to instance `__dict__` or to
    BaseAgent's dataclass class-level defaults - every one of decide()'s
    reads therefore lands on the real, live, per-agent array/side-state
    value. Names AgentView does NOT define (`decide` itself, and every
    `_foo` helper) fall through the MRO to `RuleBasedAgent`, bound to this
    same instance, so they see the same array-backed `self`.

    Two dataclass fields of `BaseAgent` genuinely have no AgentArrays column
    and no AgentSideState field pre-Task-9 (`perception_radius`,
    `survival_priority`) - both are read (never written) by
    `rule_based_agent.py` and are NOT safe to leave to BaseAgent's
    class-level dataclass default, because both vary per real agent
    (population.py spawns every agent at perception_radius=1, not the
    dataclass default of 3, and randomizes survival_priority per agent).
    `AgentSideState`/`AgentView` were extended with real per-agent fields for
    both (see views.py) rather than left to the accidental class-default
    fallback - the same category of gap Task 7's review already fixed for
    `name`/`role`/`faction`/`local_x_m`/`local_y_m`.

    Deliberately out of scope: the LLM-governor branch
    (`_decide_with_llm_governor`, `agent.async_decide`). Every
    `AgentSideState` this kernel builds is rule-based only (no `mode`/
    `llm_provider_id`/`llm_model` side-state fields exist yet - `mode` falls
    through the MRO to BaseAgent's dataclass class default, `"rule_based"`,
    which is correct for every agent this task supports). Wiring a real LLM
    branch into `step()` is future work, not a Task 9 design decision to
    invent silently.
    """


class _PreferenceAgentOnViews(AgentView, PreferenceAgent):
    """Array-backed agent with PreferenceAgent's inherited binder helpers."""


# `_observer_for_agent` and `_full_biology_update_enabled` are imported from
# src.core.shell_common above rather than copied here a third time (Task 13
# moved them there FROM src.simulation.agent_coupled_runner, which used to be
# this module's source - see shell_common.py's own module docstring for why:
# once `AgentCoupledRunner` itself needed to import `kernel.step` (Task 13),
# importing these two from agent_coupled_runner.py would have closed a
# `kernel -> agent_coupled_runner -> kernel` cycle). `src/api/state_store.py`
# keeps its own byte-identical copy of both (verified line-for-line against
# shell_common.py's), so reusing shell_common's copy here means this module
# can never drift from what either shell actually does. `src/core/shell_common.py`
# does NOT pull in `src/api` (fastapi + the module-level
# `LazySimulationController` singleton state_store.py instantiates at import
# time) or `src/simulation/agent_coupled_runner.py`: `src/core` is a lower
# layer than both, and importing from either here would be a layering
# inversion that made `import src.core.kernel` alone drag in a whole shell.


def _cfg_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _dict_cfg(config: dict, key: str) -> dict:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


@dataclass
class CoreState:
    """Everything one `step()` call needs, vectorized-side. `side` is a
    plain dict `{agent_id: AgentSideState}` mirroring `AgentArrays.ids`/
    `AgentArrays.index` - every alive-or-dead row that ever existed keeps an
    entry (rows are never reused/removed, matching the "ALIVE-ONLY
    ASSUMPTION" convention documented in kernel_biology.py).

    `events_engine` (Task 13 fix): the ONE persistent `ExtremeEventEngine`
    `step()` advances every call - originally stored as `metadata["_events_engine"]`
    (Task 9), which silently broke both shells' `WorldView.snapshot()`/
    `GridWorld.snapshot()` (`{"metadata": dict(self.metadata), ...}`, JSON-
    dumped straight to a `world_snapshots/*.json` file) the first time a run
    with `snapshot_interval > 0` actually produced one - `ExtremeEventEngine`
    is not JSON-serializable, and nothing filtered it back out of `metadata`
    before the dump. Moved to its own field, alongside `side` (another piece
    of per-run state that lives outside the plain-dict `metadata`/
    `planetary_state` JSON-safe pair), so no snapshot payload can ever pick
    it up by accident. Optional/defaulted so every existing direct
    `CoreState(agents=..., cells=..., side=..., planetary_state=...,
    metadata=...)` call (tests, both shells pre-this-fix) keeps constructing
    a valid instance unmodified - `step()` lazily creates one the first time
    it sees `None`, exactly like it used to lazily create one in `metadata`."""

    agents: AgentArrays
    cells: CellArrays
    side: dict[str, AgentSideState]
    planetary_state: dict
    metadata: dict
    events_engine: "ExtremeEventEngine | None" = None
    #: Policy del governatore in vigore per questo passo, o `None`. La
    #: deposita la shell prima di chiamare `step()`; il kernel non interroga
    #: mai un governatore e non attende mai la rete (vedi la spec, §3).
    #: Campo tipizzato e non chiave in `metadata` perche' una chiave stringa si
    #: puo' sbagliare in silenzio, un campo no.
    governor_policy: "Policy | None" = None
    #: Contatori di scatto per regola, accumulati sull'intera run: chiave
    #: `rule_text`, valore quante (cella, passo) la regola ha catturato, piu'
    #: l'else. Li crea la shell quando esiste un governatore; `None` = nessun
    #: conteggio. Coprono i passi con una policy NON vuota in vigore — a policy
    #: vuota o assente il kernel non invoca l'applicazione, per il costo zero
    #: della baseline. Fuori dal digest di parita' per costruzione:
    #: `digest_state` legge agents/cells/side, mai questo campo.
    governor_policy_hits: "dict | None" = None
    #: Le politiche degli amministratori di distretto in vigore per questo
    #: passo: `{identificativo di distretto: Policy}`. Le deposita la shell
    #: insieme a `district_ids`, sullo stesso confine di tick del governatore,
    #: perche' l'amministratore corregge la politica che vede. Vuoto o `None` =
    #: nessun decentramento, e il kernel torna a invocare la sola
    #: `apply_policy`: la baseline resta bit-exact per costruzione.
    district_policies: "dict | None" = None
    #: Griglia `[H, W]` con l'identificativo di distretto per cella, `-1` dove
    #: non ce n'e' (la cella madre e le celle non ancora assegnate, entrambe di
    #: competenza diretta del governo).
    district_ids: "object | None" = None
    #: Fattori semantici per agente sui pilastri (livello 1 dello strato SemIf
    #: degli agenti): `{agent_id: array di N_PILLARS}`. Li deposita la shell
    #: prima di `step()`, come la policy del governatore. `None` = nessuno
    #: strato: `decide_batch` non esegue alcuna operazione in piu' e la baseline
    #: resta bit-exact per costruzione. Fuori dal digest (`digest_state` legge
    #: agents/cells/side). Un agente assente dal dizionario decide rule-based.
    semantic_pillar_factors: "dict | None" = None
    #: Fattori per azione (livello 2): `{agent_id: array di N_ACTIONS}`, applicati
    #: solo alla riga passata ai binder, mai al calcolo dei pilastri.
    semantic_action_factors: "dict | None" = None
    #: Se e' un dict, `decide_batch` vi scrive per agente il margine fra i due
    #: pilastri migliori calcolato SENZA fattori (serve al braccio `hard`).
    pillar_margin_out: "dict | None" = None
    #: Contatori dello strato semantico; aggiornati solo con fattori presenti.
    semantic_counters: "dict | None" = None


@dataclass
class AgentStepRecord:
    """One acted-on agent's per-step bookkeeping, captured INSIDE the execute
    loop below at the exact pre-vitals point both shells' own loops read it
    from (src/api/state_store.py:376-407, src/simulation/agent_coupled_runner.py:
    177-238 - both BEFORE `tick_all_agents`/`tick_vitals` mutates
    health/satiety/oxygen/hydration/fatigue/stress/morale in place at stage 5
    below). This is the seam gap the first Task 12 attempt found missing (see
    .superpowers/sdd/task-12-report.md): `StepOutcome.action_results` alone
    only carries a flattened, partial dict (no `ActionRequest`/`Observation`,
    no pre-vitals agent scalars) and by the time a shell could look again
    after `step()` returns, `tick_vitals` has already overwritten those
    scalars - not just made them inconvenient to re-derive, genuinely gone.

    `action_log_row`/`replay_event` are built here via the SAME shared
    formatters both shells already import
    (`src.simulation.action_logging.build_action_log_row`,
    `src.simulation.run_artifacts.build_replay_event`) - not reimplemented -
    so there is exactly one place that knows how to turn an action into a log
    row, no matter which shell (or this kernel itself) calls it. `thought`
    mirrors `agent.memory.recent_events[-1]` read at the SAME point
    (state_store.py:404-405) - because it is captured here, before
    `tick_vitals`/`ExtremeEventEngine.advance` (stages 5/9 below) can append
    their OWN memory events for this same step, it is unambiguous which
    entry it is, closing the second half of the same gap (see the report's
    "thoughts" analysis) without needing to attribute later stages' memory
    writes after the fact.

    Caps (`max_action_log_rows`, `max_replay_rows`, "only accepted actions get
    a replay event", "only communicate actions become a conversation row")
    are deliberately NOT applied here - those are shell-side bookkeeping
    policy (state_store.py:388-405), not something every consumer of
    `StepOutcome` should be forced to agree on; the shell drains/filters this
    list itself.
    """

    agent_id: str
    request: ActionRequest
    accepted: bool
    result_data: dict
    nearby_agents: list[str]
    # `None` quando nessun consumatore la leggera': azione ACCETTATA con
    # `headless.store_memory_logs` spento, che e' il default della configurazione
    # reale. I due soli lettori sono in `shell_common.drain_agent_step_records`,
    # e ciascuno guarda solo casi in cui la riga esiste: il conteggio dei motivi
    # di rifiuto legge solo le azioni rifiutate, l'accodamento ai log solo con
    # `store_memory_logs` acceso. Vedi `_action_log_row_may_be_read`.
    action_log_row: dict | None
    replay_event: dict
    thought: str | None
    x: int
    y: int


@dataclass
class StepOutcome:
    """Everything one `step()` call produced. `flows` (src/core/redistribution.py's
    own `((from_x, from_y), (to_x, to_y), res_idx, amount)` tuples) is `[]`
    whenever redistribution is disabled (the default) or produced no flow -
    `config["redistribution"]["enabled"]` gates the whole stage, checked at
    the very end of `step()`, right where the object engines have nothing
    equivalent to call yet either. The needs matrix that produced these flows
    is NOT carried on `StepOutcome` - it never leaves
    src/core/redistribution.py's `redistribute()` call (binding modification
    #2, .superpowers/sdd/task-10-brief.md).

    `deaths`/`memory_events` are row-indexed (kernel_vitals.py's own
    vocabulary - see tick_vitals's docstring); `deaths_by_agent`/
    `memory_events_by_agent` carry the same pairs with the row swapped for
    the row's `agent_id` (`AgentArrays.ids[row]`), a friendlier public
    surface for shell consumers that only know agents by id."""

    deaths: list[tuple[int, str]] = field(default_factory=list)
    memory_events: list[tuple[int, str]] = field(default_factory=list)
    action_results: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    knowledge_gain: float = 0.0
    flows: list = field(default_factory=list)
    deaths_by_agent: list[tuple[str, str]] = field(default_factory=list)
    memory_events_by_agent: list[tuple[str, str]] = field(default_factory=list)
    agent_step_records: list[AgentStepRecord] = field(default_factory=list)


def step(state: CoreState, step_index: int, dt_days: float, config: dict, rng) -> StepOutcome:
    """Advance `state` by exactly one step, in the order derived in this
    module's docstring. `rng` is accepted but unused by anything invoked
    here: rule_based_agent.py's own randomness (e.g. the scouting
    should_move roll) goes through the stdlib `random` module, not an
    injected generator, and `ExtremeEventEngine.advance` self-seeds a
    `np.random.default_rng` from `config["seed"]` and `step_index` -
    `rng` is accepted purely as the forward seam for the redistribution
    stage (Task 10+), which will need one."""
    del rng  # unused for now; see docstring - reserved for redistribution.

    agents = state.agents
    cells = state.cells
    metadata = state.metadata
    metadata["step"] = int(step_index)
    # Both engines set world.day to the START-of-step day (agent_coupled_runner.py:167-169's
    # `day = step * days_per_step`, state_store.py:347's `start_day = self.world.day`,
    # equivalent under a constant dt_days per step_index) BEFORE observation/
    # decide/execute/vitals/death/biology/events run, and only advance it to
    # END-of-step (`end_day`) AFTER events_engine.advance - see the matching
    # write near the bottom of this function. Every `wv.log_event` call made
    # in between (agent_died, structure_warning, ...) must see the START day,
    # exactly like both engines' `WorldView`/`GridWorld.log_event` do.
    day_start = (int(step_index) - 1) * float(dt_days)
    day_end = int(step_index) * float(dt_days)
    metadata["day"] = day_start

    headless_cfg = _dict_cfg(config, "headless")
    agents_cfg = _dict_cfg(config, "agents")
    decision_mode = str(agents_cfg.get("decision_mode", "tree")).strip().lower()
    decision_sampling = str(
        agents_cfg.get("decision_sampling", "softmax")
    ).strip().lower()
    # Internal phase-B selector: vectorized became the default only after the
    # reference-vs-batch end-to-end equivalence suite passed for both sampling
    # modes.  It is intentionally not a user-facing UI option.
    decision_engine = str(
        agents_cfg.get("decision_engine", "vectorized")
    ).strip().lower()
    if decision_engine not in {"reference", "vectorized"}:
        raise ValueError(f"unknown preference decision engine: {decision_engine}")
    use_fast_observation = _cfg_bool(headless_cfg.get("fast_observation", True))
    use_rule_observation = _cfg_bool(headless_cfg.get("rule_based_observation", True))
    psychosocial_enabled = bool(_dict_cfg(config, "model").get("psychosocial_enabled", False))
    individual_survival_priority_enabled = _cfg_bool(
        agents_cfg.get("individual_survival_priority_enabled", True)
    )
    delay_minutes = float(_dict_cfg(config, "social").get("earth_mars_delay_minutes", 12.0) or 0.0)
    cell_degradation = _cfg_bool(_dict_cfg(config, "world").get("cell_degradation", True))
    # Conversione ISRU minerali -> materiale da costruzione. Zero = spenta, ed e'
    # il default: la baseline bit-exact non deve muoversi per l'aggiunta di una
    # capacita' che nessuna run precedente aveva. Vedi kernel_biology.update_cells.
    isru_material_rate = float(
        _dict_cfg(config, "colony").get("isru_material_rate", 0.0) or 0.0
    )
    # Priorita' del livello di sviluppo (laboratori, infermerie, stazioni meteo,
    # depositi) nel menu di cella. Il default e' **0,30**; 0,07 era il valore
    # storico e li teneva fuori dal taglio top-k sempre. La priorita' e' inoltre
    # proporzionale al fabbisogno scoperto: vedi la nota in
    # `cell_proposals.compute_cell_proposals`.
    development_priority = float(
        agents_cfg.get("development_build_priority", 0.30) or 0.30
    )
    full_grid_biology = _full_biology_update_enabled(config)

    # Persistent event-engine state lives on `state.events_engine` (like the
    # shells used to keep it on `self`) so it survives across `step()` calls
    # without a global - see `CoreState.events_engine`'s own docstring for
    # why this is a dedicated field and not `metadata["_events_engine"]`
    # (Task 13 fix: the latter broke JSON snapshot payloads).
    events_engine = state.events_engine
    if not isinstance(events_engine, ExtremeEventEngine):
        events_engine = ExtremeEventEngine()
        state.events_engine = events_engine
    metadata["active_event"] = events_engine.active_event
    metadata["upcoming_event"] = events_engine.upcoming_event
    metadata["individual_survival_priority_enabled"] = (
        individual_survival_priority_enabled
    )
    metadata["psychosocial_enabled"] = psychosocial_enabled

    wv = WorldView(cells, agents, state.planetary_state, metadata, state.side)

    # ---- 1-3. Observation + decision, full batch over the PRE-action alive
    # set, before any agent has acted this step (see module docstring). ----
    pre_rows = agents.alive_rows()

    # Task 14 fix (equivalence audit, first divergence found by
    # scripts/compare_v1_v2.py): rebuild `cells.occupancy` fully from the
    # CURRENT alive set here, at the TOP of the step, before decide/execute -
    # matching what `cell.agents_present` already holds on the object engine
    # at this exact point (carried over live from the end of the previous
    # step). `kernel_biology.recompute_habitability`'s density-load/pollution
    # -floor term reads `cells.occupancy` both here (via the biology stage
    # below) AND mid-step, from `WorldView.add_structure` (a BUILD action
    # executed in stage 4 below) - the array must therefore already be
    # correct BEFORE stage 4 runs, not just before biology. From here on,
    # `WorldView.move_agent` (views.py) and the death-sweep decrement below
    # keep it live for the REST of this same step, exactly like
    # `GridWorld.place_agent`/`remove_agent` keep `agents_present` live on
    # the object engine - this full rebuild only needs to run ONCE per step,
    # right here, as the correct starting point.
    cells.occupancy[...] = 0
    if pre_rows.size:
        np.add.at(
            cells.occupancy,
            (agents.y[pre_rows].astype(np.intp), agents.x[pre_rows].astype(np.intp)),
            1,
        )

    # Automatic same-cell logistics happens before decisions.  Preference
    # runs opt in when the key is absent; an explicit enabled=false remains a
    # valid experiment, while historical tree configs keep their old no-op.
    redistribution_cfg = RedistributionConfig.from_config(config)
    raw_redistribution = config.get("redistribution", {})
    raw_redistribution = raw_redistribution if isinstance(raw_redistribution, dict) else {}
    # **Il ripiego dice cio' che dice la dataclass (2026-08-31).** Prima era
    # `decision_mode == "preferences"`, cioe' ACCESA: un terzo default per una
    # quantita' sola, e diceva l'opposto degli altri due
    # (`RedistributionConfig.enabled = False` e
    # `ManualConfigOptions.redistribution_enabled = False`). Non era vivo —
    # ogni percorso reale passa da `build_manual_config`, che la chiave la
    # scrive sempre — ma i sei scenari standard in `standard_v1.yaml` NON
    # hanno quella chiave, quindi un config grezzo dato dritto al kernel
    # sarebbe girato con la logistica accesa mentre ogni run manuale la tiene
    # spenta: due simulazioni diverse sotto lo stesso nome.
    redistribution_enabled = bool(raw_redistribution.get("enabled", False))
    flows: list = []
    metadata["_automatic_logistics_enabled"] = bool(redistribution_enabled)
    metadata["_automatic_logistics_last_deposited"] = 0.0
    metadata["_automatic_logistics_last_withdrawn"] = 0.0
    metadata["_automatic_logistics_last_flow"] = 0.0
    if redistribution_enabled:
        # `compute_needs` azzera il proprio risultato fuori dalle celle insediate
        # (`needs *= settled_mask(...)`), quindi su tutta la mappa produce 64.800
        # righe per ottenerne una non nulla -- misurato: 64.800x di sperpero su
        # configurazione reale, il 15,55% della run. Restringere e' lecito senza
        # ulteriori argomenti: ogni operazione della funzione e' elementwise per
        # cella, e l'unica riduzione (`maintenance_deficit.sum(axis=-1)`) e'
        # sull'asse delle strutture DENTRO la cella. Nessun prodotto matriciale,
        # quindi nessun cambio di ordine di somma: e' la differenza rispetto a
        # `struct_fx`, dove la stessa restrizione aveva rotto la parita'.
        #
        # Il consumatore riceve comunque la griglia intera: `redistribute`
        # indicizza per coordinata, e lo zero fuori sottoinsieme e' esattamente
        # cio' che il percorso storico produce.
        if _NEEDS_SUBSET_ENABLED:
            settled = CellSubset.at_flagged_cells(cells, settled_mask(cells))
            needs = settled.scatter(
                compute_needs(settled.cells, redistribution_cfg),
                (int(cells.H), int(cells.W), C.NN),
            )
        else:
            needs = compute_needs(cells, redistribution_cfg)
        founder_rows = np.fromiter(
            (
                agents.index[aid]
                for aid, side in state.side.items()
                if side.founder_kit_reserved
                and aid in agents.index
                and agents.alive[agents.index[aid]]
            ),
            dtype=np.intp,
        )
        redistribution_out = redistribute(
            agents,
            cells,
            needs,
            redistribution_cfg,
            deposit_exempt_rows=founder_rows,
        )
        flows = redistribution_out["flows"]
        deposited = float(redistribution_out.get("deposited", 0.0) or 0.0)
        withdrawn = float(redistribution_out.get("withdrawn", 0.0) or 0.0)
        flow_amount = sum(float(flow[3]) for flow in flows)
        metadata["_automatic_logistics_last_deposited"] = deposited
        metadata["_automatic_logistics_last_withdrawn"] = withdrawn
        metadata["_automatic_logistics_last_flow"] = flow_amount
        metadata["_automatic_logistics_deposited_total"] = float(
            metadata.get("_automatic_logistics_deposited_total", 0.0)
        ) + deposited
        metadata["_automatic_logistics_withdrawn_total"] = float(
            metadata.get("_automatic_logistics_withdrawn_total", 0.0)
        ) + withdrawn
        metadata["_automatic_logistics_flow_total"] = float(
            metadata.get("_automatic_logistics_flow_total", 0.0)
        ) + flow_amount
        metadata["_automatic_logistics_flow_events"] = int(
            metadata.get("_automatic_logistics_flow_events", 0)
        ) + len(flows)
        metadata["_automatic_logistics_steps"] = int(
            metadata.get("_automatic_logistics_steps", 0)
        ) + 1
        metadata["_automatic_logistics_agent_steps"] = int(
            metadata.get("_automatic_logistics_agent_steps", 0)
        ) + int(pre_rows.size)
        if deposited > 0.0 or withdrawn > 0.0 or flow_amount > 0.0:
            metadata["_automatic_logistics_active_steps"] = int(
                metadata.get("_automatic_logistics_active_steps", 0)
            ) + 1

    if (
        state.semantic_pillar_factors is not None
        or state.semantic_action_factors is not None
    ) and (decision_mode != "preferences" or decision_engine != "vectorized"):
        raise ValueError(
            "semantic agent factors require decision_mode='preferences' and "
            "decision_engine='vectorized'"
        )
    if decision_mode == "preferences":
        agent_views = {
            agents.ids[int(row)]: _PreferenceAgentOnViews(
                agents, int(row), state.side[agents.ids[int(row)]]
            )
            for row in pre_rows
        }
        # The cell publishes a bounded menu once per step. Needs are
        # recomputed after automatic redistribution so agents react to the
        # warehouse state they actually receive, not to the pre-transfer one.
        #
        # Il menu si pubblica SOLO nelle celle che ospitano un colono che
        # decide. Nessuno legge il menu altrove: `decide_batch` e il percorso
        # scalare indicizzano entrambi `[agent.y, agent.x]`. Su una run reale
        # quelle celle sono 1-5 su 64.800, quindi la forma a griglia intera
        # spendeva il 30,9% del passo a comporre un menu per terreno vuoto.
        # Il risultato e' bit-identico: vedi src/core/cell_subset.py per
        # perche' la restrizione e' lecita e come la frontiera di EXPLORE,
        # unica formula che legge i vicini, resti calcolata su tutta la mappa.
        proposal_cells = cells
        proposal_frontier = None
        subset = None
        if _CELL_SUBSET_ENABLED:
            subset = CellSubset.at_agent_cells(cells, agents, pre_rows)
            proposal_cells = subset.cells
            proposal_frontier = subset.gather(explore_frontier(cells))
        proposal_needs = compute_needs(proposal_cells, redistribution_cfg)
        # La policy del governatore entra DENTRO `compute_cell_proposals`,
        # subito prima del taglio top-k, cosi' partecipa alla scelta del menu
        # invece di riordinare un menu gia' composto (su una priorita' appena
        # azzerata dal taglio un peso sarebbe un non-intervento). Le sue
        # condizioni si valutano sugli array di `proposal_cells`, che hanno la
        # stessa forma della `priority` — griglia intera o sottoinsieme — e
        # quindi la stessa aritmetica vale per entrambe le vie. Comunque PRIMA
        # di `publish`: dopo, gli array sono avvolti in `CellIndexedArray`, che
        # accetta letture per `(y, x)` e nessuna scrittura. Senza policy non
        # viene invocato niente, quindi la baseline resta bit-exact per
        # costruzione e non per verifica.
        tilt = None
        policy = state.governor_policy
        distretti = state.district_policies or {}
        if distretti and state.district_ids is not None:
            # Con gli amministratori la politica non va dal governo alle celle:
            # passa dai distretti, e su quelli che hanno riscritto vale la loro
            # catena al posto di quella del governo. `gather` porta la griglia
            # degli identificativi nella stessa forma della `priority`, cosi'
            # che la via compatta e quella intera passino dalla stessa aritmetica.
            ids_locali = (
                subset.gather(state.district_ids)
                if subset is not None
                else state.district_ids
            )

            def tilt(priority, _policy=policy, _distretti=distretti,
                     _ids=ids_locali, _cells=proposal_cells,
                     _hits=state.governor_policy_hits):
                return apply_layered(priority, _policy, _distretti, _ids, _cells, hits=_hits)
        elif policy is not None and policy.rules:
            def tilt(priority, _policy=policy, _cells=proposal_cells,
                     _hits=state.governor_policy_hits):
                return apply_policy(priority, _policy, _cells, hits=_hits)
        proposals = compute_cell_proposals(
            proposal_cells,
            compute_cell_masks(proposal_cells),
            proposal_needs,
            top_k=int(agents_cfg.get("cell_proposal_top_k", 5)),
            # Il pavimento dei cantieri aperti: 1,1 e' il valore con cui sono
            # state misurate tutte le campagne in archivio, e sopra il massimo
            # di ogni altra priorita' di lavoro. Vedi il commento in
            # `cell_proposals`: e' un braccio sperimentale, non una costante.
            pavimento_cantieri=float(
                agents_cfg.get("active_site_priority_floor", 1.1)
            ),
            individual_survival_priority_enabled=(
                individual_survival_priority_enabled
            ),
            frontier=proposal_frontier,
            priority_tilt=tilt,
            development_priority=development_priority,
        )
        if subset is not None:
            proposals = subset.publish(proposals)
        cell_masks = proposals.mask
        by_cell: dict[tuple[int, int], list[str]] = {}
        for row in pre_rows:
            agent_id = agents.ids[int(row)]
            key = (int(agents.x[row]), int(agents.y[row]))
            by_cell.setdefault(key, []).append(agent_id)

        # `by_cell` resta necessario a `_shared_preference_observations` e a
        # `decide_batch`. La lista dei vicini per agente, che veniva costruita
        # qui con una scansione 3x3 per ogni agente a ogni step, e' stata invece
        # rimossa: il suo unico consumatore (`_bind_life`) la scartava senza
        # leggerla. Vedi la nota in src/agents/preference_agent.py.
        if decision_sampling == "softmax":
            uniforms = decision_rng(
                int(config.get("seed", 0)), int(step_index)
            ).random(len(pre_rows))
        else:
            # Greedy has no stochastic draw by contract.
            uniforms = np.zeros(len(pre_rows), dtype=np.float64)
        observations = _shared_preference_observations(
            wv, agent_views, by_cell, fast=use_fast_observation
        )
        if decision_engine == "vectorized":
            requests = decide_batch(
                agents,
                cells,
                state.side,
                pre_rows,
                cell_masks,
                proposals.priority,
                proposals.quota,
                uniforms,
                decision_sampling,
                int(step_index),
                by_cell,
                world=wv,
                agent_views=agent_views,
                individual_survival_priority_enabled=(
                    individual_survival_priority_enabled
                ),
                semantic_pillar_factors=state.semantic_pillar_factors,
                semantic_action_factors=state.semantic_action_factors,
                pillar_margin_out=state.pillar_margin_out,
                semantic_counters=state.semantic_counters,
            )
        else:
            requests = {}
            claims: dict[tuple[int, int, int], int] = {}
            # La spedizione la equipaggia la COMUNITA', non il solo magazzino:
            # `_equip_founder_kit_from_warehouse` ha bisogno dei compagni di cella
            # per attingere alle loro sacche. Chiave stringa, quindi non collide
            # con le chiavi a tupla delle prenotazioni.
            claims["__agenti__"] = agent_views
            for index, row in enumerate(pre_rows):
                agent_id = agents.ids[int(row)]
                agent_view = agent_views[agent_id]
                requests[agent_id] = decide_preferences(
                    agent_view,
                    wv,
                    cell_masks[int(agent_view.y), int(agent_view.x)],
                    float(uniforms[index]),
                    decision_sampling,
                    claims,
                    int(step_index),
                    action_priority_row=proposals.priority[
                        int(agent_view.y), int(agent_view.x)
                    ],
                    quota_row=proposals.quota[
                        int(agent_view.y), int(agent_view.x)
                    ],
                    individual_survival_priority_enabled=(
                        individual_survival_priority_enabled
                    ),
                )
    else:
        agent_views = {
            agents.ids[int(row)]: _RuleBasedAgentOnViews(
                agents, int(row), state.side[agents.ids[int(row)]]
            )
            for row in pre_rows
        }
        observations = {}
        for agent_id, agent_view in agent_views.items():
            observer = _observer_for_agent(
                agent_view, use_fast_observation, use_rule_observation
            )
            observations[agent_id] = observer(
                wv,
                agent_id,
                agent_view.x,
                agent_view.y,
                agent_view.perception_radius,
                agent_views,
            )
        requests = {
            agent_id: agent_view.decide(observations[agent_id], wv)
            for agent_id, agent_view in agent_views.items()
        }

    # ---- 4. Sequential action execution, same (ascending-row) order the
    # object engines get from `list(self.agents.values())`. ----
    action_results: list[dict] = []
    agent_step_records: list[AgentStepRecord] = []
    # Risolto una volta per passo e non per agente: e' una lettura di
    # configurazione, e dentro il ciclo costerebbe 12.000 volte cio' che decide.
    log_rows_may_be_read = _action_log_row_may_be_read(config)
    for r in pre_rows:
        aid = agents.ids[int(r)]
        av = agent_views[aid]
        request = requests[aid]
        result = execute_action(av, agent_views, wv, request)
        action_results.append({
            "agent_id": aid,
            "action": request.action.value,
            "accepted": bool(result.accepted),
            "message": result.message,
            "data": dict(result.data) if isinstance(result.data, dict) else result.data,
        })
        # Per-agent bookkeeping (AgentStepRecord's own docstring has the full
        # rationale): captured HERE, right after this agent's own
        # execute_action and before anyone else's - the same pre-vitals point
        # both shells' loops read it from - via the shared row-builders both
        # shells already import.
        observation = observations[aid]
        result_data = dict(result.data) if isinstance(result.data, dict) else {}
        agent_step_records.append(AgentStepRecord(
            agent_id=aid,
            request=request,
            accepted=bool(result.accepted),
            result_data=result_data,
            nearby_agents=list(getattr(observation, "nearby_agents", []) or []),
            # La riga completa si costruisce solo se qualcuno la leggera':
            # con `store_memory_logs` spento -- il default della config reale --
            # il solo consumatore restante e' il conteggio dei motivi di
            # rifiuto, che riguarda le azioni RIFIUTATE. Misurato: 12.000
            # righe costruite in 40 passi, zero lette, il 7,61% della run.
            # Non e' un interruttore sui log: e' costruire cio' che serve.
            action_log_row=(
                build_action_log_row(
                    int(step_index), day_start, av, request, result, observation
                )
                if log_rows_may_be_read or not result.accepted
                else None
            ),
            replay_event=build_replay_event(int(step_index), day_start, av, request, result),
            thought=(av.memory.recent_events[-1] if av.memory.recent_events else None),
            x=av.x,
            y=av.y,
        ))

    # ---- 5. Vitals (tick_all_agents == tick_vitals), over the SAME pre-action
    # alive set tick_all_agents would have iterated. drank/ate masks mirror
    # vitals.py:153-154's own check (`agent.recent_actions[-1] ==
    # "drink_water"/"eat_food"`) - execute_action only appends to
    # recent_actions on an ACCEPTED action (action_space.py:490, after the
    # early-return on rejection), so this is exactly equivalent to checking
    # `result.accepted and request.action == ActionType.DRINK_WATER/EAT_FOOD`
    # for the LAST action a row took this step, without re-deriving that
    # logic separately from vitals.py's own source of truth. ----
    drank_mask = np.zeros(agents.n, dtype=bool)
    ate_mask = np.zeros(agents.n, dtype=bool)
    recovery_mask = np.zeros(agents.n, dtype=bool)
    for r in pre_rows:
        aid = agents.ids[int(r)]
        recent = state.side[aid].recent_actions
        if not recent:
            continue
        last = recent[-1]
        if last == ActionType.DRINK_WATER.value:
            drank_mask[int(r)] = True
        elif last == ActionType.EAT_FOOD.value:
            ate_mask[int(r)] = True
        elif last == ActionType.PHYSIOLOGICAL_RECOVERY.value:
            recovery_mask[int(r)] = True
            drank_mask[int(r)] = True
            ate_mask[int(r)] = True

    vitals_out = tick_vitals(agents, cells, dt_days, psychosocial_enabled, delay_minutes, drank_mask, ate_mask)
    # Recovery is defined as a full-week cell service. execute_action restores
    # the row before vitals (preventing a critical agent from dying mid-step),
    # and this second assignment makes the observable end-of-step state exact
    # instead of immediately subtracting one week's baseline consumption.
    if np.any(recovery_mask):
        recovery_rows = np.flatnonzero(recovery_mask)
        agents.health[recovery_rows] = 1.0
        agents.satiety[recovery_rows] = 1.0
        agents.oxygen[recovery_rows] = 1.0
        agents.hydration[recovery_rows] = 1.0
        agents.fatigue[recovery_rows] = 0.0
        agents.steps_without_water[recovery_rows] = 0
        agents.steps_without_food[recovery_rows] = 0
        if psychosocial_enabled:
            agents.stress[recovery_rows] = 0.0
            agents.morale[recovery_rows] = 1.0
    deaths_by_agent = [(agents.ids[row], cause) for row, cause in vitals_out["deaths"]]
    memory_events_by_agent = [(agents.ids[row], text) for row, text in vitals_out["memory_events"]]

    # kernel_vitals.py's own memory_events are (row, text) pairs the CALLER
    # must write into the real per-agent memory (it has no side-state
    # reference to do so itself) - mirrors agent.memory.add_event() calls
    # tick_agent_vitals/tick_agent_psychosocial make directly in the object
    # model.
    for row, text in vitals_out["memory_events"]:
        state.side[agents.ids[row]].memory.add_event(text)

    # ---- 5b. refresh_cell_alerts (vitals.py:75-97), called from inside
    # tick_agent_vitals at vitals.py:196 for EVERY agent in the PRE-death
    # alive set (tick_all_agents's own iteration), BEFORE that function's own
    # `if agent.health <= 0.0: return` - so it runs even for a row that dies
    # from THIS tick's vitals, exactly like the object engine. This call was
    # missing entirely from the kernel path (Task 14 equivalence audit): the
    # unmodified function is called directly here, on the views, exactly as
    # kernel_vitals.py's own module docstring always said it would be -
    # nothing about `refresh_cell_alerts` itself is reimplemented. Uses each
    # agent's CURRENT (post-action) cell, matching vitals.py's own call site
    # (tick_agent_vitals's `cell` argument is the agent's cell at the point
    # vitals runs, i.e. after execute_action, before biology).
    for r in pre_rows:
        aid = agents.ids[int(r)]
        av = agent_views[aid]
        refresh_cell_alerts(av, wv.get_cell(av.x, av.y))

    # ---- 6. Death sweep: `agents.alive` is already updated in place by
    # tick_vitals; log "agent_died" (matching both engines' payload) and
    # decrement `cells.occupancy` at each dying agent's own (CURRENT) cell -
    # mirrors `GridWorld.remove_agent` (grid.py:96-101), which the object
    # engines call on every death (agent_coupled_runner.py's own death-sweep
    # loop, `self.world.remove_agent(dead_id, agent.x, agent.y)`). Together
    # with the start-of-step full rebuild above and `WorldView.move_agent`'s
    # own live decrement/increment, `cells.occupancy` now never needs another
    # full-grid rebuild mid-step - it stays exactly as live as
    # `cell.agents_present` is on the object engine.
    for row, cause in vitals_out["deaths"]:
        aid = agents.ids[row]
        side = state.side[aid]
        wv.log_event(
            "agent_died",
            f"{side.name} has died.",
            agent_id=aid,
            role=side.role,
            cause=cause,
            health=float(agents.health[row]),
            satiety=float(agents.satiety[row]),
            oxygen_level=float(agents.oxygen[row]),
            hydration=float(agents.hydration[row]),
            fatigue=float(agents.fatigue[row]),
        )
        dy, dx = int(agents.y[row]), int(agents.x[row])
        cells.occupancy[dy, dx] = max(0, int(cells.occupancy[dy, dx]) - 1)

    post_rows = agents.alive_rows()

    # ---- EXTINCTION SHORT-CIRCUIT, on `pre_rows.size > 0 and post_rows.size
    # == 0` - a TRANSITION into extinction THIS step (agents were alive at
    # the start, none survived vitals/death) - matches state_store.py's own
    # unconditional `if not self.agents: return` for that case, stopping
    # before spawn/biology/structure-effects/events. This kernel has no spawn
    # stage (see 7. below), so a step that transitions to extinct can never
    # repopulate mid-step; short-circuiting here is therefore correct, NOT a
    # claim that this matches agent_coupled_runner.py's own `if not
    # self.agents and agents_list:` in general (see the module docstring's 6a
    # for the full comparison of the two engines' guards).
    #
    # `pre_rows.size > 0` (Task 13 fix, see task-13-report.md): a step that
    # STARTS with zero agents (`AgentCoupledRunner`'s own
    # `test_agent_runner_uses_days_per_step_as_decision_scale`, an
    # `agents.count: 0` "planetary background only" scenario) is NOT a
    # transition into extinction - there is nothing to decide/execute/kill,
    # but biology/structure-effects/events/redistribution have no agent
    # dependency that requires a nonempty set (kernel_biology.update_cells
    # works off `cells`/`planetary_state` alone; `events_engine.advance`
    # tolerates an empty `live_agent_views`), so this kernel now falls
    # through to stage 8 onward exactly like both object engines' own loops
    # always did when population was zero (their `agents_list`/`self.agents`
    # loops were simply no-ops, never a special case) - `metadata["day"]`
    # advances to `day_end` normally. Before this fix, `pre_rows.size == 0`
    # ALSO always satisfied the old unconditional `post_rows.size == 0`
    # check, permanently freezing `world.day` at 0 for any always-empty
    # population - a latent bug this kernel introduced (the object engines
    # never had it) that this fix closes without weakening the real
    # (nonempty -> empty) extinction short-circuit at all. ----
    if pre_rows.size > 0 and post_rows.size == 0:
        return StepOutcome(
            deaths=vitals_out["deaths"],
            memory_events=vitals_out["memory_events"],
            action_results=action_results,
            events=list(wv.events),
            knowledge_gain=0.0,
            flows=flows,
            deaths_by_agent=deaths_by_agent,
            memory_events_by_agent=memory_events_by_agent,
            agent_step_records=agent_step_records,
        )

    # ---- 7. (population spawn: out of scope - see module docstring.) ----

    # ---- 8. Biology + structure effects + colony feedback + structure wear
    # (update_biology_cells + apply_structure_effects +
    # apply_colony_resource_feedback + apply_structure_wear ==
    # kernel_biology.update_cells, already the exact vectorized transcription
    # of that whole sub-sequence in that order). `full_grid_biology` mirrors
    # `headless.biology_update_scope` exactly like both engines'
    # `_full_biology_update_enabled(config)` call (default "active" ->
    # False - see kernel_biology.py's `update_cells`/`_active_biology_mask`
    # for how the vectorized kernel reproduces that scoping). ----
    biology_out = update_cells(
        cells, agents, state.planetary_state, dt_days, cell_degradation,
        full_grid_biology, isru_material_rate,
    )

    # ---- 9. Extreme events, on the POST-death alive agent set only -
    # matches both engines calling `events_engine.advance` with `self.agents`
    # AFTER dead ids were already `del`eted from that dict. ----
    live_agent_views = {
        agents.ids[int(r)]: agent_views[agents.ids[int(r)]]
        for r in post_rows
        if agents.ids[int(r)] in agent_views
    }
    events_engine.advance(wv, live_agent_views, config, step_index)
    metadata["active_event"] = events_engine.active_event
    metadata["upcoming_event"] = events_engine.upcoming_event
    # Both engines advance world.day to end_day right after events_engine.advance
    # (agent_coupled_runner.py:271, state_store.py:462) - see the matching
    # `day_start` write near the top of this function.
    metadata["day"] = day_end

    # ---- 10. Inter-cell flow has already been produced by the automatic
    # logistics pass before decisions. Biology and events do not silently run
    # a second warehouse cycle in the same step. ----
    # ---- 11. Collect events/log. ----
    return StepOutcome(
        deaths=vitals_out["deaths"],
        memory_events=vitals_out["memory_events"],
        action_results=action_results,
        events=list(wv.events) + list(biology_out["events"]),
        knowledge_gain=float(biology_out["knowledge_gain"]),
        flows=flows,
        deaths_by_agent=deaths_by_agent,
        memory_events_by_agent=memory_events_by_agent,
        agent_step_records=agent_step_records,
    )
