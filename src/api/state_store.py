from __future__ import annotations

from collections import Counter
from pathlib import Path
import asyncio
import json
import math
import re
import threading
import time

from src.agents.action_space import execute_action
from src.agents.operational_range import (
    operational_range_from_config,
    synchronize_operational_range_config,
)
from src.agents.rule_based_agent import RuleBasedAgent
from src.agents.vitals import death_cause, tick_all_agents
from src.agents.population import maybe_spawn_agents, spawn_initial_agents
from src.core import shell_common
from src.core.kernel import step as _core_step
from src.core.views import AgentView
from src.governors.config import build_administration, build_governor, numeric_metrics
from src.governors.apply import consuntivo_per_regola
from src.governors.observation import build_picture
from src.governors.record import GovernorRecorder, _policy_to_json
from src.llm.cost_tracker import CostTracker
from src.llm.provider_registry import create_agent_providers
from src.social_network.metrics import compute_social_metrics
from src.social_network.network import SocialNetwork
from src.simulation.planetary_coupling import (
    PlanetaryCoupler,
    environmental_layer_enabled,
    sync_static_mars_environment,
    time_scale_from_config,
)
from src.simulation.colony_dynamics import compute_colony_dynamics_metrics
from src.simulation.extreme_events import event_chance_per_step, event_config_enabled
from src.simulation.run_artifacts import (
    action_productivity_metrics,
    save_run_artifacts,
    world_static_base_payload as _world_static_base_payload,
)
from src.world.cell import set_cell_degradation
from src.world.colony_site import find_safe_start
from src.world.initial_support import seed_initial_colony_support
from src.world.world_generator import WorldGenerator
from src.experiments.metrics import composite_agent_score

# GOVERNING NOTE (Task 12 migration, see .superpowers/sdd/task-12-report.md):
# `SimulationController` now runs the hot per-step physics (decide -> execute
# -> vitals -> death -> biology/structure-effects/colony-feedback/wear ->
# extreme events) through `src.core.kernel.step` ("`_core_step`" below,
# renamed on import only to avoid shadowing this module's own `step` method)
# against a vectorized `CoreState`, exactly as `src/core/kernel.py`'s module
# docstring documents. `tick_all_agents`/`death_cause` (src/agents/vitals.py)
# and `update_biology_cells`/`apply_structure_effects`/
# `apply_colony_resource_feedback`/`apply_structure_wear`/
# `research_knowledge_gain` (src/simulation/biology_update.py,
# src/simulation/step_effects.py) are NO LONGER CALLED from `step()` below -
# `src/core/kernel_biology.py`'s `update_cells` (invoked inside
# `_core_step`) is their 1:1 vectorized transcription (see its own module
# docstring for the exact correspondence, including `research_knowledge_gain`
# -> `StepOutcome.knowledge_gain`). `death_cause`/`tick_all_agents` stay
# imported here only because `death_cause` is still used by
# `tests/test_survival_rate.py`'s helpers and both remain the reference this
# module's docstrings point at for what the kernel replicates - see
# `src/core/kernel.py`'s own "the object functions remain the truth" note on
# the sibling `kernel_biology.py` module for the same discipline.
# `observe`/`observe_fast`/`observe_rule_based`/`_observer_for_agent` and
# `_fast_observation_enabled`/`_rule_based_observation_enabled`/
# `_full_biology_update_enabled` below are ALSO no longer called from `step()`
# (the kernel reads the same config flags and picks the same observer
# internally) but `_observer_for_agent`/`_full_biology_update_enabled` are
# kept as importable names here, delegating to `src.core.shell_common`'s
# canonical copies (Task 13 moved the canonical definitions there so
# `src/core/kernel.py` itself could import them without a shell-import cycle -
# see shell_common.py's own module docstring) - kept as thin wrappers, not
# removed, because `tests/core/test_kernel_step.py`/
# `test_kernel_step_bookkeeping.py` import `_full_biology_update_enabled`
# from THIS module directly.


class SimulationController:
    def __init__(self, config: dict | None = None):
        self.running = False
        self.speed = 1
        default_config = {
            "seed": 0,
            "world": {"width": 360, "height": 180, "map_profile": "balanced"},
            "climate": {"enabled": True, "source": "auto", "provider": "netcdf", "scenario": "climatology", "data_path": "data/mcd_runtime", "mcd_path": "data/mcd_runtime"},
            "agents": {"count": 15},
            "simulation": {"max_days": 3652500, "days_per_step": 3650},
            "environmental_layer": {"enabled": True, "role": "background"},
            "social": {"earth_mars_delay_minutes": 12},
        }
        self.planetary = None
        self.last_saved_run_id = ""
        self.last_saved_output_dir = ""
        self.stop_reason = ""
        self.created_by = "gui"  # headless_runner overrides with "cli"
        self._lock = threading.RLock()
        self._step_lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._build_core_state(config or default_config)
        self.log_runtime_event("backend_ready", "backend initialized", run_id=self.run_id)

    def reset(self, config: dict | None = None, preserve_last_saved: bool = False) -> None:
        self._build_core_state(config or self.config, preserve_last_saved=preserve_last_saved)
        self.stop_reason = "reset"
        self.log_runtime_event(
            "scenario_loaded",
            f"scenario loaded: {self.config.get('name', 'gui_run')}",
            run_id=self.run_id,
            agent_count=self._spawned_agent_count,
            llm_count=self._spawned_llm_count,
        )

    def _build_core_state(self, config: dict, preserve_last_saved: bool = False) -> None:
        """Shared by `__init__`/`reset()`: generates the world/agents on the
        OBJECT model exactly as before (`WorldGenerator`, `spawn_initial_agents`,
        `seed_initial_colony_support` - all UNMODIFIED), then converts to the
        vectorized `CoreState` (`CellArrays.from_world`, `AgentArrays.from_agents`,
        `AgentSideState.from_agent`) `step()` drives through `src.core.kernel`.
        The object-model `world`/`agents_obj` built here are local and
        discarded once converted - nothing after this method ever touches
        them again."""
        # `reset()` passa di qui: il consiglio della run precedente possiede un
        # thread e va chiuso PRIMA di costruirne un altro, altrimenti ogni reset
        # ne lascerebbe uno vivo per tutta la vita del processo.
        self._close_governors()
        self.config = _enforce_mcd_runtime(config)
        seed = int(self.config.get("seed", 0))
        world_cfg = self.config.get("world", {})
        width = int(world_cfg.get("width", 50))
        height = int(world_cfg.get("height", 50))
        map_profile = _map_profile_from_config(self.config)
        set_cell_degradation(bool(world_cfg.get("cell_degradation", True)))
        object_world = WorldGenerator(seed, map_profile=map_profile).generate(width, height)
        self.time_scale = time_scale_from_config(self.config)
        self.environmental_layer_enabled = environmental_layer_enabled(self.config)
        if self.environmental_layer_enabled:
            if self.planetary:
                self.planetary.reset(self.config)
            else:
                self.planetary = PlanetaryCoupler(self.config)
            self.planetary.sync_world(object_world)
        else:
            self.planetary = None
            sync_static_mars_environment(object_world, self.time_scale)
        self.cost_tracker = self._create_cost_tracker(self.config)
        agents_cfg = self.config.get("agents", {})
        agent_count = int(agents_cfg.get("count", 15))
        llm_count = int(agents_cfg.get("llm_count", 0))
        llm_providers = create_agent_providers(self.config, llm_count)
        safe_x, safe_y = self._resolve_colony_start(object_world)
        object_agents = spawn_initial_agents(
            agent_count,
            object_world,
            seed=seed,
            llm_count=llm_count,
            llm_providers=llm_providers,
            cost_tracker=self.cost_tracker,
            initial_inventory=agents_cfg.get("initial_inventory") if isinstance(agents_cfg.get("initial_inventory"), dict) else None,
            start_x=safe_x,
            start_y=safe_y,
            operational_range_m=operational_range_from_config(agents_cfg),
            role_distribution=agents_cfg.get("role_distribution"),
            role_preference_randomness=float(
                agents_cfg.get("role_preference_randomness", 0.25)
            ),
        )
        seed_initial_colony_support(object_world, self.config, seed=seed)
        self.world_static_base = _world_static_base_payload(object_world)
        self._spawned_agent_count = agent_count
        self._spawned_llm_count = llm_count

        self.core, self.world = shell_common.build_core_state(object_world, object_agents)
        self._rebuild_agent_views()

        # Founding cohort for survival_rate: births push the population above
        # the initial count, so survival tracks the founders only.
        self.initial_agent_ids = set(object_agents.keys())
        self.social = SocialNetwork(set(object_agents.keys()))
        self.run_id = self._new_run_id(self.config)
        self.output_dir: Path | None = None
        self.step_index = 0  # Current step number (1, 2, 3, ...)
        self.global_metrics: list[dict] = []
        self.validated_actions: list[dict] = []
        self.rejected_actions: list[dict] = []
        self.agent_decisions: list[dict] = []
        self.conversations: list[dict] = []
        self.thoughts: list[dict] = []
        self.dead_agents: list[dict] = []
        #: Quanti decessi lo strato amministrativo ha gia' visto.
        self._morti_gia_visti = 0
        self.replay_events: list[dict] = []
        # True action counters, independent from the capped log lists: metrics
        # must not freeze when max_action_log_rows trips (same fields as the
        # CLI engine, so the two engines stay comparable).
        self.actions_attempted_count = 0
        self.actions_accepted_count = 0
        self.actions_rejected_count = 0
        self.action_counts: Counter[str] = Counter()
        self.rejection_counts: Counter[tuple[str, str]] = Counter()
        self._action_log_truncated = False
        self._replay_log_truncated = False
        self._replay_compactions = 0
        self.active_event = None
        self.upcoming_event = None

        # Il governatore, uno per run (quindi ricostruito a ogni reset). Il suo
        # registro non nasce qui: in questa shell la cartella di output si
        # materializza al primo passo (`_ensure_output_dir`), quindi il registro
        # viene attaccato li' con `Governor.set_recorder`. Senza la sezione
        # `governors` in configurazione resta `None` e nulla di questo percorso
        # viene eseguito.
        # Lo strato SemIf dei singoli agenti vive solo nella shell headless
        # (`agent_coupled_runner`). Qui verrebbe ignorato in silenzio, e un
        # fattore ignorato in silenzio e' proprio cio' che non deve accadere.
        from src.semantic_governance.artifacts import semantic_agents_enabled

        if semantic_agents_enabled(self.config):
            raise ValueError(
                "semantic_agents is supported only by the headless runner "
                "(scripts/run_governor_experiment.py), not by the web shell"
            )
        self._governor = build_governor(self.config, seed, cost_tracker=self.cost_tracker)
        if self._governor is not None:
            # Contatori di scatto per regola, come nella shell headless: il
            # kernel li accumula, `_close_governors` li scarica su file e lo
            # stato live li espone.
            self.core.governor_policy_hits = {}
        self._governor_recorder: GovernorRecorder | None = None
        #: I contatori di scatto al tick precedente, per farne la
        #: differenza: il kernel li accumula sull'intera run.
        self._hits_precedenti: dict[str, int] = {}
        # Lo strato amministrativo, se l'interruttore lo chiede. Nasce accanto
        # al governatore e non dentro di lui: una run puo' avere amministratori
        # senza governo (decentramento puro) o governo senza amministratori
        # (l'assetto storico), e sono due condizioni sperimentali distinte.
        self._administration = build_administration(
            self.config, seed, cost_tracker=self.cost_tracker
        )
        self._admin_records: list[dict] = []
        # Le metriche NUMERICHE dell'ultimo passo concluso: sono cio' che il
        # governatore osserva, in ritardo di un passo per costruzione. Vedi la
        # spec, §3.
        self._last_metrics: dict[str, float] = {}
        if not preserve_last_saved:
            self.last_saved_run_id = ""
            self.last_saved_output_dir = ""
        self.running = False

    def _close_governors(self) -> None:
        """Chiude governatore e registro della run in corso, se ce ne sono.

        Idempotente, e con `getattr` perche' `_build_core_state` la chiama anche
        alla primissima costruzione, quando gli attributi non esistono ancora.
        Chiamata quando la run e' davvero finita — stato terminale o reset — e non
        in pausa: una pausa lascia il governatore vivo, perche' `start()` riprende
        la stessa run e un governatore chiuso non potrebbe piu' pianificare nulla.
        """
        governor = getattr(self, "_governor", None)
        if governor is not None:
            governor.close()
            hits = getattr(self.core, "governor_policy_hits", None)
            if hits is not None and self.output_dir is not None:
                (self.output_dir / "governor_policy_hits.json").write_text(
                    json.dumps(
                        dict(sorted(hits.items(), key=lambda item: item[1], reverse=True)),
                        indent=1, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
        self._governor = None
        recorder = getattr(self, "_governor_recorder", None)
        if recorder is not None:
            recorder.close()
        self._governor_recorder = None

    def _rebuild_agent_views(self) -> None:
        """`self.agents`: dict[str, AgentView] over the CURRENT alive set -
        mirrors the object engines' own `dict[str, BaseAgent]` (only alive
        agents present) closely enough that every existing consumer
        (`compute_colony_dynamics_metrics`, `compute_social_metrics`,
        `save_run_artifacts`, `_current_metrics`, `current_state`) keeps
        working unmodified against `AgentView`'s read-write facade. Rebuilt
        after every `step()` (deaths/spawn change the alive set) and once at
        `_build_core_state` time. Delegates to `shell_common.rebuild_agent_views`
        (shared with `AgentCoupledRunner`, Task 13)."""
        self.agents: dict[str, AgentView] = shell_common.rebuild_agent_views(self.core)

    def _invalidate_structure_cache(self) -> None:
        # Delegates to `shell_common.invalidate_structure_cache` (shared with
        # `AgentCoupledRunner`, Task 13 - see that function's own docstring
        # for the full rationale).
        shell_common.invalidate_structure_cache(self.world)

    def _commit_spawned_agent(self, spawned) -> None:
        """Bridges `src.agents.population.maybe_spawn_agent`'s output (a
        real, fully-formed `RuleBasedAgent` - random psychological profile
        included, see `_assign_psychological_profile`) into the vectorized
        `CoreState`: allocates a new `AgentArrays` row (`AgentArrays.spawn`)
        and a new `AgentSideState`, both populated FROM the spawned object's
        own values - not re-derived. `maybe_spawn_agent` itself (the
        probability/eligibility/trait-randomization logic) runs completely
        unmodified against `self.agents`/`self.world`; this only commits its
        result to the array representation `kernel.step()` actually
        iterates - `AgentArrays.spawn`'s own hardcoded vitals defaults are
        overwritten right after with `spawned`'s real (randomized) values so
        a newborn's traits are not silently flattened to the mean. Delegates
        to `shell_common.commit_spawned_agent` (shared with
        `AgentCoupledRunner`, Task 13)."""
        shell_common.commit_spawned_agent(self.core, self.agents, spawned)

    def start(self) -> None:
        with self._lock:
            if self.running:
                self.log_runtime_event("start_ignored", "play ignored because run is already running", run_id=self.run_id)
                return
            if self._is_terminal_stop_reason():
                previous_run_id = self.run_id
                previous_output_dir = str(self.output_dir) if self.output_dir else self.last_saved_output_dir
                current_config = json.loads(json.dumps(self.config))
                self.reset(current_config, preserve_last_saved=True)
                self.last_saved_run_id = previous_run_id
                self.last_saved_output_dir = previous_output_dir
            try:
                if self.planetary:
                    self.planetary.sync_world(self.world)
                else:
                    sync_static_mars_environment(self.world, self.time_scale)
            except Exception as exc:
                self.running = False
                self.stop_reason = f"runtime_error:{type(exc).__name__}"
                self.log_runtime_event(
                    "run_failed",
                    f"simulation preflight failed: {exc}",
                    run_id=self.run_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                self._ensure_output_dir()
                self._flush_outputs(final=True)
                return
            self.running = True
            self.stop_reason = ""
            self.log_runtime_event("run_started", "play pressed: simulation loop started", run_id=self.run_id)
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_loop, daemon=True)
            self._worker.start()

    def pause(self) -> None:
        # Lock order everywhere: _step_lock -> _lock, so flushing cannot race a step()
        # mutating the output lists on the worker thread.
        with self._step_lock:
            with self._lock:
                self.running = False
                self.stop_reason = "paused_by_user"
                self.log_runtime_event("run_paused", "pause pressed: simulation loop paused", run_id=self.run_id)
                self._flush_outputs(final=False)

    def stop(self) -> None:
        with self._lock:
            self.running = False
        with self._step_lock:
            # Idle simulation (freshly loaded or already reset, nothing simulated):
            # there is nothing to save and nothing to reset. This MUST be a no-op,
            # otherwise a second stop press would flush the empty day-0 state into
            # the previous run's folder and destroy its saved artifacts.
            if self.step_index <= 0 and not self._is_terminal_stop_reason():
                self.stop_reason = "ready_for_new_run"
                self.log_runtime_event("ready_for_new_run", "simulation idle at day 0; nothing to save", run_id=self.run_id)
                return
            current_config = json.loads(json.dumps(self.config))
            stopped_run_id = self.run_id
            already_terminal = self._is_terminal_stop_reason()
            if not already_terminal:
                self.stop_reason = "stopped_by_user"
                self.log_runtime_event("run_stopped", "stop pressed: simulation run stopped", run_id=self.run_id)
                self._ensure_output_dir()
                self._flush_outputs(final=True)
            stopped_output_dir = str(self.output_dir) if self.output_dir else ""
            self.reset(current_config, preserve_last_saved=True)
            self.last_saved_run_id = stopped_run_id
            self.last_saved_output_dir = stopped_output_dir
            self.stop_reason = "ready_for_new_run"
            self.log_runtime_event(
                "ready_for_new_run",
                "stopped run saved; simulation reset to day 0" if not already_terminal else "finished run already saved; simulation reset to day 0",
                saved_run_id=stopped_run_id,
                saved_output_dir=stopped_output_dir,
                new_run_id=self.run_id,
            )

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self.running:
                    return
                if self._reached_stop_threshold():
                    self.running = False
                    return
            try:
                self.step(1)
            except Exception as exc:
                with self._lock:
                    self.running = False
                    self.stop_reason = f"runtime_error:{type(exc).__name__}"
                    self.log_runtime_event(
                        "run_failed",
                        f"simulation step failed: {exc}",
                        run_id=self.run_id,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    self._flush_outputs(final=True)
                return
            time.sleep(max(0.05, 1.0 / max(1, self.speed)))

    def _reached_stop_threshold(self) -> bool:
        """Vero quando la run e' finita — e in quel caso chiude il consiglio.

        Il controllo passa di qui da tutte e quattro le condizioni di stop e da
        entrambe le shell che lo pilotano (`step`, `_run_loop`), quindi e' l'unico
        punto in cui "la run e' finita" e' gia' scritto una volta sola. La pausa
        NON passa di qui, ed e' voluto: riprendere una run in pausa deve trovare il
        consiglio ancora in piedi.
        """
        reached = self._evaluate_stop_threshold()
        if reached:
            self._close_governors()
        return reached

    def _evaluate_stop_threshold(self) -> bool:
        sim_cfg = self.config.get("simulation", {})
        max_days = sim_cfg.get("max_days")
        target_habitability = sim_cfg.get("target_habitability")
        max_steps = self._max_steps()
        if max_steps is not None and self.step_index >= max_steps:
            self.stop_reason = f"max_steps_reached:{max_steps}"
            self.log_runtime_event("run_stopped", f"configured step limit reached: {max_steps}", run_id=self.run_id)
            self._flush_outputs(final=True)
            return True
        if max_days is not None:
            max_days_value = int(max_days)
            if self.world.day >= max_days_value:
                self.stop_reason = f"max_days_reached:{max_days_value}"
                self.log_runtime_event("run_stopped", f"max simulated days reached: {max_days_value}", run_id=self.run_id)
                self._flush_outputs(final=True)
                return True
        if target_habitability is not None and self.world.metrics()["average_habitability"] >= float(target_habitability):
            self.stop_reason = f"target_habitability_reached:{target_habitability}"
            self.log_runtime_event("run_stopped", f"target habitability reached: {target_habitability}", run_id=self.run_id)
            self._flush_outputs(final=True)
            return True
        if not self.agents:
            self.stop_reason = "population_extinct"
            self.log_runtime_event("run_stopped", "population reached zero", run_id=self.run_id)
            self._flush_outputs(final=True)
            return True
        return False

    def step(self, count: int = 1) -> None:
        with self._step_lock:
            for _ in range(max(1, count)):
                if self._reached_stop_threshold():
                    self.running = False
                    return
                self._ensure_output_dir()
                self.step_index += 1
                step = self.step_index
                start_day = self.world.day
                self.log_runtime_event("step_started", f"step {step} started", step=step, start_day=start_day, days_per_step=self.time_scale.days_per_step)

                max_action_log_rows = _max_action_log_rows(self.config)
                max_replay_rows = _max_replay_rows(self.config)
                max_observe_replay_rows = _max_observe_replay_rows_per_step(
                    self.config
                )
                store_memory_logs = _headless_bool(self.config, "store_memory_logs", True)
                include_nearby_agent_ids = _headless_bool(self.config, "store_nearby_agent_ids", False)

                # `_core_step` (src/core/kernel.py `step()`) runs decide ->
                # execute -> vitals -> death sweep -> biology/structure-effects/
                # colony-feedback/wear -> extreme events -> (optional)
                # redistribution as ONE call - see that module's docstring for
                # the exact per-stage correspondence with what this method used
                # to do inline. `self.core.planetary_state` is resynced right
                # before the call because `PlanetaryCoupler.sync_world`/
                # `.advance`/`sync_static_mars_environment` REASSIGN
                # `world.planetary_state` (not mutate it in place) - see
                # `WorldView`'s module docstring - so `self.world.planetary_state`
                # and `self.core.planetary_state` can drift apart between calls
                # unless resynced here.
                self.core.planetary_state = self.world.planetary_state
                # Il governatore non fa mai attendere il passo (salvo regime
                # bloccante): `advance` avvia la chiamata del tick e restituisce
                # la policy IN VIGORE, che puo' essere quella del tick precedente
                # o `None`. Il quadro che riceve e' quello dell'ultimo passo
                # concluso (`_last_metrics`), non del passo che sta per essere
                # calcolato.
                # Il quadro si costruisce SOLO ai confini di tick, come nella
                # shell headless: fra un confine e l'altro `advance` non lo
                # guarderebbe nemmeno.
                if self._governor is not None:
                    if self._governor.is_tick_boundary(step):
                        if self._governor_recorder is None and self.output_dir is not None:
                            # La cartella esiste solo da `_ensure_output_dir`,
                            # qualche riga sopra: e' il primo istante in cui il
                            # registro puo' essere aperto, e va fatto prima del
                            # primo `advance` altrimenti il tick 0 non verrebbe
                            # mai registrato. Il passo 1 e' sempre un confine.
                            self._governor_recorder = GovernorRecorder(
                                self.output_dir / "governor_decisions.jsonl"
                            )
                            self._governor.set_recorder(self._governor_recorder)
                        # Contati una volta per tick, per entrambi i livelli:
                        # vedi la shell headless.
                        morti, self._morti_gia_visti = shell_common.morti_per_cella(
                            self.dead_agents, self._morti_gia_visti
                        )
                        picture = build_picture(
                            step,
                            self._last_metrics,
                            self.core.cells,
                            self.core.agents,
                            self.core.agents.alive_rows(),
                            rule_hits=self._consuntivo_regole(),
                            morti_per_cella=morti,
                        )
                        self.core.governor_policy = self._governor.advance(step, picture)
                        self._avanza_amministratori(step, morti)
                    else:
                        self.core.governor_policy = self._governor.in_force
                elif self._administration is not None:
                    # Nessun governo: gli amministratori hanno bisogno lo stesso
                    # di una cadenza, e usano quella dichiarata.
                    cadenza = max(1, int(
                        (self.config.get("governors") or {}).get("cadence_steps", 20) or 20
                    ))
                    if (step - 1) % cadenza == 0:
                        self._avanza_amministratori(step)
                outcome = _core_step(self.core, step, float(self.time_scale.days_per_step), self.config, None)

                # Drain this step's physics-produced events (rejected_action,
                # structure_built, agent_died, extreme_event_*, ...) into the
                # PERSISTENT event log, renumbering `event_id` so it stays
                # globally monotonic - `_core_step`'s own internal `WorldView`
                # is transient (a fresh instance per call, its `events` list
                # starts empty and would otherwise reset the id counter to 1
                # every step).
                for event in outcome.events:
                    event_copy = dict(event)
                    event_copy["event_id"] = f"{len(self.world.events) + 1:08d}"
                    self.world.events.append(event_copy)

                # Per-agent bookkeeping, rebuilt from `StepOutcome.agent_step_records`
                # (built by the kernel at the exact pre-vitals point this
                # bookkeeping used to be built inline at - see AgentStepRecord's
                # own docstring in src/core/kernel.py for why that timing
                # matters and how it is preserved). Drained via
                # `shell_common.drain_agent_step_records` (shared with
                # `AgentCoupledRunner`). The same headless logging controls now
                # apply to GUI and CLI runs, preventing the controller-backed
                # CLI path from silently retaining hundreds of MiB of rows.
                drained = shell_common.drain_agent_step_records(
                    outcome,
                    step,
                    start_day,
                    world=self.world,
                    social=self.social,
                    validated_actions=self.validated_actions,
                    rejected_actions=self.rejected_actions,
                    replay_events=self.replay_events,
                    conversations=self.conversations,
                    thoughts=self.thoughts,
                    action_counts=self.action_counts,
                    rejection_counts=self.rejection_counts,
                    max_action_log_rows=max_action_log_rows,
                    max_replay_rows=max_replay_rows,
                    max_observe_replay_rows_per_step=max_observe_replay_rows,
                    action_log_truncated=self._action_log_truncated,
                    replay_log_truncated=self._replay_log_truncated,
                    replay_compactions=self._replay_compactions,
                    store_memory_logs=store_memory_logs,
                    cap_thoughts=True,
                    include_nearby_agent_ids=include_nearby_agent_ids,
                )
                self.actions_attempted_count += drained["attempted"]
                self.actions_accepted_count += drained["accepted"]
                self.actions_rejected_count += drained["rejected"]
                self._action_log_truncated = drained["action_log_truncated"]
                self._replay_log_truncated = drained["replay_log_truncated"]
                self._replay_compactions = drained["replay_compactions"]

                # Death bookkeeping (agent_died was already logged by the
                # kernel, inside `outcome.events` above - do not log it twice).
                for row, cause in outcome.deaths:
                    aid = self.core.agents.ids[row]
                    agent_view = AgentView(self.core.agents, row, self.core.side[aid])
                    self.dead_agents.append({"death_step": step, "death_day": start_day, "cause": cause, **agent_view.to_dict()})

                self._rebuild_agent_views()
                self._invalidate_structure_cache()

                if not self.agents:
                    self.stop_reason = "population_extinct"
                    self.log_runtime_event("run_stopped", "population reached zero", run_id=self.run_id)
                    self.global_metrics.append(self._current_metrics(step, start_day))
                    self._flush_outputs(final=True)
                    # Estinzione: la run finisce QUI senza ripassare da
                    # `_reached_stop_threshold`, quindi il consiglio va chiuso a mano.
                    self._close_governors()
                    self.running = False
                    return

                for spawned in maybe_spawn_agents(self.agents, self.world, self.config, int(self.world.day), int(self.config.get("seed", 0))):
                    self._commit_spawned_agent(spawned)
                    self.social.add_agent(spawned.agent_id)
                self.cost_tracker.new_day()

                self.active_event = self.core.metadata.get("active_event")
                self.upcoming_event = self.core.metadata.get("upcoming_event")

                # kernel_biology.py's `update_cells` (run inside `_core_step`
                # above) already wrote the post-biology vegetation total
                # straight into `cells.vegetation`; summing it here is the
                # same aggregate `update_biology_cells`'s own return value
                # used to be (`biology["vegetation"]`), just read directly off
                # the array instead of threaded back out through `StepOutcome`.
                if self.planetary:
                    if outcome.knowledge_gain:
                        self.planetary.state.values["ConoscenzaScientifica"] = self.planetary.state.values.get("ConoscenzaScientifica", 0.0) + outcome.knowledge_gain
                    self.planetary.state.values["sumVeg"] = float(self.core.cells.vegetation.sum())

                if self.planetary:
                    self.planetary.advance(
                        self.world,
                        self.time_scale.days_per_step,
                        refresh_grid=_planetary_grid_refresh_due(self.config, step),
                    )
                else:
                    sync_static_mars_environment(self.world, self.time_scale, recompute_habitability=False)
                self.core.planetary_state = self.world.planetary_state
                # Le metriche si calcolano UNA volta sola: la riga della serie
                # storica e il quadro che il consiglio osservera' al passo
                # successivo sono la stessa cosa. `numeric_metrics` toglie cio'
                # che il quadro non sa rappresentare — con lo strato planetario
                # acceso ci sono una lista e tre stringhe — mentre la serie
                # storica resta completa, perche' quelle voci la riguardano.
                step_metrics = self._current_metrics(step, self.world.day)
                self.global_metrics.append(step_metrics)
                self._last_metrics = numeric_metrics(step_metrics)
                snapshot_interval = _snapshot_interval(self.config)
                if self.output_dir and snapshot_interval > 0 and step % snapshot_interval == 0:
                    (self.output_dir / "world_snapshots" / f"step_{step:06d}_day_{self.world.day:06d}.json").write_text(
                        json.dumps(self._world_snapshot_payload(step), ensure_ascii=False),
                        encoding="utf-8",
                    )
                self.log_runtime_event("step_completed", f"step {step} completed", step=step, day=self.world.day, events=len(self.world.events))
                if self._reached_stop_threshold():
                    self.running = False
                    return
                if self._should_flush_outputs(step):
                    self._flush_outputs(final=False)

    def _record_social_effect(self, agent_id: str, request, accepted: bool, nearby_agents: list[str]) -> None:
        # Delegates to `shell_common.record_social_effect` (shared with
        # `AgentCoupledRunner`, Task 13) - `step()`'s own drain loop above
        # calls the shared function directly; this method stays as a public
        # entry point for anything (tests, routes) that still calls it on the
        # instance.
        shell_common.record_social_effect(self.social, agent_id, request, accepted, nearby_agents, int(getattr(self.world, "step", 0)))

    async def _decide_with_llm_governor(self, agents_list: list, observations: list) -> list:
        cap = _llm_step_cap(self.config)
        if cap <= 0:
            return [
                RuleBasedAgent.decide(agent, obs, self.world) if getattr(agent, "mode", "") == "llm" else agent.decide(obs, self.world)
                for agent, obs in zip(agents_list, observations)
            ]
        requests = [None] * len(agents_list)
        llm_tasks = []
        llm_indexes = []
        llm_used = 0
        for idx, (agent, obs) in enumerate(zip(agents_list, observations)):
            if getattr(agent, "mode", "") == "llm":
                if llm_used < cap:
                    llm_used += 1
                    llm_indexes.append(idx)
                    llm_tasks.append(agent.async_decide(obs, self.world))
                else:
                    self.world.log_event("llm_governor_skipped", "LLM governor cap reached; using deterministic rule-based decision", agent_id=agent.agent_id, step=self.world.step)
                    requests[idx] = RuleBasedAgent.decide(agent, obs, self.world)
            else:
                requests[idx] = agent.decide(obs, self.world)
        if llm_tasks:
            for idx, request in zip(llm_indexes, await asyncio.gather(*llm_tasks)):
                requests[idx] = request
        return requests

    def current_state(self) -> dict:
        # _step_lock keeps the snapshot consistent with a concurrently running step()
        # (agents dict and event list are mutated there under the same lock).
        with self._step_lock, self._lock:
            metrics = dict(self.global_metrics[-1]) if self.global_metrics else self._current_metrics()
            agents = [agent.to_dict() for agent in self.agents.values()]
            events = self.world.events[-50:]
            # Il riferimento al governatore va preso QUI, sotto i lock: il
            # dizionario sotto viene composto fuori, e il thread della run puo'
            # azzerare `_governor` alla fine — fra il controllo e la lettura di
            # `misses` — facendo fallire con un `AttributeError` una semplice
            # lettura di stato.
            governor = self._governor
            policy_in_force = self.core.governor_policy
            policy_hits = dict(self.core.governor_policy_hits or {})
        # json.dumps happily emits NaN/Infinity but browsers' JSON.parse rejects
        # them, silently freezing the GUI on every websocket/REST state update.
        return _json_finite({
            "running": self.running,
            "step": self.step_index,
            "day": self.world.day,
            "simulated_year": self.world.day / 365.25,
            "days_per_step": self.time_scale.days_per_step,
            "years_per_step": self.time_scale.years_per_step,
            "stop_reason": self.stop_reason,
            "max_days": int(self.config.get("simulation", {}).get("max_days", 0) or 0),
            "step_index": self.step_index,
            "run_id": self.run_id,
            "seed": int(self.config.get("seed", 0)),
            "map_profile": self.world.metadata.get("map_profile", _map_profile_from_config(self.config)),
            "requested_map_profile": self.world.metadata.get("requested_map_profile", _map_profile_from_config(self.config)),
            "output_dir": str(self.output_dir) if self.output_dir else "",
            "last_saved_run_id": self.last_saved_run_id,
            "last_saved_output_dir": self.last_saved_output_dir,
            "metrics": metrics,
            "decision_mode": str(
                (self.config.get("agents") or {}).get("decision_mode", "tree")
            ),
            "agents": agents,
            "dead_agents": self.dead_agents,
            "recent_events": events,
            # Accepted action stream for the GUI.  This is intentionally a
            # small tail of the already-persisted replay buffer, not a second
            # unbounded log and not an agent-to-agent communication metric.
            "recent_actions": self.replay_events[-200:],
            "active_event": self.active_event,
            "upcoming_event": self.upcoming_event,
            "api_usage": self.cost_tracker.to_dict(),
            # Che cosa il governatore sta imponendo adesso, e quante volte e'
            # arrivato tardi. `misses` accanto alla policy e non in un log
            # separato: una policy ferma da dieci tick e un governatore che non
            # risponde piu' si vedono uguali dall'esterno, e questo numero li
            # distingue.
            "governors": (
                {
                    "in_force": _policy_to_json(policy_in_force),
                    "misses": governor.misses,
                    # Quante (cella, passo) ogni regola ha catturato finora,
                    # else compreso: e' il modo di vedere DA VIVA una regola
                    # in ombra, senza aspettare la fine della run.
                    "policy_hits": dict(sorted(
                        policy_hits.items(), key=lambda item: item[1], reverse=True,
                    )),
                    # L'ultimo tick in forma compatta: razionale, latenza,
                    # regole proposte e in vigore. Senza, di una run governata
                    # si vedeva soltanto la policy in vigore, cioe' il
                    # risultato e mai il ragionamento.
                    "last_tick": (
                        self._governor_recorder.last_summary()
                        if self._governor_recorder is not None
                        else None
                    ),
                }
                if governor is not None
                else {}
            ),
            # Lo strato amministrativo, mentre la run gira. Senza questo, di
            # una run decentrata si vedrebbe solo l'effetto e mai chi lo ha
            # deciso: un amministratore che si astiene sempre e una run
            # centralizzata producono lo stesso mondo, e vanno distinti.
            "administrators": (
                self._administration.riassunto()
                if getattr(self, "_administration", None) is not None
                else {}
            ),
        })

    def _consuntivo_regole(self) -> tuple:
        """Che cosa ha catturato ogni regola della politica in vigore, da un tick.

        I contatori del kernel sono cumulativi sull'intera run: qui si fa la
        differenza dall'ultimo tick e si ricorda il punto, perche' cio' che
        serve a chi riscrive la legge e' l'effetto della legge SUA, non la somma
        di tutte quelle che l'hanno preceduta.
        """
        cumulati = dict(self.core.governor_policy_hits or {})
        consuntivo = consuntivo_per_regola(
            self.core.governor_policy, cumulati, self._hits_precedenti
        )
        self._hits_precedenti = cumulati
        return consuntivo

    def _avanza_amministratori(self, step: int, morti: dict | None = None) -> None:
        """Una tornata di distretto, subito dopo quella del governo.

        **L'ordine e' il disegno.** La politica non va dal governo alle celle:
        va dal governo agli amministratori, e da questi alle celle. Chiamare
        questo metodo prima di `Governor.advance` farebbe correggere agli
        amministratori la politica del tick precedente.
        """
        amministrazione = self._administration
        if amministrazione is None:
            return
        colonia = self.config.get("colony") or {}
        amministrazione.distretti.imposta_madre(
            (int(colonia.get("start_y", 0)), int(colonia.get("start_x", 0)))
            if "start_x" in colonia and "start_y" in colonia
            else None
        )
        # Il quadro di colonia serve agli amministratori per confrontarsi con il
        # paese, ed e' l'unica grandezza che giustifica un livello locale. Costa
        # una ricostruzione per tornata (misurato: 1,29 ms su 300 agenti), non
        # per passo: a cadenza 20 e' lo 0,03 per cento del tempo di run.
        quadro_colonia = build_picture(
            step, self._last_metrics, self.core.cells, self.core.agents,
            self.core.agents.alive_rows(),
        )
        occupate = self.core.cells.occupancy > 0
        amministrazione.distretti.aggiorna(occupate)
        # L'unico ESITO che lo strato riceve, contato una volta sola per le due
        # shell: vedi `shell_common.morti_per_cella`.
        if morti is None:
            morti, self._morti_gia_visti = shell_common.morti_per_cella(
                self.dead_agents, self._morti_gia_visti
            )
        politiche = amministrazione.advance(
            step,
            self._last_metrics,
            self.core.cells,
            self.core.agents,
            self.core.agents.alive_rows(),
            self.core.governor_policy,
            indicatori_colonia=quadro_colonia.indicators,
            morti_per_cella=morti,
        )
        self.core.district_policies = politiche
        self.core.district_ids = amministrazione.distretti.griglia(occupate.shape)
        riga = {"step": int(step), **amministrazione.riassunto()}
        self._admin_records.append(riga)
        if self.output_dir is not None:
            # Una riga per tornata, nello stesso formato del registro del
            # governatore: e' cio' che il frontend di analisi rilegge per
            # ricostruire il replay senza la run.
            import json as _json

            with (self.output_dir / "administrator_decisions.jsonl").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(_json.dumps(riga, ensure_ascii=False) + "\n")

    def runs(self) -> list[dict]:
        root = Path("outputs/runs")
        if not root.exists():
            return []
        rows = []
        for path in sorted(root.iterdir(), reverse=True):
            if path.is_dir():
                rows.append({"run_id": path.name, "path": str(path)})
        return rows

    def run_detail(self, run_id: str) -> dict:
        path = Path("outputs/runs") / run_id
        summary = path / "final_metrics.json"
        return {"run_id": run_id, "exists": path.exists(), "metrics": json.loads(summary.read_text()) if summary.exists() else {}}

    def log_runtime_event(self, event_type: str, message: str, **data) -> None:
        self.world.log_event(f"runtime_{event_type}", message, **data)

    def _resolve_colony_start(self, world) -> tuple[int, int]:
        """Honor an explicit colony.start_x/start_y, else pick a safe equatorial site.

        The chosen coordinates are written back into the config so initial
        structures and prosperity-spawned settlers use the same site.
        `world` is the temporary object-model `GridWorld` `_build_core_state`
        generates before converting to `CoreState` - this runs before
        `self.world` (the vectorized `WorldView`) exists.
        """
        colony_cfg = self.config.get("colony")
        if not isinstance(colony_cfg, dict):
            colony_cfg = {}
            self.config["colony"] = colony_cfg
        if colony_cfg.get("start_x") is not None and colony_cfg.get("start_y") is not None:
            x = min(world.width - 1, max(0, int(colony_cfg["start_x"])))
            y = min(world.height - 1, max(0, int(colony_cfg["start_y"])))
        else:
            x, y = find_safe_start(world)
        colony_cfg["start_x"] = x
        colony_cfg["start_y"] = y
        return x, y

    def _create_cost_tracker(self, config: dict) -> CostTracker:
        llm_cfg = config.get("llm", {})
        return CostTracker(
            max_calls_per_run=int(llm_cfg.get("max_calls_per_run", 0) or 0),
            max_calls_per_day=int(llm_cfg.get("max_calls_per_day", 0) or 0),
            max_estimated_cost_usd=float(llm_cfg.get("max_estimated_cost_usd", 0.0) or 0.0),
        )

    def _max_steps(self) -> int | None:
        raw = self.config.get("days")
        if raw is None:
            return None
        steps = int(raw)
        return steps if steps > 0 else None

    def _is_terminal_stop_reason(self) -> bool:
        return (
            self.stop_reason == "population_extinct"
            or self.stop_reason.startswith("max_steps_reached:")
            or self.stop_reason.startswith("max_days_reached:")
            or self.stop_reason.startswith("target_habitability_reached:")
        )

    def _current_metrics(self, step: int | None = None, day: float | None = None) -> dict:
        current_step = self.step_index if step is None else step
        current_day = self.world.day if day is None else day
        self._invalidate_structure_cache()
        initial_agent_count = int((self.config.get("agents") or {}).get("count", len(self.agents) or 1))
        # Survival = founders still alive / founders. The population count can
        # exceed the founding cohort through births; survival stays in [0, 1].
        initial_ids = getattr(self, "initial_agent_ids", None)
        if initial_ids:
            survival_rate = sum(1 for aid in initial_ids if aid in self.agents) / max(1, len(initial_ids))
        else:
            survival_rate = min(1.0, len(self.agents) / max(1, initial_agent_count))
        actions_attempted = self.actions_attempted_count
        avg_health = sum(agent.health for agent in self.agents.values()) / max(1, len(self.agents))
        metrics = {
            "step": current_step,
            "day": current_day,
            "simulated_year": current_day / 365.25,
            "days_per_step": self.time_scale.days_per_step,
            **self.world.metrics(),
            **compute_social_metrics(self.social),
            **compute_colony_dynamics_metrics(self.agents, self.world, self.config, self.social),
            "population": len(self.agents),
            "initial_agent_count": float(initial_agent_count),
            "survival_rate": survival_rate,
            "average_agent_health": avg_health,
            "actions_attempted": float(actions_attempted),
            "actions_accepted": float(self.actions_accepted_count),
            "action_acceptance_rate": self.actions_accepted_count / max(1, actions_attempted),
            "action_rejection_rate": self.actions_rejected_count / max(1, actions_attempted),
            "action_diversity": float(len(self.action_counts)),
            **action_productivity_metrics(
                self.action_counts,
                self.rejection_counts,
                accepted=self.actions_accepted_count,
            ),
            "llm_governor_skips": float(sum(1 for event in self.world.events if event.get("type") == "llm_governor_skipped")),
        }
        metrics.update(composite_agent_score(metrics))
        return metrics

    def _ensure_output_dir(self) -> None:
        if self.output_dir:
            return
        # Never write into a folder that already holds a saved run: a rerun with
        # the same name (or a stray stop) must not overwrite finished artifacts.
        candidate = Path("outputs/runs") / self.run_id
        if (candidate / "run_metadata.json").exists():
            base = self.run_id
            suffix = 2
            while (Path("outputs/runs") / f"{base}_{suffix}" / "run_metadata.json").exists():
                suffix += 1
            self.run_id = f"{base}_{suffix}"
            candidate = Path("outputs/runs") / self.run_id
            self.log_runtime_event(
                "output_dir_renamed",
                f"run folder '{base}' already contains a saved run; writing to '{self.run_id}'",
                run_id=self.run_id,
            )
        self.output_dir = candidate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if _snapshot_interval(self.config) > 0:
            (self.output_dir / "world_snapshots").mkdir(parents=True, exist_ok=True)

    def _flush_outputs(self, final: bool) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if _snapshot_interval(self.config) > 0:
            (self.output_dir / "world_snapshots").mkdir(exist_ok=True)
        save_run_artifacts(
            self.output_dir,
            run_id=self.run_id,
            config=self.config,
            world=self.world,
            world_static_base=self.world_static_base,
            agents=self.agents,
            dead_agents=self.dead_agents,
            global_metrics=self.global_metrics,
            validated_actions=self.validated_actions,
            rejected_actions=self.rejected_actions,
            agent_decisions=self.agent_decisions,
            conversations=self.conversations,
            thoughts=self.thoughts,
            replay_events=self.replay_events,
            social=self.social,
            cost_tracker=self.cost_tracker,
            time_scale=self.time_scale,
            current_metrics=self._current_metrics(),
            climate_metadata=self._climate_metadata(),
            stop_reason=self.stop_reason,
            created_by=self.created_by,
            final=final,
            active_event=self.active_event,
            upcoming_event=self.upcoming_event,
            actions_attempted=self.actions_attempted_count,
            actions_rejected=self.actions_rejected_count,
            action_counts=self.action_counts,
            rejection_counts=self.rejection_counts,
            action_log_truncated=self._action_log_truncated,
            replay_log_truncated=self._replay_log_truncated,
            replay_compactions=self._replay_compactions,
            store_memory_logs=_headless_bool(self.config, "store_memory_logs", True),
        )

    def _should_flush_outputs(self, step: int) -> bool:
        interval = _output_flush_interval_steps(self.config)
        return interval <= 1 or step % interval == 0

    def _world_snapshot_payload(self, step: int) -> dict:
        snapshot = self.world.snapshot()
        snapshot.update(
            {
                "step": step,
                "metrics": self._current_metrics(step, self.world.day),
                "agents": [agent.to_dict() for agent in self.agents.values()],
                "dead_agents": list(self.dead_agents),
                "active_event": self.active_event,
                "upcoming_event": self.upcoming_event,
                # Stesso blocco della shell headless: il replay per snapshot
                # colora i distretti da qui.
                "administrators": (
                    self._administration.riassunto()
                    if getattr(self, "_administration", None) is not None
                    else None
                ),
            }
        )
        if shell_common.snapshot_compatto_richiesto(self.config):
            return shell_common.compatta_snapshot(snapshot)
        return snapshot

    def _climate_metadata(self) -> dict:
        if self.planetary:
            return self.planetary.climate_provider.metadata()
        return {"source": "static_mars_baseline", "environmental_layer_enabled": False}

    def _new_run_id(self, config: dict) -> str:
        base = _safe_run_name(str(config.get("name", "gui_run")))
        map_profile = _map_profile_from_config(config)
        if not map_profile or map_profile == "balanced":
            return base
        return _safe_run_name(f"{base}_{map_profile}_seed{int(config.get('seed', 0))}")


def _json_finite(value):
    """Replace non-finite numbers (NaN/Infinity) with 0.0 so the payload stays
    parseable by the browser's strict JSON.parse."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else 0.0


def _safe_run_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_") or "gui_run"


def _map_profile_from_config(config: dict) -> str:
    world_cfg = config.get("world", {}) if isinstance(config.get("world", {}), dict) else {}
    return str(world_cfg.get("map_profile") or config.get("map_profile") or "balanced")


def _enforce_mcd_runtime(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    synchronize_operational_range_config(cfg)
    climate = cfg.get("climate")
    if not isinstance(climate, dict):
        climate = {}
    scenario = str(climate.get("scenario") or "climatology")
    cfg["climate"] = {
        **climate,
        "enabled": True,
        "source": "auto",
        "provider": "netcdf",
        "scenario": scenario,
        "data_path": "data/mcd_runtime",
        "mcd_path": "data/mcd_runtime",
    }
    return cfg


def _event_config_enabled(config: dict, events_cfg: dict) -> bool:
    return event_config_enabled(config, events_cfg)


def _event_chance_per_step(config: dict, events_cfg: dict) -> float:
    return event_chance_per_step(config, events_cfg)


def _max_action_log_rows(config: dict) -> int:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    try:
        return max(1, int(headless.get("max_action_log_rows", 50_000) or 50_000))
    except (TypeError, ValueError):
        return 50_000


def _max_replay_rows(config: dict) -> int:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    try:
        return max(1, int(headless.get("max_replay_rows", 250_000) or 250_000))
    except (TypeError, ValueError):
        return 250_000


def _max_observe_replay_rows_per_step(config: dict) -> int:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    try:
        return max(
            0,
            int(headless.get("max_observe_replay_rows_per_step", 10) or 0),
        )
    except (TypeError, ValueError):
        return 10


def _headless_bool(config: dict, key: str, default: bool) -> bool:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = headless.get(key, default)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def _fast_observation_enabled(config: dict) -> bool:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = headless.get("fast_observation", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def _rule_based_observation_enabled(config: dict) -> bool:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = headless.get("rule_based_observation", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def _observer_for_agent(agent, use_fast_observation: bool, use_rule_observation: bool):
    # Delegates to `shell_common._observer_for_agent` (the canonical copy
    # `src/core/kernel.py` itself imports) - see this module's own header
    # comment for why a wrapper stays here instead of a bare re-export.
    return shell_common._observer_for_agent(agent, use_fast_observation, use_rule_observation)


def _full_biology_update_enabled(config: dict) -> bool:
    # Delegates to `shell_common._full_biology_update_enabled` (see
    # `_observer_for_agent` above, same rationale).
    return shell_common._full_biology_update_enabled(config)


def _snapshot_interval(config: dict) -> int:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = config.get("snapshot_interval", headless.get("snapshot_interval", 0))
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _output_flush_interval_steps(config: dict) -> int:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = headless.get("output_flush_interval_steps", headless.get("flush_interval_steps", 10))
    try:
        return max(1, int(raw or 10))
    except (TypeError, ValueError):
        return 10


def _planetary_grid_refresh_due(config: dict, step: int) -> bool:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = headless.get("planetary_grid_refresh_interval_steps", 100)
    try:
        interval = max(1, int(raw or 100))
    except (TypeError, ValueError):
        interval = 100
    return interval <= 1 or step % interval == 0


def _llm_step_cap(config: dict) -> int:
    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    raw = llm_cfg.get("max_calls_per_step", llm_cfg.get("governor_calls_per_step", 5))
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 5


class LazySimulationController:
    def __init__(self):
        self._instance: SimulationController | None = None
        self._preview_world = None
        self._lock = threading.RLock()

    def _get(self) -> SimulationController:
        with self._lock:
            if self._instance is None:
                self._instance = SimulationController()
            return self._instance

    @property
    def initialized(self) -> bool:
        return self._instance is not None

    def current_state(self) -> dict:
        if self._instance is None:
            return _placeholder_state()
        return self._instance.current_state()

    def read_world(self):
        if self._instance is not None:
            return self._instance.world
        with self._lock:
            if self._preview_world is None:
                self._preview_world = WorldGenerator(0, map_profile="balanced").generate(360, 180)
            return self._preview_world

    def reset(self, config: dict | None = None, preserve_last_saved: bool = False) -> None:
        with self._lock:
            if self._instance is None:
                self._instance = SimulationController(config)
                self._preview_world = None
                return
            self._instance.reset(config, preserve_last_saved=preserve_last_saved)
            self._preview_world = None

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


def _placeholder_state() -> dict:
    return {
        "running": False,
        "step": 0,
        "day": 0,
        "simulated_year": 0.0,
        "days_per_step": 0,
        "years_per_step": 0.0,
        "stop_reason": "awaiting_configuration",
        "max_days": 0,
        "step_index": 0,
        "run_id": "",
        "seed": 0,
        "map_profile": "balanced",
        "requested_map_profile": "balanced",
        "output_dir": "",
        "last_saved_run_id": "",
        "last_saved_output_dir": "",
        "metrics": {},
        "decision_mode": "preferences",
        "agents": [],
        "dead_agents": [],
        "recent_events": [],
        "recent_actions": [],
        "active_event": None,
        "upcoming_event": None,
        "api_usage": {"calls": 0, "successful_calls": 0, "failed_calls": 0, "estimated_cost_usd": 0.0},
    }


controller = LazySimulationController()
