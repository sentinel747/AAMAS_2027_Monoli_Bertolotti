from __future__ import annotations

from pathlib import Path
import asyncio
from collections import Counter
import json
import time

from src.agents.rule_based_agent import RuleBasedAgent
from src.agents.operational_range import (
    operational_range_from_config,
    synchronize_operational_range_config,
)
from src.agents.population import maybe_spawn_agents, spawn_initial_agents
from src.core import shell_common
from src.core.kernel import step as _core_step
from src.core.views import AgentView
from src.governors.config import build_administration, build_governor, numeric_metrics
from src.governors.apply import consuntivo_per_regola
from src.governors.observation import build_picture
from src.governors.record import GovernorRecorder
from src.llm.cost_tracker import CostTracker
from src.llm.provider_registry import create_agent_providers
from src.social_network.metrics import compute_social_metrics
from src.social_network.network import SocialNetwork
from src.simulation.step_effects import (
    CAPIENZA_PER_DEPOSITO,
    CAPIENZA_PER_STRUTTURA,
    RISORSE_A_TETTO,
    SCORTA_PRO_CAPITE,
)
from src.simulation.planetary_coupling import (
    PlanetaryCoupler,
    environmental_layer_enabled,
    sync_static_mars_environment,
    time_scale_from_config,
)
from src.world.cell import set_cell_degradation
from src.world.colony_site import find_safe_start
from src.world.initial_support import seed_initial_colony_support
from src.world.world_generator import WorldGenerator
from src.experiments.metrics import composite_agent_score
from src.simulation.colony_dynamics import compute_colony_dynamics_metrics
from src.simulation.run_artifacts import (
    action_productivity_metrics,
    build_replay_event as _build_replay_event,
    save_run_artifacts,
    world_static_base_payload as _world_static_base_payload,
)

# GOVERNING NOTE (Task 13 migration, see .superpowers/sdd/task-13-report.md):
# `AgentCoupledRunner` now runs the hot per-step physics (decide -> execute ->
# vitals -> death -> biology/structure-effects/colony-feedback/wear -> extreme
# events) through `src.core.kernel.step` ("`_core_step`" below, renamed on
# import only to avoid any confusion with this module's own step numbering)
# against a vectorized `CoreState`, exactly like `SimulationController`
# (src/api/state_store.py, Task 12) - both shells share the orchestration in
# `src/core/shell_common.py` (state conversion, agent-view rebuilding, spawn
# commit, social-effect recording, per-step bookkeeping drain) so they cannot
# silently drift apart on how they build state or drain `StepOutcome`. See
# `src/core/kernel.py`'s own module docstring for the exact per-stage
# correspondence with what this module used to do inline.
#
# `execute_action`/`tick_all_agents`/`death_cause` (src/agents/action_space.py,
# src/agents/vitals.py) and `update_biology_cells`/`apply_structure_effects`/
# `apply_colony_resource_feedback`/`apply_structure_wear`/
# `research_knowledge_gain` (src/simulation/biology_update.py,
# src/simulation/step_effects.py) are NO LONGER CALLED from `run_async` below -
# `src/core/kernel_biology.py`'s `update_cells` (invoked inside `_core_step`)
# is their 1:1 vectorized transcription (see its own module docstring for the
# exact correspondence). `step_effects.py`/`biology_update.py` are NOT deleted
# and NOT modified - they stay the object-engine REFERENCE the equivalence
# tests (`tests/core/test_kernel_biology_equivalence.py`,
# `tests/core/test_kernel_step.py`, `tests/core/test_kernel_step_bookkeeping.py`)
# import directly.
#
# `_observer_for_agent`/`_full_biology_update_enabled` are GONE from this
# module (they used to live here, unused-but-kept, and `src/core/kernel.py`
# imported its copies from here) - Task 13 moved their canonical definitions
# to `src/core/shell_common.py` instead, because this module now needs to
# import `src.core.kernel.step` itself: leaving them here would have closed a
# `kernel -> agent_coupled_runner -> kernel` import cycle the moment this
# module started importing the kernel. `src/api/state_store.py` still keeps
# its own byte-identical copy (thin wrappers delegating to shell_common's, as
# of Task 13) for the tests that import `_full_biology_update_enabled` from
# it directly; nothing imported either name from THIS module, so nothing was
# left behind here.
#
# LLM-governor decisions (`_decide_with_llm_governor`, `agent.async_decide`)
# are, per `kernel.py`'s own docstring, "deliberately out of scope": every
# agent is decided through the rule-based path inside `_core_step` regardless
# of `mode` - identical to `SimulationController`'s own Task-12 concession
# (see its report's "Concerns" section). `_decide_with_llm_governor` stays
# defined below (unused by `run_async` now) for the same reason
# `state_store.py` keeps its own copy: no external caller depends on it being
# removed, and removing it would be a bigger diff than the migration itself
# needs. See task-13-report.md for the two existing tests this concession
# required updating (`test_experiment_runner_records_async_llm_attempts`,
# `test_headless_llm_governor_limits_calls_per_step`).


class AgentCoupledRunner:
    def __init__(self, config: dict):
        synchronize_operational_range_config(config)
        self.config = config
        seed = int(config.get("seed", 0))
        self.time_scale = time_scale_from_config(config)
        world_cfg = config.get("world", {})
        map_profile = str(world_cfg.get("map_profile") or config.get("map_profile") or "balanced")
        set_cell_degradation(bool(world_cfg.get("cell_degradation", True)))
        self.world = WorldGenerator(seed, map_profile=map_profile).generate(int(world_cfg.get("width", 50)), int(world_cfg.get("height", 50)))
        
        # Honor an explicit colony site from config, else find a safe zone near the equator
        if "colony" not in config or not isinstance(config.get("colony"), dict):
            config["colony"] = {}
        colony_cfg = config["colony"]
        if colony_cfg.get("start_x") is not None and colony_cfg.get("start_y") is not None:
            safe_x = min(self.world.width - 1, max(0, int(colony_cfg["start_x"])))
            safe_y = min(self.world.height - 1, max(0, int(colony_cfg["start_y"])))
        else:
            safe_x, safe_y = find_safe_start(self.world)
        colony_cfg["start_x"] = safe_x
        colony_cfg["start_y"] = safe_y

        self.environmental_layer_enabled = environmental_layer_enabled(config)
        self.planetary = PlanetaryCoupler(config) if self.environmental_layer_enabled else None
        if self.planetary:
            self.planetary.sync_world(self.world)
        else:
            sync_static_mars_environment(self.world, self.time_scale)
        agents_cfg = config.get("agents", {})
        llm_cfg = config.get("llm", {})
        requested_agent_count = int(agents_cfg.get("count", 5))
        headless_cfg = _headless_config(config)
        aggregate_threshold = int(headless_cfg.get("aggregate_threshold_agents", 50_000) or 50_000)
        self.aggregate_mode = _cfg_bool(headless_cfg.get("aggregate_mode", requested_agent_count >= aggregate_threshold))
        aggregate_sample_agents = max(1, int(headless_cfg.get("aggregate_sample_agents", 200) or 200))
        spawn_count = min(requested_agent_count, aggregate_sample_agents) if self.aggregate_mode else requested_agent_count
        self.cost_tracker = CostTracker(
            max_calls_per_run=int(llm_cfg.get("max_calls_per_run", 0) or 0),
            max_calls_per_day=int(llm_cfg.get("max_calls_per_day", 0) or 0),
            max_estimated_cost_usd=float(llm_cfg.get("max_estimated_cost_usd", 0.0) or 0.0),
        )
        llm_count = min(int(agents_cfg.get("llm_count", 0)), spawn_count)
        llm_providers = create_agent_providers(config, llm_count)
        self.agents = spawn_initial_agents(
            spawn_count,
            self.world,
            seed=seed,
            llm_count=llm_count,
            llm_providers=llm_providers,
            cost_tracker=self.cost_tracker,
            initial_inventory=agents_cfg.get("initial_inventory") if isinstance(agents_cfg.get("initial_inventory"), dict) else None,
            start_x=safe_x,
            start_y=safe_y,
            total_population=requested_agent_count,
            operational_range_m=operational_range_from_config(agents_cfg),
            role_distribution=agents_cfg.get("role_distribution"),
            role_preference_randomness=float(
                agents_cfg.get("role_preference_randomness", 0.25)
            ),
        )
        seed_initial_colony_support(self.world, config, seed=seed)
        self.world_static_base = _world_static_base_payload(self.world)
        self.initial_agent_count = requested_agent_count
        # Founding cohort for survival_rate: births push the population above
        # the initial count, so survival tracks the founders only.
        self.initial_agent_ids = set(self.agents.keys())
        self.aggregate_state = _initial_aggregate_state(config, requested_agent_count, self.world) if self.aggregate_mode else None

        # Task 13: convert the object-model world/agents (built above,
        # UNMODIFIED - same `WorldGenerator`/`spawn_initial_agents`/
        # `seed_initial_colony_support` the pre-migration engine used) into a
        # vectorized `CoreState` plus a persistent `WorldView`/`AgentView`
        # facade, the SAME conversion `SimulationController._build_core_state`
        # uses (`src/core/shell_common.py`, shared - Task 12/13). `object_world`/
        # `object_agents` are local and discarded once converted - nothing
        # after this line ever touches them again (aggregate mode's own
        # `_run_aggregate` never reads `self.agents`/`self.world` for physics
        # either way, so this conversion is harmless overhead there, not a
        # behavior change - see task-13-report.md).
        object_world, object_agents = self.world, self.agents
        self.core, self.world = shell_common.build_core_state(object_world, object_agents)
        # Interruttore sperimentale (2026-09-26), spento per default: vedi
        # `governors.apply._power_coverage`.
        governi_cfg = config.get("governors", {}) if isinstance(config.get("governors"), dict) else {}
        if str(governi_cfg.get("copertura_elettrica", "giacenza")).strip().lower() == "vera":
            self.core.cells.copertura_vera_attiva = True
        self.agents = shell_common.rebuild_agent_views(self.core)

        self.social = SocialNetwork()
        for agent_id in self.agents:
            self.social.add_agent(agent_id)
        self.global_metrics: list[dict] = []
        self.validated_actions: list[dict] = []
        self.rejected_actions: list[dict] = []
        self.agent_decisions: list[dict] = []
        self.conversations: list[dict] = []
        self.thoughts: list[dict] = []
        self.dead_agents: list[dict] = []
        #: Quanti decessi lo strato amministrativo ha gia' visto. Serve a
        #: dargli i morti NUOVI di ogni tornata invece del cumulato.
        self._morti_gia_visti = 0
        self.replay_events: list[dict] = []
        self.action_counts: Counter[str] = Counter()
        self.rejection_counts: Counter[tuple[str, str]] = Counter()
        self.actions_attempted_count = 0
        self.actions_accepted_count = 0
        self.actions_rejected_count = 0
        self._action_log_truncated = False
        self._replay_log_truncated = False
        self._replay_compactions = 0
        self.llm_governor_skips = 0
        self._run_started_at = 0.0
        self.stop_reason = ""

        # Extreme Events State: `kernel.step()` owns the ONE persistent
        # `ExtremeEventEngine` internally (`self.core.events_engine`, see
        # `CoreState`'s own docstring in kernel.py) - matches
        # `SimulationController`'s own Task-12 migration (its
        # `self.events_engine` is gone the same way). `self.active_event`/
        # `self.upcoming_event` are read back from `self.core.metadata` after
        # each `_core_step` call instead.
        self.active_event: dict | None = None
        self.upcoming_event: dict | None = None

        # Il governatore. Nasce qui perche' la configurazione e' gia' nota, ma
        # il suo registro no: la cartella di output arriva soltanto a
        # `run_async`, e viene attaccata li' con `Governor.set_recorder`. Senza
        # la sezione `governors` in configurazione questo e' `None`, il campo
        # `CoreState.governor_policy` resta a `None` e il kernel non chiama
        # nemmeno `apply_policy`: la baseline resta bit-exact per costruzione,
        # non per verifica.
        self._governor = build_governor(
            self.config, int(self.config.get("seed", 0)), cost_tracker=self.cost_tracker
        )
        if self._governor is not None:
            # I contatori di scatto per regola: il kernel li accumula dentro
            # `apply_policy`, la run li scarica in `governor_policy_hits.json`.
            # Senza governo ne' amministratori restano `None` e il kernel non
            # conta niente: la baseline resta bit-exact per costruzione.
            self.core.governor_policy_hits = {}
        self._governor_recorder: GovernorRecorder | None = None
        #: I contatori di scatto al tick precedente, per farne la
        #: differenza: il kernel li accumula sull'intera run.
        self._hits_precedenti: dict[str, int] = {}
        # Lo strato amministrativo. Nasce accanto al governatore e non dentro
        # di lui: una run puo' avere amministratori senza governo e viceversa,
        # e sono due condizioni sperimentali distinte.
        self._administration = build_administration(
            self.config, int(self.config.get("seed", 0)), cost_tracker=self.cost_tracker
        )
        if self._administration is not None and getattr(self._administration, "attivo", False):
            # Gli amministratori possono governare SENZA governatore: le loro
            # catene passano comunque da `apply_layered`, quindi i contatori
            # vanno accesi anche qui. Prima restavano spenti e il braccio
            # «amministratori soli» riportava zero regole applicate pur avendo
            # riscritto, che e' indistinguibile da un braccio inerte.
            self.core.governor_policy_hits = self.core.governor_policy_hits or {}
        self._admin_path: Path | None = None
        # Lo strato SemIf dei singoli agenti: condizione sperimentale distinta
        # da governo e amministrazione. Senza la sezione `semantic_agents` resta
        # `None` e i campi semantici di `CoreState` restano `None`: il kernel non
        # esegue alcuna operazione in piu'.
        from src.semantic_governance.agent_layer import AgentLayerSettings, SemanticAgentLayer

        impostazioni_agenti = AgentLayerSettings.from_config(self.config)
        self._semantic_agents = None
        if impostazioni_agenti is not None:
            agenti_cfg = self.config.get("agents") or {}
            # Stessi default del kernel (`src/core/kernel.py`): decision_mode
            # "tree", decision_engine "vectorized".
            modo = str(agenti_cfg.get("decision_mode", "tree")).strip().lower()
            motore = str(agenti_cfg.get("decision_engine", "vectorized")).strip().lower()
            if modo != "preferences" or motore != "vectorized":
                raise ValueError(
                    "semantic_agents requires agents.decision_mode='preferences' "
                    "and agents.decision_engine='vectorized'"
                )
            run_id = str(
                impostazioni_agenti.semantic.get("run_id")
                or f"semif-agents-seed-{int(self.config.get('seed', 0))}"
            )
            self._semantic_agents = SemanticAgentLayer(impostazioni_agenti, run_id=run_id)
        # Le metriche NUMERICHE dell'ULTIMO passo concluso, cioe' quelle che il
        # governatore osserva. Sono in ritardo di un passo per costruzione (la shell
        # le calcola dopo `_core_step`) ed e' voluto: un governatore agisce su
        # rapporti, e ricalcolarle a inizio passo solo per lui raddoppierebbe la
        # parte piu' cara della contabilita' per passo. Vedi la spec, §3.
        self._last_metrics: dict[str, float] = {}

    def run(self, days: int | None = None, output_dir: str | Path | None = None) -> Path | None:
        return asyncio.run(self.run_async(days, output_dir))

    async def run_async(self, days: int | None = None, output_dir: str | Path | None = None) -> Path | None:
        """La run, piu' la garanzia che il governatore venga sempre chiuso.

        `Governor` possiede un thread e un event loop propri: se nessuno chiama
        `close()` restano appesi anche quando la run finisce per eccezione, e con
        un provider vero una chiamata in volo resterebbe aperta. Il corpo sta in
        `_run_async_inner` invece che dentro un `try` qui, cosi' che il `finally`
        copra ANCHE l'uscita anticipata della modalita' aggregata.
        """
        result = None
        try:
            result = await self._run_async_inner(days, output_dir)
        finally:
            self._close_governors()
            if self._semantic_agents is not None:
                self._semantic_agents.telemetry["kernel_counters"] = dict(
                    self.core.semantic_counters or {}
                )
                self._semantic_agents.close()
        if result is not None:
            from src.semantic_governance.artifacts import write_semantic_artifacts

            write_semantic_artifacts(
                result,
                self.config,
                elapsed_seconds=time.perf_counter() - self._run_started_at,
            )
        return result

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

    def _morti_dall_ultima_tornata(self) -> dict:
        """I decessi nuovi da dare allo strato amministrativo.

        Il conteggio vive in `shell_common` perche' le due shell devono darne
        uno identico: vedi `shell_common.morti_per_cella`.
        """
        per_cella, self._morti_gia_visti = shell_common.morti_per_cella(
            self.dead_agents, self._morti_gia_visti
        )
        return per_cella

    def _avanza_amministratori(self, step: int, morti: dict | None = None) -> None:
        """Una tornata di distretto, subito dopo quella del governo.

        L'ordine e' il disegno: la politica va dal governo agli amministratori
        e da questi alle celle, mai direttamente. Invertirlo farebbe correggere
        loro la politica del tick precedente.
        """
        amministrazione = self._administration
        if amministrazione is None:
            return
        colonia = self.config.get("colony") or {}
        if "start_x" in colonia and "start_y" in colonia:
            amministrazione.distretti.imposta_madre(
                (int(colonia["start_y"]), int(colonia["start_x"]))
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
        self.core.district_policies = amministrazione.advance(
            step,
            self._last_metrics,
            self.core.cells,
            self.core.agents,
            self.core.agents.alive_rows(),
            self.core.governor_policy,
            indicatori_colonia=quadro_colonia.indicators,
            morti_per_cella=morti if morti is not None else self._morti_dall_ultima_tornata(),
        )
        self.core.district_ids = amministrazione.distretti.griglia(occupate.shape)
        if self._admin_path is not None:
            riga = {"step": int(step), **amministrazione.riassunto()}
            with self._admin_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(riga, ensure_ascii=False) + "\n")

    def _close_governors(self) -> None:
        """Chiude governatore e registro. Idempotente: la run puo' finire piu' volte."""
        if self._governor is not None:
            self._governor.close()
            self._governor = None
        if self._governor_recorder is not None:
            self._governor_recorder.close()
            self._governor_recorder = None

    async def _run_async_inner(self, days: int | None = None, output_dir: str | Path | None = None) -> Path | None:
        days = int(days if days is not None else self.config.get("days", 100))
        output_path = Path(output_dir) if output_dir else None
        headless_cfg = _headless_config(self.config)
        snapshot_interval = _snapshot_interval(self.config)
        log_interval = max(1, int(headless_cfg.get("log_interval_steps", self.config.get("log_interval_steps", 10)) or 10))
        use_fast_observation = _cfg_bool(headless_cfg.get("fast_observation", True))
        use_rule_observation = _cfg_bool(headless_cfg.get("rule_based_observation", True))
        max_action_log_rows = int(headless_cfg.get("max_action_log_rows", 50_000) or 50_000)
        max_replay_rows = int(headless_cfg.get("max_replay_rows", 250_000) or 250_000)
        max_observe_replay_rows = max(
            0,
            int(headless_cfg.get("max_observe_replay_rows_per_step", 10) or 0),
        )
        store_memory_logs = _cfg_bool(headless_cfg.get("store_memory_logs", len(self.agents) <= 10_000))
        include_nearby_agent_ids = _cfg_bool(headless_cfg.get("store_nearby_agent_ids", False))
        self._run_started_at = time.perf_counter()
        if self.aggregate_mode:
            if self._semantic_agents is not None:
                # La modalita' aggregata non passa dal kernel decisionale: lo
                # strato verrebbe ignorato in silenzio.
                raise ValueError("semantic_agents is not supported in aggregate mode")
            return self._run_aggregate(days, output_path, log_interval)
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            if snapshot_interval > 0:
                (output_path / "world_snapshots").mkdir(exist_ok=True)
        if output_path is not None and self._administration is not None:
            self._admin_path = output_path / "administrator_decisions.jsonl"
            # Svuotato all'avvio per la stessa ragione del registro del
            # governatore: una run rieseguita nella stessa cartella accodava le
            # proprie tornate a quelle della run che sostituisce. Le singole
            # righe restano in aggiunta, cosi' una run interrotta lascia il suo
            # parziale.
            self._admin_path.write_text("", encoding="utf-8")
        if output_path is not None and self._governor is not None:
            # Il registro esiste solo se la run ha dove scrivere. Senza cartella la
            # run resta valida ma non rieseguibile, ed e' il caso dei test.
            self._governor_recorder = GovernorRecorder(
                output_path / "governor_decisions.jsonl"
            )
            self._governor.set_recorder(self._governor_recorder)
        if output_path is not None and self._semantic_agents is not None:
            self._semantic_agents.attach_output(output_path)
        print(
            "[Headless] start "
            f"steps={days} agents={len(self.agents)} llm_agents={sum(1 for a in self.agents.values() if getattr(a, 'mode', '') == 'llm')} "
            f"llm_step_cap={_llm_step_cap(self.config)} fast_observation={use_fast_observation} "
            f"rule_observation={use_rule_observation} "
            f"snapshots={'off' if snapshot_interval <= 0 else 'every ' + str(snapshot_interval)}",
            flush=True,
        )
        for step in range(days):
            # 1-based step numbering, matching SimulationController.step_index so
            # timeseries/logs from both engines line up row by row.
            step_number = step + 1
            start_day = self.world.day
            # Captured BEFORE `_core_step`, matching the pre-Task-13 loop's
            # own `agents_list = list(self.agents.values())` (agent_coupled_
            # runner.py:236) semantics: a step that STARTS empty (e.g.
            # `agents.count: 0`, a "planetary background only" scenario - see
            # `tests/test_planetary_coupling.py::test_agent_runner_uses_days_per_step_as_decision_scale`)
            # is not extinction, just an empty colony, and must NOT stop the
            # run - only a TRANSITION from nonempty to empty this step is a
            # real extinction event (see the `if not self.agents:` check
            # below, and `kernel.py`'s own matching "EXTINCTION SHORT-CIRCUIT"
            # docstring note for why the kernel itself now draws the same
            # distinction).
            agents_nonempty_at_start = bool(self.agents)

            # `_core_step` (src/core/kernel.py `step()`) runs decide -> execute
            # -> vitals -> death sweep -> biology/structure-effects/colony-
            # feedback/wear -> extreme events -> (optional) redistribution as
            # ONE call - see that module's docstring for the exact per-stage
            # correspondence with what this loop used to do inline (observe/
            # decide/execute loop, `tick_all_agents`, death sweep,
            # `update_biology_cells`/`apply_structure_effects`/
            # `apply_colony_resource_feedback`/`apply_structure_wear`,
            # `events_engine.advance`). `self.core.planetary_state` is
            # resynced right before the call because
            # `PlanetaryCoupler.sync_world`/`.advance`/
            # `sync_static_mars_environment` REASSIGN `world.planetary_state`
            # (not mutate it in place - see `WorldView`'s module docstring),
            # so `self.world.planetary_state` and `self.core.planetary_state`
            # can drift apart between calls unless resynced here (matches
            # `SimulationController.step`).
            self.core.planetary_state = self.world.planetary_state
            # Il governatore non fa mai attendere il passo (salvo regime
            # bloccante): `advance` avvia la chiamata del tick e restituisce la
            # policy IN VIGORE, che puo' essere quella del tick precedente o
            # `None`. Il quadro che riceve e' quello dell'ultimo passo concluso
            # (`_last_metrics`), non del passo che sta per essere calcolato.
            #
            # Il quadro si costruisce SOLO ai confini di tick: fra un confine e
            # l'altro `advance` non lo guarderebbe nemmeno, e costruirlo a ogni
            # passo costava lo 0,58% del passo (1,29 ms su 300 agenti) per
            # buttarne via diciannove su venti a cadenza 20.
            if self._governor is not None:
                if self._governor.is_tick_boundary(step_number):
                    # I morti dall'ultimo tick si contano UNA volta e vanno a
                    # entrambi i livelli: il contatore avanza, e chiamarlo due
                    # volte darebbe al secondo un elenco vuoto.
                    morti = self._morti_dall_ultima_tornata()
                    picture = build_picture(
                        step_number,
                        self._last_metrics,
                        self.core.cells,
                        self.core.agents,
                        self.core.agents.alive_rows(),
                        rule_hits=self._consuntivo_regole(),
                        morti_per_cella=morti,
                    )
                    self.core.governor_policy = self._governor.advance(step_number, picture)
                    self._avanza_amministratori(step_number, morti)
                else:
                    self.core.governor_policy = self._governor.in_force
            elif self._administration is not None:
                cadenza = max(1, int(
                    (self.config.get("governors") or {}).get("cadence_steps", 20) or 20
                ))
                if (step_number - 1) % cadenza == 0:
                    self._avanza_amministratori(step_number)
            if self._semantic_agents is not None:
                self._semantic_agents.advance(step_number, self.core)
            outcome = _core_step(self.core, step_number, float(self.time_scale.days_per_step), self.config, None)

            # Drain this step's physics-produced events (rejected_action,
            # structure_built, agent_died, extreme_event_*, ...) into the
            # PERSISTENT event log, renumbering `event_id` so it stays
            # globally monotonic - `_core_step`'s own internal `WorldView` is
            # transient (a fresh instance per call, its `events` list starts
            # empty and would otherwise reset the id counter to 1 every step).
            for event in outcome.events:
                event_copy = dict(event)
                event_copy["event_id"] = f"{len(self.world.events) + 1:08d}"
                self.world.events.append(event_copy)

            # Per-agent bookkeeping, rebuilt from `StepOutcome.agent_step_records`
            # (built by the kernel at the exact pre-vitals point this
            # bookkeeping used to be built inline at - see `AgentStepRecord`'s
            # own docstring in src/core/kernel.py). Drained via
            # `shell_common.drain_agent_step_records` (shared with
            # `SimulationController`, Task 12/13); `store_memory_logs`
            # reproduces this shell's own pre-Task-13 large-population gate,
            # `cap_thoughts=False` reproduces its own pre-Task-13 uncapped
            # `thoughts` list (unlike the GUI shell, which caps it - see that
            # function's own docstring).
            drained = shell_common.drain_agent_step_records(
                outcome,
                step_number,
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
                cap_thoughts=False,
                include_nearby_agent_ids=include_nearby_agent_ids,
            )
            self.actions_attempted_count += drained["attempted"]
            self.actions_accepted_count += drained["accepted"]
            self.actions_rejected_count += drained["rejected"]
            self._action_log_truncated = drained["action_log_truncated"]
            self._replay_log_truncated = drained["replay_log_truncated"]
            self._replay_compactions = drained["replay_compactions"]

            # Death bookkeeping (agent_died was already logged by the kernel,
            # inside `outcome.events` above - do not log it twice).
            for row, cause in outcome.deaths:
                aid = self.core.agents.ids[row]
                agent_view = AgentView(self.core.agents, row, self.core.side[aid])
                self.dead_agents.append({"death_step": step_number, "death_day": start_day, "cause": cause, **agent_view.to_dict()})

            self.agents = shell_common.rebuild_agent_views(self.core)
            shell_common.invalidate_structure_cache(self.world)

            if not self.agents and agents_nonempty_at_start:
                # A TRANSITION into extinction this step (population was
                # nonempty at the start, now empty) - matches
                # `agents_nonempty_at_start`'s own docstring above and
                # `kernel.py`'s own matching "EXTINCTION SHORT-CIRCUIT" note:
                # `_core_step` takes the same distinction and leaves
                # `metadata["day"]` (hence `self.world.day`) at its
                # START-of-step value on this path, matching this loop's own
                # pre-Task-13 `self.world.day = day` (the start-of-step day)
                # for a real extinction. A step that STARTS empty (population
                # was already zero) falls through instead - `_core_step`
                # itself already fell through to advance biology/events/day
                # normally for that case (see kernel.py), so there is nothing
                # to special-case here beyond not stopping the run.
                self.global_metrics.append(self._current_metrics(step_number, self.world.day))
                self.stop_reason = "population_extinct"
                break

            for spawned in maybe_spawn_agents(self.agents, self.world, self.config, int(self.world.day), int(self.config.get("seed", 0))):
                self._commit_spawned_agent(spawned)
                self.social.add_agent(spawned.agent_id)
            self.cost_tracker.new_day()

            self.active_event = self.core.metadata.get("active_event")
            self.upcoming_event = self.core.metadata.get("upcoming_event")

            # kernel_biology.py's `update_cells` (run inside `_core_step`
            # above) already wrote the post-biology vegetation total straight
            # into `cells.vegetation`; summing it here is the same aggregate
            # `update_biology_cells`'s old return value was
            # (`biology["vegetation"]`), just read directly off the array
            # instead of threaded back out through `StepOutcome`.
            if self.planetary:
                if outcome.knowledge_gain:
                    self.planetary.state.values["ConoscenzaScientifica"] = self.planetary.state.values.get("ConoscenzaScientifica", 0.0) + outcome.knowledge_gain
                self.planetary.state.values["sumVeg"] = float(self.core.cells.vegetation.sum())

            if self.planetary:
                self.planetary.advance(
                    self.world,
                    self.time_scale.days_per_step,
                    refresh_grid=_planetary_grid_refresh_due(self.config, step_number, days),
                )
            else:
                sync_static_mars_environment(self.world, self.time_scale, recompute_habitability=False)
            self.core.planetary_state = self.world.planetary_state
            end_day = self.world.day
            # Le metriche si calcolano UNA volta sola: la riga della serie storica e
            # il quadro che il consiglio osservera' al passo successivo sono la
            # stessa cosa. Ricalcolarle per il governatore raddoppierebbe la parte
            # piu' cara della contabilita' per passo senza dire nulla di piu'.
            # `numeric_metrics` toglie cio' che il quadro non sa rappresentare —
            # con lo strato planetario acceso ci sono una lista e tre stringhe —
            # e la serie storica resta invece completa, perche' quelle voci la
            # riguardano.
            step_metrics = self._current_metrics(step_number, end_day)
            self.global_metrics.append(step_metrics)
            self._last_metrics = numeric_metrics(step_metrics)
            if output_path and snapshot_interval > 0 and (step_number % snapshot_interval == 0 or step == days - 1):
                (output_path / "world_snapshots" / f"step_{step_number:06d}_day_{end_day:06d}.json").write_text(
                    json.dumps(self._world_snapshot_payload(step_number, end_day), ensure_ascii=False),
                    encoding="utf-8",
                )
            if step_number % log_interval == 0 or step == days - 1 or not self.agents:
                self._print_headless_progress(step_number, days, end_day)
        self.stop_reason = self.stop_reason or f"max_steps_reached:{days}"
        if output_path and len(self.global_metrics) > 0:
            self.save(output_path)
        if output_path and self.core.governor_policy_hits is not None:
            # Quante (cella, passo) ogni regola ha catturato, else compreso.
            # E' la risposta di run all'ombra fra regole: una regola larga
            # scritta per prima affama le successive, e questo file lo dice
            # senza strumentare nulla. Copre i passi con una policy non vuota
            # in vigore. Ordinato per conteggio: la prima riga e' la politica
            # che ha governato davvero.
            (output_path / "governor_policy_hits.json").write_text(
                json.dumps(
                    dict(sorted(
                        self.core.governor_policy_hits.items(),
                        key=lambda item: item[1], reverse=True,
                    )),
                    indent=1, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return output_path

    def _run_aggregate(self, days: int, output_path: Path | None, log_interval: int) -> Path | None:
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        state = self.aggregate_state or _initial_aggregate_state(self.config, self.initial_agent_count, self.world)
        # **L'avviso non e' un abbellimento.** La modalita' si accende DA SOLA
        # sopra `aggregate_threshold_agents`, e chi lancia la run puo' non
        # accorgersi di aver cambiato modello. Misurata contro l'ABM sulla
        # stessa configurazione, sovrastima la popolazione del 43% e non
        # produce alcun decesso dove l'ABM ne conta trentuno.
        print(
            "[HeadlessAggregate] ATTENZIONE: questa NON e' la simulazione ad "
            "agenti. E' una estrapolazione aggregata a infrastruttura fissa, "
            "non calibrata sull'ABM e non utilizzabile per confrontare "
            "politiche. Vedi docs/benchmarks/2026-08-30-lo-stallo-della-colonia.md.",
            flush=True,
        )
        print(
            "[HeadlessAggregate] start "
            f"steps={days} population={state['population']:.0f} llm_step_cap={_llm_step_cap(self.config)}",
            flush=True,
        )
        for step in range(days):
            day = step * self.time_scale.days_per_step
            end_day = (step + 1) * self.time_scale.days_per_step
            self.world.step = step
            self.world.day = end_day
            _advance_aggregate_state(state, self.config, self.time_scale.days_per_step)
            self.global_metrics.append(_aggregate_metrics(step, end_day, state, self.config, self.initial_agent_count))
            if (step + 1) % log_interval == 0 or step == days - 1:
                elapsed = max(1.0e-9, time.perf_counter() - self._run_started_at)
                macro_updates_per_second = (step + 1) / elapsed
                print(
                    "[HeadlessAggregate] "
                    f"step={step + 1}/{days} day={end_day:.0f} population={state['population']:.0f} "
                    f"food={state['food']:.1f} water={state['water']:.1f} materials={state['construction_material']:.1f} "
                    f"tools={state['tools']:.1f} occupied_m2={state['occupied_m2']:.0f} speed={macro_updates_per_second:.0f} macro-updates/s",
                    flush=True,
                )
        self.aggregate_state = state
        if output_path:
            self.save(output_path)
        return output_path

    async def _decide_with_llm_governor(self, agents_list: list, observations: list, step: int) -> list:
        cap = _llm_step_cap(self.config)
        if cap <= 0:
            return [RuleBasedAgent.decide(agent, obs, self.world) if getattr(agent, "mode", "") == "llm" else agent.decide(obs, self.world) for agent, obs in zip(agents_list, observations)]

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
                    self.llm_governor_skips += 1
                    requests[idx] = RuleBasedAgent.decide(agent, obs, self.world)
            else:
                requests[idx] = agent.decide(obs, self.world)
        if llm_tasks:
            for idx, request in zip(llm_indexes, await asyncio.gather(*llm_tasks)):
                requests[idx] = request
        return requests

    def _print_headless_progress(self, step_done: int, total_steps: int, day: float) -> None:
        elapsed = max(1.0e-9, time.perf_counter() - self._run_started_at)
        steps_per_second = step_done / elapsed
        agent_steps = self.actions_attempted_count / elapsed
        top_actions = ", ".join(f"{name}:{count}" for name, count in self.action_counts.most_common(4)) or "none"
        acceptance = self.actions_accepted_count / max(1, self.actions_attempted_count)
        print(
            "[Headless] "
            f"step={step_done}/{total_steps} day={day:.0f} population={len(self.agents)} deaths={len(self.dead_agents)} "
            f"acceptance={acceptance:.2%} llm_calls={self.cost_tracker.calls} llm_skips={self.llm_governor_skips} "
            f"speed={steps_per_second:.2f} steps/s {agent_steps:.0f} agent-steps/s actions=[{top_actions}]",
            flush=True,
        )

    def _world_snapshot_payload(self, step: int, day: int | float) -> dict:
        snapshot = self.world.snapshot()
        snapshot.update(
            {
                "step": step,
                "day": day,
                "metrics": self._current_metrics(step, day),
                "agents": [agent.to_dict() for agent in self.agents.values()],
                "dead_agents": list(self.dead_agents),
                "active_event": self.active_event,
                "upcoming_event": self.upcoming_event,
                # Lo strato amministrativo com'era all'ultima tornata: celle di
                # ogni distretto e decisione presa. E' cio' che il replay per
                # snapshot colora; senza, di una run decentrata si rivedrebbe
                # solo l'effetto e mai chi lo ha deciso.
                "administrators": (
                    self._administration.riassunto()
                    if self._administration is not None
                    else None
                ),
            }
        )
        if shell_common.snapshot_compatto_richiesto(self.config):
            return shell_common.compatta_snapshot(snapshot)
        return snapshot

    def _record_social_effect(self, agent_id: str, request, accepted: bool, nearby_agents: list[str]) -> None:
        # Delegates to `shell_common.record_social_effect` (shared with
        # `SimulationController`, Task 12/13) - `run_async`'s own drain loop
        # above calls the shared function directly; this method stays as a
        # public entry point for anything (tests) that still calls it on the
        # instance (see tests/test_social_network.py).
        shell_common.record_social_effect(self.social, agent_id, request, accepted, nearby_agents, int(getattr(self.world, "step", 0)))

    def _commit_spawned_agent(self, spawned) -> None:
        """Bridges `maybe_spawn_agent`'s output into the vectorized
        `CoreState`, exactly like `SimulationController._commit_spawned_agent`
        (Task 12) - delegates to `shell_common.commit_spawned_agent` (shared,
        Task 13). Kept as a public method (not inlined at the one call site
        in `run_async`) because `tests/test_survival_rate.py`'s `_force_birth`
        helper looks it up via `getattr(engine, "_commit_spawned_agent",
        None)` to detect a core-backed engine."""
        shell_common.commit_spawned_agent(self.core, self.agents, spawned)

    def _current_metrics(self, step: int, day: float) -> dict:
        shell_common.invalidate_structure_cache(self.world)
        actions_attempted = self.actions_attempted_count
        avg_health = sum(agent.health for agent in self.agents.values()) / max(1, len(self.agents))
        # Survival = founders still alive / founders. The population count can
        # exceed the founding cohort through births; survival stays in [0, 1].
        initial_ids = getattr(self, "initial_agent_ids", None)
        if initial_ids:
            survival_rate = sum(1 for aid in initial_ids if aid in self.agents) / max(1, len(initial_ids))
        else:
            survival_rate = min(1.0, len(self.agents) / max(1, self.initial_agent_count))
        metrics = {
            "step": step,
            "day": day,
            "simulated_year": day / 365.25,
            "days_per_step": self.time_scale.days_per_step,
            **self.world.metrics(),
            **compute_social_metrics(self.social),
            **compute_colony_dynamics_metrics(self.agents, self.world, self.config, self.social),
            "population": len(self.agents),
            "initial_agent_count": float(self.initial_agent_count),
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
            "llm_governor_skips": float(self.llm_governor_skips),
        }
        metrics.update(composite_agent_score(metrics))
        return metrics

    def save(self, output_dir: Path) -> None:
        save_run_artifacts(
            output_dir,
            run_id=output_dir.name,
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
            current_metrics={} if self.aggregate_mode else self._current_metrics(len(self.global_metrics), self.world.day),
            climate_metadata=self._climate_metadata(),
            stop_reason=self.stop_reason or "completed",
            created_by="cli",
            final=True,
            active_event=self.active_event,
            upcoming_event=self.upcoming_event,
            actions_attempted=self.actions_attempted_count,
            actions_rejected=self.actions_rejected_count,
            action_counts=self.action_counts,
            rejection_counts=self.rejection_counts,
            aggregate_state=self.aggregate_state,
            action_log_truncated=self._action_log_truncated,
            replay_log_truncated=self._replay_log_truncated,
            replay_compactions=self._replay_compactions,
            store_memory_logs=_cfg_bool(_headless_config(self.config).get("store_memory_logs", len(self.agents) <= 10_000)),
        )

    def _climate_metadata(self) -> dict:
        if self.planetary:
            return self.planetary.climate_provider.metadata()
        return {"source": "static_mars_baseline", "environmental_layer_enabled": False}


def _headless_config(config: dict) -> dict:
    value = config.get("headless", {})
    return value if isinstance(value, dict) else {}


def _cfg_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _planetary_grid_refresh_due(config: dict, step: int, total_steps: int) -> bool:
    headless = config.get("headless", {}) if isinstance(config.get("headless", {}), dict) else {}
    raw = headless.get("planetary_grid_refresh_interval_steps", 100)
    try:
        interval = max(1, int(raw or 100))
    except (TypeError, ValueError):
        interval = 100
    force_final = _cfg_bool(headless.get("force_final_planetary_grid_refresh", False))
    return (force_final and step >= total_steps) or interval <= 1 or step % interval == 0


def _snapshot_interval(config: dict) -> int:
    raw = config.get("snapshot_interval", config.get("headless", {}).get("snapshot_interval", 0) if isinstance(config.get("headless", {}), dict) else 0)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _llm_step_cap(config: dict) -> int:
    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    raw = llm_cfg.get("max_calls_per_step", llm_cfg.get("governor_calls_per_step", 5))
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 5


def _initial_aggregate_state(config: dict, population: int, world) -> dict:
    agents_cfg = config.get("agents", {}) if isinstance(config.get("agents", {}), dict) else {}
    inventory = agents_cfg.get("initial_inventory", {}) if isinstance(agents_cfg.get("initial_inventory", {}), dict) else {}
    colony_cfg = config.get("colony", {}) if isinstance(config.get("colony", {}), dict) else {}
    structures = colony_cfg.get("initial_structures", {}) if isinstance(colony_cfg.get("initial_structures", {}), dict) else {}
    habitat_units = float(structures.get("habitat", 0) or 0)
    greenhouse_units = float(structures.get("greenhouse", 0) or 0)
    storage_units = float(structures.get("storage_depot", 0) or 0)
    occupied_m2 = max(25.0 * population, habitat_units * 1_200.0 + greenhouse_units * 900.0 + storage_units * 700.0)
    return {
        "population": float(population),
        "food": float(inventory.get("food", 5.0)) * population,
        "water": float(inventory.get("water", 4.0)) * population,
        "construction_material": float(inventory.get("construction_material", 5.0)) * population,
        "tools": float(inventory.get("tools", 0.0)) * population,
        "occupied_m2": occupied_m2,
        "habitat_units": habitat_units,
        "greenhouse_units": greenhouse_units,
        "storage_units": storage_units,
        "deaths": 0.0,
    }


def _advance_aggregate_state(state: dict, config: dict, dt_days: float) -> None:
    """Estrapolazione aggregata: infrastruttura FISSA, nessuna massa creata.

    **Che cosa e' e che cosa non e' (misurato il 2026-08-30).** Confrontata con
    l'ABM sulla stessa configurazione — duecento coloni, 1500 passi — questa
    modalita' dava popolazione 335 contro 235, **zero morti contro trentuno**,
    cibo 135.533 contro 516 e materiale 7.761 contro zero. Non era una stima
    imprecisa dell'ABM: era un secondo modello, e ne creava la massa dal nulla.

    Due termini sono stati **rimossi** perche' fabbricavano materia:
    i depositi producevano trenta unita' di materiale l'anno ciascuno (un
    deposito immagazzina, non estrae) e il materiale generava attrezzi senza
    consumarsi — l'esatto contrario della ritaratura del 25 agosto, che nell'ABM
    ha reso gli attrezzi un prodotto del materiale che li compone. Le giacenze
    hanno inoltre il **tetto di magazzino** dell'ABM (`SCORTA_PRO_CAPITE`),
    senza il quale il cibo cresceva di centotrenta volte in ventotto anni.

    Resta una estrapolazione **deliberatamente grezza e conservativa**:
    l'infrastruttura non cresce, quindi la stima e' un limite inferiore su ogni
    grandezza che dipenda dal costruire. Non e' un sostituto di una prova ABM e
    non va usata per confrontare politiche.
    """
    years = max(1.0 / 365.25, float(dt_days) / 365.25)
    population = max(0.0, state["population"])
    greenhouse_units = state.get("greenhouse_units", 0.0)
    habitat_units = state.get("habitat_units", 0.0)
    tools_factor = 1.0 + min(0.6, state.get("tools", 0.0) / max(1.0, population) * 0.04)

    state["food"] += greenhouse_units * 140.0 * years * tools_factor
    state["water"] += greenhouse_units * 55.0 * years * tools_factor

    state["food"] -= population * 0.38 * years
    state["water"] -= population * 0.62 * years
    state["construction_material"] -= population * 0.05 * years

    # Stesso tetto della simulazione ad agenti: il surplus oltre la capienza
    # non si conserva. Senza di esso una colonia con mille passi di scorta
    # torna insensibile a qualunque crisi.
    tetto = max(
        population * SCORTA_PRO_CAPITE,
        (habitat_units + greenhouse_units + state.get("storage_units", 0.0))
        * CAPIENZA_PER_STRUTTURA,
    ) + state.get("storage_units", 0.0) * CAPIENZA_PER_DEPOSITO
    for risorsa in RISORSE_A_TETTO:
        if risorsa in state and state[risorsa] > tetto:
            state[risorsa] = tetto
    shortage = max(0.0, -state["food"]) + max(0.0, -state["water"])
    if shortage > 0.0:
        deaths = min(population, max(1.0, shortage * 0.015))
        state["population"] = max(0.0, population - deaths)
        state["deaths"] += deaths
        state["food"] = max(0.0, state["food"])
        state["water"] = max(0.0, state["water"])

    capacity = max(1.0, habitat_units * 80.0 + state["occupied_m2"] / 35.0)
    if state["population"] < capacity and state["food"] > state["population"] * 2.0 and state["water"] > state["population"] * 2.0:
        population_cfg = config.get("population", {}) if isinstance(config.get("population", {}), dict) else {}
        if population_cfg.get("enabled", True):
            growth_rate = float(population_cfg.get("annual_growth_rate", 0.018) or 0.018)
            max_agents = float(population_cfg.get("max_agents", max(capacity, state["population"])) or max(capacity, state["population"]))
            growth = min(max_agents - state["population"], state["population"] * growth_rate * years)
            if growth > 0:
                state["population"] += growth
                state["occupied_m2"] += growth * 35.0


def _aggregate_metrics(step: int, day: float, state: dict, config: dict, initial_population: int) -> dict:
    population = max(0.0, state["population"])
    return {
        "step": step,
        "day": day,
        "simulated_year": day / 365.25,
        "days_per_step": float(config.get("simulation", {}).get("days_per_step", 1) if isinstance(config.get("simulation", {}), dict) else 1),
        "population": population,
        "initial_agent_count": float(initial_population),
        # No per-agent identity in aggregate mode: approximate cohort survival
        # from the cumulative death count, clamped to [0, 1].
        "survival_rate": max(0.0, min(1.0, (float(initial_population) - float(state.get("deaths", 0.0) or 0.0)) / max(1.0, float(initial_population)))),
        "average_agent_health": 1.0 if state.get("deaths", 0.0) <= 0 else max(0.0, population / max(1.0, population + state["deaths"])),
        "food_stock": state["food"],
        "water_stock": state["water"],
        "construction_material_stock": state["construction_material"],
        "tools_stock": state["tools"],
        "colony_area_m2": state["occupied_m2"],
        "colony_built_area_m2": state["occupied_m2"],
        "colony_claimed_area_m2": state["occupied_m2"],
        "actions_attempted": 0.0,
        "actions_accepted": 0.0,
        "action_acceptance_rate": 1.0,
        "action_rejection_rate": 0.0,
        "action_diversity": 0.0,
        "llm_governor_skips": 0.0,
        "aggregate_mode": 1.0,
    }


def map_profile_from_config(config: dict) -> str:
    world_cfg = config.get("world", {}) if isinstance(config.get("world", {}), dict) else {}
    return str(world_cfg.get("map_profile") or config.get("map_profile") or "balanced")


