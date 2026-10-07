"""Igiene di log e metriche azioni (audit run 120726_testlogic5_1000steps).

1. events.jsonl aveva step=None su OGNI evento: log_event ora timbra lo step
   corrente del mondo quando il chiamante non lo passa.
2. Il motore GUI calcolava actions_attempted/accepted/rejected/diversity dalle
   liste di log, che si congelano al cap max_action_log_rows (150k raggiunto a
   step 958): ora usa contatori veri come il motore CLI, e il cap emette un
   evento esplicito invece di troncare in silenzio.
"""

from collections import Counter
from types import SimpleNamespace

from src.agents.action_space import ActionRequest, ActionType
from src.api.state_store import SimulationController
from src.core import shell_common
from src.simulation.run_artifacts import build_action_summary
from src.social_network.network import SocialNetwork
from src.world.world_generator import WorldGenerator


BASE_CONFIG = {
    "name": "hygiene_unit",
    "seed": 3,
    "days": 6,
    "world": {"width": 8, "height": 8},
    "agents": {"count": 2, "llm_count": 0},
    "simulation": {"days_per_step": 1, "max_days": 50},
    "population": {"enabled": False},
    "climate": {"enabled": False},
    "environmental_layer": {"enabled": False},
}


def test_log_event_stamps_current_world_step():
    world = WorldGenerator(seed=2).generate(6, 6)
    world.step = 41
    world.log_event("population_grew", "test settler", agent_id="agent_099")
    assert world.events[-1]["step"] == 41
    # Il chiamante puo' comunque passare uno step esplicito.
    world.log_event("custom", "explicit step", step=7)
    assert world.events[-1]["step"] == 7


def test_gui_engine_action_metrics_survive_log_cap():
    config = {**BASE_CONFIG, "headless": {"max_action_log_rows": 3}}
    sim = SimulationController(config)
    sim.step(4)  # 2 agenti x 4 step = 8 azioni tentate, log cappato a 3

    metrics = sim._current_metrics(sim.step_index, sim.world.day)
    assert len(sim.validated_actions) + len(sim.rejected_actions) <= 3
    assert metrics["actions_attempted"] == 8.0
    assert metrics["actions_accepted"] + int(metrics["actions_attempted"] * metrics["action_rejection_rate"] + 0.5) == 8
    assert metrics["action_diversity"] >= 1.0
    assert 0.0 <= metrics["productive_action_rate"] <= 1.0
    assert 0.0 <= metrics["passive_action_rate"] <= 1.0

    truncation_events = [e for e in sim.world.events if e["type"] == "action_log_truncated"]
    assert len(truncation_events) == 1  # una sola volta, non a ogni riga


def test_gui_engine_counters_reset_with_scenario():
    config = {**BASE_CONFIG, "headless": {"max_action_log_rows": 3}}
    sim = SimulationController(config)
    sim.step(2)
    assert sim.actions_attempted_count == 4
    sim.reset({**config, "name": "hygiene_unit_reset"})
    assert sim.actions_attempted_count == 0
    assert sim.actions_accepted_count == 0
    assert sim.actions_rejected_count == 0
    assert len(sim.action_counts) == 0
    assert len(sim.rejection_counts) == 0


def test_controller_respects_disabled_memory_logs():
    config = {**BASE_CONFIG, "headless": {"store_memory_logs": False}}
    sim = SimulationController(config)
    sim.step(2)

    assert sim.actions_attempted_count == 4
    assert sim.validated_actions == []
    assert sim.rejected_actions == []
    assert sim.thoughts == []
    # Replay is intentionally independent from verbose in-memory logs and
    # remains available for visual analysis.
    assert sim.replay_events


def test_replay_compaction_preserves_early_and_latest_steps():
    world = WorldGenerator(seed=5).generate(4, 4)
    replay_events: list[dict] = []
    replay_log_truncated = False
    replay_compactions = 0

    for step in range(1, 9):
        request = ActionRequest("a0", ActionType.REST)
        record = SimpleNamespace(
            agent_id="a0",
            request=request,
            accepted=True,
            nearby_agents=[],
            action_log_row={"step": step, "nearby_agents_count": 0, "nearby_agents": ["a1", "a2"]},
            replay_event={"step": step, "day": float(step), "agent_id": "a0"},
            result_data={},
            thought=None,
            x=1,
            y=2,
        )
        drained = shell_common.drain_agent_step_records(
            SimpleNamespace(agent_step_records=[record]),
            step,
            float(step),
            world=world,
            social=SocialNetwork({"a0"}),
            validated_actions=[],
            rejected_actions=[],
            replay_events=replay_events,
            conversations=[],
            thoughts=[],
            action_counts=Counter(),
            rejection_counts=Counter(),
            max_action_log_rows=10,
            max_replay_rows=4,
            action_log_truncated=False,
            replay_log_truncated=replay_log_truncated,
            replay_compactions=replay_compactions,
        )
        replay_log_truncated = drained["replay_log_truncated"]
        replay_compactions = drained["replay_compactions"]

    retained_steps = [row["step"] for row in replay_events]
    assert len(replay_events) <= 4
    assert retained_steps[0] == 1
    assert retained_steps[-1] == 8
    assert replay_log_truncated is True
    assert replay_compactions >= 1
    assert len([event for event in world.events if event["type"] == "replay_log_compacted"]) == 1


def test_replay_aggregates_dense_observations_but_keeps_other_actions():
    records = []
    for index in range(25):
        records.append(
            SimpleNamespace(
                agent_id=f"observer_{index:02d}",
                request=ActionRequest(f"observer_{index:02d}", ActionType.OBSERVE),
                accepted=True,
                nearby_agents=[],
                action_log_row={"message": "observed"},
                replay_event={
                    "step": 1,
                    "day": 0.0,
                    "agent_id": f"observer_{index:02d}",
                    "action": "observe",
                },
                result_data={},
                thought=None,
                x=1,
                y=1,
            )
        )
    records.append(
        SimpleNamespace(
            agent_id="builder",
            request=ActionRequest("builder", ActionType.BUILD_GREENHOUSE),
            accepted=True,
            nearby_agents=[],
            action_log_row={"message": "built"},
            replay_event={
                "step": 1,
                "day": 0.0,
                "agent_id": "builder",
                "action": "build_greenhouse",
            },
            result_data={},
            thought=None,
            x=1,
            y=1,
        )
    )
    replay_events: list[dict] = []
    action_counts = Counter()

    drained = shell_common.drain_agent_step_records(
        SimpleNamespace(agent_step_records=records),
        1,
        0.0,
        world=WorldGenerator(seed=5).generate(4, 4),
        social=SocialNetwork({record.agent_id for record in records}),
        validated_actions=[],
        rejected_actions=[],
        replay_events=replay_events,
        conversations=[],
        thoughts=[],
        action_counts=action_counts,
        rejection_counts=Counter(),
        max_action_log_rows=100,
        max_replay_rows=100,
        max_observe_replay_rows_per_step=3,
        action_log_truncated=False,
    )

    assert len([row for row in replay_events if row["action"] == "observe"]) == 3
    assert len([row for row in replay_events if row["action"] == "build_greenhouse"]) == 1
    summary = next(row for row in replay_events if row["action"] == "observe_summary")
    assert summary["result_data"] == {
        "accepted_observations": 25,
        "sampled_observations": 3,
        "omitted_observations": 22,
    }
    assert drained["replay_observations_omitted"] == 22
    assert action_counts == Counter({"observe": 25, "build_greenhouse": 1})


def test_replay_aggregates_idle_fallbacks_without_losing_counts():
    records = [
        SimpleNamespace(
            agent_id=f"idle_{index:02d}",
            request=ActionRequest(f"idle_{index:02d}", ActionType.DO_NOTHING),
            accepted=True,
            nearby_agents=[],
            action_log_row={"message": "idle"},
            replay_event={
                "step": 1,
                "day": 0.0,
                "agent_id": f"idle_{index:02d}",
                "action": "do_nothing",
            },
            result_data={},
            thought=None,
            x=1,
            y=1,
        )
        for index in range(25)
    ]
    replay_events: list[dict] = []
    action_counts = Counter()

    drained = shell_common.drain_agent_step_records(
        SimpleNamespace(agent_step_records=records),
        1,
        0.0,
        world=WorldGenerator(seed=6).generate(4, 4),
        social=SocialNetwork({record.agent_id for record in records}),
        validated_actions=[],
        rejected_actions=[],
        replay_events=replay_events,
        conversations=[],
        thoughts=[],
        action_counts=action_counts,
        rejection_counts=Counter(),
        max_action_log_rows=100,
        max_replay_rows=100,
        max_observe_replay_rows_per_step=3,
        action_log_truncated=False,
    )

    assert len([row for row in replay_events if row["action"] == "do_nothing"]) == 3
    summary = next(row for row in replay_events if row["action"] == "do_nothing_summary")
    assert summary["result_data"] == {
        "accepted_idle_actions": 25,
        "sampled_idle_actions": 3,
        "omitted_idle_actions": 22,
    }
    assert drained["replay_passive_rows_omitted"] == 22
    assert action_counts == Counter({"do_nothing": 25})


def test_action_summary_reports_productive_share_from_complete_counters():
    summary = build_action_summary(
        Counter({"observe": 90, "build_greenhouse": 10, "rest": 5}),
        Counter({("build_greenhouse", "duplicate"): 2}),
        attempted=105,
        rejected=2,
    )

    assert summary["accepted"] == 103
    assert summary["productive_actions_accepted"] == 13.0
    assert summary["productive_action_rate"] == 13 / 103
    assert summary["passive_action_rate"] == 90 / 103


def test_action_log_can_omit_nearby_agent_ids_but_keep_count():
    request = ActionRequest("a0", ActionType.REST)
    record = SimpleNamespace(
        agent_id="a0",
        request=request,
        accepted=True,
        nearby_agents=["a1", "a2"],
        action_log_row={"nearby_agents_count": 2, "nearby_agents": ["a1", "a2"]},
        replay_event={"step": 1, "agent_id": "a0"},
        result_data={},
        thought=None,
        x=1,
        y=2,
    )
    validated_actions: list[dict] = []

    shell_common.drain_agent_step_records(
        SimpleNamespace(agent_step_records=[record]),
        1,
        0.0,
        world=WorldGenerator(seed=5).generate(4, 4),
        social=SocialNetwork({"a0", "a1", "a2"}),
        validated_actions=validated_actions,
        rejected_actions=[],
        replay_events=[],
        conversations=[],
        thoughts=[],
        action_counts=Counter(),
        rejection_counts=Counter(),
        max_action_log_rows=10,
        max_replay_rows=10,
        action_log_truncated=False,
        include_nearby_agent_ids=False,
    )

    assert validated_actions == [{"nearby_agents_count": 2}]
    assert record.action_log_row["nearby_agents"] == ["a1", "a2"]


def test_rejection_summary_survives_disabled_memory_logs():
    request = ActionRequest("a0", ActionType.COLLECT_MATERIALS)
    record = SimpleNamespace(
        agent_id="a0",
        request=request,
        accepted=False,
        nearby_agents=[],
        action_log_row={"message": "no materials found in this cell"},
        replay_event={},
        result_data={},
        thought=None,
        x=1,
        y=2,
    )
    validated_actions: list[dict] = []
    rejected_actions: list[dict] = []
    action_counts = Counter()
    rejection_counts = Counter()
    world = WorldGenerator(seed=5).generate(4, 4)

    drained = shell_common.drain_agent_step_records(
        SimpleNamespace(agent_step_records=[record]),
        1,
        0.0,
        world=world,
        social=SocialNetwork({"a0"}),
        validated_actions=validated_actions,
        rejected_actions=rejected_actions,
        replay_events=[],
        conversations=[],
        thoughts=[],
        action_counts=action_counts,
        rejection_counts=rejection_counts,
        max_action_log_rows=10,
        max_replay_rows=10,
        action_log_truncated=False,
        store_memory_logs=False,
    )
    summary = build_action_summary(
        action_counts,
        rejection_counts,
        attempted=drained["attempted"],
        rejected=drained["rejected"],
    )

    assert rejected_actions == []
    assert summary["rejected"] == 1
    assert summary["by_action"]["collect_materials"]["rejection_reasons"] == {
        "no materials found in this cell": 1
    }
