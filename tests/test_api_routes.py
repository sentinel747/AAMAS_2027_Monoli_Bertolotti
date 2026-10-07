from fastapi.testclient import TestClient
from pathlib import Path
import shutil

from src.api.main import app


def test_api_routes_smoke():
    client = TestClient(app)
    assert client.get("/api/state/current").status_code == 200
    assert "gpt" in client.get("/api/llm/providers").json()
    assert client.get("/api/world/chunk").json()["width"] == 360
    assert len(client.get("/api/agents").json()) >= 5
    assert client.post("/api/simulation/step").status_code == 200
    start_state = client.post("/api/simulation/start").json()
    assert "max_days" in start_state
    assert client.post("/api/simulation/pause").status_code == 200
    stop_state = client.post("/api/simulation/stop").json()
    assert stop_state["stop_reason"] == "ready_for_new_run"
    configured = client.post(
        "/api/scenarios/load",
        json={
            "seed": 0,
            "world": {"width": 10, "height": 10},
            "agents": {"count": 4, "llm_count": 2},
            "simulation": {"days_per_step": 3650, "max_days": 36500},
            "llm": {"provider": "fallback"},
            "population": {"enabled": False},
        },
    ).json()
    assert len(configured["agents"]) == 4
    assert sum(1 for agent in configured["agents"] if agent["mode"] == "llm") == 2


def test_world_chunk_supports_colony_focus_bounds():
    client = TestClient(app)
    client.post(
        "/api/scenarios/load",
        json={
            "seed": 2,
            "world": {"width": 12, "height": 8},
            "agents": {"count": 1, "llm_count": 0},
            "simulation": {"days_per_step": 1, "max_days": 10},
            "population": {"enabled": False},
        },
    )

    chunk = client.get("/api/world/chunk?x=-20&y=-10&size=5").json()

    assert chunk["x"] == 0
    assert chunk["y"] == 0
    assert chunk["width"] == 5
    assert chunk["height"] == 5
    assert chunk["world_width"] == 12
    assert chunk["world_height"] == 8


def test_run_analysis_reads_persisted_run_files():
    run_id = "test_analysis_unit"
    run_dir = Path("outputs/runs") / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_metrics.json").write_text('{"average_habitability":0.42}', encoding="utf-8")
    (run_dir / "api_usage.json").write_text('{"calls":2,"successful_calls":1,"failed_calls":1,"tokens_in":10,"tokens_out":5,"tokens_total":15}', encoding="utf-8")
    (run_dir / "token_usage.json").write_text('{"calls":2,"tokens_in":10,"tokens_out":5,"tokens_total":15}', encoding="utf-8")
    (run_dir / "llm_usage.jsonl").write_text('{"provider":"gpt","model":"gpt-4o-mini","tokens_in":10,"tokens_out":5}\n', encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        '{"type":"llm_call_completed","day":1,"data":{"provider":"gpt","error":"boom"}}\n',
        encoding="utf-8",
    )
    (run_dir / "agent_decisions.jsonl").write_text('{"agent_id":"a","llm_provider":"gpt:gpt-4o-mini"}\n', encoding="utf-8")
    (run_dir / "dead_agents.jsonl").write_text('{"agent_id":"a","death_step":1}\n', encoding="utf-8")
    (run_dir / "replay_events.jsonl").write_text('{"step":1,"agent_id":"a","action":"move","x":2,"y":3}\n', encoding="utf-8")
    (run_dir / "world_static_base.json").write_text('{"width":4,"height":3,"cells":[]}', encoding="utf-8")
    (run_dir / "state_timeseries.csv").write_text("step,day,population\n1,3650,5\n", encoding="utf-8")

    client = TestClient(app)
    data = client.get(f"/api/runs/{run_id}/analysis").json()

    assert data["run_id"] == run_id
    assert data["overview"]["events"] == 1
    assert data["overview"]["dead_agents"] == 1
    assert data["overview"]["tokens_total"] == 15
    assert data["api_usage"]["calls"] == 2
    assert data["token_usage"]["tokens_in"] == 10
    assert data["insights"]["event_providers"]["gpt"] == 1
    assert data["streams"]["llm_usage"][0]["model"] == "gpt-4o-mini"
    assert data["streams"]["dead_agents"][0]["death_step"] == 1
    assert data["streams"]["replay_events"][0]["action"] == "move"
    assert data["replay_static_base"]["width"] == 4
    assert data["streams"]["state_timeseries"][0]["population"] == "5"
    shutil.rmtree(run_dir, ignore_errors=True)


def test_gui_days_limit_is_step_limit_and_terminal_run_starts_fresh():
    client = TestClient(app)
    configured = client.post(
        "/api/scenarios/load",
        json={
            "name": "step_limit_unit",
            "seed": 7,
            "days": 2,
            "world": {"width": 8, "height": 8, "map_profile": "ice_rich"},
            "agents": {"count": 1, "llm_count": 0},
            "simulation": {"days_per_step": 1, "max_days": 100},
            "llm": {"provider": "fallback"},
            "population": {"enabled": False},
        },
    ).json()
    assert configured["step"] == 0
    assert configured["map_profile"] == "ice_rich"
    assert configured["seed"] == 7

    client.post("/api/simulation/step")
    finished = client.post("/api/simulation/step").json()

    assert finished["step"] == 2
    assert finished["day"] == 2
    assert finished["stop_reason"] == "max_steps_reached:2"
    saved_run_id = finished["run_id"]
    saved_output_dir = Path(finished["output_dir"])
    assert saved_output_dir.exists()

    metadata_before_stop = (saved_output_dir / "run_metadata.json").read_text(encoding="utf-8")
    stopped = client.post("/api/simulation/stop").json()
    metadata_after_stop = (saved_output_dir / "run_metadata.json").read_text(encoding="utf-8")

    assert '"stop_reason": "max_steps_reached:2"' in metadata_after_stop
    assert '"map_profile": "ice_rich"' in metadata_after_stop
    assert '"seed": 7' in metadata_after_stop
    assert metadata_after_stop == metadata_before_stop
    assert stopped["stop_reason"] == "ready_for_new_run"
    assert stopped["last_saved_run_id"] == saved_run_id


def test_loaded_run_name_is_used_for_next_output_folder(tmp_path, monkeypatch):
    # Hermetic cwd: run folders are created relative to it, and a leftover
    # folder with the same name would trigger the no-overwrite suffix.
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    first = client.post(
        "/api/scenarios/load",
        json={
            "name": "Prima Run Test",
            "seed": 4,
            "days": 1,
            "world": {"width": 8, "height": 8},
            "agents": {"count": 1, "llm_count": 0},
            "simulation": {"days_per_step": 1, "max_days": 10},
            "population": {"enabled": False},
            "climate": {"enabled": False},
            "environmental_layer": {"enabled": False},
        },
    ).json()
    assert first["run_id"] == "prima_run_test"

    second = client.post(
        "/api/scenarios/load",
        json={
            "name": "Seconda Run Finale",
            "seed": 4,
            "days": 1,
            "world": {"width": 8, "height": 8},
            "agents": {"count": 1, "llm_count": 0},
            "simulation": {"days_per_step": 1, "max_days": 10},
            "population": {"enabled": False},
            "climate": {"enabled": False},
            "environmental_layer": {"enabled": False},
        },
    ).json()
    assert second["run_id"] == "seconda_run_finale"

    finished = client.post("/api/simulation/step").json()
    output_dir = Path(finished["output_dir"])
    assert output_dir.name == "seconda_run_finale"
    assert (output_dir / "config.yaml").read_text(encoding="utf-8").find('"name": "Seconda Run Finale"') >= 0


def test_missing_days_and_max_days_has_no_implicit_step_or_day_limit():
    client = TestClient(app)
    configured = client.post(
        "/api/scenarios/load",
        json={
            "name": "no_limits_unit",
            "seed": 9,
            "world": {"width": 8, "height": 8},
            "agents": {"count": 1, "llm_count": 0},
            "simulation": {"days_per_step": 10},
            "population": {"enabled": False},
        },
    ).json()
    assert configured["max_days"] == 0

    for _ in range(4):
        state = client.post("/api/simulation/step").json()

    assert state["step"] == 4
    assert state["day"] == 40
    assert not str(state["stop_reason"]).startswith("max_steps_reached")
    assert not str(state["stop_reason"]).startswith("max_days_reached")
