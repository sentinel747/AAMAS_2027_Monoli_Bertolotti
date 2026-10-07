"""Regression tests: pressing stop must never overwrite a saved run, and the
state payload must always be strict-JSON parseable (no NaN/Infinity)."""

import json
import math

import pytest

from src.api.state_store import SimulationController, _json_finite


@pytest.fixture()
def controller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "name": "stop_safety_run",
        "seed": 0,
        "days": 3,
        "simulation": {"days_per_step": 7},
        "world": {"width": 20, "height": 20, "map_profile": "balanced"},
        "agents": {"count": 3, "llm_count": 0},
        "colony": {"start_x": 5, "start_y": 5, "initial_structures": {"habitat": 1, "greenhouse": 1}},
        "climate": {"enabled": False},
        "environmental_layer": {"enabled": False},
        "extreme_events": {"enabled": False, "chance_per_step": 0},
        "population": {"enabled": False},
        "llm": {"max_calls_per_step": 0},
        "headless": {"log_interval_steps": 1},
    }
    return SimulationController(config)


def test_stop_on_idle_simulation_is_a_noop(controller, tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / controller.run_id
    controller.stop()
    assert not run_dir.exists()
    assert controller.stop_reason == "ready_for_new_run"


def test_double_stop_does_not_wipe_saved_run(controller, tmp_path):
    controller.step(2)
    controller.stop()  # saves the 2-step run and resets to day 0
    saved_dir = tmp_path / "outputs" / "runs" / controller.last_saved_run_id
    assert saved_dir.exists()
    metrics_before = json.loads((saved_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert metrics_before.get("simulation_step") == 2

    controller.stop()  # second press on the freshly reset sim: must be a no-op
    metrics_after = json.loads((saved_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert metrics_after.get("simulation_step") == 2
    assert (saved_dir / "state_timeseries.csv").stat().st_size > 0


def test_new_run_with_same_name_does_not_overwrite_previous(controller, tmp_path):
    controller.step(2)
    controller.stop()
    first_dir = tmp_path / "outputs" / "runs" / controller.last_saved_run_id
    first_metrics = (first_dir / "final_metrics.json").read_text(encoding="utf-8")

    controller.step(1)  # new run with the unchanged config name
    controller.stop()
    assert controller.last_saved_run_id != first_dir.name
    second_dir = tmp_path / "outputs" / "runs" / controller.last_saved_run_id
    assert second_dir.exists() and second_dir != first_dir
    assert (first_dir / "final_metrics.json").read_text(encoding="utf-8") == first_metrics


def test_current_state_is_strict_json_parseable(controller):
    controller.step(1)
    # Poison a metric the way numpy edge cases do, then verify the payload.
    controller.global_metrics[-1]["poisoned_metric"] = float("nan")
    controller.global_metrics[-1]["poisoned_inf"] = float("inf")
    payload = controller.current_state()
    text = json.dumps(payload, allow_nan=False)  # raises on NaN/Infinity
    assert json.loads(text)["metrics"]["poisoned_metric"] == 0.0


def test_json_finite_sanitizes_nested_payloads():
    payload = {"a": float("nan"), "b": [1.0, float("-inf"), {"c": float("inf")}], "d": "x", "e": True, "f": None}
    clean = _json_finite(payload)
    assert clean == {"a": 0.0, "b": [1.0, 0.0, {"c": 0.0}], "d": "x", "e": True, "f": None}
    assert math.isfinite(clean["a"])
