from pathlib import Path

from src.model.dynamic_model import load_static_terraformazione_model
from src.simulation.engine import SimulationEngine


def test_simulation_one_step_and_short_run():
    model = load_static_terraformazione_model()
    model.config.end_time = 3
    engine = SimulationEngine(model, base_dir="Terraformazione")
    state = engine.initialize()
    next_state = engine.step(state)
    assert next_state.time == 1.0
    result = engine.run(max_steps=3)
    assert len(result.states) == 4


def test_run_until_respects_end_time():
    model = load_static_terraformazione_model()
    model.config.end_time = 2
    engine = SimulationEngine(model, base_dir="Terraformazione")
    result = engine.run_until()
    assert result.states[-1].time == 2


def test_baseline_outputs_exist():
    assert Path("outputs/baseline_results/state_timeseries.csv").exists()
    assert Path("outputs/baseline_results/final_state.json").exists()
