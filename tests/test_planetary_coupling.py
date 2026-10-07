from src.simulation.agent_coupled_runner import AgentCoupledRunner
from src.simulation.planetary_coupling import PlanetaryCoupler
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def test_planetary_coupler_exposes_scaled_time_and_metrics():
    world = WorldGenerator(seed=11).generate(10, 10)
    coupler = PlanetaryCoupler({"simulation": {"days_per_step": 365, "max_days": 36500}})
    coupler.sync_world(world)
    world.day += 365
    metrics = coupler.advance(world)

    assert metrics["days_per_step"] == 365
    assert metrics["simulated_year"] > 0.9
    assert metrics["surface_pressure_pa"] > 0
    assert "planetary_terraforming_progress_index" in metrics


def test_infrastructure_increases_engineering_activity():
    world = WorldGenerator(seed=12).generate(10, 10)
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, 0, 0))
    world.add_structure(Structure(StructureType.GREENHOUSE, 1, 0))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, 2, 0))
    coupler = PlanetaryCoupler({"simulation": {"days_per_step": 365, "max_days": 36500}})
    coupler.sync_world(world)
    world.day += 365
    metrics = coupler.advance(world)

    assert metrics["engineering_activity_index"] > 0.05
    assert metrics["o2_pa"] > 0


def test_agent_runner_uses_days_per_step_as_decision_scale():
    runner = AgentCoupledRunner(
        {
            "seed": 1,
            "days": 2,
            "simulation": {"days_per_step": 180, "max_days": 3600},
            "world": {"width": 8, "height": 8},
            "agents": {"count": 0, "llm_count": 0},
        }
    )
    runner.run(days=2)

    assert runner.global_metrics[-1]["day"] == 360
    assert runner.global_metrics[-1]["days_per_step"] == 180


def test_planetary_grid_refresh_runs_on_soa_facade():
    """Regressione: il refresh periodico della griglia planetaria legge
    cell.baseline_dust_level / baseline_radiation_level /
    baseline_temperature_modifier dalla facciata SoA (CellView). Questi baseline
    sono catturati a worldgen e prima non erano esposti dalla facciata, quindi
    alla PRIMA occorrenza del refresh (di default ogni 100 step, latente nelle
    run brevi) `planetary_coupling._apply_climate_to_grid` sollevava
    AttributeError. Qui l'intervallo e' forzato a 2 così il refresh scatta
    subito e attraversa il percorso incriminato."""
    runner = AgentCoupledRunner(
        {
            "seed": 3,
            "days": 2,
            "simulation": {"days_per_step": 30, "max_days": 3600},
            "environmental_layer": {"enabled": True},
            "world": {"width": 8, "height": 8},
            "agents": {"count": 5, "llm_count": 0},
            "headless": {"planetary_grid_refresh_interval_steps": 2},
        }
    )
    runner.run(days=2)  # step 2 -> 2 % 2 == 0 -> grid refresh -> legge i baseline

    assert runner.planetary is not None
    cell = runner.world.get_cell(4, 4)
    # esposti e popolati (worldgen li imposta per ogni cella, mai None qui).
    assert cell.baseline_dust_level is not None
    assert cell.baseline_radiation_level is not None
    assert cell.baseline_temperature_modifier is not None


def test_agent_runner_can_use_static_mars_environment_without_planetary_background():
    runner = AgentCoupledRunner(
        {
            "seed": 2,
            "days": 1,
            "simulation": {"days_per_step": 90, "max_days": 3600},
            "environmental_layer": {"enabled": False},
            "world": {"width": 8, "height": 8},
            "agents": {"count": 0, "llm_count": 0},
        }
    )
    runner.run(days=1)

    metrics = runner.global_metrics[-1]
    assert runner.planetary is None
    assert runner.world.metadata["environmental_layer_enabled"] is False
    assert metrics["environmental_layer_enabled"] == 0.0
    assert metrics["surface_pressure_pa"] > 0
    assert metrics["colony_environment_progress_index"] == 0.0
