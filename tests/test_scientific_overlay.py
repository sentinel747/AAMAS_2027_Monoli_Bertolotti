import pytest

from src.model.dynamic_model import load_static_terraformazione_model
from src.model.scientific_overlay import apply_scientific_corrections
from src.simulation.constraints import scalarize
from src.simulation.engine import SimulationEngine


def test_static_model_uses_mars_scientific_overlay():
    model = load_static_terraformazione_model()
    state = SimulationEngine(model).initialize()
    values = state.values

    pressure_sum = sum(scalarize(values[name]) for name in ["CO2", "N2", "O2", "CH4", "AltriGas"])
    pressure_sum += scalarize(values["SeasonalCO2PressureSwing"])

    assert model.metadata["scientific_overlay"]["enabled"]
    assert values["TempMediaMarte"] == pytest.approx(-63.0, abs=12.0)
    assert values["PressioneAtmosferica"] == pytest.approx(pressure_sum, rel=1e-9)
    assert values["LiquidWaterStability"] == 0
    assert values["RadiationShieldingIndex"] < 0.1
    assert values["VegetationSuitability"] == 0


def test_overlay_allows_plausible_terraformed_state_only_after_thresholds():
    model = load_static_terraformazione_model()
    values = SimulationEngine(model).initialize().values
    values.update(
        {
            "CO2": 7000.0,
            "N2": 70000.0,
            "O2": 21000.0,
            "CH4": 5.0,
            "AltriGas": 30000.0,
            "AcquaSuperficiale": 1.0e16,
            "UmiditaSuolo": 100.0,
            "CapacitaMaxSuolo": 200.0,
            "NutrientiDelSuolo": [50.0, 50.0, 10.0],
            "sumVeg": 25000.0,
            "Veg": 25000.0,
        }
    )

    apply_scientific_corrections(model, values, time=100.0, timestep=1.0)

    assert values["LiquidWaterStability"] > 0.1
    assert values["RadiationShieldingIndex"] > 0.9
    assert values["VegetationSuitability"] > 0.25
    assert values["Condizione_Avviamento_Veg"] == 1.0
    assert values["BreathabilityIndex"] > 0.4
