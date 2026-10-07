from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.mars_climate import MarsClimateProvider
from src.model.dynamic_model import load_static_terraformazione_model
from src.simulation.constraints import scalarize
from src.simulation.engine import SimulationEngine
from src.world.grid import GridWorld
from src.world.mars_geometry import (
    MARS_MEAN_SURFACE_PRESSURE_PA,
    MARS_MEAN_TEMPERATURE_C,
    MARS_SOL_HOURS,
    MARS_SURFACE_GRAVITY_M_S2,
    MARS_YEAR_EARTH_DAYS,
)
from src.world.terrain import TerrainType


DEFAULT_DAYS_PER_STEP = 365
DEFAULT_MAX_SIMULATED_DAYS = 365250


@dataclass
class TimeScale:
    days_per_step: int = DEFAULT_DAYS_PER_STEP
    max_days: int = DEFAULT_MAX_SIMULATED_DAYS

    @property
    def years_per_step(self) -> float:
        return self.days_per_step / 365.25


def time_scale_from_config(config: dict[str, Any] | None) -> TimeScale:
    config = config or {}
    sim_cfg = config.get("simulation", {}) or {}
    days_per_step = int(sim_cfg.get("days_per_step", config.get("days_per_step", DEFAULT_DAYS_PER_STEP)))
    max_days = int(sim_cfg.get("max_days", config.get("max_days", DEFAULT_MAX_SIMULATED_DAYS)))
    return TimeScale(days_per_step=max(1, days_per_step), max_days=max(1, max_days))


def environmental_layer_enabled(config: dict[str, Any] | None) -> bool:
    config = config or {}
    layer_cfg = config.get("environmental_layer", {})
    if isinstance(layer_cfg, dict) and "enabled" in layer_cfg:
        return _as_bool(layer_cfg.get("enabled"))
    planetary_cfg = config.get("planetary", {})
    if isinstance(planetary_cfg, dict) and "enabled" in planetary_cfg:
        return _as_bool(planetary_cfg.get("enabled"))
    return True


def sync_static_mars_environment(world: GridWorld, time_scale: TimeScale, recompute_habitability: bool = True) -> dict[str, float]:
    metrics = {
        "simulation_step": int(world.step),
        "simulated_day": float(world.day),
        "simulated_year": float(world.day) / 365.25,
        "days_per_step": float(time_scale.days_per_step),
        "years_per_step": time_scale.years_per_step,
        "environmental_layer_enabled": 0.0,
        "environmental_layer_mode": 0.0,
        "mars_sol_hours": MARS_SOL_HOURS,
        "mars_year_earth_days": MARS_YEAR_EARTH_DAYS,
        "mars_surface_gravity_m_s2": MARS_SURFACE_GRAVITY_M_S2,
        "surface_pressure_pa": MARS_MEAN_SURFACE_PRESSURE_PA,
        "pressure_pa": MARS_MEAN_SURFACE_PRESSURE_PA,
        "mean_temperature_c": MARS_MEAN_TEMPERATURE_C,
        "dust_opacity_index": 0.18,
        "solar_flux_w_m2": 586.2,
        "relative_humidity_index": 0.0,
        "wind_speed_m_s": 5.0,
        "liquid_water_stability": 0.0,
        "radiation_shielding_index": 0.0,
        "vegetation_suitability": 0.0,
        "breathability_index": 0.0,
        "human_survivability_index": 0.0,
        "soil_formation_index": 0.0,
        "engineering_activity_index": 0.0,
        "colony_environment_progress_index": 0.0,
        "planetary_terraforming_progress_index": 0.0,
        "scientific_knowledge": 0.0,
        "tech_tier": 0.0,
        "cost_reduction": 0.0,
        "wear_reduction": 0.0,
        "yield_bonus": 0.0,
    }
    world.planetary_state = metrics
    world.metadata["environmental_layer_enabled"] = False
    world.metadata["environmental_layer_mode"] = "static_mars_baseline"
    if recompute_habitability:
        world.recompute_habitability()
    return metrics


class PlanetaryCoupler:
    """Couples local agent-built infrastructure to the static planetary model."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.time_scale = time_scale_from_config(config)
        self.climate_provider = MarsClimateProvider(self.config)
        self.model = load_static_terraformazione_model()
        self.model.config.timestep = float(self.time_scale.days_per_step)
        self.model.config.end_time = float(self.time_scale.max_days)
        self.engine = SimulationEngine(self.model)
        self.state = self.engine.initialize(self.model.config)
        self._seed_baseline_atmosphere()

    def _seed_baseline_atmosphere(self) -> None:
        """The generated ODE model starts with an empty atmosphere (all gas
        stocks at 0 Pa); seed it with the real Mars baseline so blended
        pressure/gas metrics start from ~610 Pa instead of vacuum."""
        values = self.state.values
        if scalarize(values.get("PressioneAtmosferica", 0.0)) >= 1.0:
            return
        composition = {
            "CO2": 0.951,
            "N2": 0.028,
            "AltriGas": 0.0197,
            "O2": 0.0013,
        }
        for name, fraction in composition.items():
            if scalarize(values.get(name, 0.0)) <= 0.0:
                values[name] = MARS_MEAN_SURFACE_PRESSURE_PA * fraction
        values["PressioneAtmosferica"] = sum(
            scalarize(values.get(name, 0.0)) for name in ("CH4", "CO2", "N2", "O2", "AltriGas")
        )

    def reset(self, config: dict[str, Any] | None = None) -> None:
        self.__init__(config)

    def sync_world(self, world: GridWorld, apply_climate_grid: bool = True, recompute_habitability: bool = True) -> None:
        world.planetary_state = self.public_metrics(world)
        world.metadata["environmental_layer_enabled"] = True
        world.metadata["environmental_layer_mode"] = "planetary_background"
        if apply_climate_grid:
            self._apply_climate_to_grid(world)
            if recompute_habitability:
                world.recompute_habitability()

    def advance(self, world: GridWorld, days_per_step: int | None = None, refresh_grid: bool = True) -> dict[str, float]:
        if days_per_step is not None:
            self.time_scale.days_per_step = max(1, int(days_per_step))
        self.model.config.timestep = float(self.time_scale.days_per_step)
        self.model.config.end_time = max(float(self.time_scale.max_days), self.state.time + self.model.config.timestep)
        self._apply_infrastructure_forcing(world)
        self.state = self.engine.step(self.state, self.model.config)
        self.sync_world(world, apply_climate_grid=refresh_grid, recompute_habitability=refresh_grid)
        return world.planetary_state

    def public_metrics(self, world: GridWorld | None = None) -> dict[str, float]:
        values = self.state.values
        day = float(world.day if world is not None else self.state.time)
        climate = self.climate_provider.sample(
            day,
            latitude_deg=_world_latitude(world),
            elevation_m=_world_elevation(world),
            longitude_deg=_world_longitude(world),
        )
        _ensure_mcd_sample_ready(climate, self.climate_provider)
        model_pressure = scalarize(values.get("PressioneAtmosferica", 0.0))
        model_temp = scalarize(values.get("TempMediaMarte", 0.0))
        model_dust = scalarize(values.get("DustOpacityIndex", 0.0))
        engineering = scalarize(values.get("EngineeringActivityIndex", 0.0))
        pressure = _blend_climate_baseline(model_pressure, climate.pressure_pa, engineering, weight=0.72)
        temp = _blend_climate_baseline(model_temp, climate.mean_temperature_c, engineering, weight=0.58)
        dust = _blend_climate_baseline(model_dust, climate.dust_opacity, engineering, weight=0.68)
        metrics = {
            "simulation_step": int(getattr(self.state, 'step', int(values.get('step', 0)))),
            "simulated_day": day,
            "simulated_year": day / 365.25,
            "days_per_step": float(self.time_scale.days_per_step),
            "years_per_step": self.time_scale.years_per_step,
            "surface_pressure_pa": pressure,
            "pressure_pa": pressure,
            "mean_temperature_c": temp,
            "co2_pa": scalarize(values.get("CO2", 0.0)),
            "o2_pa": scalarize(values.get("O2", 0.0)),
            "n2_pa": scalarize(values.get("N2", 0.0)),
            "dust_opacity_index": dust,
            "solar_flux_w_m2": climate.solar_flux_w_m2,
            "solar_flux_toa_w_m2": climate.solar_flux_toa_w_m2,
            "relative_humidity_index": climate.relative_humidity,
            "wind_speed_m_s": climate.wind_speed_m_s,
            "mcd_pressure_pa_raw": climate.pressure_pa,
            "mcd_mean_temperature_c_raw": climate.mean_temperature_c,
            "mcd_dust_opacity_raw": climate.dust_opacity,
            "mcd_solar_flux_w_m2_raw": climate.solar_flux_w_m2,
            "mcd_relative_humidity_raw": climate.relative_humidity,
            "mcd_wind_speed_m_s_raw": climate.wind_speed_m_s,
            "liquid_water_stability": scalarize(values.get("LiquidWaterStability", 0.0)),
            "radiation_shielding_index": scalarize(values.get("RadiationShieldingIndex", 0.0)),
            "vegetation_suitability": scalarize(values.get("VegetationSuitability", 0.0)),
            "breathability_index": scalarize(values.get("BreathabilityIndex", 0.0)),
            "human_survivability_index": scalarize(values.get("HumanSurvivabilityIndex", 0.0)),
            "soil_formation_index": scalarize(values.get("SoilFormationIndex", 0.0)),
            "engineering_activity_index": scalarize(values.get("EngineeringActivityIndex", 0.0)),
            "colony_environment_progress_index": scalarize(values.get("TerraformingProgressIndex", 0.0)),
            "planetary_terraforming_progress_index": scalarize(values.get("TerraformingProgressIndex", 0.0)),
            "scientific_knowledge": scalarize(values.get("ConoscenzaScientifica", 0.0)),
            "environmental_layer_enabled": 1.0,
        }
        metrics.update(self.get_active_bonuses(metrics["scientific_knowledge"]))
        metrics.update(climate.to_metrics())
        return metrics

    def get_active_bonuses(self, knowledge: float) -> dict[str, Any]:
        """Determine active colony-wide bonuses based on scientific knowledge."""
        return active_bonuses(knowledge)

    def _apply_infrastructure_forcing(self, world: GridWorld) -> None:
        counts = world.structure_type_counts()
        metrics = world.metrics(include_planetary=False)
        years = self.time_scale.years_per_step
        oxygen_plants = counts.get("oxygen_plant", 0)
        greenhouses = counts.get("greenhouse", 0)
        heaters = counts.get("heater", 0)
        solar_arrays = counts.get("solar_array", 0)
        habitats = counts.get("habitat", 0) + counts.get("shelter", 0)
        labs = counts.get("research_lab", 0)
        engineering = min(
            1.0,
            oxygen_plants * 0.040
            + greenhouses * 0.030
            + heaters * 0.030
            + solar_arrays * 0.015
            + habitats * 0.010
            + labs * 0.050,
        )
        values = self.state.values
        values["EngineeringActivityIndex"] = engineering
        values["O2"] = scalarize(values.get("O2", 0.0)) + oxygen_plants * 0.08 * years
        values["AltriGas"] = scalarize(values.get("AltriGas", 0.0)) + (heaters * 0.04 + solar_arrays * 0.006 + labs * 0.02) * years
        values["AcquaSuperficiale"] = max(0.0, scalarize(values.get("AcquaSuperficiale", 0.0)) + greenhouses * 2.5e8 * years)
        values["UmiditaSuolo"] = max(0.0, scalarize(values.get("UmiditaSuolo", 0.0)) + greenhouses * 0.05 * years)
        vegetation_units = metrics.get("vegetation", 0.0) * 500.0 + greenhouses * 200.0
        values["sumVeg"] = max(scalarize(values.get("sumVeg", 0.0)), vegetation_units)
        values["Veg"] = max(scalarize(values.get("Veg", 0.0)), vegetation_units)

    def _apply_climate_to_grid(self, world: GridWorld) -> None:
        metrics = world.planetary_state
        dust = float(metrics.get("dust_opacity_index", metrics.get("climate_dust_opacity", 0.18)) or 0.18)
        pressure = float(metrics.get("surface_pressure_pa", 700.0) or 700.0)
        mean_temp = float(metrics.get("mean_temperature_c", -63.0) or -63.0)
        # **Il canale solare usa il flusso al TOP DELL'ATMOSFERA (2026-08-30).**
        # `solar_flux_w_m2` e' gia' attenuato dalla polvere, e usarlo qui
        # accanto a `dust_factor` faceva entrare la polvere DUE volte nelle
        # stesse due grandezze di cella: dimostrato per sostituzione, perche'
        # quel flusso valeva `586,2 x (1 - 0,45 tau)` e quindi
        # `solar_factor = 1 - 0,45 tau`. Il 41,8% dell'effetto termico della
        # polvere e il 22,3% di quello radiativo arrivavano da una variabile
        # che si chiama «solare» e di solare non aveva nulla.
        solar_flux = float(
            metrics.get("solar_flux_toa_w_m2")
            or metrics.get("solar_flux_w_m2", 0.0)
            or 0.0
        )
        relative_humidity = float(metrics.get("relative_humidity_index", 0.0) or 0.0)
        wind_speed = float(metrics.get("wind_speed_m_s", 0.0) or 0.0)
        dust_factor = max(0.0, min(1.0, dust / 0.85))
        pressure_shielding = max(0.0, min(0.45, pressure / 30000.0))
        solar_factor = max(0.0, min(1.5, solar_flux / 586.2))
        humidity_factor = max(0.0, min(1.0, relative_humidity))
        wind_factor = max(0.0, min(2.0, wind_speed / 20.0))
        for row in world.cells:
            for cell in row:
                base_dust = cell.baseline_dust_level if cell.baseline_dust_level is not None else cell.dust_level
                base_rad = cell.baseline_radiation_level if cell.baseline_radiation_level is not None else cell.radiation_level
                base_temp = cell.baseline_temperature_modifier if cell.baseline_temperature_modifier is not None else cell.local_temperature_modifier
                terrain_dust = 0.14 if cell.terrain == TerrainType.DUST_FIELD else 0.0
                cell.dust_level = max(0.0, min(1.0, base_dust * (0.55 + 0.75 * dust_factor + 0.2 * wind_factor) + terrain_dust * dust_factor))
                cell.radiation_level = max(0.25, min(1.8, base_rad * (1.0 - pressure_shielding) + dust_factor * 0.04 + (1.0 - solar_factor) * 0.03))
                cell.local_temperature_modifier = base_temp + (mean_temp + 63.0) * 0.035 - dust_factor * 0.8 + (solar_factor - 1.0) * 1.5 + humidity_factor * 0.4


def _blend_climate_baseline(model_value: float, climate_value: float, engineering: float, weight: float) -> float:
    # Engineering shifts trust from observed climatology toward the dynamic
    # model, which carries the terraforming effects MCD cannot see. This is
    # sound only because the model is seeded at the real Mars baseline
    # (_seed_baseline_atmosphere); from vacuum it would drag pressure to ~0.
    if climate_value <= 0 and model_value > 0:
        return model_value
    anchoring = max(0.15, min(0.95, weight * (1.0 - min(0.85, engineering * 0.65))))
    return climate_value * anchoring + model_value * (1.0 - anchoring)


def _world_latitude(world: GridWorld | None) -> float:
    if world is None:
        return -4.6
    return float(world.metadata.get("center_latitude_deg", -4.6))


def _world_longitude(world: GridWorld | None) -> float:
    if world is None:
        return 137.4
    return float(world.metadata.get("center_longitude_deg", 137.4))


def _world_elevation(world: GridWorld | None) -> float:
    if world is None:
        return -2500.0
    return float(world.metadata.get("mean_elevation_m", -2500.0))


def _ensure_mcd_sample_ready(sample, provider: MarsClimateProvider) -> None:
    if sample.source not in ("mcd_call_mcd", "mcd_netcdf"):
        raise RuntimeError(
            "MCD call_mcd runtime is required but the Fortran/Python interface did not return data. "
            f"Detected source={sample.source}, path={sample.data_path or '<empty>'}, scenario={provider.scenario}."
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def active_bonuses(knowledge: float) -> dict[str, Any]:
    """I bonus di colonia attivi a una data conoscenza scientifica.

    Estratta dal metodo perche' ha due chiamanti: il layer planetario, che
    ricava la conoscenza dal proprio modello, e `GridWorld.metrics`, che la
    accumula dai laboratori di ricerca quando il layer e' spento. Duplicare le
    soglie avrebbe significato che la stessa colonia sbloccava tecnologie
    diverse a seconda di quale sottosistema le calcolava.
    """
    bonuses: dict[str, Any] = {
        "tech_tier": 0,
        "cost_reduction": 0.0,
        "wear_reduction": 0.0,
        "yield_bonus": 0.0,
        "unlocked_techs": [],
    }
    if knowledge >= 50:
        bonuses["tech_tier"] = 1
        bonuses["cost_reduction"] = 0.15  # costruzioni piu' economiche del 15%
        bonuses["unlocked_techs"].append("Advanced Materials")
    if knowledge >= 150:
        bonuses["tech_tier"] = 2
        bonuses["wear_reduction"] = 0.25  # usura delle strutture piu' lenta
        bonuses["unlocked_techs"].append("Automated Maintenance")
    if knowledge >= 400:
        bonuses["tech_tier"] = 3
        bonuses["yield_bonus"] = 0.30  # resa di O2 e cibo maggiore del 30%
        bonuses["unlocked_techs"].append("Bio-Engineered Ecosystems")
    return bonuses
