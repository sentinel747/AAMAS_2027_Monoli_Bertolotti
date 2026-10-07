from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .schema import DynamicModel, Equation, Variable


@dataclass(frozen=True)
class MarsPhysicsConstants:
    solar_constant_w_m2: float = 586.2
    stefan_boltzmann: float = 5.670374419e-8
    mars_year_sols: float = 669.6
    earth_pressure_pa: float = 101325.0
    mars_reference_pressure_pa: float = 700.0
    mars_reference_co2_pa: float = 665.0
    mars_reference_albedo: float = 0.25
    max_accessible_co2_pressure_pa: float = 0.07 * 101325.0
    water_triple_point_pa: float = 611.657
    water_gas_constant: float = 461.5
    default_atmosphere_volume_m3: float = 8.66e17
    mars_surface_area_m2: float = 1.4437e14
    vegetation_reference_units: float = 20000.0


CONSTANTS = MarsPhysicsConstants()


DERIVED_VARIABLES: dict[str, tuple[str, str]] = {
    "SolarFluxMars": ("W/m2", "Solar flux after orbital distance and dust opacity."),
    "DustOpacityIndex": ("0..1", "Seasonal and episodic atmospheric dust forcing."),
    "RadiativeEquilibriumTemp": ("degC", "Blackbody equilibrium temperature from solar flux and albedo."),
    "GreenhouseWarming": ("degC", "Approximate warming from CO2, CH4, water vapor, and engineered gases."),
    "SeasonalCO2PressureSwing": ("Pa", "Reversible seasonal CO2 pressure exchange with polar caps."),
    "LiquidWaterStability": ("0..1", "Pure or briny surface water stability score."),
    "LiquidSurfaceWater": ("kg", "Surface water mass weighted by liquid-water stability."),
    "SurfaceIceWater": ("kg", "Surface water mass behaving as ice or unstable brine."),
    "RadiationShieldingIndex": ("0..1", "Atmospheric shielding proxy against surface radiation."),
    "VegetationSuitability": ("0..1", "Combined pressure, temperature, water, CO2, nutrient, and radiation score."),
    "BreathabilityIndex": ("0..1", "Human-breathable atmosphere proxy based on O2 and total pressure."),
    "HumanSurvivabilityIndex": ("0..1", "Open-surface human survivability proxy."),
    "SoilFormationIndex": ("0..1", "Weathering, moisture, nutrient, and biomass soil-development proxy."),
    "CarbonCycleFlux": ("Pa/step", "Net photosynthesis-respiration pressure exchange between CO2 and O2."),
    "AtmosphericEscapeLoss": ("Pa/step", "Thin-atmosphere photochemical and escape-loss proxy."),
    "EngineeringActivityIndex": ("0..1", "Terraforming infrastructure and industrial activity intensity."),
    "TimeCompressionFactor": ("sol/step", "Simulated sols represented by one model step."),
    "SimulatedYearsPerStep": ("yr/step", "Simulated years represented by one model step."),
    "TerraformingProgressIndex": ("0..1", "Aggregate physics-first terraforming progress score."),
}

SCIENTIFIC_OUTPUTS = [
    "SolarFluxMars",
    "DustOpacityIndex",
    "RadiativeEquilibriumTemp",
    "GreenhouseWarming",
    "LiquidWaterStability",
    "RadiationShieldingIndex",
    "VegetationSuitability",
    "BreathabilityIndex",
    "HumanSurvivabilityIndex",
    "SoilFormationIndex",
    "EngineeringActivityIndex",
    "TimeCompressionFactor",
    "SimulatedYearsPerStep",
    "TerraformingProgressIndex",
]


def enable_scientific_overlay(model: DynamicModel) -> DynamicModel:
    metadata = model.metadata.setdefault("scientific_overlay", {})
    if metadata.get("enabled"):
        return model
    metadata.update(
        {
            "enabled": True,
            "version": "mars_physics_v1",
            "runtime_source": "static_python_only",
            "notes": [
                "STGraph is now frozen source material; runtime corrections are pure Python.",
                "Mars CO2 mobilization is capped near the NASA/MAVEN all-accessible-source estimate.",
                "Water, vegetation, and human survivability use smooth physical suitability scores.",
            ],
        }
    )
    for name, (unit, comment) in DERIVED_VARIABLES.items():
        if name not in model.variables:
            model.variables[name] = Variable(
                original_name=name,
                safe_name=name,
                variable_type="auxiliary",
                unit=unit,
                metadata={"scientific_overlay": True, "comment": comment},
            )
            model.equations[name] = Equation(
                variable=name,
                expression="scientific_overlay",
                kind="scientific_overlay",
                original_expression=comment,
                supported=True,
            )
    for name in SCIENTIFIC_OUTPUTS:
        if name not in model.config.selected_outputs:
            model.config.selected_outputs.append(name)
    return model


def has_scientific_overlay(model: DynamicModel) -> bool:
    return bool(model.metadata.get("scientific_overlay", {}).get("enabled"))


def apply_scientific_corrections(
    model: DynamicModel,
    values: dict[str, Any],
    previous_values: dict[str, Any] | None = None,
    time: float = 0.0,
    timestep: float = 1.0,
) -> list[str]:
    warnings: list[str] = []
    dt = max(0.0, float(timestep or 1.0))
    previous = previous_values or {}

    engineering = _clamp(_scalar(values.get("EngineeringActivityIndex"), 0.03), 0.0, 1.0)
    co2 = _rate_limited_gas("CO2", values, previous, CONSTANTS.mars_reference_co2_pa, 0.0005, 0.020, engineering, dt, CONSTANTS.max_accessible_co2_pressure_pa)
    n2 = _rate_limited_gas("N2", values, previous, 20.0, 0.0002, 0.010, engineering, dt, 78000.0)
    o2 = _rate_limited_gas("O2", values, previous, 1.0, 0.00005, 0.012, engineering, dt, 32000.0)
    ch4 = _rate_limited_gas("CH4", values, previous, 0.0, 0.0001, 0.003, engineering, dt, 3000.0)
    other_gases = _rate_limited_gas("AltriGas", values, previous, 0.0, 0.00005, 0.015, engineering, dt, 50000.0)

    raw_pressure = co2 + n2 + o2 + ch4 + other_gases
    pressure_swing = _seasonal_co2_pressure_swing(co2, raw_pressure, time)
    pressure = max(0.0, raw_pressure + pressure_swing)

    vegetation_total = max(_scalar(values.get("sumVeg"), 0.0), _sum_value(values.get("Veg")))
    vegetation_cover = _clamp(vegetation_total / CONSTANTS.vegetation_reference_units, 0.0, 1.0)
    surface_water = max(0.0, _scalar(values.get("AcquaSuperficiale"), 0.0))
    nubi = max(0.0, _scalar(values.get("Nubi"), 0.0))
    polar_area_fraction = _polar_area_fraction(values)
    dust = _dust_opacity(time)
    cloud_fraction = _clamp(nubi / 1.0e15, 0.0, 1.0)
    water_fraction = _clamp(surface_water / (CONSTANTS.mars_surface_area_m2 * 1000.0), 0.0, 1.0)
    albedo = _clamp(
        CONSTANTS.mars_reference_albedo
        + 0.18 * math.sqrt(polar_area_fraction)
        + 0.08 * cloud_fraction
        + 0.05 * dust
        - 0.06 * vegetation_cover
        - 0.04 * water_fraction,
        0.12,
        0.65,
    )

    solar_flux = CONSTANTS.solar_constant_w_m2 * (1.0 - 0.45 * dust)
    radiative_temp_k = ((solar_flux * (1.0 - albedo)) / (4.0 * CONSTANTS.stefan_boltzmann)) ** 0.25
    radiative_temp_c = radiative_temp_k - 273.15
    greenhouse = _greenhouse_warming(co2, ch4, other_gases, pressure)
    target_temp_c = radiative_temp_c + greenhouse
    prev_temp = _scalar(previous.get("TempMediaMarte"), target_temp_c)
    if previous_values is None:
        temp_c = target_temp_c
    else:
        relaxation = _clamp(dt / 90.0, 0.02, 0.25)
        temp_c = prev_temp + (target_temp_c - prev_temp) * relaxation

    saturation_pa = _saturation_vapor_pressure_pa(temp_c)
    atmosphere_volume = max(1.0, _scalar(values.get("VolumeAtmosfera"), CONSTANTS.default_atmosphere_volume_m3))
    vapor_capacity = saturation_pa / (CONSTANTS.water_gas_constant * max(140.0, temp_c + 273.15)) * atmosphere_volume
    vapor = max(0.0, _scalar(values.get("VaporeAcqueo"), 0.0))
    relative_humidity = _clamp(vapor / vapor_capacity if vapor_capacity > 0 else 0.0, 0.0, 1.5)
    condensation = max(0.0, vapor - vapor_capacity)

    liquid_water_stability = _liquid_water_stability(temp_c, pressure)
    liquid_surface_water = surface_water * liquid_water_stability
    surface_ice_water = surface_water - liquid_surface_water

    soil_capacity = max(1.0, _scalar(values.get("CapacitaMaxSuolo"), 1.0))
    soil_moisture = max(0.0, _scalar(values.get("UmiditaSuolo"), 0.0))
    soil_moisture_fraction = _clamp(soil_moisture / soil_capacity, 0.0, 1.5)
    water_score = max(liquid_water_stability, _smoothstep(0.03, 0.45, soil_moisture_fraction), _smoothstep(0.45, 0.95, relative_humidity) * 0.4)
    radiation_shielding = _clamp(1.0 - math.exp(-pressure / 30000.0), 0.0, 1.0)

    nutrient_total = _sum_value(values.get("NutrientiDelSuolo"))
    nutrient_score = _clamp(0.25 + 0.35 * vegetation_cover + 0.40 * _smoothstep(1.0, 100.0, nutrient_total), 0.0, 1.0)
    temp_veg_score = _smoothstep(-8.0, 18.0, temp_c) * (1.0 - _smoothstep(42.0, 65.0, temp_c))
    pressure_veg_score = _smoothstep(8000.0, 45000.0, pressure)
    co2_score = _smoothstep(40.0, 300.0, co2)
    vegetation_suitability = _clamp(temp_veg_score * pressure_veg_score * water_score * co2_score * radiation_shielding * nutrient_score, 0.0, 1.0)

    breathability = _breathability_index(o2, pressure)
    human_temp_score = _smoothstep(-10.0, 15.0, temp_c) * (1.0 - _smoothstep(35.0, 55.0, temp_c))
    human_survivability = _clamp(breathability * human_temp_score * radiation_shielding * max(0.25, water_score), 0.0, 1.0)
    soil_formation = _clamp(
        0.20 * _smoothstep(-20.0, 20.0, temp_c) * _smoothstep(1000.0, 20000.0, pressure)
        + 0.25 * water_score
        + 0.30 * vegetation_cover
        + 0.25 * nutrient_score,
        0.0,
        1.0,
    )

    carbon_flux = _carbon_cycle_flux(co2, o2, vegetation_cover, vegetation_suitability, dt)
    co2 = max(0.0, co2 - carbon_flux)
    o2 = max(0.0, o2 + carbon_flux)

    escape_loss = _atmospheric_escape_loss(pressure, ch4, vapor, dt)
    ch4 = max(0.0, ch4 - escape_loss)
    vapor_escape = min(vapor, max(0.0, vapor * (1.0 - radiation_shielding) * 1.0e-6 * dt))
    vapor = max(0.0, vapor - vapor_escape - condensation)
    nubi = max(0.0, nubi + condensation)

    pressure = max(0.0, co2 + n2 + o2 + ch4 + other_gases + pressure_swing)
    terraforming_progress = _clamp(
        0.18 * _smoothstep(8000.0, 70000.0, pressure)
        + 0.16 * _smoothstep(-20.0, 15.0, temp_c)
        + 0.14 * liquid_water_stability
        + 0.14 * radiation_shielding
        + 0.14 * vegetation_suitability
        + 0.10 * soil_formation
        + 0.14 * breathability,
        0.0,
        1.0,
    )

    _write(model, values, "CO2", co2)
    _write(model, values, "N2", n2)
    _write(model, values, "O2", o2)
    _write(model, values, "CH4", ch4)
    _write(model, values, "AltriGas", other_gases)
    _write(model, values, "PressioneAtmosferica", pressure)
    _write(model, values, "Pressione_calcolata", pressure)
    _write(model, values, "AlbedoSuperficie", albedo)
    _write(model, values, "TempMediaMarte", temp_c)
    _write(model, values, "GasSerra", co2 + ch4 + other_gases)
    _write(model, values, "EffettoSerra", 1.0 if greenhouse > 3.0 and pressure > 1000.0 else 0.0)
    _write(model, values, "VaporeAcqueo", vapor)
    _write(model, values, "Nubi", nubi)
    _write(model, values, "RimozioneFotodissociativa", vapor_escape)
    _write(model, values, "PressSaturazioneVapore", saturation_pa)
    _write(model, values, "CapacitaMaxVapore", vapor_capacity)
    _write(model, values, "UmiditaRelativa", relative_humidity)
    _write(model, values, "UmiditaPercentuale", soil_moisture_fraction)
    _write(model, values, "Condizione_Avviamento", 1.0 if liquid_water_stability > 0.25 and pressure >= 10000.0 else 0.0)
    _write(model, values, "Condizione_Avviamento_Veg", 1.0 if vegetation_suitability >= 0.35 else 0.0)
    _write(model, values, "Condizione_Avviamento_Pop", 1.0 if human_survivability >= 0.55 and vegetation_total >= 20000.0 else 0.0)

    _write(model, values, "SolarFluxMars", solar_flux)
    _write(model, values, "DustOpacityIndex", dust)
    _write(model, values, "RadiativeEquilibriumTemp", radiative_temp_c)
    _write(model, values, "GreenhouseWarming", greenhouse)
    _write(model, values, "SeasonalCO2PressureSwing", pressure_swing)
    _write(model, values, "LiquidWaterStability", liquid_water_stability)
    _write(model, values, "LiquidSurfaceWater", liquid_surface_water)
    _write(model, values, "SurfaceIceWater", surface_ice_water)
    _write(model, values, "RadiationShieldingIndex", radiation_shielding)
    _write(model, values, "VegetationSuitability", vegetation_suitability)
    _write(model, values, "BreathabilityIndex", breathability)
    _write(model, values, "HumanSurvivabilityIndex", human_survivability)
    _write(model, values, "SoilFormationIndex", soil_formation)
    _write(model, values, "CarbonCycleFlux", carbon_flux)
    _write(model, values, "AtmosphericEscapeLoss", escape_loss)
    _write(model, values, "EngineeringActivityIndex", engineering)
    _write(model, values, "TimeCompressionFactor", dt)
    _write(model, values, "SimulatedYearsPerStep", dt / 365.25)
    _write(model, values, "TerraformingProgressIndex", terraforming_progress)
    return warnings


def _write(model: DynamicModel, values: dict[str, Any], name: str, value: float) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        value = 0.0
    values[name] = value
    variable = model.variables.get(name)
    if variable:
        values[variable.safe_name] = value


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return float(np.nanmean(value.astype(float)))
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return float(np.nanmean(np.array(value, dtype=float)))
        return float(value)
    except Exception:
        return default


def _sum_value(value: Any) -> float:
    try:
        if isinstance(value, np.ndarray):
            return float(np.nansum(value.astype(float)))
        if isinstance(value, (list, tuple)):
            return float(np.nansum(np.array(value, dtype=float)))
        return max(0.0, float(value))
    except Exception:
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rate_limited_gas(
    name: str,
    values: dict[str, Any],
    previous: dict[str, Any],
    default: float,
    natural_rate_pa_per_sol: float,
    engineered_rate_pa_per_sol: float,
    engineering: float,
    timestep: float,
    cap: float,
) -> float:
    target = _clamp(_scalar(values.get(name), default), 0.0, cap)
    if not previous:
        return target
    previous_value = _clamp(_scalar(previous.get(name), default), 0.0, cap)
    max_increase = (natural_rate_pa_per_sol + engineered_rate_pa_per_sol * engineering) * max(1.0, timestep)
    if target > previous_value + max_increase:
        return previous_value + max_increase
    max_decrease = natural_rate_pa_per_sol * 0.5 * max(1.0, timestep)
    if target < previous_value - max_decrease:
        return previous_value - max_decrease
    return target


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _seasonal_co2_pressure_swing(co2: float, pressure: float, time: float) -> float:
    phase = 2.0 * math.pi * (time % CONSTANTS.mars_year_sols) / CONSTANTS.mars_year_sols
    amplitude = 0.15 * math.exp(-pressure / 6000.0)
    return co2 * amplitude * math.sin(phase)


def _dust_opacity(time: float) -> float:
    seasonal = (0.5 + 0.5 * math.sin(2.0 * math.pi * (time % CONSTANTS.mars_year_sols) / CONSTANTS.mars_year_sols)) ** 4
    storm = max(0.0, math.sin(2.0 * math.pi * ((time - 180.0) % 1100.0) / 1100.0)) ** 14
    return _clamp(0.06 + 0.18 * seasonal + 0.55 * storm, 0.0, 0.85)


def _polar_area_fraction(values: dict[str, Any]) -> float:
    surface = max(1.0, _scalar(values.get("SuperficieMarte"), CONSTANTS.mars_surface_area_m2))
    polar_area = max(0.0, _scalar(values.get("CalottaPolareNord"), 0.0)) + max(0.0, _scalar(values.get("CalottaPolareSud"), 0.0))
    return _clamp(polar_area / surface, 0.0, 1.0)


def _greenhouse_warming(co2: float, ch4: float, other_gases: float, pressure: float) -> float:
    co2_extra = max(0.0, co2 - CONSTANTS.mars_reference_co2_pa)
    co2_term = 10.0 * math.log1p(co2_extra / 1000.0)
    pressure_term = 4.0 * math.log1p(max(0.0, pressure - CONSTANTS.mars_reference_pressure_pa) / 5000.0)
    methane_term = min(12.0, 3.0 * math.log1p(ch4))
    engineered_term = min(55.0, 12.0 * math.log1p(other_gases / 1000.0))
    return 3.0 + co2_term + pressure_term + methane_term + engineered_term


def _saturation_vapor_pressure_pa(temp_c: float) -> float:
    if temp_c >= 0.0:
        return 610.94 * math.exp((17.625 * temp_c) / (temp_c + 243.04))
    return 610.94 * math.exp((22.587 * temp_c) / (temp_c + 273.86))


def _liquid_water_stability(temp_c: float, pressure: float) -> float:
    pressure_score = _smoothstep(CONSTANTS.water_triple_point_pa, 10000.0, pressure)
    brine_score = _smoothstep(-23.0, 4.0, temp_c)
    pure_water_score = _smoothstep(0.0, 12.0, temp_c)
    return _clamp(pressure_score * max(0.35 * brine_score, pure_water_score), 0.0, 1.0)


def _breathability_index(o2: float, pressure: float) -> float:
    o2_low = _smoothstep(12000.0, 19000.0, o2)
    o2_high_penalty = 1.0 - _smoothstep(30000.0, 55000.0, o2)
    pressure_score = _smoothstep(45000.0, 85000.0, pressure) * (1.0 - _smoothstep(140000.0, 200000.0, pressure))
    return _clamp(o2_low * o2_high_penalty * pressure_score, 0.0, 1.0)


def _carbon_cycle_flux(co2: float, o2: float, vegetation_cover: float, suitability: float, timestep: float) -> float:
    photosynthesis = min(co2, 0.04 * vegetation_cover * suitability * max(1.0, timestep))
    # Respiration scales with the *living* biosphere (same cover x suitability
    # driver as photosynthesis) and can only draw down a fraction of the O2
    # pool per step: a barren planet must not silently convert engineered
    # oxygen back into CO2 (the old (1-suitability) driver consumed all
    # industrial O2 every step while suitability was 0).
    respiration = min(o2 * 0.25, 0.015 * vegetation_cover * suitability * max(1.0, timestep))
    return photosynthesis - respiration


def _atmospheric_escape_loss(pressure: float, ch4: float, vapor: float, timestep: float) -> float:
    shielding = _clamp(1.0 - math.exp(-pressure / 30000.0), 0.0, 1.0)
    methane_loss = ch4 * (1.0 - shielding) * 2.5e-4 * max(1.0, timestep)
    water_equivalent_loss = min(0.05, vapor * 1.0e-18 * (1.0 - shielding) * max(1.0, timestep))
    return methane_loss + water_equivalent_loss
