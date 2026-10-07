from __future__ import annotations

import math
from dataclasses import dataclass, field

from .resources import ResourceBundle
from .structures import Structure
from .terrain import TERRAIN_HABITABILITY, TERRAIN_TRAVERSAL_RISK, TerrainType, terrain_hazard_tier, traversal_risk_for_cell
from .colony_site import colony_site_score
from .occupancy import occupancy_capacity_from_housing
from .structures import StructureType

# Ecological bounds shared by biology and habitability scoring.
VEGETATION_CARRYING_CAPACITY = 10.0
POLLUTION_DEGRADATION_CEILING = 0.35

# Runtime toggle (config `world.cell_degradation`, default ON): with it OFF the
# cells do not degrade under occupancy — pollution_risk stays 0, the density
# floor is skipped and the SYSTEM ALERTs derived from these conditions are
# never raised in agent memory. Set once per run by both engines.
_CELL_DEGRADATION_ENABLED = True


def set_cell_degradation(enabled: bool) -> None:
    global _CELL_DEGRADATION_ENABLED
    _CELL_DEGRADATION_ENABLED = bool(enabled)


def cell_degradation_enabled() -> bool:
    return _CELL_DEGRADATION_ENABLED


@dataclass
class Cell:
    x: int
    y: int
    terrain: TerrainType
    elevation: float = 0.0
    local_temperature_modifier: float = 0.0
    radiation_level: float = 1.0
    dust_level: float = 0.0
    baseline_radiation_level: float | None = None
    baseline_dust_level: float | None = None
    baseline_temperature_modifier: float | None = None
    polar_severity: float = 0.0
    water_ice: float = 0.0
    liquid_water: float = 0.0
    resources: ResourceBundle = field(default_factory=ResourceBundle)
    habitability_score: float = 0.0
    vegetation_biomass: float = 0.0
    is_spawn: bool = False
    
    # New biological/pedological fields
    proto_soil_development: float = 0.0  # 0.0 (regolith) to 1.0 (mature soil)
    nutrients: dict[str, float] = field(default_factory=lambda: {"N": 0.0, "P": 0.0, "C": 0.0})
    organic_matter: float = 0.0
    
    structures: list[Structure] = field(default_factory=list)
    construction_sites: dict[str, float] = field(default_factory=dict) # type: progress (0-100)
    agents_present: list[str] = field(default_factory=list)
    agent_positions_m: dict[str, dict[str, float]] = field(default_factory=dict)
    geometry: dict[str, float] = field(default_factory=dict)
    explored: bool = False
    faction_influence: dict[str, float] = field(default_factory=dict)
    pollution_risk: float = 0.0

    def occupancy_capacity(self) -> float:
        counts = {
            structure_type: sum(
                1 for structure in self.structures if structure.type == structure_type
            )
            for structure_type in (
                StructureType.SHELTER,
                StructureType.HABITAT,
                StructureType.INFIRMARY,
            )
        }
        return occupancy_capacity_from_housing(
            counts[StructureType.SHELTER],
            counts[StructureType.HABITAT],
            counts[StructureType.INFIRMARY],
        )

    def overcrowding_excess(self) -> float:
        return max(0.0, len(self.agents_present) - self.occupancy_capacity())

    def structure_heat(self) -> float:
        """Local warming (degrees C) contributed by structures such as heaters."""
        return sum(float(s.local_effect.get("temperature", 0.0)) for s in self.structures)

    def structure_food_effect(self) -> float:
        """Effetto `food` complessivo delle strutture, gia' pesato per efficienza.

        Gemello di `structure_heat`, e serve alla stessa ragione: la facciata
        vettoriale lo legge da `struct_fx` senza materializzare l'elenco delle
        strutture, che in una metrica per passo costerebbe un oggetto per
        struttura per cella.
        """
        return sum(float(s.local_effect.get("food", 0.0)) for s in self.structures)

    def structure_water_effect(self) -> float:
        """Effetto `water` complessivo delle strutture. Vedi `structure_food_effect`."""
        return sum(float(s.local_effect.get("water", 0.0)) for s in self.structures)

    def update_biology(self, planetary_state: dict, dt: float):
        """Update biological and soil processes based on scienzedellaterra.md principles.

        dt is expressed in days. The vegetation balance dv/dt = g - k*v (Liebig-limited
        source, linear mineralization sink) is integrated with the exact exponential
        solution so deep-time steps (e.g. 3650 days/step) remain numerically stable.
        """
        temp_c = float(planetary_state.get("mean_temperature_c", -63.0))
        pressure = float(planetary_state.get("pressure_pa", 600.0))
        liquid_stability = float(planetary_state.get("liquid_water_stability", 0.0))
        radiation = self.radiation_level * (1.0 - float(planetary_state.get("radiation_shielding_index", 0.0)))

        # 0. Ice <-> liquid water exchange: once the planetary layer makes liquid
        # water stable and temperatures rise above freezing, part of the local ice
        # melts; it refreezes when conditions revert. Dormant on baseline Mars.
        exchange_fraction = min(1.0, dt / 365.25)
        effective_temp_c = temp_c + self.local_temperature_modifier + self.structure_heat()
        if liquid_stability > 0.0 and effective_temp_c > 0.0 and self.water_ice > 0.0:
            melted = self.water_ice * min(0.5, liquid_stability * 0.05 * exchange_fraction * max(0.0, effective_temp_c) / 10.0)
            self.water_ice -= melted
            self.liquid_water += melted
        elif self.liquid_water > 0.0 and (effective_temp_c < -5.0 or liquid_stability <= 0.0):
            refrozen = self.liquid_water * min(1.0, 0.5 * exchange_fraction)
            self.liquid_water -= refrozen
            self.water_ice += refrozen

        # 1. Pedogenesis: Rock -> Regolith -> Proto-soil
        # Accelerated by temperature (Arrhenius-like), water, and existing biomass (lichens)
        weathering_rate = 0.0001 * dt * max(0, (temp_c + 20) / 40) * (liquid_stability + 0.1)
        if self.vegetation_biomass > 0.1:
            weathering_rate *= (1.0 + 0.5 * self.vegetation_biomass)

        self.proto_soil_development = min(1.0, self.proto_soil_development + weathering_rate)

        # 2. Vegetation Growth (Limited by Liebig's Law of the Minimum)
        # Factors: Temp, Pressure, Water, Soil, Nutrients (N, P, C)
        f_temp = max(0, 1.0 - abs(temp_c - 15) / 35) if temp_c > -10 else 0
        f_press = min(1.0, pressure / 40000.0) if pressure > 6000 else 0
        f_water = min(1.0, (self.liquid_water + self.water_ice * 0.2) / 5.0)
        f_soil = 0.1 + 0.9 * self.proto_soil_development

        # Nutrient limiting factor (Liebig)
        n_ratio = self.nutrients["N"] / 10.0
        p_ratio = self.nutrients["P"] / 2.0
        c_ratio = self.nutrients["C"] / 5.0
        f_nutrients = min(n_ratio, p_ratio, c_ratio, 1.0)

        growth_potential = f_temp * f_press * f_water * f_soil * (1.0 - radiation * 0.5)
        growth_per_day = 0.1 * max(0.0, growth_potential) * max(0.0, f_nutrients)
        decay_per_day = 0.02 * max(0, (temp_c + 10) / 30)

        before = self.vegetation_biomass
        if decay_per_day > 0.0:
            decay_factor = math.exp(-decay_per_day * dt)
            equilibrium = growth_per_day / decay_per_day
            new_veg = equilibrium + (before - equilibrium) * decay_factor
            decayed = before * (1.0 - decay_factor)
        else:
            new_veg = before + growth_per_day * dt
            decayed = 0.0
        new_veg = max(0.0, min(VEGETATION_CARRYING_CAPACITY, new_veg))
        grown = max(0.0, new_veg - before + decayed)
        self.vegetation_biomass = new_veg

        # Nutrient cycle: growth takes N/P/C up from the soil, mineralization returns it.
        self.organic_matter += decayed * 0.5
        self.nutrients["N"] = max(0.0, self.nutrients["N"] + decayed * 0.1 - grown * 0.1)
        self.nutrients["P"] = max(0.0, self.nutrients["P"] + decayed * 0.02 - grown * 0.02)
        self.nutrients["C"] = max(0.0, self.nutrients["C"] + decayed * 0.4 - grown * 0.4)

        # 3. Degradation from prolonged agent presence (over-exploitation):
        # sustained occupancy pushes the cell toward a bounded degradation ceiling,
        # abandoned cells slowly recover.
        years = dt / 365.25
        unsupported_occupants = self.overcrowding_excess()
        if not _CELL_DEGRADATION_ENABLED:
            self.pollution_risk = 0.0
        elif unsupported_occupants > 0:
            degradation_ceiling = min(
                POLLUTION_DEGRADATION_CEILING, 0.05 * unsupported_occupants
            )
            if self.pollution_risk < degradation_ceiling:
                self.pollution_risk = min(
                    degradation_ceiling,
                    self.pollution_risk + unsupported_occupants * 0.05 * years,
                )
        else:
            self.pollution_risk = max(0.0, self.pollution_risk - 0.02 * years)

        self.recompute_habitability(planetary_state)
    def recompute_habitability(self, planetary_state: dict | None = None) -> float:
        score = TERRAIN_HABITABILITY[self.terrain]
        score += min(self.water_ice, 10.0) * 0.005
        score += min(self.liquid_water, 5.0) * 0.02
        score += min(self.vegetation_biomass, VEGETATION_CARRYING_CAPACITY) * 0.01
        score -= self.radiation_level * 0.03
        score -= self.dust_level * 0.02
        score -= self.polar_severity * 0.11
        effective_temperature_modifier = self.local_temperature_modifier + self.structure_heat()
        score -= max(0.0, -effective_temperature_modifier - 10.0) * 0.003

        # Instantaneous congestion load (area-scaled) acts as a floor under the
        # persistent over-exploitation degradation accumulated in update_biology.
        if _CELL_DEGRADATION_ENABLED:
            area_m2 = max(1.0, float(self.geometry.get("area_m2", 1.0)))
            density_load = len(self.agents_present) * 6.0 + len(self.structures) * 20.0
            density_pollution = min(0.08, density_load / area_m2 * 25_000.0)
            self.pollution_risk = max(self.pollution_risk, density_pollution)
        score -= self.pollution_risk * 0.45  # Severe penalty for ecological over-exploitation

        if planetary_state:
            shielding = float(planetary_state.get("radiation_shielding_index", 0.0))
            liquid_stability = float(planetary_state.get("liquid_water_stability", 0.0))
            vegetation_suitability = float(planetary_state.get("vegetation_suitability", 0.0))
            temp_c = float(planetary_state.get("mean_temperature_c", -63.0))
            temp_score = max(0.0, min(1.0, (temp_c + 60.0) / 80.0))
            score += shielding * 0.06
            score += liquid_stability * min(self.water_ice + self.resources.ice, 10.0) * 0.01
            score += vegetation_suitability * 0.08
            score += temp_score * 0.03
            score -= (1.0 - shielding) * 0.04
        for structure in self.structures:
            score += structure.local_effect.get("habitability", 0.0)
        self.habitability_score = max(0.0, min(1.0, score))
        return self.habitability_score

    def to_public_dict(self) -> dict:
        tr = traversal_risk_for_cell(self)
        return {
            "x": self.x,
            "y": self.y,
            "terrain": self.terrain.value,
            "terrain_base_risk": float(TERRAIN_TRAVERSAL_RISK.get(self.terrain, 0.2)),
            "traversal_risk": tr,
            "hazard_tier": terrain_hazard_tier(tr),
            "elevation": self.elevation,
            "temperature_modifier": self.local_temperature_modifier,
            "polar_severity": self.polar_severity,
            "radiation": self.radiation_level,
            "dust": self.dust_level,
            "water_ice": self.water_ice,
            "liquid_water": self.liquid_water,
            "resources": self.resources.to_dict(),
            "habitability": self.habitability_score,
            "vegetation_biomass": self.vegetation_biomass,
            "proto_soil": self.proto_soil_development,
            "nutrients": dict(self.nutrients),
            "organic_matter": self.organic_matter,
            "structures": [s.to_dict() for s in self.structures],
            "construction_sites": dict(self.construction_sites),
            "agents_present": list(self.agents_present),
            "agent_positions_m": {agent_id: dict(position) for agent_id, position in self.agent_positions_m.items()},
            "geometry": dict(self.geometry),
            "explored": self.explored,
            "faction_influence": dict(self.faction_influence),
            "pollution_risk": self.pollution_risk,
            "occupancy_capacity": self.occupancy_capacity(),
            "overcrowding_excess": self.overcrowding_excess(),
            "colony_site_score": colony_site_score(self),
        }
