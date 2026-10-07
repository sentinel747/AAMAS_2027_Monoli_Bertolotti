from __future__ import annotations

import math
import random

from src.core import reachability_batch
from src.world.perception import LocalObservation
from src.world.navigation import (
    wrapped_chebyshev_distance,
    wrapped_manhattan_distance,
)
from src.world.structures import STRUTTURE_DI_AVVIAMENTO, BUILD_COSTS, StructureType
from src.world.terrain import edificabile, traversal_risk_for_cell

from .action_space import ActionRequest, ActionType, _combined_can_afford
from .base_agent import BaseAgent
from .build_policy import (
    COLONISTS_PER_STRUCTURE,
    cell_accepts_arrival,
    local_life_support_capacity,
    permessi_di_viaggio,
    posti_di_cantiere,
    copertura_completa,
    structure_saturated,
)


ROLE_PRESETS = {
    # coordinate_every 3 made ~1 action in 4 a status chat once survival needs
    # were met (24% of the 1000-step audit run): 6 keeps the social network
    # alive at half the action cost.
    "colonist": {"coordinate_every": 6, "cooperate_min": 0.50, "risk_limit": 0.72},
}

COMMUNICATION_LINES = {
    "medical": [
        "I can spare medical support. Hold position and stabilize.",
        "You are below safe vitals. I am transferring emergency aid.",
    ],
    "food": [
        "Food buffer available. Sharing one ration now.",
        "Your satiety is low. Take this food and recover.",
    ],
    "water": [
        "Water buffer available. Sharing ice for hydration now.",
        "Your hydration is low. Take this ice and drink before continuing.",
    ],
    "shelter": [
        "Local risk is high. Prioritizing shelter and recovery coverage.",
        "We need a safe tile here before pushing farther.",
    ],
    "greenhouse": [
        "Food and biomass are the next bottleneck. Coordinating greenhouse work.",
        "Greenhouse priority: stabilize food before expanding.",
    ],
    "oxygen": [
        "Oxygen support is the next build priority.",
        "Life support gap detected. Coordinating oxygen infrastructure.",
    ],
    "resources": [
        "I am collecting construction inputs for the next build.",
        "Resource buffer is low. Switching to collection work.",
    ],
    "colony": [
        "Colony support is stable. Continue infrastructure expansion.",
        "This sector can support more development. Maintain coordinated progress.",
    ],
}

STRUCTURE_ACTIONS = {
    StructureType.SHELTER: ActionType.BUILD_SHELTER,
    StructureType.SOLAR_ARRAY: ActionType.BUILD_SOLAR_ARRAY,
    StructureType.OXYGEN_PLANT: ActionType.BUILD_OXYGEN_PLANT,
    StructureType.GREENHOUSE: ActionType.BUILD_GREENHOUSE,
    StructureType.WATER_EXTRACTOR: ActionType.BUILD_WATER_EXTRACTOR,
    StructureType.HEATER: ActionType.BUILD_HEATER,
    StructureType.INFIRMARY: ActionType.BUILD_INFIRMARY,
    StructureType.HABITAT: ActionType.BUILD_HABITAT,
    StructureType.RESEARCH_LAB: ActionType.BUILD_RESEARCH_LAB,
    StructureType.STORAGE_DEPOT: ActionType.BUILD_STORAGE_DEPOT,
    StructureType.WEATHER_STATION: ActionType.BUILD_WEATHER_STATION,
}

EARLY_INFRASTRUCTURE_DELAY_STEPS = 6
WATER_FORECAST_RESERVE_STEPS = 14
WATER_FORECAST_RATION_PER_STEP = 0.1

# Periodic scouting expeditions: healthy, well-stocked colonists leave the
# colony toward unexplored cells and come back, so the colony keeps mapping
# its surroundings instead of freezing in place after the first year.
SCOUT_INTERVAL_STEPS = 24
# Maximum outbound work expressed in macro-cell equivalents. The runtime
# multiplies it by the metric steps needed to cross one cell, so lowering the
# unified vision/movement range slows a mission instead of truncating it after
# the same 30 weekly actions used near the 59 km setting.
SCOUT_MAX_OUT_STEPS = 30
# A single scout covers only a short, safely reversible leg. Wider coverage is
# obtained by handing the frontier to supported outposts, not by stretching
# one colonist's round trip until physiological overrides can strand them.
SCOUT_RADIUS_CELLS = 3
SCOUT_MIN_WATER_RESERVE = 3.4
# A range-2 round trip costs ~4.8 food in rations (24 moves each way at 0.1):
# launching with 2.0 starved scouts mid-return (validation v2, agents 075/118).
SCOUT_MIN_FOOD = 4.0
# Expedition logistics derives crossing time from the configured movement
# distance and local Mars-cell geometry. One ration is consumed per step.
SCOUT_RATION_PER_STEP = 0.1
SCOUT_TARGET_WATER_RESERVE = 7.0
SCOUT_TARGET_FOOD = 4.0

# Settlement expeditions: when the home cell already satisfies every coverage
# ratio and is crowded, provisioned high-curiosity colonists head to a virgin
# cell nearby and found an outpost, greenhouse first (its water/food support
# makes the new site self-sufficient within a couple of steps).
SETTLE_INTERVAL_STEPS = 60
SETTLE_MIN_CROWD = 14
SETTLE_RADIUS_CELLS = 2
# Crossing one ~59 km cell costs ~12 moves = 1.2 water in rations: launch with
# a full canteen and turn back while the walk home is still affordable — the
# first validation run lost 27 settlers to dehydration on the frontier with
# thinner margins (6.0 launch / 2.0 abort).
SETTLE_MIN_WATER_RESERVE = 7.0
SETTLE_ABORT_WATER_RESERVE = 3.5
SETTLE_MIN_FOOD = 4.0
SETTLE_MAX_OUT_STEPS = 40
SETTLE_MIN_CURIOSITY = 0.45
# The vectorized biology kernel consumes a tiny amount of carried building
# material at the end of every weekly step.  A kit matching the greenhouse
# cost exactly therefore arrived one crossing later at 2.9998 < 3.0 and could
# no longer open the promised site.  This is logistics slack, not a second
# building: the unused fraction remains in the founder's inventory.
SETTLE_BUILD_MATERIAL_BUFFER = 0.05
#: Cio' che una comunita' puo' mettere in una spedizione senza mettere a
#: rischio chi resta: i minerali e il materiale di un colono non lo tengono
#: in vita, l'acqua e il cibo si'.
RISORSE_INDUSTRIALI = ("construction_material", "minerals", "energy")

# Le quattro strutture dell'avviamento hanno una definizione sola, in
# `src/world/structures.py`: le leggono il kit del fondatore, i posti di
# cantiere e la dotazione di regolito di ogni cella. L'alloggio fa parte
# dell'avviamento e va riservato nel kit come gli altri, altrimenti il ricovero
# consuma cio' che era destinato all'impianto d'ossigeno e un avamposto resta
# senza supporto vitale per il resto della run.
SETTLE_BOOTSTRAP_STRUCTURES = STRUTTURE_DI_AVVIAMENTO
# **Erano due elenchi per la stessa cosa (2026-09-01).** Il kit del fondatore e
# la griglia che rende operativo l'avamposto sono lo stesso insieme — le
# strutture che `local_life_support_capacity` mette in `min` — e tenerli
# separati significa che aggiungerne una a uno solo dei due manda il fondatore
# in giro con il kit giusto e la lista dei lavori sbagliata. E' successo il
# giorno stesso in cui il pozzo idrico e' entrato fra i requisiti.
SETTLE_OPERATIONAL_STRUCTURES = STRUTTURE_DI_AVVIAMENTO
# Crowd dispersal to a neighbor cell is a ~12-move walk too: agents leave only
# with enough canteen to come back if the new cell has no water support yet.
DISPERSAL_MIN_WATER_RESERVE = 3.5

# A build action rejected this many times in the recent window goes on
# cooldown: the agent abandons the plan instead of looping on rejections.
BUILD_REJECTION_THRESHOLD = 2
BUILD_REJECTION_WINDOW_STEPS = 8
# Collect actions back off after a single empty grab: the cell pool is shared
# and refills as a trickle, so a second attempt inside the window is almost
# always another empty grab.
COLLECT_REJECTION_THRESHOLD = 1

class RuleBasedAgent(BaseAgent):
    def _structure_targets(self, world, structure_types: set[StructureType]):
        """Return a per-step cached list of cells containing support types."""
        cache_step = int(getattr(world, "step", 0))
        cache_state = getattr(world, "_structure_target_cache", None)
        if not cache_state or cache_state[0] != cache_step:
            cache_state = (cache_step, {})
            world._structure_target_cache = cache_state
        cache_key = tuple(
            sorted(structure_type.value for structure_type in structure_types)
        )
        targets = cache_state[1].get(cache_key)
        if targets is None:
            # Both GridWorld and the vectorized WorldView maintain a compact
            # set of occupied structure coordinates. Iterating those positions
            # avoids two Python scans of all 64,800 Mars cells per step when
            # every agent evaluates its food/water return budget.
            positions = getattr(world, "_structure_positions", None)
            if positions is None:
                positions = (
                    (cell.x, cell.y)
                    for row in world.cells
                    for cell in row
                    if cell.structures
                )
            targets = []
            for x, y in positions:
                cell = world.get_cell(int(x), int(y))
                if any(
                    structure.type in structure_types
                    for structure in cell.structures
                ):
                    targets.append(cell)
            cache_state[1][cache_key] = targets
        return targets

    def _movement_steps_per_cell(self, world) -> int:
        """Conservative steps needed to cross one adjacent macro-cell."""
        cell = world.get_cell(self.x, self.y)
        width_m = float(cell.geometry.get("width_m", 1.0))
        height_m = float(cell.geometry.get("height_m", 1.0))
        movement_m = max(
            0.1, float(getattr(self, "movement_distance_m_per_step", 59_000.0))
        )
        return max(1, math.ceil(math.hypot(width_m, height_m) / movement_m))

    def _support_distance_cells(
        self,
        world,
        structure_types: set[StructureType],
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> int | None:
        origin_x = self.x if x is None else int(x)
        origin_y = self.y if y is None else int(y)
        type_key = tuple(
            sorted(structure_type.value for structure_type in structure_types)
        )

        # Task 3/4 della migrazione Rust. La formulazione storica (sotto, ramo
        # `scalar`) ricalcola il minimo su tutti i target per OGNI origine e lo
        # memorizza per origine. Il profilo post-0b la misura fra le voci piu'
        # costose della decisione. I backend `numpy` e `rust` calcolano invece
        # l'intero campo di distanza una volta per step e servono ogni query come
        # un accesso ad array.
        #
        # Il risultato deve restare identico: stesso minimo, stessa distanza
        # Chebyshev avvolta, stesso `None` quando non ci sono target. La parita'
        # bit-exact fra i tre backend e' verificata dall'harness di parita' sui
        # seed golden.
        backend = reachability_batch.configured_backend()
        if backend != "scalar":
            targets = self._structure_targets(world, structure_types)
            if not targets:
                return None
            width = int(world.width)
            height = int(world.height)
            if 0 <= origin_x < width and 0 <= origin_y < height:
                field = reachability_batch.cached_support_distance_field(
                    world, type_key, targets, backend
                )
                distance = int(field[origin_y * width + origin_x])
                # Il campo non puo' contenere il sentinella qui: `targets` non e'
                # vuoto, quindi ogni cella ha almeno un supporto a distanza
                # finita. Il controllo protegge da un disallineamento futuro fra
                # il ramo "nessun target" e il contenuto del campo.
                return None if distance == reachability_batch.NO_SUPPORT else distance
            # Origine fuori griglia: nessun chiamante attuale la produce (i
            # candidati vengono da `world.neighbors`, che avvolge x e limita y),
            # ma indicizzare il campo con coordinate fuori intervallo darebbe
            # silenziosamente la distanza di un'altra cella. Si ricade sul
            # calcolo diretto, che e' definito ovunque.

        cache_step = int(getattr(world, "step", 0))
        cache_state = getattr(world, "_support_distance_cache", None)
        if not cache_state or cache_state[0] != cache_step:
            cache_state = (cache_step, {})
            world._support_distance_cache = cache_state
        cache_key = (type_key, origin_x, origin_y)
        if cache_key in cache_state[1]:
            return cache_state[1][cache_key]
        targets = self._structure_targets(world, structure_types)
        if not targets:
            cache_state[1][cache_key] = None
            return None
        distance = min(
            wrapped_chebyshev_distance(
                origin_x, origin_y, target.x, target.y, world.width
            )
            for target in targets
        )
        cache_state[1][cache_key] = distance
        return distance

    def _return_ration_requirement(
        self,
        world,
        structure_types: set[StructureType],
        *,
        x: int | None = None,
        y: int | None = None,
        safety_steps: int = 2,
    ) -> tuple[float, int | None]:
        distance = self._support_distance_cells(
            world, structure_types, x=x, y=y
        )
        if distance is None:
            return float("inf"), None
        travel_steps = distance * self._movement_steps_per_cell(world)
        return (travel_steps + max(0, safety_steps)) * SCOUT_RATION_PER_STEP, distance

    def _can_visit_cell_and_return(self, world, x: int, y: int) -> bool:
        """Whether carried rations cover the proposed move and safe return."""
        food_required, food_distance = self._return_ration_requirement(
            world, {StructureType.GREENHOUSE}, x=x, y=y
        )
        water_required, water_distance = self._return_ration_requirement(
            world,
            {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
            x=x,
            y=y,
        )
        # Minimal unit worlds may intentionally contain no settlement. Real
        # runs seed initial support before decisions; preserve free exploration
        # only when there is no known home of either kind to return to.
        if food_distance is None and water_distance is None:
            return True
        if food_distance is None or water_distance is None:
            return False
        ration_after_move = SCOUT_RATION_PER_STEP
        return (
            float(self.inventory.food) - ration_after_move >= food_required
            and float(self.inventory.water + self.inventory.ice) - ration_after_move
            >= water_required
        )

    def _return_for_ration_budget(self, world) -> ActionRequest | None:
        """Start the trip home before the remaining ration budget is consumed."""
        food_required, food_distance = self._return_ration_requirement(
            world, {StructureType.GREENHOUSE}
        )
        if (
            food_distance is not None
            and food_distance > 0
            and float(self.inventory.food) <= food_required
        ):
            return self._move_toward_structure(
                world,
                {StructureType.GREENHOUSE},
                "food return budget reached: heading to greenhouse support",
            )
        water_required, water_distance = self._return_ration_requirement(
            world,
            {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
        )
        if (
            water_distance is not None
            and water_distance > 0
            and float(self.inventory.water + self.inventory.ice) <= water_required
        ):
            return self._move_toward_structure(
                world,
                {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
                "low water reserve: return budget reached; heading to life support",
            )
        return None

    def _is_move_viable(self, x: int, y: int) -> bool:
        for f in getattr(self.memory, "recent_failures", [])[-5:]:
            if f["action"] == ActionType.MOVE.value and f["target"] and f["target"].get("x") == x and f["target"].get("y") == y:
                return False
        return True

    def _process_task_queue(self, cell, existing, world) -> ActionRequest | None:
        if not hasattr(self.memory, "task_queue") or not self.memory.task_queue:
            return None
        # Process the first task
        task = self.memory.task_queue.pop(0)
        if task["type"] == "build" and task["target"] == "greenhouse" and StructureType.GREENHOUSE not in existing:
            if self.inventory.can_afford(BUILD_COSTS[StructureType.GREENHOUSE]):
                req = ActionRequest(self.agent_id, ActionType.BUILD_GREENHOUSE, message="building greenhouse as requested by team")
                if not any(f["action"] == req.action.value for f in getattr(self.memory, "recent_failures", [])[-3:]):
                    return req
        elif task["type"] == "collect" and task["target"] == "ice":
            if cell.water_ice > 0.1 or cell.resources.ice > 0.1:
                return ActionRequest(self.agent_id, ActionType.COLLECT_ICE, message="collecting ice as requested by team")
            else:
                move_req = self._move_toward_nearby_ice(world)
                if move_req:
                    move_req.message = "moving toward ice as requested by team"
                    return move_req
        return None

    def decide(self, observation: LocalObservation, world) -> ActionRequest:
        cell = world.get_cell(self.x, self.y)
        existing = {s.type for s in cell.structures}

        emergency = self._emergency_action(cell, world)
        if emergency:
            return emergency

        water_return = self._water_return_action(cell, world)
        if water_return:
            return water_return

        # Rest cannot fix starvation or dehydration: when deprived with no local
        # food/water source, forcing rest would trap the agent in a death spiral.
        critical_deprivation = (
            (int(getattr(self, "steps_without_food", 0)) >= 8 and self.inventory.food < 0.1 and not _has_food_source(cell))
            or (int(getattr(self, "steps_without_water", 0)) >= 3 and not _has_water_source(self, cell))
        )

        if self.fatigue > 0.85 and not critical_deprivation:
            return ActionRequest(self.agent_id, ActionType.REST, message="extreme fatigue; forced rest to prevent collapse")

        if self.health < 0.40 and self._should_recover(cell, observation) and not critical_deprivation:
            return ActionRequest(self.agent_id, ActionType.REST, message="critical health; forced rest to prevent death from environmental exposure")

        forecast = self._water_forecast_action(world, cell, observation)
        if forecast:
            return forecast

        water_security = self._water_security_action(world, cell, observation)   
        if water_security:
            return water_security

        food_security = self._food_security_action(world, cell, observation)     
        if food_security:
            return food_security

        oxygen_security = self._oxygen_security_action(world, cell, observation) 
        if oxygen_security:
            return oxygen_security

        if aid := self._nearby_agent_to_help(observation):
            return aid

        if self.stress_index > 0.72 and observation.nearby_agents and self._should_coordinate(observation):
            return ActionRequest(
                self.agent_id,
                ActionType.COMMUNICATE,
                {"agent_id": self._communication_target(observation)},
                "Stress is high. Requesting a crew check-in before continuing work.",
            )

        if not self._has_critical_neighbor(observation) and self._should_coordinate(observation) and self._coordination_topic(existing, observation) in {"shelter", "resources"}:
            return ActionRequest(
                self.agent_id,
                ActionType.COMMUNICATE,
                {"agent_id": self._communication_target(observation)},
                self._communication_line(self._coordination_topic(existing, observation)),
            )

        if self._should_recover(cell, observation):
            return ActionRequest(self.agent_id, ActionType.REST, message="recovering before higher-risk work")

        urgent_build = self._critical_infrastructure(existing, observation, world)
        if urgent_build:
            return urgent_build

        # _resource_action books intra-step pool claims, so it must run only
        # where its proposal is actually used: a claim from a discarded plan
        # would lock the stock away from colonists who would collect it.
        resource_needed = self._needs_resources(existing, cell)
        if resource_needed:
            resource_action = self._resource_action(cell, existing, world)
            if resource_action:
                return resource_action

        # Founding outranks the marginal local build: with settlement checked
        # only in the movement fallback, any colonist who finally held the
        # founder kit was intercepted here first — the demand builder spent
        # the materials on the Nth local greenhouse/solar and the expedition
        # rung never saw a provisioned candidate (three 400-step validations
        # with ZERO launches despite open valve gates). The stagger makes
        # this check a cheap no-op on almost every turn.
        settlement = self._settlement_action(world, cell)
        if settlement:
            return settlement

        planned_build = self._planned_infrastructure(existing, observation, world)
        if planned_build:
            return planned_build

        if self._should_coordinate(observation):
            topic = self._coordination_topic(existing, observation)
            return ActionRequest(
                self.agent_id,
                ActionType.COMMUNICATE,
                {"agent_id": self._communication_target(observation)},
                self._communication_line(topic),
            )

        if not resource_needed:
            resource_action = self._resource_action(cell, existing, world)
            if resource_action:
                return resource_action

        return self._movement_or_observe(world, observation)

    def _nearby_agent_to_help(self, observation: LocalObservation) -> ActionRequest | None:
        if self.cooperation <= 0.38:
            return None
        for neighbor in sorted(observation.nearby_agents_details, key=lambda row: (row.get("health", 1.0), row.get("hydration", 1.0), row.get("satiety", 1.0))):
            target = neighbor["agent_id"]
            name = neighbor.get("name", target)
            if neighbor.get("health", 1.0) < 0.42 and self.inventory.med_kits >= 1.0:
                return ActionRequest(
                    self.agent_id,
                    ActionType.SHARE_RESOURCE,
                    {"agent_id": target, "resources": {"med_kits": 1.0}},
                    self._communication_line("medical").replace(".", f" for {name}.", 1),
                )
            if neighbor.get("hydration", 1.0) < 0.35 and (self.inventory.water >= 1.0 or self.inventory.ice >= 1.0):
                offered = {"water": 1.0} if self.inventory.water >= 1.0 else {"ice": 1.0}
                return ActionRequest(
                    self.agent_id,
                    ActionType.SHARE_RESOURCE,
                    {"agent_id": target, "resources": offered},
                    self._communication_line("water").replace(".", f" with {name}.", 1),
                )
            if neighbor.get("satiety", 1.0) < 0.35 and self.inventory.food >= 2.0:
                return ActionRequest(
                    self.agent_id,
                    ActionType.SHARE_RESOURCE,
                    {"agent_id": target, "resources": {"food": 1.0}},
                    self._communication_line("food").replace(".", f" with {name}.", 1),
                )
        return None

    def _has_critical_neighbor(self, observation: LocalObservation) -> bool:
        return any(neighbor.get("health", 1.0) < 0.42 or neighbor.get("hydration", 1.0) < 0.35 or neighbor.get("satiety", 1.0) < 0.35 for neighbor in observation.nearby_agents_details)

    def _water_return_action(self, cell, world) -> ActionRequest | None:
        """Hard survival override: with the dehydration clock running out, drop
        everything, drink if possible, otherwise head straight back to the
        nearest water-support cell."""
        if int(getattr(self, "steps_without_water", 0)) < 4:
            return None
        if _has_water_source(self, cell):
            return ActionRequest(self.agent_id, ActionType.DRINK_WATER, message="dehydration emergency: drinking now")
        move = self._move_toward_structure(
            world,
            {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
            "dehydration emergency: returning to water-support cell",
        )
        if move:
            return move
        return self._move_toward_nearby_ice(world) or self._move_toward_polar_ice(world)

    def _emergency_action(self, cell, world) -> ActionRequest | None:
        risk = traversal_risk_for_cell(cell)
        if risk > 0.85 and self.health < 0.60:
            escape = self._move_to_less_crowded_safe_neighbor(world)
            if escape:
                escape.message = f"emergency escape from high-risk terrain ({cell.terrain.value})"
                return escape

        if self.inventory.med_kits >= 1 and (self.health < 0.55 or self.oxygen_level < 0.32 or self.hydration < 0.28):
            return ActionRequest(self.agent_id, ActionType.USE_MED_KIT, message="emergency medical stabilization")
        if self.hydration < 0.70 and _has_water_source(self, cell):
            return ActionRequest(self.agent_id, ActionType.DRINK_WATER, message="drinking before dehydration risk")
        if self.hydration < 0.78 and self.inventory.ice < 1.0 and (cell.water_ice > 0.1 or cell.resources.ice > 0.1):
            return ActionRequest(self.agent_id, ActionType.COLLECT_ICE, message="collecting ice for drinking water")
        if self.satiety < 0.70 and self.inventory.food >= 0.1:
            return ActionRequest(self.agent_id, ActionType.EAT_FOOD, message="eating before starvation risk")
        if self.satiety < 0.78 and self.inventory.food < 1.0 and _has_food_source(cell):
            return ActionRequest(self.agent_id, ActionType.FORAGE, message="foraging from available local food source")
        return None

    def _water_forecast_action(self, world, cell, observation: LocalObservation) -> ActionRequest | None:
        if observation.agent_count < 4:
            return None
        reserve = self.inventory.water + self.inventory.ice
        local_agents = max(1, int(observation.agent_count))
        local_ice = max(0.0, cell.water_ice + cell.resources.ice)
        water_support = sum(float(structure.local_effect.get("water", 0.0)) for structure in cell.structures)
        # Colony-wide balance, kept DELIBERATELY pessimistic at scale: a
        # per-capita rewrite (my 14-step need vs my canteen + my share of the
        # wells) was tried in the run5 expansion audit and REVERTED — it was
        # formally more correct but this pessimistic pressure is the colony's
        # de-facto greenhouse engine, and without it the validated economy
        # collapsed (greenhouses 68 -> 32, ISRU 0.53 -> 0.33 at 400 steps)
        # while expeditions stayed blocked anyway. Known side effect: at high
        # crowd the forecast intercepts kit-ready founders into local builds;
        # unblocking the expedition machine needs a dedicated founder-
        # commitment design, tracked as an open item.
        projected_need = local_agents * WATER_FORECAST_RATION_PER_STEP * WATER_FORECAST_RESERVE_STEPS
        projected_supply = local_ice + reserve + water_support * WATER_FORECAST_RESERVE_STEPS
        pressure = projected_need - projected_supply
        if pressure <= 0.0 and reserve >= 4.5:
            return None

        if local_ice > 0.1 and (reserve < 6.0 or pressure > 0.0):
            return ActionRequest(
                self.agent_id,
                ActionType.COLLECT_ICE,
                message="forecast water deficit: extracting ice before colony rations collapse",
            )

        build_pressure = (pressure > 0.0 or (local_agents >= 4 and reserve < 4.5)) and not self._on_expedition
        if build_pressure:
            priorities = [StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY]
            if self.oxygen_level < 0.82:
                priorities.insert(0, StructureType.OXYGEN_PLANT)
            build = self._first_buildable(priorities, {s.type for s in cell.structures}, world=world, cell=cell)
            if build:
                build.message = "forecast water deficit: building water-support infrastructure before mass dehydration"
                return build

        if reserve <= 3.2:
            if self.inventory.food < SCOUT_MIN_FOOD:
                # The ice road is an expedition and runs on carried rations:
                # in the 1000-step 200-colonist run the colony-wide forecast
                # pushed food-poor colonists 3 cells out to buried ice, where
                # 11 starved together at the ice field with FULL hydration.
                # Short on food, walk home instead: the colony feeds AND waters.
                home = self._move_toward_structure(
                    world,
                    {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
                    "forecast water deficit but rations short: returning to colony support",
                )
                if home:
                    return home
            nearby = self._move_toward_nearby_ice(world)
            if nearby:
                nearby.message = "forecast water deficit: moving toward nearby ice before emergency"
                return nearby
        return None

    def _water_security_action(self, world, cell, observation: LocalObservation) -> ActionRequest | None:
        reserve = self.inventory.water + self.inventory.ice
        local_water_support = any(float(s.local_effect.get("water", 0.0)) > 0.0 for s in cell.structures)
        steps_without_water = int(getattr(self, "steps_without_water", 0))
        move_streak = self._recent_action_streak(ActionType.MOVE.value)
        # The travel streak-breaker only stops for a drink when hydration is
        # actually draining: forcing a sip at full hydration every 4th step
        # taxed the walk home by ~33% while the starvation clock ran (the 13
        # dead of the 1000-step 200-colonist run all show the move-move-move-
        # drink loop with hydration 1.0 and satiety 0.0).
        if steps_without_water >= 2 or (move_streak >= 3 and reserve <= 2.8 and self.hydration < 0.92):
            if _has_water_source(self, cell):
                return ActionRequest(self.agent_id, ActionType.DRINK_WATER, message="breaking dehydration streak before more movement")
            if cell.water_ice > 0.1 or cell.resources.ice > 0.1:
                return ActionRequest(self.agent_id, ActionType.COLLECT_ICE, message="stopping travel to extract local ice for water")
            support_move = self._move_toward_structure(world, {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY}, "returning to water-capable life support")
            if support_move:
                return support_move
            ice_move = self._move_toward_nearby_ice(world) or self._move_toward_polar_ice(world)
            if ice_move:
                ice_move.message = "water emergency: moving toward nearest reachable ice"
                return ice_move
        if reserve >= 3.2 or local_water_support:
            return None
        if cell.water_ice > 0.1 or cell.resources.ice > 0.1:
            return ActionRequest(self.agent_id, ActionType.COLLECT_ICE, message="securing water reserve from local ice before rations run out")
        if reserve <= 2.4 and not self._on_expedition:
            existing = {s.type for s in cell.structures}
            priorities = [StructureType.GREENHOUSE]
            if StructureType.SHELTER in existing and StructureType.OXYGEN_PLANT in existing:
                priorities.append(StructureType.HABITAT)
            build = self._first_buildable(priorities, existing, world=world, cell=cell, ignore_water_buffer=True)
            if build:
                build.message = "building water-capable life support before emergency rations run out"
                return build
        if reserve <= 1.8 or (self.hydration < 0.55 and reserve <= 2.4):
            if self.inventory.food < SCOUT_MIN_FOOD:
                # Same ration budget as scouts: no ice pilgrimage on an empty
                # pantry — walking home rescues water AND food at once.
                home = self._move_toward_structure(
                    world,
                    {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
                    "water low with rations short: returning to colony support",
                )
                if home:
                    return home
            local_ice_search = self._move_toward_nearby_ice(world) if self.autonomy_preference > 0.72 else None
            if local_ice_search:
                return local_ice_search
            return self._move_toward_polar_ice(world)
        return None

    def _food_security_action(self, world, cell, observation: LocalObservation) -> ActionRequest | None:
        if self.inventory.food <= 1.5 and _has_food_source(cell):
            return ActionRequest(self.agent_id, ActionType.FORAGE, message="stocking food before emergency rations run out")
        if self.inventory.food > 4.2:
            return None
        if self._on_expedition:
            # Expeditions live on carried rations: no security construction
            # sites scattered along the route (the abort logic walks the agent
            # home before rations run out).
            return None
        existing = {s.type for s in cell.structures}
        build = self._first_buildable([StructureType.GREENHOUSE, StructureType.HABITAT], existing, world=world, cell=cell)
        if build:
            build.message = "building food production before emergency rations run out"
            return build
        # The walk home must start while rations still cover it: one cell is
        # ~12 moves at 0.1 food each, and the starvation clock allows 14 steps.
        if self.inventory.food <= 2.6:
            return self._move_toward_structure(world, {StructureType.GREENHOUSE, StructureType.HABITAT}, "moving toward colony food production")
        return None

    def _oxygen_security_action(self, world, cell, observation: LocalObservation) -> ActionRequest | None:
        local_oxygen_support = any(float(s.local_effect.get("oxygen", 0.0)) > 0.0 for s in cell.structures)
        oxygen_pressure = self.oxygen_level < 0.82 or self.inventory.oxygen < 0.8
        if not oxygen_pressure or local_oxygen_support:
            return None
        build = self._first_buildable([StructureType.OXYGEN_PLANT, StructureType.HABITAT, StructureType.INFIRMARY], {s.type for s in cell.structures}, world=world, cell=cell)
        if build:
            build.message = "building oxygen life support before suit reserves run out"
            return build
        if self.oxygen_level < 0.72 or self.inventory.oxygen < 0.5:
            return self._move_toward_structure(world, {StructureType.OXYGEN_PLANT, StructureType.HABITAT, StructureType.INFIRMARY}, "moving toward colony oxygen support")
        return None

    def _should_recover(self, cell, observation: LocalObservation) -> bool:
        shelter_bonus = self._has_recovery_support(cell)
        vital_pressure = self.health < 0.68 or self.oxygen_level < 0.48 or self.hydration < 0.45 or self.fatigue > 0.72
        psychosocial_pressure = self.stress_index > 0.78 or (self.morale < 0.35 and self.stress_index > 0.58)
        environmental_pressure = observation.local_danger > 1.45 and self.health < 0.86
        # Comfort rest at a shelter is safe only where water flows or the
        # canteen outlasts the stay: a dry shelter must not park thirsty agents.
        watered_rest = shelter_bonus and self.fatigue > 0.55 and (
            any(float(s.local_effect.get("water", 0.0)) > 0.0 for s in cell.structures)
            or (self.inventory.water + self.inventory.ice) >= 2.4
        )
        return vital_pressure or psychosocial_pressure or environmental_pressure or watered_rest

    def _has_recovery_support(self, cell) -> bool:
        return any(s.type in {StructureType.SHELTER, StructureType.HABITAT, StructureType.INFIRMARY} for s in cell.structures)

    def _critical_infrastructure(self, existing: set[StructureType], observation: LocalObservation, world) -> ActionRequest | None:
        # 1. Continue in-progress construction
        if observation.construction_sites:
            for struct_val in sorted(observation.construction_sites, key=lambda k: observation.construction_sites[k], reverse=True):
                structure_name = str(struct_val).split("@", 1)[0]
                try:
                    struct_type = StructureType(structure_name)
                except ValueError:
                    continue
                if struct_type not in STRUCTURE_ACTIONS:
                    continue
                if self._local_structure_exists(struct_type, world.get_cell(self.x, self.y)):
                    continue
                # A finished site turns queued "continue" proposals into brand-new
                # sites at execution time, so saturation must gate continuation
                # too or buildings chain-build forever.
                if self._structure_saturated(struct_type, world.get_cell(self.x, self.y)):
                    continue
                if self._recently_rejected(STRUCTURE_ACTIONS[struct_type], world):
                    continue
                return ActionRequest(
                    self.agent_id,
                    STRUCTURE_ACTIONS[struct_type],
                    message=f"continuing construction of {structure_name} ({observation.construction_sites[struct_val]:.0f}%)",
                )

        priorities: list[StructureType] = []
        if observation.local_danger > 1.55 or self.health < 0.82:
            priorities.append(StructureType.SHELTER)
        if self.oxygen_level < 0.70:
            priorities.append(StructureType.OXYGEN_PLANT)
        survival_grid_ready = self._local_survival_grid_ready(existing)
        if self.hydration < 0.62 or self.satiety < 0.88 or (self.inventory.food < 4.2 and self.inventory.ice >= 1.0):
            priorities.append(StructureType.GREENHOUSE)
        if self.health < 0.58 or (survival_grid_ready and (self.health < 0.82 or self.inventory.med_kits < 1)):
            priorities.append(StructureType.INFIRMARY)
        cell = world.get_cell(self.x, self.y)
        return self._first_buildable(priorities, existing, world=world, cell=cell)

    def _planned_infrastructure(self, existing: set[StructureType], observation: LocalObservation, world) -> ActionRequest | None:
        """General-colonist development plan (single role, no specialist branches).

        Survival first (food/water, oxygen, shelter), then base expansion once the
        local survival grid exists: power, habitat, care, forecasting, science.
        """
        cell = world.get_cell(self.x, self.y)
        if not self._field_ready_for_infrastructure(world):
            return None
        priorities: list[StructureType] = []
        survival_grid_ready = self._local_survival_grid_ready(existing)
        established = survival_grid_ready or getattr(world, "step", 0) >= 30
        reserve = self.inventory.water + self.inventory.ice

        # Homesteading: a provisioned pioneer on ground with no greenhouse or
        # habitat plants the greenhouse deliberately, not out of hunger — this
        # is what turns dispersal into safe organic expansion (in v7 pioneers
        # with full rations idled on virgin cells because every build rule was
        # need-driven, and expansion collapsed to 4 cells).
        settled_here = any(s.type in {StructureType.GREENHOUSE, StructureType.HABITAT} for s in cell.structures)
        # Threshold 4.5, not 5.0: dispersers top up at the well to ~6.0 and a
        # one-cell crossing costs ~1.2 rations, so pioneers land at ~4.8 — the
        # 5.0 gate rejected virtually every provisioned drifter (zero
        # homesteads in the 960-step 300-founder run). 4.5 still clears the
        # 3.5 abort floor plus the greenhouse build's own water cost.
        if not settled_here and reserve >= 4.5 and self.inventory.food >= 3.0:
            priorities.append(StructureType.GREENHOUSE)

        if reserve < 3.0 or self.inventory.food < 4.2:
            priorities.append(StructureType.GREENHOUSE)
        if self.oxygen_level < 0.90:
            priorities.append(StructureType.OXYGEN_PLANT)
        if observation.local_danger > 0.90:
            priorities.append(StructureType.SHELTER)

        if survival_grid_ready or settled_here:
            # Demand-driven life support: the coverage ratios in build_policy
            # stop these once the cell serves its population, and the types
            # are proposed by proportional deficit — with a fixed order the
            # greenhouse (first in line, same 1/7 ratio) shadowed the solar
            # array out of the build slot (18 vs 32 arrays in validation v4).
            # Settled cells qualify even without the full survival grid: the
            # old oxygen-plant prerequisite locked satellite settlements out
            # of their own power and oxygen — in the 1000-step 200-colonist
            # run ALL 32 solar arrays and 20 oxygen plants sat in the mother
            # cell while 35 outpost greenhouses added colony-wide load, and
            # the power margin decayed 1.09 -> 0.66.
            colonists = max(1, len(cell.agents_present))
            deficits: list[tuple[float, StructureType]] = []
            for structure in (StructureType.GREENHOUSE, StructureType.OXYGEN_PLANT, StructureType.SOLAR_ARRAY):
                needed = max(1, -(-colonists // COLONISTS_PER_STRUCTURE[structure]))
                have = sum(1 for s in cell.structures if s.type == structure)
                if have < needed:
                    deficits.append((have / needed, structure))
            priorities.extend(structure for _, structure in sorted(deficits, key=lambda pair: pair[0]))
            if StructureType.SHELTER in existing and StructureType.OXYGEN_PLANT in existing:
                priorities.append(StructureType.HABITAT)
            if observation.local_danger > 0.85 or self.health < 0.82:
                priorities.append(StructureType.INFIRMARY)
        if established:
            # Support infrastructure follows homes: on cells without local
            # food+water production these become wasteland bait — drifters
            # flocked to heater/weather micro-sites and starved there (the
            # v4/v5 validation death clusters were exactly such cells).
            if settled_here:
                priorities.extend([StructureType.WEATHER_STATION, StructureType.HEATER, StructureType.STORAGE_DEPOT])
                # Gate on the habitability of the local colony cell (which includes
                # structure effects), not the planet-wide average: the global mean
                # stays ~0.004 for decades and made research labs unreachable.
                if cell.habitability_score >= 0.015:
                    priorities.append(StructureType.RESEARCH_LAB)
        return self._first_buildable(priorities, existing, world=world, cell=cell)

    def _local_survival_grid_ready(self, existing: set[StructureType]) -> bool:
        return StructureType.OXYGEN_PLANT in existing and (
            StructureType.GREENHOUSE in existing or StructureType.HABITAT in existing
        )

    def _first_buildable(
        self,
        priorities: list[StructureType],
        existing: set[StructureType],
        world=None,
        cell=None,
        ignore_water_buffer: bool = False,
    ) -> ActionRequest | None:
        if world is not None and not self._field_ready_for_infrastructure(world):
            priorities = [s for s in priorities if s in {StructureType.SHELTER, StructureType.OXYGEN_PLANT}]
            if (self.inventory.water + self.inventory.ice) >= 2.5 and (self.inventory.food < 4.2 or self.hydration < 0.75):
                priorities.append(StructureType.GREENHOUSE)
        for structure in priorities:
            # The greenhouse costs water: never pour the survival buffer into
            # mortar — keep at least 1.5 rations after the build. The water
            # emergency is the exception: for a stranded agent a greenhouse IS
            # the water rescue (2 steps to a working well), while walking home
            # costs ~12 moves per cell.
            if (
                structure == StructureType.GREENHOUSE
                and not ignore_water_buffer
                and (self.inventory.water + self.inventory.ice) < 2.5
            ):
                continue
            if self._build_blocked(structure, cell):
                continue
            if cell is not None and self._structure_saturated(structure, cell):
                continue
            if self._recently_rejected(STRUCTURE_ACTIONS[structure], world):
                continue
            if self.inventory.can_afford(BUILD_COSTS[structure]):
                return ActionRequest(
                    self.agent_id,
                    STRUCTURE_ACTIONS[structure],
                    message=f"building {structure.value} based on rule-based survival plan",
                )
        return None

    def _resource_action(self, cell, existing: set[StructureType], world=None) -> ActionRequest | None:
        if self.satiety < 0.88 and _has_food_source(cell):
            return ActionRequest(self.agent_id, ActionType.FORAGE, message="building food buffer")
        if self.hydration < 0.85 and _has_water_source(self, cell):
            return ActionRequest(self.agent_id, ActionType.DRINK_WATER, message="building hydration buffer")
        if self.inventory.ice < 2.0 and (cell.water_ice > 0.1 or cell.resources.ice > 0.1):
            return ActionRequest(self.agent_id, ActionType.COLLECT_ICE, message="collecting ice for greenhouse and life support")
        # Resource gathering if inventory is low. Two layers break the crowd
        # contention on the shared cell pool: the intra-step claim ledger stops
        # proposals once the visible stock is already spoken for, and a single
        # empty grab puts the action on cooldown for the whole window.
        can_scavenge = not self._recently_rejected(ActionType.COLLECT_MATERIALS, world, COLLECT_REJECTION_THRESHOLD)
        can_mine = not self._recently_rejected(ActionType.COLLECT_MINERALS, world, COLLECT_REJECTION_THRESHOLD)
        if self.inventory.construction_material < 3.0:
            if can_scavenge and self._claim_cell_pool(world, cell, "construction_material"):
                return ActionRequest(self.agent_id, ActionType.COLLECT_MATERIALS, message="scavenging construction materials")
            if can_mine and self._claim_cell_pool(world, cell, "minerals"):
                return ActionRequest(self.agent_id, ActionType.COLLECT_MINERALS, message="extracting minerals and materials")

        if self.inventory.minerals < 3.0 and can_mine and self._claim_cell_pool(world, cell, "minerals"):
            return ActionRequest(self.agent_id, ActionType.COLLECT_MINERALS, message="collecting minerals for construction")
        if self.inventory.construction_material < 4.0 and can_mine and self._claim_cell_pool(world, cell, "minerals"):
            return ActionRequest(self.agent_id, ActionType.COLLECT_MINERALS, message="collecting construction material")
        if StructureType.GREENHOUSE in existing and self.inventory.food < 3.0:
            return ActionRequest(self.agent_id, ActionType.FORAGE, message="stocking greenhouse food")
        return None

    def _field_ready_for_infrastructure(self, world) -> bool:
        if getattr(world, "step", 0) >= EARLY_INFRASTRUCTURE_DELAY_STEPS:
            return True
        productive_actions = {
            ActionType.COLLECT_ICE.value,
            ActionType.COLLECT_MINERALS.value,
            ActionType.COLLECT_MATERIALS.value,
            ActionType.EXPLORE.value,
        }
        return any(action in productive_actions for action in self.recent_actions)

    def _needs_resources(self, existing: set[StructureType], cell) -> bool:
        desired = [StructureType.SHELTER, StructureType.SOLAR_ARRAY, StructureType.OXYGEN_PLANT, StructureType.GREENHOUSE]
        if StructureType.SHELTER in existing and StructureType.OXYGEN_PLANT in existing:
            desired.append(StructureType.HABITAT)
        # Demand follows the coverage ratios, not mere existence: a cell with 7
        # greenhouses serving 200 colonists still needs more of them.
        return any(
            not self._structure_saturated(structure, cell) and not self.inventory.can_afford(BUILD_COSTS[structure])
            for structure in desired
        )

    def _should_coordinate(self, observation: LocalObservation) -> bool:
        preset = ROLE_PRESETS["colonist"]
        cooperate_min = max(0.28, preset["cooperate_min"] - max(0.0, self.stress_index - 0.55) * 0.18)
        if not observation.nearby_agents or self.cooperation < cooperate_min:
            return False
        recent_comm = sum(1 for action in self.recent_actions[-3:] if action == ActionType.COMMUNICATE.value)
        cadence_hit = int(getattr(self, "actions_taken", len(self.recent_actions))) % int(preset["coordinate_every"]) == 0
        # Only acute PERSONAL needs count as signals. Chronic cell-wide facts
        # are not news to crewmates standing in the same cell: the "good news"
        # habitability clause and the ambient local_danger clause were both
        # permanently true in developed colony cells (structures raise the
        # score, crowding raises the danger), so communicate ran at the 1-in-4
        # recent-window ceiling (23.9% of actions in run3 and again in
        # validation v1) instead of the intended 1-in-6 cadence.
        need_signal = (
            self.health < 0.82
            or self.oxygen_level < 0.72
            or self.hydration < 0.68
            or self.inventory.food <= 1.5
            or self.stress_index > 0.62
        )
        return recent_comm == 0 and (cadence_hit or need_signal)

    def _coordination_topic(self, existing: set[StructureType], observation: LocalObservation) -> str:
        if self.health < 0.82 or observation.local_danger > 1.20:
            return "shelter"
        if self.inventory.ice < 2.0 or self.inventory.minerals < 3.0:
            return "resources"
        if StructureType.SHELTER not in existing:
            return "shelter"
        if StructureType.GREENHOUSE not in existing:
            return "greenhouse"
        if StructureType.OXYGEN_PLANT not in existing:
            return "oxygen"
        return "colony"

    def _communication_line(self, topic: str) -> str:
        lines = COMMUNICATION_LINES.get(topic, COMMUNICATION_LINES["colony"])
        index = (int(getattr(self, "actions_taken", len(self.recent_actions))) + sum(ord(ch) for ch in self.agent_id)) % len(lines)
        return lines[index]

    def _communication_target(self, observation: LocalObservation) -> str | None:
        nearby = observation.nearby_agents
        if not nearby:
            return None
        # Rotate deterministically through the (sorted) neighbor list: a fixed
        # nearby_agents[0] made the lexicographically first colonist the hub
        # of the entire network — 98.9% of the 34.6k messages of the
        # 1000-step run went to agent_000 (degree 408, median edge weight 1,
        # network density 0.01: a call center, not a society).
        index = (int(getattr(self, "actions_taken", 0)) + sum(ord(ch) for ch in self.agent_id)) % len(nearby)
        return nearby[index]

    def _movement_or_observe(self, world, observation: LocalObservation) -> ActionRequest:
        current = world.get_cell(self.x, self.y)
        # Settlement lives higher in decide() now (above planned builds);
        # calling it here too would double-step the expedition state machine.
        scout = self._scouting_action(world, current)
        if scout:
            return scout
        reserve = self.inventory.water + self.inventory.ice
        has_life_support = any(
            s.type in {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.OXYGEN_PLANT, StructureType.SHELTER, StructureType.INFIRMARY}
            for s in current.structures
        )
        local_water_support = any(float(s.local_effect.get("water", 0.0)) > 0.0 for s in current.structures)
        # Resting requires water on tap or a canteen that outlasts the rest:
        # shelter-only suburbs counted as "life support" and parked thirsty
        # agents until the 7-step dehydration clock beat the ~12-move walk home
        # (18 of 19 deaths in the 1000-step post-expansion audit run).
        if has_life_support and (self.oxygen_level < 0.82 or self.fatigue > 0.45) and (local_water_support or reserve >= 2.4):
            return ActionRequest(self.agent_id, ActionType.REST, message="holding near local life support instead of wandering")
        if local_water_support and reserve < 6.0:
            # Idle top-up at the well keeps colonists pioneer-ready: a full
            # canteen is what lets dispersal cross into virgin ground and
            # still beat the return-budget floor while the homestead goes up.
            return ActionRequest(self.agent_id, ActionType.REFILL_WATER, message="topping up the canteen at the local well")
        if not local_water_support and (reserve < 4.0 or self.inventory.food < 2.5):
            # Under-provisioned drifters die mid-crossing: one cell is ~12
            # moves while the dehydration clock is 7 steps and starvation 14,
            # so the walk home starts while rations still cover it. Flat
            # floors: a distance-scaled budget (validation v5) bounced
            # pioneers off virgin borders before they could homestead, and
            # with the junk-base gate the deep-out stragglers it targeted no
            # longer exist.
            home = self._move_toward_structure(
                world,
                {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
                "low provisions away from water support: returning to the colony",
            )
            if home:
                return home
        if self._recent_action_streak(ActionType.MOVE.value) >= 4 and reserve < 3.0:
            return ActionRequest(self.agent_id, ActionType.OBSERVE, message="pausing repeated travel to reassess water route")
        if reserve < 1.6 and not _has_water_source(self, current):
            distress_target = self._communication_target(observation)
            if distress_target:
                return ActionRequest(self.agent_id, ActionType.COMMUNICATE, {"agent_id": distress_target}, "water reserves low; requesting local support instead of unsafe travel")
            return ActionRequest(self.agent_id, ActionType.OBSERVE, message="water reserves low; no crew in range to signal")
        if (
            len(current.agents_present) > 18
            and self.health > 0.65
            and self.fatigue < 0.70
            and reserve >= DISPERSAL_MIN_WATER_RESERVE
            and self.inventory.food >= 2.5
        ):
            # Plain crowd dispersal (the v3 organic-expansion channel): the
            # homestead rule turns provisioned drifters into greenhouse
            # founders, and the support-follows-homes gate removes the junk
            # micro-bases they used to die at. Requiring pre-supported targets
            # or founder kits here strangled expansion to 3-4 cells (v6-v9).
            dispersal = self._move_to_less_crowded_safe_neighbor(world)
            if dispersal:
                return dispersal
        neighbors = world.neighbors(self.x, self.y, 1)
        if not neighbors:
            return ActionRequest(self.agent_id, ActionType.OBSERVE)

        candidates = [cell for cell in neighbors if not cell.explored] or neighbors
        preset = ROLE_PRESETS["colonist"]
        if self.survival_priority > 0.45:
            safe = [
                cell for cell in candidates
                if (traversal_risk_for_cell(cell) < preset["risk_limit"] or self.risk_tolerance > 0.62)
                and len(cell.agents_present) < 12
            ]
            if safe:
                candidates = safe

        target = max(candidates, key=lambda candidate: self._cell_score(candidate))
        should_move = random.random() < max(0.18, self.curiosity * 0.65)
        if should_move:
            return ActionRequest(self.agent_id, ActionType.MOVE, {"x": target.x, "y": target.y}, "moving toward best local opportunity")
        return ActionRequest(self.agent_id, ActionType.OBSERVE, message="monitoring local situation")

    def _move_to_less_crowded_safe_neighbor(self, world) -> ActionRequest | None:
        current_count = len(world.get_cell(self.x, self.y).agents_present)
        candidates = [
            cell for cell in world.neighbors(self.x, self.y, 1)
            if len(cell.agents_present) <= max(4, current_count - 4)
            and edificabile(cell)  # su una montagna non si fonda nulla
            and traversal_risk_for_cell(cell) < 0.55
        ]
        if not candidates:
            return None
        target = max(candidates, key=lambda cell: (cell.habitability_score, -len(cell.agents_present), -traversal_risk_for_cell(cell)))
        return ActionRequest(self.agent_id, ActionType.MOVE, {"x": target.x, "y": target.y}, "dispersing from overcrowded cell to safer nearby ground")

    def _move_toward_nearby_ice(self, world) -> ActionRequest | None:
        visible = [
            cell for cell in world.neighbors(self.x, self.y, self.perception_radius)
            if cell.water_ice + cell.resources.ice > 0.35
            and traversal_risk_for_cell(cell) < 0.55
            and len(cell.agents_present) < 10
        ]
        if not visible:
            return None

        target = max(
            visible,
            key=lambda cell: (
                cell.water_ice + cell.resources.ice,
                cell.habitability_score,
                -len(cell.agents_present),
                -traversal_risk_for_cell(cell),
            ),
        )
        if target.x == self.x and target.y == self.y:
            return None

        current_distance = wrapped_manhattan_distance(
            target.x, target.y, self.x, self.y, world.width
        )
        candidates = [
            cell for cell in world.neighbors(self.x, self.y, 1)
            if wrapped_manhattan_distance(
                target.x, target.y, cell.x, cell.y, world.width
            ) < current_distance
            and len(cell.agents_present) < 10
            and traversal_risk_for_cell(cell) < 0.58
        ]
        if not candidates:
            return None
        next_cell = max(candidates, key=lambda cell: (cell.habitability_score, -len(cell.agents_present), -traversal_risk_for_cell(cell)))
        return ActionRequest(self.agent_id, ActionType.MOVE, {"x": next_cell.x, "y": next_cell.y}, "moving toward nearby buried ice instead of polar cap")

    def _move_toward_polar_ice(self, world) -> ActionRequest | None:
        north_distance = self.y
        south_distance = world.height - 1 - self.y
        target_y = 0 if north_distance <= south_distance else world.height - 1
        candidates = world.neighbors(self.x, self.y, 1)
        if not candidates:
            return None
        current_distance = abs(self.y - target_y)
        forward = [cell for cell in candidates if abs(cell.y - target_y) < current_distance]
        if not forward:
            forward = candidates
        viable = [
            cell for cell in forward
            if len(cell.agents_present) < 8
            and traversal_risk_for_cell(cell) < 0.55
        ]
        if not viable and self.hydration < 0.35:
            viable = [
                cell for cell in forward
                if len(cell.agents_present) < 8
                and traversal_risk_for_cell(cell) < 0.68
            ]
        if not viable:
            return None
        target = max(
            viable,
            key=lambda candidate: (
                candidate.water_ice + candidate.resources.ice,
                -abs(candidate.y - target_y),
                -traversal_risk_for_cell(candidate),
                candidate.habitability_score,
            ),
        )
        return ActionRequest(
            self.agent_id,
            ActionType.MOVE,
            {"x": target.x, "y": target.y},
            "moving cautiously toward peripheral ice; accepting extreme polar exposure risk",
        )

    def _move_toward_structure(self, world, structure_types: set[StructureType], message: str) -> ActionRequest | None:
        # All decisions in one simulation step see the same pre-action world.
        # Cache the expensive full-grid structure lookup by step and type set;
        # hundreds of agents returning to the same life-support network must
        # not rescan every cell independently.  The cache is replaced at the
        # next step, so structures completed during action execution become
        # visible to the following decision phase exactly as before.
        targets = self._structure_targets(world, structure_types)
        if not targets:
            return None
        # **Chebyshev e non Manhattan, perche' si cammina in diagonale.** Il
        # budget di razioni per rientrare e' calcolato con la distanza di
        # Chebyshev (`_support_distance_cells`), che e' la metrica di viaggio:
        # il movimento va agli otto vicini. Scegliendo il bersaglio con
        # Manhattan si sceglieva un rifugio diverso da quello su cui il budget
        # era stato calcolato, e si partiva con una razione in meno.
        target_cell = min(
            targets,
            key=lambda cell: wrapped_chebyshev_distance(
                cell.x, cell.y, self.x, self.y, world.width
            ),
        )
        if target_cell.x == self.x and target_cell.y == self.y:
            return None
        candidates = world.neighbors(self.x, self.y, 1)
        current_distance = wrapped_manhattan_distance(
            target_cell.x, target_cell.y, self.x, self.y, world.width
        )
        forward = [
            cell for cell in candidates
            if wrapped_manhattan_distance(
                target_cell.x, target_cell.y, cell.x, cell.y, world.width
            ) < current_distance
        ]
        if not forward:
            forward = candidates
        viable = [
            cell for cell in forward
            if len(cell.agents_present) < 10
            and traversal_risk_for_cell(cell) < 0.62
        ]
        if not viable:
            viable = forward
        next_cell = max(
            viable,
            key=lambda cell: (
                -wrapped_manhattan_distance(
                    target_cell.x, target_cell.y, cell.x, cell.y, world.width
                ),
                cell.habitability_score,
                -traversal_risk_for_cell(cell),
            ),
        )
        return ActionRequest(self.agent_id, ActionType.MOVE, {"x": next_cell.x, "y": next_cell.y}, message)

    def _structure_saturated(self, structure: StructureType, cell) -> bool:
        return structure_saturated(structure, cell)

    def _recently_rejected(self, action: ActionType, world=None, threshold: int = BUILD_REJECTION_THRESHOLD) -> bool:
        failures = getattr(self.memory, "recent_failures", [])
        current_step = int(getattr(world, "step", 0)) if world is not None else None
        count = 0
        for failure in failures:
            if failure.get("action") != action.value:
                continue
            step = failure.get("step")
            if current_step is not None and step is not None and step < current_step - BUILD_REJECTION_WINDOW_STEPS:
                continue
            count += 1
        return count >= threshold

    def _claim_cell_pool(self, world, cell, resource: str, grab: float = 1.0) -> bool:
        """Intra-step claim ledger for shared cell pools.

        All agents decide on the same world snapshot, so a thin shared stock
        (e.g. the storage-depot material trickle) attracts the whole crowd and
        only the first executor succeeds: everyone else burns the step on a
        rejected grab. Claims are transient decision-phase bookkeeping shared
        through the world object; the ledger self-resets when the step changes,
        so neither engine needs an explicit cleanup hook.
        """
        available = float(getattr(cell.resources, resource, 0.0))
        if available <= 0.1:
            return False
        if world is None:
            return True
        step = int(getattr(world, "step", 0))
        ledger = getattr(world, "_collect_claims", None)
        if not ledger or ledger[0] != step:
            ledger = (step, {})
            world._collect_claims = ledger
        key = (self.x, self.y, resource)
        claimed = ledger[1].get(key, 0.0)
        if available - claimed <= 0.1:
            return False
        ledger[1][key] = claimed + grab
        return True

    @property
    def _on_expedition(self) -> bool:
        return self.scout_phase is not None or self.settle_phase is not None

    def _settlement_action(
        self, world, current, resource_claims: dict | None = None
    ) -> ActionRequest | None:
        if self.settle_phase == "out":
            return self._continue_settle_out(world)
        return self._maybe_start_settlement(world, current, resource_claims)

    def _maybe_start_settlement(
        self, world, current, resource_claims: dict | None = None
    ) -> ActionRequest | None:
        actions_taken = int(getattr(self, "actions_taken", 0))
        if self.settle_next_at < 0:
            # Deterministic per-agent stagger, like scouting: the colony must
            # not empty out in one synchronized founding wave.
            self.settle_next_at = SETTLE_INTERVAL_STEPS + sum(ord(ch) for ch in self.agent_id) % 29
            return None
        if actions_taken < self.settle_next_at or self.scout_phase is not None:
            return None
        preparing_founder = bool(self.founder_kit_reserved)
        existing = {s.type for s in current.structures}
        # Only a mature cell spawns outposts: every core coverage ratio is
        # satisfied, so marginal work here has little value, and the crowd
        # justifies opening new ground.
        # **Maturita', non saturazione (2026-08-30).** La domanda qui e' «questa
        # cella copre chi ci vive?», non «posso costruirne un'altra?»: usare la
        # seconda, che dal 2026-08-30 porta il margine di crescita, alzerebbe la
        # soglia e spegnerebbe le spedizioni. Vedi `copertura_completa`.
        core_covered = (
            self._local_survival_grid_ready(existing)
            and copertura_completa(StructureType.GREENHOUSE, current)
            and copertura_completa(StructureType.OXYGEN_PLANT, current)
            and copertura_completa(StructureType.SHELTER, current)
        )
        # Pressure valve: at scale the oxygen ratio (1/10, mineral-bound)
        # saturates far later than the crowd grows — with 300 founders the
        # 960-step run launched ZERO expeditions because the mother cell never
        # covered O2, and expansion collapsed onto the two dispersal-adjacent
        # cells. Full greenhouse saturation cannot be the bar either: in a
        # growing cell it is a moving target the demand builder chases from
        # below (instrumented probe: average gap ~10 greenhouses on ~45
        # needed, so the valve never opened). Three quarters of coverage is
        # food-secure enough — the overflow crowd is exactly what should
        # leave and found its own greenhouse elsewhere.
        crowd = len(current.agents_present)
        greenhouses_needed = max(1, -(-crowd // COLONISTS_PER_STRUCTURE[StructureType.GREENHOUSE]))
        greenhouses_have = sum(1 for s in current.structures if s.type == StructureType.GREENHOUSE)
        pressure_valve = (
            crowd >= SETTLE_MIN_CROWD * 3
            and greenhouses_have >= 0.75 * greenhouses_needed
        )
        if (
            not preparing_founder
            and ((not core_covered and not pressure_valve) or crowd < SETTLE_MIN_CROWD)
        ):
            self.founder_kit_reserved = False
            self.settle_next_at = actions_taken + self._settle_interval()
            return None
        initial_candidate_unsafe = (
            self.curiosity < SETTLE_MIN_CURIOSITY
            or self.health < 0.80
            or self.fatigue > 0.50
            or self.oxygen_level < 0.85
        )
        committed_founder_unsafe = (
            self.health < 0.55
            or self.fatigue > 0.85
            or self.oxygen_level < 0.55
        )
        if (
            initial_candidate_unsafe
            if not preparing_founder
            else committed_founder_unsafe
        ):
            self.founder_kit_reserved = False
            self.settle_next_at = actions_taken + self._settle_interval()
            return None
        # From this point the agent is an eligible founder.  Preserve any
        # water, food and building materials it gathers across the short
        # preparation retries instead of donating the surplus back to the
        # warehouse before the following decision.
        self.founder_kit_reserved = True
        self._equip_founder_kit_from_warehouse(current, resource_claims)
        reserve = self.inventory.water + self.inventory.ice
        kit_targets = self._founder_kit_targets()
        founder_kit_ready = all(
            float(getattr(self.inventory, resource, 0.0)) + 1.0e-6 >= target
            for resource, target in kit_targets.items()
        )
        if not founder_kit_ready:
            # Kit-preparation mode with a SHORT retry: the full interval here
            # made the launch unreachable — refills add +1 water per turn and
            # the ordinary build loop spends collected materials on local
            # sites within a couple of steps, so by the next check (~60
            # actions later) the kit was gone again. Probe on the 300-founder
            # run: 139 of 139 material-kit failures returned None silently
            # and NO expedition ever launched in 960 steps.
            if reserve < SETTLE_MIN_WATER_RESERVE and any(float(s.local_effect.get("water", 0.0)) > 0.0 for s in current.structures):
                self.settle_next_at = actions_taken + 2
                return ActionRequest(
                    self.agent_id,
                    ActionType.REFILL_WATER,
                    target={"reserve_target": 8.0},
                    message="refilling canteens before settlement expedition",
                )
            if self.inventory.food < SETTLE_MIN_FOOD and _has_food_source(current):
                self.settle_next_at = actions_taken + 2
                return ActionRequest(
                    self.agent_id,
                    ActionType.FORAGE,
                    target={"food_target": SETTLE_MIN_FOOD},
                    message="stocking rations before settlement expedition",
                )
            if (
                self.inventory.construction_material
                < kit_targets["construction_material"]
                or self.inventory.minerals < kit_targets["minerals"]
            ):
                if self._claim_cell_pool(world, current, "construction_material"):
                    self.settle_next_at = actions_taken + 2
                    return ActionRequest(self.agent_id, ActionType.COLLECT_MATERIALS, message="gathering the founder kit before settlement expedition")
                if self._claim_cell_pool(world, current, "minerals"):
                    self.settle_next_at = actions_taken + 2
                    return ActionRequest(self.agent_id, ActionType.COLLECT_MINERALS, message="gathering the founder kit before settlement expedition")
            self.settle_next_at = actions_taken + self._settle_interval()
            self.founder_kit_reserved = False
            return None
        target = self._pick_settlement_target(world)
        if target is None:
            self.founder_kit_reserved = False
            self.settle_next_at = actions_taken + self._settle_interval()
            return None
        self.settle_phase = "out"
        self.settle_steps = 0
        self.settle_target = (target.x, target.y)
        return self._move_toward_cell(world, target.x, target.y, f"settlement expedition: founding an outpost at ({target.x},{target.y})")

    def _equip_founder_kit_from_warehouse(
        self, current, resource_claims: dict | None = None
    ) -> bool:
        """Atomically allocate an available founder kit from the cell store.

        Choosing the Explore pillar is the actual agent decision.  Moving
        warehouse stock into the chosen founder's knapsack is automatic cell
        logistics, just like ordinary redistribution: it must not consume one
        seven-day action for every bottle or ration.  The all-or-nothing check
        prevents partial reservations from draining a scarce warehouse.
        """
        targets = self._founder_kit_targets()
        missing = {
            resource: max(
                0.0,
                target - float(getattr(self.inventory, resource, 0.0)),
            )
            for resource, target in targets.items()
        }
        claims = resource_claims if resource_claims is not None else {}

        def unclaimed(resource: str) -> float:
            warehouse = float(getattr(current.resources, resource, 0.0))
            reserved = float(
                claims.get(
                    ("resource", int(current.y), int(current.x), resource),
                    0.0,
                )
            )
            return max(0.0, warehouse - reserved)

        # **La spedizione la equipaggia la comunita', non il solo magazzino
        # (2026-08-30).** Misurato: la colonia teneva 146 minerali nelle sacche
        # dei coloni e **zero** in magazzino, e il kit guardava solo il
        # magazzino. I fondatori partivano percio' con circa **2,03 minerali e
        # 2,08 di materiale** contro i **7,0 e 12,05** che la griglia di
        # avviamento richiede, arrivavano potendo costruire un solo pannello
        # solare, e l'avamposto restava senza serra ne' impianto d'ossigeno —
        # cioe' a capienza ZERO, quindi inabitabile per chiunque altro. Sei
        # avamposti fondati, sei avamposti vuoti.
        #
        # Le risorse **industriali** si possono chiedere ai compagni di cella:
        # i minerali di un colono non lo tengono in vita. Acqua e cibo no, e
        # restano al magazzino piu' alla borraccia del fondatore, che il ramo
        # di preparazione sa gia' riempire.
        compagni = self._compagni_di_cella(current, resource_claims)

        def manca(resource: str, amount: float) -> bool:
            """`unclaimed + somma dei compagni + 1e-6 < amount`, con uscita anticipata.

            La somma dei compagni si accumula da zero nello stesso ordine di
            prima; gli inventari non sono mai negativi, quindi la somma parziale
            non decresce e appena il confronto e' deciso ci si puo' fermare: il
            booleano e' identico. Misura (2026-09-24): a 2000 coloni questa somma
            su tutta la cella madre valeva ~13% del passo.
            """
            base = unclaimed(resource)
            if base + 1.0e-6 >= amount:
                return False
            if resource not in RISORSE_INDUSTRIALI:
                return True
            parziale = 0
            for c in compagni:
                parziale += float(getattr(c.inventory, resource, 0.0))
                if base + parziale + 1.0e-6 >= amount:
                    return False
            return True

        if any(manca(resource, amount) for resource, amount in missing.items()):
            return False
        for resource, amount in missing.items():
            if amount <= 0.0:
                continue
            residuo = amount
            warehouse = float(getattr(current.resources, resource, 0.0))
            preso = min(warehouse, residuo)
            setattr(current.resources, resource, warehouse - preso)
            residuo -= preso
            if residuo > 1.0e-9 and resource in RISORSE_INDUSTRIALI:
                for compagno in compagni:
                    if residuo <= 1.0e-9:
                        break
                    ha = float(getattr(compagno.inventory, resource, 0.0))
                    quota = min(ha, residuo)
                    if quota <= 0.0:
                        continue
                    setattr(compagno.inventory, resource, ha - quota)
                    residuo -= quota
            personal = float(getattr(self.inventory, resource, 0.0))
            setattr(self.inventory, resource, personal + (amount - residuo))
        return True

    def _compagni_di_cella(self, current, resource_claims: dict | None) -> list:
        """Gli altri residenti della cella, in ordine deterministico.

        Il registro arriva dal ledger del passo (`claims["__agenti__"]`), che e'
        gia' il canale con cui le prenotazioni viaggiano fino a qui. Quando non
        c'e' -- percorsi di test che chiamano il metodo da soli -- la lista e'
        vuota e il comportamento torna quello di prima, cioe' solo magazzino.
        """
        registro = (resource_claims or {}).get("__agenti__") or {}
        mio = str(getattr(self, "agent_id", ""))
        compagni = []
        for identificativo in sorted(current.agents_present, key=str):
            if str(identificativo) == mio:
                continue
            compagno = registro.get(identificativo)
            if compagno is not None and getattr(compagno, "inventory", None) is not None:
                compagni.append(compagno)
        return compagni

    @staticmethod
    def _founder_kit_targets() -> dict[str, float]:
        """Travel reserves plus one local greenhouse/solar/O2 starter grid."""
        targets = {
            "water": SETTLE_MIN_WATER_RESERVE,
            "food": SETTLE_MIN_FOOD,
            "construction_material": SETTLE_BUILD_MATERIAL_BUFFER,
            "minerals": 0.0,
            "energy": 0.0,
        }
        # **L'acqua della griglia va riservata come tutto il resto (2026-08-30).**
        # Il ciclo iterava solo materiale, minerali ed energia, e ometteva
        # l'acqua: l'unica struttura di avviamento che la richiede e' la
        # **serra**, cioe' esattamente quella che da' supporto vitale. Il
        # fondatore arrivava con l'acqua da viaggio ormai bevuta, non poteva
        # costruire la serra, e l'avamposto restava a capienza ZERO — quindi
        # inabitabile per chiunque altro. Misurato: sei avamposti fondati, sei
        # avamposti vuoti, tutti con impianto d'ossigeno e ricovero ma senza
        # serra.
        #
        # Un'unita' d'acqua su un kit da sette non cambia la fattibilita' della
        # partenza; cambia se all'arrivo esiste una colonia o un pannello solare.
        for structure_type in SETTLE_BOOTSTRAP_STRUCTURES:
            cost = BUILD_COSTS[structure_type]
            for resource in ("construction_material", "minerals", "energy", "water"):
                targets[resource] += float(getattr(cost, resource, 0.0))
        return targets

    def _continue_settle_out(self, world) -> ActionRequest | None:
        self.settle_steps += 1
        reserve = self.inventory.water + self.inventory.ice
        at_target = (
            self.settle_target is not None
            and (self.x, self.y) == self.settle_target
        )
        # Travel limits protect a founder while it is still crossing exposed
        # terrain.  Once it reaches the nominated cell, closing the mission on
        # the same limits can orphan a partially completed bootstrap for the
        # rest of a long run: stochastic preferences legitimately interleave
        # the four construction projects over many more than 40 decisions.
        # Keep the commitment alive on-site and recover from the greenhouse;
        # only a true health emergency may release the founder before the grid
        # is operational.
        abort = (
            self.settle_target is None
            or self.health < 0.55
            or (
                not at_target
                and (
                    reserve < SETTLE_ABORT_WATER_RESERVE
                    or self.inventory.food < 1.0
                    or self.fatigue > 0.85
                    or self.settle_steps > SETTLE_MAX_OUT_STEPS
                )
            )
        )
        if abort:
            # Drop the mission and fall through: the ordinary water/food
            # security rules walk the agent back to life support on their own.
            self._end_settlement()
            return None
        tx, ty = self.settle_target
        if (self.x, self.y) == (tx, ty):
            cell = world.get_cell(self.x, self.y)
            if self._bootstrap_grid_ready(cell):
                self._end_settlement()
                return ActionRequest(
                    self.agent_id,
                    ActionType.OBSERVE,
                    message="surveying the operational settlement site",
                )
            if reserve < SETTLE_ABORT_WATER_RESERVE and _has_water_source(
                self, cell
            ):
                return ActionRequest(
                    self.agent_id,
                    ActionType.REFILL_WATER,
                    target={"reserve_target": SETTLE_MIN_WATER_RESERVE},
                    message="refilling locally while completing the settlement bootstrap",
                )
            if self.inventory.food < 1.0 and any(
                structure.type == StructureType.GREENHOUSE
                for structure in cell.structures
            ):
                return ActionRequest(
                    self.agent_id,
                    ActionType.FORAGE,
                    target={"food_target": SETTLE_MIN_FOOD},
                    message="harvesting locally while completing the settlement bootstrap",
                )
            if self.fatigue > 0.65 or self.health < 0.72:
                return ActionRequest(
                    self.agent_id,
                    ActionType.REST,
                    message="recovering while completing the settlement bootstrap",
                )
            build = self._bootstrap_build_action(cell)
            if build:
                build.message = "founding new settlement: completing the local life-support grid"
                return build
            return ActionRequest(
                self.agent_id,
                ActionType.OBSERVE,
                message="waiting for bootstrap inputs at the new settlement",
            )
        move = self._move_toward_cell(world, tx, ty, f"settlement expedition toward ({tx},{ty})")
        if move:
            return move
        self._end_settlement()
        return None

    @staticmethod
    def _bootstrap_grid_ready(cell) -> bool:
        existing = {structure.type for structure in cell.structures}
        return all(
            structure_type in existing
            for structure_type in SETTLE_OPERATIONAL_STRUCTURES
        )

    def _bootstrap_build_action(self, cell) -> ActionRequest | None:
        """Continue or open exactly the structures in the founder kit."""
        for site_key in sorted(cell.construction_sites):
            structure_name = str(site_key).split("@", 1)[0]
            for structure_type in SETTLE_OPERATIONAL_STRUCTURES:
                if structure_type.value == structure_name:
                    return ActionRequest(
                        self.agent_id,
                        STRUCTURE_ACTIONS[structure_type],
                        message=f"continuing outpost {structure_type.value}",
                    )
        existing = {structure.type for structure in cell.structures}
        for structure_type in SETTLE_OPERATIONAL_STRUCTURES:
            if structure_type in existing:
                continue
            if _combined_can_afford(
                self.inventory,
                cell.resources,
                BUILD_COSTS[structure_type],
            ):
                return ActionRequest(
                    self.agent_id,
                    STRUCTURE_ACTIONS[structure_type],
                    message=f"building outpost {structure_type.value}",
                )
        return None

    def _end_settlement(self) -> None:
        self.founder_kit_reserved = False
        self.settle_phase = None
        self.settle_target = None
        self.settle_steps = 0
        self.settle_next_at = int(getattr(self, "actions_taken", 0)) + self._settle_interval()

    def _settle_interval(self) -> int:
        return max(20, int(SETTLE_INTERVAL_STEPS * (1.6 - self.curiosity)))

    def _pick_settlement_target(self, world):
        candidates = [
            cell
            # **Una struttura sparsa non squalifica una cella (2026-08-30).**
            # Il filtro era `not cell.structures`: bastava che un colono di
            # passaggio vi posasse un pannello solare perche' la cella fosse
            # esclusa **per sempre** da ogni futura spedizione. E siccome un
            # pannello da solo non da' supporto vitale, quella cella non poteva
            # nemmeno diventare abitabile da se': misurato, sei o otto celle
            # ciascuna con uno o due pannelli, capienza ZERO, occupanti ZERO,
            # e nessuna spedizione poteva piu' sceglierle.
            #
            # Il criterio giusto non e' «vergine» ma «non e' gia' un
            # insediamento»: una cella che non sostiene ancora nessuno e' un
            # bersaglio legittimo — anzi migliore, perche' ha gia' qualcosa.
            # **Una regola sola per chi puo' stare in un cantiere
            # (2026-09-01).** Qui c'era `len(agents_present) < 6`, e in
            # `cell_accepts_arrival` c'era `popolazione == 0`: il selettore
            # sceglieva un bersaglio che la regola di arrivo poi rifiutava, e
            # il rifiuto costava l'intero passo al colono. Ora entrambe leggono
            # `posti_di_cantiere`, cioe' quanti dei quattro requisiti di
            # abitabilita' mancano ancora.
            #
            # E' caduta con esse `not cell.construction_sites`: un cantiere
            # aperto era motivo di ESCLUSIONE, quindi nessun rinforzo poteva
            # raggiungere l'avamposto proprio mentre ci si lavorava. E' il posto
            # dove i rinforzi servono di piu'.
            for cell in world.neighbors(self.x, self.y, SETTLE_RADIUS_CELLS)
            if local_life_support_capacity(cell) <= 0
            and len(cell.agents_present) < posti_di_cantiere(cell)
            and traversal_risk_for_cell(cell) < 0.55
            and float(getattr(cell, "polar_severity", 0.0)) < 0.5
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda cell: (
                (cell.water_ice + cell.resources.ice) * 0.05
                + cell.resources.minerals * 0.03
                + cell.resources.construction_material * 0.03
                + cell.habitability_score
                + (0.3 if cell.explored else 0.0)
                - wrapped_manhattan_distance(
                    cell.x, cell.y, self.x, self.y, world.width
                ) * 0.02
            ),
        )

    def _scouting_action(self, world, current) -> ActionRequest | None:
        if self.settle_phase is not None:
            return None
        if self.scout_phase == "out":
            return self._continue_scout_out(world)
        if self.scout_phase == "back":
            return self._continue_scout_back(world)
        return self._maybe_start_scout(world, current)

    def _maybe_start_scout(self, world, current) -> ActionRequest | None:
        actions_taken = int(getattr(self, "actions_taken", 0))
        if self.scout_next_at < 0:
            # Deterministic per-agent stagger so the colony does not empty out
            # in a single synchronized expedition wave.
            self.scout_next_at = self._scout_interval() + sum(ord(ch) for ch in self.agent_id) % 13
            return None
        if actions_taken < self.scout_next_at:
            return None
        if (
            self.health < 0.75
            or self.oxygen_level < 0.75
            or self.fatigue > 0.50
            or self.hydration < 0.70
            or self.satiety < 0.60
        ):
            return None
        # Stock up at the colony before leaving: canteens toward the expedition
        # target, food from local production. Range scales with what is carried.
        reserve = self.inventory.water + self.inventory.ice
        if reserve < SCOUT_TARGET_WATER_RESERVE and any(float(s.local_effect.get("water", 0.0)) > 0.0 for s in current.structures):
            return ActionRequest(
                self.agent_id,
                ActionType.REFILL_WATER,
                target={"reserve_target": 8.0},
                message="refilling canteens before scouting expedition",
            )
        if self.inventory.food < SCOUT_TARGET_FOOD and _has_food_source(current):
            return ActionRequest(
                self.agent_id,
                ActionType.FORAGE,
                target={"food_target": SCOUT_TARGET_FOOD},
                message="stocking rations before scouting expedition",
            )
        if reserve < SCOUT_MIN_WATER_RESERVE or self.inventory.food < SCOUT_MIN_FOOD:
            return None
        plan = self._pick_scout_plan(world, current)
        if plan is None:
            self.scout_next_at = actions_taken + self._scout_interval()
            return None
        target, home = plan
        self.scout_phase = "out"
        self.scout_steps = 0
        self.scout_target = (target.x, target.y)
        self.scout_home = (home.x, home.y)
        relay = (home.x, home.y) != (self.x, self.y)
        message = (
            f"relay scouting toward ({target.x},{target.y}); "
            f"handoff to operational outpost ({home.x},{home.y})"
            if relay
            else f"scouting shared frontier toward ({target.x},{target.y})"
        )
        return self._move_toward_cell(world, target.x, target.y, message)

    def _continue_scout_out(self, world) -> ActionRequest | None:
        self.scout_steps += 1
        reserve = self.inventory.water + self.inventory.ice
        # Turn back while there are still enough rations for the walk home:
        # farther out means a larger reserve is required to keep going.
        rations_home = self._scout_rations_to_return(world)
        abort = (
            self.scout_target is None
            or reserve < max(2.0, rations_home)
            or int(getattr(self, "steps_without_water", 0)) >= 2
            or self.inventory.food < max(1.0, rations_home)
            or self.fatigue > 0.75
            or self.health < 0.60
            or self.scout_steps
            > SCOUT_MAX_OUT_STEPS * self._movement_steps_per_cell(world)
        )
        if abort:
            self.scout_phase = "back"
            return self._continue_scout_back(world)
        tx, ty = self.scout_target
        if (self.x, self.y) == (tx, ty):
            self.scout_phase = "back"
            return ActionRequest(self.agent_id, ActionType.OBSERVE, message=f"scouting survey of cell ({tx},{ty}) complete; heading home")
        move = self._move_toward_cell(world, tx, ty, f"scouting expedition toward ({tx},{ty})")
        if move:
            return move
        self.scout_phase = "back"
        return self._continue_scout_back(world)

    def _continue_scout_back(self, world) -> ActionRequest | None:
        home = self.scout_home
        if home is None or (self.x, self.y) == tuple(home):
            self._end_scout_mission()
            return None
        move = self._move_toward_cell(world, home[0], home[1], "returning to colony from scouting expedition")
        if move:
            return move
        self._end_scout_mission()
        return None

    def _end_scout_mission(self) -> None:
        self.scout_phase = None
        self.scout_target = None
        self.scout_home = None
        self.scout_steps = 0
        self.scout_next_at = int(getattr(self, "actions_taken", 0)) + self._scout_interval()

    def _scout_interval(self) -> int:
        return max(8, int(SCOUT_INTERVAL_STEPS * (1.6 - self.curiosity)))

    def _scout_rations_to_return(self, world) -> float:
        """Rations (water or food) needed to walk back home, one cell of margin."""
        home = self.scout_home
        if not home:
            return 0.0
        distance_cells = wrapped_chebyshev_distance(
            self.x, self.y, home[0], home[1], world.width
        )
        return (
            distance_cells + 1
        ) * self._movement_steps_per_cell(world) * SCOUT_RATION_PER_STEP

    def _pick_scout_plan(self, world, current=None):
        """Return ``(unexplored_target, supported_return_base)``.

        Exploration advances only along the shared known/unknown boundary.
        Once the current base has mapped every reachable frontier cell, a
        colonist may take a relay assignment whose return point is another
        populated, fully supported outpost. This moves a bounded number of
        scouts outward without manufacturing residents or scanning the full
        64,800-cell planet for every decision.
        """
        current = current or world.get_cell(self.x, self.y)
        reserve = self.inventory.water + self.inventory.ice
        ration_budget = min(reserve - 1.0, self.inventory.food - 1.0)
        ration_per_cell = (
            self._movement_steps_per_cell(world) * SCOUT_RATION_PER_STEP
        )
        if ration_budget <= 0.0 or ration_per_cell <= 0.0:
            return None
        # This is a budget in macro-cell legs, not in raw decisions. With a
        # sub-cell operational range, one leg consumes several weekly moves.
        # Do not force a minimum adjacent-cell expedition: if food/water cannot
        # cover an outbound and return leg at the selected metric range, the
        # physically correct result is to wait for a nearer relay base or more
        # provisions.
        total_route_budget = int(ration_budget / ration_per_cell)

        bases = self._scout_support_bases(world, current)
        for base in bases:
            transfer_distance = wrapped_manhattan_distance(
                self.x,
                self.y,
                base.x,
                base.y,
                world.width,
            )
            # Conservative route: current -> base -> target -> base. The
            # executor may take the shorter direct outbound path, but never
            # receives a larger ration allowance because of it.
            remaining_route = total_route_budget - transfer_distance
            max_range = min(SCOUT_RADIUS_CELLS, remaining_route // 2)
            if max_range < 1:
                continue
            candidates = self._scout_frontier_candidates(
                world,
                base,
                max_range,
            )
            if not candidates:
                continue

            def frontier_score(cell) -> tuple[float, float, int, int]:
                unknown_neighbors = sum(
                    1
                    for adjacent in world.neighbors(cell.x, cell.y, 1)
                    if not adjacent.explored
                )
                resource_score = (
                    (cell.water_ice + cell.resources.ice) * 0.04
                    + cell.resources.minerals * 0.03
                    + cell.resources.construction_material * 0.03
                    + cell.habitability_score
                )
                distance = wrapped_manhattan_distance(
                    cell.x,
                    cell.y,
                    base.x,
                    base.y,
                    world.width,
                )
                return (
                    float(unknown_neighbors),
                    float(resource_score) - distance * 0.01,
                    -cell.y,
                    -cell.x,
                )

            return max(candidates, key=frontier_score), base
        return None

    def _scout_support_bases(self, world, current) -> list:
        """Current base first, then nearest fully operational outposts."""
        bases = [current]
        seen = {(int(current.x), int(current.y))}
        positions = getattr(world, "_structure_positions", ()) or ()
        ordered_positions = sorted(
            ((int(x), int(y)) for x, y in positions),
            key=lambda position: (
                wrapped_manhattan_distance(
                    self.x,
                    self.y,
                    position[0],
                    position[1],
                    world.width,
                ),
                position[1],
                position[0],
            ),
        )
        for x, y in ordered_positions:
            if (x, y) in seen:
                continue
            cell = world.get_cell(x, y)
            population = len(cell.agents_present)
            if population <= 0 or local_life_support_capacity(cell) < population:
                continue
            bases.append(cell)
            seen.add((x, y))
        return bases

    def _scout_frontier_candidates(self, world, base, radius: int) -> list:
        """Cached unexplored cells touching the shared explored map."""
        cache_step = int(getattr(world, "step", 0))
        cache_state = getattr(world, "_scout_frontier_cache", None)
        if not cache_state or cache_state[0] != cache_step:
            cache_state = (cache_step, {})
            world._scout_frontier_cache = cache_state
        cache_key = (int(base.x), int(base.y), int(radius))
        cached = cache_state[1].get(cache_key)
        if cached is not None:
            return cached
        candidates = [
            cell
            for cell in world.neighbors(base.x, base.y, radius)
            if not cell.explored
            and traversal_risk_for_cell(cell) < 0.60
            and float(getattr(cell, "polar_severity", 0.0)) < 0.5
            and (
                # An occupied base is implicitly known even in unit worlds
                # that have not run the shared-observation initializer yet.
                wrapped_chebyshev_distance(
                    cell.x,
                    cell.y,
                    base.x,
                    base.y,
                    world.width,
                )
                <= 1
                or any(
                    adjacent.explored
                    for adjacent in world.neighbors(cell.x, cell.y, 1)
                )
            )
        ]
        cache_state[1][cache_key] = candidates
        return candidates

    def _pick_scout_target(self, world):
        """Compatibility wrapper returning only the target cell."""
        plan = self._pick_scout_plan(world, world.get_cell(self.x, self.y))
        return None if plan is None else plan[0]

    def _move_toward_cell(self, world, target_x: int, target_y: int, message: str) -> ActionRequest | None:
        if (self.x, self.y) == (target_x, target_y):
            return None
        candidates = world.neighbors(self.x, self.y, 1)
        if not candidates:
            return None
        current_distance = wrapped_manhattan_distance(
            target_x, target_y, self.x, self.y, world.width
        )
        forward = [
            cell for cell in candidates
            if wrapped_manhattan_distance(
                target_x, target_y, cell.x, cell.y, world.width
            ) < current_distance
        ] or candidates
        # **Il passo si sceglie fra le celle che accetteranno davvero il
        # viaggiatore (2026-09-01).** Il filtro era `len(agents_present) < 10`,
        # cioe' una terza regola per una domanda che ne aveva gia' una:
        # `cell_accepts_arrival`. La rotta puntava percio' su celle che alla
        # porta venivano respinte, e il rifiuto non e' un ritardo -- il ciclo
        # delle preferenze non ha ripiego, quindi il colono restava fermo a fare
        # NULLA per quel passo. Misurato: 30,1% delle decisioni finiva a
        # `do_nothing`, e il 92,6% dei rifiuti erano ricognizioni respinte da
        # cantieri gia' al completo.
        #
        # La cascata e' avanti, poi la deviazione, poi la rinuncia: se nessuna
        # cella intorno accetta il viaggiatore la rotta non esiste, e i
        # chiamanti sanno gia' cosa fare (la ricognizione torna a casa, la
        # spedizione rinuncia) invece di proporre un passo impossibile.
        origine = world.get_cell(self.x, self.y)
        consentito, transito = permessi_di_viaggio(self)

        def percorribile(cell) -> bool:
            return traversal_risk_for_cell(cell) < 0.62 and cell_accepts_arrival(
                origine, cell, allow_pioneer=consentito, allow_transit=transito
            )

        viable = [cell for cell in forward if percorribile(cell)]
        if not viable:
            viable = [cell for cell in candidates if percorribile(cell)]
        if not viable:
            return None
        next_cell = max(
            viable,
            key=lambda cell: (
                -wrapped_manhattan_distance(
                    target_x, target_y, cell.x, cell.y, world.width
                ),
                cell.habitability_score,
                -traversal_risk_for_cell(cell),
            ),
        )
        return ActionRequest(self.agent_id, ActionType.EXPLORE, {"x": next_cell.x, "y": next_cell.y}, message)

    def _recent_action_streak(self, action: str) -> int:
        count = 0
        for item in reversed(self.recent_actions):
            if item != action:
                break
            count += 1
        return count

    def _build_blocked(self, structure: StructureType, cell) -> bool:
        return self._local_structure_exists(structure, cell) or self._local_construction_site_key(structure) in cell.construction_sites

    def _local_structure_exists(self, structure: StructureType, cell) -> bool:
        return any(
            s.type == structure and self._local_distance_to_structure(s) <= 250.0
            for s in cell.structures
        )

    def _local_distance_to_structure(self, structure) -> float:
        dx = float(getattr(structure, "local_x_m", 0.0)) - float(self.local_x_m)
        dy = float(getattr(structure, "local_y_m", 0.0)) - float(self.local_y_m)
        return (dx * dx + dy * dy) ** 0.5

    def _local_construction_site_key(self, structure: StructureType) -> str:
        return f"{structure.value}@{round(float(self.local_x_m))}:{round(float(self.local_y_m))}"

    def _cell_score(self, cell) -> float:
        risk = traversal_risk_for_cell(cell)
        food_value = cell.vegetation_biomass * 0.025
        water_value = (cell.water_ice + cell.resources.ice) * 0.024
        mineral_value = cell.resources.minerals * 0.020
        material_value = cell.resources.construction_material * 0.022
        structure_value = len(cell.structures) * 0.035
        polar_penalty = float(getattr(cell, "polar_severity", 0.0)) * (0.55 + self.survival_priority * 0.45)
        pollution_penalty = getattr(cell, "pollution_risk", 0.0) * 1.25
        
        # Avoid overcrowded cells (agents > 10)
        density_penalty = max(0.0, len(cell.agents_present) - 5) * 0.10
        if len(cell.agents_present) >= 12:
            density_penalty += 0.6
            
        safety_penalty = risk * (2.5 - self.risk_tolerance * 1.45 + self.survival_priority * 0.55)
        return cell.habitability_score + food_value + water_value + mineral_value + material_value + structure_value - safety_penalty - density_penalty - polar_penalty - pollution_penalty


def _has_food_source(cell) -> bool:
    return any(s.type == StructureType.GREENHOUSE for s in cell.structures) or (
        cell.vegetation_biomass > 0.25 and cell.proto_soil_development >= 0.35
    )


def _has_water_source(agent: RuleBasedAgent, cell) -> bool:
    return (
        agent.inventory.water >= 0.1
        or agent.inventory.ice >= 0.1
        or cell.liquid_water >= 0.1
        or cell.water_ice + cell.resources.ice >= 0.1
        or any(float(s.local_effect.get("water", 0.0)) > 0.0 for s in cell.structures)
    )
