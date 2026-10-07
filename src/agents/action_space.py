from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.world.grid import GridWorld
from src.world.resources import ResourceBundle
from src.agents.vitals import (
    RAZIONE_ACQUA,
    RAZIONE_CIBO,
    RAZIONE_OSSIGENO,
    RISTORO_ACQUA,
    RISTORO_CIBO,
)
from src.simulation.step_effects import (
    CAPACITA_MANUTENZIONE,
    COSTO_MANUTENZIONE_PER_INTEGRITA,
    RIPARAZIONE_MANUTENZIONE,
)
from src.world.structures import BUILD_COSTS, Structure, StructureType
from src.world.terrain import traversal_risk_for_cell, edificabile

from .build_policy import structure_saturated

RESOURCE_EPSILON = 1.0e-6
BUILD_SITE_RADIUS_M = 250.0
BUILD_SITE_CLEARANCE_M = 750.0
WATER_RESERVE_CAP = 8.0
WATER_REFILL_PER_ACTION = 1.0
# ``COLLECT_ICE`` is weekly field work, not construction.  The previous
# five-call state machine stored ``ice_extraction`` inside construction_sites:
# interrupted collectors consequently left permanent, positionless "building
# sites" on empty cells.  A 0.4-unit yield preserves the old long-run rate
# (2 units every five actions) while every accepted action now has a complete,
# mass-conserving result.
ICE_COLLECTION_YIELD_PER_ACTION = 0.4


def _movement_target_local(agent_id: str, step: int, x: int, y: int, geometry: dict) -> tuple[float, float]:
    """Return a reproducible, continuous destination inside the target cell.

    MOVE/EXPLORE historically aimed at the macro-cell centre, which made
    independent agents collapse onto visible lines and central clusters.  A
    stable digest gives every accepted movement a real sub-cell destination
    without relying on process-randomized ``hash()`` or global RNG state.
    Including the step lets an agent revisit the same cell at a different
    physical point while seeded runs remain exactly reproducible.
    """
    digest = hashlib.blake2b(
        f"{agent_id}|{int(step)}|{int(x)}|{int(y)}".encode("utf-8"),
        digest_size=16,
        person=b"mars-move-v3",
    ).digest()
    ux = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    uy = int.from_bytes(digest[8:], "big") / float(2**64 - 1)
    return (
        ux * float(geometry.get("width_m", 1.0)),
        uy * float(geometry.get("height_m", 1.0)),
    )


def arretrato_manutenzione(cell) -> float:
    """L'arretrato INTERO della cella: quanta integrita' manca al parco.

    E' la grandezza che `compute_needs` pubblica come bisogno e da cui discende
    quante persone la cella puo' mandare a manutenere in un passo. Non e' cio'
    che una singola azione ripara: per quello vedi `integrita_da_ripristinare`.
    """
    return sum(
        min(RIPARAZIONE_MANUTENZIONE, max(0.0, 1.0 - s.integrity))
        for s in cell.structures
    )


def integrita_da_ripristinare(cell) -> float:
    """Quanta integrita' UNA manutenzione ripristina, cioe' il lavoro di un passo.

    Fonte unica per il preventivo di `validate_action` e per il pagamento di
    `execute_action`: il costo fisso di 1,0 era scritto in DUE posti, e
    commisurarlo in uno solo lasciava l'altro a respingere azioni che il
    secondo avrebbe accettato.

    **Limitata a `CAPACITA_MANUTENZIONE` dal 2026-08-27.** Senza il tetto una
    manutenzione era il rifacimento dell'intero parco pagato da una persona
    sola, e in una cella matura nessuno poteva permetterselo. Sotto la capienza
    nulla cambia: una passata ripara tutto, come prima.
    """
    return min(CAPACITA_MANUTENZIONE, arretrato_manutenzione(cell))


def costo_manutenzione(cell) -> ResourceBundle:
    """Il materiale che la manutenzione di questa cella richiede."""
    return ResourceBundle(
        construction_material=integrita_da_ripristinare(cell)
        * COSTO_MANUTENZIONE_PER_INTEGRITA
    )


def _combined_can_afford(agent_inventory, cell_resources, cost: ResourceBundle) -> bool:
    """Return whether agent knapsack plus current-cell warehouse cover ``cost``."""
    return all(
        float(getattr(agent_inventory, field, 0.0))
        + float(getattr(cell_resources, field, 0.0))
        + RESOURCE_EPSILON
        >= float(getattr(cost, field, 0.0))
        for field in cost.__dataclass_fields__
    )


def _remove_combined(agent_inventory, cell_resources, cost: ResourceBundle) -> bool:
    """Pay a work cost from the agent first, then from the cell warehouse.

    The operation is deterministic and mass-conserving.  The preference still
    chooses the worker; the warehouse only removes the artificial requirement
    that all project inputs must fit in that worker's personal knapsack.
    """
    if not _combined_can_afford(agent_inventory, cell_resources, cost):
        return False
    for field in cost.__dataclass_fields__:
        remaining = float(getattr(cost, field, 0.0))
        if remaining <= 0.0:
            continue
        personal = float(getattr(agent_inventory, field, 0.0))
        from_personal = min(personal, remaining)
        setattr(agent_inventory, field, personal - from_personal)
        remaining -= from_personal
        if remaining > RESOURCE_EPSILON:
            warehouse = float(getattr(cell_resources, field, 0.0))
            setattr(cell_resources, field, max(0.0, warehouse - remaining))
    return True


class ActionType(str, Enum):
    OBSERVE = "observe"
    MOVE = "move"
    EXPLORE = "explore"
    COLLECT_ICE = "collect_ice"
    COLLECT_MINERALS = "collect_minerals"
    COLLECT_MATERIALS = "collect_materials"
    FORAGE = "forage"
    BUILD_SHELTER = "build_shelter"
    BUILD_SOLAR_ARRAY = "build_solar_array"
    BUILD_OXYGEN_PLANT = "build_oxygen_plant"
    BUILD_GREENHOUSE = "build_greenhouse"
    BUILD_WATER_EXTRACTOR = "build_water_extractor"
    BUILD_HEATER = "build_heater"
    BUILD_INFIRMARY = "build_infirmary"
    BUILD_HABITAT = "build_habitat"
    BUILD_RESEARCH_LAB = "build_research_lab"
    BUILD_STORAGE_DEPOT = "build_storage_depot"
    BUILD_WEATHER_STATION = "build_weather_station"
    COMMUNICATE = "communicate"
    SHARE_RESOURCE = "share_resource"
    PHYSIOLOGICAL_RECOVERY = "physiological_recovery"
    REST = "rest"
    EAT_FOOD = "eat_food"
    DRINK_WATER = "drink_water"
    REFILL_WATER = "refill_water"
    USE_MED_KIT = "use_med_kit"
    MAINTAIN_STRUCTURE = "maintain_structure"
    DO_NOTHING = "do_nothing"


BUILD_ACTIONS = {
    ActionType.BUILD_SHELTER: StructureType.SHELTER,
    ActionType.BUILD_SOLAR_ARRAY: StructureType.SOLAR_ARRAY,
    ActionType.BUILD_OXYGEN_PLANT: StructureType.OXYGEN_PLANT,
    ActionType.BUILD_GREENHOUSE: StructureType.GREENHOUSE,
    ActionType.BUILD_WATER_EXTRACTOR: StructureType.WATER_EXTRACTOR,
    ActionType.BUILD_HEATER: StructureType.HEATER,
    ActionType.BUILD_INFIRMARY: StructureType.INFIRMARY,
    ActionType.BUILD_HABITAT: StructureType.HABITAT,
    ActionType.BUILD_RESEARCH_LAB: StructureType.RESEARCH_LAB,
    ActionType.BUILD_STORAGE_DEPOT: StructureType.STORAGE_DEPOT,
    ActionType.BUILD_WEATHER_STATION: StructureType.WEATHER_STATION,
}


@dataclass
class ActionRequest:
    agent_id: str
    action: ActionType
    target: Any = None  # coordinates {x, y} or target_agent_id
    message: str | None = None
    magnitude: float = 1.0


@dataclass
class ActionResult:
    accepted: bool
    action: str
    message: str
    data: dict = field(default_factory=dict)


def valid_actions_for(agent, world: GridWorld) -> list[str]:
    actions = [a.value for a in ActionType]

    individual_survival = bool(
        world.metadata.get("individual_survival_priority_enabled", True)
    )
    if individual_survival:
        actions.remove(ActionType.PHYSIOLOGICAL_RECOVERY.value)
    
    # If critically ill, restrict actions strictly to survival
    if individual_survival and getattr(agent, "health", 1.0) < 0.2:
        allowed = {ActionType.OBSERVE.value, ActionType.REST.value, ActionType.EAT_FOOD.value, ActionType.DRINK_WATER.value, ActionType.REFILL_WATER.value, ActionType.USE_MED_KIT.value, ActionType.COMMUNICATE.value, ActionType.DO_NOTHING.value}
        actions = [a for a in actions if a in allowed]
    
    cell = world.get_cell(agent.x, agent.y)
    if ActionType.COLLECT_ICE.value in actions and cell.water_ice <= 0 and cell.resources.ice <= 0:
        actions.remove(ActionType.COLLECT_ICE.value)
    if ActionType.COLLECT_MINERALS.value in actions and cell.resources.minerals <= 0:
        actions.remove(ActionType.COLLECT_MINERALS.value)
    if ActionType.COLLECT_MATERIALS.value in actions and cell.resources.construction_material <= 0:
        actions.remove(ActionType.COLLECT_MATERIALS.value)
    if ActionType.FORAGE.value in actions and not _has_food_source(cell):
        actions.remove(ActionType.FORAGE.value)
    if ActionType.USE_MED_KIT.value in actions and getattr(agent.inventory, "med_kits", 0) < 1:
        actions.remove(ActionType.USE_MED_KIT.value)
    if ActionType.EAT_FOOD.value in actions and getattr(agent.inventory, "food", 0) + RESOURCE_EPSILON < 0.1:
        actions.remove(ActionType.EAT_FOOD.value)
    if ActionType.DRINK_WATER.value in actions and not _has_drink_source(agent, cell):
        actions.remove(ActionType.DRINK_WATER.value)
    if ActionType.REFILL_WATER.value in actions and (
        not _has_refill_source(cell)
        or _water_reserve(agent) + RESOURCE_EPSILON >= WATER_RESERVE_CAP
    ):
        actions.remove(ActionType.REFILL_WATER.value)
    
    # Check build affordability only if NOT already in progress
    _si_puo_costruire = edificabile(cell)
    for action, structure_type in BUILD_ACTIONS.items():
        if action.value in actions:
            # La geografia vieta il cantiere dove il terreno non lo sostiene.
            if not _si_puo_costruire:
                actions.remove(action.value)
                continue
            if has_structure_at_local_position(cell, structure_type, agent.local_x_m, agent.local_y_m):
                actions.remove(action.value)
                continue
            in_progress = _construction_site_key_for_agent(cell, structure_type, agent) in cell.construction_sites
            if not in_progress and structure_saturated(structure_type, cell):
                actions.remove(action.value)
                continue
            if not in_progress and not _combined_can_afford(
                agent.inventory, cell.resources, BUILD_COSTS[structure_type]
            ):
                actions.remove(action.value)
    return actions


def validate_action(agent, world: GridWorld, request: ActionRequest) -> ActionResult:
    target = request.target if isinstance(request.target, dict) else {}
    cell = world.get_cell(agent.x, agent.y)

    if (
        request.action == ActionType.PHYSIOLOGICAL_RECOVERY
        and bool(world.metadata.get("individual_survival_priority_enabled", True))
    ):
        return ActionResult(
            False,
            request.action.value,
            "physiological recovery is available only in cell-only priority mode",
        )

    if request.action in {ActionType.MOVE, ActionType.EXPLORE}:
        tx = int(target.get("x", agent.x))
        ty = int(target.get("y", agent.y))
        if not world.in_bounds(tx, ty):
            return ActionResult(False, request.action.value, "target cell is outside world")
        raw_dx = abs(tx - agent.x)
        wrapped_dx = min(raw_dx, world.width - raw_dx)
        if wrapped_dx > 1 or abs(ty - agent.y) > 1:
            return ActionResult(False, request.action.value, "move target is too far")
        # Migration admission belongs to the preference decision binder.  The
        # executor remains a policy-neutral action primitive so explicit
        # movement, survival routes and alternate decision engines retain the
        # same public MOVE/EXPLORE semantics.
    
    if request.action in BUILD_ACTIONS:
        structure_type = BUILD_ACTIONS[request.action]
        in_progress = _construction_site_key_for_agent(cell, structure_type, agent) in cell.construction_sites
        proposal_population = target.get("coverage_population")
        if has_structure_at_local_position(cell, structure_type, agent.local_x_m, agent.local_y_m):
            return ActionResult(False, request.action.value, f"{structure_type.value} already exists at this local position")
        # Opening yet another site of a saturated type is refused here (not only
        # in the policy): with cell-wide shared sites, a crowd working the same
        # site would otherwise chain-complete copies within a single step.
        if not in_progress and structure_saturated(
            structure_type, cell, proposal_population
        ):
            return ActionResult(False, request.action.value, f"{structure_type.value} coverage already sufficient in this cell")
        if not in_progress and not _combined_can_afford(
            agent.inventory, cell.resources, BUILD_COSTS[structure_type]
        ):
            return ActionResult(False, request.action.value, f"insufficient resources for {structure_type.value}")

    if request.action == ActionType.MAINTAIN_STRUCTURE:
        if not cell.structures:
            return ActionResult(False, request.action.value, "no structures to maintain in this cell")
        if integrita_da_ripristinare(cell) <= RESOURCE_EPSILON:
            return ActionResult(
                False, request.action.value, "no structure needs maintenance here"
            )
        if not _combined_can_afford(
            agent.inventory, cell.resources, costo_manutenzione(cell)
        ):
            return ActionResult(False, request.action.value, "insufficient materials for maintenance")

    if request.action == ActionType.REFILL_WATER:
        if _water_reserve(agent) + RESOURCE_EPSILON >= WATER_RESERVE_CAP:
            return ActionResult(False, request.action.value, "water reserve already full")
        if not _has_refill_source(cell):
            return ActionResult(False, request.action.value, "no local source for refilling")
    
    if request.action == ActionType.SHARE_RESOURCE and not target.get("agent_id"):
        return ActionResult(False, request.action.value, "target agent_id required for sharing")
    if request.action in {ActionType.COMMUNICATE, ActionType.SHARE_RESOURCE}:
        target_id = target.get("agent_id") or target.get("target_agent_id")
        if target_id and target_id not in _nearby_agent_ids(agent, world):
            return ActionResult(False, request.action.value, f"target agent {target_id} is too far")
            
    return ActionResult(True, request.action.value, "validated")


def _nearby_agent_ids(agent, world: GridWorld) -> set[str]:
    radius = 1
    nearby = set()
    for yy in range(max(0, agent.y - radius), min(world.height, agent.y + radius + 1)):
        for raw_x in range(agent.x - radius, agent.x + radius + 1):
            xx = raw_x % world.width
            nearby.update(a for a in world.get_cell(xx, yy).agents_present if a != agent.agent_id)
    return nearby


def execute_action(agent, agents: dict, world: GridWorld, request: ActionRequest) -> ActionResult:
    validation = validate_action(agent, world, request)
    if not validation.accepted:
        world.log_event("rejected_action", validation.message, agent_id=agent.agent_id, action=request.action.value)
        agent.memory.add_failure(request.action.value, request.target, step=int(getattr(world, "step", 0)))
        return validation
    cell = world.get_cell(agent.x, agent.y)
    target = request.target if isinstance(request.target, dict) else {}
    if request.action == ActionType.MOVE or request.action == ActionType.EXPLORE:
        tx = int(target.get("x", agent.x))
        ty = int(target.get("y", agent.y))
        
        origin_cell = world.get_cell(agent.x, agent.y)
        target_cell = world.get_cell(tx, ty)
        origin_geom = origin_cell.geometry
        target_geom = target_cell.geometry

        generated_local_x, generated_local_y = _movement_target_local(
            agent.agent_id,
            int(getattr(world, "step", 0)),
            tx,
            ty,
            target_geom,
        )
        target_local_x = min(
            float(target_geom.get("width_m", 1.0)),
            max(0.0, float(target.get("local_x_m", generated_local_x))),
        )
        target_local_y = min(
            float(target_geom.get("height_m", 1.0)),
            max(0.0, float(target.get("local_y_m", generated_local_y))),
        )
        # Persist the concrete choice in the request/replay artifact.  The
        # policy still chooses MOVE/EXPLORE and the macro-cell; the executor
        # resolves that proposal to a real metric point.
        target.setdefault("local_x_m", target_local_x)
        target.setdefault("local_y_m", target_local_y)
        
        origin_lat = math.radians(float(origin_geom.get("center_lat_deg", 0.0)))
        origin_lon = math.radians(float(origin_geom.get("center_lon_deg", 0.0)))
        cell_lat = math.radians(float(target_geom.get("center_lat_deg", 0.0)))
        cell_lon = math.radians(float(target_geom.get("center_lon_deg", 0.0)))
        
        lon_delta = (cell_lon - origin_lon + math.pi) % (2.0 * math.pi) - math.pi
        mean_lat = (origin_lat + cell_lat) * 0.5
        planet_radius_m = 3389.5 * 1000.0 # MARS_MEAN_RADIUS_KM
        
        cell_center_x = lon_delta * math.cos(mean_lat) * planet_radius_m
        cell_center_y = -(cell_lat - origin_lat) * planet_radius_m
        
        agent_x_from_origin = agent.local_x_m - float(origin_geom.get("width_m", 1.0)) * 0.5
        agent_y_from_origin = agent.local_y_m - float(origin_geom.get("height_m", 1.0)) * 0.5
        
        target_x_from_origin = (
            cell_center_x
            + target_local_x
            - float(target_geom.get("width_m", 1.0)) * 0.5
        )
        target_y_from_origin = (
            cell_center_y
            + target_local_y
            - float(target_geom.get("height_m", 1.0)) * 0.5
        )

        dx = target_x_from_origin - agent_x_from_origin
        dy = target_y_from_origin - agent_y_from_origin
        
        dist = math.hypot(dx, dy)
        # Kept as a dedicated runtime field for artifact/API compatibility;
        # scenario normalization synchronizes it with perception from the
        # single operational range selected by the user.
        max_dist = max(0.0, float(getattr(agent, "movement_distance_m_per_step", 59_000.0)))
        
        if dist > max_dist:
            dx = (dx / dist) * max_dist
            dy = (dy / dist) * max_dist
            actual_dist = max_dist
        else:
            actual_dist = dist
            
        new_x_from_origin = agent_x_from_origin + dx
        new_y_from_origin = agent_y_from_origin + dy
        
        half_width = float(origin_geom.get("width_m", 1.0)) * 0.5
        half_height = float(origin_geom.get("height_m", 1.0)) * 0.5
        
        new_local_x = new_x_from_origin + half_width
        new_local_y = new_y_from_origin + half_height
        
        new_cell_x = agent.x
        new_cell_y = agent.y
        
        # Resolve boundaries
        while True:
            current_geom = world.get_cell(new_cell_x, new_cell_y).geometry
            width_m = float(current_geom.get("width_m", 1.0))
            height_m = float(current_geom.get("height_m", 1.0))
            
            if new_local_x < 0:
                new_cell_x -= 1
                if new_cell_x < 0: new_cell_x = world.width - 1
                new_local_x += float(world.get_cell(new_cell_x, new_cell_y).geometry.get("width_m", 1.0))
                continue
            elif new_local_x > width_m:
                new_cell_x += 1
                if new_cell_x >= world.width: new_cell_x = 0
                new_local_x -= width_m
                continue
                
            if new_local_y < 0:
                if new_cell_y > 0:
                    new_cell_y -= 1
                    new_local_y += float(world.get_cell(new_cell_x, new_cell_y).geometry.get("height_m", 1.0))
                    continue
                else:
                    new_local_y = 0.0
            elif new_local_y > height_m:
                if new_cell_y < world.height - 1:
                    new_cell_y += 1
                    new_local_y -= height_m
                    continue
                else:
                    new_local_y = height_m
            break
            
        agent.local_x_m = new_local_x
        agent.local_y_m = new_local_y
        
        world.move_agent(agent.agent_id, agent.x, agent.y, new_cell_x, new_cell_y, agent.local_x_m, agent.local_y_m)
        agent.x, agent.y = new_cell_x, new_cell_y
        if request.action == ActionType.EXPLORE:
            world.mark_explored(new_cell_x, new_cell_y)
        
        target_cell = world.get_cell(new_cell_x, new_cell_y)
        agent.memory.remember_cell(new_cell_x, new_cell_y, target_cell.to_public_dict())

        # Overcrowding/pollution SYSTEM ALERTs are transient state, refreshed
        # every step by vitals against the CURRENT cell (see
        # vitals.refresh_cell_alerts) — appending them here flooded memory
        # with stale warnings from cells the agent had long left.

        distance_fraction = actual_dist / max(1.0, float(target_cell.geometry.get("width_m", 59000.0)))
        apply_environmental_entry(agent, target_cell, world, distance_fraction)
        result = ActionResult(
            True,
            request.action.value,
            "moved",
            {
                "x": new_cell_x,
                "y": new_cell_y,
                "local_x_m": float(new_local_x),
                "local_y_m": float(new_local_y),
                "distance_m": float(actual_dist),
                "target_local_x_m": float(target_local_x),
                "target_local_y_m": float(target_local_y),
            },
        )
    elif request.action == ActionType.COLLECT_ICE:
        # Fold any pre-fix progress into the first atomic extraction so a
        # loaded/long-lived world neither loses completed work nor keeps the
        # legacy pseudo-site forever.  25% represented one prior work action.
        legacy_progress = float(cell.construction_sites.get("ice_extraction", 0.0))
        if "ice_extraction" in cell.construction_sites:
            del cell.construction_sites["ice_extraction"]
        work_units = 1.0 + max(0.0, min(100.0, legacy_progress)) / 25.0
        before_ice = float(cell.water_ice) + float(cell.resources.ice)
        amount = min(ICE_COLLECTION_YIELD_PER_ACTION * work_units, before_ice)
        from_surface = min(float(cell.water_ice), amount)
        cell.water_ice = max(0.0, float(cell.water_ice) - from_surface)
        cell.resources.ice = max(
            0.0,
            float(cell.resources.ice) - (amount - from_surface),
        )
        world.adjust_total_ice_cache(
            (float(cell.water_ice) + float(cell.resources.ice)) - before_ice
        )
        agent.inventory.ice += amount
        agent.fatigue = min(1.0, agent.fatigue + 0.08)
        result = ActionResult(
            True,
            request.action.value,
            f"estratto {amount:.2f} ghiaccio",
            {
                "amount": amount,
                "legacy_progress_recovered": legacy_progress,
            },
        )
    elif request.action == ActionType.COLLECT_MINERALS:
        # **Via il premio di raffinazione (2026-08-25).** L'azione trasferiva
        # `amount` di minerali E aggiungeva `amount * 0,4` di materiale da
        # costruzione senza che nulla diminuisse: misurato, 499,5 unita'
        # fabbricate dal nulla in duecento passi, il 14% dell'intera giacenza
        # della colonia.
        #
        # **Perche' toglierlo e non farlo pagare al giacimento.** Ripartire la
        # massa estratta (1,0 di minerale + 0,4 di materiale da un prelievo di
        # 1,4) conserva anch'essa, ma toglie il 29% dei minerali alla colonia,
        # e i minerali sono la risorsa che stringe: una serra ne chiede 1 e i
        # coloni ne portano 0,73 a testa. Il mondo ha gia' il proprio
        # giacimento di materiale — `construction_material = minerali x 0,35`
        # nelle stesse celle, che `COLLECT_MATERIALS` raccoglie — e il premio
        # ne era un doppione gratuito. Tolto, la massa totale estraibile da una
        # cella resta 1,35 volte il suo giacimento minerale invece di 1,0, e la
        # ripartizione fra le due risorse torna quella che la geologia dichiara.
        amount = min(1.0, cell.resources.minerals)
        cell.resources.minerals -= amount
        agent.inventory.minerals += amount
        agent.fatigue = min(1.0, agent.fatigue + 0.12)
        result = ActionResult(True, request.action.value, f"collected {amount:.2f} minerals", {"amount": amount})
    elif request.action == ActionType.COLLECT_MATERIALS:
        amount = min(1.0, cell.resources.construction_material)
        if amount > 0:
            cell.resources.construction_material -= amount
            agent.inventory.construction_material += amount
            agent.fatigue = min(1.0, agent.fatigue + 0.15)
            result = ActionResult(True, request.action.value, f"scavenged {amount:.2f} construction materials", {"amount": amount})
        else:
            result = ActionResult(False, request.action.value, "no materials found in this cell")
    elif request.action == ActionType.EAT_FOOD:
        if agent.inventory.food + RESOURCE_EPSILON < 0.1:
            return ActionResult(False, request.action.value, "no food in inventory")
        agent.inventory.food = max(0.0, agent.inventory.food - 0.1)
        agent.satiety = min(1.0, agent.satiety + RISTORO_CIBO)
        agent.health = min(1.0, agent.health + 0.05)
        result = ActionResult(True, request.action.value, "ate food", {"satiety": agent.satiety})
    elif request.action == ActionType.DRINK_WATER:
        # Bere dall'impianto preleva dalla giacenza della cella come ogni
        # altra fonte: il ramo aggiungeva idratazione senza che nulla
        # diminuisse, gemello del difetto gia' chiuso in REFILL_WATER.
        if (
            _structure_water_support(cell) > 0
            and cell.resources.water + RESOURCE_EPSILON >= 0.1
        ):
            cell.resources.water = max(0.0, float(cell.resources.water) - 0.1)
            source = "life support structure"
        elif agent.inventory.water + RESOURCE_EPSILON >= 0.1:
            agent.inventory.water = max(0.0, agent.inventory.water - 0.1)
            source = "inventory water"
        elif agent.inventory.ice + RESOURCE_EPSILON >= 0.1:
            agent.inventory.ice = max(0.0, agent.inventory.ice - 0.1)
            source = "inventory ice"
        elif cell.liquid_water >= 0.1:
            cell.liquid_water -= 0.1
            source = "local liquid water"
        elif cell.water_ice + cell.resources.ice >= 0.1:
            before_ice = cell.water_ice + cell.resources.ice
            amount = min(0.1, cell.water_ice + cell.resources.ice)
            from_cell = min(cell.water_ice, amount)
            cell.water_ice -= from_cell
            cell.resources.ice = max(0.0, cell.resources.ice - (amount - from_cell))
            world.adjust_total_ice_cache((cell.water_ice + cell.resources.ice) - before_ice)
            source = "local ice"
        else:
            return ActionResult(False, request.action.value, "no drinkable water or ice source")
        agent.hydration = min(1.0, agent.hydration + RISTORO_ACQUA)
        agent.health = min(1.0, agent.health + 0.03)
        result = ActionResult(True, request.action.value, f"drank water from {source}", {"hydration": agent.hydration, "source": source})
    elif request.action == ActionType.REFILL_WATER:
        capacity = max(0.0, WATER_RESERVE_CAP - _water_reserve(agent))
        reserve_target = min(
            WATER_RESERVE_CAP,
            max(0.0, float(target.get("reserve_target", 0.0) or 0.0)),
        )
        requested = max(
            WATER_REFILL_PER_ACTION,
            reserve_target - _water_reserve(agent),
        )
        amount = min(requested, capacity)
        if _structure_water_support(cell) > 0 and cell.resources.water > RESOURCE_EPSILON:
            # **Si preleva, non si crea (2026-08-25).** Questo ramo aggiungeva
            # `amount` all'inventario senza sottrarlo da nulla: bastava un
            # impianto nella cella perche' l'acqua fosse infinita, e la massa
            # non si conservava. Restava invisibile finche' il supporto di
            # struttura era cosi' abbondante da rendere il rifornimento inutile;
            # normalizzato quel supporto, i coloni hanno cominciato a rifornirsi
            # davvero — misurato: da 30 azioni in duecento passi a 3.662 in
            # duecentosessanta, e l'acqua della colonia da 300 unita' a 199.774
            # in ottocento passi, in crescita illimitata.
            amount = min(amount, float(cell.resources.water))
            cell.resources.water = max(0.0, float(cell.resources.water) - amount)
            agent.inventory.water += amount
            source = "life support structure"
        elif cell.liquid_water > RESOURCE_EPSILON:
            amount = min(amount, float(cell.liquid_water))
            cell.liquid_water -= amount
            agent.inventory.water += amount
            source = "local liquid water"
        else:
            before_ice = cell.water_ice + cell.resources.ice
            amount = min(amount, before_ice)
            from_surface = min(cell.water_ice, amount)
            cell.water_ice -= from_surface
            cell.resources.ice = max(
                0.0, cell.resources.ice - (amount - from_surface)
            )
            world.adjust_total_ice_cache(
                (cell.water_ice + cell.resources.ice) - before_ice
            )
            agent.inventory.ice += amount
            source = "local ice"
        result = ActionResult(
            True,
            request.action.value,
            f"refilled water reserve from {source}",
            {"reserve": _water_reserve(agent), "amount": amount, "source": source},
        )
    elif request.action == ActionType.FORAGE:
        # Prioritize gathering from greenhouses first, then cell biomass
        amount = 0.0
        yield_multiplier = 1.0
        food_target = max(
            0.0, float(target.get("food_target", 0.0) or 0.0)
        )
        requested_food = max(
            1.0,
            food_target - float(getattr(agent.inventory, "food", 0.0)),
        )
        
        # TECH BONUS: Yield bonus from scientific progress
        tech_yield = float(world.planetary_state.get("yield_bonus", 0.0))
        yield_multiplier *= (1.0 + tech_yield)

        if any(s.type == StructureType.GREENHOUSE for s in cell.structures):
            # **Si raccoglie il raccolto, non se ne crea uno (2026-08-25).**
            # Il ramo produceva `requested_food` — cioe' esattamente quanto
            # l'agente avesse chiesto — senza attingere a nulla: una serra
            # rendeva il cibo gratuito e illimitato per chiunque le stesse
            # accanto (misurato: 154 unita' create in 105 raccolte). La serra
            # produce nella giacenza della cella a ogni passo
            # (`RESA_CIBO`); raccogliere e' prelevare da quella.
            amount = min(requested_food * yield_multiplier, float(cell.resources.food))
            cell.resources.food = max(0.0, float(cell.resources.food) - amount)
        elif _has_cultivated_biomass(cell):
            # MARTIAN STYLE CULTIVATION: open-field yield scales with soil development
            soil_mult = 0.4 + (cell.proto_soil_development * 1.6)  # 0.4x to 2.0x yield
            yield_multiplier *= soil_mult
            harvested = min(1.0, cell.vegetation_biomass)
            amount = harvested * yield_multiplier
            cell.vegetation_biomass = max(0.0, cell.vegetation_biomass - harvested * 0.1)

        if amount > 0:
            agent.inventory.food += amount
            agent.fatigue = min(1.0, agent.fatigue + 0.10)
            msg = f"foraged {amount:.2f} food"
            if cell.proto_soil_development > 0.5:
                msg += f" (harvested from rich soil: {cell.proto_soil_development*100:.0f}%)"
            result = ActionResult(True, request.action.value, msg, {"amount": amount})
        else:
            result = ActionResult(False, request.action.value, "no food found. develop soil to enable local cultivation.")
    elif request.action in BUILD_ACTIONS:
        from src.world.structures import BUILD_TIME
        structure_type = BUILD_ACTIONS[request.action]
        costs = BUILD_COSTS[structure_type]
        build_time = BUILD_TIME[structure_type]
        
        # TECH BONUS: Cost reduction from scientific progress
        tech_reduction = float(world.planetary_state.get("cost_reduction", 0.0))
        
        tool_bonus = min(0.40, getattr(agent.inventory, "tools", 0.0) * 0.08)
        if tool_bonus > 0.0:
            build_time = max(1, build_time - int(tool_bonus * 4.0))
            tech_reduction = max(tech_reduction, tool_bonus * 0.35)

        final_mult = 1.0 - tech_reduction
        
        if final_mult < 1.0:
            costs = ResourceBundle.from_dict({k: v * final_mult for k, v in costs.to_dict().items()})

        # Check if we are continuing or starting
        site_key = _construction_site_key_for_agent(cell, structure_type, agent)
        current_progress = cell.construction_sites.get(site_key, 0.0)
        
        if current_progress == 0:
            # Deduct resources ONLY when starting
            if _remove_combined(agent.inventory, cell.resources, costs):
                cell.construction_sites[site_key] = 0.0
                site_x_m, site_y_m = _construction_site_position(site_key, agent)
                world.log_event(
                    "construction_started",
                    f"started building {structure_type.value}",
                    agent_id=agent.agent_id,
                    x=agent.x,
                    y=agent.y,
                    local_x_m=site_x_m,
                    local_y_m=site_y_m,
                    site_key=site_key,
                )
            else:
                return ActionResult(False, request.action.value, f"insufficient resources for {structure_type.value}")
        
        # Advance construction
        increment = 100.0 / build_time
        new_progress = min(100.0, cell.construction_sites[site_key] + increment)
        cell.construction_sites[site_key] = new_progress
        
        # Fatigue and physical strain (construction is hard work!)
        agent.fatigue = min(1.0, agent.fatigue + 0.35)
        agent.health = max(0.0, agent.health - 0.05)
        
        if new_progress >= 100.0:
            # Finished buildings stand where the site was opened, not where the
            # last worker happens to be standing.
            site_x_m, site_y_m = _construction_site_position(site_key, agent)
            structure = Structure(structure_type, agent.x, agent.y, local_x_m=site_x_m, local_y_m=site_y_m, owner=agent.agent_id)
            world.add_structure(structure)
            del cell.construction_sites[site_key]
            
            result = ActionResult(True, request.action.value, f"finished building {structure_type.value}", {"structure": structure.to_dict()})
        else:
            result = ActionResult(True, request.action.value, f"construction of {structure_type.value} at {new_progress:.0f}%", {"progress": new_progress})
    elif request.action == ActionType.COMMUNICATE:
        target_id = target.get("agent_id") or target.get("target_agent_id")
        world.log_event("conversation", request.message or "status update", agent_id=agent.agent_id, target=target_id, x=agent.x, y=agent.y)
        result = ActionResult(True, request.action.value, "message sent", {"message": request.message, "target": target_id})
    elif request.action == ActionType.SHARE_RESOURCE:
        target_agent = agents.get(target.get("agent_id"))
        bundle = ResourceBundle.from_dict(target.get("resources", {}))
        if target_agent and agent.inventory.remove(bundle):
            target_agent.inventory.add(bundle)
            result = ActionResult(True, request.action.value, "resource shared", {"target": target_agent.agent_id})
            world.log_event("resource_shared", "resource shared", agent_id=agent.agent_id, target=target_agent.agent_id, resources=bundle.to_dict())
        else:
            result = ActionResult(False, request.action.value, "resource share failed")
    elif request.action == ActionType.MAINTAIN_STRUCTURE:
        # **Il costo segue cio' che si ripara (2026-08-25).** Prima era
        # un'unita' fissa di materiale qualunque fosse il parco riparato, e in
        # una cella con sessanta strutture comprava ventiquattro unita' di
        # integrita'. Misurato sulla vecchia taratura: 117 azioni in duecento
        # passi, 117 unita' di materiale, il 26% della spesa della colonia,
        # spese per riparare un'usura di 0,004 complessivi.
        strutture = list(cell.structures)
        # **Istantanea PRIMA del ciclo, e assegnazione invece di incremento.**
        # Sul motore ad array l'integrita' e' per (cella, tipo) e non per
        # istanza: tutte le facciate di uno stesso tipo scrivono nella stessa
        # casella, quindi `s.integrity += 0,4` ripetuto per N istanze
        # applicherebbe 0,4 x N. Con l'usura ferma a zero la differenza era
        # invisibile — entrambi i motori saturavano a 1,0 — e sarebbe diventata
        # una divergenza reale nel momento stesso in cui l'usura ha cominciato
        # a mordere.
        prima = [s.integrity for s in strutture]
        arretrato = arretrato_manutenzione(cell)
        ripristino = min(CAPACITA_MANUTENZIONE, arretrato)
        if ripristino <= RESOURCE_EPSILON:
            return ActionResult(
                False, request.action.value, "no structure needs maintenance here"
            )
        if not _remove_combined(agent.inventory, cell.resources, costo_manutenzione(cell)):
            return ActionResult(False, request.action.value, "insufficient materials for maintenance")

        # **La passata si spalma su tutto il parco (2026-08-27).** Quando
        # l'arretrato supera la capienza di un colono, la riparazione non
        # sceglie alcune strutture e ne abbandona altre: le solleva tutte in
        # proporzione. Sceglierne un sottoinsieme richiederebbe un ordine, e un
        # ordine sulle ISTANZE non esiste sul motore ad array, dove l'integrita'
        # e' per (cella, tipo). Con `fattore` = 1 — arretrato sotto la capienza
        # — l'aritmetica torna esattamente quella di prima.
        fattore = ripristino / arretrato if arretrato > 0.0 else 0.0
        for s, i in zip(strutture, prima):
            s.integrity = min(
                1.0, i + min(RIPARAZIONE_MANUTENZIONE, max(0.0, 1.0 - i)) * fattore
            )

        agent.fatigue = min(1.0, agent.fatigue + 0.15)
        result = ActionResult(
            True,
            request.action.value,
            f"maintained structures (+{ripristino:.2f} integrity)",
            {"integrity": [s.integrity for s in strutture], "restored": ripristino},
        )
    elif request.action == ActionType.PHYSIOLOGICAL_RECOVERY:
        # **Il ripristino si paga (2026-08-25).** L'azione portava salute,
        # sazieta', ossigeno e idratazione a 1,0 e azzerava i contatori di
        # privazione senza costo ne' massa: una guarigione completa dal nulla,
        # che nella modalita' in cui e' disponibile ("cell-only", cioe' con
        # `individual_survival_priority_enabled` spento) annullava per intero
        # l'economia delle razioni. Ora costa cibo, acqua e ossigeno come un
        # rifornimento vero, prelevati dal magazzino della cella, e senza
        # scorte non avviene.
        #
        # Nota: il toggle psicosociale governa soltanto l'azzeramento di stress
        # e morale piu' sotto; il ripristino dei vitali avveniva comunque.
        costo_ripristino = ResourceBundle(
            food=RAZIONE_CIBO, water=RAZIONE_ACQUA, oxygen=RAZIONE_OSSIGENO
        )
        if not _remove_combined(agent.inventory, cell.resources, costo_ripristino):
            return ActionResult(
                False,
                request.action.value,
                "no supplies for cell-coordinated physiological recovery",
            )
        agent.health = 1.0
        agent.satiety = 1.0
        agent.oxygen_level = 1.0
        agent.hydration = 1.0
        agent.fatigue = 0.0
        agent.steps_without_water = 0
        agent.steps_without_food = 0
        if bool(world.metadata.get("psychosocial_enabled", False)):
            agent.stress_index = 0.0
            agent.morale = 1.0
        result = ActionResult(
            True,
            request.action.value,
            "completed cell-coordinated physiological recovery",
            {
                "health": agent.health,
                "satiety": agent.satiety,
                "oxygen_level": agent.oxygen_level,
                "hydration": agent.hydration,
                "fatigue": agent.fatigue,
            },
        )
    elif request.action == ActionType.REST:
        bonus = 0.05 + cell.habitability_score * 0.07
        if any(s.type == StructureType.SHELTER for s in cell.structures):
            bonus += 0.08
            agent.oxygen_level = min(1.0, agent.oxygen_level + 0.08)
        if any(s.type == StructureType.HABITAT for s in cell.structures):
            bonus += 0.10
            agent.oxygen_level = min(1.0, agent.oxygen_level + 0.10)
            agent.hydration = min(1.0, agent.hydration + 0.04)
        if any(s.type == StructureType.INFIRMARY for s in cell.structures):
            bonus += 0.12
            agent.health = min(1.0, agent.health + 0.08)
            agent.oxygen_level = min(1.0, agent.oxygen_level + 0.06)
        agent.fatigue = max(0.0, agent.fatigue - bonus)
        agent.health = min(1.0, agent.health + 0.02)
        result = ActionResult(True, request.action.value, "rested", {"fatigue": agent.fatigue})
    elif request.action == ActionType.USE_MED_KIT:
        if agent.inventory.med_kits >= 1:
            agent.inventory.med_kits -= 1
            agent.health = min(1.0, agent.health + 0.3)
            result = ActionResult(True, request.action.value, "used med kit")
        else:
            result = ActionResult(False, request.action.value, "no med kits")
    elif request.action == ActionType.OBSERVE:
        result = ActionResult(True, request.action.value, "observed area")
    else:
        result = ActionResult(True, "do_nothing", "agent idle")
    
    agent.recent_actions.append(request.action.value)
    agent.actions_taken = int(getattr(agent, "actions_taken", 0)) + 1
    if len(agent.recent_actions) > 60:
        del agent.recent_actions[:-60]
    return result


def apply_environmental_entry(agent, cell, world: GridWorld, distance_fraction: float = 1.0) -> None:
    """Called after MOVE/EXPLORE into a cell: radiation, fractured ground, pits, lava-tube skylights, dust."""
    risk = traversal_risk_for_cell(cell)
    
    dmg = risk * (0.045 + (1.0 - agent.risk_tolerance) * 0.065) * distance_fraction
    
    agent.health = max(0.0, agent.health - dmg)
    
    agent.fatigue = max(0.0, min(1.0, agent.fatigue + risk * 0.095 * distance_fraction))
    if risk >= 0.55:
        agent.memory.add_event(f"Hazard exposure on {cell.terrain.value}: traversal_risk≈{risk:.2f}")
    if risk >= 0.75:
        world.log_event(
            "hazard_environment",
            f"{agent.name} encountered hazardous terrain: {cell.terrain.value}",
            agent_id=agent.agent_id,
            terrain=cell.terrain.value,
            risk=risk,
        )


def _structure_water_support(cell) -> float:
    return sum(float(s.local_effect.get("water", 0.0)) for s in cell.structures)


def _has_cultivated_biomass(cell) -> bool:
    return cell.vegetation_biomass > 0.25 and cell.proto_soil_development >= 0.35


def _has_food_source(cell) -> bool:
    return any(s.type == StructureType.GREENHOUSE for s in cell.structures) or _has_cultivated_biomass(cell)


def _has_drink_source(agent, cell) -> bool:
    return (
        getattr(agent.inventory, "water", 0.0) + RESOURCE_EPSILON >= 0.1
        or getattr(agent.inventory, "ice", 0.0) + RESOURCE_EPSILON >= 0.1
        or cell.liquid_water >= 0.1
        or cell.water_ice + cell.resources.ice >= 0.1
        or _structure_water_support(cell) > 0
    )


def _has_refill_source(cell) -> bool:
    return (
        cell.liquid_water > RESOURCE_EPSILON
        or cell.water_ice + cell.resources.ice > RESOURCE_EPSILON
        or _structure_water_support(cell) > 0
    )


def _water_reserve(agent) -> float:
    return float(getattr(agent.inventory, "water", 0.0)) + float(
        getattr(agent.inventory, "ice", 0.0)
    )


def _construction_site_key_for_agent(cell, structure_type: StructureType, agent) -> str:
    # Continuation is cell-wide: any open site of this type in the cell is the
    # shared work site, so helpers join it instead of being asked to afford a
    # brand-new build at their own local position.
    existing_key = _nearest_construction_site_key(
        cell,
        structure_type,
        float(agent.local_x_m),
        float(agent.local_y_m),
        radius_m=float("inf"),
    )
    if existing_key:
        return existing_key
    site_x_m, site_y_m = _nearest_free_structure_site(
        cell,
        float(agent.local_x_m),
        float(agent.local_y_m),
    )
    return _construction_site_key(structure_type, site_x_m, site_y_m)


def _construction_site_key(structure_type: StructureType, local_x_m: float, local_y_m: float) -> str:
    return f"{structure_type.value}@{round(local_x_m)}:{round(local_y_m)}"


def _construction_site_position(site_key: str, agent) -> tuple[float, float]:
    try:
        _, raw_position = site_key.split("@", 1)
        raw_x, raw_y = raw_position.split(":", 1)
        return float(raw_x), float(raw_y)
    except ValueError:
        return float(agent.local_x_m), float(agent.local_y_m)


def _construction_site_positions(cell) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    for key in cell.construction_sites:
        if "@" not in str(key):
            continue
        try:
            _, raw_position = str(key).split("@", 1)
            raw_x, raw_y = raw_position.split(":", 1)
            positions.append((float(raw_x), float(raw_y)))
        except ValueError:
            continue
    return positions


def _structure_site_is_clear(
    cell,
    local_x_m: float,
    local_y_m: float,
    radius_m: float = BUILD_SITE_CLEARANCE_M,
) -> bool:
    occupied = [
        (float(structure.local_x_m), float(structure.local_y_m))
        for structure in cell.structures
    ]
    occupied.extend(_construction_site_positions(cell))
    return all(
        math.hypot(site_x - local_x_m, site_y - local_y_m) > radius_m
        for site_x, site_y in occupied
    )


def _nearest_free_structure_site(
    cell,
    preferred_x_m: float,
    preferred_y_m: float,
    spacing_m: float = BUILD_SITE_CLEARANCE_M + 1.0,
) -> tuple[float, float]:
    """Allocate the nearest deterministic free site inside a macro-cell.

    Different building types used to reuse the worker's exact coordinates,
    causing multiple colored map markers to cover one another. Existing
    structures and every open construction site now reserve physical space;
    a square spiral selects the closest free alternative without RNG or a
    failed seven-day action.
    """
    geometry = getattr(cell, "geometry", {}) or {}
    width_m = max(1.0, float(geometry.get("width_m", 1.0)))
    height_m = max(1.0, float(geometry.get("height_m", 1.0)))
    preferred_x_m = min(width_m, max(0.0, preferred_x_m))
    preferred_y_m = min(height_m, max(0.0, preferred_y_m))
    if _structure_site_is_clear(cell, preferred_x_m, preferred_y_m):
        return preferred_x_m, preferred_y_m

    max_ring = int(math.ceil(max(width_m, height_m) / spacing_m)) + 1
    for ring in range(1, max_ring + 1):
        offsets = {
            (dx, dy)
            for dx in range(-ring, ring + 1)
            for dy in range(-ring, ring + 1)
            if max(abs(dx), abs(dy)) == ring
        }
        for dx, dy in sorted(
            offsets,
            key=lambda item: (
                item[0] * item[0] + item[1] * item[1],
                item[1],
                item[0],
            ),
        ):
            candidate_x = preferred_x_m + dx * spacing_m
            candidate_y = preferred_y_m + dy * spacing_m
            if not (0.0 <= candidate_x <= width_m and 0.0 <= candidate_y <= height_m):
                continue
            if _structure_site_is_clear(cell, candidate_x, candidate_y):
                return candidate_x, candidate_y
    # A 59 km cell cannot realistically exhaust 750 m sites at current
    # populations. Keeping the preferred point is a defensive last resort for
    # malformed zero-sized geometries; validation still remains deterministic.
    return preferred_x_m, preferred_y_m


def _nearest_construction_site_key(
    cell,
    structure_type: StructureType,
    local_x_m: float,
    local_y_m: float,
    radius_m: float = BUILD_SITE_RADIUS_M,
) -> str | None:
    prefix = f"{structure_type.value}@"
    best_key = None
    best_distance = radius_m
    for key in cell.construction_sites:
        if not key.startswith(prefix):
            continue
        _, raw_position = key.split("@", 1)
        try:
            raw_x, raw_y = raw_position.split(":", 1)
            site_x = float(raw_x)
            site_y = float(raw_y)
        except ValueError:
            continue
        dx = site_x - local_x_m
        dy = site_y - local_y_m
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= best_distance:
            best_key = key
            best_distance = distance
    return best_key


def has_structure_at_local_position(
    cell,
    structure_type: StructureType,
    local_x_m: float,
    local_y_m: float,
    radius_m: float = BUILD_SITE_RADIUS_M,
) -> bool:
    for structure in cell.structures:
        if structure.type != structure_type:
            continue
        if math.hypot(float(structure.local_x_m) - float(local_x_m), float(structure.local_y_m) - float(local_y_m)) <= radius_m:
            return True
    return False
