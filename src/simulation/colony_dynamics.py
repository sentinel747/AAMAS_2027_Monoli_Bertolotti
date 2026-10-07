from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np

from src.core import constants as C
from src.social_network.metrics import compute_social_metrics
from src.social_network.network import SocialNetwork
from src.world.grid import GridWorld
from src.world.colony_site import best_colony_site, colony_site_average, colony_site_score
from src.world.resources import ResourceBundle, indice_di_messa_in_comune, scorta_colonia
from src.simulation.step_effects import (
    CARICO_ELETTRICO,
    RESA_ENERGIA,
    RESA_ENERGIA_PER_CARICO,
    solar_yield,
)
from src.agents.vitals import RAZIONE_ACQUA, RAZIONE_CIBO
from src.world.structures import StructureType
from src.world.occupancy import (
    housing_slots_from_structures,
    occupancy_capacity_from_housing,
)


STRUCTURE_FOOTPRINT_M2 = {
    StructureType.SHELTER: 180.0,
    StructureType.HABITAT: 420.0,
    StructureType.GREENHOUSE: 650.0,
    StructureType.OXYGEN_PLANT: 260.0,
    StructureType.SOLAR_ARRAY: 900.0,
    StructureType.HEATER: 90.0,
    StructureType.RESEARCH_LAB: 320.0,
    StructureType.STORAGE_DEPOT: 480.0,
    StructureType.INFIRMARY: 300.0,
    StructureType.WEATHER_STATION: 80.0,
}

DEFAULT_INITIAL_INVENTORY = ResourceBundle(energy=2, oxygen=2, food=5.0, water=5.0, construction_material=5, minerals=3, med_kits=2.0)

RESOURCE_INPUT_MASS_KG = {
    "food": 1.0,
    "water": 1.0,
    "oxygen": 0.5,
    "construction_material": 8.0,
    "tools": 12.0,
    "energy": 15.0,
    "minerals": 4.0,
    "med_kits": 2.0,
    "ice": 1.0,
    "biomass": 1.0,
    "knowledge": 0.0,
}

INITIAL_STRUCTURE_ALIASES = {structure_type.value: structure_type for structure_type in StructureType}
STRUCTURE_INPUT_UNITS = {
    StructureType.SHELTER: 12.0,
    StructureType.HABITAT: 32.0,
    StructureType.GREENHOUSE: 26.0,
    StructureType.OXYGEN_PLANT: 20.0,
    # Il pozzo e' l'analogo idrico dell'impianto d'ossigeno — perforazione,
    # riscaldamento, condensazione — e pesa come lui, come gia' costa come lui
    # in `BUILD_COSTS`.
    StructureType.WATER_EXTRACTOR: 20.0,
    StructureType.SOLAR_ARRAY: 16.0,
    StructureType.HEATER: 10.0,
    StructureType.RESEARCH_LAB: 28.0,
    StructureType.STORAGE_DEPOT: 12.0,
    StructureType.INFIRMARY: 22.0,
    StructureType.WEATHER_STATION: 8.0,
}
# Diurnal-mean surface solar flux (W/m2) at which solar arrays deliver
# nominal output (clear-sky equatorial Mars from the MCD climatology).
SOLAR_FLUX_REFERENCE_W_M2 = 115.0

STRUCTURE_INPUT_MASS_KG = {
    StructureType.SHELTER: 8_000.0,
    StructureType.HABITAT: 28_000.0,
    StructureType.GREENHOUSE: 18_000.0,
    StructureType.OXYGEN_PLANT: 14_000.0,
    StructureType.WATER_EXTRACTOR: 14_000.0,
    StructureType.SOLAR_ARRAY: 10_000.0,
    StructureType.HEATER: 4_000.0,
    StructureType.RESEARCH_LAB: 22_000.0,
    StructureType.STORAGE_DEPOT: 7_000.0,
    StructureType.INFIRMARY: 16_000.0,
    StructureType.WEATHER_STATION: 3_000.0,
}


def compute_colony_dynamics_metrics(agents: Mapping[str, object], world: GridWorld, config: dict, social: SocialNetwork) -> dict[str, float]:
    """
    Deterministic socio-operational metrics inspired by Mars-base ABM and analogue-mission literature.
    They make the lower simulation layer auditable while LLM/rule agents remain the cognitive layer.
    """
    agent_list = list(agents.values())
    population = len(agent_list)
    structure_cells, counts, colony_built_area_m2, counts_efficaci = _structure_snapshot(world)
    social_metrics = compute_social_metrics(social)
    logistics_metrics = _automatic_logistics_metrics(world)

    # **La coesione legge la cooperazione che ESISTE (2026-08-30).**
    #
    # In modalita' preferenze `COMMUNICATE` e `SHARE_RESOURCE` sono servizi di
    # cella e il pilastro sociale e' deliberatamente vuoto, quindi il grafo
    # sociale non ha archi: misurato, **82 nodi e ZERO archi**. La conseguenza
    # non era dichiarata da nessuna parte, ed era che sei metriche pubblicate
    # diventavano degeneri — `average_trust`, `network_density`,
    # `communication_frequency`, `conflict_intensity` e `social_stability_score`
    # identicamente nulle — e che `cohesion_index` si riduceva a
    # `0,30 x morale`: verificato aritmeticamente, 0,246363 contro 0,246363.
    # Non misurava la coesione: era il morale riscalato.
    #
    # La cooperazione pero' **c'e'**, e il modello la misura gia': deposita,
    # preleva e fa fluire risorse fra celle, e `logistics_cooperation_index` ne
    # e' l'indice, limitato in [0,1]. Il codice usava gia' questo canale per
    # `cooperation_index`; qui si estende lo stesso precedente alle grandezze
    # che restavano vuote. `social_graph_active` dice a chi legge quale dei due
    # canali ha prodotto i numeri, cosi' nessuno scambia uno zero per una misura.
    grafo_sociale_attivo = bool(getattr(social, "edges", None))
    coop_logistica = float(logistics_metrics["logistics_cooperation_index"])
    # **La messa in comune, e la correzione di un errore del 2026-08-30.** Il
    # canale di cella era stato agganciato a `logistics_cooperation_index`,
    # cioe' alla redistribuzione automatica — che nella stessa campagna e'
    # stata portata a SPENTA di default. Le due modifiche si annullavano: la
    # metrica restava a zero esattamente come il grafo a coppie che doveva
    # sostituire. Misurato: `social_cooperation_channel_index` piatto a 0,000
    # per tutti i 1000 passi della run `verifica1000_seed9`.
    #
    # Il canale che c'e' SEMPRE e' il magazzino condiviso: `_remove_combined`
    # paga ogni opera prima dalla sacca di chi lavora e poi dal magazzino
    # della cella. `indice_di_messa_in_comune` ne misura lo stato, e sulla
    # stessa run vale da 0,46 (energia) a 0,88 (materiale): una grandezza
    # viva, non uno zero travestito. La redistribuzione, quando e' accesa,
    # aumenta la messa in comune e quindi e' gia' dentro questo indice.
    messa_in_comune = indice_di_messa_in_comune(
        [agent.inventory for agent in agent_list], structure_cells
    )
    # **La KPI canonica include il terzo canale (2026-08-31).** Combinava il
    # grafo a coppie e la redistribuzione, che in modalita' a preferenze e con
    # il default attuale sono ENTRAMBI zero: `cooperation_index` restava piatta
    # a 0,000 su tutta la run pur dichiarandosi «canonical KPI». Il magazzino
    # condiviso e' il canale che c'e' sempre, e va nella stessa combinazione a
    # complemento: chi coopera per uno qualunque dei tre modi coopera.
    effective_cooperation = 1.0 - (
        (1.0 - social_metrics["cooperation_index"])
        * (1.0 - logistics_metrics["logistics_cooperation_index"])
        * (1.0 - messa_in_comune)
    )

    if grafo_sociale_attivo:
        stabilita = social_metrics["social_stability_score"]
        scambi = min(
            1.0, social_metrics["communication_frequency"] / max(1.0, len(agent_list) * 2.0)
        )
    else:
        # La cooperazione di cella e' collettiva, non a coppie: si legge dove
        # avviene davvero invece di ricostruire archi che il modello non crea.
        stabilita = messa_in_comune
        scambi = messa_in_comune

    avg_stress = _avg(agent_list, "stress_index")
    avg_morale = _avg(agent_list, "morale")
    avg_compliance = _avg(agent_list, "protocol_compliance")
    avg_autonomy = _avg(agent_list, "autonomy_preference")

    # **La resa solare e' quella della simulazione, non una seconda formula
    # (2026-08-25).** Questa metrica derivava la resa dal flusso del layer
    # climatico mentre la produzione di energia la deriva dalla polvere della
    # cella (`step_effects.solar_yield`): due modelli diversi della stessa
    # grandezza, e il margine riportato non era il margine subito. Ora si
    # misura cella per cella con la funzione che produce davvero l'energia.
    # Il flusso resta come ripiego quando non ci sono celle con strutture.
    if structure_cells:
        solar_efficiency = sum(
            solar_yield(cell.dust_level) for cell in structure_cells
        ) / len(structure_cells)
    else:
        solar_flux = float(world.planetary_state.get("solar_flux_w_m2", 0.0) or 0.0)
        solar_efficiency = (
            min(1.0, max(0.25, solar_flux / SOLAR_FLUX_REFERENCE_W_M2))
            if solar_flux > 0.0
            else 1.0
        )
    # **L'interruttore delle capienze (2026-09-14).** Spento, e' il calcolo con
    # cui sono state misurate tutte le campagne in archivio. Acceso, toglie
    # dall'ossigeno il termine `habitat x 0,35` --- che e' l'effetto CIBO
    # dell'habitat, copiato per errore: la tabella degli effetti dice
    # `{habitability, population_support, food, water}`, ossigeno non c'e' ---
    # e pesa i conteggi per l'integrita' delle strutture, perche' la produzione
    # vera scala con quella e queste righe la ignoravano. Resta lineare, e
    # quindi ancora ottimista sotto integrita' 0,4, dove la fisica dimezza.
    capienze_corrette = bool(
        (config.get("engine_fixes") or {}).get("capacity", False)
    )
    c = counts_efficaci if capienze_corrette else counts
    power_capacity = c[StructureType.SOLAR_ARRAY] * 1.5 * solar_efficiency
    oxygen_capacity = (
        c[StructureType.OXYGEN_PLANT] * 1.0 + c[StructureType.INFIRMARY] * 0.6
        + (0.0 if capienze_corrette else counts[StructureType.HABITAT] * 0.35)
    )
    food_capacity = c[StructureType.GREENHOUSE] * 1.0 + c[StructureType.HABITAT] * 0.35
    tool_capacity = c[StructureType.RESEARCH_LAB] * 0.8 + c[StructureType.STORAGE_DEPOT] * 0.5 + c[StructureType.SOLAR_ARRAY] * 0.2
    shelter_capacity = float(
        housing_slots_from_structures(
            counts[StructureType.SHELTER],
            counts[StructureType.HABITAT],
            counts[StructureType.INFIRMARY],
        )
    )
    isru_capacity = counts[StructureType.OXYGEN_PLANT] + counts[StructureType.GREENHOUSE] + counts[StructureType.RESEARCH_LAB] * 0.6 + counts[StructureType.STORAGE_DEPOT] * 0.35

    need = max(1.0, float(population))
    # **Le scorte pubblicate erano le sole sacche (corretto il 2026-08-30).**
    # `food_stock`, `water_stock`, `material_stock` e `tool_stock` finiscono nei
    # margini, nella prosperita' e nell'indice composito del paper: misurati su
    # una colonia di duecento coloni a quattrocento passi, riportavano 714 di
    # cibo su 1929 realmente posseduti e 980 di materiale su 3954, cioe' il 37%
    # e il 25%. E' lo stesso difetto gia' corretto sulla natalita' il
    # 2026-08-27, sopravvissuto in questo secondo percorso: la ragione per cui
    # ora la somma la fa **una funzione sola**.
    inventari = [agent.inventory for agent in agent_list]
    food_stock = scorta_colonia(inventari, structure_cells, "food")
    water_stock = scorta_colonia(inventari, structure_cells, "water")
    material_stock = scorta_colonia(inventari, structure_cells, "construction_material")
    tool_stock = scorta_colonia(inventari, structure_cells, "tools")
    occupied_cells = _occupied_cells(world, structure_cells)
    best_site = _cached_best_colony_site(world, config)
    best_site_score = best_site["score"]
    current_site_score = colony_site_average(occupied_cells)
    # **Il rimpianto confronta due massimi (2026-08-31).** Confrontava il
    # MASSIMO su tutta la mappa con la MEDIA delle celle occupate: due
    # statistiche diverse della stessa funzione, quindi un rimpianto
    # strutturalmente positivo anche quando la colonia occupa la cella
    # migliore del pianeta. Misurato sulla run `verifica1000_seed9`: il sito
    # migliore ERA (0,80), cioe' la cella madre, e il rimpianto pubblicato
    # valeva comunque 0,319. «Quanto meglio avremmo potuto fare» si risponde
    # con la miglior cella che teniamo contro la miglior cella che esiste.
    best_held_site_score = max(
        (colony_site_score(cell) for cell in occupied_cells), default=0.0
    )
    colony_claimed_area_m2 = sum(float(cell.geometry.get("area_m2", 0.0)) for cell in occupied_cells)
    colony_area_m2 = colony_built_area_m2
    colony_density_per_km2 = population / max(1e-9, colony_claimed_area_m2 / 1_000_000.0)
    occupancy_metrics = _occupancy_pressure_metrics(world, occupied_cells)
    eclss_margin = min(1.5, (oxygen_capacity + food_capacity + shelter_capacity * 0.35) / need)
    # **Il margine e' ora il rapporto FISICO fra cio' che i pannelli producono
    # e cio' che gli impianti consumano**, con le stesse costanti che la
    # simulazione applica. Prima era una formula parallela: normalizzava su
    # `max(1.0, carico)` — cioe' su un carico minimo di 1 che non esiste in
    # nessun luogo del modello — e restituiva 0,70 al piano di costruzione
    # mentre nessuna energia veniva consumata affatto. Un margine che non
    # misura nulla e' peggio di un margine assente: sembra un dato.
    energia_prodotta = power_capacity * RESA_ENERGIA
    carico_elettrico = sum(
        counts[tipo] * peso for tipo, peso in CARICO_ELETTRICO.items()
    ) * RESA_ENERGIA_PER_CARICO
    power_margin = (
        min(1.5, energia_prodotta / carico_elettrico) if carico_elettrico > 0.0 else 1.5
    )
    food_margin = min(2.0, (food_stock + food_capacity) / max(1.0, need * 2.0))
    material_margin = min(2.0, (material_stock + tool_stock + tool_capacity) / max(1.0, need * 1.5))
    habitat_pressure = max(0.0, 1.0 - shelter_capacity / need)
    life_support_reliability = _clamp01(0.20 + eclss_margin * 0.38 + power_margin * 0.18 + min(1.0, isru_capacity / need) * 0.14 - avg_stress * 0.12)
    colony_prosperity_index = _clamp01(life_support_reliability * 0.35 + min(1.0, food_margin) * 0.25 + min(1.0, material_margin) * 0.20 + (1.0 - habitat_pressure) * 0.20)
    earth_input_metrics = compute_earth_input_package_metrics(config, population)

    social_cfg = config.get("social", {}) if isinstance(config.get("social", {}), dict) else {}
    delay_minutes = float(social_cfg.get("earth_mars_delay_minutes", 12.0) or 0.0)
    autonomy_need = _clamp01(delay_minutes / 22.0)
    governance_autonomy_score = _clamp01((avg_autonomy * 0.35 + avg_compliance * 0.30 + stabilita * 0.35) - autonomy_need * 0.10)
    cohesion_index = _clamp01(stabilita * 0.55 + avg_morale * 0.30 + scambi * 0.15)
    conflict_risk_index = _clamp01(avg_stress * 0.45 + (1.0 - avg_morale) * 0.25 + social_metrics["conflict_intensity"] * 0.05 + habitat_pressure * 0.20)
    mission_operational_readiness = _clamp01(life_support_reliability * 0.38 + cohesion_index * 0.24 + avg_compliance * 0.20 + min(1.0, isru_capacity / max(1.0, need * 0.35)) * 0.18)
    inter_colony_metrics = compute_inter_colony_cooperation_metrics(agents, social, config)

    return {
        **logistics_metrics,
        # In preference mode resource cooperation is an implicit cell service,
        # not a SHARE_RESOURCE action/social edge. Combine both channels so
        # the canonical KPI does not report zero while real redistribution is
        # continuously serving the colony.
        "cooperation_index": _clamp01(effective_cooperation),
        "crew_stress_index": avg_stress,
        "crew_morale_index": avg_morale,
        "protocol_compliance_index": avg_compliance,
        "earth_mars_delay_minutes": delay_minutes,
        "life_support_reliability": life_support_reliability,
        "eclss_margin": eclss_margin,
        "power_margin": power_margin,
        "food_stock": food_stock,
        "water_stock": water_stock,
        "material_stock": material_stock,
        "tool_stock": tool_stock,
        "tool_capacity_index": min(1.0, tool_capacity / max(1.0, need * 0.5)),
        "food_margin": food_margin,
        "material_margin": material_margin,
        "colony_area_m2": colony_area_m2,
        "colony_built_area_m2": colony_built_area_m2,
        "colony_claimed_area_m2": colony_claimed_area_m2,
        # **L'espansione in celle, non solo in metri quadri (2026-09-06).** I
        # metri quadri costruiti crescono anche senza uscire dalla cella madre;
        # il numero di celle occupate (coloni o strutture) e di celle con
        # strutture e' l'estensione territoriale che il piano di espansione
        # misura e che sulla mappa si vede a occhio.
        "occupied_cells": float(len(occupied_cells)),
        "structure_cells": float(len(structure_cells)),
        "colony_density_per_km2": colony_density_per_km2,
        **occupancy_metrics,
        "colony_site_score": current_site_score,
        "best_colony_site_score": best_site_score,
        "best_colony_site_x": best_site["x"],
        "best_colony_site_y": best_site["y"],
        "best_held_site_score": best_held_site_score,
        "colony_site_regret": max(0.0, best_site_score - best_held_site_score),
        **earth_input_metrics,
        "colony_prosperity_index": colony_prosperity_index,
        "isru_capacity_index": min(1.0, isru_capacity / max(1.0, need * 0.5)),
        "habitat_pressure_index": habitat_pressure,
        **_local_ration_shortfall(occupied_cells),
        # Quale canale ha prodotto stabilita' e scambi: 1 = grafo sociale a
        # coppie, 0 = cooperazione di cella. Senza questo, uno zero da grafo
        # vuoto e uno zero da comunita' inerte sono indistinguibili.
        "social_graph_active": float(grafo_sociale_attivo),
        "social_cooperation_channel_index": stabilita,
        "resource_pooling_index": messa_in_comune,
        "governance_autonomy_score": governance_autonomy_score,
        "cohesion_index": cohesion_index,
        "conflict_risk_index": conflict_risk_index,
        **inter_colony_metrics,
        "mission_operational_readiness": mission_operational_readiness,
    }


def _local_ration_shortfall(occupied_cells: list) -> dict[str, float]:
    """Quanti coloni stanno, ORA, in una cella che non puo' dar loro la razione.

    **Perche' mancava, e cosa costava.** Ogni metrica di supporto vitale di
    questo modello misura la CAPIENZA — quante strutture ci sono — e nessuna
    misurava la SCORTA dove la gente sta davvero. Il risultato, misurato sulla
    run `verifica3000_seed9`: 161 decessi di cui **132 per fame**, mentre
    `life_support_reliability` valeva 1,0 ed `eclss_margin` era al massimo di
    1,5. Al passo 1400 la cella madre aveva dodici serre, tredici occupanti,
    **zero unita' di cibo in magazzino** e un supporto pro capite di 0,049
    contro la soglia di 0,168 — e la colonia, nel suo insieme, ne teneva oltre
    mille altrove. Non e' un difetto di produzione: e' di **logistica**, e con
    la redistribuzione spenta (il default) nulla sposta il cibo fra le celle.

    Le due condizioni sono quelle che `tick_vitals` valuta davvero:
    l'impianto eroga solo se `food_support > 0` **e** la giacenza copre la
    razione. Chi non riceve la razione ripiega sulla propria sacca, e quando
    anche quella e' vuota il contatore di privazione parte.
    """
    totale = 0
    senza_cibo = 0
    senza_acqua = 0
    for cell in occupied_cells:
        occupanti = len(getattr(cell, "agents_present", []) or [])
        if occupanti <= 0:
            continue
        totale += occupanti
        if not (cell.structure_food_effect() > 0.0 and float(cell.resources.food) >= RAZIONE_CIBO):
            senza_cibo += occupanti
        if not (cell.structure_water_effect() > 0.0 and float(cell.resources.water) >= RAZIONE_ACQUA):
            senza_acqua += occupanti
    if totale <= 0:
        return {"local_food_shortfall_index": 0.0, "local_water_shortfall_index": 0.0,
                "underfed_population_now": 0.0}
    return {
        "local_food_shortfall_index": senza_cibo / totale,
        "local_water_shortfall_index": senza_acqua / totale,
        "underfed_population_now": float(senza_cibo),
    }


def _automatic_logistics_metrics(world: GridWorld) -> dict[str, float]:
    metadata = getattr(world, "metadata", {}) or {}
    deposited = float(metadata.get("_automatic_logistics_deposited_total", 0.0) or 0.0)
    withdrawn = float(metadata.get("_automatic_logistics_withdrawn_total", 0.0) or 0.0)
    flow_total = float(metadata.get("_automatic_logistics_flow_total", 0.0) or 0.0)
    agent_steps = float(metadata.get("_automatic_logistics_agent_steps", 0.0) or 0.0)
    # Two resource units served per colonist-step represent full automatic
    # logistics utilization. The bound is explicit and cumulative, so the KPI
    # is comparable across run length and population size.
    logistics_index = _clamp01((withdrawn + flow_total) / max(1.0, agent_steps * 2.0))
    return {
        "automatic_logistics_enabled": float(bool(metadata.get("_automatic_logistics_enabled", False))),
        "automatic_resource_deposited_total": deposited,
        "automatic_resource_withdrawn_total": withdrawn,
        "automatic_intercell_flow_total": flow_total,
        "automatic_intercell_flow_events": float(metadata.get("_automatic_logistics_flow_events", 0) or 0),
        "automatic_redistribution_steps": float(metadata.get("_automatic_logistics_steps", 0) or 0),
        "automatic_redistribution_active_steps": float(metadata.get("_automatic_logistics_active_steps", 0) or 0),
        "automatic_logistics_agent_steps": agent_steps,
        "automatic_resource_deposited_last_step": float(metadata.get("_automatic_logistics_last_deposited", 0.0) or 0.0),
        "automatic_resource_withdrawn_last_step": float(metadata.get("_automatic_logistics_last_withdrawn", 0.0) or 0.0),
        "automatic_intercell_flow_last_step": float(metadata.get("_automatic_logistics_last_flow", 0.0) or 0.0),
        "logistics_cooperation_index": logistics_index,
    }


def _occupancy_pressure_metrics(world: GridWorld, occupied_cells: list) -> dict[str, float]:
    arrays = getattr(world, "_cells", None)
    if arrays is not None and hasattr(arrays, "occupancy"):
        occupancy = arrays.occupancy.astype(np.float64)
        capacity = occupancy_capacity_from_housing(
            arrays.struct_count[:, :, C.S[StructureType.SHELTER]],
            arrays.struct_count[:, :, C.S[StructureType.HABITAT]],
            arrays.struct_count[:, :, C.S[StructureType.INFIRMARY]],
        )
        excess = np.maximum(0.0, occupancy - capacity)
        if occupancy.size and float(occupancy.max()) > 0.0:
            densest = np.unravel_index(int(np.argmax(occupancy)), occupancy.shape)
            max_occupancy = float(occupancy[densest])
            capacity_at_max = float(capacity[densest])
        else:
            max_occupancy = capacity_at_max = 0.0
        return {
            "max_cell_occupancy": max_occupancy,
            "max_cell_occupancy_capacity": capacity_at_max,
            "max_cell_overcrowding_excess": float(excess.max(initial=0.0)),
            "overcrowded_cell_count": float(np.count_nonzero(excess > 0.0)),
        }

    if not occupied_cells:
        return {
            "max_cell_occupancy": 0.0,
            "max_cell_occupancy_capacity": 0.0,
            "max_cell_overcrowding_excess": 0.0,
            "overcrowded_cell_count": 0.0,
        }
    densest = max(occupied_cells, key=lambda cell: len(cell.agents_present))
    excesses = [float(cell.overcrowding_excess()) for cell in occupied_cells]
    return {
        "max_cell_occupancy": float(len(densest.agents_present)),
        "max_cell_occupancy_capacity": float(densest.occupancy_capacity()),
        "max_cell_overcrowding_excess": max(excesses, default=0.0),
        "overcrowded_cell_count": float(sum(value > 0.0 for value in excesses)),
    }


def _structure_cells(world: GridWorld) -> list:
    cells = getattr(world, "_cells", None)
    if cells is not None and hasattr(cells, "structure_positions"):
        return [world.get_cell(x, y) for x, y in cells.structure_positions()]
    positions = getattr(world, "_structure_positions", set())
    if not positions:
        positions = {
            (x, y)
            for y, row in enumerate(world.cells)
            for x, cell in enumerate(row)
            if cell.structures
        }
        world._structure_positions = positions
    return [world.get_cell(x, y) for x, y in positions if world.in_bounds(x, y) and world.get_cell(x, y).structures]


def _structure_snapshot(world: GridWorld) -> tuple[list, Counter, float, Counter]:
    """Read counts and built area directly from vector columns when present.

    Il quarto valore e' il conteggio PESATO PER INTEGRITA': un impianto al
    quaranta per cento conta 0,4. Serve all'interruttore delle capienze, che
    senza di esso continuerebbe a dichiarare la capacita' di impianti nuovi.
    """
    cells = getattr(world, "_cells", None)
    if cells is not None and hasattr(cells, "struct_count"):
        structure_cells = [
            world.get_cell(x, y) for x, y in cells.structure_positions()
        ]
        count_totals = np.sum(cells.struct_count, axis=(0, 1), dtype=np.int64)
        integrity_totals = np.sum(cells.struct_integrity, axis=(0, 1), dtype=np.float64)
        counts = Counter(
            {
                structure_type: int(count_totals[index])
                for index, structure_type in enumerate(C.STRUCTURES)
                if count_totals[index] > 0
            }
        )
        built_area = sum(
            STRUCTURE_FOOTPRINT_M2.get(structure_type, 150.0)
            * float(integrity_totals[index])
            for index, structure_type in enumerate(C.STRUCTURES)
        )
        efficaci = Counter(
            {
                structure_type: float(integrity_totals[index])
                for index, structure_type in enumerate(C.STRUCTURES)
                if count_totals[index] > 0
            }
        )
        return structure_cells, counts, built_area, efficaci

    structure_cells = _structure_cells(world)
    structures = [structure for cell in structure_cells for structure in cell.structures]
    counts = Counter(structure.type for structure in structures)
    built_area = sum(
        STRUCTURE_FOOTPRINT_M2.get(structure.type, 150.0) * structure.integrity
        for structure in structures
    )
    efficaci = Counter()
    for structure in structures:
        efficaci[structure.type] += float(structure.integrity)
    return structure_cells, counts, built_area, efficaci


def _occupied_cells(world: GridWorld, structure_cells: list) -> list:
    by_position = {(cell.x, cell.y): cell for cell in structure_cells}
    for x, y in getattr(world, "_agent_positions", {}).values():
        if world.in_bounds(x, y):
            by_position[(x, y)] = world.get_cell(x, y)
    return list(by_position.values())


def _cached_best_colony_site(world: GridWorld, config: dict) -> dict[str, float]:
    step = int(getattr(world, "step", 0) or 0)
    cache = world.metadata.get("_best_colony_site_cache") if isinstance(world.metadata, dict) else None
    interval = _best_site_refresh_interval(config)
    if isinstance(cache, dict) and step - int(cache.get("step", -interval - 1)) < interval:
        return {
            "x": float(cache.get("x", -1.0)),
            "y": float(cache.get("y", -1.0)),
            "score": float(cache.get("score", 0.0)),
        }

    all_cells = (cell for row in world.cells for cell in row)
    cell, score = best_colony_site(all_cells)
    value = {
        "step": step,
        "x": float(cell.x) if cell is not None else -1.0,
        "y": float(cell.y) if cell is not None else -1.0,
        "score": float(score),
    }
    world.metadata["_best_colony_site_cache"] = value
    return {"x": value["x"], "y": value["y"], "score": value["score"]}


def _best_site_refresh_interval(config: dict) -> int:
    metrics_cfg = config.get("metrics", {}) if isinstance(config.get("metrics"), dict) else {}
    headless_cfg = config.get("headless", {}) if isinstance(config.get("headless"), dict) else {}
    raw = metrics_cfg.get("best_site_refresh_interval_steps", headless_cfg.get("best_site_refresh_interval_steps", 100))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 100


def compute_earth_input_package_metrics(config: dict, population: int) -> dict[str, float]:
    """Proxy metrics for comparing initial Earth-supplied colony packages."""
    agents_cfg = config.get("agents", {}) if isinstance(config.get("agents"), dict) else {}
    colony_cfg = config.get("colony", {}) if isinstance(config.get("colony"), dict) else {}
    initial_inventory = agents_cfg.get("initial_inventory") if isinstance(agents_cfg.get("initial_inventory"), dict) else None
    inventory = ResourceBundle.from_dict({**DEFAULT_INITIAL_INVENTORY.to_dict(), **initial_inventory}) if initial_inventory else DEFAULT_INITIAL_INVENTORY
    initial_structures = colony_cfg.get("initial_structures")
    if not isinstance(initial_structures, dict):
        initial_structures = agents_cfg.get("initial_structures") if isinstance(agents_cfg.get("initial_structures"), dict) else {}

    need = max(1.0, float(population))
    inventory_totals = {key: float(value) * need for key, value in inventory.to_dict().items()}
    resource_mass_kg = sum(inventory_totals[key] * RESOURCE_INPUT_MASS_KG.get(key, 0.0) for key in inventory_totals)
    resource_units = (
        inventory_totals["food"]
        + inventory_totals["water"]
        + inventory_totals["oxygen"] * 0.5
        + inventory_totals["construction_material"] * 2.0
        + inventory_totals["tools"] * 4.0
        + inventory_totals["energy"] * 3.0
        + inventory_totals["minerals"] * 1.5
        + inventory_totals["med_kits"] * 1.2
        + inventory_totals["ice"]
    )

    structure_counts: Counter[StructureType] = Counter()
    for key, value in initial_structures.items():
        structure_type = INITIAL_STRUCTURE_ALIASES.get(str(key))
        if structure_type is None:
            continue
        structure_counts[structure_type] += max(0, int(value or 0))

    structure_units = sum(STRUCTURE_INPUT_UNITS[structure_type] * count for structure_type, count in structure_counts.items())
    structure_mass_kg = sum(STRUCTURE_INPUT_MASS_KG[structure_type] * count for structure_type, count in structure_counts.items())
    habitat_slots = float(
        housing_slots_from_structures(
            structure_counts[StructureType.SHELTER],
            structure_counts[StructureType.HABITAT],
            structure_counts[StructureType.INFIRMARY],
        )
    )
    life_support_slots = (
        structure_counts[StructureType.OXYGEN_PLANT] * 1.0
        + structure_counts[StructureType.HABITAT] * 0.35
        + structure_counts[StructureType.INFIRMARY] * 0.6
    )
    power_slots = structure_counts[StructureType.SOLAR_ARRAY] * 1.5
    food_slots = structure_counts[StructureType.GREENHOUSE] * 1.0 + structure_counts[StructureType.HABITAT] * 0.35

    food_water_readiness = min(1.0, (inventory_totals["food"] + inventory_totals["water"]) / (need * 14.0))
    material_readiness = min(1.0, (inventory_totals["construction_material"] + inventory_totals["minerals"] + inventory_totals["tools"] * 2.0) / (need * 10.0))
    habitat_readiness = min(1.0, habitat_slots / need)
    eclss_readiness = min(1.0, (life_support_slots + power_slots * 0.35) / need)
    food_generation_readiness = min(1.0, food_slots / max(1.0, need * 0.35))
    sufficiency = _clamp01(
        food_water_readiness * 0.28
        + material_readiness * 0.22
        + habitat_readiness * 0.20
        + eclss_readiness * 0.20
        + food_generation_readiness * 0.10
    )

    total_units = resource_units + structure_units
    total_mass_kg = resource_mass_kg + structure_mass_kg
    return {
        "earth_input_package_units": total_units,
        "earth_input_mass_proxy_kg": total_mass_kg,
        "earth_input_per_capita_units": total_units / need,
        "earth_input_per_capita_mass_proxy_kg": total_mass_kg / need,
        "earth_input_sufficiency_index": sufficiency,
        "earth_input_deficit_index": 1.0 - sufficiency,
    }


def compute_inter_colony_cooperation_metrics(agents: Mapping[str, object], social: SocialNetwork, config: dict) -> dict[str, float]:
    assignments = _agent_colony_assignments(agents, config)
    colonies = sorted(set(assignments.values()))
    colony_count = len(colonies)
    if colony_count <= 1:
        return {
            "inter_colony_count": float(colony_count),
            "inter_colony_edge_count": 0.0,
            "inter_colony_communication_frequency": 0.0,
            "inter_colony_trade_events": 0.0,
            "inter_colony_conflict_intensity": 0.0,
            "inter_colony_cooperation_index": 0.0,
            "inter_colony_cooperation_need_index": 0.0,
            "inter_colony_cooperation_gap_index": 0.0,
        }

    colony_sizes = Counter(assignments.values())
    possible_inter_edges = 0
    for index, colony in enumerate(colonies):
        for other in colonies[index + 1 :]:
            possible_inter_edges += colony_sizes[colony] * colony_sizes[other]
    possible = max(1.0, float(possible_inter_edges))

    communication = 0.0
    trade = 0.0
    alliance = 0.0
    conflict = 0.0
    edge_count = 0.0
    for (a, b), edge in social.edges.items():
        if assignments.get(a) == assignments.get(b):
            continue
        edge_count += 1.0
        communication += float(edge.communication_frequency)
        trade += float(edge.trade)
        alliance += float(edge.alliance)
        conflict += float(edge.conflict)

    resource_imbalance = _colony_resource_imbalance(agents, assignments)
    social_cfg = config.get("social", {}) if isinstance(config.get("social"), dict) else {}
    delay_pressure = _clamp01(float(social_cfg.get("earth_mars_delay_minutes", 12.0) or 0.0) / 22.0)
    colony_count_pressure = _clamp01((colony_count - 1) / 4.0)
    delivered = _clamp01((communication * 0.15 + trade * 0.55 + alliance * 2.0) / possible)
    need = _clamp01(0.20 + resource_imbalance * 0.35 + colony_count_pressure * 0.25 + min(1.0, conflict / possible) * 0.10 + delay_pressure * 0.10)

    return {
        "inter_colony_count": float(colony_count),
        "inter_colony_edge_count": edge_count,
        "inter_colony_communication_frequency": communication,
        "inter_colony_trade_events": trade,
        "inter_colony_conflict_intensity": conflict,
        "inter_colony_cooperation_index": delivered,
        "inter_colony_cooperation_need_index": need,
        "inter_colony_cooperation_gap_index": max(0.0, need - delivered),
    }


def _agent_colony_assignments(agents: Mapping[str, object], config: dict) -> dict[str, str]:
    configured = _configured_colony_assignments(config)
    assignments: dict[str, str] = {}
    for agent_id, agent in agents.items():
        colony_id = getattr(agent, "colony_id", None) or configured.get(agent_id)
        assignments[agent_id] = str(colony_id) if colony_id is not None else "colony_0"
    return assignments


def _configured_colony_assignments(config: dict) -> dict[str, str]:
    for section_name in ("colonies", "social", "agents"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in ("agent_colonies", "colony_assignments", "assignments"):
            value = section.get(key)
            if isinstance(value, dict):
                return {str(agent_id): str(colony_id) for agent_id, colony_id in value.items()}
    return {}


def _colony_resource_imbalance(agents: Mapping[str, object], assignments: dict[str, str]) -> float:
    totals: dict[str, float] = {}
    for agent_id, agent in agents.items():
        inventory = getattr(agent, "inventory", None)
        value = 0.0
        if inventory is not None:
            value = (
                float(getattr(inventory, "food", 0.0))
                + float(getattr(inventory, "water", 0.0))
                + float(getattr(inventory, "construction_material", 0.0))
                + float(getattr(inventory, "tools", 0.0)) * 2.0
                + float(getattr(inventory, "oxygen", 0.0)) * 0.5
            )
        colony_id = assignments.get(agent_id, "colony_0")
        totals[colony_id] = totals.get(colony_id, 0.0) + value
    if len(totals) <= 1:
        return 0.0
    values = list(totals.values())
    return _clamp01((max(values) - min(values)) / max(1.0, sum(values) / len(values)))


def _avg(items: list[object], attr: str) -> float:
    if not items:
        return 0.0
    return sum(float(getattr(item, attr, 0.0)) for item in items) / len(items)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
