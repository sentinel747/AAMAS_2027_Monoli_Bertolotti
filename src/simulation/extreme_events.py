from __future__ import annotations

"""Shared cell-based catastrophic events (dust storm fronts, solar flares).

One engine used by BOTH SimulationController (GUI/headless) and
AgentCoupledRunner (experiments) so event dynamics, logging and the
active_event payload (cells/trail_cells consumed by the frontends) are
identical regardless of how a run was launched.

**Gli eventi agiscono sulla FISICA del mondo, non con danni ad hoc
(2026-08-25).** Prima toccavano soltanto `agent.health`, `agent.fatigue` e
`structure.integrity` con costanti proprie, e il risultato era misurabile:
su 400 passi con 300 coloni, diciotto eventi arrivati e settantuno passi
sotto evento, la salute totale persa era **0,03 punti** e le morti zero.
La causa stava tutta in una riga — bastava UNA struttura con integrita' >
0,3 nella cella per dichiarare al riparo TUTTI gli occupanti, e il 99,9%
degli agenti colpiti (1513 su 1514) risultava protetto.

La correzione non inventa numeri nuovi: usa i canali che il simulatore gia'
modella. Una tempesta alza la **polvere** delle celle investite, un
brillamento la **radiazione**, e da li' in poi la catena esiste gia' —
`kernel_biology.recompute_habitability` sottrae `radiation * 0,03` e
`dust * 0,02` all'abitabilita', che `kernel_vitals` legge per la salute dei
coloni; `apply_structure_wear` aggiunge `dust * 0,010` e `radiation * 0,003`
di usura per anno. Un evento diventa cosi' un fatto ambientale che nessun
riparo annulla, invece di un danno che un tetto qualsiasi cancella.

**Le magnitudini vengono dal MCD, non da una scelta di gusto.** Campionato
sul dataset reale alla stessa latitudine e stagione: in climatologia
l'opacita' vale 0,224 con 527 W/m2 di flusso solare, nello scenario di
tempesta globale (`strm`) vale 5,000 con 57,8 W/m2 — un fattore ventidue
sulla polvere e un crollo dell'89% sull'irraggiamento. E' la condizione che
uccise Opportunity, ed e' la ragione per cui `DUST_STORM_OPACITY_SURGE`
satura il fattore polvere della cella invece di incrementarlo di poco.

**Il riparo ha ora una capienza.** Un habitat protegge cinque coloni e un
rifugio uno solo, gli stessi rapporti che `COLONISTS_PER_STRUCTURE` usa per
dimensionare la colonia: una capitale attrezzata continua a proteggere i
suoi, un avamposto con due rifugi e dieci abitanti no. E' la differenza fra
un modello in cui il riparo e' una proprieta' della cella e uno in cui e'
una risorsa contesa.
"""

import numpy as np

from src.world.grid import GridWorld
from src.world.structures import StructureType


#: Quanto una tempesta alza la polvere della cella investita, in unita' di
#: `dust_level` (che `_apply_climate_to_grid` tiene in [0, 1] e la biologia
#: legge come `dust`). Il MCD misura l'opacita' passare da 0,224 a 5,000 fra
#: climatologia e tempesta globale: sul canale normalizzato del simulatore
#: quel salto satura, quindi una tempesta piena porta la cella al massimo.
DUST_STORM_OPACITY_SURGE = 0.85

#: Quanto un brillamento alza la radiazione della cella, in unita' di
#: `radiation_level` (che `_apply_climate_to_grid` tiene in [0,25, 1,8]).
#: Un evento a severita' piena raddoppia grosso modo la radiazione di fondo
#: di una cella equatoriale, che parte intorno a 0,6.
SOLAR_FLARE_RADIATION_SURGE = 0.60

#: Quanti coloni ripara ciascuna struttura. Sono i rapporti con cui la
#: colonia si dimensiona (`COLONISTS_PER_STRUCTURE`): un habitat vale cinque
#: persone, un rifugio una. Oltre la capienza, chi resta fuori e' esposto.
SHELTER_CAPACITY = {StructureType.HABITAT: 5, StructureType.SHELTER: 1}


def event_config_enabled(config: dict, events_cfg: dict) -> bool:
    """Catastrophic events are explicit opt-in for reproducible baselines."""
    raw = config.get("extreme_events_enabled", events_cfg.get("enabled", False))
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def event_chance_per_step(config: dict, events_cfg: dict) -> float:
    raw = events_cfg.get("chance_per_step", config.get("extreme_event_chance_per_step", 0.05))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.05
    return max(0.0, min(1.0, value))


class ExtremeEventEngine:
    def __init__(self):
        self.active_event: dict | None = None
        self.upcoming_event: dict | None = None
        self.event_timer: int = 0
        #: Il delta ambientale applicato al passo precedente, per cella:
        #: {(x, y): (delta_polvere, delta_radiazione)}. Va ripristinato prima
        #: di applicare quello nuovo, perche' il fronte si muove e una cella
        #: gia' attraversata deve tornare al suo clima. Senza, la mappa
        #: accumulerebbe polvere a ogni passaggio e la tempesta diventerebbe
        #: permanente.
        self._environment_delta: dict[tuple[int, int], tuple[float, float]] = {}

    def reset(self) -> None:
        self.active_event = None
        self.upcoming_event = None
        self.event_timer = 0
        self._environment_delta = {}

    def advance(self, world: GridWorld, agents: dict, config: dict, step_index: int) -> None:
        """Advance detection, movement and impacts of global extreme events."""
        events_cfg = config.get("extreme_events", {}) if isinstance(config.get("extreme_events"), dict) else {}
        if not event_config_enabled(config, events_cfg):
            self.reset()
            world.metadata["active_event"] = None
            world.metadata["upcoming_event"] = None
            return

        rng = np.random.default_rng(int(config.get("seed", 0)) + step_index * 7919)

        if self.active_event:
            self.event_timer -= 1
            if self.event_timer <= 0:
                world.log_event(
                    "event_ended",
                    f"Extreme event {self.active_event.get('type', 'unknown')} has passed.",
                    step=step_index,
                    type=self.active_event.get("type"),
                )
                self.active_event = None
                self._restore_environment(world)

        if self.upcoming_event:
            self.upcoming_event["countdown"] = int(self.upcoming_event.get("countdown", 1)) - 1
            if self.upcoming_event["countdown"] <= 0:
                duration = int(rng.integers(3, 6))
                self.active_event = {
                    **self.upcoming_event,
                    "started_step": step_index,
                    "duration_steps": duration,
                    "cells": [],
                    "trail_cells": [],
                }
                self.upcoming_event = None
                self.event_timer = duration
                world.log_event(
                    "extreme_event_started",
                    f"CRITICAL: {self.active_event['type']} is crossing the map.",
                    step=step_index,
                    type=self.active_event["type"],
                    severity=float(self.active_event["severity"]),
                )

        if not self.active_event and not self.upcoming_event:
            chance = event_chance_per_step(config, events_cfg)
            if rng.random() < chance:
                event_type = str(rng.choice(["Global Dust Storm", "Solar Flare"]))
                self.upcoming_event = {
                    "type": event_type,
                    "countdown": int(rng.integers(1, 4)),
                    "severity": float(rng.uniform(0.5, 1.0)),
                }
                world.log_event(
                    "event_detected",
                    f"Long-range sensors detect a potential {event_type} approaching.",
                    step=step_index,
                    type=event_type,
                    countdown=self.upcoming_event["countdown"],
                    severity=float(self.upcoming_event["severity"]),
                )

        if self.active_event:
            current_cells = self._event_cells_for_current_step(world, self.active_event, step_index)
            previous_trail = self.active_event.get("trail_cells", [])
            trail_by_key = {f"{cell['x']},{cell['y']}": cell for cell in previous_trail if isinstance(cell, dict)}
            for cell in current_cells:
                trail_by_key[f"{cell['x']},{cell['y']}"] = cell
            self.active_event["cells"] = current_cells
            self.active_event["trail_cells"] = list(trail_by_key.values())[-260:]
            self.active_event["remaining_steps"] = max(0, self.event_timer)
            self._apply_event_impacts(world, agents, self.active_event, current_cells)

        world.metadata["active_event"] = self.active_event
        world.metadata["upcoming_event"] = self.upcoming_event

    def _event_cells_for_current_step(self, world: GridWorld, event: dict, step_index: int) -> list[dict]:
        width = max(1, world.width)
        height = max(1, world.height)
        started = int(event.get("started_step", step_index))
        duration = max(1, int(event.get("duration_steps", 1)))
        local_step = max(0, step_index - started)
        progress = min(1.0, local_step / max(1, duration - 1))
        severity = float(event.get("severity", 0.7))
        radius = max(1, int(round((2 + severity * 3) * max(width, height) / 50)))
        cells: list[dict] = []

        if event.get("type") == "Solar Flare":
            center_y = int(round(progress * (height - 1)))
            drift_x = int(round((0.5 + 0.35 * np.sin((step_index + severity) * 0.9)) * (width - 1)))
            for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
                span = max(2, radius + int(abs(y - center_y) * 0.5))
                for x in range(max(0, drift_x - span), min(width, drift_x + span + 1)):
                    cells.append({"x": x, "y": y})
            return cells

        center_x = int(round(progress * (width - 1)))
        wave = max(1, int(round(radius * 1.4)))
        for y in range(height):
            offset = int(round(np.sin((y + step_index) * 0.33) * wave))
            x0 = center_x + offset
            for x in range(max(0, x0 - radius), min(width, x0 + radius + 1)):
                cells.append({"x": x, "y": y})
        return cells

    def _restore_environment(self, world: GridWorld) -> None:
        """Riporta al clima le celle che il fronte ha lasciato.

        Sottrae esattamente il delta applicato, invece di riscrivere un
        valore assoluto: fra un passo e l'altro il refresh planetario o la
        biologia possono aver mosso quelle grandezze per conto loro, e
        sovrascriverle cancellerebbe il loro lavoro.
        """
        for (x, y), (d_dust, d_rad) in self._environment_delta.items():
            try:
                cell = world.get_cell(x, y)
            except (IndexError, KeyError):
                continue
            if d_dust:
                cell.dust_level = max(0.0, cell.dust_level - d_dust)
            if d_rad:
                cell.radiation_level = max(0.0, cell.radiation_level - d_rad)
        self._environment_delta = {}

    def _sheltered_capacity(self, cell) -> int:
        """Quanti coloni quella cella puo' mettere al riparo.

        Contano le strutture ancora in piedi: sotto il 30% di integrita' un
        habitat non ripara piu' nessuno, che e' la stessa soglia usata prima
        per la protezione binaria.
        """
        return sum(
            SHELTER_CAPACITY.get(structure.type, 0)
            for structure in cell.structures
            if structure.integrity > 0.3
        )

    def _apply_event_impacts(self, world: GridWorld, agents: dict, event: dict, affected_cells: list[dict]) -> None:
        """Effetti dell'evento: prima sull'ambiente, poi su chi ci vive.

        L'ordine conta. L'ambiente e' il canale che nessun riparo annulla e
        che raggiunge la colonia per le vie che il modello gia' percorre —
        abitabilita' della cella, e da li' salute; usura delle strutture. I
        danni diretti restano, ma come cio' che sono: l'effetto immediato su
        chi si trova all'aperto quando il fronte passa.
        """
        severity = float(event.get("severity", 0.7))
        event_type = event.get("type")
        affected = {(int(cell["x"]), int(cell["y"])) for cell in affected_cells}

        # --- 1. l'ambiente ------------------------------------------------
        self._restore_environment(world)
        delta_dust = DUST_STORM_OPACITY_SURGE * severity if event_type == "Global Dust Storm" else 0.0
        delta_rad = SOLAR_FLARE_RADIATION_SURGE * severity if event_type == "Solar Flare" else 0.0
        if delta_dust or delta_rad:
            nuovo: dict[tuple[int, int], tuple[float, float]] = {}
            for x, y in affected:
                try:
                    cell = world.get_cell(x, y)
                except (IndexError, KeyError):
                    continue
                # Si memorizza il delta EFFETTIVAMENTE applicato, non quello
                # richiesto: i due differiscono ogni volta che il limite
                # superiore morde (una cella gia' polverosa satura a 1,0), e
                # sottrarre in ripristino il delta nominale lascerebbe la
                # cella piu' pulita di come la si era trovata. Trovato da
                # `test_the_front_leaves_the_cells_it_has_passed_as_it_found_them`.
                applicato_dust = applicato_rad = 0.0
                if delta_dust:
                    prima = cell.dust_level
                    cell.dust_level = min(1.0, prima + delta_dust)
                    applicato_dust = cell.dust_level - prima
                if delta_rad:
                    prima = cell.radiation_level
                    cell.radiation_level = min(1.8, prima + delta_rad)
                    applicato_rad = cell.radiation_level - prima
                nuovo[(x, y)] = (applicato_dust, applicato_rad)
            self._environment_delta = nuovo

        # --- 2. chi e' all'aperto ----------------------------------------
        # La capienza si consuma: i primi occupanti trovano posto, gli altri
        # restano fuori. L'ordine e' quello del dizionario degli agenti, che
        # e' stabile a parita' di seme.
        posti_rimasti: dict[tuple[int, int], int] = {}
        for agent in agents.values():
            chiave = (agent.x, agent.y)
            if chiave not in affected:
                continue
            cell = world.get_cell(agent.x, agent.y)
            if chiave not in posti_rimasti:
                posti_rimasti[chiave] = self._sheltered_capacity(cell)
            if posti_rimasti[chiave] > 0:
                posti_rimasti[chiave] -= 1
                is_sheltered = True
            else:
                is_sheltered = False
            if event_type == "Solar Flare":
                if not is_sheltered:
                    agent.health = max(0.0, agent.health - 0.10 * severity)
                    agent.memory.add_event("SYSTEM ALERT: Solar flare crossing this cell; seek shelter.")
            elif event_type == "Global Dust Storm":
                agent.fatigue = min(1.0, agent.fatigue + 0.12 * severity)
                if not is_sheltered:
                    agent.health = max(0.0, agent.health - 0.035 * severity)
                agent.memory.add_event("SYSTEM ALERT: Dust storm front crossing this cell.")

        # --- 3. le strutture ---------------------------------------------
        for x, y in affected:
            cell = world.get_cell(x, y)
            for structure in cell.structures:
                wear = 0.015 * severity
                if event_type == "Global Dust Storm" and structure.type == StructureType.SOLAR_ARRAY:
                    wear *= 2.5
                structure.integrity = max(0.0, structure.integrity - wear)
