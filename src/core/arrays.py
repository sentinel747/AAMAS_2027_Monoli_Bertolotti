from __future__ import annotations

import os

import numpy as np

from src.agents.pillars import N_PILLARS
from src.core import constants as C
from src.world.structures import StructureType
from src.world.terrain import TerrainType

_MISSION_NONE, MISSION_SCOUT_OUT, MISSION_SCOUT_BACK, MISSION_SETTLE_OUT = 0, 1, 2, 3

# Modalita' di verifica dell'indice posizioni (vedi `AgentArrays.positions_index`).
# Costosa per costruzione: ricostruisce l'indice a ogni accesso per confrontarlo
# con quello memorizzato. Serve nei test di parita', non nelle run normali.
_VERIFY_POSITION_INDEX = os.environ.get("MARSABM_VERIFY_INDEX", "0") == "1"

# Toggle di sola misura del braccio di controllo (Task 0b del piano di migrazione
# Rust): quando e' spento, `CellView.agents_present` torna a ricalcolare la
# scansione a ogni lettura. Serve a confrontare Rust contro un Python con lo
# stesso precompute, non contro un Python che ricalcola invarianti.
_POSITION_INDEX_ENABLED = os.environ.get("MARSABM_PRECOMPUTE", "1") != "0"

_F32_FIELDS = ("health", "oxygen", "hydration", "satiety", "fatigue",
               "stress", "morale", "cooperation", "compliance", "autonomy",
               "risk_tolerance", "curiosity")
# attributo sorgente sull'oggetto BaseAgent per ogni colonna float
_F32_SOURCE = {"oxygen": "oxygen_level", "stress": "stress_index",
               "compliance": "protocol_compliance", "autonomy": "autonomy_preference"}

# sentinella fill values per campi che non devono inizializzarsi a zero
_FILL = {
    "target_x": -1,
    "target_y": -1,
    "mission_next_at": -1,
    "settle_next_at": -1,
    "last_comm_step": -10**6,
    "pref": 1.0 / N_PILLARS,
    "skill": 1.0,
}


class AgentArrays:
    def __init__(self, capacity: int):
        self.x = np.zeros(capacity, dtype=np.int16)
        self.y = np.zeros(capacity, dtype=np.int16)
        self.alive = np.zeros(capacity, dtype=np.bool_)
        # Task 14 fix (equivalence audit, controller decision - exact
        # equivalence): float64, not float32 - the object engine's own
        # `BaseAgent` attributes are plain Python floats (float64), and the
        # v1<->v2 comparison (scripts/compare_v1_v2.py) found float32
        # truncation was the source of tiny, real (if usually harmless)
        # per-step drift on health/inventory/vitals.
        for name in _F32_FIELDS:
            setattr(self, name, np.zeros(capacity, dtype=np.float64))
        self.steps_without_water = np.zeros(capacity, dtype=np.int16)
        self.steps_without_food = np.zeros(capacity, dtype=np.int16)
        self.inv = np.zeros((capacity, C.NR), dtype=np.float64)
        self.mission = np.zeros(capacity, dtype=np.int8)
        self.target_x = np.full(capacity, _FILL["target_x"], dtype=np.int16)
        self.target_y = np.full(capacity, _FILL["target_y"], dtype=np.int16)
        self.mission_steps = np.zeros(capacity, dtype=np.int16)
        # scout_next_at is stored in mission_next_at (the historical column);
        # settle_next_at gets its own column so the two independent cadences
        # (see rule_based_agent.py's SCOUT_INTERVAL_STEPS/SETTLE_INTERVAL_STEPS)
        # cannot clobber each other through AgentView (views.py).
        self.mission_next_at = np.full(capacity, _FILL["mission_next_at"], dtype=np.int32)
        self.settle_next_at = np.full(capacity, _FILL["settle_next_at"], dtype=np.int32)
        self.home_x = np.zeros(capacity, dtype=np.int16)
        self.home_y = np.zeros(capacity, dtype=np.int16)
        self.last_comm_step = np.full(capacity, _FILL["last_comm_step"], dtype=np.int32)
        self.comm_rotation = np.zeros(capacity, dtype=np.int32)
        self.pref = np.full(
            (capacity, N_PILLARS), 1.0 / N_PILLARS, dtype=np.float64
        )
        self.skill = np.ones((capacity, N_PILLARS), dtype=np.float64)
        self.n = 0
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        # --- Indice per-cella degli agenti vivi (perf, nessun cambio di
        # comportamento) -----------------------------------------------------
        # `CellView.agents_present` (src/core/views.py) rifaceva una scansione
        # O(n_vivi) piu' due gather NumPy a OGNI lettura: 255.531 chiamate in
        # 100 step nel profilo, quasi tutte per un semplice `len(...)`. Un
        # sondaggio con cache non invalidata ha misurato un limite superiore di
        # circa 1,26x end-to-end, quindi vale la pena mantenere un indice vero.
        #
        # L'invalidazione e' a epoca: `touch_positions()` va chiamata a OGNI
        # scrittura di `x`, `y` o `alive`. I punti di scrittura sono pochi e
        # centralizzati (`from_agents`, `spawn`, i setter di `AgentView.x/.y`,
        # il colpo di mortalita' in kernel_vitals). Perche' l'invariante non
        # resti affidato alla memoria di chi tocchera' questi array in futuro,
        # `MARSABM_VERIFY_INDEX=1` ricostruisce l'indice a ogni accesso e lo
        # confronta con quello memorizzato: la suite di parita' gira anche in
        # quella modalita'.
        self._pos_epoch = 0
        self._pos_index: dict[tuple[int, int], list[str]] | None = None
        self._pos_index_epoch = -1

    def touch_positions(self) -> None:
        """Invalida l'indice per-cella.

        Da chiamare dopo ogni scrittura di ``x``, ``y`` o ``alive``. E'
        volutamente a granularita' grossa: invalidare l'intero indice costa una
        ricostruzione O(n_vivi) alla prossima lettura, mentre un aggiornamento
        incrementale richiederebbe di conoscere la posizione precedente in ogni
        punto di scrittura, cioe' proprio il tipo di accoppiamento che rende
        fragile un'invalidazione manuale.
        """
        self._pos_epoch += 1

    def _build_positions_index(self) -> dict[tuple[int, int], list[str]]:
        index: dict[tuple[int, int], list[str]] = {}
        xs, ys = self.x, self.y
        # Ordine di riga crescente: e' lo stesso ordine che produceva la vecchia
        # `agents_present` (`np.flatnonzero` restituisce indici crescenti), e
        # l'ordine degli id e' osservabile - `action_space.py` lo usa per
        # costruire l'insieme dei vicini.
        for row in self.alive_rows():
            r = int(row)
            index.setdefault((int(xs[r]), int(ys[r])), []).append(self.ids[r])
        return index

    def positions_index(self) -> dict[tuple[int, int], list[str]]:
        """Mappa ``(x, y) -> [agent_id, ...]`` dei soli agenti vivi."""
        if self._pos_index_epoch != self._pos_epoch or self._pos_index is None:
            self._pos_index = self._build_positions_index()
            self._pos_index_epoch = self._pos_epoch
        elif _VERIFY_POSITION_INDEX:
            fresh = self._build_positions_index()
            if fresh != self._pos_index:
                raise AssertionError(
                    "indice posizioni non invalidato: una scrittura di x/y/alive "
                    "non ha chiamato touch_positions(). "
                    f"memorizzato={self._pos_index} ricostruito={fresh}"
                )
        return self._pos_index

    @classmethod
    def from_agents(cls, agents: dict, capacity_margin: float = 1.5) -> "AgentArrays":
        capacity = max(8, int(len(agents) * capacity_margin))
        aa = cls(capacity)
        for agent_id in agents:  # ordine di inserimento del dict: deterministico
            agent = agents[agent_id]
            row = aa.n
            aa.ids.append(agent_id)
            aa.index[agent_id] = row
            aa.x[row], aa.y[row] = agent.x, agent.y
            aa.home_x[row], aa.home_y[row] = agent.x, agent.y
            aa.alive[row] = agent.health > 0.0
            for name in _F32_FIELDS:
                src = _F32_SOURCE.get(name, name)
                getattr(aa, name)[row] = float(getattr(agent, src, 0.0))
            preferences = getattr(agent, "pillar_preferences", None)
            if preferences is not None:
                aa.pref[row] = np.asarray(preferences, dtype=np.float64)
            skills = getattr(agent, "pillar_skills", None)
            if skills is not None:
                aa.skill[row] = np.asarray(skills, dtype=np.float64)
            aa.steps_without_water[row] = int(getattr(agent, "steps_without_water", 0))
            aa.steps_without_food[row] = int(getattr(agent, "steps_without_food", 0))
            for res, value in agent.inventory.to_dict().items():
                aa.inv[row, C.R[res]] = value
            aa.n += 1
        aa.touch_positions()  # x/y/alive appena popolati per tutte le righe
        return aa

    def alive_rows(self) -> np.ndarray:
        return np.flatnonzero(self.alive[: self.n])

    def spawn(self, agent_id: str, x: int, y: int, inv_row: np.ndarray) -> int:
        if self.n >= self.x.shape[0]:
            self._grow()
        row = self.n
        self.ids.append(agent_id)
        self.index[agent_id] = row
        self.x[row], self.y[row] = x, y
        self.home_x[row], self.home_y[row] = x, y
        self.alive[row] = True
        self.health[row] = self.oxygen[row] = self.hydration[row] = self.satiety[row] = 1.0
        self.morale[row], self.stress[row] = 0.82, 0.15
        self.cooperation[row], self.compliance[row], self.autonomy[row] = 0.7, 0.85, 0.5
        self.inv[row] = inv_row
        self.pref[row] = 1.0 / N_PILLARS
        self.skill[row] = 1.0
        self.n += 1
        self.touch_positions()  # nuova riga viva in (x, y)
        return row

    def _grow(self) -> None:
        old = self.x.shape[0]
        new = old * 2
        for name, arr in list(vars(self).items()):
            if isinstance(arr, np.ndarray):
                shape = (new,) + arr.shape[1:]
                fill_value = _FILL.get(name, 0)
                grown = np.full(shape, fill_value, dtype=arr.dtype)
                grown[:old] = arr
                setattr(self, name, grown)


TERRAIN_ORDER: tuple[TerrainType, ...] = tuple(TerrainType)
_T_INDEX = {t: i for i, t in enumerate(TERRAIN_ORDER)}

_CELL_F32 = (  # (attributo CellArrays, attributo Cell)
    ("elevation", "elevation"), ("temp_mod", "local_temperature_modifier"),
    ("radiation", "radiation_level"), ("dust", "dust_level"),
    ("polar", "polar_severity"), ("water_ice", "water_ice"),
    ("liquid_water", "liquid_water"), ("habitability", "habitability_score"),
    ("pollution", "pollution_risk"), ("vegetation", "vegetation_biomass"),
    ("proto_soil", "proto_soil_development"), ("organic", "organic_matter"),
)


class CellArrays:
    def __init__(self, height: int, width: int):
        self.H, self.W = height, width
        self.terrain = np.zeros((height, width), dtype=np.int8)
        # Task 14 fix (equivalence audit, controller decision - exact
        # equivalence): float64, not float32 - see AgentArrays.__init__'s
        # matching note above.
        for name, _src in _CELL_F32:
            setattr(self, name, np.zeros((height, width), dtype=np.float64))
        # Baseline climatici per-cella, catturati UNA volta a worldgen
        # (Cell.baseline_*, world_generator.py:97-99) e poi SOLO letti dal
        # refresh periodico del layer planetario (planetary_coupling.py:281-283).
        # La facciata CellView deve esporli o quel percorso solleva
        # AttributeError (accade solo quando scatta un refresh di griglia, non
        # ai primi step - per questo era latente). NaN codifica il `None` del
        # modello a oggetti (fallback al valore vivo al momento della lettura).
        self.baseline_dust = np.full((height, width), np.nan, dtype=np.float64)
        self.baseline_radiation = np.full((height, width), np.nan, dtype=np.float64)
        self.baseline_temp = np.full((height, width), np.nan, dtype=np.float64)
        self.nut_n = np.zeros((height, width), dtype=np.float64)
        self.nut_p = np.zeros((height, width), dtype=np.float64)
        self.nut_c = np.zeros((height, width), dtype=np.float64)
        self.area_m2 = np.ones((height, width), dtype=np.float64)
        self.explored = np.zeros((height, width), dtype=np.bool_)
        # Task 12 (src/core/views.py CellView.is_spawn): set once at colony-
        # seeding time (src/world/initial_support.py:52), never reassigned
        # afterward - a plain bool column is sufficient (no setter needed on
        # the CellView side either, see its docstring).
        self.is_spawn = np.zeros((height, width), dtype=np.bool_)
        self.cell_res = np.zeros((height, width, C.NR), dtype=np.float64)
        self.struct_count = np.zeros((height, width, C.NS), dtype=np.int16)
        self.struct_integrity = np.zeros((height, width, C.NS), dtype=np.float64)
        self.site_progress = np.full((height, width, C.NS), -1.0, dtype=np.float64)
        self.struct_fx = np.zeros((height, width, C.NE), dtype=np.float64)
        self.occupancy = np.zeros((height, width), dtype=np.int16)
        # Bookkeeping for ConstructionSitesProxy (src/core/views.py): the exact
        # construction_sites key string (e.g. "greenhouse@12:34") per
        # (y, x, structure_index), plus backward-compatible custom keys that
        # are not a StructureType. These are plain dicts, not ndarrays, so they
        # are NOT part of the vectorized state. COLLECT_ICE is now atomic and
        # no longer creates the former "ice_extraction" custom key.
        self._site_keys: dict = {}
        self._site_extra: dict = {}
        # Sparse per-instance metadata.  Counts/integrity remain in the dense
        # SoA arrays used by the hot simulation path; only the comparatively
        # small number of built structures needs an individual sub-cell
        # position for geometry-aware actions and faithful map rendering.
        # Key: (y, x, structure_index), value: insertion-ordered
        # (local_x_m, local_y_m, owner) tuples.
        self._structure_instances: dict[
            tuple[int, int, int], list[tuple[float, float, str | None]]
        ] = {}
        # --- Structure view cache (perf, no behaviour change) ---------------
        # `CellView.structures` (src/core/views.py) rebuilds a fresh list of
        # StructureView wrappers on every access; the profile showed this
        # dominates (~68% self-time - StructureView materialization hammered by
        # `structure_saturated` during the decide phase, where nothing mutates
        # structures). Cache the materialized list per (y, x). Invalidated ONLY
        # where the arrays change OUTSIDE a StructureView setter:
        #   - WorldView.add_structure (struct_count++): `invalidate_structures_at`
        #   - kernel_biology's bulk wear (struct_integrity -= ...): `clear_structure_cache`
        # The StructureView.integrity setter keeps its OWN `_integrity_cache`
        # coherent with the array, so it deliberately does NOT invalidate:
        # re-materializing mid maintenance/wear loop would re-read a
        # partially-updated mean and break the integrity idempotence documented
        # on StructureView. `struct_count` is only ever incremented (structures
        # are never removed in the array engine), so there is no decrement path.
        # Values returned are bit-identical to the fresh path (same
        # `struct_integrity/count` division), so exact equivalence is preserved.
        self._struct_cache: dict = {}
        self._structure_positions_cache: tuple[tuple[int, int], ...] | None = None
        self._cache_stats = {
            "structure_view_hits": 0,
            "structure_view_misses": 0,
            "structure_empty_fast_paths": 0,
            "structure_position_hits": 0,
            "structure_position_misses": 0,
            "structure_invalidated_entries": 0,
        }

    def invalidate_structures_at(self, y: int, x: int) -> None:
        """Drop the cached StructureView list for one cell. Call after any
        change to struct_count/struct_integrity for (y, x) that is NOT routed
        through a StructureView setter (i.e. WorldView.add_structure)."""
        if self._struct_cache.pop((int(y), int(x)), None) is not None:
            self._cache_stats["structure_invalidated_entries"] += 1
        self._structure_positions_cache = None

    def structure_positions(self) -> tuple[tuple[int, int], ...]:
        """Return cached ``(x, y)`` coordinates with at least one structure."""
        if self._structure_positions_cache is None:
            self._cache_stats["structure_position_misses"] += 1
            ys_xs = np.argwhere(np.any(self.struct_count > 0, axis=2))
            self._structure_positions_cache = tuple(
                (int(x), int(y)) for y, x in ys_xs
            )
        else:
            self._cache_stats["structure_position_hits"] += 1
        return self._structure_positions_cache

    def clear_structure_cache(self) -> None:
        """Drop every cached StructureView list. Call after a BULK mutation of
        struct_integrity/struct_count (e.g. kernel_biology's vectorized wear)."""
        self._cache_stats["structure_invalidated_entries"] += len(
            self._struct_cache
        )
        self._struct_cache.clear()

    def cache_metrics(self) -> dict[str, float]:
        """Stable scalar telemetry for persisted performance diagnostics."""
        stats = {key: float(value) for key, value in self._cache_stats.items()}
        hits = stats["structure_view_hits"] + stats["structure_position_hits"]
        misses = stats["structure_view_misses"] + stats["structure_position_misses"]
        stats["structure_cache_hit_rate"] = hits / max(1.0, hits + misses)
        stats["structure_cached_cell_count"] = float(len(self._struct_cache))
        return stats

    @classmethod
    def from_world(cls, world) -> "CellArrays":
        height, width = len(world.cells), len(world.cells[0])
        ca = cls(height, width)
        for y, row in enumerate(world.cells):
            for x, cell in enumerate(row):
                ca.terrain[y, x] = _T_INDEX[cell.terrain]
                for name, src in _CELL_F32:
                    getattr(ca, name)[y, x] = float(getattr(cell, src, 0.0))
                ca.nut_n[y, x] = cell.nutrients.get("N", 0.0)
                ca.nut_p[y, x] = cell.nutrients.get("P", 0.0)
                ca.nut_c[y, x] = cell.nutrients.get("C", 0.0)
                # Baseline statici (None -> NaN, cioe' "usa il valore vivo").
                _bd = getattr(cell, "baseline_dust_level", None)
                _br = getattr(cell, "baseline_radiation_level", None)
                _bt = getattr(cell, "baseline_temperature_modifier", None)
                ca.baseline_dust[y, x] = np.nan if _bd is None else float(_bd)
                ca.baseline_radiation[y, x] = np.nan if _br is None else float(_br)
                ca.baseline_temp[y, x] = np.nan if _bt is None else float(_bt)
                ca.area_m2[y, x] = float(cell.geometry.get("area_m2", 1.0) or 1.0)
                ca.explored[y, x] = bool(cell.explored)
                ca.is_spawn[y, x] = bool(getattr(cell, "is_spawn", False))
                # Task 14 fix (equivalence audit): mirror `cell.agents_present`
                # (always live on the object engine - GridWorld.place_agent/
                # remove_agent keep it current as agents move) instead of
                # leaving `occupancy` zero-initialized. `kernel.step()` also
                # rebuilds this fully from the live alive set at the top of
                # every call (see its own docstring), so this seed mostly
                # matters for any read of `occupancy` that happens before the
                # first `step()` call ever runs against this `CellArrays`.
                ca.occupancy[y, x] = len(getattr(cell, "agents_present", ()) or ())
                for res, value in cell.resources.to_dict().items():
                    ca.cell_res[y, x, C.R[res]] = value
                for s in cell.structures:
                    si = C.S[s.type]
                    ca.struct_count[y, x, si] += 1
                    ca._structure_instances.setdefault((y, x, si), []).append(
                        (
                            float(getattr(s, "local_x_m", 0.0)),
                            float(getattr(s, "local_y_m", 0.0)),
                            getattr(s, "owner", None),
                        )
                    )
                    # SOMMA integrita' per tipo: la media (usata da recompute_struct_fx)
                    # e' calcolata a valle. ATTENZIONE (spec sez.1, sacrificio accettato):
                    # il modello a oggetti applica la soglia 0.4 per-struttura, l'aggregato
                    # la applica su media*count. I due valori divergono quando le
                    # integrita' individuali cavalcano la soglia 0.4 (es. 1.0 e 0.3:
                    # oggetti 1.0+0.15=1.15, aggregato eff(0.65)*2=1.3); coincidono quando
                    # tutte le integrita' del tipo stanno dalla stessa parte della soglia.
                    ca.struct_integrity[y, x, si] += float(s.integrity)
                for key, progress in cell.construction_sites.items():
                    st = _site_key_to_type(key)
                    if st is not None:
                        ca.site_progress[y, x, C.S[st]] = float(progress)
        ca.recompute_struct_fx()
        return ca

    def recompute_struct_fx(self, ys=None, xs=None) -> None:
        """Ricalcola struct_fx. Con ys/xs (array di indici o interi) ricalcola solo
        le celle indicate (es. dopo build/maintain/decay puntuali); senza argomenti
        ricalcola l'intera griglia."""
        if ys is None or xs is None:
            count = self.struct_count.astype(np.float64)
            integrity = self.struct_integrity
            mean_int = np.divide(integrity, count, out=np.zeros_like(integrity), where=count > 0)
            eff = np.where(mean_int > 0.4, mean_int, mean_int * 0.5) * count
            # [H,W,NS] @ [NS,NE] -> [H,W,NE]
            self.struct_fx = eff @ C.STRUCT_FX_M
        else:
            count = self.struct_count[ys, xs].astype(np.float64)
            integrity = self.struct_integrity[ys, xs]
            mean_int = np.divide(integrity, count, out=np.zeros_like(integrity), where=count > 0)
            eff = np.where(mean_int > 0.4, mean_int, mean_int * 0.5) * count
            self.struct_fx[ys, xs] = eff @ C.STRUCT_FX_M

    def structure_heat(self) -> np.ndarray:
        return self.struct_fx[:, :, C.E["temperature"]]


def _site_key_to_type(key: str):
    # Formato reale (src/agents/action_space.py:_construction_site_key):
    # f"{structure_type.value}@{round(local_x_m)}:{round(local_y_m)}", es. "greenhouse@2:3".
    # Alcune chiavi non rappresentano una struttura (es. "ice_extraction" per
    # l'estrazione ghiaccio in azione action_space.py:275): in quel caso non c'e'
    # "@" e StructureType(key) fallisce, quindi la chiave viene ignorata per
    # site_progress (nessuna colonna strutturale corrispondente).
    try:
        return StructureType(key.split("@")[0]) if "@" in key else StructureType(key)
    except ValueError:
        return None
