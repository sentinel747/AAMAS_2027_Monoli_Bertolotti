"""Read-write facades over the SoA arrays (`AgentArrays`/`CellArrays`) that
reproduce the JSON shape of `Cell.to_public_dict` (src/world/cell.py) and
`BaseAgent.to_dict` (src/agents/base_agent.py) field-for-field, so the GUI and
API routes stay byte-for-byte compatible while the hot simulation loop moves
to vectorized arrays.

Beyond reads, `AgentView`/`CellView` are write-through proxies: assigning
`agent.fatigue = ...`, `agent.inventory.ice += ...`, `cell.water_ice -= ...`,
opening/advancing/finishing `cell.construction_sites[...]`, and so on, all
land straight in the backing arrays. This is what lets
`src/agents/action_space.py` (`execute_action`, which mutates state in
place) and `src/simulation/extreme_events.py` run completely unmodified
against array-backed state instead of `BaseAgent`/`Cell` objects - see
`tests/core/test_views_writeback.py` for the proof (Task 7).

Cold, non-vectorized per-agent data (memory, recent actions, current goal,
actions_taken, local sub-cell offsets, static identity - name, role,
faction - and a handful of fields the arrays never gained a column for -
perception radius, execution mode, LLM provider/model) lives in
`AgentSideState`, one per agent, owned by the shell layer outside the hot
loop. Where `AgentSideState` does not carry a field yet, `AgentView.to_dict`
falls back to a type-correct neutral default via `getattr(..., default)` -
the moment a real value is wired into `AgentSideState` (Task 12) it starts
flowing through automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.agents.memory import AgentMemory
from src.core import constants as C
from src.core.arrays import (
    AgentArrays,
    CellArrays,
    TERRAIN_ORDER,
    _MISSION_NONE,
    MISSION_SCOUT_BACK,
    MISSION_SCOUT_OUT,
    MISSION_SETTLE_OUT,
    _POSITION_INDEX_ENABLED,
    _site_key_to_type,
)
from src.core.kernel_biology import recompute_habitability
from src.world.cell import cell_degradation_enabled
from src.world.colony_site import colony_site_score
from src.world.mars_geometry import cell_geometry
from src.world.structures import StructureType
from src.world.occupancy import occupancy_capacity_from_housing
from src.world.terrain import TERRAIN_TRAVERSAL_RISK, terrain_hazard_tier, traversal_risk_for_cell


@dataclass
class AgentSideState:
    """Cold per-agent state that is not vectorized in `AgentArrays`."""

    memory: AgentMemory = field(default_factory=AgentMemory)
    recent_actions: list = field(default_factory=list)
    current_goal: str = "survive and improve local habitability"
    actions_taken: int = 0
    # Sub-cell offsets: not vectorized (see module docstring), but
    # action_space.py's BUILD/MOVE actions read/write them unconditionally,
    # so they are wired into AgentSideState here rather than left as a
    # getattr-with-default fallback that only `to_dict` sees.
    local_x_m: float = 0.0
    local_y_m: float = 0.0
    # Static identity fields: action_space.py's `apply_environmental_entry`
    # (hazardous MOVE/EXPLORE entry, risk >= 0.75) reads `agent.name`
    # unconditionally to log the event - AgentView had no such property and
    # raised AttributeError. role/faction are wired in for symmetry (both
    # already flow through `to_dict`'s getattr fallback).
    name: str = ""
    role: str = ""
    faction: str | None = None
    # Task 9 (src/core/kernel.py): rule_based_agent.py's decide() reads both
    # of these unconditionally (perception_radius for the scouting neighbor
    # radius, survival_priority for hazard/recovery scoring) - neither is a
    # vectorized AgentArrays column nor did AgentView expose a property for
    # either, so an AgentView driving decide() would have silently fallen
    # through Python's MRO to BaseAgent's dataclass CLASS-level default
    # (perception_radius=3, survival_priority=0.8) for every agent instead of
    # its real per-agent value. Both vary per real agent: population.py seeds
    # every spawned agent at perception_radius=1 (not the dataclass default
    # 3), and randomizes survival_priority per agent
    # (`rng.gauss(agent.survival_priority, 0.10)` clipped to [0.45, 0.98]).
    # Neither field is ever reassigned after spawn (grep confirms only
    # population.py's spawn-time write), so a read-only property here is
    # sufficient - decide()/execute_action() never set either one.
    perception_radius: int = 3
    survival_priority: float = 0.8
    # Task 12 (src/api/state_store.py): wired for GUI display fidelity, not
    # for decision logic - the kernel always drives every agent through the
    # rule-based path (see kernel.py's module docstring, "Deliberately out of
    # scope: the LLM-governor branch"), regardless of what `mode` reports.
    # Kept as separate runtime columns for existing consumers, but populated
    # from the same user-facing operational range.
    mode: str = "rule_based"
    llm_provider_id: str | None = None
    llm_model: str | None = None
    perception_radius_m: float = 59_000.0
    movement_distance_m_per_step: float = 59_000.0
    # Cold mission-preparation flag: it changes only the automatic warehouse
    # deposit mask and therefore does not justify another hot SoA column.
    founder_kit_reserved: bool = False

    @classmethod
    def from_agent(cls, agent) -> "AgentSideState":
        return cls(
            memory=agent.memory,
            recent_actions=list(agent.recent_actions),
            current_goal=agent.current_goal,
            actions_taken=int(getattr(agent, "actions_taken", 0)),
            local_x_m=float(getattr(agent, "local_x_m", 0.0)),
            local_y_m=float(getattr(agent, "local_y_m", 0.0)),
            name=str(getattr(agent, "name", "")),
            role=str(getattr(agent, "role", "")),
            faction=getattr(agent, "faction", None),
            perception_radius=int(getattr(agent, "perception_radius", 3)),
            survival_priority=float(getattr(agent, "survival_priority", 0.8)),
            mode=str(getattr(agent, "mode", "rule_based")),
            llm_provider_id=getattr(agent, "llm_provider_id", None),
            llm_model=getattr(agent, "llm_model", None),
            perception_radius_m=float(getattr(agent, "perception_radius_m", 59_000.0)),
            movement_distance_m_per_step=float(
                getattr(agent, "movement_distance_m_per_step", 59_000.0)
            ),
            founder_kit_reserved=bool(
                getattr(agent, "founder_kit_reserved", False)
            ),
        )


class ResourceProxy:
    """Write-through view of one resource bundle row/vector.

    Same field set and order as `ResourceBundle` (`src/world/resources.py`),
    same `add`/`remove`/`can_afford` semantics (resources.py:20-39), but reads
    and writes go straight into the backing `ndarray` row
    (`AgentArrays.inv[row]` or `CellArrays.cell_res[y, x]`) instead of a
    dataclass instance, so mutations like `agent.inventory.ice += amount`
    (src/agents/action_space.py) land directly in the array.
    """

    __slots__ = ("_row",)

    def __init__(self, row: np.ndarray):
        object.__setattr__(self, "_row", row)

    def __getattr__(self, name: str) -> float:
        idx = C.R.get(name)
        if idx is None:
            raise AttributeError(name)
        return float(self._row[idx])

    def __setattr__(self, name: str, value: float) -> None:
        idx = C.R.get(name)
        if idx is None:
            raise AttributeError(name)
        self._row[idx] = value

    def to_dict(self) -> dict[str, float]:
        return {name: float(self._row[idx]) for name, idx in C.R.items()}

    def add(self, other) -> None:
        for name, idx in C.R.items():
            self._row[idx] = float(self._row[idx]) + float(getattr(other, name))

    def remove(self, other) -> bool:
        if not self.can_afford(other):
            return False
        for name, idx in C.R.items():
            self._row[idx] = float(self._row[idx]) - float(getattr(other, name))
        return True

    def can_afford(self, cost) -> bool:
        return all(float(self._row[idx]) >= float(getattr(cost, name)) for name, idx in C.R.items())


class NutrientsProxy:
    """Dict-like write-through view of one cell's N/P/C nutrient pools
    (`CellArrays.nut_n/nut_p/nut_c`)."""

    _ATTRS = {"N": "nut_n", "P": "nut_p", "C": "nut_c"}

    def __init__(self, cells: CellArrays, y: int, x: int):
        self._c, self.y, self.x = cells, y, x

    def __getitem__(self, key: str) -> float:
        return float(getattr(self._c, self._ATTRS[key])[self.y, self.x])

    def __setitem__(self, key: str, value: float) -> None:
        getattr(self._c, self._ATTRS[key])[self.y, self.x] = float(value)

    def __contains__(self, key: str) -> bool:
        return key in self._ATTRS

    def __iter__(self):
        return iter(self._ATTRS)

    def __len__(self) -> int:
        return len(self._ATTRS)

    def keys(self):
        return list(self._ATTRS)

    def items(self):
        return [(k, self[k]) for k in self._ATTRS]

    def to_dict(self) -> dict[str, float]:
        return {k: self[k] for k in self._ATTRS}


def _site_keys_store(cells: CellArrays) -> dict:
    # Bookkeeping dict owned by the (persistent) CellArrays instance
    # (`CellArrays.__init__`): site_progress only holds one float per (cell,
    # structure type), so the exact key string used to open a site (which
    # encodes the sub-cell position, e.g. "greenhouse@12:34") is cached here
    # for continuation lookups (`_nearest_construction_site_key` in
    # action_space.py) to find the same key again across calls. CellView
    # instances are recreated on every `get_cell`, so this cannot live there.
    # NOT part of the vectorized array state - see the TODO on
    # `CellArrays.__init__` for the rebuild-drops-it consequence. The getattr
    # fallback below only guards against a CellArrays built before that field
    # existed (e.g. an unpickled instance from an older run).
    store = getattr(cells, "_site_keys", None)
    if store is None:
        store = {}
        cells._site_keys = store
    return store


def _site_extra_store(cells: CellArrays) -> dict:
    # Compatibility storage for legacy/custom construction_sites keys that do
    # not encode a real structure type.  COLLECT_ICE no longer writes its
    # former "ice_extraction" pseudo-site here, but old in-memory worlds can
    # still expose and clear it safely.
    store = getattr(cells, "_site_extra", None)
    if store is None:
        store = {}
        cells._site_extra = store
    return store


class ConstructionSitesProxy:
    """dict-like write-through view of one cell's `construction_sites`
    (`cell.construction_sites[key] = ...`, `del cell.construction_sites[key]`,
    `key in cell.construction_sites`, iteration - all used by
    src/agents/action_space.py's BUILD_* actions; non-structural keys remain
    supported for backward compatibility)."""

    def __init__(self, cells: CellArrays, y: int, x: int):
        self._c, self.y, self.x = cells, y, x

    def _extra(self) -> dict:
        """La sacca dei siti non strutturali di questa cella, in SOLA LETTURA.

        **Leggere non deve creare stato (2026-08-28).** Prima questo accessore
        era un `setdefault`, quindi bastava *interrogare* `construction_sites`
        — cosa che il salvataggio degli artifact fa per ogni cella, per
        comporre `cell_infrastructure.json` — perche' comparisse una voce vuota
        nel registro. Effetto misurato: l'impronta dello stato cambiava fra una
        run salvata e la stessa run non salvata, sul solo campo
        `cells.site_extra`, e ogni verifica basata sul digest ne risultava
        invalidata. Il comportamento della simulazione non cambiava — le voci
        create sono dizionari vuoti — ma un percorso di sola osservazione che
        muta cio' che osserva rende inutilizzabile l'oracolo che il progetto usa
        per tutto il resto.

        Il dizionario restituito quando la voce esiste E' quello memorizzato,
        quindi la cancellazione attraverso di esso continua a funzionare; quando
        non esiste si restituisce un vuoto usa-e-getta, e `__delitem__` solleva
        `KeyError` esattamente come prima.
        """
        return _site_extra_store(self._c).get((self.y, self.x), {})

    def _extra_scrivibile(self) -> dict:
        """La stessa sacca, ma creandola: la usa solo chi scrive davvero."""
        return _site_extra_store(self._c).setdefault((self.y, self.x), {})

    def _si(self, key: str) -> int | None:
        st = _site_key_to_type(key)
        return None if st is None else C.S[st]

    def __contains__(self, key: str) -> bool:
        si = self._si(key)
        if si is not None:
            return bool(self._c.site_progress[self.y, self.x, si] >= 0.0)
        return key in self._extra()

    def __getitem__(self, key: str) -> float:
        si = self._si(key)
        if si is not None:
            v = float(self._c.site_progress[self.y, self.x, si])
            if v < 0.0:
                raise KeyError(key)
            return v
        extra = self._extra()
        if key not in extra:
            raise KeyError(key)
        return extra[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: str, value: float) -> None:
        si = self._si(key)
        if si is not None:
            self._c.site_progress[self.y, self.x, si] = float(value)
            _site_keys_store(self._c)[(self.y, self.x, si)] = key
        else:
            self._extra_scrivibile()[key] = float(value)

    def __delitem__(self, key: str) -> None:
        si = self._si(key)
        if si is not None:
            if self._c.site_progress[self.y, self.x, si] < 0.0:
                raise KeyError(key)
            self._c.site_progress[self.y, self.x, si] = -1.0
            _site_keys_store(self._c).pop((self.y, self.x, si), None)
        else:
            del self._extra()[key]

    def __iter__(self):
        keys_store = _site_keys_store(self._c)
        for si, st in enumerate(C.STRUCTURES):
            if self._c.site_progress[self.y, self.x, si] >= 0.0:
                yield keys_store.get((self.y, self.x, si), f"{st.value}@0:0")
        yield from self._extra()

    def __len__(self) -> int:
        n_struct = int(np.count_nonzero(self._c.site_progress[self.y, self.x] >= 0.0))
        return n_struct + len(self._extra())

    def keys(self):
        return list(self)

    def items(self):
        return [(k, self[k]) for k in self]


# Precomputed nonzero (effect_name, base_value) pairs per structure index.
# `StructureView.local_effect` used to rescan every effect column (C.E.items())
# and re-index C.STRUCT_FX_M on each call - ~590k calls per 200-agent step, the
# top facade hotspot after the structures-list cache. base_value is the exact
# float the inline expression produced (`float(C.STRUCT_FX_M[si, idx])`), and
# the key order matches C.E.items(), so the resulting dict is bit-identical and
# exact equivalence is preserved.
_STRUCT_FX_NONZERO = tuple(
    tuple((name, float(C.STRUCT_FX_M[si, idx])) for name, idx in C.E.items() if C.STRUCT_FX_M[si, idx])
    for si in range(C.NS)
)


class StructureView:
    """One live structure of a given type in a cell; unlike the plain
    `Structure` dataclass this writes `.integrity` straight back into
    `CellArrays.struct_integrity` (used by action_space.py's
    MAINTAIN_STRUCTURE: `for s in cell.structures: s.integrity += ...`).

    Individual per-structure integrities are not tracked - only the
    per-type sum is (see `CellArrays.from_world`'s "sacrifice accepted"
    note) - so every one of the `count` structures of a type shares the
    same mean integrity, exactly like the pre-existing read-only view did.

    Because of that sharing, a getter that re-reads the mean live would
    compound across a loop such as action_space.py's MAINTAIN_STRUCTURE
    (`for s in cell.structures: s.integrity = min(1.0, s.integrity + 0.4)`)
    or extreme_events.py's wear pass: `CellView.structures` builds every
    `StructureView` of a type from the SAME starting mean in one call, but
    once iteration 1 writes the new mean back, iteration 2's *live* getter
    would read the already-updated value and apply the delta twice - e.g.
    two 0.5 shelters repaired by +0.4 give the object engine 0.9/0.9 (mean
    0.9) but would give a live-reading array engine 0.9 then 1.0 (mean 1.0).
    Caching the integrity at construction time fixes this: every view built
    from the same `cell.structures` call reads the same starting mean, so
    each independently computes the same `f(mean)` and writes the same
    `count * f(mean)` back - idempotent, and matching the object engine
    where every structure's setter also starts from the same pre-loop value.
    """

    def __init__(
        self,
        cells: CellArrays,
        y: int,
        x: int,
        si: int,
        owner: str | None = None,
        local_x_m: float = 0.0,
        local_y_m: float = 0.0,
    ):
        self._c, self.y, self.x, self._si = cells, y, x, si
        self.type = C.STRUCTURES[si]
        self.local_x_m = float(local_x_m)
        self.local_y_m = float(local_y_m)
        self.owner = owner
        self._integrity_cache = self._read_mean_integrity()

    def _read_mean_integrity(self) -> float:
        count = int(self._c.struct_count[self.y, self.x, self._si])
        if count <= 0:
            return 0.0
        return float(self._c.struct_integrity[self.y, self.x, self._si]) / count

    @property
    def integrity(self) -> float:
        return self._integrity_cache

    @integrity.setter
    def integrity(self, value: float) -> None:
        count = int(self._c.struct_count[self.y, self.x, self._si])
        if count <= 0:
            return
        self._c.struct_integrity[self.y, self.x, self._si] = float(value) * count
        self._integrity_cache = float(value)
        self._c.recompute_struct_fx(self.y, self.x)

    @property
    def efficiency(self) -> float:
        integrity = self.integrity
        if integrity <= 0:
            return 0.0
        return integrity if integrity > 0.4 else integrity * 0.5

    @property
    def local_effect(self) -> dict[str, float]:
        eff = self.efficiency
        return {name: base * eff for name, base in _STRUCT_FX_NONZERO[self._si]}

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "x": self.x,
            "y": self.y,
            "local_x_m": float(self.local_x_m),
            "local_y_m": float(self.local_y_m),
            "owner": self.owner,
            "integrity": float(self.integrity),
            "efficiency": float(self.efficiency),
        }


class CellView:
    """Read-write facade over one cell backed by `CellArrays`. Reads pull
    straight from the arrays; the setters below write straight back into
    them, so mutations from src/agents/action_space.py and
    src/simulation/extreme_events.py (`cell.water_ice -= ...`,
    `cell.vegetation_biomass = ...`, ...) land in the array state."""

    def __init__(
        self,
        cells: CellArrays,
        x: int,
        y: int,
        planetary_state: dict,
        agents: AgentArrays | None = None,
        side_states: dict[str, AgentSideState] | None = None,
    ):
        self._c, self.x, self.y, self._ps, self._agents, self._side_states = (
            cells,
            int(x),
            int(y),
            planetary_state,
            agents,
            side_states,
        )

    # -- terrain / hazard -------------------------------------------------
    @property
    def terrain(self):
        return TERRAIN_ORDER[int(self._c.terrain[self.y, self.x])]

    @property
    def elevation(self) -> float:
        return float(self._c.elevation[self.y, self.x])

    @property
    def local_temperature_modifier(self) -> float:
        return float(self._c.temp_mod[self.y, self.x])

    @local_temperature_modifier.setter
    def local_temperature_modifier(self, value: float) -> None:
        self._c.temp_mod[self.y, self.x] = value

    @property
    def polar_severity(self) -> float:
        return float(self._c.polar[self.y, self.x])

    @property
    def radiation_level(self) -> float:
        return float(self._c.radiation[self.y, self.x])

    @radiation_level.setter
    def radiation_level(self, value: float) -> None:
        self._c.radiation[self.y, self.x] = value

    @property
    def dust_level(self) -> float:
        return float(self._c.dust[self.y, self.x])

    @dust_level.setter
    def dust_level(self, value: float) -> None:
        self._c.dust[self.y, self.x] = value

    # Baseline climatici statici (catturati a worldgen, letti dal refresh del
    # layer planetario). Read-only: nulla li riscrive dopo la generazione. NaN
    # nel array == None sul modello a oggetti (planetary_coupling.py li tratta
    # con `... if baseline is not None else valore_vivo`).
    @property
    def baseline_dust_level(self):
        v = float(self._c.baseline_dust[self.y, self.x])
        return None if np.isnan(v) else v

    @property
    def baseline_radiation_level(self):
        v = float(self._c.baseline_radiation[self.y, self.x])
        return None if np.isnan(v) else v

    @property
    def baseline_temperature_modifier(self):
        v = float(self._c.baseline_temp[self.y, self.x])
        return None if np.isnan(v) else v

    @property
    def water_ice(self) -> float:
        return float(self._c.water_ice[self.y, self.x])

    @water_ice.setter
    def water_ice(self, value: float) -> None:
        self._c.water_ice[self.y, self.x] = value

    @property
    def liquid_water(self) -> float:
        return float(self._c.liquid_water[self.y, self.x])

    @liquid_water.setter
    def liquid_water(self, value: float) -> None:
        self._c.liquid_water[self.y, self.x] = value

    @property
    def habitability_score(self) -> float:
        return float(self._c.habitability[self.y, self.x])

    @habitability_score.setter
    def habitability_score(self, value: float) -> None:
        self._c.habitability[self.y, self.x] = value

    @property
    def vegetation_biomass(self) -> float:
        return float(self._c.vegetation[self.y, self.x])

    @vegetation_biomass.setter
    def vegetation_biomass(self, value: float) -> None:
        self._c.vegetation[self.y, self.x] = value

    @property
    def proto_soil_development(self) -> float:
        return float(self._c.proto_soil[self.y, self.x])

    @proto_soil_development.setter
    def proto_soil_development(self, value: float) -> None:
        self._c.proto_soil[self.y, self.x] = value

    @property
    def organic_matter(self) -> float:
        return float(self._c.organic[self.y, self.x])

    @organic_matter.setter
    def organic_matter(self, value: float) -> None:
        self._c.organic[self.y, self.x] = value

    @property
    def pollution_risk(self) -> float:
        return float(self._c.pollution[self.y, self.x])

    @pollution_risk.setter
    def pollution_risk(self, value: float) -> None:
        self._c.pollution[self.y, self.x] = value

    @property
    def nutrients(self) -> "NutrientsProxy":
        return NutrientsProxy(self._c, self.y, self.x)

    @property
    def explored(self) -> bool:
        return bool(self._c.explored[self.y, self.x])

    @explored.setter
    def explored(self, value: bool) -> None:
        self._c.explored[self.y, self.x] = bool(value)

    @property
    def resources(self) -> ResourceProxy:
        return ResourceProxy(self._c.cell_res[self.y, self.x])

    # -- fields CellArrays never gained a column for (Task 12 wires these) -
    @property
    def is_spawn(self) -> bool:
        # Real per-cell flag (CellArrays.is_spawn), populated at
        # `CellArrays.from_world` from the object model's `cell.is_spawn` -
        # set once, at colony-seeding time (src/world/initial_support.py:52),
        # never reassigned afterward (grep confirms the only write site), so
        # a read-only property is sufficient here too.
        return bool(self._c.is_spawn[self.y, self.x])

    @property
    def geometry(self) -> dict:
        # `area_m2` is the one field CellArrays actually stores as a live
        # array column (src/core/kernel_biology.py:221 reads it directly for
        # pollution density) - read it from there so this stays consistent
        # with what the biology kernel sees. Every other geometry field is a
        # PURE function of (x, y, W, H) - src/world/mars_geometry.py's
        # cell_geometry, the exact same helper world_generator.py calls once
        # per cell at world-gen time to build the object model's
        # cell.geometry dict (src/world/world_generator.py:88-96) - so
        # recomputing it here on read is exact, not an approximation, and
        # needs no extra stored/vectorized state. Task 12: this closes the
        # up-to-0.16-weighted `colony_site_score` divergence from
        # `center_lat_deg` being absent (src/world/colony_site.py:20,113-114)
        # plus the `area_km2` term (:23).
        geom = cell_geometry(self.x, self.y, self._c.W, self._c.H)
        return {
            "center_lat_deg": geom.center_lat_deg,
            "center_lon_deg": geom.center_lon_deg,
            "width_m": geom.width_m,
            "height_m": geom.height_m,
            "area_m2": float(self._c.area_m2[self.y, self.x]),
            "area_km2": geom.area_km2,
        }

    @property
    def faction_influence(self) -> dict:
        return {}

    @property
    def agent_positions_m(self) -> dict:
        if self._agents is None or self._side_states is None:
            return {}
        return {
            agent_id: {
                "x": float(self._side_states[agent_id].local_x_m),
                "y": float(self._side_states[agent_id].local_y_m),
            }
            for agent_id in self.agents_present
            if agent_id in self._side_states
        }

    @property
    def construction_sites(self) -> "ConstructionSitesProxy":
        # CellArrays.site_progress only keeps one progress value per structure
        # type per cell (-1.0 == no site); the object model's key also encodes
        # the sub-cell local_x_m/local_y_m of the specific site, which is lost
        # once aggregated (multiple sites of the same type in this cell share
        # one progress value - see CellArrays.from_world's "sacrifice
        # accepted" note). ConstructionSitesProxy is a dict-like write-through
        # facade so action_space.py's `cell.construction_sites[key] = ...`,
        # `del cell.construction_sites[key]` and `key in cell.construction_sites`
        # all work unmodified.
        return ConstructionSitesProxy(self._c, self.y, self.x)

    @property
    def structures(self) -> list["StructureView"]:
        # Cached on CellArrays per (y, x): `get_cell` builds a fresh CellView
        # each call, so the cache MUST live on the persistent array state, not
        # on this transient facade. Invalidation is handled at the two mutation
        # sites that bypass a StructureView setter (add_structure, bulk wear);
        # see CellArrays._struct_cache. Returned values are bit-identical to a
        # fresh materialization, so exact equivalence is preserved.
        key = (self.y, self.x)
        if not np.any(self._c.struct_count[self.y, self.x]):
            # Empty cells dominate a planetary grid. Returning immediately
            # avoids filling the persistent cache with tens of thousands of
            # empty lists that bulk structure wear would invalidate each step.
            self._c._cache_stats["structure_empty_fast_paths"] += 1
            return []
        cached = self._c._struct_cache.get(key)
        if cached is not None:
            self._c._cache_stats["structure_view_hits"] += 1
            return cached
        self._c._cache_stats["structure_view_misses"] += 1
        out: list[StructureView] = []
        for si in range(C.NS):
            count = int(self._c.struct_count[self.y, self.x, si])
            instances = self._c._structure_instances.get((self.y, self.x, si), ())
            for index in range(count):
                if index < len(instances):
                    local_x_m, local_y_m, owner = instances[index]
                else:
                    # Defensive compatibility for direct/manual count-array
                    # mutations: a stable low-discrepancy point is preferable
                    # to stacking every unknown instance at (0, 0).
                    geom = self.geometry
                    local_x_m = ((index + 1) * 0.61803398875 % 1.0) * float(geom["width_m"])
                    local_y_m = ((index + 1) * 0.75487766625 % 1.0) * float(geom["height_m"])
                    owner = None
                out.append(
                    StructureView(
                        self._c,
                        self.y,
                        self.x,
                        si,
                        owner=owner,
                        local_x_m=local_x_m,
                        local_y_m=local_y_m,
                    )
                )
        self._c._struct_cache[key] = out
        return out

    def structure_counts(self) -> dict:
        """Conteggi per tipo letti dalla colonna, senza materializzare le viste.

        `C.STRUCTURES` e' `tuple(StructureType)`, quindi l'indice di colonna
        copre ogni tipo e il dizionario prodotto qui e' esattamente quello che
        il ciclo su `structures` costruirebbe -- ma in O(NS) invece che in
        O(numero di strutture), che nella cella di colonia sono ~180. Vedi
        `src/agents/build_policy.py::structure_counts` per la misura che ha
        motivato questa via.
        """
        row = self._c.struct_count[self.y, self.x]
        return {C.STRUCTURES[si]: int(row[si]) for si in range(C.NS)}

    @property
    def agents_present(self) -> list[str]:
        # Task 0b (piano di migrazione Rust): questa lettura era una scansione
        # O(n_vivi) ripetuta 255.531 volte in 100 step, quasi sempre per un
        # semplice `len(...)`. Ora consulta l'indice per-cella mantenuto da
        # `AgentArrays.positions_index()`, invalidato a epoca dai punti di
        # scrittura di x/y/alive.
        #
        # La lista restituita e' CONDIVISA con l'indice e non va mutata. Tutti i
        # chiamanti attuali sono in sola lettura (`len`, iterazione, test di
        # verita' - grep su src/ e tests/), quindi non viene copiata: copiarla
        # reintrodurrebbe un costo per-chiamata proprio dove lo si e' tolto.
        if self._agents is None:
            # No AgentArrays reference (standalone CellView): neutral default.
            return []
        if _POSITION_INDEX_ENABLED:
            return self._agents.positions_index().get((self.x, self.y), [])
        rows = self._agents.alive_rows()
        if rows.size == 0:
            return []
        ax = self._agents.x[rows]
        ay = self._agents.y[rows]
        mask = (ax == self.x) & (ay == self.y)
        return [self._agents.ids[int(i)] for i in rows[mask]]

    def occupancy_capacity(self) -> float:
        counts = self._c.struct_count[self.y, self.x]
        return occupancy_capacity_from_housing(
            counts[C.S[StructureType.SHELTER]],
            counts[C.S[StructureType.HABITAT]],
            counts[C.S[StructureType.INFIRMARY]],
        )

    def overcrowding_excess(self) -> float:
        return max(
            0.0,
            float(self._c.occupancy[self.y, self.x]) - self.occupancy_capacity(),
        )

    def structure_heat(self) -> float:
        return float(self._c.struct_fx[self.y, self.x, C.E["temperature"]])

    def structure_food_effect(self) -> float:
        return float(self._c.struct_fx[self.y, self.x, C.E["food"]])

    def structure_water_effect(self) -> float:
        return float(self._c.struct_fx[self.y, self.x, C.E["water"]])

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
            "agent_positions_m": {agent_id: dict(pos) for agent_id, pos in self.agent_positions_m.items()},
            "geometry": dict(self.geometry),
            "explored": self.explored,
            "faction_influence": dict(self.faction_influence),
            "pollution_risk": self.pollution_risk,
            "occupancy_capacity": self.occupancy_capacity(),
            "overcrowding_excess": self.overcrowding_excess(),
            "colony_site_score": colony_site_score(self),
        }


class AgentView:
    """Read-write facade over one agent row backed by `AgentArrays` plus its
    cold `AgentSideState` counterpart. Reads pull straight from the arrays
    (or the side state for cold fields); the setters below write straight
    back, so mutations from src/agents/action_space.py, rule_based_agent.py
    and src/simulation/extreme_events.py (`agent.fatigue = min(1.0, ...)`,
    `agent.inventory.ice += amount`, ...) land in the array state."""

    def __init__(self, agents: AgentArrays, row: int, side_state: AgentSideState):
        self._a, self.row, self._side = agents, int(row), side_state

    @property
    def agent_id(self) -> str:
        return self._a.ids[self.row]

    # -- static identity (AgentSideState; see its docstring) ---------------
    # Read-only, like agent_id: action_space.py/rule_based_agent.py/
    # extreme_events.py only ever read these (e.g. `apply_environmental_entry`
    # logging `agent.name` on hazardous MOVE/EXPLORE entry), never assign them.
    @property
    def name(self) -> str:
        return self._side.name

    @property
    def role(self) -> str:
        return self._side.role

    @property
    def faction(self) -> str | None:
        return self._side.faction

    # -- read-only, side-state-backed traits (see AgentSideState docstring) -
    # Read-only for the same reason as name/role/faction above: nothing in
    # rule_based_agent.py or action_space.py ever assigns either one.
    @property
    def perception_radius(self) -> int:
        return int(self._side.perception_radius)

    @property
    def survival_priority(self) -> float:
        return float(self._side.survival_priority)

    # -- Task 12: read-only, side-state-backed (see AgentSideState docstring) -
    # Needed by `src/simulation/action_logging.py`'s `build_action_log_row`/
    # `src/simulation/run_artifacts.py`'s `build_replay_event`, both of which
    # read `getattr(agent, "mode"/"perception_radius_m", ...)` on the AGENT
    # object itself (not on a side-state reference) - without a property here
    # every call through an `AgentView` would silently fall back to the
    # getattr default instead of the real per-agent value.
    @property
    def mode(self) -> str:
        return self._side.mode

    @property
    def llm_provider_id(self) -> str | None:
        return self._side.llm_provider_id

    @property
    def llm_model(self) -> str | None:
        return self._side.llm_model

    @property
    def perception_radius_m(self) -> float:
        return float(self._side.perception_radius_m)

    @property
    def movement_distance_m_per_step(self) -> float:
        return float(self._side.movement_distance_m_per_step)

    @property
    def x(self) -> int:
        return int(self._a.x[self.row])

    @x.setter
    def x(self, value: int) -> None:
        self._a.x[self.row] = int(value)
        self._a.touch_positions()

    @property
    def y(self) -> int:
        return int(self._a.y[self.row])

    @y.setter
    def y(self, value: int) -> None:
        self._a.y[self.row] = int(value)
        self._a.touch_positions()

    @property
    def inventory(self) -> ResourceProxy:
        return ResourceProxy(self._a.inv[self.row])

    # -- vitals / decision traits (AgentArrays._F32_FIELDS) ---------------
    @property
    def health(self) -> float:
        return float(self._a.health[self.row])

    @health.setter
    def health(self, value: float) -> None:
        self._a.health[self.row] = value

    @property
    def satiety(self) -> float:
        return float(self._a.satiety[self.row])

    @satiety.setter
    def satiety(self, value: float) -> None:
        self._a.satiety[self.row] = value

    @property
    def oxygen_level(self) -> float:
        return float(self._a.oxygen[self.row])

    @oxygen_level.setter
    def oxygen_level(self, value: float) -> None:
        self._a.oxygen[self.row] = value

    @property
    def hydration(self) -> float:
        return float(self._a.hydration[self.row])

    @hydration.setter
    def hydration(self, value: float) -> None:
        self._a.hydration[self.row] = value

    @property
    def fatigue(self) -> float:
        return float(self._a.fatigue[self.row])

    @fatigue.setter
    def fatigue(self, value: float) -> None:
        self._a.fatigue[self.row] = value

    @property
    def stress_index(self) -> float:
        return float(self._a.stress[self.row])

    @stress_index.setter
    def stress_index(self, value: float) -> None:
        self._a.stress[self.row] = value

    @property
    def morale(self) -> float:
        return float(self._a.morale[self.row])

    @morale.setter
    def morale(self, value: float) -> None:
        self._a.morale[self.row] = value

    @property
    def cooperation(self) -> float:
        return float(self._a.cooperation[self.row])

    @cooperation.setter
    def cooperation(self, value: float) -> None:
        self._a.cooperation[self.row] = value

    @property
    def protocol_compliance(self) -> float:
        return float(self._a.compliance[self.row])

    @protocol_compliance.setter
    def protocol_compliance(self, value: float) -> None:
        self._a.compliance[self.row] = value

    @property
    def autonomy_preference(self) -> float:
        return float(self._a.autonomy[self.row])

    @autonomy_preference.setter
    def autonomy_preference(self, value: float) -> None:
        self._a.autonomy[self.row] = value

    @property
    def risk_tolerance(self) -> float:
        return float(self._a.risk_tolerance[self.row])

    @risk_tolerance.setter
    def risk_tolerance(self, value: float) -> None:
        self._a.risk_tolerance[self.row] = value

    @property
    def curiosity(self) -> float:
        return float(self._a.curiosity[self.row])

    @curiosity.setter
    def curiosity(self, value: float) -> None:
        self._a.curiosity[self.row] = value

    @property
    def pillar_preferences(self) -> np.ndarray:
        """Live view of this agent's normalized, lifetime-stable preferences."""
        return self._a.pref[self.row]

    @property
    def pillar_skills(self) -> np.ndarray:
        """Live view of the role-shaped six-pillar capability multipliers."""
        return self._a.skill[self.row]

    @property
    def steps_without_water(self) -> int:
        return int(self._a.steps_without_water[self.row])

    @steps_without_water.setter
    def steps_without_water(self, value: int) -> None:
        self._a.steps_without_water[self.row] = int(value)

    @property
    def steps_without_food(self) -> int:
        return int(self._a.steps_without_food[self.row])

    @steps_without_food.setter
    def steps_without_food(self, value: int) -> None:
        self._a.steps_without_food[self.row] = int(value)

    # -- sub-cell offsets (see AgentSideState docstring) -------------------
    @property
    def local_x_m(self) -> float:
        return float(getattr(self._side, "local_x_m", 0.0))

    @local_x_m.setter
    def local_x_m(self, value: float) -> None:
        self._side.local_x_m = float(value)

    @property
    def local_y_m(self) -> float:
        return float(getattr(self._side, "local_y_m", 0.0))

    @local_y_m.setter
    def local_y_m(self, value: float) -> None:
        self._side.local_y_m = float(value)

    # -- cold state (AgentSideState) ---------------------------------------
    @property
    def memory(self) -> AgentMemory:
        return self._side.memory

    @property
    def recent_actions(self) -> list:
        # Live list (not a copy): `agent.recent_actions.append(...)` and
        # `del agent.recent_actions[:-60]` in action_space.py must mutate the
        # actual side-state list.
        return self._side.recent_actions

    @property
    def current_goal(self) -> str:
        return self._side.current_goal

    @current_goal.setter
    def current_goal(self, value: str) -> None:
        self._side.current_goal = value

    @property
    def actions_taken(self) -> int:
        return int(getattr(self._side, "actions_taken", 0))

    @actions_taken.setter
    def actions_taken(self, value: int) -> None:
        self._side.actions_taken = int(value)

    # -- mission state: scout/settle expeditions ---------------------------
    # Mapped onto the shared mission/target_x/target_y/mission_steps/
    # home_x/home_y columns of AgentArrays. `mission` (int8) says which kind
    # is currently active: 0=none, 1=SCOUT_OUT, 2=SCOUT_BACK, 3=SETTLE_OUT.
    # target_x/y and mission_steps are safe to share between scout and
    # settle because rule_based_agent.py always fully ends one mission
    # (resetting its target/steps to None/0) before the other kind can start
    # - the two are never concurrently active. scout_home reuses home_x/
    # home_y (otherwise unused after AgentArrays construction/spawn) and is
    # gated on `mission` being scout-related so an at-rest agent reads None
    # instead of its spawn position.
    #
    # scout_next_at and settle_next_at do NOT share a column (each has its
    # own AgentArrays field: mission_next_at / settle_next_at). They cannot
    # be aliased like target_x/y above: rule_based_agent.decide() runs
    # `_settlement_action` before `_scouting_action` every tick, and
    # `_maybe_start_settlement` writes its stagger (min 20 steps,
    # `_settle_interval()`) on most bail-out paths while a mission is NOT
    # active - i.e. concurrently with scouting's own pending stagger (min 8
    # steps, `_scout_interval()`). A shared column would let the settle
    # cadence overwrite/starve the scout cadence (or vice versa) even though
    # the two are never both *actively under way* at once.
    @property
    def scout_phase(self) -> str | None:
        m = int(self._a.mission[self.row])
        if m == MISSION_SCOUT_OUT:
            return "out"
        if m == MISSION_SCOUT_BACK:
            return "back"
        return None

    @scout_phase.setter
    def scout_phase(self, value: str | None) -> None:
        if value == "out":
            self._a.mission[self.row] = MISSION_SCOUT_OUT
        elif value == "back":
            self._a.mission[self.row] = MISSION_SCOUT_BACK
        elif value is None:
            self._a.mission[self.row] = _MISSION_NONE
        else:
            raise ValueError(f"invalid scout_phase: {value!r}")

    @property
    def settle_phase(self) -> str | None:
        return "out" if int(self._a.mission[self.row]) == MISSION_SETTLE_OUT else None

    @settle_phase.setter
    def settle_phase(self, value: str | None) -> None:
        if value == "out":
            self._a.mission[self.row] = MISSION_SETTLE_OUT
        elif value is None:
            self._a.mission[self.row] = _MISSION_NONE
        else:
            raise ValueError(f"invalid settle_phase: {value!r}")

    @property
    def scout_target(self) -> tuple[int, int] | None:
        tx, ty = int(self._a.target_x[self.row]), int(self._a.target_y[self.row])
        return None if tx < 0 else (tx, ty)

    @scout_target.setter
    def scout_target(self, value: tuple[int, int] | None) -> None:
        tx, ty = (-1, -1) if value is None else value
        self._a.target_x[self.row], self._a.target_y[self.row] = int(tx), int(ty)

    @property
    def settle_target(self) -> tuple[int, int] | None:
        return self.scout_target

    @settle_target.setter
    def settle_target(self, value: tuple[int, int] | None) -> None:
        self.scout_target = value

    @property
    def scout_steps(self) -> int:
        return int(self._a.mission_steps[self.row])

    @scout_steps.setter
    def scout_steps(self, value: int) -> None:
        self._a.mission_steps[self.row] = int(value)

    @property
    def settle_steps(self) -> int:
        return self.scout_steps

    @settle_steps.setter
    def settle_steps(self, value: int) -> None:
        self.scout_steps = value

    @property
    def scout_next_at(self) -> int:
        return int(self._a.mission_next_at[self.row])

    @scout_next_at.setter
    def scout_next_at(self, value: int) -> None:
        self._a.mission_next_at[self.row] = int(value)

    @property
    def settle_next_at(self) -> int:
        return int(self._a.settle_next_at[self.row])

    @settle_next_at.setter
    def settle_next_at(self, value: int) -> None:
        self._a.settle_next_at[self.row] = int(value)

    @property
    def founder_kit_reserved(self) -> bool:
        return bool(self._side.founder_kit_reserved)

    @founder_kit_reserved.setter
    def founder_kit_reserved(self, value: bool) -> None:
        self._side.founder_kit_reserved = bool(value)

    @property
    def scout_home(self) -> tuple[int, int] | None:
        m = int(self._a.mission[self.row])
        if m not in (MISSION_SCOUT_OUT, MISSION_SCOUT_BACK):
            return None
        return (int(self._a.home_x[self.row]), int(self._a.home_y[self.row]))

    @scout_home.setter
    def scout_home(self, value: tuple[int, int] | None) -> None:
        # Unlike scout_target/settle_target (target_x/y have a real "unset"
        # sentinel, -1), home_x/home_y have no sentinel of their own: they are
        # seeded with the agent's spawn position (AgentArrays.spawn/from_agents)
        # and the getter above already gates on `mission` to read None while at
        # rest. Writing (0,0) here on `scout_home = None` (as `_end_scout_mission`
        # does) would permanently clobber that spawn position with the map
        # corner - the read side already agrees on None, so simply skip the
        # write instead of encoding None as (0, 0).
        if value is None:
            return
        hx, hy = value
        self._a.home_x[self.row], self._a.home_y[self.row] = int(hx), int(hy)

    def to_dict(self) -> dict:
        s = self._side
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "x": self.x,
            "y": self.y,
            "local_x_m": self.local_x_m,
            "local_y_m": self.local_y_m,
            # Not vectorized and not (yet) on AgentSideState: neutral defaults
            # of the correct type, matched to BaseAgent's own defaults where
            # one exists. Task 12 wires real values by extending
            # AgentSideState (getattr picks them up automatically).
            "perception_radius": getattr(s, "perception_radius", 3),
            "perception_radius_m": getattr(s, "perception_radius_m", 59_000.0),
            "movement_distance_m_per_step": getattr(
                s, "movement_distance_m_per_step", 59_000.0
            ),
            "pillar_preferences": list(self.pillar_preferences),
            "pillar_skills": list(self.pillar_skills),
            "faction": self.faction,
            "inventory": self.inventory.to_dict(),
            "health": self.health,
            "satiety": self.satiety,
            "oxygen_level": self.oxygen_level,
            "hydration": self.hydration,
            "fatigue": self.fatigue,
            "steps_without_water": self.steps_without_water,
            "steps_without_food": self.steps_without_food,
            "stress_index": self.stress_index,
            "morale": self.morale,
            "protocol_compliance": self.protocol_compliance,
            "autonomy_preference": self.autonomy_preference,
            "current_goal": s.current_goal,
            "recent_actions": list(s.recent_actions[-10:]),
            "mode": getattr(s, "mode", "rule_based"),
            "llm_provider_id": getattr(s, "llm_provider_id", None),
            "llm_model": getattr(s, "llm_model", None),
            "memory_summary": s.memory.summarize(),
        }


class _LazyCellRow:
    """Row of `CellView`s built on demand; supports `row[x]` and iteration."""

    __slots__ = ("_c", "_agents", "_side_states", "_ps", "_y")

    def __init__(
        self,
        cells: CellArrays,
        agents: AgentArrays | None,
        side_states: dict[str, AgentSideState] | None,
        planetary_state: dict,
        y: int,
    ):
        self._c, self._agents, self._side_states, self._ps, self._y = (
            cells,
            agents,
            side_states,
            planetary_state,
            y,
        )

    def __getitem__(self, x: int) -> CellView:
        return CellView(
            self._c,
            x,
            self._y,
            self._ps,
            self._agents,
            self._side_states,
        )

    def __iter__(self):
        for x in range(self._c.W):
            yield self[x]

    def __len__(self) -> int:
        return self._c.W


class _LazyCellGrid:
    """`cells`-like 2D lazy sequence: `grid[y][x]` and `for row in grid: ...`."""

    __slots__ = ("_c", "_agents", "_side_states", "_ps")

    def __init__(
        self,
        cells: CellArrays,
        agents: AgentArrays | None,
        side_states: dict[str, AgentSideState] | None,
        planetary_state: dict,
    ):
        self._c, self._agents, self._side_states, self._ps = (
            cells,
            agents,
            side_states,
            planetary_state,
        )

    def __getitem__(self, y: int) -> _LazyCellRow:
        return _LazyCellRow(
            self._c,
            self._agents,
            self._side_states,
            self._ps,
            y,
        )

    def __iter__(self):
        for y in range(self._c.H):
            yield self[y]

    def __len__(self) -> int:
        return self._c.H


class WorldView:
    """Minimal `GridWorld`-shaped facade over `CellArrays`/`AgentArrays` -
    covers what `colony_dynamics.py` and the API routes consume, plus the
    handful of `world.*` calls src/agents/action_space.py makes while
    mutating state (`log_event`, `adjust_total_ice_cache`, `add_structure`,
    `move_agent`)."""

    def __init__(
        self,
        cells: CellArrays,
        agents: AgentArrays,
        planetary_state: dict,
        metadata: dict,
        side_states: dict[str, AgentSideState] | None = None,
    ):
        self._cells, self._agents, self.planetary_state, self.metadata = cells, agents, planetary_state, metadata
        self._side_states = side_states
        self.events: list[dict] = []

    # day/step read live from `metadata` instead of a value copied once at
    # construction time: GridWorld.day/.step (src/world/grid.py) are plain
    # mutable fields the shell assigns every step/day
    # (agent_coupled_runner.py, state_store.py: `self.world.step = ...`); a
    # WorldView snapshotted once at construction would keep reporting the
    # step/day it was built with forever, so every event logged through it
    # (`log_event` below) would carry a stale step/day. The setters mirror
    # that mutability by writing back into the same shared `metadata` dict.
    @property
    def day(self) -> int:
        return int(self.metadata.get("day", 0)) if isinstance(self.metadata, dict) else 0

    @day.setter
    def day(self, value: int) -> None:
        if isinstance(self.metadata, dict):
            self.metadata["day"] = int(value)

    @property
    def step(self) -> int:
        return int(self.metadata.get("step", 0)) if isinstance(self.metadata, dict) else 0

    @step.setter
    def step(self, value: int) -> None:
        if isinstance(self.metadata, dict):
            self.metadata["step"] = int(value)

    @property
    def width(self) -> int:
        return self._cells.W

    @property
    def height(self) -> int:
        return self._cells.H

    @property
    def cells(self) -> _LazyCellGrid:
        return _LazyCellGrid(
            self._cells,
            self._agents,
            self._side_states,
            self.planetary_state,
        )

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._cells.W and 0 <= y < self._cells.H

    def get_cell(self, x: int, y: int) -> CellView:
        if not self.in_bounds(x, y):
            raise IndexError(f"cell out of bounds: {x},{y}")
        return CellView(
            self._cells,
            x,
            y,
            self.planetary_state,
            self._agents,
            self._side_states,
        )

    def get_agents_in_range(self, x: int, y: int, radius: int) -> list[str]:
        # Bounding-box (Chebyshev) radius, matching GridWorld.get_agents_in_range
        # (src/world/grid.py) exactly.
        rows = self._agents.alive_rows()
        if rows.size == 0:
            return []
        ax = self._agents.x[rows].astype(np.int32)
        ay = self._agents.y[rows].astype(np.int32)
        raw_dx = np.abs(ax - int(x))
        wrapped_dx = np.minimum(raw_dx, self.width - raw_dx)
        mask = (wrapped_dx <= radius) & (np.abs(ay - int(y)) <= radius)
        return [self._agents.ids[int(i)] for i in rows[mask]]

    def neighbors(self, x: int, y: int, radius: int = 1) -> list[CellView]:
        out: list[CellView] = []
        seen: set[tuple[int, int]] = set()
        for yy in range(max(0, y - radius), min(self.height, y + radius + 1)):
            for raw_x in range(x - radius, x + radius + 1):
                xx = raw_x % self.width
                key = (xx, yy)
                if key == (x, y) or key in seen:
                    continue
                seen.add(key)
                out.append(self.get_cell(xx, yy))
        return out

    def mark_explored(self, x: int, y: int) -> None:
        # Mirrors GridWorld.mark_explored (src/world/grid.py:142-146). Task 9
        # (src/core/kernel.py) is the first caller that drives
        # src/world/perception.py's observe()/observe_rule_based() against a
        # WorldView instead of a real GridWorld - both call `world.mark_explored`
        # unconditionally, which WorldView had no method for at all (an
        # AttributeError on the very first observation, not a subtle value
        # mismatch). `_explored_count` is bookkeeping only (never read by
        # rule_based_agent.py/action_space.py), so this only needs to match
        # the visible behavior: explored flips to True at most once, and the
        # counter increments exactly on that transition.
        if not bool(self._cells.explored[y, x]):
            self._cells.explored[y, x] = True
            self.metadata["_explored_count"] = int(self.metadata.get("_explored_count", 0)) + 1

    def log_event(self, event_type: str, message: str, **data) -> None:
        # Mirrors GridWorld.log_event (src/world/grid.py); appends to this
        # WorldView's own `events` list rather than a shared GridWorld one.
        event = {
            "event_id": f"{len(self.events) + 1:08d}",
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            "step": data.pop("step", getattr(self, "step", None)),
            "day": getattr(self, "day", 0),
            "type": event_type,
            "message": message,
            "data": data,
        }
        self.events.append(event)

    def adjust_total_ice_cache(self, delta: float) -> None:
        self.metadata["_total_ice_cache"] = max(0.0, float(self.metadata.get("_total_ice_cache", 0.0)) + float(delta))

    def place_agent(
        self,
        agent_id: str,
        x: int,
        y: int,
        local_x_m: float | None = None,
        local_y_m: float | None = None,
    ) -> None:
        # No-op, deliberately: unlike GridWorld, there is no separate per-cell
        # agents_present list to populate - CellView.agents_present already
        # derives membership live from AgentArrays.x/y. Task 12
        # (src/api/state_store.py's SimulationController): called by
        # src/agents/population.py's maybe_spawn_agent/spawn_initial_agents
        # BEFORE the new agent has an AgentArrays row at all (the shell
        # commits the real row right after - see
        # SimulationController._commit_spawned_agent) - there is nothing to
        # do here yet; presence becomes live automatically the moment that
        # row exists.
        return None

    def recompute_habitability(self) -> None:
        # Mirrors GridWorld.recompute_habitability (src/world/grid.py:255-268)
        # - called by src/simulation/planetary_coupling.py's `sync_world`/
        # `sync_static_mars_environment` after a climate update. The FULL
        # vectorized formula (kernel_biology.recompute_habitability, already
        # imported above for `add_structure`) recomputes the whole grid when
        # called without ys/xs - a real, complete recompute, not the
        # incremental approximation `add_structure`'s own docstring
        # describes replacing.
        recompute_habitability(self._cells, self.planetary_state, cell_degradation_enabled())

    @property
    def _agent_positions(self) -> dict[str, tuple[int, int]]:
        # Live equivalent of GridWorld._agent_positions (src/world/grid.py),
        # consumed via `getattr(world, "_agent_positions", {})` by
        # src/simulation/colony_dynamics.py's `_occupied_cells`. Computed
        # fresh from AgentArrays on every access (no cache, so never stale) -
        # unlike `_structure_positions` (see
        # SimulationController._invalidate_structure_cache's docstring),
        # nothing ever tries to ASSIGN to this attribute, so a read-only
        # property is safe and simplest here.
        rows = self._agents.alive_rows()
        return {self._agents.ids[int(r)]: (int(self._agents.x[r]), int(self._agents.y[r])) for r in rows}

    @property
    def _structure_positions(self) -> set[tuple[int, int]]:
        """Live cached structure index backed by ``CellArrays``."""
        return set(self._cells.structure_positions())

    def add_structure(self, structure) -> None:
        # Mirrors GridWorld.add_structure's bookkeeping (struct_count/
        # struct_integrity/struct_fx). GridWorld.add_structure then calls
        # Cell.recompute_habitability, which reruns the FULL habitability
        # formula (src/world/cell.py:154-188) - terrain/water/vegetation/
        # radiation/dust/polar/pollution/planetary_state terms, not just
        # structures.
        #
        # This used to only reproduce the formula's incremental tail (sum
        # each structure's local_effect["habitability"], clamp to [0, 1]),
        # added onto the existing `cells.habitability` value - a documented
        # approximation that could drift if some other habitability input
        # changed without going through here. Task 8 (src/core/kernel_biology.py)
        # vectorized the FULL formula, so we now call it here exactly as the
        # object engine does: a real, complete recompute for just this one
        # cell (ys=y, xs=x), not an incremental patch. This is exact, not an
        # approximation, and removes the drift risk entirely.
        y, x = int(structure.y), int(structure.x)
        si = C.S[structure.type]
        self._cells.struct_count[y, x, si] += 1
        self._cells.struct_integrity[y, x, si] += float(structure.integrity)
        self._cells._structure_instances.setdefault((y, x, si), []).append(
            (
                float(getattr(structure, "local_x_m", 0.0)),
                float(getattr(structure, "local_y_m", 0.0)),
                getattr(structure, "owner", None),
            )
        )
        # struct_count changed outside a StructureView setter: the cached
        # StructureView list for this cell now has the wrong length, so drop it
        # (the very next reader - e.g. structure_saturated on a co-located
        # builder - must see the new count).
        self._cells.invalidate_structures_at(y, x)

        self._cells.recompute_struct_fx(y, x)
        recompute_habitability(self._cells, self.planetary_state, cell_degradation_enabled(), y, x)
        self.log_event(
            "structure_built",
            f"{structure.type.value} built at ({x},{y})",
            structure=structure.to_dict(),
        )

    def structure_type_counts(self) -> dict[str, int]:
        """Return structure totals directly from the canonical count array."""
        return {
            structure_type.value: int(
                self._cells.struct_count[..., structure_i].sum()
            )
            for structure_i, structure_type in enumerate(C.STRUCTURES)
        }

    def move_agent(
        self,
        agent_id: str,
        old_x: int,
        old_y: int,
        new_x: int,
        new_y: int,
        local_x_m: float | None = None,
        local_y_m: float | None = None,
    ) -> bool:
        # Unlike GridWorld, there is no separate per-cell agents_present list
        # to update: CellView.agents_present already derives membership live
        # from AgentArrays.x/y, and the caller (action_space.py) sets
        # `agent.x, agent.y = new_cell_x, new_cell_y` itself right after this
        # call. This only validates bounds and logs, matching GridWorld's
        # return value and event.
        #
        # Task 14 fix (equivalence audit): `cells.occupancy` is NOT derived
        # live the way `agents_present` is - it is a genuine cached array
        # (`kernel_biology.recompute_habitability`'s density-load/pollution-
        # floor term and its own degradation accumulation both read it
        # directly, see kernel_biology.py:452/222). GridWorld.move_agent
        # keeps `cell.agents_present` current on every move (remove_agent then
        # place_agent, grid.py:112-127); mirror that here so a MID-STEP
        # `WorldView.add_structure` call (views.py, triggered by a BUILD
        # action later in the SAME step's action-execution pass) sees the
        # true current population of a cell agents already moved into/out of
        # this step, not a stale pre-step count. `kernel.step()` rebuilds
        # `cells.occupancy` fully from the alive set once at the top of every
        # call (see its own docstring) - this keeps it correct for the REST
        # of that same step as agents actually move.
        if not self.in_bounds(new_x, new_y):
            return False
        if self.in_bounds(old_x, old_y) and (old_x, old_y) != (new_x, new_y):
            self._cells.occupancy[old_y, old_x] = max(0, int(self._cells.occupancy[old_y, old_x]) - 1)
            self._cells.occupancy[new_y, new_x] += 1
        self.log_event("agent_moved", f"{agent_id} moved to ({new_x},{new_y})", agent_id=agent_id, x=new_x, y=new_y)
        return True

    # -- Task 12: shell-facing read methods (GridWorld.metrics/.snapshot) ----
    # Neither is called by any pinned logic module (rule_based_agent.py,
    # action_space.py, vitals.py, extreme_events.py) - both exist purely for
    # `SimulationController`'s `_current_metrics`/`_world_snapshot_payload`
    # (src/api/state_store.py). Unlike `GridWorld.metrics` (src/world/grid.py:
    # 270-306), which keeps incremental caches (`_habitability_cache`,
    # `_structure_positions`, ...) to avoid re-scanning every `Cell` object on
    # every call, these read straight off `CellArrays`' own vectorized
    # columns - a numpy reduction over the whole grid is already the cheap
    # path here, so no cache is needed (or could silently drift from the
    # array state the way an object-model cache theoretically could).
    def metrics(self, include_planetary: bool = True) -> dict[str, float]:
        total_cells = max(1, self._cells.W * self._cells.H)
        # L'integrita' media per struttura in piedi. **Non e' un dettaglio
        # cosmetico**: la resa di una struttura scala con l'efficienza al
        # QUADRATO (`eff_adj ** 2` in `kernel_biology`), e sotto 0,4 si aggiunge
        # un dimezzamento, quindi una struttura al 30% produce circa un
        # ventiduesimo di una al 100%. Nessuna metrica lo registrava, e la prima
        # run governata vera ha reso la lacuna evidente: il braccio LLM ha speso
        # da due a quattro volte e mezzo piu' azioni di manutenzione della
        # baseline, e non c'era modo di dire se fosse prudenza o spreco, perche'
        # le strutture non vengono mai perse -- si limitano a produrre di meno.
        #
        # Costo: una riduzione in piu' sullo stesso array che `struct_count.sum()`
        # gia' percorre in questa stessa funzione, che di riduzioni sull'intera
        # griglia ne fa gia' otto.
        standing = float(self._cells.struct_count.sum())
        metrics: dict[str, float] = {
            "average_habitability": float(np.mean(self._cells.habitability)),
            "exploration_coverage": float(np.count_nonzero(self._cells.explored)) / total_cells,
            "total_ice": float(self._cells.water_ice.sum() + self._cells.cell_res[..., C.R["ice"]].sum()),
            "vegetation": float(np.sum(self._cells.vegetation)),
            "structures_built": standing,
            "structure_integrity_mean": (
                float(self._cells.struct_integrity.sum()) / standing if standing else 0.0
            ),
            "greenhouses": float(self._cells.struct_count[..., C.S[StructureType.GREENHOUSE]].sum()),
            "oxygen_plants": float(self._cells.struct_count[..., C.S[StructureType.OXYGEN_PLANT]].sum()),
        }
        conoscenza_colonia = float(self._cells.cell_res[..., C.R["knowledge"]].sum())

        if include_planetary:
            metrics.update(self.planetary_state)

        # **Dopo il merge, e non prima.** `planetary_state` porta
        # `scientific_knowledge: 0.0` fra i propri default anche a layer
        # disattivato, quindi scrivendo prima il valore della colonia verrebbe
        # sovrascritto da quello zero. Il layer, quando produce davvero un
        # valore, continua a vincere: li' la grandezza ha una dinamica propria.
        #
        # Senza questo blocco la colonia poteva costruire decine di laboratori
        # di ricerca e vedere la conoscenza scientifica ferma a zero e il
        # livello tecnologico mai avanzare — due parti dello stesso modello che
        # si contraddicevano.
        if not float(metrics.get("scientific_knowledge", 0.0) or 0.0):
            metrics["scientific_knowledge"] = conoscenza_colonia
            # I bonus vanno ricalcolati con essa: `get_active_bonuses` gira solo
            # dentro il layer planetario, quindi senza questa riga la conoscenza
            # si accumulerebbe senza sbloccare nulla.
            from src.simulation.planetary_coupling import active_bonuses

            for chiave, valore in active_bonuses(conoscenza_colonia).items():
                if not isinstance(valore, list):
                    metrics[chiave] = float(valore)
        return metrics

    def snapshot(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "day": self.day,
            "metadata": dict(self.metadata),
            "cells": [self.get_cell(x, y).to_public_dict() for y in range(self.height) for x in range(self.width)],
            "events": self.events[-200:],
        }

    # -- Task 12 review carry-over: the live GUI map/chunk API routes
    # (src/api/routes_world.py) call `world.compact_chunk`/
    # `world.render_map_payload` on whatever `controller.read_world()`
    # returns - a `WorldView` once a scenario is loaded. Both mirror
    # `GridWorld`'s own versions (src/world/grid.py:178-253) field-for-field
    # and byte-for-byte in shape, built from `CellArrays`' vectorized columns
    # instead of a per-`Cell` Python loop - see
    # tests/core/test_views_world_routes.py for the payload-equivalence test
    # this pins.
    def compact_chunk(self, x: int = 0, y: int = 0, zoom: int = 1, size: int = 50) -> dict:
        size = max(1, int(size))
        start_x = max(0, min(self.width - 1, int(x)))
        start_y = max(0, min(self.height - 1, int(y)))
        max_x = min(self.width, start_x + size)
        max_y = min(self.height, start_y + size)
        cells = [self.get_cell(xx, yy).to_public_dict() for yy in range(start_y, max_y) for xx in range(start_x, max_x)]
        return {
            "x": start_x,
            "y": start_y,
            "zoom": zoom,
            "width": max_x - start_x,
            "height": max_y - start_y,
            "world_width": self.width,
            "world_height": self.height,
            "cells": cells,
        }

    def render_map_payload(self, include_static: bool = False) -> dict:
        """Compact full-planet snapshot for GUI polling (flat row-major
        arrays, index = y * width + x) - matches GridWorld.render_map_payload
        exactly, including its rounding, but reads straight off the
        vectorized columns (`.flatten()` on an (H, W) array is row-major by
        default, the same y-then-x order the object engine's nested
        `for row in self.cells: for cell in row` loop produces) instead of
        building one dict per cell first.
        """
        payload: dict = {
            "world_width": self.width,
            "world_height": self.height,
            "day": int(self.day),
        }
        if include_static:
            payload["static"] = {
                "terrain": [int(v) for v in self._cells.terrain.flatten().tolist()],
                "terrain_types": [t.value for t in TERRAIN_ORDER],
                "elevation": [round(float(v), 2) for v in self._cells.elevation.flatten().tolist()],
                "polar_severity": [round(float(v), 2) for v in self._cells.polar.flatten().tolist()],
            }
        payload["dynamic"] = {
            "habitability": [round(float(v), 3) for v in self._cells.habitability.flatten().tolist()],
            "vegetation": [round(float(v), 3) for v in self._cells.vegetation.flatten().tolist()],
            "water_ice": [round(float(v), 2) for v in self._cells.water_ice.flatten().tolist()],
            "dust": [round(float(v), 2) for v in self._cells.dust.flatten().tolist()],
            "radiation": [round(float(v), 2) for v in self._cells.radiation.flatten().tolist()],
            "explored": [int(v) for v in self._cells.explored.flatten().tolist()],
        }
        structures: list[dict] = []
        for x, y in self._cells.structure_positions():
            structures.extend(
                {
                    "x": x,
                    "y": y,
                    "type": structure.type.value,
                    "local_x_m": float(structure.local_x_m),
                    "local_y_m": float(structure.local_y_m),
                }
                for structure in self.get_cell(x, y).structures
            )
        payload["structures"] = structures
        return payload
