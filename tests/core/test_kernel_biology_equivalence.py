import numpy as np
import pytest
from src.agents.base_agent import BaseAgent
from src.world.cell import Cell, set_cell_degradation
from src.world.grid import GridWorld
from src.world.resources import ResourceBundle
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.kernel_biology import update_cells, _serialized_reserve_draw
from src.simulation.step_effects import (
    apply_colony_resource_feedback, apply_structure_effects, apply_structure_wear,
    research_knowledge_gain)

FIELDS = ("water_ice", "liquid_water", "proto_soil_development",
          "vegetation_biomass", "pollution_risk", "habitability_score")
COL = {"proto_soil_development": "proto_soil", "vegetation_biomass": "vegetation",
       "pollution_risk": "pollution", "habitability_score": "habitability"}


def _grid(width: int, height: int) -> GridWorld:
    # GridWorld has no width/height/seed convenience constructor (plain dataclass
    # requiring a pre-built cells grid) — build it the same way
    # tests/core/test_cell_arrays.py and tests/core/test_kernel_vitals_equivalence.py do.
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)] for y in range(height)]
    return GridWorld(width=width, height=height, cells=cells)


def _scenario():
    w = _grid(5, 4)
    c = w.get_cell(2, 2)
    # NOTE: must go through GridWorld.add_structure (not cell.structures.append)
    # so world._structure_positions (grid.py:32,39-44) — computed once in
    # __post_init__ from the cells passed at construction time — actually
    # knows about this cell. apply_structure_effects/apply_colony_resource_
    # feedback/apply_structure_wear all iterate structure_cells(world), which
    # trusts that spatial index; a plain list .append() would leave the
    # structure invisible to all three and silently no-op them.
    w.add_structure(Structure(type=StructureType.GREENHOUSE, x=2, y=2))
    w.add_structure(Structure(type=StructureType.SOLAR_ARRAY, x=2, y=2))
    c.water_ice, c.liquid_water = 4.0, 0.5
    c.nutrients.update({"N": 5.0, "P": 1.0, "C": 3.0})
    w.get_cell(0, 0).water_ice = 2.0
    agents = {"a0": BaseAgent(agent_id="a0", name="a0", role="colonist", x=2, y=2)}
    # IMPORTANT: `ps` must be the SAME dict object as `w.planetary_state`, not a
    # copy. apply_structure_effects(w) (called below by the object pipeline)
    # reads `world.planetary_state` internally for its own (sticking, for a
    # structured cell) recompute_habitability call — if that were a different
    # dict than the one this test passes to update_biology/update_cells, the
    # two engines would compute habitability from different planetary inputs
    # and diverge by exactly the planetary_state-gated terms (shielding,
    # liquid_stability, vegetation_suitability, temp_score), independent of
    # any kernel bug.
    w.planetary_state.setdefault("mean_temperature_c", -63.0)
    w.planetary_state.setdefault("pressure_pa", 600.0)
    w.planetary_state.setdefault("liquid_water_stability", 0.0)
    ps = w.planetary_state
    return w, agents, ps


def test_update_cells_matches_object_pipeline():
    w, agents, ps = _scenario()
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = 3650.0
    # pipeline a oggetti (ordine dei motori: biologia -> effects -> feedback -> wear)
    for row in w.cells:
        for cell in row:
            cell.agents_present = [a.agent_id for a in agents.values()
                                   if (a.x, a.y) == (cell.x, cell.y)]
            cell.update_biology(ps, dt)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, dt)
    apply_structure_wear(w, dt)
    # pipeline vettoriale
    ca.occupancy[:] = 0
    np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)
    update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)
    for y, row in enumerate(w.cells):
        for x, cell in enumerate(row):
            for f in FIELDS:
                col = COL.get(f, f)
                assert np.isclose(getattr(ca, col)[y, x], getattr(cell, f), atol=1e-4), (
                    f"({x},{y}).{f}: obj={getattr(cell, f)} vec={getattr(ca, col)[y, x]}")

    # Important 2 (partial): agents' per-capita food/construction_material
    # draws (step_effects.py:81-85, kernel_biology.py:337-340) were never
    # compared against the object pipeline anywhere - only cell-side fields
    # were. `_scenario()`'s agent "a0" starts with BaseAgent's default
    # inventory (food/construction_material > 0), so the flat per-step draw
    # is exercised on both sides here.
    row = aa.index["a0"]
    assert np.isclose(aa.inv[row, C.R["food"]], agents["a0"].inventory.food, atol=1e-5)
    assert np.isclose(
        aa.inv[row, C.R["construction_material"]], agents["a0"].inventory.construction_material, atol=1e-5
    )


def test_wear_reduces_aggregated_integrity_like_objects():
    # The name promises an object-model comparison, but this used to only
    # assert `sum < before` on the vectorized side alone. `_scenario()` has
    # exactly one structure per type per cell (2,2) (a GREENHOUSE and a
    # SOLAR_ARRAY), so aggregate == per-instance there and an EXACT
    # np.isclose against `sum(s.integrity for s in cell.structures)` is
    # available for free - no aggregation-sacrifice (see kernel_biology.py's
    # module docstring) can hide behind this scenario.
    w, agents, ps = _scenario()
    cell = w.get_cell(2, 2)
    ca = CellArrays.from_world(w)
    aa = AgentArrays.from_agents(agents)
    before = ca.struct_integrity[2, 2].sum()

    for row in w.cells:
        for c in row:
            c.agents_present = [a.agent_id for a in agents.values() if (a.x, a.y) == (c.x, c.y)]
            c.update_biology(ps, 3650.0)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, 3650.0)
    apply_structure_wear(w, 3650.0)

    update_cells(ca, aa, ps, 3650.0, cell_degradation=True, full_grid=True)

    assert ca.struct_integrity[2, 2].sum() < before
    obj_sum = sum(s.integrity for s in cell.structures)
    assert np.isclose(ca.struct_integrity[2, 2].sum(), obj_sum, atol=1e-5), (
        f"obj={obj_sum} vec={ca.struct_integrity[2, 2].sum()}")


def test_ice_liquid_exchange_melt_and_refreeze_branches():
    # cell.py:86-93. Same planetary liquid_water_stability (> 0) for the whole
    # grid, but per-cell local_temperature_modifier pushes one cell's
    # effective_temp_c above 0 (melt branch, cell.py:86-89) and the other's
    # below -5 (refreeze branch, cell.py:90-93) — exercising both branches
    # of the if/elif in the SAME step.
    w = _grid(2, 1)
    melt_cell = w.get_cell(0, 0)
    melt_cell.water_ice = 5.0
    refreeze_cell = w.get_cell(1, 0)
    refreeze_cell.liquid_water = 3.0
    refreeze_cell.local_temperature_modifier = -20.0
    agents: dict = {}
    ps = {"mean_temperature_c": 2.0, "pressure_pa": 600.0, "liquid_water_stability": 0.5}
    dt = 3650.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0

    for row in w.cells:
        for cell in row:
            cell.agents_present = []
            cell.update_biology(ps, dt)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, dt)
    apply_structure_wear(w, dt)

    update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)

    assert melt_cell.water_ice < 5.0, "sanity: object model really melted ice"
    assert refreeze_cell.water_ice > 0.0, "sanity: object model really refroze liquid"
    assert np.isclose(ca.water_ice[0, 0], melt_cell.water_ice, atol=1e-4)
    assert np.isclose(ca.liquid_water[0, 0], melt_cell.liquid_water, atol=1e-4)
    assert np.isclose(ca.water_ice[0, 1], refreeze_cell.water_ice, atol=1e-4)
    assert np.isclose(ca.liquid_water[0, 1], refreeze_cell.liquid_water, atol=1e-4)


def test_cell_degradation_false_matches_object_and_keeps_pollution_zero():
    # cell.py:144-145 and :167-171 — the SAME _CELL_DEGRADATION_ENABLED toggle
    # must gate both the occupancy-driven pollution accrual AND the density
    # floor in recompute_habitability.
    w = _grid(2, 1)
    cell = w.get_cell(0, 0)
    agent = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0)
    agents = {"a0": agent}
    ps = {"mean_temperature_c": -63.0, "pressure_pa": 600.0, "liquid_water_stability": 0.0}
    dt = 3650.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0
    np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)

    set_cell_degradation(False)
    try:
        for row in w.cells:
            for c in row:
                c.agents_present = [aid for aid, a in agents.items() if (a.x, a.y) == (c.x, c.y)]
                c.update_biology(ps, dt)
        apply_structure_effects(w)
        apply_colony_resource_feedback(w, agents, dt)
        apply_structure_wear(w, dt)

        update_cells(ca, aa, ps, dt, cell_degradation=False, full_grid=True)
    finally:
        set_cell_degradation(True)

    assert cell.pollution_risk == 0.0
    assert ca.pollution[0, 0] == 0.0
    assert np.isclose(ca.habitability[0, 0], cell.habitability_score, atol=1e-4)


def test_greenhouse_fertilization_caps_match_object_model():
    # step_effects.py:47-50 caps: vegetation<=10, N<=20, P<=4, C<=10.
    w = _grid(1, 1)
    cell = w.get_cell(0, 0)
    w.add_structure(Structure(type=StructureType.GREENHOUSE, x=0, y=0, integrity=1.0))
    cell.vegetation_biomass = 9.99
    cell.nutrients.update({"N": 19.99, "P": 3.99, "C": 9.99})
    agents: dict = {}
    ps = {"mean_temperature_c": -63.0, "pressure_pa": 600.0, "liquid_water_stability": 0.0}
    dt = 3650.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0

    for row in w.cells:
        for c in row:
            c.agents_present = []
            c.update_biology(ps, dt)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, dt)
    apply_structure_wear(w, dt)

    update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)

    assert cell.vegetation_biomass <= 10.0 + 1e-6
    assert cell.nutrients["N"] <= 20.0 + 1e-6
    assert cell.nutrients["P"] <= 4.0 + 1e-6
    assert cell.nutrients["C"] <= 10.0 + 1e-6
    assert np.isclose(ca.vegetation[0, 0], cell.vegetation_biomass, atol=1e-4)
    assert np.isclose(ca.nut_n[0, 0], cell.nutrients["N"], atol=1e-4)
    assert np.isclose(ca.nut_p[0, 0], cell.nutrients["P"], atol=1e-4)
    assert np.isclose(ca.nut_c[0, 0], cell.nutrients["C"], atol=1e-4)


def test_multiple_agents_same_cell_share_scarce_reserve_without_creating_matter():
    # step_effects.py:89-96 mass-conservation trap: 3 agents on one cell with
    # only 0.3 energy in stock (refill step is 0.25) must draw SEQUENTIALLY —
    # first agent gets 0.25 (stock -> 0.05), second gets the remaining 0.05
    # (stock -> 0.0), third gets nothing. A naive vectorized gather would let
    # all three draw up to 0.25 from the same undrained stock (0.75 total,
    # manufacturing 0.45 units of energy out of nothing).
    w = _grid(1, 1)
    cell = w.get_cell(0, 0)
    w.add_structure(Structure(type=StructureType.SOLAR_ARRAY, x=0, y=0, integrity=1.0))
    cell.resources.energy = 0.3
    agents = {f"a{i}": BaseAgent(agent_id=f"a{i}", name=f"a{i}", role="colonist", x=0, y=0)
              for i in range(3)}
    ps = {"mean_temperature_c": -63.0, "pressure_pa": 600.0, "liquid_water_stability": 0.0}
    dt = 3650.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0
    np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)

    for row in w.cells:
        for c in row:
            c.agents_present = list(agents.keys())
            c.update_biology(ps, dt)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, dt)
    apply_structure_wear(w, dt)

    update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)

    assert np.isclose(cell.resources.energy, ca.cell_res[0, 0, C.R["energy"]], atol=1e-5)
    for i in range(3):
        row = aa.index[f"a{i}"]
        obj_val = agents[f"a{i}"].inventory.energy
        assert np.isclose(aa.inv[row, C.R["energy"]], obj_val, atol=1e-5), f"a{i} energy mismatch"
    # sanity: sequential depletion really happened. BaseAgent's default
    # inventory already carries energy=2.0 (base_agent.py), so with a scarce
    # 0.3-unit cell stock the three agents can't all get the same draw:
    # strictly decreasing per-agent gains is the signature of the sequential
    # (object-model) draw order, not a naive parallel gather (which would
    # give all three an identical draw from the same undrained stock).
    per_agent = [agents[f"a{i}"].inventory.energy for i in range(3)]
    assert per_agent[0] > per_agent[1] > per_agent[2] == 2.0  # 3rd agent: stock ran out, no draw
    assert ca.cell_res[0, 0, C.R["energy"]] < 1e-4, "scarce stock must be fully drained"


def test_wear_emits_structure_warning_event_below_020_mean_integrity():
    w = _grid(1, 1)
    cell = w.get_cell(0, 0)
    # NOTE: use GridWorld.add_structure (not cell.structures.append), same
    # reasoning as _scenario() above - keeps world._structure_positions in
    # sync, the pattern the other tests in this file already follow.
    w.add_structure(Structure(type=StructureType.SHELTER, x=0, y=0, integrity=0.21))
    cell.dust_level = 5.0
    cell.radiation_level = 5.0
    agents: dict = {}
    ps = {"mean_temperature_c": -63.0, "pressure_pa": 600.0, "liquid_water_stability": 0.0,
          "wind_speed_m_s": 40.0}
    dt = 3650.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0

    out = update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)

    si = C.S[StructureType.SHELTER]
    assert ca.struct_integrity[0, 0, si] < 0.2
    warnings = [e for e in out["events"] if e["type"] == "structure_warning"]
    assert len(warnings) == 1
    assert warnings[0]["x"] == 0 and warnings[0]["y"] == 0
    assert "shelter" in warnings[0]["message"]


# ---------------------------------------------------------------------------
# Important 1: squared-efficiency colony production (kernel_biology.py:324,
# `eff2_agg = (eff_adj ** 2) * count_f`) had zero equivalence coverage. Every
# other scenario in this file uses integrity=1.0 everywhere, where
# efficiency == efficiency**2 (1.0 == 1.0**2), so even a full field-by-field
# comparison could not have told the correct eff**2 apart from a wrong eff.
# This scenario straddles the 0.4 efficiency cliff on two structure types
# (GREENHOUSE at 0.6 stays on the >0.4 side; HABITAT at 0.3 falls to the
# *0.5 side) and adds a STORAGE_DEPOT/RESEARCH_LAB pair (also straddling 0.4)
# so the colony-feedback resource fields that still move - food, tools, water -
# actually move, and `construction_material` (no longer produced by a depot
# since 2026-08-25) is checked to stay at zero on BOTH engines. `cell.resources.tools` is pre-seeded so
# tool_factor > 1.0, exercising that multiplier alongside eff2_agg.
# ---------------------------------------------------------------------------


def test_colony_feedback_squared_efficiency_matches_object_pipeline_across_040_threshold():
    w = _grid(1, 1)
    cell = w.get_cell(0, 0)
    w.add_structure(Structure(type=StructureType.GREENHOUSE, x=0, y=0, integrity=0.6))
    w.add_structure(Structure(type=StructureType.HABITAT, x=0, y=0, integrity=0.3))
    w.add_structure(Structure(type=StructureType.STORAGE_DEPOT, x=0, y=0, integrity=0.6))
    w.add_structure(Structure(type=StructureType.RESEARCH_LAB, x=0, y=0, integrity=0.3))
    cell.resources.tools = 5.0  # tool_factor = 1 + min(0.75, 5.0*0.05) = 1.25 > 1.0
    # Corrente in magazzino: dal 2026-08-25 serra, habitat e infermeria
    # pagano un carico elettrico, e senza copertura non producono nulla. Qui
    # non c'e' un pannello, quindi la scorta va messa esplicitamente o il test
    # misurerebbe il blackout invece della soglia di efficienza.
    cell.resources.energy = 5.0
    # Materiale in magazzino: dal 2026-08-25 gli attrezzi si fabbricano con il
    # materiale da costruzione invece di comparire dal nulla, quindi senza
    # scorta il laboratorio non ne produce e il test misurerebbe la penuria
    # invece della soglia di efficienza.
    cell.resources.construction_material = 2.0
    agents: dict = {}
    ps = {"mean_temperature_c": -63.0, "pressure_pa": 600.0, "liquid_water_stability": 0.0}
    dt = 3650.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0

    for row in w.cells:
        for c in row:
            c.agents_present = []
            c.update_biology(ps, dt)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, dt)
    apply_structure_wear(w, dt)

    update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)

    # `construction_material` resta nell'elenco confrontato: dal 2026-08-25 i
    # depositi immagazzinano invece di fabbricare, quindi il valore atteso e'
    # ZERO — ed e' proprio la parita' sullo zero che verifica che entrambi i
    # motori abbiano smesso di produrlo, non solo uno.
    for field in ("food", "construction_material", "tools", "water", "energy", "oxygen"):
        obj_val = getattr(cell.resources, field)
        vec_val = ca.cell_res[0, 0, C.R[field]]
        assert np.isclose(vec_val, obj_val, atol=1e-4), f"{field}: obj={obj_val} vec={vec_val}"
    # Sanity: efficiencies really do straddle 0.4 (0.6 stays linear, 0.3*0.5
    # halves) so this scenario could not pass with efficiency==1 everywhere.
    # Il deposito non compare piu' fra i testimoni perche' non produce nulla;
    # il laboratorio (0,3, sotto soglia) resta rappresentato dai `tools`.
    assert cell.resources.food > 0.0
    # Il materiale non e' piu' prodotto dai depositi ed e' ora CONSUMATO dagli
    # attrezzi: deve essere sceso sotto la scorta iniziale.
    assert cell.resources.construction_material < 2.0
    assert cell.resources.tools > 5.0
    assert cell.resources.water > 0.0


# ---------------------------------------------------------------------------
# Important 2: fields the headline equivalence test never compared.
# ---------------------------------------------------------------------------


def test_organic_and_nutrients_mid_range_match_object_pipeline():
    # organic_matter/nut_n/nut_p/nut_c were compared nowhere, and every
    # existing scenario either doesn't touch them or (with dt=3650, a
    # greenhouse, and 10 years of growth) saturates vegetation/nutrients at
    # their ceilings, where the `decayed*k - grown*k` arithmetic stops being
    # differentially tested (both engines just sit on the clamp). This
    # scenario uses a short dt (100 days) and mid-range starting values so
    # vegetation settles well below VEGETATION_CARRYING_CAPACITY and no
    # nutrient clamp engages, only the decay/growth cycle (cell.py:133-137 /
    # kernel_biology.py:281-284).
    w = _grid(1, 1)
    cell = w.get_cell(0, 0)
    cell.vegetation_biomass = 2.0
    cell.proto_soil_development = 0.5
    cell.liquid_water = 1.0
    cell.water_ice = 3.0
    cell.organic_matter = 1.0
    cell.nutrients.update({"N": 5.0, "P": 1.0, "C": 2.5})
    agents: dict = {}
    ps = {"mean_temperature_c": 10.0, "pressure_pa": 50000.0, "liquid_water_stability": 0.5}
    dt = 100.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0

    for row in w.cells:
        for c in row:
            c.agents_present = []
            c.update_biology(ps, dt)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, dt)
    apply_structure_wear(w, dt)

    update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)

    # Sanity: genuinely mid-range, not saturated at any bound.
    assert 0.0 < cell.vegetation_biomass < 10.0 - 1e-6
    assert cell.nutrients["N"] > 0.0 and cell.nutrients["N"] < 20.0
    assert cell.organic_matter > 1.0  # decay really added organic matter

    assert np.isclose(ca.organic[0, 0], cell.organic_matter, atol=1e-4)
    assert np.isclose(ca.nut_n[0, 0], cell.nutrients["N"], atol=1e-4)
    assert np.isclose(ca.nut_p[0, 0], cell.nutrients["P"], atol=1e-4)
    assert np.isclose(ca.nut_c[0, 0], cell.nutrients["C"], atol=1e-4)


def test_knowledge_gain_matches_object_research_knowledge_gain():
    # kernel_biology.py's `knowledge_gain` (line ~408) was never compared to
    # `research_knowledge_gain` (step_effects.py:99-110). Cover both additive
    # terms - research_assets*0.5 (a RESEARCH_LAB) and tool_stock*0.03 (an
    # agent holding tools) - and the dt/365.0 (knowledge) vs dt/365.25 (years,
    # used elsewhere in the same function) distinction: any dt not a multiple
    # of 365.25 makes the two divisors numerically distinguishable.
    w = _grid(1, 1)
    w.add_structure(Structure(type=StructureType.RESEARCH_LAB, x=0, y=0, integrity=1.0))
    agent = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0,
                       inventory=ResourceBundle(tools=4.0))
    agents = {"a0": agent}
    ps = {"mean_temperature_c": -63.0, "pressure_pa": 600.0, "liquid_water_stability": 0.0}
    dt = 3650.0

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0
    np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)

    for row in w.cells:
        for c in row:
            c.agents_present = ["a0"] if (c.x, c.y) == (0, 0) else []
            c.update_biology(ps, dt)
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, dt)
    obj_gain = research_knowledge_gain(w, agents, dt)
    apply_structure_wear(w, dt)

    out = update_cells(ca, aa, ps, dt, cell_degradation=True, full_grid=True)

    assert obj_gain > 0.0  # sanity: both terms actually contributed
    assert np.isclose(obj_gain, (1 * 0.5 + 4.0 * 0.03) * (dt / 365.0), atol=1e-9)
    assert np.isclose(out["knowledge_gain"], obj_gain, atol=1e-4), (
        f"obj={obj_gain} vec={out['knowledge_gain']}")


# ---------------------------------------------------------------------------
# Minor: undefended row-order assumption in `_serialized_reserve_draw`.
# ---------------------------------------------------------------------------


def test_ordine_del_tetto_di_magazzino_diverge_di_una_ricarica():
    """I due motori applicano il tetto in due punti diversi, e si vede qui.

    **Il punto e' dichiarato lo stesso e non lo e'.** Il commento di entrambi i
    motori dice che il tetto si applica «dopo tutte le aggiunte del passo, una
    volta sola, in entrambi i motori nello stesso punto: e' cio' che rende la
    parita' verificabile». In realta':

        motore a oggetti (step_effects.py)   aggiunte -> prelievi -> TETTO
        kernel           (kernel_biology.py) aggiunte -> TETTO -> prelievi

    L'unica grandezza toccata da entrambi e' l'ossigeno --- `RISORSE_A_TETTO` e'
    (food, water, oxygen) e i prelievi di riserva riguardano energia e ossigeno
    --- ed e' precisamente l'unica che `test_update_cells_matches_object_pipeline`
    non confronta: `FIELDS` copre ghiaccio, acqua, suolo, vegetazione,
    inquinamento e abitabilita', piu' cibo e materiale nelle sacche. Il difetto
    stava nella sola grandezza fuori dal confronto, ed e' rimasto invisibile
    per tutta la campagna di parita'.

    **Quanto vale.** Esattamente `AGENT_REFILL_PER_STEP` per colono che
    preleva e per passo, e solo in una cella gia' satura: col tetto prima la
    cella chiude a `tetto - ricarica`, col tetto dopo a `tetto`. Il colono
    riceve la sua ricarica in tutti e due i casi, e la massa e' conservata in
    tutti e due --- cambia solo quanta ne finisce nell'eccedenza che si
    disperde.

    **Perche' il valore predefinito resta quello che diverge.** Misurato su
    cinque semi appaiati del braccio `none`, mille passi: quattro run su cinque
    sono identiche bit per bit nei due ordini, la quinta resta identica per
    settecento passi e poi biforca (-23 vivi, +200 morti a fine run). E' quindi
    un'amplificazione caotica di una perturbazione minima, non una distorsione
    sistematica, e resta un ordine di grandezza sotto il rumore di
    campionamento del modello misurato in campagna (438 morti). Allinearlo
    adesso cambierebbe il motore a meta' di una campagna in corso, che e'
    esattamente la deriva di configurazione che questo progetto evita. Questo
    test fissa la divergenza cosi' com'e', perche' smetta di essere silenziosa,
    e fallisce di proposito il giorno in cui qualcuno cambia l'ordine senza
    aggiornare la nota.
    """
    from src.core import kernel_biology
    from src.core.constants import AGENT_REFILL_PER_STEP

    def _mondo():
        w = _grid(3, 3)
        w.add_structure(Structure(type=StructureType.OXYGEN_PLANT, x=1, y=1))
        w.add_structure(Structure(type=StructureType.HABITAT, x=1, y=1))
        c = w.get_cell(1, 1)
        # Molto oltre qualunque tetto: la cella e' satura per costruzione.
        c.resources.oxygen = 100.0
        c.resources.energy = 50.0
        w.planetary_state.setdefault("mean_temperature_c", -63.0)
        w.planetary_state.setdefault("pressure_pa", 600.0)
        w.planetary_state.setdefault("liquid_water_stability", 0.0)
        a = BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1)
        a.inventory.oxygen = 0.0   # a secco: preleva di sicuro
        c.agents_present = ["a0"]
        return w, {"a0": a}

    def _kernel(tetto_dopo):
        w, agents = _mondo()
        aa = AgentArrays.from_agents(agents)
        ca = CellArrays.from_world(w)
        ca.occupancy[:] = 0
        np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)
        prima = kernel_biology.TETTO_DOPO_PRELIEVI
        kernel_biology.TETTO_DOPO_PRELIEVI = tetto_dopo
        try:
            update_cells(ca, aa, w.planetary_state, 7.0,
                         cell_degradation=True, full_grid=True)
        finally:
            kernel_biology.TETTO_DOPO_PRELIEVI = prima
        return (float(ca.cell_res[1, 1, C.R["oxygen"]]),
                float(aa.inv[aa.index["a0"], C.R["oxygen"]]))

    w, agents = _mondo()
    apply_structure_effects(w)
    apply_colony_resource_feedback(w, agents, 7.0)
    ossigeno_oggetti = w.get_cell(1, 1).resources.oxygen
    muta_oggetti = agents["a0"].inventory.oxygen

    cella_prima, muta_prima = _kernel(False)
    cella_dopo, muta_dopo = _kernel(True)

    assert muta_prima == pytest.approx(muta_oggetti) == pytest.approx(muta_dopo), (
        "il colono riceve la stessa ricarica in ogni ordine: e' la CELLA a "
        "chiudere su due valori diversi"
    )
    assert cella_prima == pytest.approx(ossigeno_oggetti - AGENT_REFILL_PER_STEP), (
        "divergenza attesa col valore predefinito: il kernel taglia prima di "
        "servire, quindi la cella chiude una ricarica sotto il motore a oggetti"
    )
    assert cella_dopo == pytest.approx(ossigeno_oggetti), (
        "con MARSABM_TETTO_DOPO_PRELIEVI i due motori coincidono: e' la prova "
        "che l'interruttore agisce davvero su questo punto e non altrove"
    )


def test_serialized_reserve_draw_requires_ascending_row_order():
    # `_serialized_reserve_draw`'s per-cell Python loop (kernel_biology.py)
    # replicates the object model's sequential `for agent in agents.values()`
    # draw order (step_effects.py:83-96) only because `AgentArrays.from_agents`
    # assigns rows in dict insertion order and `alive_rows()` returns them
    # ascending. Pin the assumption: rows passed out of ascending order must
    # raise, rather than silently reordering who drains a scarce cell stock
    # first if a future row-reuse allocator broke that correspondence.
    cell_res = np.zeros((1, 1, C.NR), dtype=np.float32)
    cell_res[0, 0, C.R["energy"]] = 1.0
    inv = np.zeros((3, C.NR), dtype=np.float32)
    ys = np.array([0, 0, 0], dtype=np.intp)
    xs = np.array([0, 0, 0], dtype=np.intp)

    rows_out_of_order = np.array([2, 0, 1], dtype=np.intp)
    with pytest.raises(AssertionError):
        _serialized_reserve_draw(
            cell_res, C.R["energy"], inv, C.R["energy"], rows_out_of_order, ys, xs, 3.0, 0.25
        )

    # sanity: ascending order is accepted (the real call pattern).
    rows_ascending = np.array([0, 1, 2], dtype=np.intp)
    _serialized_reserve_draw(
        cell_res, C.R["energy"], inv, C.R["energy"], rows_ascending, ys, xs, 3.0, 0.25
    )
