import numpy as np
from src.agents.base_agent import BaseAgent
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.needs import RedistributionConfig, compute_needs
from src.core.redistribution import redistribute


def _grid(width: int, height: int) -> GridWorld:
    # GridWorld has no width/height/seed convenience constructor (plain dataclass
    # requiring a pre-built cells grid) - same pattern as the other tests/core/*.py.
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)] for y in range(height)]
    return GridWorld(width=width, height=height, cells=cells)


def _setup(n_agents=6, seed=1):
    w = _grid(8, 8)
    w.get_cell(2, 2).structures.append(Structure(type=StructureType.SHELTER, x=2, y=2))
    w.get_cell(6, 6).structures.append(Structure(type=StructureType.SHELTER, x=6, y=6))
    rng = np.random.default_rng(seed)
    agents = {}
    for i in range(n_agents):
        x, y = (2, 2) if i % 2 == 0 else (6, 6)
        a = BaseAgent(agent_id=f"a{i}", name=f"a{i}", role="colonist", x=x, y=y)
        a.inventory.water = float(rng.uniform(0.0, 8.0))
        a.inventory.food = float(rng.uniform(0.0, 8.0))
        agents[a.agent_id] = a
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0
    np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)
    ca.cell_res[2, 2, C.R["water"]] = 20.0  # cella ricca
    return aa, ca


def total_mass(aa, ca, res):
    return float(aa.inv[: aa.n, C.R[res]].sum() + ca.cell_res[:, :, C.R[res]].sum())


def test_mass_is_exactly_conserved():
    aa, ca = _setup()
    cfg = RedistributionConfig()
    before = {res: total_mass(aa, ca, res) for res in C.RESOURCES}
    needs = compute_needs(ca, cfg)
    redistribute(aa, ca, needs, cfg)
    for res in C.RESOURCES:
        assert np.isclose(total_mass(aa, ca, res), before[res], atol=1e-5), res
    # Mass conservation alone does not rule out over-drawing a cell's stock
    # negative while a symmetric over-credit elsewhere keeps the SUM
    # unchanged (a stale-read withdraw, where every agent sharing a cell
    # reads the same undrained stock via a naive np.add.at scatter, does
    # exactly this: conserves total mass but drives cell_res negative). Both
    # array mins must stay non-negative for the conservation check to mean
    # anything physically.
    assert ca.cell_res.min() >= 0.0
    assert aa.inv[: aa.n].min() >= 0.0


def test_mass_is_exactly_conserved_with_uneven_sharing_and_flows():
    """Signal check for the mass-conservation test above: a scenario with an
    ODD number of agents sharing each settlement (so the multi-resident fair
    withdrawal path actually runs) plus a flow-eligible gap between settlements wide
    enough that stage 3 moves real mass - a naive gather/scatter without the
    serialization would double count or lose mass exactly in this shared-cell
    case, so this is not redundant with the simpler 6-agent-split test above."""
    aa, ca = _setup(n_agents=7, seed=2)
    ca.cell_res[2, 2, C.R["water"]] = 100.0
    cfg = RedistributionConfig(flow_radius=10, flow_rate_per_step=3.0)
    before = {res: total_mass(aa, ca, res) for res in C.RESOURCES}
    needs = compute_needs(ca, cfg)
    out = redistribute(aa, ca, needs, cfg)
    assert out["flows"], "scenario must actually exercise stage 3 to be a meaningful conservation check"
    for res in C.RESOURCES:
        assert np.isclose(total_mass(aa, ca, res), before[res], atol=1e-5), res
    # See test_mass_is_exactly_conserved's comment: sum-conservation alone
    # would not catch a stale-read over-draw that goes negative on one side
    # and over-credits the other by the same amount.
    assert ca.cell_res.min() >= 0.0
    assert aa.inv[: aa.n].min() >= 0.0


def test_agents_end_at_or_below_knapsack_when_pool_allows():
    aa, ca = _setup()
    ca.cell_res[2, 2, C.R["water"]] = 100.0
    cfg = RedistributionConfig()
    needs = compute_needs(ca, cfg)
    redistribute(aa, ca, needs, cfg)
    on_rich = (aa.x[: aa.n] == 2) & (aa.y[: aa.n] == 2)
    water = aa.inv[: aa.n, C.R["water"]][on_rich]
    # tutti alla soglia-zaino (coperti dal pool ricco: deficit colmato, surplus versato)
    assert np.allclose(water, cfg.knapsack_targets["water"], atol=1e-5)


def test_flow_moves_water_toward_needier_settlement():
    aa, ca = _setup()
    ca.cell_res[2, 2, C.R["water"]] = 100.0
    cfg = RedistributionConfig(flow_radius=10, flow_rate_per_step=5.0)
    poor_before = float(ca.cell_res[6, 6, C.R["water"]])
    needs = compute_needs(ca, cfg)
    out = redistribute(aa, ca, needs, cfg)
    assert ca.cell_res[6, 6, C.R["water"]] > poor_before
    assert any(f[2] == C.R["water"] for f in out["flows"])


def test_agents_outside_settlements_are_untouched():
    aa, ca = _setup()
    row = 0
    aa.x[row], aa.y[row] = 4, 0  # cella non insediata
    ca.occupancy[:] = 0
    np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)
    inv_before = aa.inv[row].copy()
    cfg = RedistributionConfig()
    redistribute(aa, ca, compute_needs(ca, cfg), cfg)
    assert np.allclose(aa.inv[row], inv_before)


def test_founder_reservation_skips_deposit_but_keeps_withdrawal_enabled():
    aa, ca = _single_cell_setup(n_agents=2, water_stock=1.0)
    cfg = RedistributionConfig()
    water = C.R["water"]
    aa.inv[: aa.n] = 0.0
    aa.inv[0, water] = 7.0
    aa.inv[1, water] = 5.0
    before = total_mass(aa, ca, "water")

    out = redistribute(
        aa,
        ca,
        compute_needs(ca, cfg),
        cfg,
        deposit_exempt_rows=np.asarray([0], dtype=np.intp),
    )

    # The founder keeps its expedition reserve; the ordinary resident still
    # deposits down to the configured target.  The founder remains eligible
    # for stage-2 withdrawals (irrelevant here because it is already above
    # target), so the exemption is deliberately one-way.
    assert aa.inv[0, water] == 7.0
    assert aa.inv[1, water] == cfg.knapsack_targets["water"]
    assert np.isclose(
        out["deposited"], 5.0 - cfg.knapsack_targets["water"]
    )
    assert np.isclose(total_mass(aa, ca, "water"), before, atol=1e-5)


def _single_cell_setup(n_agents: int, water_stock: float, pos=(2, 2)):
    """Minimal single-settlement rig for the shared-cell withdraw-split test
    below: every agent starts with water=0 (BaseAgent's dataclass default,
    4.0, is already above the 2.0 knapsack target and would generate a
    stage-1 deposit that muddies the scenario, so it is explicitly zeroed)
    and shares the same cell, whose warehouse is seeded to `water_stock`."""
    x, y = pos
    w = _grid(4, 4)
    w.get_cell(x, y).structures.append(Structure(type=StructureType.SHELTER, x=x, y=y))
    agents = {}
    for i in range(n_agents):
        a = BaseAgent(agent_id=f"a{i}", name=f"a{i}", role="colonist", x=x, y=y)
        a.inventory.water = 0.0
        agents[a.agent_id] = a
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[:] = 0
    np.add.at(ca.occupancy, (aa.y[:aa.n], aa.x[:aa.n]), 1)
    ca.cell_res[y, x, C.R["water"]] = water_stock
    return aa, ca


def test_shared_cell_withdraw_splits_fairly_when_stock_below_total_deficit():
    """Closes the gap the two conservation tests above do not cover: neither
    forces `stock < sum(per-agent deficit)` on a single SHARED cell, so
    neither would catch a stale-read over-draw (every agent reading the same
    undrained stock via a naive `np.add.at` scatter-subtract) - that variant
    still conserves total mass exactly (it manufactures a negative cell_res
    balanced by an equal over-credit to every agent's inventory), so it slips
    past `np.isclose(total_mass, before)` undetected; only a non-negativity
    check or an exact-split check like this one catches it.

    Hand-verified scenario: 3 agents share one settled cell, all starting at
    water=0.0 (knapsack target 2.0, so deficit=2.0 each - 6.0 total), but the
    cell warehouse holds only 1.0 unit - less than even a single agent's
    deficit. The warehouse must be split proportionally: identical deficits
    receive identical shares and stock never becomes negative."""
    aa, ca = _single_cell_setup(n_agents=3, water_stock=1.0)
    cfg = RedistributionConfig()
    before_total = total_mass(aa, ca, "water")
    needs = compute_needs(ca, cfg)
    redistribute(aa, ca, needs, cfg)
    assert np.allclose(aa.inv[:3, C.R["water"]], [1.0 / 3.0] * 3), aa.inv[:3, C.R["water"]]
    assert ca.cell_res[2, 2, C.R["water"]] == 0.0
    assert np.isclose(total_mass(aa, ca, "water"), before_total, atol=1e-5)
    assert ca.cell_res.min() >= 0.0
    assert aa.inv[:3].min() >= 0.0


def test_med_kits_participate_in_knapsack_but_never_in_flows():
    """med_kits has a knapsack target (stages 1-2) but no column in
    src.core.constants.NEEDS, so it must never appear as a stage-3 flow
    resource index - proves _FLOW_NEED_MAP's deliberate omission actually
    holds at runtime, not just in a docstring."""
    aa, ca = _setup()
    for row in range(aa.n):
        aa.inv[row, C.R["med_kits"]] = 5.0  # well above the default 1.0 target -> surplus to deposit
    cfg = RedistributionConfig(flow_radius=10, flow_rate_per_step=5.0)
    needs = compute_needs(ca, cfg)
    out = redistribute(aa, ca, needs, cfg)
    assert all(f[2] != C.R["med_kits"] for f in out["flows"])


def test_adjacent_covered_cell_supplies_outpost_construction_inputs():
    """A covered mother cell may supply the survival gaps of an adjacent
    outpost even though both cells have occupants retaining personal targets.
    Construction material, minerals and energy keep separate demand ledgers.
    """
    w = _grid(4, 4)
    mother = w.get_cell(1, 1)
    outpost = w.get_cell(2, 1)
    for structure_type in (
        StructureType.SHELTER,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        mother.structures.append(
            Structure(type=structure_type, x=mother.x, y=mother.y)
        )
    outpost.structures.append(
        Structure(type=StructureType.GREENHOUSE, x=outpost.x, y=outpost.y)
    )
    agents = {
        "mother": BaseAgent("mother", "mother", "colonist", 1, 1),
        "outpost": BaseAgent("outpost", "outpost", "colonist", 2, 1),
    }
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    ca.occupancy[1, 1] = 1
    ca.occupancy[1, 2] = 1
    for resource in ("construction_material", "minerals", "energy"):
        ca.cell_res[1, 1, C.R[resource]] = 10.0
    cfg = RedistributionConfig(flow_radius=1, flow_rate_per_step=5.0)
    before = {resource: total_mass(aa, ca, resource) for resource in C.RESOURCES}

    out = redistribute(aa, ca, compute_needs(ca, cfg), cfg)

    flowed = {C.RESOURCES[int(flow[2])] for flow in out["flows"]}
    assert {"construction_material", "minerals", "energy"} <= flowed
    assert ca.cell_res[1, 2, C.R["construction_material"]] > 0.0
    assert ca.cell_res[1, 2, C.R["minerals"]] > 0.0
    assert ca.cell_res[1, 2, C.R["energy"]] > 0.0
    for resource in C.RESOURCES:
        assert np.isclose(total_mass(aa, ca, resource), before[resource], atol=1e-5)


def test_default_flow_reaches_adjacent_but_not_distant_settlement():
    aa, ca = _setup()
    cfg = RedistributionConfig()
    assert cfg.flow_radius == 1
    ca.cell_res[2, 2, C.R["water"]] = 100.0
    # Existing second settlement is four cells away, outside the default.
    out = redistribute(aa, ca, compute_needs(ca, cfg), cfg)
    assert not any(flow[2] == C.R["water"] for flow in out["flows"])
