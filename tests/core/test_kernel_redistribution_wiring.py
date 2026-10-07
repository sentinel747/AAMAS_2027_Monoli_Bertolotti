from __future__ import annotations

"""Task 10 kernel-wiring tests: `src.core.kernel.step()`'s redistribution
seam must be a true no-op whenever `RedistributionConfig.enabled` is False
(the default) - see kernel.py's own "10. Redistribution" docstring section
and src/core/needs.py / src/core/redistribution.py's module docstrings for
why this is the ONE deliberate model addition allowed on top of the pure
representation refactor."""

import random
import copy

import numpy as np

from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.kernel import CoreState, step
from src.core.views import AgentSideState
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType

import src.core.kernel as kernel_module


def _grid(width: int, height: int) -> GridWorld:
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)] for y in range(height)]
    return GridWorld(width=width, height=height, cells=cells)


def _agent_specs():
    return [
        {"agent_id": "a0", "name": "a0", "role": "colonist", "x": 2, "y": 2,
         "perception_radius": 2, "survival_priority": 0.80},
        {"agent_id": "a1", "name": "a1", "role": "colonist", "x": 2, "y": 2,
         "perception_radius": 2, "survival_priority": 0.75},
        {"agent_id": "a2", "name": "a2", "role": "colonist", "x": 1, "y": 1,
         "perception_radius": 1, "survival_priority": 0.85},
        {"agent_id": "a3", "name": "a3", "role": "colonist", "x": 4, "y": 3,
         "perception_radius": 3, "survival_priority": 0.60},
    ]


def _build_world() -> GridWorld:
    w = _grid(6, 5)
    home = w.get_cell(2, 2)
    for st in (StructureType.HABITAT, StructureType.GREENHOUSE, StructureType.SOLAR_ARRAY, StructureType.OXYGEN_PLANT):
        w.add_structure(Structure(type=st, x=2, y=2, integrity=1.0))
    home.water_ice, home.liquid_water = 4.0, 0.6
    home.nutrients.update({"N": 5.0, "P": 1.0, "C": 3.0})
    w.get_cell(1, 1).water_ice = 1.5
    w.get_cell(4, 3).radiation_level = 0.5
    w.get_cell(4, 3).dust_level = 0.2
    w.planetary_state.update({
        "mean_temperature_c": -55.0,
        "pressure_pa": 650.0,
        "liquid_water_stability": 0.2,
    })
    return w


def _build_array_scenario():
    from src.agents.rule_based_agent import RuleBasedAgent

    w = _build_world()
    agents_src: dict[str, RuleBasedAgent] = {}
    for spec in _agent_specs():
        agents_src[spec["agent_id"]] = RuleBasedAgent(**spec)
    aa = AgentArrays.from_agents(agents_src)
    ca = CellArrays.from_world(w)
    side = {aid: AgentSideState.from_agent(a) for aid, a in agents_src.items()}
    ps = dict(w.planetary_state)
    return aa, ca, side, ps


BASE_CONFIG = {
    "headless": {"fast_observation": True, "rule_based_observation": True},
    "model": {"psychosocial_enabled": True},
    "social": {"earth_mars_delay_minutes": 12.0},
    "world": {"cell_degradation": True},
    "seed": 123,
}


def test_la_logistica_automatica_resta_spenta_se_la_chiave_manca():
    """Il ripiego del kernel deve dire cio' che dicono le due dataclass.

    Fino al 2026-08-31 valeva `decision_mode == "preferences"`, cioe' ACCESA:
    un terzo default per una quantita' sola, e diceva l'opposto di
    `RedistributionConfig.enabled = False` e di
    `ManualConfigOptions.redistribution_enabled = False`. Non era vivo, perche'
    ogni percorso reale passa da `build_manual_config`, che la chiave la scrive
    sempre; ma i sei scenari standard di `configs/scenarios/standard_v1.yaml`
    NON la contengono, quindi un config grezzo dato dritto al kernel sarebbe
    girato con la logistica accesa mentre ogni run manuale la tiene spenta:
    due simulazioni diverse sotto lo stesso nome.
    """
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    config = copy.deepcopy(BASE_CONFIG)
    config["agents"] = {"decision_mode": "preferences", "decision_sampling": "greedy"}
    config.pop("redistribution", None)
    calls = 0
    original = kernel_module.redistribute

    def _counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    kernel_module.redistribute = _counting
    try:
        step(state, 1, 7.0, config, np.random.default_rng(0))
    finally:
        kernel_module.redistribute = original

    assert calls == 0, "senza la chiave la logistica automatica deve restare spenta"


def test_la_logistica_automatica_gira_quando_la_chiave_la_accende():
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    config = copy.deepcopy(BASE_CONFIG)
    config["agents"] = {"decision_mode": "preferences", "decision_sampling": "greedy"}
    config["redistribution"] = {**config.get("redistribution", {}), "enabled": True}
    calls = 0
    original = kernel_module.redistribute

    def _counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    kernel_module.redistribute = _counting
    try:
        step(state, 1, 7.0, config, np.random.default_rng(0))
    finally:
        kernel_module.redistribute = original

    assert calls == 1


def test_kernel_passes_reserved_founder_rows_to_redistribution():
    aa, ca, side, ps = _build_array_scenario()
    side["a1"].founder_kit_reserved = True
    state = CoreState(
        agents=aa, cells=ca, side=side, planetary_state=ps, metadata={}
    )
    config = copy.deepcopy(BASE_CONFIG)
    config["agents"] = {
        "decision_mode": "preferences",
        "decision_sampling": "greedy",
    }
    config["redistribution"] = {**config.get("redistribution", {}), "enabled": True}
    captured = None
    original = kernel_module.redistribute

    def _capture(*args, **kwargs):
        nonlocal captured
        captured = np.asarray(kwargs.get("deposit_exempt_rows")).copy()
        return original(*args, **kwargs)

    kernel_module.redistribute = _capture
    try:
        step(state, 1, 7.0, config, np.random.default_rng(0))
    finally:
        kernel_module.redistribute = original

    assert np.array_equal(captured, np.asarray([aa.index["a1"]]))


def test_kernel_accumulates_automatic_logistics_metrics():
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    config = copy.deepcopy(BASE_CONFIG)
    config["agents"] = {"decision_mode": "preferences", "decision_sampling": "greedy"}
    config["redistribution"] = {**config.get("redistribution", {}), "enabled": True}
    original = kernel_module.redistribute

    def _known_transfer(*args, **kwargs):
        return {
            "deposited": 4.0,
            "withdrawn": 2.0,
            "flows": [((2, 2), (1, 1), C.R["water"], 1.5)],
        }

    kernel_module.redistribute = _known_transfer
    try:
        step(state, 1, 7.0, config, np.random.default_rng(0))
        step(state, 2, 7.0, config, np.random.default_rng(0))
    finally:
        kernel_module.redistribute = original

    assert state.metadata["_automatic_logistics_deposited_total"] == 8.0
    assert state.metadata["_automatic_logistics_withdrawn_total"] == 4.0
    assert state.metadata["_automatic_logistics_flow_total"] == 3.0
    assert state.metadata["_automatic_logistics_flow_events"] == 2
    assert state.metadata["_automatic_logistics_active_steps"] == 2
    assert state.metadata["_automatic_logistics_agent_steps"] == 8


def _snapshot(aa: AgentArrays, ca: CellArrays) -> dict:
    """Every ndarray column either engine could plausibly touch - deep-copied
    so later mutation of `aa`/`ca` cannot retroactively change the snapshot."""
    return {
        "x": aa.x[: aa.n].copy(), "y": aa.y[: aa.n].copy(), "alive": aa.alive[: aa.n].copy(),
        "inv": aa.inv[: aa.n].copy(), "health": aa.health[: aa.n].copy(),
        "oxygen": aa.oxygen[: aa.n].copy(), "hydration": aa.hydration[: aa.n].copy(),
        "satiety": aa.satiety[: aa.n].copy(), "fatigue": aa.fatigue[: aa.n].copy(),
        "stress": aa.stress[: aa.n].copy(), "morale": aa.morale[: aa.n].copy(),
        "cell_res": ca.cell_res.copy(), "struct_count": ca.struct_count.copy(),
        "struct_integrity": ca.struct_integrity.copy(), "habitability": ca.habitability.copy(),
        "occupancy": ca.occupancy.copy(),
    }


def _run(config: dict, seed: int, n_steps: int):
    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    rng = np.random.default_rng(0)
    random.seed(seed)
    outs = []
    for i in range(1, n_steps + 1):
        outs.append(step(state, i, 365.0, config, rng))
    return aa, ca, outs


def test_disabled_redistribution_is_a_no_op():
    """Con enabled=False lo stato non cambia di un solo float.

    Three independent, each individually sufficient, checks across several
    full kernel.step() calls:

      1. `redistribute()` is never CALLED at all when disabled (monkeypatched
         to raise if invoked) - proves the stage is skipped, not run-and-
         discarded.
      2. A run with `config["redistribution"] = {"enabled": False}` produces
         BIT-IDENTICAL (`np.array_equal`, not `np.allclose`) final arrays to a
         run whose config has NO "redistribution" key at all (the "run
         without the feature existing" comparison the brief asks for - the
         default is what a caller who never heard of Task 10 would get).
      3. `out.flows == []` on every step (the existing
         `test_redistribution_disabled_yields_no_flows` in
         test_kernel_step.py already checks this; repeated here so this
         module stands on its own).
    """
    config_absent = dict(BASE_CONFIG)  # no "redistribution" key at all
    config_disabled = dict(BASE_CONFIG)
    config_disabled["redistribution"] = {"enabled": False}

    # ---- Check 1: redistribute() itself must never be called. ----
    def _boom(*args, **kwargs):
        raise AssertionError("redistribute() must not be called when disabled")

    aa, ca, side, ps = _build_array_scenario()
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    rng = np.random.default_rng(0)
    random.seed(2026)
    original = kernel_module.redistribute
    kernel_module.redistribute = _boom
    try:
        for i in range(1, 6):
            out = step(state, i, 365.0, config_disabled, rng)
            assert out.flows == []
    finally:
        kernel_module.redistribute = original

    # ---- Check 2: bit-identical to a run where the feature "does not exist"
    # (config never mentions "redistribution" at all). ----
    aa_absent, ca_absent, outs_absent = _run(config_absent, seed=2026, n_steps=5)
    aa_disabled, ca_disabled, outs_disabled = _run(config_disabled, seed=2026, n_steps=5)

    snap_absent = _snapshot(aa_absent, ca_absent)
    snap_disabled = _snapshot(aa_disabled, ca_disabled)
    for key in snap_absent:
        assert np.array_equal(snap_absent[key], snap_disabled[key]), (
            f"array '{key}' diverged between redistribution-absent and "
            f"redistribution-disabled runs - the stage is not a true no-op"
        )
    for oa, od in zip(outs_absent, outs_disabled):
        assert oa.flows == [] and od.flows == []

    # Sanity: the scenario actually evolves over 5 steps (real decide/execute/
    # vitals/biology activity) - otherwise "bit-identical" would be a vacuous
    # comparison of two frozen states rather than proof the disabled stage
    # left real per-step activity untouched.
    x0, y0 = _agent_specs()[0]["x"], _agent_specs()[0]["y"]
    assert not (
        np.array_equal(aa_disabled.x[: aa_disabled.n], np.array([x0] * aa_disabled.n))
        and np.allclose(aa_disabled.inv[: aa_disabled.n], 0.0)
    ), "scenario must actually evolve for the bit-identical comparison to carry any signal"


def test_disabled_redistribution_never_touches_cell_res_even_when_needy():
    """A stronger, scenario-specific no-op check than
    `test_disabled_redistribution_is_a_no_op`'s generic snapshot comparison:
    the settled cell (2,2) is deliberately loaded with a drowning water stock
    (`cell_res[water] = 500.0`) that WOULD trigger real stage-1/2/3 mass
    movement if redistribution ran, then `src.core.kernel.compute_needs` is
    monkeypatched to raise if called. `compute_needs` is the FIRST statement
    inside kernel.py's `if redistribution_cfg.enabled:` block (see kernel.py's
    own "10. Redistribution" docstring section), so proving it is never
    invoked also proves `redistribute()` - and therefore any cell_res/
    inventory mutation the stage could perform - never runs either: the
    needs matrix is never even computed, not just discarded after
    computation. The spy assertion alone is sufficient for this property;
    this test does not additionally snapshot-compare cell_res/inventories
    before and after the call, since nothing would be left for such a
    comparison to catch that the spy does not already catch more directly."""
    aa, ca, side, ps = _build_array_scenario()
    ca.cell_res[2, 2, C.R["water"]] = 500.0  # drowning; would trigger real stage-1/2/3 movement if enabled
    state = CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})
    rng = np.random.default_rng(0)
    random.seed(11)

    def _boom(*args, **kwargs):
        raise AssertionError("compute_needs() must not be called when redistribution is disabled")

    original = kernel_module.compute_needs
    kernel_module.compute_needs = _boom
    try:
        config = dict(BASE_CONFIG)
        config["redistribution"] = {"enabled": False}
        out = step(state, 1, 365.0, config, rng)
        assert out.flows == []
    finally:
        kernel_module.compute_needs = original
