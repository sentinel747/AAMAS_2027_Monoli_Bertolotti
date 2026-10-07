"""Task 10 (redistribution half): the ONE deliberate model addition on top of
the pure representation refactor (src/core/*). Disabled by default
(`RedistributionConfig.enabled = False`, src/core/needs.py) so the baseline
stays provably equivalent to the pre-refactor object engines - see
tests/core/test_kernel_redistribution_wiring.py's
`test_disabled_redistribution_is_a_no_op`.

This module remains purely a resource mechanism: the cell acts as a warehouse
and can ship surplus between nearby settled cells. The v3 preference policy
now separately turns the same need snapshot into a bounded task menu in
``src/core/cell_proposals.py``. It publishes priorities/quotas, never a named
agent assignment - see
`tests/core/test_needs.py::test_needs_matrix_not_imported_by_any_agents_module`,
which keeps ``src.core.kernel`` as the narrow integration seam.

Three stages, run in this order, each exactly mass-conserving:
  1. Deposit: every settled agent's surplus above its knapsack target (for
     each resource in `cfg.knapsack_targets`) is scattered into its cell's
     warehouse (`cells.cell_res`). Pure vectorized `np.add.at` scatter-add -
     commutative, so no read-after-write hazard even when several agents
     share a cell.
  2. Withdraw: every settled agent below its knapsack target draws from its
     cell warehouse. If stock is insufficient, residents receive the same
     fraction of their individual deficit. The operation is mass-conserving
     and does not reward insertion order.
  3. Flow: among settled cells only, a donor cell (one whose own need for a
     resource is currently zero) ships up to `flow_rate_per_step` units
     toward the neediest settled cell within `flow_radius` (Chebyshev
     distance), for as long as that recipient still has unmet need. Only
     resources with a real needs-matrix column can flow this way
     (`_FLOW_NEED_MAP` below) - `med_kits` participates in stages 1-2 (it has
     a knapsack target) but not stage 3 (there is no `med_kits` column in
     `src.core.constants.NEEDS`).
"""

from __future__ import annotations

import numpy as np

from src.core import constants as C
from src.core.needs import RedistributionConfig, settled_mask
from src.world.structures import BUILD_COSTS, StructureType

# Maps a resource index name to the needs-matrix column it feeds for stage 3.
# `construction_material` and `minerals` both feed the aggregate "materials"
# need (see src/core/needs.py's own materials-need docstring for why the two
# are combined). Deliberately does NOT include `med_kits`: NEEDS
# (src/core/constants.py) has no matching column, so a `med_kits` donor could
# never find a needy recipient - it stays a stage-1/2-only resource.
_FLOW_NEED_MAP: dict[str, str] = {
    "water": "water",
    "food": "food",
    "oxygen": "oxygen",
}


def _infrastructure_flow_demand(needs: np.ndarray, resource: str) -> np.ndarray:
    """Resource units required to close each cell's survival coverage gap.

    Residents already retain their configured personal target in stage 1.
    Using the generic materials target as the donor gate made every populated
    warehouse appear needy and disabled construction flows in real runs.
    """
    demand = np.zeros(needs.shape[:2], dtype=np.float64)
    for need_name, structure_type in (
        ("build_solar", StructureType.SOLAR_ARRAY),
        ("build_oxygen_plant", StructureType.OXYGEN_PLANT),
        ("build_greenhouse", StructureType.GREENHOUSE),
    ):
        cost = float(getattr(BUILD_COSTS[structure_type], resource, 0.0))
        if cost > 0.0:
            demand += needs[..., C.ND[need_name]] * cost
    shelter_cost = float(
        getattr(BUILD_COSTS[StructureType.SHELTER], resource, 0.0)
    )
    if shelter_cost > 0.0:
        demand += needs[..., C.ND["habitat_pressure"]] * shelter_cost
    return demand


def _flow_demand(needs: np.ndarray, resource: str) -> np.ndarray:
    need_name = _FLOW_NEED_MAP.get(resource)
    if need_name is not None:
        return needs[..., C.ND[need_name]].astype(np.float64, copy=True)
    return _infrastructure_flow_demand(needs, resource)


def _need_weighted_withdraw(cell_res, ri: int, inv, ii: int, rows: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                            deficit: np.ndarray) -> float:
    """Fill same-cell deficits fairly without manufacturing resources.

    When the warehouse cannot satisfy everybody, every resident receives the
    same fraction of their own deficit.  The old ascending-row drain could
    give the entire stock to the first agent and nothing to identical peers.
    """
    if rows.size == 0:
        return 0.0
    assert np.all(rows[:-1] <= rows[1:]), (
        "_need_weighted_withdraw requires rows in deterministic order"
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(rows.size):
        key = (int(ys[i]), int(xs[i]))
        groups.setdefault(key, []).append(i)

    withdrawn = 0.0
    for key, idxs in groups.items():
        y, x = key
        local = np.asarray(idxs, dtype=np.intp)
        local_deficit = np.maximum(0.0, deficit[local])
        total_deficit = float(local_deficit.sum())
        stock = max(0.0, float(cell_res[y, x, ri]))
        if total_deficit <= 0.0 or stock <= 0.0:
            continue
        scale = min(1.0, stock / total_deficit)
        take = local_deficit * scale
        total_take = float(take.sum())
        cell_res[y, x, ri] -= total_take
        inv[rows[local], ii] += take
        withdrawn += total_take
    return withdrawn


def redistribute(
    agents,
    cells,
    needs: np.ndarray,
    cfg: RedistributionConfig,
    deposit_exempt_rows: np.ndarray | None = None,
) -> dict:
    """Run the three stages described in this module's docstring. `needs` is
    computed ONCE by the caller (`src.core.needs.compute_needs`) before this
    call and mutated in place only by stage 3 (each satisfied unit of need is
    subtracted as it is delivered) - stages 1-2 do not recompute it, matching
    the reference plan's own design (a caller wanting a fully up-to-date
    needs snapshot mid-stage would call `compute_needs` again itself).

    KNOWN CHARACTERISTIC, not a bug: because `needs` is the pre-stage-1/2
    snapshot, a cell drained of its own donatable surplus by its OWN agents
    during stage 2 can still show up as a stage-3 donor - `needs[fy, fx, ni]
    > 0.0` reflects the deficit as it stood before this call, not the deficit
    after stage 2 actually served that cell's agents. A caller enabling this
    feature for a real run should be aware a cell can ship away resources in
    stage 3 that its own residents needed and received only moments earlier
    in stage 2, within the same `redistribute()` call.

    `deposit_exempt_rows` protects agents that are actively assembling or
    carrying a founder kit from stage-1 deposits.  They still participate in
    stage-2 withdrawals, and all other residents retain the normal leveling
    behavior.  This is a reservation of existing mass, never resource
    creation.

    Returns `{"deposited": float, "withdrawn": float, "flows": list[tuple]}`
    where each flow is `((from_x, from_y), (to_x, to_y), res_idx, amount)`.
    """
    settled = settled_mask(cells)
    rows = agents.alive_rows()
    ay, ax = agents.y[rows], agents.x[rows]
    in_settled = settled[ay, ax]
    srows, sy, sx = rows[in_settled], ay[in_settled], ax[in_settled]
    deposited = 0.0
    withdrawn = 0.0
    exempt = np.asarray(
        [] if deposit_exempt_rows is None else deposit_exempt_rows,
        dtype=np.intp,
    )
    deposit_allowed = ~np.isin(srows, exempt) if srows.size else np.empty(0, dtype=np.bool_)

    # ---- Stage 1 + 2: knapsack leveling for every resource with a target. ----
    # Early guard hoisted OUT of the per-resource loop: `srows` does not
    # change across iterations, so checking it once here (instead of on every
    # pass, as before) skips the whole stage 1+2 loop in one comparison when
    # no settled agent exists - stage 3 below does not depend on `srows` at
    # all (it only reads `settled`/`needs`/`cells.cell_res`), so it must NOT
    # be skipped by this guard.
    if srows.size:
        for res, target in sorted(cfg.knapsack_targets.items()):
            ri = C.R[res]
            surplus = np.maximum(0.0, agents.inv[srows, ri] - target)
            surplus[~deposit_allowed] = 0.0
            agents.inv[srows, ri] -= surplus
            np.add.at(cells.cell_res[:, :, ri], (sy, sx), surplus)
            deposited += float(surplus.sum())

            deficit = np.maximum(0.0, target - agents.inv[srows, ri])
            withdrawn += _need_weighted_withdraw(cells.cell_res, ri, agents.inv, ri, srows, sy, sx, deficit)

    # ---- Stage 3: flows between settled cells within flow_radius. ----
    flows: list[tuple] = []
    ys, xs = np.nonzero(settled)
    if len(ys) > 1:
        flow_resources = (
            *sorted(_FLOW_NEED_MAP),
            "construction_material",
            "energy",
            "minerals",
        )
        for res in flow_resources:
            ri = C.R[res]
            remaining_need = _flow_demand(needs, res)
            for k in range(len(ys)):
                fy, fx = int(ys[k]), int(xs[k])
                if remaining_need[fy, fx] > 0.0:  # a cell in deficit does not donate
                    continue
                stock = float(cells.cell_res[fy, fx, ri])
                budget = min(cfg.flow_rate_per_step, stock)
                if budget <= 0.0:
                    continue
                d = np.maximum(np.abs(ys - fy), np.abs(xs - fx))
                cand = (d > 0) & (d <= cfg.flow_radius)
                if not cand.any():
                    continue
                cn = remaining_need[ys[cand], xs[cand]]
                if cn.max() <= 0.0:
                    continue
                j = np.flatnonzero(cand)[int(np.argmax(cn))]
                ty, tx = int(ys[j]), int(xs[j])
                amount = min(budget, float(remaining_need[ty, tx]))
                if amount <= 0.0:
                    continue
                cells.cell_res[fy, fx, ri] -= amount
                cells.cell_res[ty, tx, ri] += amount
                remaining_need[ty, tx] = max(
                    0.0, remaining_need[ty, tx] - amount
                )
                flows.append(((fx, fy), (tx, ty), ri, amount))

    return {"deposited": deposited, "withdrawn": withdrawn, "flows": flows}
