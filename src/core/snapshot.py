"""Task 11: `SnapshotBuffer`, a NEW, additive, high-frequency columnar
recording channel for the core kernel (`src/core/kernel.py`'s `CoreState`/
`StepOutcome`).

GOVERNING CONSTRAINT (task-11-brief.md): this module changes nothing about
simulation behavior - it only *observes* `AgentArrays`/`CellArrays`/
`StepOutcome` after `src.core.kernel.step` has already run and copies a light
columnar slice of them into pre-allocated numpy buffers, later flushed to a
single compressed `.npz`. It does NOT replace the existing per-step JSON
payloads the shells emit (`src/api/state_store.py`'s `_world_snapshot_payload`,
`src/simulation/agent_coupled_runner.py`'s twin) - those stay exactly as they
are, and this module is not wired into either shell by this task. Nothing
here reads wall-clock time: `step`/`day` are supplied by the caller (usually
the same `step_index`/`day_start` a `kernel.step` call already used), matching
the deterministic-replay discipline the rest of `src/core` follows.

Per-step data recorded:
  - scalars: `steps` (the caller's `step` index), `day`, `death_count`
    (`len(outcome.deaths)`), `knowledge_gain` (`outcome.knowledge_gain`),
    `flow_count` (`len(outcome.flows)`) - light `StepOutcome` summaries, not
    the outcome itself (memory events / action results / raw flows are not
    columnar and are exactly what the untouched JSON payload channel already
    carries).
  - per-agent columns (`AGENT_COLUMNS` below), each a `[n]` slice of the
    matching `AgentArrays` column at append time: `x, y, alive, health,
    oxygen, hydration, satiety, fatigue, stress, morale`.
  - light per-cell aggregates (not the full `[H, W, ...]` grids - this is the
    "light aggregate summaries for cells" the spec calls for, not a second
    full-grid snapshot channel): mean habitability, total occupancy, total
    struct_count, and the total of every resource column in `cell_res`
    (`[NR]` per step).

Growth: both the step dimension and the per-step agent-column width grow by
doubling on overflow (the same pattern `AgentArrays._grow` already uses in
`src/core/arrays.py`), so `capacity_steps` is a sizing hint, not a hard cap -
`append` never raises just because more steps or more agents arrive than the
constructor guessed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays

# Agent column name -> dtype. Order matches the brief's own list.
AGENT_COLUMNS: dict[str, Any] = {
    "x": np.int16,
    "y": np.int16,
    "alive": np.bool_,
    "health": np.float32,
    "oxygen": np.float32,
    "hydration": np.float32,
    "satiety": np.float32,
    "fatigue": np.float32,
    "stress": np.float32,
    "morale": np.float32,
}


class SnapshotBuffer:
    """Pre-allocated columnar recorder. `capacity_steps` sizes the initial
    step dimension; both it and the per-step agent-column width grow (double)
    on overflow rather than raising."""

    def __init__(self, capacity_steps: int) -> None:
        self._capacity_steps = max(1, int(capacity_steps))
        self._capacity_agents = 0
        self._n_steps = 0

        self.steps = np.zeros(self._capacity_steps, dtype=np.int64)
        self.day = np.zeros(self._capacity_steps, dtype=np.float64)
        self.death_count = np.zeros(self._capacity_steps, dtype=np.int32)
        self.knowledge_gain = np.zeros(self._capacity_steps, dtype=np.float32)
        self.flow_count = np.zeros(self._capacity_steps, dtype=np.int32)

        self.cell_habitability_mean = np.zeros(self._capacity_steps, dtype=np.float32)
        self.cell_occupancy_total = np.zeros(self._capacity_steps, dtype=np.int64)
        self.cell_struct_count_total = np.zeros(self._capacity_steps, dtype=np.int64)
        self.cell_resource_total = np.zeros((self._capacity_steps, C.NR), dtype=np.float32)

        self._agent_cols: dict[str, np.ndarray] = {}

    # ---- growth helpers, mirroring AgentArrays._grow's doubling pattern ----

    def _grow_steps(self) -> None:
        new_capacity = self._capacity_steps * 2
        for name in (
            "steps", "day", "death_count", "knowledge_gain", "flow_count",
            "cell_habitability_mean", "cell_occupancy_total", "cell_struct_count_total",
        ):
            old = getattr(self, name)
            grown = np.zeros((new_capacity,) + old.shape[1:], dtype=old.dtype)
            grown[: self._capacity_steps] = old
            setattr(self, name, grown)
        old_res = self.cell_resource_total
        grown_res = np.zeros((new_capacity, C.NR), dtype=old_res.dtype)
        grown_res[: self._capacity_steps] = old_res
        self.cell_resource_total = grown_res
        for name, old in self._agent_cols.items():
            grown = np.zeros((new_capacity, old.shape[1]), dtype=old.dtype)
            grown[: self._capacity_steps] = old
            self._agent_cols[name] = grown
        self._capacity_steps = new_capacity

    def _alloc_agent_columns(self, n: int) -> None:
        self._capacity_agents = max(n, 1)
        for name, dtype in AGENT_COLUMNS.items():
            self._agent_cols[name] = np.zeros((self._capacity_steps, self._capacity_agents), dtype=dtype)

    def _grow_agents(self, n: int) -> None:
        new_capacity = max(self._capacity_agents * 2, n)
        for name, old in self._agent_cols.items():
            grown = np.zeros((old.shape[0], new_capacity), dtype=old.dtype)
            grown[:, : self._capacity_agents] = old
            self._agent_cols[name] = grown
        self._capacity_agents = new_capacity

    # ---- public API ----

    def append(self, step: int, day: float, agents: AgentArrays, cells: CellArrays, outcome) -> None:
        """Copy a columnar slice of `agents`/`cells`/`outcome` at this step.
        `outcome` is a `src.core.kernel.StepOutcome` (only light scalar
        summaries are kept - see module docstring)."""
        n = int(agents.n)
        if not self._agent_cols:
            self._alloc_agent_columns(n)
        elif n > self._capacity_agents:
            self._grow_agents(n)

        if self._n_steps >= self._capacity_steps:
            self._grow_steps()

        idx = self._n_steps
        self.steps[idx] = int(step)
        self.day[idx] = float(day)
        self.death_count[idx] = len(outcome.deaths)
        self.knowledge_gain[idx] = float(outcome.knowledge_gain)
        self.flow_count[idx] = len(outcome.flows)

        self.cell_habitability_mean[idx] = float(cells.habitability.mean()) if cells.habitability.size else 0.0
        self.cell_occupancy_total[idx] = int(cells.occupancy.sum())
        self.cell_struct_count_total[idx] = int(cells.struct_count.sum())
        self.cell_resource_total[idx] = cells.cell_res.reshape(-1, C.NR).sum(axis=0)

        for name in AGENT_COLUMNS:
            dest = self._agent_cols[name]
            dest[idx, :] = 0
            if n:
                dest[idx, :n] = getattr(agents, name)[:n]

        self._n_steps += 1

    def flush_npz(self, path: Path) -> Path:
        """Write every recorded step (`[: self._n_steps]` of each buffer) to
        one `np.savez_compressed` `.npz` at `path`. Returns `path` (as a
        `Path`) for convenient chaining.

        `agent_*` keys are ALWAYS present in the written file, even when
        `append()` was never called: `_agent_cols` (and therefore the
        per-agent column arrays) is only populated on the first `append()`,
        so a buffer flushed with zero appends falls back to emitting one
        empty (`shape (n_steps, 0)`, i.e. `(0, 0)` when nothing was ever
        appended) array per `AGENT_COLUMNS` entry instead of omitting the
        keys outright. Callers that always expect `agent_x`/`agent_health`/
        etc. to be present (e.g. a shell loading this .npz into a UI) can
        rely on the key existing, just possibly empty."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = self._n_steps
        arrays: dict[str, np.ndarray] = {
            "steps": self.steps[:n],
            "day": self.day[:n],
            "death_count": self.death_count[:n],
            "knowledge_gain": self.knowledge_gain[:n],
            "flow_count": self.flow_count[:n],
            "cell_habitability_mean": self.cell_habitability_mean[:n],
            "cell_occupancy_total": self.cell_occupancy_total[:n],
            "cell_struct_count_total": self.cell_struct_count_total[:n],
            "cell_resource_total": self.cell_resource_total[:n],
        }
        if self._agent_cols:
            for name, col in self._agent_cols.items():
                arrays[f"agent_{name}"] = col[:n, : self._capacity_agents]
        else:
            for name, dtype in AGENT_COLUMNS.items():
                arrays[f"agent_{name}"] = np.zeros((n, 0), dtype=dtype)
        np.savez_compressed(path, **arrays)
        return path
