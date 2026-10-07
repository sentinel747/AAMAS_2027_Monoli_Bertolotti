"""Final maps for the AAMAS paper: reference world, seed 6, ungoverned vs governor+administrators.

Same logic as scripts/genera_figura_mappe_finali.py (shared window, shared colour
scale, every cell outlined), English labels, one-column width.
Usage: python scripts/paper/figura_mappe_finali.py  (writes figures/final_maps.pdf/.png)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
SEED = 6
RUNS = [
    ("Ungoverned", ROOT / f"runs/paper_qwen/ctrl_none/none_seed{SEED}", "#222222"),
    ("LLM governor + administrators", ROOT / f"runs/paper_qwen/variante_D/llm_completo-pD+amm_seed{SEED}", "#c2410c"),
]
OUT = ROOT / "figures" / "final_maps"


def load(run: Path):
    snap = sorted((run / "world_snapshots").glob("step_*.json"))[-1]
    d = json.loads(snap.read_text(encoding="utf-8"))
    cells = {}
    for c in d["cells"]:
        n = len(c.get("agents_present") or [])
        if n or c.get("structures"):
            cells[(c["x"], c["y"])] = n
    return cells, d["metrics"].get("population", 0)


data = [(name, *load(run), colour) for name, run, colour in RUNS]
xs = [x for _, cells, _, _ in data for x, _ in cells]
ys = [y for _, cells, _, _ in data for _, y in cells]
side = max(max(xs) - min(xs), max(ys) - min(ys)) + 3
x0 = (min(xs) + max(xs)) // 2 - side // 2
y0 = (min(ys) + max(ys)) // 2 - side // 2
vmax = max(max(cells.values()) for _, cells, _, _ in data)

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.95), constrained_layout=True)
for ax, (name, cells, pop, colour) in zip(axes, data):
    grid = np.full((side, side), np.nan)
    for (x, y), v in cells.items():
        grid[y - y0, x - x0] = max(v, 0.5)  # 0.5 = structures only, no colonist
    ax.imshow(grid, cmap="YlOrRd", norm=LogNorm(vmin=0.5, vmax=vmax),
              interpolation="nearest", origin="upper")
    ax.set_xticks(np.arange(-0.5, side, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, side, 1), minor=True)
    ax.xaxis.remove_overlapping_locs = False
    ax.yaxis.remove_overlapping_locs = False
    ax.grid(which="minor", color="#6b6b6b", linewidth=0.15, alpha=0.7)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color("#999999")
    ax.set_title(name, fontsize=7, pad=3, color=colour)
    ax.set_xlabel(f"{len(cells)} cells, {pop} colonists", fontsize=6.5, labelpad=2)
    print(name, len(cells), pop)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
print("written", OUT.with_suffix(".pdf"))
