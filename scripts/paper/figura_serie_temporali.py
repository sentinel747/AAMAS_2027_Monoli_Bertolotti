"""Paper figure: occupied cells and population over 1000 steps, by model.

Rows: occupied cells (the paper's territory measure, `occupied_cells` in
state_timeseries.csv, equal to `celle_totali` in results.json) and population.
Columns: Qwen3.8-27B (runs/paper_qwen) and gpt-oss-20b (runs/paper_oss).
In each panel every governed run of the seven prompt arms (governor and
administrators) is a thin line at low opacity, their mean is bold, and the five
ungoverned colonies of the same campaign are grey and dashed.

Usage: python scripts/paper/figura_serie_temporali.py  (writes figures/time_series.pdf/.png)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "time_series"
BRACCI = ["variante_0", "variante_A", "variante_B", "variante_C", "variante_D",
          "variante_D_rep2", "variante_F"]
MODELLI = [("Qwen3.8-27B", ROOT / "runs/paper_qwen", "#c2410c"),
           ("gpt-oss-20b", ROOT / "runs/paper_oss", "#1d4ed8")]
SERIE = [("occupied_cells", "occupied cells"), ("population", "colonists")]
GRIGIO = "#6b6f78"


def leggi(cartella: Path):
    """step, {colonna: array} for every seed run found under a campaign arm."""
    fuori = []
    for run in sorted(cartella.glob("*seed*")):
        f = run / "state_timeseries.csv"
        if not f.exists():
            continue
        righe = list(csv.DictReader(f.open(encoding="utf-8")))
        if len(righe) < 1000:  # incomplete run: not in the paper
            continue
        passi = np.array([float(r["step"]) for r in righe])
        fuori.append((passi, {c: np.array([float(r[c]) for r in righe]) for c, _ in SERIE}))
    return fuori


plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, assi = plt.subplots(2, 2, figsize=(3.35, 2.9), sharex=True, sharey="row",
                         constrained_layout=True)
for j, (nome, campagna, colore) in enumerate(MODELLI):
    base = leggi(campagna / "ctrl_none")
    gov = [r for b in BRACCI for r in leggi(campagna / b)]
    print(f"{nome}: {len(gov)} governed runs, {len(base)} ungoverned")
    for i, (col, etichetta) in enumerate(SERIE):
        ax = assi[i][j]
        for passi, d in gov:
            ax.plot(passi, d[col], color=colore, lw=0.5, alpha=0.12)
        for passi, d in base:
            ax.plot(passi, d[col], color=GRIGIO, lw=0.6, alpha=0.45, ls=(0, (3, 2)))
        n = min(len(p) for p, _ in gov)
        ax.plot(gov[0][0][:n], np.mean([d[col][:n] for _, d in gov], axis=0),
                color=colore, lw=1.4, label="governed (mean)")
        n = min(len(p) for p, _ in base)
        ax.plot(base[0][0][:n], np.mean([d[col][:n] for _, d in base], axis=0),
                color=GRIGIO, lw=1.2, ls=(0, (3, 2)), label="ungoverned (mean)")
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.tick_params(length=2, pad=1.5)
        if i == 0:
            ax.set_title(nome, fontsize=7, color=colore, pad=3)
        if j == 0:
            ax.set_ylabel(etichetta, fontsize=7)
        if i == 1:
            ax.set_xlabel("step (week)", fontsize=6.5)
            ax.set_xticks([0, 250, 500, 750, 1000])
h, l = assi[0][0].get_legend_handles_labels()
assi[0][0].legend(h, l, fontsize=5.5, frameon=False, loc="upper left", handlelength=1.8)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
print("written", OUT.with_suffix(".pdf"))
