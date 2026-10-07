"""Paper figure: seconds per decision, generating the law vs choosing among valid laws.

Medians with interquartile ranges, read from the run records:
- Qwen3.8-27B generating the law: governor_decisions.jsonl (`latency_s`) of the
  four executions of the reference arm (runs/controllo_copertura_20260926/
  variante_D_vera{,_rep2}_s*, runs/paper_qwen/variante_D{,_rep2});
- Qwen3.8-27B and Jev choosing (System 1): semantic_decisions.jsonl (`latency_ms`)
  of runs/confronto_system1/tesi/{qwen,jev}_s1_s*, the same main setting.

Usage: python scripts/paper/figura_latenza.py  (writes figures/latency.pdf/.png)
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "figures" / "latency"


def generazione(schemi):
    v = []
    for s in schemi:
        for f in glob.glob(str(RUNS / s / "**" / "governor_decisions.jsonl"), recursive=True):
            for riga in open(f, encoding="utf-8"):
                g = json.loads(riga).get("governor") or {}
                if g.get("latency_s"):
                    v.append(float(g["latency_s"]))
    return v


def scelta(schemi):
    v = []
    for s in schemi:
        for f in glob.glob(str(RUNS / s / "**" / "semantic_decisions.jsonl"), recursive=True):
            for riga in open(f, encoding="utf-8"):
                x = json.loads(riga).get("latency_ms")
                if x is not None:
                    v.append(float(x) / 1000.0)
    return v


BARRE = [
    ("Qwen3.8, writes the law", generazione(["controllo_copertura_20260926/variante_D_vera_s*",
                                              "controllo_copertura_20260926/variante_D_vera_rep2_s*",
                                              "paper_qwen/variante_D", "paper_qwen/variante_D_rep2"]), "#6b6f78"),
    ("Qwen3.8, chooses", scelta(["confronto_system1/tesi/qwen_s1_s*"]), "#c2410c"),
    ("Jev, chooses", scelta(["confronto_system1/tesi/jev_s1_s*"]), "#1d4ed8"),
]

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.35, 1.25), constrained_layout=True)
for y, (nome, v, colore) in enumerate(BARRE):
    q1, med, q3 = st.quantiles(v, n=4)[0], st.median(v), st.quantiles(v, n=4)[2]
    ax.barh(y, med, color=colore, height=0.55)
    ax.errorbar(med, y, xerr=[[med - q1], [q3 - med]], fmt="none", ecolor="#222222", lw=0.7, capsize=2)
    ax.text(q3 * 1.15, y, f"{med:.2g} s" if med < 10 else f"{med:.0f} s", va="center", fontsize=6.5)
    print(f"{nome}: n={len(v)}, median {med:.2f} s, IQR {q1:.2f}-{q3:.2f} s")
ax.set_yticks(range(len(BARRE)), [b[0] for b in BARRE])
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(0.1, 200)
ax.set_xlabel("seconds per decision (log scale)", fontsize=6.5)
ax.grid(axis="x", lw=0.3, alpha=0.4)
ax.tick_params(length=2, pad=1.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
print("written", OUT.with_suffix(".pdf"))
