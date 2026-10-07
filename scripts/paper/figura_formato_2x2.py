# -*- coding: utf-8 -*-
"""Figura del quadrato 2x2: formato (scrive / sceglie) per impostazione della run.

Territorio a fine run meno quello della colonia senza governo dello stesso
seme, un punto per seme (3-7), il trattino e' la media. Due pannelli:
impostazione della tesi (1000 passi, cadenza 25; confronto con
`runs/base_qwen/ctrl_none`) e impostazione della campagna System 1 v4 (2000
passi, cadenza 20; confronto con `campaign_v2h none_s*`).

Bracci:
- Qwen scrive: `runs/controllo_copertura_20260926/variante_D_vera*` piu'
  `runs/paper_qwen/variante_D{,_rep2}` (tesi, quattro esecuzioni con la stessa
  configurazione) e `runs/confronto_system1/system1/qwen_llm*` (System 1, due);
- Qwen sceglie: `runs/confronto_system1/tesi/qwen_s1_s*` e
  `campaign_v4_20260924/qwen_s*`;
- Jev sceglie: `runs/confronto_system1/tesi/jev_s1_s*` e `campaign_v4_20260924/jev_s*`.

Scrive in italiano (nota) e in inglese (articolo):
    python docs/nota_professore/figura_formato_2x2.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
QUI = Path(__file__).resolve().parent
PAPER = ROOT / "figures"
SEMI = range(3, 8)
COLORI = {"scrive": "#C1440E", "qwen": "#2E6F95", "jev": "#7B61A8"}


def celle(p, seme):
    rs = json.loads(Path(ROOT / p).read_text(encoding="utf-8"))
    for r in rs if isinstance(rs, list) else [rs]:
        if int(r.get("seed", seme)) == seme:
            return int(r["espansione"]["celle_totali"])
    raise KeyError(p)


def dati():
    bt = {s: celle("runs/base_qwen/ctrl_none/results.json", s) for s in SEMI}
    bv = {s: celle(f"runs/jev_semif_experiments/campaign_v2h_20260923/none_s{s}/results.json", s) for s in SEMI}
    d = lambda pat, base: [celle(pat.format(s=s), s) - base[s] for s in SEMI]  # noqa: E731
    return {
        "tesi": [
            ("scrive", [d("runs/controllo_copertura_20260926/variante_D_vera_s{s}/results.json", bt),
                        d("runs/controllo_copertura_20260926/variante_D_vera_rep2_s{s}/results.json", bt),
                        d("runs/paper_qwen/variante_D/results.json", bt),
                        d("runs/paper_qwen/variante_D_rep2/results.json", bt)]),
            ("qwen", [d("runs/confronto_system1/tesi/qwen_s1_s{s}/results.json", bt)]),
            ("jev", [d("runs/confronto_system1/tesi/jev_s1_s{s}/results.json", bt)]),
        ],
        "s1": [
            ("scrive", [d("runs/confronto_system1/system1/qwen_llm_s{s}/results.json", bv),
                        d("runs/confronto_system1/system1/qwen_llm_rep2_s{s}/results.json", bv)]),
            ("qwen", [d("runs/jev_semif_experiments/campaign_v4_20260924/qwen_s{s}/results.json", bv)]),
            ("jev", [d("runs/jev_semif_experiments/campaign_v4_20260924/jev_s{s}/results.json", bv)]),
        ],
    }


TESTI = {
    "it": dict(pannelli={"tesi": "impostazione della tesi\n1000 passi, cadenza 25",
                         "s1": "impostazione System 1\n2000 passi, cadenza 20"},
               righe={"scrive": "Qwen scrive la legge", "qwen": "Qwen sceglie", "jev": "Jev sceglie"},
               asse="celle occupate meno la colonia senza governo dello stesso seme"),
    "en": dict(pannelli={"tesi": "main setting: 1000 steps, law every 25",
                         "s1": "System-1 setting: 2000 steps, law every 20"},
               righe={"scrive": "Qwen writes", "qwen": "Qwen chooses", "jev": "Jev chooses"},
               asse="occupied cells minus the ungoverned colony of the same seed"),
}


def disegna(lingua, D, out, larghezza, altezza, font, impilati=False):
    T = TESTI[lingua]
    plt.rcParams.update({"font.size": font, "font.family": "DejaVu Sans"})
    fig, assi = plt.subplots(2 if impilati else 1, 1 if impilati else 2, figsize=(larghezza, altezza),
                             sharey=True, sharex=True)
    ordine = ["scrive", "qwen", "jev"]
    for ax, chiave in zip(assi, ("tesi", "s1")):
        ax.axvline(0, color="#1f2430", lw=0.9)
        for y, (braccio, esecuzioni) in enumerate(sorted(D[chiave], key=lambda b: ordine.index(b[0]))):
            y = 2 - y
            tutti = [v for e in esecuzioni for v in e]
            for k, e in enumerate(esecuzioni):
                off = (k - (len(esecuzioni) - 1) / 2) * 0.16
                ax.scatter(e, [y + off] * len(e), s=16, color=COLORI[braccio], alpha=0.85, zorder=3,
                           edgecolor="white", linewidth=0.5)
            m = st.mean(tutti)
            ax.plot([m, m], [y - 0.28, y + 0.28], color="#1f2430", lw=2.2, zorder=4)
            ax.text(m, y + 0.34, f"{m:+.0f}", ha="center", va="bottom", fontsize=font - 0.5, color="#1f2430")
        ax.set_title(T["pannelli"][chiave], fontsize=font)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([T["righe"][b] for b in ordine])
        ax.grid(axis="x", color="#e4e1dd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(-0.6, 2.75)
    fig.supxlabel(T["asse"], fontsize=font)
    fig.tight_layout()
    for est in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + est), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    D = dati()
    disegna("it", D, QUI / "formato_2x2", 9.0, 3.6, 10)
    PAPER.mkdir(exist_ok=True)
    disegna("en", D, PAPER / "formato_2x2", 3.35, 3.7, 7, impilati=True)
    print("scritte formato_2x2 (it) e figures/formato_2x2 (en)")


if __name__ == "__main__":
    main()
