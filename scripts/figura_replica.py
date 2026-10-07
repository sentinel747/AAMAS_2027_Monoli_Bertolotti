# -*- coding: utf-8 -*-
"""La figura della sezione «Che cosa replica su due modelli».

Undici effetti sul territorio occupato, in due pannelli affiancati, uno per
modello. Su ogni riga i cinque punti sono le differenze appaiate per seme, il
tratto verticale e' la media, la banda grigia e' il rumore della campagna
misurato dalle esecuzioni ripetute. La regola di riproduzione si legge senza
leggere la tabella: un effetto regge se i cinque punti stanno fuori dalla banda
dallo stesso lato in ENTRAMBI i pannelli; si inverte se stanno fuori da lati
opposti; e' un falso positivo di una campagna sola se stanno fuori in un
pannello e dentro nell'altro.

Stile allineato a `figure_confronto.py`: stesso corpo del carattere, stesso
grigio della baseline, stesso arancio delle coppie. I colori degli esiti
riprendono quelli del preambolo della tesi (verde di cio' che ne esce, ambra di
cio' che filtra), cosi' che la figura parli la stessa lingua dei diagrammi.

Uso:  python scripts/figura_replica.py
Scrive Tesi_LaTex/figures/replica_due_modelli.pdf e .png.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esiti_campagna import EFFETTI, RIPETIZIONE, differenze  # noqa: E402

CAMPAGNE = (("runs/base_qwen", "Qwen3.8-27B"), ("runs/base_pulita", "gpt-oss:20b"))

#: L'ordine delle righe e' quello della tabella in tesi: prima cio' che regge,
#: poi cio' che si inverte, poi il resto.
ORDINE = [
    ("governare: D meno baseline", "governare"),
    ("senza amministratori: gov meno D", "senza amministratori"),
    ("senza governatore: amm meno D", "senza governatore"),
    ("cecita': cieco meno D", "cecità di entrambi i livelli"),
    ("la gerarchia (it): B meno A", "la gerarchia (it)"),
    ("la lingua: D meno B", "la lingua (D meno B)"),
    ("la parola (it): A meno 0", "la parola (it)"),
    ("la parola (en): C meno F", "la parola (en)"),
    ("la gerarchia (en): D meno C", "la gerarchia (en)"),
    ("la lingua: C meno A", "la lingua (C meno A)"),
    ("la lingua: F meno 0", "la lingua (F meno 0)"),
]

COLORE_RIPRODOTTO = "#2F6B45"     # bordoEsito
COLORE_CONTRADDETTO = "#8A2F2F"   # tintaRetro
COLORE_UNA_SOLA = "#B5651D"       # bordoFiltro
COLORE_NESSUNA = "#4a4a4a"        # COLORE_BASE
COLORE_BANDA = "#d9d9d9"


def _forte(d: list[float], rumore: float) -> tuple[bool, int]:
    """(supera il rumore con 5/5 concordi, segno)."""
    n = len(d)
    media = sum(d) / n
    concordi = max(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
    return (concordi == n and abs(media) > rumore), (1 if media > 0 else -1)


def main() -> int:
    dati = {}
    for base, nome in CAMPAGNE:
        d = {et: differenze(base, a, b, "territorio") for et, a, b in EFFETTI}
        r = [abs(x) for x in differenze(base, *RIPETIZIONE, "territorio")]
        dati[nome] = (d, sum(r) / len(r))

    # L'esito per riga, con la stessa regola della tesi.
    esiti = {}
    for chiave, _ in ORDINE:
        forti = [_forte(dati[n][0][chiave], dati[n][1]) for _, n in CAMPAGNE]
        if all(f for f, _ in forti):
            esiti[chiave] = (COLORE_RIPRODOTTO if forti[0][1] == forti[1][1]
                             else COLORE_CONTRADDETTO)
        elif any(f for f, _ in forti):
            esiti[chiave] = COLORE_UNA_SOLA
        else:
            esiti[chiave] = COLORE_NESSUNA

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7, "pdf.fonttype": 42, "figure.dpi": 150,
    })
    fig, assi = plt.subplots(1, 2, figsize=(6.4, 3.6), sharey=True, sharex=True)
    y = list(range(len(ORDINE)))[::-1]

    for ax, (_, nome) in zip(assi, CAMPAGNE):
        d, rumore = dati[nome]
        ax.axvspan(-rumore, rumore, color=COLORE_BANDA, zorder=0, lw=0)
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        for (chiave, _), yy in zip(ORDINE, y):
            v = d[chiave]
            col = esiti[chiave]
            ax.scatter(v, [yy] * len(v), s=12, color=col, alpha=0.75, zorder=3, lw=0)
            m = sum(v) / len(v)
            ax.plot([m, m], [yy - 0.32, yy + 0.32], color=col, lw=1.8, zorder=4)
        ax.set_title(f"{nome}   ·   rumore ±{rumore:.1f} celle".replace(".", ","))
        ax.set_xlabel("differenza di celle occupate, per seme")
        ax.grid(axis="x", color="#eeeeee", lw=0.5, zorder=0)
        for lato in ("top", "right"):
            ax.spines[lato].set_visible(False)

    assi[0].set_yticks(y)
    assi[0].set_yticklabels([et for _, et in ORDINE])
    assi[0].tick_params(axis="y", length=0)

    legenda = [
        plt.Line2D([], [], color=COLORE_RIPRODOTTO, lw=2, label="riprodotto"),
        plt.Line2D([], [], color=COLORE_CONTRADDETTO, lw=2, label="contraddetto"),
        plt.Line2D([], [], color=COLORE_UNA_SOLA, lw=2, label="forte in una campagna sola"),
        plt.Line2D([], [], color=COLORE_NESSUNA, lw=2, label="in nessuna"),
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()

    out = Path("Tesi_LaTex/figures/replica_due_modelli.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"scritta {out} (+ .png)")
    for chiave, et in ORDINE:
        print(f"  {et:32s} {esiti[chiave]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
