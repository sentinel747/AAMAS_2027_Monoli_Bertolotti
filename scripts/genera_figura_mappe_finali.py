# -*- coding: utf-8 -*-
r"""Lo stato finale della colonia, baseline contro LLM (Gov+Amm), sullo stesso seme.

Il risultato piu' solido della campagna dei mondi e' che le celle occupate
scendono su cinque mondi su cinque. E' un numero, e un numero di celle non dice
che \emph{forma} abbia la colonia. Questa figura mostra le due mappe a fine run,
per ogni mondo, dallo stesso seme e alla stessa scala: si vede che la differenza
non e' una colonia piu' piccola ma una colonia meno sparpagliata, con gli stessi
coloni e meno avamposti.

**Le due mappe di una riga condividono il riquadro e la scala di colore.** Un
riquadro adattato a ciascun pannello farebbe sembrare uguali due estensioni
diverse, che e' esattamente cio' che la figura deve far vedere.

Uso:
    python scripts/genera_figura_mappe_finali.py --seme 3
    python scripts/genera_figura_mappe_finali.py --seme 3 --presentazione   # slide di laurea

**La variante per le slide mostra un mondo solo, il frammentato.** Otto mappe
in una colonna stretta, proiettate, diventano francobolli illeggibili; la slide
cita il raggio del mondo frammentato, e due mappe grandi di quel mondo dicono
la stessa cosa a chi le guarda da tre metri. Le etichette sono quelle del resto
della presentazione («senza governo», «con governo LLM»), non quelle tecniche.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# mondo -> (etichetta, cartella baseline, cartella coppia)
MONDI = [
    ("ricco di ghiaccio", "runs/mondi/ice_rich/ctrl_none",
     "runs/mondi/ice_rich/llm_completo_amm"),
    ("riferimento", "runs/mondi/ice_rich/ctrl_none", None),  # sostituito sotto
    ("frammentato", "runs/mondi/fragmented/ctrl_none",
     "runs/mondi/fragmented/llm_completo_amm"),
    ("rischio alto", "runs/mondi/high_hazard/ctrl_none",
     "runs/mondi/high_hazard/llm_completo_amm"),
    ("risorse scarse", "runs/mondi/scarce_resources/ctrl_none",
     "runs/mondi/scarce_resources/llm_completo_amm_v2"),
]
# Il mondo di riferimento non ha snapshot nella baseline (la campagna e'
# anteriore agli snapshot a ogni tornata): resta fuori, e la didascalia lo dice.
MONDI = [m for m in MONDI if m[2] is not None]


def ultimo_snapshot(braccio: Path, seme: int) -> Path | None:
    for cartella in sorted(braccio.glob(f"*seed{seme}")):
        istantanee = sorted((cartella / "world_snapshots").glob("step_*.json"))
        if istantanee:
            return istantanee[-1]
    return None


def popolazione_per_cella(percorso: Path) -> tuple[dict[tuple[int, int], int], int, int]:
    dati = json.loads(percorso.read_text(encoding="utf-8"))
    mappa: dict[tuple[int, int], int] = {}
    for c in dati["cells"]:
        n = len(c.get("agents_present") or [])
        if n or c.get("structures"):
            mappa[(c["x"], c["y"])] = n
    return mappa, dati["metrics"].get("population", 0), len(mappa)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seme", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--presentazione", action="store_true",
                    help="solo il mondo frammentato, mappe grandi, etichette delle slide")
    args = ap.parse_args()
    pres = args.presentazione
    if args.out is None:
        args.out = ROOT / ("Tesi_LaTex/PPT_Tesi_LaTex/media/mappe_frammentato" if pres
                           else "Tesi_LaTex/figures/mappe_finali")
    mondi = [m for m in MONDI if m[0] == "frammentato"] if pres else MONDI
    nomi = ("senza governo", "con governo LLM") if pres else ("baseline", "LLM (Gov+Amm)")
    f_tit, f_lab = (13, 11) if pres else (8, 6.5)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "figure.dpi": 150,
    })

    righe = []
    for etichetta, base_rel, coppia_rel in mondi:
        b = ultimo_snapshot(ROOT / base_rel, args.seme)
        c = ultimo_snapshot(ROOT / coppia_rel, args.seme)
        if b is None or c is None:
            print(f"salto {etichetta}: manca uno snapshot")
            continue
        righe.append((etichetta, popolazione_per_cella(b), popolazione_per_cella(c)))
    if not righe:
        raise SystemExit("nessun mondo con entrambi gli snapshot")

    n = len(righe)
    # **Titoli corti e uguali, e layout vincolato.** Con `tight_layout` un titolo
    # piu' lungo restringe il proprio riquadro, e due mappe della stessa riga
    # finiscono disegnate a scale diverse: la figura direbbe il contrario di
    # quello che deve dire. Le intestazioni stanno solo sulla prima riga, i
    # conteggi sotto ogni pannello.
    fig, assi = plt.subplots(n, 2, figsize=(6.4, 3.5) if pres else (5.2, 1.75 * n), squeeze=False,
                             constrained_layout=True)
    massimo = max(max(m.values() or [1]) for _, (m, _, _), _ in righe)
    massimo = max(massimo, max(max(m.values() or [1]) for _, _, (m, _, _) in righe))

    # **Un riquadro solo, uguale in ogni pannello della figura.** Prima il
    # riquadro era l'unione delle due mappe di ciascuna riga, quindi mondi
    # diversi comparivano su griglie di dimensione diversa e l'estensione di
    # una colonia non si poteva confrontare a occhio con quella di un'altra.
    # Ora si prende il lato massimo su tutte le righe e lo si usa ovunque,
    # centrato sul baricentro del riquadro di ciascuna riga.
    finestre = []
    for _, (mb, _, _), (mc, _, _) in righe:
        xs = [x for x, _ in list(mb) + list(mc)]
        ys = [y for _, y in list(mb) + list(mc)]
        finestre.append((min(xs), max(xs), min(ys), max(ys)))
    lato = max(max(x1 - x0, y1 - y0) for x0, x1, y0, y1 in finestre) + 3

    for i, (etichetta, (mb, pb, cb), (mc, pc, cc)) in enumerate(righe):
        fx0, fx1, fy0, fy1 = finestre[i]
        x0 = (fx0 + fx1) // 2 - lato // 2
        y0 = (fy0 + fy1) // 2 - lato // 2
        x1, y1 = x0 + lato - 1, y0 + lato - 1
        for j, (mappa, pop, celle, nome) in enumerate(
                ((mb, pb, cb, nomi[0]), (mc, pc, cc, nomi[1]))):
            griglia = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan)
            for (x, y), v in mappa.items():
                griglia[y - y0, x - x0] = max(v, 0.5)   # 0,5 = cella con sole strutture
            ax = assi[i][j]
            ax.imshow(griglia, cmap="YlOrRd", norm=LogNorm(vmin=0.5, vmax=massimo),
                      interpolation="nearest", origin="upper")
            # Il bordo di ogni cella deve restare leggibile anche nelle zone
            # dense: la mappa comunica così occupazione discreta, non un campo
            # continuo sfumato.
            ax.set_xticks(np.arange(-0.5, griglia.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, griglia.shape[0], 1), minor=True)
            # Matplotlib scarta le tacche minori che cadono su una maggiore:
            # senza questo alcune linee della griglia mancano e le celle
            # sembrano larghe il doppio.
            ax.xaxis.remove_overlapping_locs = False
            ax.yaxis.remove_overlapping_locs = False
            ax.grid(which="minor", color="#6b6b6b", linewidth=0.3 if pres else 0.22, alpha=0.75)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#999999")
            if i == 0:
                ax.set_title(nome, fontsize=f_tit, pad=6 if pres else 4,
                             color=("#c2410c" if pres and j == 1 else "#222222"))
            ax.set_xlabel(f"{celle} celle, {pop} coloni", fontsize=f_lab, labelpad=4 if pres else 2)
            if j == 0 and not pres:
                ax.set_ylabel(etichetta, fontsize=8)
        print(f"{etichetta}: baseline {cb} celle / {pb} coloni; "
              f"LLM (Gov+Amm) {cc} celle / {pc} coloni")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not pres:
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=220 if pres else 150)
    plt.close(fig)
    print(f"scritto {args.out}.png" + ("" if pres else " / .pdf"))


if __name__ == "__main__":
    main()
