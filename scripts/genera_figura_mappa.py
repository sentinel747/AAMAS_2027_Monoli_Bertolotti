# -*- coding: utf-8 -*-
"""Rigenera la figura della mappa minerale della tesi.

La versione precedente di `Tesi_LaTex/figura_mappa.tex` era stata prodotta da
uno script che non e' mai entrato nel repository, e disegnava il pianeta
**prima** della correzione del regolito del 2026-08-31: una ventina di
addensamenti isolati su una mappa vuota, con una didascalia che dichiarava
«i giacimenti coprono il 6,4 per cento delle celle». Su quel pianeta i coloni
non avevano nulla da estrarre.

Oggi il minerale ha un fondo diffuso piu' le vene, e la stessa misura vale
85,5 per cento. Un campo continuo non si disegna con un cerchio per cella
--- sarebbero cinquantacinquemila cerchi in un file TikZ --- quindi la mappa
diventa un'immagine e il resto della figura resta LaTeX.

    python scripts/genera_figura_mappa.py

Non tocca nulla fuori da `Tesi_LaTex/figures/` e `Tesi_LaTex/figura_mappa.tex`,
e non consuma alcuna API esterna: genera solo mondi deterministici in locale.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.world.colony_site import FASCIA_EQUATORIALE_RIGHE, find_safe_start
from src.world.world_generator import WorldGenerator

LARGHEZZA, ALTEZZA = 360, 180
SEMI = (9, 21, 33)
SEME_DISEGNATO = 9

USCITA_PNG = REPO / "Tesi_LaTex" / "figures" / "mappa_risorse_seme9.png"
USCITA_TEX = REPO / "Tesi_LaTex" / "figura_mappa.tex"

# Dal regolito chiaro alla vena scura: la luminosita' scende in modo monotono,
# cosi' una stampa in bianco e nero legge ancora la stessa scala.
REGOLITE = LinearSegmentedColormap.from_list(
    "regolite",
    ["#F2E7D6", "#E0BE8C", "#C98A4B", "#9E5A2A", "#6B3117"],
)
# Dal secco al ghiaccio: stessa monotonia di luminosita' della scala del
# regolito, cosi' le due mappe si leggono con lo stesso criterio anche
# stampate in bianco e nero.
GHIACCIO = LinearSegmentedColormap.from_list(
    "ghiaccio",
    ["#F7F4EC", "#CFE3EC", "#93C4DB", "#4E8FBF", "#1F4E79"],
)


def campo_minerale(seme: int):
    mondo = WorldGenerator(seed=seme).generate(width=LARGHEZZA, height=ALTEZZA)
    campo = np.array(
        [[mondo.cells[y][x].resources.minerals for x in range(LARGHEZZA)] for y in range(ALTEZZA)],
        dtype=float,
    )
    return mondo, campo


def campo_ghiaccio(mondo) -> "np.ndarray":
    """Il ghiaccio d'acqua per cella, dalla stessa mappa gia' generata."""
    return np.array(
        [[mondo.cells[y][x].water_ice for x in range(LARGHEZZA)] for y in range(ALTEZZA)],
        dtype=float,
    )


def disegna(campo: np.ndarray, ghiaccio: np.ndarray, sito: tuple[int, int]) -> None:
    """Due pannelli sovrapposti: il minerale, che dice dove si costruisce, e il
    ghiaccio, che dice dove si beve. Sono due geografie diverse, ed e' questo
    che la figura deve far vedere: il minerale e' un campo continuo con vene,
    il ghiaccio si concentra alle calotte, cioe' FUORI dalla fascia entro cui
    il sito puo' essere scelto."""
    USCITA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, assi = plt.subplots(2, 1, figsize=(7.2, 7.4), dpi=200,
                             gridspec_kw={"hspace": 0.28})
    for ax, dati, scala, etichetta in (
        (assi[0], campo, REGOLITE, "minerali per cella"),
        (assi[1], ghiaccio, GHIACCIO, "ghiaccio d'acqua per cella"),
    ):
        _pannello(fig, ax, dati, scala, etichetta, sito)
    fig.savefig(USCITA_PNG, bbox_inches="tight")
    plt.close(fig)


def _pannello(fig, ax, campo, scala, etichetta_barra, sito) -> None:
    imm = ax.imshow(campo, cmap=scala, origin="upper", interpolation="nearest",
                    extent=(0, LARGHEZZA, ALTEZZA, 0))

    centro = ALTEZZA // 2
    alto = centro - FASCIA_EQUATORIALE_RIGHE
    basso = centro + FASCIA_EQUATORIALE_RIGHE + 1

    # La fascia entro cui il sito viene scelto: bordi netti, nessun velo che
    # alteri i colori del campo sottostante.
    for y in (alto, basso):
        ax.axhline(y, color="#1F3B57", lw=1.0, ls="-")
    ax.axhline(centro, color="#1F3B57", lw=0.8, ls=(0, (4, 3)), alpha=0.8)
    # Le annotazioni cadono sopra il campo, che e' fitto: senza il riquadro
    # bianco le lettere si confondono con la trama delle celle.
    riquadro = dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.86,
                    edgecolor="none")
    ax.annotate(
        "fascia equatoriale: il sito si sceglie qui dentro",
        xy=(6, alto - 4), color="#1F3B57", fontsize=8, va="bottom", ha="left",
        bbox=riquadro,
    )
    ax.annotate("equatore", xy=(LARGHEZZA - 6, centro - 3), color="#1F3B57",
                fontsize=8, va="bottom", ha="right", bbox=riquadro)

    x, y = sito
    ax.plot([x], [y + 0.5], marker="P", markersize=9, markerfacecolor="#F5F5F5",
            markeredgecolor="#122B44", markeredgewidth=1.2, linestyle="none")
    ax.annotate(f"sito scelto ({x},{y})", xy=(x + 7, y + 14), color="#122B44",
                fontsize=8, va="center", ha="left", bbox=riquadro,
                arrowprops=dict(arrowstyle="-", color="#122B44", lw=0.8))

    ax.set_xlim(0, LARGHEZZA)
    ax.set_ylim(ALTEZZA, 0)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("colonna della griglia (360 macro-celle)", fontsize=8)
    ax.set_ylabel("riga (180)", fontsize=8)
    ax.tick_params(labelsize=7)
    for lato in ax.spines.values():
        lato.set_color("#555555")
        lato.set_linewidth(0.6)

    barra = fig.colorbar(imm, ax=ax, fraction=0.030, pad=0.02)
    barra.set_label(etichetta_barra, fontsize=8)
    barra.ax.tick_params(labelsize=7)


def main() -> None:
    mondi = {}
    for seme in SEMI:
        mondo, campo = campo_minerale(seme)
        mondi[seme] = (mondo, campo, find_safe_start(mondo))

    mondo9, campo9, sito9 = mondi[SEME_DISEGNATO]
    ghiaccio9 = campo_ghiaccio(mondo9)
    disegna(campo9, ghiaccio9, sito9)

    quota_con_minerali = 100.0 * float((campo9 > 0).mean())
    mediana = float(np.median(campo9))
    massimo = float(campo9.max())
    # «Sopra la mediana» sarebbe il cinquanta per cento per definizione e non
    # direbbe nulla: la soglia delle vene e' una volta e mezza la mediana.
    quota_vene = 100.0 * float((campo9 > 1.5 * mediana).mean())

    # Il ghiaccio: quanto ce n'e', e soprattutto DOVE. La quota polare si
    # misura fuori dalla fascia equatoriale, che e' l'unica in cui il sito
    # puo' essere scelto: se il ghiaccio sta quasi tutto li', la colonia nasce
    # per forza lontana dall'acqua, ed e' un vincolo del pianeta e non una
    # scelta del protocollo.
    centro = ALTEZZA // 2
    dentro = slice(centro - FASCIA_EQUATORIALE_RIGHE, centro + FASCIA_EQUATORIALE_RIGHE + 1)
    ghiaccio_tot = float(ghiaccio9.sum())
    quota_ghiaccio_fascia = 100.0 * float(ghiaccio9[dentro, :].sum()) / max(ghiaccio_tot, 1e-9)
    mediana_ghiaccio_fascia = float(np.median(ghiaccio9[dentro, :]))
    massimo_ghiaccio = float(ghiaccio9.max())

    def virgola(x: float, cifre: int = 2) -> str:
        return f"{x:.{cifre}f}".replace(".", "{,}")

    righe_siti = []
    for seme in SEMI:
        mondo, campo, (x, y) = mondi[seme]
        righe_siti.append(f"il seme {seme} sceglie $({x},{y})$")
    elenco_siti = ", ".join(righe_siti)

    # Stringa **grezza**: senza `r` la sequenza `\begin` diventerebbe un
    # ritorno indietro seguito da «egin», e il file uscirebbe rotto.
    tex = rf"""% Generato da scripts/genera_figura_mappa.py: non modificare a mano.
% Mappa del seme {SEME_DISEGNATO}, griglia {LARGHEZZA}x{ALTEZZA}, misure ricalcolate
% sul mondo corrente. La versione precedente di questo file disegnava il
% pianeta prima della correzione del regolito del 2026-08-31 e dichiarava una
% copertura del 6,4 per cento: oggi vale {quota_con_minerali:.1f}.
\begin{{figure}}[htbp]
 \centering
 \includegraphics[width=\linewidth]{{mappa_risorse_seme9.png}}
 \caption{{La geografia che decide l'esito, disegnata dalla mappa reale del
 seme {SEME_DISEGNATO}. Il minerale non è un giacimento isolato ma un campo: lo possiede
 il ${virgola(quota_con_minerali, 1)}$~per cento delle celle, con mediana ${virgola(mediana)}$ e
 massimo ${virgola(massimo)}$; le vene sono il ${virgola(quota_vene, 1)}$~per cento che
 supera una volta e mezza la mediana. Le due bande chiare in alto e in basso
 sono le calotte polari, dove il minerale lascia il posto al ghiaccio. Fra le
 due righe continue sta la fascia equatoriale entro cui il sito viene scelto,
 perché fuori di essa la severità polare cambierebbe la condizione sperimentale
 invece di offrire un'alternativa, e la croce segna il sito che questa mappa
 produce. Semi diversi producono siti diversi: {elenco_siti}.
 Il pannello inferiore disegna il ghiaccio d'acqua sulla stessa mappa, e mostra
 la geografia opposta: il minerale e' un campo continuo che copre l'equatore,
 il ghiaccio si concentra alle calotte, e nella fascia in cui il sito puo'
 essere scelto ne resta soltanto il ${virgola(quota_ghiaccio_fascia, 1)}$~per cento del totale,
 con mediana ${virgola(mediana_ghiaccio_fascia)}$ contro un massimo planetario di ${virgola(massimo_ghiaccio)}$.
 La colonia nasce quindi dove c'e' da costruire e lontano da dove c'e' da bere,
 e l'acqua resta un problema di produzione e non di raccolta: e' un vincolo del
 pianeta, non una scelta del protocollo.}}
 \label{{fig:mappa-giacimenti}}
\end{{figure}}
"""
    USCITA_TEX.write_text(tex, encoding="utf-8")

    print(f"  ok   {USCITA_PNG.relative_to(REPO)}")
    print(f"  ok   {USCITA_TEX.relative_to(REPO)}")
    print(f"       celle con minerali {quota_con_minerali:.2f}%, mediana {mediana:.2f}, massimo {massimo:.2f}")
    for seme in SEMI:
        _, _, (x, y) = mondi[seme]
        print(f"       seme {seme}: sito ({x},{y})")


if __name__ == "__main__":
    main()
