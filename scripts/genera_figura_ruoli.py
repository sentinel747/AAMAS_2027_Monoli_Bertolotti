# -*- coding: utf-8 -*-
"""Genera la figura dei gruppi professionali (commento 63 sul PDF annotato).

**Che cosa deve far vedere.** I sei gruppi non hanno azioni esclusive ne' una
visione privilegiata: la sola differenza sta in due profili numerici, la
preferenza fra i sei pilastri e la capacita' su ciascuno di essi. La figura
mette i due profili uno accanto all'altro, con il colono generico come riga di
riferimento, cosi' si vede subito che nessun gruppo ha uno zero (nessuno e'
escluso da nulla) e che le differenze sono di grado e non di natura.

I numeri non sono ricopiati: si leggono da `src/agents/role_profiles.py`, cosi'
la figura non puo' divergere dal simulatore.

    python scripts/genera_figura_ruoli.py

Non consuma alcuna API esterna: legge costanti locali e disegna.
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

from src.agents.role_profiles import DEFAULT_ROLE_DISTRIBUTION, ROLE_ORDER, ROLE_PROFILES

USCITA_PNG = REPO / "Tesi_LaTex" / "figures" / "profili_ruoli.png"
USCITA_TEX = REPO / "Tesi_LaTex" / "figura_ruoli.tex"

PILASTRI = ("sostentamento", "risorse", "costruzione", "vita", "sociale", "esplorazione")
NOMI = {
    "biologist": "biologo",
    "technician": "tecnico",
    "engineer": "ingegnere",
    "medic": "medico",
    "coordinator": "coordinatore",
    "explorer": "esploratore",
}
# Una tinta per pilastro, la stessa nei due pannelli.
TINTE = ("#3A6EA5", "#B5651D", "#2F6B45", "#8A2F2F", "#6A5ACD", "#C99B0A")


def disegna() -> dict:
    ruoli = list(ROLE_ORDER)
    pref = np.array([ROLE_PROFILES[r].preferences for r in ruoli], dtype=float)
    cap = np.array([ROLE_PROFILES[r].skills for r in ruoli], dtype=float)
    generico = np.array(ROLE_PROFILES["colonist"].preferences, dtype=float)

    USCITA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, (sx, dx) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200,
                                 gridspec_kw={"wspace": 0.26})

    etichette = [NOMI[r] for r in ruoli]
    y = np.arange(len(ruoli))

    # Sinistra: la preferenza, barre impilate che sommano a uno.
    sinistra = np.zeros(len(ruoli))
    for j, (nome, tinta) in enumerate(zip(PILASTRI, TINTE)):
        sx.barh(y, pref[:, j], left=sinistra, color=tinta, edgecolor="white",
                linewidth=0.6, label=nome, height=0.68)
        sinistra = sinistra + pref[:, j]
    for j, quota in enumerate(generico):
        # Il colono generico: sei quote uguali, tratteggiate come riferimento.
        sx.axvline(float(generico[:j + 1].sum()), color="black", lw=0.5,
                   ls=(0, (2, 2)), alpha=0.35)
    sx.set_yticks(y, etichette, fontsize=9)
    sx.invert_yaxis()
    sx.set_xlim(0, 1)
    sx.set_xlabel("quota di preferenza (somma 1)", fontsize=9)
    sx.set_title("Fra che cosa sceglie", fontsize=10)
    sx.tick_params(labelsize=8)

    # Destra: la capacita', un moltiplicatore attorno a uno.
    larghezza = 0.13
    for j, (nome, tinta) in enumerate(zip(PILASTRI, TINTE)):
        dx.barh(y + (j - 2.5) * larghezza, cap[:, j] - 1.0, left=1.0,
                color=tinta, edgecolor="white", linewidth=0.4, height=larghezza)
    dx.axvline(1.0, color="black", lw=0.8)
    dx.set_yticks(y, etichette, fontsize=9)
    dx.invert_yaxis()
    dx.set_xlim(0.7, 1.55)
    dx.set_xlabel("capacità (moltiplicatore, 1 = colono generico)", fontsize=9)
    dx.set_title("Quanto rende quando lo fa", fontsize=10)
    dx.tick_params(labelsize=8)

    for ax in (sx, dx):
        for lato in ax.spines.values():
            lato.set_color("#555555")
            lato.set_linewidth(0.6)

    sx.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.16),
              frameon=False)

    fig.savefig(USCITA_PNG, bbox_inches="tight")
    plt.close(fig)

    minimo = float(pref.min())
    massimo = float(pref.max())
    return {"minimo": minimo, "massimo": massimo,
            "cap_min": float(cap.min()), "cap_max": float(cap.max())}


def main() -> None:
    m = disegna()

    def virgola(x: float, cifre: int = 2) -> str:
        return f"{x:.{cifre}f}".replace(".", "{,}")

    quote = ", ".join(
        f"{NOMI[r]} {DEFAULT_ROLE_DISTRIBUTION[r]:.0f}\\,\\%" for r in ROLE_ORDER
    )

    tex = rf"""% Generato da scripts/genera_figura_ruoli.py: non modificare a mano.
% I numeri vengono da src/agents/role_profiles.py.
\begin{{figure}}[htbp]
 \centering
 \includegraphics[width=\linewidth]{{profili_ruoli.png}}
 \caption{{I sei gruppi professionali, letti dai profili del simulatore. A
 sinistra la preferenza: la quota con cui ciascun gruppo sceglie fra i sei
 pilastri, che somma sempre a uno, con le righe tratteggiate a segnare il
 colono generico, per cui le sei quote sono uguali. A destra la capacità: di
 quanto rende più o meno del generico quando quell'attività la svolge davvero.
 Nessuna quota è mai nulla, il minimo vale ${virgola(m['minimo'])}$ e il massimo
 ${virgola(m['massimo'])}$: nessun gruppo è escluso da alcuna attività e nessuno
 ne ha l'esclusiva, e le capacità restano fra ${virgola(m['cap_min'])}$ e
 ${virgola(m['cap_max'])}$. Il ruolo inclina la scelta, non la sostituisce, ed è
 la condizione che tiene i coloni confrontabili fra bracci. La distribuzione
 iniziale predefinita è {quote}.}}
 \label{{fig:profili-ruoli}}
\end{{figure}}
"""
    USCITA_TEX.write_text(tex, encoding="utf-8")
    print(f"  ok   {USCITA_PNG.relative_to(REPO)}")
    print(f"  ok   {USCITA_TEX.relative_to(REPO)}")
    print(f"       preferenze fra {m['minimo']:.2f} e {m['massimo']:.2f}, "
          f"capacità fra {m['cap_min']:.2f} e {m['cap_max']:.2f}")


if __name__ == "__main__":
    main()
