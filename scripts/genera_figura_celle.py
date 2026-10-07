# -*- coding: utf-8 -*-
"""Due macro-celle affiancate, disegnate dalle coordinate reali di una run.

La figura che stava in tesi veniva da una run al passo 3000 il cui script non
era mai entrato nel repository, e non era piu' rigenerabile. Questa lo e': legge
uno snapshot del mondo, sceglie la cella madre e una cella di distretto, e
scrive il sorgente TikZ.

**Perche' due celle e non una.** Una cella sola mostra dove finiscono le
strutture, ma non dice nulla sul fatto che la colonia sia concentrata: e'
esattamente la grandezza su cui LLM (Gov+Amm) agisce
(le celle occupate scendono su cinque mondi su cinque). Affiancare la cella
madre a una cella di distretto, nella stessa run e alla stessa scala, rende
visibile il rapporto: sono due ordini di grandezza.

Uso:
    python scripts/genera_figura_celle.py \
        --snapshot runs/campagna_scarsa_v3/llm_completo_amm/llm_completo+amm_seed3/world_snapshots/step_001000_day_007000.json \
        --out Tesi_LaTex/figura_celle.tex
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Proiezione assonometrica: il quadrato della cella diventa un rombo largo
# 2*MEZZA_BASE e alto 2*MEZZA_ALTEZZA. Il rapporto fra i due e' la pendenza
# apparente, ed e' lo stesso della figura che questa sostituisce (~3,15).
MEZZA_BASE = 3.55
MEZZA_ALTEZZA = 1.127
# Impronta di una struttura, in frazione di mezza base: piccola abbastanza da
# non nascondere le vicine nella cella madre, che ne ha quasi duemila.
IMPRONTA = 0.0118
SOLCHI = 10  # righe del reticolo del terreno

# Colore e altezza per tipo. L'altezza codifica il tipo quanto il colore: a
# stampa in bianco e nero la figura resta leggibile.
TIPI = {
    "habitat":         ("tipoHabitat",      "B03A2E", 0.46, "habitat"),
    "greenhouse":      ("tipoSerra",        "2E8B57", 0.30, "serra"),
    "oxygen_plant":    ("tipoOssigeno",     "2E86C1", 0.38, "impianto O$_2$"),
    "shelter":         ("tipoRicovero",     "E08A3C", 0.20, "ricovero"),
    "solar_array":     ("tipoPannello",     "E8BE2A", 0.09, "pannello"),
    "storage_depot":   ("tipoDeposito",     "7F8C8D", 0.26, "deposito"),
    "infirmary":       ("tipoInfermeria",   "C64B8C", 0.34, "infermeria"),
    "research_lab":    ("tipoLaboratorio",  "12958F", 0.32, "laboratorio"),
    "heater":          ("tipoRiscaldatore", "7D4CA8", 0.24, "riscaldatore"),
    "weather_station": ("tipoMeteo",        "34495E", 0.28, "stazione meteo"),
    "water_extractor": ("tipoEstrattore",   "1F6FB2", 0.22, "estrattore d'acqua"),
}
IGNOTO = ("tipoAltro", "95A5A6", 0.18, "altro")
#: Quanto sale il solido piu' alto sopra il punto in cui poggia. Le
#: intestazioni vanno messe sopra QUESTO e non sopra il piano, altrimenti le
#: strutture del fondo della cella madre ci finiscono dentro.
ALTEZZA_MASSIMA = max(v[2] for v in TIPI.values())


def proietta(fx: float, fy: float) -> tuple[float, float]:
    """Frazioni di cella in [0,1]^2 -> coordinate del disegno."""
    return MEZZA_BASE * (fx - fy), MEZZA_ALTEZZA * (fx + fy)


def piano(dx: float, dy: float) -> list[str]:
    """Il rombo del terreno e il suo reticolo, traslati di (dx, dy)."""
    ang = [proietta(0, 0), proietta(1, 0), proietta(1, 1), proietta(0, 1)]
    p = " -- ".join(f"({x + dx:.3f},{y + dy:.3f})" for x, y in ang)
    righe = [f"  \\fill[black!7] {p} -- cycle;",
             f"  \\draw[black!55, line width=0.4pt] {p} -- cycle;"]
    for i in range(1, SOLCHI):
        t = i / SOLCHI
        for a, b in (((t, 0.0), (t, 1.0)), ((0.0, t), (1.0, t))):
            x1, y1 = proietta(*a)
            x2, y2 = proietta(*b)
            righe.append(
                f"  \\draw[black!16, line width=0.15pt] "
                f"({x1 + dx:.3f},{y1 + dy:.3f}) -- ({x2 + dx:.3f},{y2 + dy:.3f});")
    return righe


def volume(cx: float, cy: float, colore: str, h: float) -> list[str]:
    """Un prisma che POGGIA su (cx, cy) e si alza, con le due facce visibili.

    **Le strutture si alzano, non sprofondano.** La prima versione metteva la
    faccia superiore sul punto di appoggio e il corpo sotto di esso: i solidi
    scendevano in $-y$ e la colonia sembrava scavata nel terreno invece che
    costruita sopra. Qui (cx, cy) e' il vertice basso dell'impronta a terra, il
    rombo superiore sta `h` piu' in alto, e le due facce laterali collegano i
    due rombi.

    Le facce disegnate sono quelle che partono dal vertice basso, verso destra
    e verso sinistra: sono le sole che il punto di vista mostra. Disegnare
    invece la faccia alta di destra, come faceva il file scritto a mano che
    questa figura sostituisce, produce un solido che non esiste.
    """
    fw = IMPRONTA * MEZZA_BASE
    fh = IMPRONTA * MEZZA_ALTEZZA
    stile = "draw=black!45, line width=0.12pt"
    sopra = (f"({cx:.3f},{cy + h:.3f}) -- ({cx + fw:.3f},{cy + h + fh:.3f}) -- "
             f"({cx:.3f},{cy + h + 2 * fh:.3f}) -- ({cx - fw:.3f},{cy + h + fh:.3f})")
    destra = (f"({cx:.3f},{cy:.3f}) -- ({cx + fw:.3f},{cy + fh:.3f}) -- "
              f"({cx + fw:.3f},{cy + fh + h:.3f}) -- ({cx:.3f},{cy + h:.3f})")
    sinistra = (f"({cx:.3f},{cy:.3f}) -- ({cx - fw:.3f},{cy + fh:.3f}) -- "
                f"({cx - fw:.3f},{cy + fh + h:.3f}) -- ({cx:.3f},{cy + h:.3f})")
    return [f"  \\fill[{colore}!74!black, {stile}] {sinistra} -- cycle;",
            f"  \\fill[{colore}!58!white, {stile}] {destra} -- cycle;",
            f"  \\fill[{colore}, {stile}] {sopra} -- cycle;"]


def pannello(cella: dict, dx: float, dy: float) -> list[str]:
    """Il terreno piu' tutte le strutture, dal fondo verso l'osservatore."""
    righe = piano(dx, dy)
    larg = cella["geometry"]["width_m"]
    alt = cella["geometry"]["height_m"]
    solidi = []
    for s in cella["structures"]:
        fx = min(max(s.get("local_x_m", 0.0) / larg, 0.0), 1.0)
        fy = min(max(s.get("local_y_m", 0.0) / alt, 0.0), 1.0)
        colore, _, h, _ = TIPI.get(s.get("type"), IGNOTO)
        x, y = proietta(fx, fy)
        solidi.append((fx + fy, x + dx, y + dy, colore, h))
    # Dal fondo (fx+fy grande) verso il davanti: i solidi vicini coprono i lontani.
    solidi.sort(key=lambda t: -t[0])
    for _, x, y, colore, h in solidi:
        righe.extend(volume(x, y, colore, h))
    return righe


def etichetta(testo: str, dx: float, dy: float) -> str:
    return (f"  \\node[font=\\small\\bfseries, align=center] at "
            f"({dx:.3f},{dy:.3f}) {{{testo}}};")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--distretto", type=int, default=None,
                    help="cella di distretto: indice nella lista ordinata per "
                         "numero di strutture (default: la piu' costruita)")
    args = ap.parse_args()

    dati = json.loads(args.snapshot.read_text(encoding="utf-8"))
    celle = {(c["x"], c["y"]): c for c in dati["cells"]}
    amm = dati["administrators"]
    my, mx = amm["mother_cell"]          # lo snapshot la scrive come [riga, colonna]
    madre = celle[(mx, my)]

    # Le celle di distretto sono quelle che un amministratore ha in carico.
    di_distretto = {}
    for d in amm["last_round"]:
        for y, x in d["cells"]:
            if (x, y) != (mx, my) and (x, y) in celle:
                di_distretto[(x, y)] = d
    candidate = sorted(di_distretto, key=lambda k: -len(celle[k]["structures"]))
    scelta = candidate[args.distretto or 0]
    vicina = celle[scelta]
    quale = di_distretto[scelta]

    passo = 2 * MEZZA_BASE + 0.9
    righe = ["% Generato da scripts/genera_figura_celle.py: non modificare a mano."]
    for chiave, (nome, hexc, _, _) in TIPI.items():
        righe.append(f"\\definecolor{{{nome}}}{{HTML}}{{{hexc}}}")
    righe.append(f"\\definecolor{{{IGNOTO[0]}}}{{HTML}}{{{IGNOTO[1]}}}")
    righe += [
        f"% cella madre ({mx},{my}) {len(madre['structures'])} strutture; "
        f"cella di distretto ({scelta[0]},{scelta[1]}) {len(vicina['structures'])}",
        "\\begin{figure}[htbp]",
        " \\centering",
        " \\resizebox{\\textwidth}{!}{%",
        " \\begin{tikzpicture}[x=1cm, y=1cm]",
    ]
    righe += pannello(madre, 0.0, 0.0)
    righe += pannello(vicina, passo, 0.0)
    righe.append(etichetta(
        f"cella madre\\\\{len(madre['structures'])} strutture, "
        f"{len(madre['agents_present'])} coloni", 0.0, 2 * MEZZA_ALTEZZA + ALTEZZA_MASSIMA + 0.45))
    righe.append(etichetta(
        f"cella di distretto\\\\{len(vicina['structures'])} strutture, "
        f"{len(vicina['agents_present'])} coloni", passo, 2 * MEZZA_ALTEZZA + ALTEZZA_MASSIMA + 0.45))
    # La quota lungo il bordo inferiore sinistro della cella madre, fuori dal piano.
    x1, y1 = proietta(0.0, 1.0)
    x2, y2 = proietta(0.0, 0.0)
    km = round(madre["geometry"]["width_m"] / 1000)
    righe.append(
        f"  \\draw[|-|, black!70, line width=0.4pt] ({x1 - 0.62:.3f},{y1 - 0.20:.3f}) -- "
        f"({x2 - 0.62:.3f},{y2 - 0.20:.3f}) node[midway, below, sloped, font=\\scriptsize] "
        f"{{{km} km}};")

    # Legenda: conteggi della sola cella madre, in ordine di frequenza.
    conta = Counter(s.get("type") for s in madre["structures"])
    voci = [t for t, _ in conta.most_common()]
    x0, y0 = -MEZZA_BASE - 0.3, -0.95
    for i, tipo in enumerate(voci):
        nome, _, _, leggibile = TIPI.get(tipo, IGNOTO)
        cx = x0 + (i % 3) * (passo * 2 / 3 + 0.2)
        cy = y0 - (i // 3) * 0.34
        righe.append(f"  \\fill[{nome}, draw=black!45, line width=0.12pt] "
                     f"({cx:.2f},{cy:.2f}) rectangle ({cx + 0.24:.2f},{cy + 0.19:.2f});")
        righe.append(f"  \\node[anchor=west, font=\\scriptsize] at ({cx + 0.32:.2f},"
                     f"{cy + 0.09:.2f}) {{{leggibile} ({conta[tipo]})}};")

    riscritta = "ha riscritto" if not quale["accepted_government_policy"] else "ha accettato"
    righe += [
        " \\end{tikzpicture}",
        " }",
        f" \\caption{{Due macro-celle della stessa colonia al passo {dati['step']}, "
        f"disegnate dalle coordinate reali della run (mondo di riferimento, seme 3, "
        f"LLM (Gov+Amm), cioè governatore più amministratori). A sinistra la cella madre, dove la "
        f"colonia si insedia; a destra una cella di distretto, il cui amministratore in "
        f"quella tornata {riscritta} la legge del governatore. Le due celle sono alla "
        f"stessa scala e misurano {km} chilometri di lato: quello che cambia è la "
        f"densità. Altezza e colore codificano entrambi il tipo, secondo la legenda, e "
        f"nient'altro; la legenda conta le strutture della cella madre. Il rapporto fra "
        f"le due, {len(madre['structures'])} strutture contro "
        f"{len(vicina['structures'])}, è la forma concreta di ciò che le figure del "
        f"capitolo cinque misurano come celle occupate: la colonia non si distribuisce, "
        f"si addensa attorno al sito di sbarco e manda avanguardie.}}",
        " \\label{fig:celle-assonometriche}",
        "\\end{figure}",
    ]
    args.out.write_text("\n".join(righe) + "\n", encoding="utf-8")
    print(f"scritto {args.out}: madre ({mx},{my}) {len(madre['structures'])} strutture, "
          f"distretto ({scelta[0]},{scelta[1]}) {len(vicina['structures'])}")


if __name__ == "__main__":
    main()
