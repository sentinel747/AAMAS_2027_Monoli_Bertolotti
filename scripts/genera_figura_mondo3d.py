# -*- coding: utf-8 -*-
"""La colonia in tre dimensioni, a tre momenti della stessa run.

**A che cosa serve.** Le mappe piatte della sezione sui mondi dicono dove la
colonia sta; non dicono come cresce ne' come si divide. Questa figura prende le
istantanee che la simulazione salva a ogni tornata e ne disegna tre, a inizio,
meta' e fine run, come rilievo tridimensionale centrato sulla cella madre: il
suolo porta la quota reale della cella con le tinte del terreno usate
dall'interfaccia, le colonne portano gli abitanti e il colore porta il distretto
amministrativo. Si vede quindi in una figura sola la crescita dell'insediamento,
la comparsa dei distretti e il rapporto fra il nucleo e la corona.

**Che cosa e' vero e che cosa e' convenzione.** Quota, terreno, popolazione,
strutture e assegnazione ai distretti vengono dall'istantanea e non sono
ritoccati; le tinte del terreno sono le stesse dell'interfaccia
(`frontend/src/rendering/marsSurface.ts`). Sono convenzioni: l'esagerazione
verticale del rilievo; l'altezza delle colonne sul logaritmo della popolazione,
perche' la cella madre ne ospita milleduecento e la piu' grande delle altre
settantaquattro, e su scala lineare o di radice il nucleo schiaccerebbe tutto il
resto; i colori dei distretti, arbitrari e utili solo a distinguerli. Le tre
istantanee condividono il riquadro, calcolato sull'ultima, cosi' che la crescita
si legga per confronto.

Uso:
    python scripts/genera_figura_mondo3d.py
    python scripts/genera_figura_mondo3d.py --run runs/mondi/fragmented/llm_completo_amm --seme 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

#: Le tinte del terreno dell'interfaccia, da `TERRAIN_ALBEDO` in
#: `frontend/src/rendering/marsSurface.ts`: la figura della tesi e la vista che
#: si usa per guardare le run devono mostrare lo stesso pianeta.
TINTE_TERRENO = {
    "regolith_plain": (178, 99, 62),
    "crater": (110, 58, 36),
    "mountain": (156, 106, 72),
    "canyon": (124, 65, 40),
    "ice_deposit": (239, 234, 228),
    "mineral_rich_area": (169, 127, 74),
    "dust_field": (208, 138, 90),
    "lava_tube": (58, 35, 28),
    "frozen_basin": (185, 169, 155),
}
TINTA_DEFAULT = (170, 94, 58)
#: La cella madre non appartiene a nessun distretto: resta di competenza diretta
#: del governo, e nella figura ha un colore proprio perche' quella e' una
#: proprieta' del disegno sperimentale e non un caso.
TINTA_MADRE = "#2f3b4a"
TINTA_OPERE = "#8d8d8d"

ESAGERA = 0.42      # esagerazione verticale del rilievo
ALTEZZA = 0.62      # scala delle colonne sul logaritmo della popolazione


def _rgb(t: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(v / 255 for v in t)


def istantanee(braccio: Path, seme: int) -> list[Path]:
    for cartella in sorted(braccio.glob(f"*seed{seme}")):
        f = sorted((cartella / "world_snapshots").glob("step_*.json"))
        if f:
            return f
    return []


def distretto_per_cella(snap: dict) -> dict[tuple[int, int], int]:
    """(x, y) -> indice del distretto, dall'ultima tornata amministrativa."""
    fuori: dict[tuple[int, int], int] = {}
    for d in (snap.get("administrators") or {}).get("last_round") or []:
        if "district" not in d:
            continue
        for c in d.get("cells") or []:
            # Il registro scrive le celle come (y, x), come in analisi_decentramento.
            fuori[(int(c[1]), int(c[0]))] = int(d["district"])
    return fuori


def centro_e_raggio(snap: dict, margine: int = 2) -> tuple[int, int, int]:
    """Cella madre e raggio che contiene tutto cio' che e' occupato."""
    celle = snap["cells"]
    pop = {(c["x"], c["y"]): len(c.get("agents_present") or []) for c in celle}
    madre = (snap.get("administrators") or {}).get("mother_cell")
    if madre:
        cx, cy = int(madre[1]), int(madre[0])
    else:
        cx, cy = max(pop, key=pop.get) if pop else (0, 0)
    occupate = [(c["x"], c["y"]) for c in celle
                if c.get("agents_present") or c.get("structures")]
    r = max((max(abs(x - cx), abs(y - cy)) for x, y in occupate), default=3)
    return cx, cy, r + margine


def disegna(ax, snap: dict, centro: tuple[int, int, int], colore_distretto) -> None:
    cx, cy, r = centro
    celle = snap["cells"]
    pop = {(c["x"], c["y"]): len(c.get("agents_present") or []) for c in celle}
    quota = {(c["x"], c["y"]): float(c.get("elevation") or 0.0) for c in celle}
    terreno = {(c["x"], c["y"]): c.get("terrain") for c in celle}
    opere = {(c["x"], c["y"]): len(c.get("structures") or []) for c in celle}
    distretti = distretto_per_cella(snap)
    madre = (snap.get("administrators") or {}).get("mother_cell")
    # Il registro scrive la cella madre come (y, x), come le celle dei distretti.
    madre = (int(madre[1]), int(madre[0])) if madre else None

    dentro = lambda x, y: cx - r <= x <= cx + r and cy - r <= y <= cy + r  # noqa: E731

    # Il suolo, una piastrella per cella nota.
    for (x, y), z in quota.items():
        if not dentro(x, y):
            continue
        base = z * ESAGERA
        ax.bar3d(x - cx - 0.5, y - cy - 0.5, base - 0.10, 1, 1, 0.10,
                 color=_rgb(TINTE_TERRENO.get(terreno.get((x, y)), TINTA_DEFAULT)),
                 shade=True, edgecolor="#00000014", linewidth=0.12, zsort="max")

    # Le celle con sole opere e nessun abitante: la corona.
    for (x, y), s in opere.items():
        if s <= 0 or pop.get((x, y), 0) > 0 or not dentro(x, y):
            continue
        ax.bar3d(x - cx - 0.22, y - cy - 0.22, quota.get((x, y), 0.0) * ESAGERA,
                 0.44, 0.44, 0.12 + min(s, 8) * 0.03,
                 color=TINTA_OPERE, shade=True, edgecolor="none", alpha=0.9, zsort="max")

    # Le colonne abitate, dal fondo verso l'osservatore perche' si sovrappongano bene.
    for (x, y), n in sorted(pop.items(), key=lambda kv: (kv[0][1], -kv[0][0])):
        if n <= 0 or not dentro(x, y):
            continue
        h = ALTEZZA * math.log1p(n)
        e_madre = madre is not None and (x, y) == madre
        d = distretti.get((x, y))
        col = TINTA_MADRE if e_madre else (colore_distretto(d) if d is not None else "#9a9a9a")
        w = 0.66 if e_madre else 0.56
        ax.bar3d(x - cx - w / 2, y - cy - w / 2, quota.get((x, y), 0.0) * ESAGERA,
                 w, w, h, color=col, shade=True,
                 edgecolor="#00000033", linewidth=0.18, zsort="max")

    ax.set_xlim(-r - 0.5, r + 0.5)
    ax.set_ylim(-r - 0.5, r + 0.5)
    ax.set_zlim(-2.0, ALTEZZA * math.log1p(max(pop.values() or [1])) + 1.4)
    ax.set_box_aspect((1, 1, 0.46))
    ax.view_init(elev=34, azim=-56)
    ax.set_axis_off()          # niente assi, niente riquadro: solo il mondo
    ax.patch.set_alpha(0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/campagna_scarsa_v3/llm_completo_amm")
    ap.add_argument("--seme", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RADICE / "Tesi_LaTex/figures/mondo3d")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.patches import Patch

    plt.rcParams.update({"pdf.fonttype": 42, "figure.dpi": 220})

    f = istantanee(RADICE / args.run, args.seme)
    if not f:
        raise SystemExit(f"nessuna istantanea in {args.run} per il seme {args.seme}")
    scelte = [f[0], f[len(f) // 2], f[-1]]
    ultimo = json.loads(scelte[-1].read_text(encoding="utf-8"))
    centro = centro_e_raggio(ultimo)

    tavolozza = list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)

    def colore_distretto(d: int):
        return tavolozza[d % len(tavolozza)]

    fig = plt.figure(figsize=(7.2, 2.35))
    fig.patch.set_alpha(0.0)
    for i, percorso in enumerate(scelte):
        snap = ultimo if i == 2 else json.loads(percorso.read_text(encoding="utf-8"))
        ax = fig.add_subplot(1, 3, i + 1, projection="3d", computed_zorder=False)
        n = sum(len(c.get("agents_present") or []) for c in snap["cells"])
        occ = sum(1 for c in snap["cells"]
                  if c.get("agents_present") or c.get("structures"))
        dist = (snap.get("administrators") or {}).get("districts") or 0
        # **Il riquadro tridimensionale va fatto traboccare.** Matplotlib riserva
        # a un asse 3d una regione quadrata, e con un rilievo schiacciato e la
        # vista dall'alto il disegno ne occupa solo la fascia centrale: a riquadro
        # contenuto la figura sarebbe per due terzi vuota. Allargando l'asse oltre
        # la propria cella il disegno riempie la colonna, e il titolo si scrive a
        # parte perche' quello del riquadro finirebbe fuori pagina.
        ax.set_position([i / 3 - 0.015, -0.46, 1 / 3 + 0.03, 1.80])
        disegna(ax, snap, centro, colore_distretto)
        fig.text(i / 3 + 1 / 6, 0.99,
                 f"passo {int(snap.get('step') or 0)}\n"
                 f"{n} coloni, {occ} celle, {dist} distretti",
                 ha="center", va="top", fontsize=7.5, linespacing=1.35)

    voci = [
        Patch(facecolor=TINTA_MADRE, label="cella madre (governo)"),
        Patch(facecolor=colore_distretto(1), label="cella di un distretto"),
        Patch(facecolor=TINTA_OPERE, label="solo opere, nessun abitante"),
        Patch(facecolor=_rgb(TINTE_TERRENO["regolith_plain"]), label="suolo, alla quota reale"),
    ]
    fig.legend(handles=voci, loc="lower center", ncol=4, frameon=False,
               fontsize=6.6, bbox_to_anchor=(0.5, -0.015), handlelength=1.1)
    # Niente subplots_adjust: la posizione di ciascun asse e' fissata sopra.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(args.out.with_suffix(ext), transparent=True, bbox_inches="tight")
    plt.close(fig)
    ritaglia(args.out)
    print(f"scritto {args.out}.png / .pdf")
    return 0


def ritaglia(base: Path, margine: float = 4.0) -> None:
    """Toglie il bordo vuoto da PNG e PDF.

    `bbox_inches='tight'` ritaglia al riquadro degli assi, non al disegno: un
    asse tridimensionale occupa una regione quadrata di cui il rilievo riempie
    solo una fascia, quindi resta un bordo vuoto che in pagina rimpicciolisce la
    figura. Qui si misura il contenuto davvero disegnato e si ritaglia a quello.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    png = base.with_suffix(".png")
    if png.exists():
        with Image.open(png) as im:
            im = im.convert("RGBA")
            riquadro = im.getchannel("A").getbbox()
            if riquadro:
                m = int(margine)
                riquadro = (max(0, riquadro[0] - m), max(0, riquadro[1] - m),
                            min(im.width, riquadro[2] + m), min(im.height, riquadro[3] + m))
                im.crop(riquadro).save(png)
                frazione = (riquadro[0] / im.width, riquadro[1] / im.height,
                            riquadro[2] / im.width, riquadro[3] / im.height)
            else:
                return
    else:
        return

    pdf = base.with_suffix(".pdf")
    if not pdf.exists():
        return
    try:
        import fitz
    except ImportError:
        return
    # Le stesse frazioni misurate sul PNG, applicate al riquadro del PDF: i due
    # file escono dalla stessa figura, quindi la proporzione e' la stessa.
    d = fitz.open(pdf)
    p = d[0]
    r = p.rect
    p.set_cropbox(fitz.Rect(
        r.x0 + frazione[0] * r.width, r.y0 + frazione[1] * r.height,
        r.x0 + frazione[2] * r.width, r.y0 + frazione[3] * r.height))
    d.save(pdf.with_suffix(".tmp.pdf"))
    d.close()
    pdf.unlink()
    pdf.with_suffix(".tmp.pdf").rename(pdf)


if __name__ == "__main__":
    raise SystemExit(main())
