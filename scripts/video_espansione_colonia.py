# -*- coding: utf-8 -*-
"""Video per la presentazione di laurea: la stessa colonia, con e senza governo.

In alto due mappe dall'alto che crescono in parallelo, istantanea dopo
istantanea (una ogni 25 passi), con passaggi sfumati; in basso le due curve del
territorio che si disegnano in sincrono. Stessa run della figura del globo in
tesi (`runs/base_pulita`, seme 7: `ctrl_none` contro `variante_D`), stesso
riquadro e stessa scala di colore nelle due mappe, cosi' che le estensioni si
confrontino a occhio. Sfondo bianco, 1920x1080, H.264: si incorpora in PowerPoint.

Le mappe non sono ritoccate: ogni quadratino e' una cella con coloni o opere,
il colore e' il numero di coloni (scala logaritmica); le celle con sole opere
sono grigie. La curva e' `occupied_cells` di `state_timeseries.csv`, che a fine
run coincide con `celle_totali`.

Uso:
    python scripts/video_espansione_colonia.py
    python scripts/video_espansione_colonia.py --seme 3 --base runs/... --gov runs/...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INCHIOSTRO = "#1f2430"
SECONDARIO = "#6b6f78"
TINTA_BASE = "#8a8d93"   # colonia senza governo
TINTA_GOV = "#c1440e"    # colonia governata (ruggine marziana)
OPERE = "#d9d6d2"        # celle con sole opere
COLONI = LinearSegmentedColormap.from_list(
    "coloni", ["#f6d7b8", "#eba36a", "#d9652b", "#a8320b", "#5e1a05"])


def cartella_seme(braccio: Path, seme: int) -> Path:
    return next(d for d in sorted(braccio.glob(f"*seed{seme}")) if d.is_dir())


def istantanee(cartella: Path) -> list[dict]:
    out = []
    for f in sorted((cartella / "world_snapshots").glob("step_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        celle = {}
        for c in d["cells"]:
            n = len(c.get("agents_present") or [])
            if n or c.get("structures"):
                celle[(c["x"], c["y"])] = n
        out.append({"passo": int(d["step"]), "celle": celle,
                    "coloni": int(d["metrics"].get("population", 0))})
    return out


def curva(cartella: Path) -> list[float]:
    with (cartella / "state_timeseries.csv").open(encoding="utf-8") as fh:
        return [float(r["occupied_cells"]) for r in csv.DictReader(fh)]


def fotogramma(i, base, gov, curve, riquadro, massimo, passi_tot, percorso: Path):
    b, g = base[i], gov[i]
    x0, y0, lato = riquadro
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    griglia = fig.add_gridspec(2, 2, height_ratios=(3.1, 1.0), left=0.05, right=0.95,
                               top=0.86, bottom=0.08, hspace=0.32, wspace=0.10)
    for j, (snap, titolo, tinta) in enumerate(
            ((b, "Senza governo", TINTA_BASE),
             (g, "Governatore e amministratori LLM", TINTA_GOV))):
        ax = fig.add_subplot(griglia[0, j])
        img = np.full((lato, lato), np.nan)
        opere = np.full((lato, lato), np.nan)
        for (x, y), n in snap["celle"].items():
            if 0 <= x - x0 < lato and 0 <= y - y0 < lato:
                if n > 0:
                    img[y - y0, x - x0] = n
                else:
                    opere[y - y0, x - x0] = 1
        ax.imshow(opere, cmap=LinearSegmentedColormap.from_list("o", [OPERE, OPERE]),
                  interpolation="nearest", origin="upper")
        ax.imshow(img, cmap=COLONI, norm=LogNorm(vmin=1, vmax=massimo),
                  interpolation="nearest", origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e3e1de"); s.set_linewidth(1.2)
        ax.set_title(titolo, fontsize=26, color=tinta, fontweight="bold", pad=14, loc="left")
        n_celle = len(snap["celle"])
        ax.text(0.0, -0.06, f"{n_celle} {'cella' if n_celle == 1 else 'celle'}", transform=ax.transAxes,
                fontsize=30, color=INCHIOSTRO, fontweight="bold", va="top")
        ax.text(1.0, -0.068, f"{snap['coloni']} coloni", transform=ax.transAxes,
                fontsize=22, color=SECONDARIO, va="top", ha="right")
    fig.text(0.05, 0.93, f"passo {g['passo']:>4} / {passi_tot}", fontsize=24,
             color=SECONDARIO, family="monospace")
    ax = fig.add_subplot(griglia[1, :])
    fino = g["passo"]
    for serie, tinta, nome in ((curve[0], TINTA_BASE, "senza governo"),
                               (curve[1], TINTA_GOV, "governata")):
        ax.plot(range(1, min(fino, len(serie)) + 1), serie[:fino], color=tinta, lw=4)
        if fino <= len(serie):
            ax.scatter([fino], [serie[fino - 1]], color=tinta, s=90, zorder=3)
            if fino >= 150:  # prima le due curve coincidono e le etichette si sovrappongono
                ax.text(fino + 12, serie[fino - 1], nome, color=tinta, fontsize=18, va="center")
    ax.set_xlim(0, passi_tot * 1.13)
    ax.set_ylim(0, max(max(curve[0]), max(curve[1])) * 1.15)
    ax.set_ylabel("celle occupate", fontsize=18, color=SECONDARIO)
    ax.tick_params(colors=SECONDARIO, labelsize=15)
    for lato_ in ("top", "right"):
        ax.spines[lato_].set_visible(False)
    for lato_ in ("left", "bottom"):
        ax.spines[lato_].set_color("#c9c6c1")
    ax.grid(axis="y", color="#eeece9", lw=1)
    fig.savefig(percorso, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="runs/base_pulita/ctrl_none")
    ap.add_argument("--gov", default="runs/base_pulita/variante_D")
    ap.add_argument("--seme", type=int, default=7)
    ap.add_argument("--passi", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sfumature", type=int, default=8, help="fotogrammi fra due istantanee")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "Tesi_LaTex/PPT_Tesi_LaTex/media/espansione_colonia.mp4")
    args = ap.parse_args()

    cb, cg = cartella_seme(ROOT / args.base, args.seme), cartella_seme(ROOT / args.gov, args.seme)
    base, gov = istantanee(cb), istantanee(cg)
    n = min(len(base), len(gov))
    base, gov = base[:n], gov[:n]
    curve = (curva(cb), curva(cg))

    # Un riquadro solo per le due mappe, sull'unione delle celle a fine run.
    tutte = list(base[-1]["celle"]) + list(gov[-1]["celle"])
    xs, ys = [x for x, _ in tutte], [y for _, y in tutte]
    lato = max(max(xs) - min(xs), max(ys) - min(ys)) + 5
    riquadro = ((min(xs) + max(xs)) // 2 - lato // 2, (min(ys) + max(ys)) // 2 - lato // 2, lato)
    massimo = max(max(s["celle"].values() or [1]) for s in base + gov)

    lavoro = Path(tempfile.mkdtemp(prefix="video_colonia_"))
    chiavi = []
    for i in range(n):
        p = lavoro / f"chiave_{i:03d}.png"
        fotogramma(i, base, gov, curve, riquadro, massimo, args.passi, p)
        chiavi.append(p)
        print(f"istantanea {i + 1}/{n}", end="\r")
    print()

    # Passaggi sfumati fra istantanee consecutive, poi una pausa sull'ultima.
    k = 0
    for i in range(n):
        a = Image.open(chiavi[i]).convert("RGB")
        if i + 1 < n:
            b = Image.open(chiavi[i + 1]).convert("RGB")
            for t in range(args.sfumature):
                Image.blend(a, b, t / args.sfumature).save(lavoro / f"f_{k:05d}.png"); k += 1
        else:
            for _ in range(args.fps * 3):
                a.save(lavoro / f"f_{k:05d}.png"); k += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", str(lavoro / "f_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(args.out)], check=True)
    shutil.copy(chiavi[-1], args.out.with_suffix(".png"))   # copertina = fotogramma finale
    shutil.copy(chiavi[0], args.out.with_name(args.out.stem + "_inizio.png"))
    shutil.rmtree(lavoro, ignore_errors=True)
    print(f"scritto {args.out} ({k} fotogrammi, {k / args.fps:.1f} s)")
    print(f"fine run: senza governo {len(base[-1]['celle'])} celle / {base[-1]['coloni']} coloni; "
          f"governata {len(gov[-1]['celle'])} celle / {gov[-1]['coloni']} coloni")


if __name__ == "__main__":
    main()
