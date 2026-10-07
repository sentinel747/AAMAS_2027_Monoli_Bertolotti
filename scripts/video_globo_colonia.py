# -*- coding: utf-8 -*-
"""Video per la presentazione di laurea: la colonia che si espande sul pianeta.

Due pannelli affiancati sullo stesso Marte, con la stessa luce e la stessa camera
obliqua "da drone": a sinistra la colonia senza governo, a destra la stessa
colonia con governatore e amministratori linguistici. Si parte dai due globi
interi al passo 25 e si scende fin sopra il sito di atterraggio, dove si vedono
gli edifici; poi la colonia cresce istantanea dopo istantanea (una ogni 25
passi) e la camera si allarga con lei, seguendo il raggio della colonia piu'
estesa, fino al passo 1000; infine si risale al globo intero, per vedere quanta
parte del pianeta ciascuna occupa. Sotto ogni pannello le celle occupate e i coloni, dalle metriche delle
istantanee.

Stessa run e stesso seme della figura del globo in tesi (`runs/base_pulita`,
seme 7, `ctrl_none` contro `variante_D`): a fine run 175 celle e 1621 coloni
senza governo, 77 celle e 1877 coloni con governo. Il pianeta e' quello di
`scripts/globo_marte.py`; gli edifici sono i modelli per tipo di
`scripts/edifici_marte.py`, disposti senza sovrapposizioni su una griglia di posti
(come nell'interfaccia) dimensionata sulla cella piu' affollata delle due run,
cosi' che abbiano la stessa grandezza nei due pannelli e per tutto il video.

Uso:
    python scripts/video_globo_colonia.py
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import globo_marte as G  # noqa: E402
import edifici_marte as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INCHIOSTRO = (31, 36, 48)
SECONDARIO = (107, 111, 120)
GRIGIO = (138, 141, 147)
RUGGINE = (193, 68, 14)
FONT = "C:/Windows/Fonts/calibri.ttf"
FONT_B = "C:/Windows/Fonts/calibrib.ttf"

W, H = 1920, 1080
PW, PH = 880, 740                      # pannello
PANNELLI = [(60, 110), (980, 110)]     # angolo in alto a sinistra
SS = 1.5
LONTANO = (2.6, 0.0)                   # (distanza in raggi, inclinazione in gradi)
MEDIO = (0.42, 42.0)                   # la colonia intera
VICINO = (0.019, 55.0)                 # la cella madre, con gli edifici


def font(dim, grassetto=False):
    return ImageFont.truetype(FONT_B if grassetto else FONT, dim)


def morbido(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def fra(a, b, e):
    """Da una posa della camera all'altra: distanza geometrica, inclinazione lineare."""
    return a[0] * (b[0] / a[0]) ** e, a[1] + (b[1] - a[1]) * e


class Braccio:
    def __init__(self, cartella: Path, titolo: str, tinta):
        self.titolo, self.tinta = titolo, tinta
        self.file = sorted((cartella / "world_snapshots").glob("step_*.json"))
        self.pianeta = G.Pianeta(cartella / "world_static_base.json")

    def carica(self, i):
        self.snap = json.loads(self.file[i].read_text(encoding="utf-8"))
        self.pianeta.applica(self.snap)
        self.albedo = self.pianeta.albedo()
        m = self.snap["metrics"]
        self.passo = int(self.snap["step"])
        self.celle = int(round(m.get("occupied_cells", len(G.celle_occupate(self.snap)))))
        self.coloni = int(m.get("population", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seme", type=int, default=7)
    ap.add_argument("--base", type=Path, default=ROOT / "runs/base_pulita/ctrl_none")
    ap.add_argument("--gov", type=Path, default=ROOT / "runs/base_pulita/variante_D")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", type=Path, default=ROOT / "Tesi_LaTex/PPT_Tesi_LaTex/media/colonia_globo.mp4")
    ap.add_argument("--prova", type=Path, default=None, help="solo un fotogramma a fine run, in questo PNG")
    args = ap.parse_args()

    def seme(b):
        return next(d for d in sorted(b.glob(f"*seed{args.seme}")) if d.is_dir())

    bracci = [Braccio(seme(args.base), "Senza governo", GRIGIO),
              Braccio(seme(args.gov), "Governatore e amministratori LLM", RUGGINE)]
    n = min(len(b.file) for b in bracci)
    for b in bracci:
        b.carica(0)
    cx, cy = max(G.celle_occupate(bracci[1].snap).items(), key=lambda kv: kv[1])[0]
    lat_c, lon_c = map(float, G.cella_a_latlon(cx, cy))
    zona = (cx - 24, cy - 24, cx + 25, cy + 25)
    fine = (cx - 11, cy - 11, cx + 12, cy + 12)
    vicina = (cx - 3, cy - 3, cx + 4, cy + 4)
    ultime = [json.loads(b.file[n - 1].read_text(encoding="utf-8")) for b in bracci]
    PASSO = E.passo_posti(*ultime)      # stessa griglia, e stessa grandezza degli edifici, nei due pannelli

    def camera(posa, e_rot=0.0):
        d, incl = posa
        # durante la discesa il pianeta ruota fino a portare il sito al centro;
        # da vicino il bersaglio e' appena oltre il centro della cella madre
        spinta = 0.12 * max(0.0, 1 - (d - VICINO[0]) / (MEDIO[0] - VICINO[0])) if d < MEDIO[0] else 0.0
        return G.CameraObliqua(lat_c + 14 * e_rot + spinta, lon_c + 38 * e_rot, d, incl, int(PW * SS), int(PH * SS))

    luce = E.luce_da_camera(camera(MEDIO))

    def livelli(b, globale=None, fitta=False):
        tg = globale if globale is not None else b.pianeta.cuoci(b.albedo, (0, 0, 360, 180), 4)
        liv = [tg, b.pianeta.cuoci(b.albedo, zona, 24), [(b.pianeta.cuoci(b.albedo, fine, 64), fine, 64)]]
        if fitta:
            liv[2].append((b.pianeta.cuoci(b.albedo, vicina, 160), vicina, 160))
        return liv

    def pannello(b, liv, scena, cam):
        img, _ = G.disegna_obliquo(cam, liv[0], 4, liv[1], zona, 24, livelli_fini=liv[2])
        im = Image.fromarray(img).convert("RGBA")
        scena.disegna(im, cam)
        return np.asarray(im.convert("RGB").resize((PW, PH), Image.LANCZOS)).astype(np.float32)

    m = Image.new("L", (PW, PH), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, PW - 1, PH - 1], radius=26, fill=255)
    maschera = (np.asarray(m).astype(np.float32) / 255)[..., None]

    def componi(pannelli, passo, numeri):
        tela = Image.new("RGB", (W, H), "white")
        for (x0, y0), pan, b, (celle, coloni) in zip(PANNELLI, pannelli, bracci, numeri):
            riquadro = np.asarray(tela.crop((x0, y0, x0 + PW, y0 + PH))).astype(np.float32)
            tela.paste(Image.fromarray((pan * maschera + riquadro * (1 - maschera)).astype(np.uint8)), (x0, y0))
            d = ImageDraw.Draw(tela)
            d.rounded_rectangle([x0 - 2, y0 - 2, x0 + PW + 1, y0 + PH + 1], radius=28, outline=b.tinta, width=5)
            cxp = x0 + PW // 2
            d.text((cxp, 70), b.titolo, font=font(40, True), fill=b.tinta, anchor="mm")
            d.text((cxp - 20, 935), f"{celle}", font=font(64, True), fill=b.tinta, anchor="rm")
            d.text((cxp - 5, 942), "cella occupata" if celle == 1 else "celle occupate", font=font(28), fill=SECONDARIO, anchor="lm")
            d.text((cxp, 1000), f"{coloni} coloni", font=font(30), fill=SECONDARIO, anchor="mm")
        d = ImageDraw.Draw(tela)
        d.text((W / 2, 28), f"passo {passo}  ·  anno {passo * 7 / 365.25:.0f}", font=font(26), fill=SECONDARIO, anchor="mm")
        return tela

    if args.prova:
        for b in bracci:
            b.carica(n - 1)
        pose = {"medio": MEDIO, "vicino": VICINO}
        for nome, posa in pose.items():
            fitta = nome == "vicino"
            sc = [E.Scena(b.snap, luce, PASSO, fine=fitta) for b in bracci]
            pan = [pannello(b, livelli(b, fitta=fitta), s, camera(posa)) for b, s in zip(bracci, sc)]
            componi(pan, bracci[0].passo, [(b.celle, b.coloni) for b in bracci]).save(
                args.prova.with_name(args.prova.stem + f"_{nome}.png"))
        return 0

    fps = args.fps
    tmp = Path(tempfile.mkdtemp(prefix="video_globo_"))
    k = 0

    def scrivi(tela):
        nonlocal k
        tela.save(tmp / f"f{k:05d}.png")
        k += 1

    # La camera parte vicina, sul sito di atterraggio, e si allarga con la
    # colonia: la distanza segue il raggio della colonia piu' estesa delle due
    # (in celle, dalla cella madre), un'istantanea avanti perche' chi cresce
    # resti in quadro. Stessa camera nei due pannelli.
    raggi = []
    for i in range(n):
        r = 0
        for b in bracci:
            s = json.loads(b.file[i].read_text(encoding="utf-8"))
            r = max([r] + [max(abs(x - cx), abs(y - cy)) for x, y in G.celle_occupate(s)])
        raggi.append(r)
    raggi = np.maximum.accumulate(np.array(raggi, float))
    raggi = np.convolve(np.pad(raggi, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")  # senza scatti
    raggi = np.maximum.accumulate(np.append(raggi[1:], raggi[-1]))                          # un passo avanti

    def posa(r):
        d = VICINO[0] + 0.044 * r
        e = min(1.0, math.log(d / VICINO[0]) / math.log(MEDIO[0] / VICINO[0]))
        return d, VICINO[1] + (MEDIO[1] - VICINO[1]) * e

    def fotogramma(cam, liv, sc, passo, numeri):
        return componi([pannello(b, l, s, cam) for b, l, s in zip(bracci, liv, sc)], passo, numeri)

    liv = [livelli(b, fitta=True) for b in bracci]
    sc = [E.Scena(b.snap, luce, PASSO, fine=True) for b in bracci]
    numeri = [(b.celle, b.coloni) for b in bracci]
    # A: dal globo intero si scende sul sito di atterraggio, gia' da vicino
    for i in range(int(3.0 * fps)):
        e = morbido((i / fps - 0.5) / 2.5)
        tela = fotogramma(camera(fra(LONTANO, posa(raggi[0]), e), 1 - e), liv, sc, bracci[0].passo, numeri)
        scrivi(tela)
    for _ in range(int(0.8 * fps)):
        scrivi(tela)
    # B: la colonia cresce, un'istantanea ogni 25 passi, e la camera si allarga
    for i in range(1, n):
        for b in bracci:
            b.carica(i)
        da, a = posa(raggi[i - 1]), posa(raggi[i])
        vicino = a[0] < 0.1
        liv = [livelli(b, globale=l[0], fitta=vicino) for b, l in zip(bracci, liv)]
        sc = [E.Scena(b.snap, luce, PASSO, fine=vicino) for b in bracci]
        numeri = [(b.celle, b.coloni) for b in bracci]
        for j in range(8):
            if j == 0 or abs(a[0] - da[0]) > 1e-6:
                tela = fotogramma(camera(fra(da, a, (j + 1) / 8)), liv, sc, bracci[0].passo, numeri)
            scrivi(tela)
    passo = bracci[0].passo
    poster = tela
    for _ in range(int(1.4 * fps)):
        scrivi(poster)
    # C: si risale al globo intero, con le colonie finali sul pianeta
    liv = [[b.pianeta.cuoci(b.albedo, (0, 0, 360, 180), 4)] + l[1:] for b, l in zip(bracci, liv)]
    fine_posa = posa(raggi[-1])
    for i in range(int(3.0 * fps)):
        e = morbido(i / (3.0 * fps - 1))
        tela = fotogramma(camera(fra(fine_posa, LONTANO, e)), liv, sc, passo, numeri)
        scrivi(tela)
    finale = tela
    for _ in range(int(2.0 * fps)):
        scrivi(finale)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    poster.save(args.out.with_suffix(".png"))
    finale.save(args.out.with_name(args.out.stem + "_intero.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(args.out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"scritto {args.out} ({k} fotogrammi, {k / fps:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
