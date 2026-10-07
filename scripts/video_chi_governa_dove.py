# -*- coding: utf-8 -*-
"""Video per la presentazione di laurea: chi governa dove.

Il secondo tratto del comportamento osservato in tesi e' la divisione dei ruoli
per geografia: il governatore tiene la cella madre, dove vive quasi tutta la
popolazione; gli amministratori governano i distretti di frontiera, dove cade
quasi ogni morte (tesi, cap. 6: fra il 73 e l'81% dei coloni nella cella madre,
fra il 97 e il 100% dei decessi fuori di essa). Questo video lo fa vedere sulla
stessa run del video della colonia (`runs/base_pulita/variante_D`, seme 7), sul
pianeta di `scripts/globo_marte.py`, con una camera obliqua "da drone":

- gli edifici sono modelli per tipo (`edifici_marte.py`: cupole, habitat, serre,
  pannelli, serbatoi...), disposti senza sovrapposizioni su una griglia di posti
  (come nell'interfaccia) dimensionata sulla cella piu' affollata;
- la cella madre, contornata di scuro, e' di competenza del governatore;
- i distretti degli amministratori sono segnati dai loro confini sul suolo, con
  una tinta leggera (colori arbitrari, servono solo a distinguerli); quando in
  una tornata l'amministratore di un distretto riscrive la legge, il distretto
  si accende di ruggine;
- ogni morto e' una croce nella posizione esatta in cui e' morto, e resta;
- le nascite dell'intervallo sono impulsi verdi sulla cella, di area
  proporzionale al loro numero.

A destra, dalle istantanee e dai registri: la quota di coloni che vive nella
cella madre e la quota cumulata di morti fuori di essa. Su questo seme: 80% e
100% (583 morti su 583). Le nascite nella cella madre sono il 58%, il resto
sparso sulle altre 76 celle: il video le mostra, ma non ne fa un contatore,
perche' la tesi afferma la geografia della popolazione e dei decessi.

Uso:
    python scripts/video_chi_governa_dove.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import globo_marte as G  # noqa: E402
import edifici_marte as E  # noqa: E402
from video_globo_colonia import GRIGIO, INCHIOSTRO, RUGGINE, SECONDARIO, font, morbido  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
PX, PY, PW, PH = 50, 100, 1040, 900     # pannello della vista
SS = 1.5
MADRE_BORDO = (31, 36, 48)
VERDE = (46, 160, 67)
MORTO = (225, 29, 72)
#: Tinte dei distretti: solo per distinguerli, nessun significato.
TINTE = [(96, 165, 250), (192, 132, 252), (250, 204, 21), (45, 212, 191),
         (244, 114, 182), (163, 230, 53), (129, 140, 248), (251, 146, 60)]
#: Camera: dal globo intero alla vista obliqua sulla colonia.
LONTANO = (2.6, 0.0)       # (distanza in raggi, inclinazione in gradi)
VICINO = (0.30, 46.0)


def carica_eventi(cartella: Path):
    morti = [json.loads(r) for r in (cartella / "dead_agents.jsonl").read_text(encoding="utf-8").splitlines() if r.strip()]
    nascite = []
    with (cartella / "events.jsonl").open(encoding="utf-8") as fh:
        for riga in fh:
            if '"population_grew"' in riga:
                e = json.loads(riga)
                nascite.append((int(e["step"]), int(e["data"]["x"]), int(e["data"]["y"])))
    riscritture = {}
    with (cartella / "administrator_decisions.jsonl").open(encoding="utf-8") as fh:
        for riga in fh:
            r = json.loads(riga)
            riscritture[int(r["step"])] = {
                int(d["district"]) for d in r.get("last_round") or []
                if "district" in d and not d.get("accepted_government_policy")
                and not d.get("provider_failure") and not d.get("malformed")}
    return morti, nascite, riscritture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "runs/base_pulita/variante_D")
    ap.add_argument("--seme", type=int, default=7)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", type=Path, default=ROOT / "Tesi_LaTex/PPT_Tesi_LaTex/media/chi_governa_dove.mp4")
    ap.add_argument("--prova", type=Path, default=None, help="solo il fotogramma finale, in questo PNG")
    args = ap.parse_args()
    cartella = next(d for d in sorted(args.run.glob(f"*seed{args.seme}")) if d.is_dir())
    file = sorted((cartella / "world_snapshots").glob("step_*.json"))
    pianeta = G.Pianeta(cartella / "world_static_base.json")
    morti, nascite, riscritture = carica_eventi(cartella)

    snap0 = json.loads(file[0].read_text(encoding="utf-8"))
    madre = max(G.celle_occupate(snap0).items(), key=lambda kv: kv[1])[0]
    lat_c, lon_c = map(float, G.cella_a_latlon(*madre))
    zona = (madre[0] - 24, madre[1] - 24, madre[0] + 25, madre[1] + 25)
    fine = (madre[0] - 11, madre[1] - 11, madre[0] + 12, madre[1] + 12)

    def camera(e):
        d = LONTANO[0] * (VICINO[0] / LONTANO[0]) ** e
        incl = LONTANO[1] + (VICINO[1] - LONTANO[1]) * e
        return G.CameraObliqua(lat_c, lon_c, d, incl, int(PW * SS), int(PH * SS))

    def poligono_cella(cam, x, y, passi=4):
        """Contorno della cella sul suolo, suddiviso perche' segua la curvatura."""
        t = np.linspace(-0.5, 0.5, passi + 1)
        bordo = ([(a, -0.5) for a in t] + [(0.5, b) for b in t[1:]] +
                 [(a, 0.5) for a in t[::-1][1:]] + [(-0.5, b) for b in t[::-1][1:-1]])
        P = np.array([G.punto_superficie(x, y, a, b, 0.2) for a, b in bordo])
        px, _, vis = cam.proietta(P)
        return [tuple(p) for p in px] if vis.all() else None

    def lato_cella(cam, x, y, lato, passi=4):
        t = np.linspace(-0.5, 0.5, passi + 1)
        pts = {"n": [(a, -0.5) for a in t], "s": [(a, 0.5) for a in t],
               "o": [(-0.5, b) for b in t], "e": [(0.5, b) for b in t]}[lato]
        P = np.array([G.punto_superficie(x, y, a, b, 0.2) for a, b in pts])
        px, _, vis = cam.proietta(P)
        return [tuple(p) for p in px] if vis.all() else None

    def punto(cam, fx, fy):
        px, _, vis = cam.proietta(np.array([G.punto_superficie(fx, fy, 0, 0, 0.3)]))
        return (px[0], vis[0])

    def pannello(snap, passo_da, passo_a, cam, livelli, scena, overlay=True):
        tex_g, tex_z, tex_f = livelli[:3]
        piu_fini = [(tex_f, fine, 64)] + ([livelli[3]] if len(livelli) > 3 else [])
        img, _ = G.disegna_obliquo(cam, tex_g, 4, tex_z, zona, 24, livelli_fini=piu_fini)
        base = Image.fromarray(img).convert("RGBA")
        if overlay:
            velo = Image.new("RGBA", base.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(velo)
            distretto = {}
            for dist in (snap.get("administrators") or {}).get("last_round") or []:
                if "district" in dist:
                    for c in dist.get("cells") or []:
                        distretto[(int(c[1]), int(c[0]))] = int(dist["district"])
            riscritti = riscritture.get(int(snap["step"]), set())
            # tinta leggera, e ruggine dove l'amministratore riscrive
            for (x, y), k in distretto.items():
                pol = poligono_cella(cam, x, y)
                if pol:
                    if k in riscritti:
                        d.polygon(pol, fill=RUGGINE + (120,))
                    else:
                        d.polygon(pol, fill=TINTE[k % len(TINTE)] + (46,))
            # confini fra distretti diversi (o verso l'esterno)
            vicini = {"n": (0, -1), "s": (0, 1), "o": (-1, 0), "e": (1, 0)}
            for (x, y), k in distretto.items():
                colore = RUGGINE if k in riscritti else TINTE[k % len(TINTE)]
                for lato, (dx, dy) in vicini.items():
                    if distretto.get((x + dx, y + dy)) != k:
                        seg = lato_cella(cam, x, y, lato)
                        if seg:
                            d.line(seg, fill=colore + (235,), width=5)
            base.alpha_composite(velo)
        scena.disegna(base, cam)
        if overlay:
            velo = Image.new("RGBA", base.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(velo)
            pol = poligono_cella(cam, madre[0], madre[1], passi=6)
            if pol:
                d.line(pol + [pol[0]], fill=MADRE_BORDO + (255,), width=8)
            for m in morti:
                if m["death_step"] > passo_a:
                    continue
                fx = m["x"] + (m.get("local_x_m") or 29500) / 59163.0 - 0.5
                fy = m["y"] + (m.get("local_y_m") or 29500) / 59166.0 - 0.5
                (u, v), vis = punto(cam, fx, fy)
                if vis:
                    s = 11 if m["death_step"] > passo_da else 8
                    alfa = 255 if m["death_step"] > passo_da else 200
                    for colore, w in (((255, 255, 255, alfa), 7), (MORTO + (alfa,), 4)):
                        d.line([(u - s, v - s), (u + s, v + s)], fill=colore, width=w)
                        d.line([(u - s, v + s), (u + s, v - s)], fill=colore, width=w)
            conta = collections.Counter((x, y) for st, x, y in nascite if passo_da < st <= passo_a)
            for (x, y), n in conta.items():
                (u, v), vis = punto(cam, x, y)
                if vis:
                    r = 8 + 7 * math.sqrt(n)      # anello: non copre la cella madre
                    d.ellipse([u - r, v - r * 0.6, u + r, v + r * 0.6], outline=(255, 255, 255, 200), width=8)
                    d.ellipse([u - r, v - r * 0.6, u + r, v + r * 0.6], outline=VERDE + (255,), width=5)
            base.alpha_composite(velo)
        return np.asarray(base.convert("RGB").resize((PW, PH), Image.LANCZOS)).astype(np.float32)

    # maschera del pannello: rettangolo ad angoli arrotondati
    m = Image.new("L", (PW, PH), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, PW - 1, PH - 1], radius=28, fill=255)
    maschera = (np.asarray(m).astype(np.float32) / 255)[..., None]

    def componi(pan, passo, quota_madre, quota_fuori, n_morti):
        tela = Image.new("RGB", (W, H), "white")
        riquadro = np.asarray(tela.crop((PX, PY, PX + PW, PY + PH))).astype(np.float32)
        tela.paste(Image.fromarray((pan * maschera + riquadro * (1 - maschera)).astype(np.uint8)), (PX, PY))
        d = ImageDraw.Draw(tela)
        d.rounded_rectangle([PX - 2, PY - 2, PX + PW + 1, PY + PH + 1], radius=30, outline=RUGGINE, width=4)
        x0 = 1150
        d.text((x0, 70), f"passo {passo}", font=font(30), fill=SECONDARIO, anchor="lm")
        d.text((x0 + 170, 70), f"anno {passo * 7 / 365.25:.0f}", font=font(26), fill=GRIGIO, anchor="lm")
        d.text((x0, 195), f"{quota_madre:.0f}%", font=font(96, True), fill=INCHIOSTRO, anchor="ls")
        d.text((x0, 240), "dei coloni vive nella cella madre", font=font(32), fill=SECONDARIO, anchor="ls")
        d.text((x0, 278), "la governa il governatore", font=font(28, True), fill=INCHIOSTRO, anchor="ls")
        d.text((x0, 420), f"{quota_fuori:.0f}%" if n_morti else "–", font=font(96, True), fill=RUGGINE, anchor="ls")
        d.text((x0, 465), f"dei morti cade fuori dalla cella madre ({n_morti})", font=font(32), fill=SECONDARIO, anchor="ls")
        d.text((x0, 503), "nei distretti degli amministratori", font=font(28, True), fill=RUGGINE, anchor="ls")
        y = 640
        for tipo, testo in (("madre", "cella madre: governatore"), ("distretto", "confini dei distretti: amministratori"),
                            ("acceso", "l'amministratore riscrive la legge"), ("edificio", "edifici della colonia"),
                            ("nascita", "nascite"), ("morto", "morti")):
            if tipo == "madre":
                d.rectangle([x0, y - 14, x0 + 28, y + 14], outline=MADRE_BORDO, width=5)
            elif tipo == "distretto":
                for j, t in enumerate(TINTE[:3]):
                    d.line([(x0 + j * 10, y + 12), (x0 + j * 10 + 12, y - 12)], fill=t, width=5)
            elif tipo == "acceso":
                d.rectangle([x0, y - 14, x0 + 28, y + 14], fill=(226, 156, 125), outline=RUGGINE, width=3)
            elif tipo == "edificio":
                for j, t in enumerate(("#f5f5f4", "#60a5fa", "#4ade80")):
                    d.rectangle([x0 + j * 10, y - 6 - j * 3, x0 + j * 10 + 9, y + 12], fill=t, outline=(90, 90, 96))
            elif tipo == "nascita":
                d.ellipse([x0 + 2, y - 9, x0 + 26, y + 9], fill=(164, 214, 172), outline=VERDE, width=3)
            else:
                d.line([(x0 + 4, y - 10), (x0 + 24, y + 10)], fill=MORTO, width=4)
                d.line([(x0 + 4, y + 10), (x0 + 24, y - 10)], fill=MORTO, width=4)
            d.text((x0 + 48, y), testo, font=font(30), fill=INCHIOSTRO, anchor="lm")
            y += 58
        return tela

    def quote(snap, passo):
        pop = G.celle_occupate(snap)
        tot = sum(pop.values()) or 1
        mm = [m for m in morti if m["death_step"] <= passo]
        fuori = sum((m["x"], m["y"]) != madre for m in mm)
        return 100 * pop.get(madre, 0) / tot, (100 * fuori / len(mm) if mm else 0.0), len(mm)

    fps = args.fps
    tmp = Path(tempfile.mkdtemp(prefix="video_geografia_"))
    k = 0

    def scrivi(t):
        nonlocal k
        t.save(tmp / f"f{k:05d}.png")
        k += 1

    def livelli(snap, globale=None):
        pianeta.applica(snap)
        alb = pianeta.albedo()
        tg = globale if globale is not None else pianeta.cuoci(alb, (0, 0, 360, 180), 4)
        return tg, pianeta.cuoci(alb, zona, 24), pianeta.cuoci(alb, fine, 64)

    luce = E.luce_da_camera(camera(1.0))   # fissa: le ombre non si muovono con la camera
    # griglia dei posti fissata sull'ultima istantanea (la piu' affollata): gli
    # edifici hanno la stessa grandezza per tutto il video e non si toccano mai
    PASSO = E.passo_posti(json.loads(file[-1].read_text(encoding="utf-8")))

    def scena(snap, fine=False):
        return E.Scena(snap, luce, PASSO, fine=fine)

    if args.prova:
        ultimo = json.loads(file[-1].read_text(encoding="utf-8"))
        liv = livelli(ultimo)
        pan = pannello(ultimo, int(json.loads(file[-2].read_text(encoding="utf-8"))["step"]),
                       int(ultimo["step"]), camera(1.0), liv, scena(ultimo))
        componi(pan, int(ultimo["step"]), *quote(ultimo, int(ultimo["step"]))).save(args.prova)
        return 0

    # La camera parte vicina, sopra la cella madre, e si allarga con la colonia:
    # la distanza segue il raggio della colonia (in celle, dalla cella madre),
    # un'istantanea avanti perche' chi cresce resti in quadro.
    raggi = []
    for f in file:
        s = json.loads(f.read_text(encoding="utf-8"))
        raggi.append(max([0] + [max(abs(x - madre[0]), abs(y - madre[1])) for x, y in G.celle_occupate(s)]))
    raggi = np.maximum.accumulate(np.array(raggi, float))
    raggi = np.convolve(np.pad(raggi, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    raggi = np.maximum.accumulate(np.append(raggi[1:], raggi[-1]))
    PRESSO = (0.019, 56.0)

    def posa(r):
        d = PRESSO[0] + 0.044 * r
        e = min(1.0, math.log(d / PRESSO[0]) / math.log(VICINO[0] / PRESSO[0]))
        return d, PRESSO[1] + (VICINO[1] - PRESSO[1]) * e

    def cam_posa(p, e_rot=0.0):
        d, incl = p
        spinta = 0.12 * max(0.0, 1 - math.log(d / PRESSO[0]) / math.log(VICINO[0] / PRESSO[0]))
        return G.CameraObliqua(lat_c + 12 * e_rot + spinta, lon_c + 30 * e_rot, d, incl, int(PW * SS), int(PH * SS))

    vicina = (madre[0] - 3, madre[1] - 3, madre[0] + 4, madre[1] + 4)      # texture fitta da vicino

    def livelli_vicini(snap, globale=None, fitta=True):
        liv = livelli(snap, globale)
        return liv + ((pianeta.cuoci(pianeta.albedo(), vicina, 160), vicina, 160),) if fitta else liv

    snap = snap0
    liv = livelli_vicini(snap)
    passo = int(snap["step"])
    q = quote(snap, passo)
    sc = scena(snap, fine=True)
    for i in range(int(3.0 * fps)):          # dal globo intero fin sopra la cella madre
        e = morbido((i / fps - 0.5) / 2.5)
        p = (LONTANO[0] * (posa(raggi[0])[0] / LONTANO[0]) ** e, LONTANO[1] + (posa(raggi[0])[1] - LONTANO[1]) * e)
        tela = componi(pannello(snap, 0, passo, cam_posa(p, 1 - e), liv, sc), passo, *q)
        scrivi(tela)
    for _ in range(int(0.8 * fps)):
        scrivi(tela)
    for i in range(1, len(file)):            # la colonia cresce, la camera si allarga
        passo_prima = passo
        snap = json.loads(file[i].read_text(encoding="utf-8"))
        da, a = posa(raggi[i - 1]), posa(raggi[i])
        vicino = a[0] < 0.1
        liv = livelli_vicini(snap, globale=liv[0], fitta=vicino)
        sc = scena(snap, fine=vicino)
        passo = int(snap["step"])
        q = quote(snap, passo)
        for j in range(8):
            if j == 0 or abs(a[0] - da[0]) > 1e-6:
                e = (j + 1) / 8
                p = (da[0] * (a[0] / da[0]) ** e, da[1] + (a[1] - da[1]) * e)
                tela = componi(pannello(snap, passo_prima, passo, cam_posa(p), liv, sc), passo, *q)
            scrivi(tela)
    finale = tela
    for _ in range(int(2.5 * fps)):
        scrivi(finale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    finale.save(args.out.with_suffix(".png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(args.out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"scritto {args.out} ({k} fotogrammi, {k / fps:.1f} s); finale: madre {q[0]:.1f}%, morti fuori {q[1]:.1f}% su {q[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
