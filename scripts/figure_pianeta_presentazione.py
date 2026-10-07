# -*- coding: utf-8 -*-
"""Immagini ferme del pianeta per la presentazione, nello stile dei video.

Stesso pianeta di `scripts/globo_marte.py` e dei due video (stessa ricetta di
superficie dell'interfaccia, stessa luce), su fondo trasparente, dalla stessa
run e dallo stesso seme (`runs/base_pulita`, seme 7):

- `pianeta_titolo.png`: il globo intero con la colonia governata al passo 1000;
- `pianeta_intero.png`: il globo intero con la colonia senza governo al passo
  1000 (175 celle, circa 608.000 km^2), e accanto la Francia metropolitana alla
  stessa scala (circa 550.000 km^2): i contorni sono di Natural Earth
  (`data/sagome/francia_italia.json`), riportati da chilometri terrestri a
  chilometri marziani e posati a est della colonia;
- `pianeta_atterraggio.png`: il sito di atterraggio da vicino al passo 25, con
  la camera obliqua: gli edifici in rilievo (altezze esagerate, convenzione di
  `globo_marte.py`) e i 300 coloni come puntini nelle loro posizioni.

Uso:
    python scripts/figure_pianeta_presentazione.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import globo_marte as G  # noqa: E402
import edifici_marte as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "Tesi_LaTex/PPT_Tesi_LaTex/media"
BASE = ROOT / "runs/base_pulita/ctrl_none/none_seed7"
GOV = ROOT / "runs/base_pulita/variante_D/llm_completo-pD+amm_seed7"


def carica(cartella: Path, indice: int):
    file = sorted((cartella / "world_snapshots").glob("step_*.json"))
    p = G.Pianeta(cartella / "world_static_base.json")
    snap = json.loads(file[indice].read_text(encoding="utf-8"))
    p.applica(snap)
    return p, snap


def rendi(p, snap, lato, R, lat0, lon0, zona, oblo=False):
    """Disco (o oblò) del pianeta su fondo trasparente, con le strutture."""
    alb = p.albedo()
    tg = p.cuoci(alb, (0, 0, 360, 180), 4)
    tz = p.cuoci(alb, zona, 24)
    img, dentro = G.disegna_globo(tg, 4, tz, zona, 24, lato, R, lat0, lon0)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    st = G.coloni_e_strutture(snap)
    if st:
        xs = np.array([s[0] for s in st])
        ys = np.array([s[1] for s in st])
        la, lo = G.cella_a_latlon(xs, ys)
        px, py, vis = G.diretta(la, lo, R, lat0, lon0, lato)
        r = max(0.8, 0.075 * R * math.pi / 180)
        for a, c, v, s in zip(px, py, vis, st):
            if v:
                d.rectangle([a - r, c - r, a + r, c + r], fill=s[2])
    # trasparenza: il disco pieno, l'alone sfuma, fuori nulla
    c = (np.arange(lato) + 0.5 - lato / 2)
    x, y = np.meshgrid(c, c)
    rho = np.sqrt(x * x + y * y)
    if oblo:
        alfa = np.clip(lato / 2 - rho, 0, 1) * 255
    else:
        rr = rho / R
        alfa = np.where(rr < 1.0, 255, np.clip((1.035 - rr) / 0.035, 0, 1) ** 2 * 200)
    im.putalpha(Image.fromarray(alfa.astype(np.uint8)))
    return im


def sagoma(im, R, lat0, lon0, lato, paese, lat_t, lon_t):
    """Contorno del paese (lon, lat terrestri) portato su Marte a pari chilometri,
    centrato in (lat_t, lon_t) marziani, e disegnato sul globo."""
    dati = json.loads((ROOT / "data/sagome/francia_italia.json").read_text(encoding="utf-8"))
    poligoni = dati["paesi"][paese]
    tutti = np.array([pt for pol in poligoni for pt in pol])
    lon_m, lat_m = tutti[:, 0].mean(), tutti[:, 1].mean()
    velo = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(velo)
    fondo_x, fondo_y = [], []
    for pol in poligoni:
        a = np.array(pol)
        x_km = (a[:, 0] - lon_m) * math.cos(math.radians(lat_m)) * 111.32
        y_km = (a[:, 1] - lat_m) * 110.57
        la = lat_t + np.degrees(y_km / G.RAGGIO_KM)
        lo = lon_t + np.degrees(x_km / (G.RAGGIO_KM * math.cos(math.radians(lat_t))))
        px, py, vis = G.diretta(la, lo, R, lat0, lon0, lato)
        if vis.all():
            pts = list(zip(px.tolist(), py.tolist()))
            d.polygon(pts, fill=(255, 255, 255, 120))
            d.line(pts + [pts[0]], fill=(31, 36, 48, 255), width=5)
            fondo_x += px.tolist()
            fondo_y += py.tolist()
    if fondo_x:
        f = ImageFont.truetype("C:/Windows/Fonts/calibrib.ttf", 64)
        cx = (min(fondo_x) + max(fondo_x)) / 2
        d.text((cx, max(fondo_y) + 18), paese, font=f, fill=(255, 255, 255, 255), anchor="mt",
               stroke_width=6, stroke_fill=(31, 36, 48, 255))
    im.alpha_composite(velo)


def vista_atterraggio(p, snap, lat_c, lon_c, madre, W=2000, H=1700):
    zona = (madre[0] - 24, madre[1] - 24, madre[0] + 25, madre[1] + 25)
    fine = (madre[0] - 3, madre[1] - 3, madre[0] + 4, madre[1] + 4)
    alb = p.albedo()
    tg, tz, tf = p.cuoci(alb, (0, 0, 360, 180), 4), p.cuoci(alb, zona, 24), p.cuoci(alb, fine, 160)
    cam = G.CameraObliqua(lat_c, lon_c, 0.028, 58, W, H)
    img, _ = G.disegna_obliquo(cam, tg, 4, tz, zona, 24, livelli_fini=[(tf, fine, 160)])
    im = Image.fromarray(img).convert("RGBA")
    E.edifici(im, cam, snap)
    # coloni: puntini nelle loro posizioni, appena sopra il suolo
    pts = [G.punto_superficie(a["x"], a["y"], (a.get("local_x_m") or 29500) / 59163 - 0.5,
                              (a.get("local_y_m") or 29500) / 59166 - 0.5, 0.15) for a in snap["agents"]]
    px, _, vis = cam.proietta(np.array(pts))
    d = ImageDraw.Draw(im)
    for (u, v), ok in zip(px, vis):
        if ok:
            d.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(74, 222, 128, 255), outline=(20, 70, 40, 255), width=2)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, W - 1, H - 1], radius=70, fill=255)
    im.putalpha(m)
    return im


def vista_colonia(p, snap, lat_c, lon_c, madre, W=2000, H=1700):
    """La colonia governata a fine run, dal drone, con gli edifici dettagliati."""
    zona = (madre[0] - 24, madre[1] - 24, madre[0] + 25, madre[1] + 25)
    fine = (madre[0] - 4, madre[1] - 4, madre[0] + 5, madre[1] + 5)
    alb = p.albedo()
    tg, tz, tf = p.cuoci(alb, (0, 0, 360, 180), 4), p.cuoci(alb, zona, 24), p.cuoci(alb, fine, 160)
    cam = G.CameraObliqua(lat_c + 0.12, lon_c, 0.019, 55, W, H)
    img, _ = G.disegna_obliquo(cam, tg, 4, tz, zona, 24, livelli_fini=[(tf, fine, 160)])
    im = Image.fromarray(img).convert("RGBA")
    E.edifici(im, cam, snap)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, W - 1, H - 1], radius=70, fill=255)
    im.putalpha(m)
    return im


def main() -> int:
    MEDIA.mkdir(parents=True, exist_ok=True)
    lato = 2000
    pg, sg = carica(GOV, -1)
    madre = max(G.celle_occupate(json.loads(sorted((GOV / "world_snapshots").glob("step_*.json"))[0]
                                             .read_text(encoding="utf-8"))).items(), key=lambda kv: kv[1])[0]
    lat_c, lon_c = map(float, G.cella_a_latlon(*madre))
    zona = (madre[0] - 24, madre[1] - 24, madre[0] + 25, madre[1] + 25)

    # titolo: globo intero, colonia governata, lievemente inclinato
    rendi(pg, sg, lato, 960, lat_c + 10, lon_c + 16, zona).resize((1000, 1000), Image.LANCZOS) \
        .save(MEDIA / "pianeta_titolo.png")
    # il mondo: globo intero, colonia senza governo al passo 1000
    pb, sb = carica(BASE, -1)
    intero = rendi(pb, sb, lato, 960, lat_c, lon_c, zona)
    sagoma(intero, 960, lat_c, lon_c, lato, "Francia", lat_c + 4.0, lon_c + 27.0)
    intero.resize((1000, 1000), Image.LANCZOS).save(MEDIA / "pianeta_intero.png")
    # atterraggio: da vicino, passo 25
    p0, s0 = carica(GOV, 0)
    vista_atterraggio(p0, s0, lat_c, lon_c, madre).resize((1000, 850), Image.LANCZOS) \
        .save(MEDIA / "pianeta_atterraggio.png")
    # chiusura: la colonia governata al passo 1000, dal drone
    pg2, sg2 = carica(GOV, -1)
    finale = vista_colonia(pg2, sg2, lat_c, lon_c, madre)
    finale.resize((1000, 850), Image.LANCZOS).save(MEDIA / "colonia_finale_drone.png")
    print("scritte pianeta_titolo.png, pianeta_intero.png, pianeta_atterraggio.png, colonia_finale_drone.png in", MEDIA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
