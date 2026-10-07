"""Paper figure: the largest ungoverned colony on the globe, France at the same scale.

Needs runs/paper_qwen/ctrl_none/none_seed7 with its world_static_base.json.
Usage: python scripts/paper/figura_globo.py  (writes figures/mars_globe.jpg/.png)

The colony is at the centre of the disk (camera on the mother cell), where the
globe's scale is R px per R_M km. France is projected orthographically on a
sphere of the same radius, viewed head-on at its own centre, so it carries the
same slight curvature as the planet, and is placed beside the globe at the
colony's height. A scale bar gives 500 km at that same scale.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import globo_marte as G  # noqa: E402
import figure_pianeta_presentazione as F  # noqa: E402

RUN = ROOT / "runs/paper_qwen/ctrl_none/none_seed7"
OUT = ROOT / "figures/mars_globe"
LATO, R = 2000, 960          # canvas side and disk radius of the globe render (px)
R_M = G.RAGGIO_KM            # Mars radius used by the renderer (km)
PX_PER_KM = R / R_M          # scale at the centre of the disk
EXTRA = 760                  # width added on the right for France
INK = (31, 36, 48, 255)

snaps = sorted((RUN / "world_snapshots").glob("step_*.json"))
first = json.loads(snaps[0].read_text(encoding="utf-8"))
madre = max(G.celle_occupate(first).items(), key=lambda kv: kv[1])[0]
lat_c, lon_c = map(float, G.cella_a_latlon(*madre))
zona = (madre[0] - 24, madre[1] - 24, madre[0] + 25, madre[1] + 25)

p, snap = F.carica(RUN, -1)
globo = F.rendi(p, snap, LATO, R, lat_c, lon_c, zona)

tela = Image.new("RGBA", (LATO + EXTRA, LATO), (255, 255, 255, 255))
tela.alpha_composite(globo, (0, 0))
d = ImageDraw.Draw(tela)

# France: km offsets from its centroid -> angles on a sphere of radius R_M ->
# orthographic projection seen head-on (same curvature as the globe's centre)
dati = json.loads((ROOT / "data/sagome/francia_italia.json").read_text(encoding="utf-8"))
poligoni = dati["paesi"]["Francia"]
tutti = np.array([pt for pol in poligoni for pt in pol])
lon_m, lat_m = tutti[:, 0].mean(), tutti[:, 1].mean()
cx, cy = LATO + EXTRA / 2 - 40, LATO / 2      # France centre on the canvas, at the colony's height
fx, fy = [], []
for pol in poligoni:
    a = np.array(pol)
    x_km = (a[:, 0] - lon_m) * math.cos(math.radians(lat_m)) * 111.32
    y_km = (a[:, 1] - lat_m) * 110.57
    phi = y_km / R_M
    lam = x_km / (R_M * np.cos(phi))
    X = R * np.cos(phi) * np.sin(lam)
    Y = R * np.sin(phi)
    pts = list(zip((cx + X).tolist(), (cy - Y).tolist()))
    if len(pts) > 2:
        d.polygon(pts, fill=(226, 228, 233, 255), outline=INK, width=5)
        fx += (cx + X).tolist()
        fy += (cy - Y).tolist()

def font(nomi, size):
    for n in nomi:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


f_big = font(["C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"], 66)
f_small = font(["C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf", "Arial.ttf"], 54)
d.text(((min(fx) + max(fx)) / 2, max(fy) + 26), "France", font=f_big, fill=INK, anchor="mt")
d.text(((min(fx) + max(fx)) / 2, max(fy) + 104), "same scale", font=f_small, fill=(90, 94, 104, 255), anchor="mt")

# scale bar: 500 km at the scale of the disk centre
L = 500 * PX_PER_KM
x0, y0 = (min(fx) + max(fx)) / 2 - L / 2, min(fy) - 120
d.line([(x0, y0), (x0 + L, y0)], fill=INK, width=8)
for x in (x0, x0 + L):
    d.line([(x, y0 - 18), (x, y0 + 18)], fill=INK, width=8)
d.text(((x0 + x0 + L) / 2, y0 - 30), "500 km", font=f_small, fill=INK, anchor="mb")

im = tela.convert("RGB").resize(((LATO + EXTRA) * 3 // 5, LATO * 3 // 5), Image.LANCZOS)
OUT.parent.mkdir(parents=True, exist_ok=True)
im.save(OUT.with_suffix(".jpg"), quality=90)
im.save(OUT.with_suffix(".png"), optimize=True)
print(f"France width {max(fx) - min(fx):.0f}px = {(max(fx) - min(fx)) / PX_PER_KM:.0f} km; written {OUT}.jpg/.png")
