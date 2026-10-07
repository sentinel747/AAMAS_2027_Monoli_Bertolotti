# -*- coding: utf-8 -*-
"""Il globo di Marte della simulazione, disegnato in Python per i video.

**Che cosa e' vero e che cosa e' convenzione.** Terreno, quota, ghiaccio,
polvere, abitabilita', vegetazione, esplorazione, strutture e coloni vengono dal
mondo statico della run (`world_static_base.json`) e dalle sue istantanee
(`world_snapshots/`), senza ritocchi. La superficie ripete la ricetta
dell'interfaccia (`frontend/src/rendering/marsSurface.ts`): stesse tinte del
terreno, stesso rumore frattale per le regioni scure e per la grana del suolo,
stesso ombreggiamento del rilievo, stessi colori delle strutture. Le differenze
sono due, entrambe di resa: la superficie e' quella "realistica" dell'interfaccia
(celle sfumate) e non quella a quadrati, perche' in un video la griglia distrae
dal pianeta; e il rumore e' espresso in unita' di cella invece che di pixel di
texture, cosi' che la texture globale e quella ad alta risoluzione intorno alla
colonia combacino senza cucitura. La luce, l'alone e l'esagerazione del rilievo
sono convenzioni di disegno.

Proiezione ortografica: un punto (lat, lon) e' visto come da lontano, centrato
su (lat0, lon0), con raggio del disco R pixel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# --- Palette dell'interfaccia (marsSurface.ts) -------------------------------------
TERRAIN_ALBEDO = {
    "regolith_plain": (178, 99, 62), "crater": (110, 58, 36), "mountain": (156, 106, 72),
    "canyon": (124, 65, 40), "ice_deposit": (239, 234, 228), "mineral_rich_area": (169, 127, 74),
    "dust_field": (208, 138, 90), "lava_tube": (58, 35, 28), "frozen_basin": (185, 169, 155),
}
FALLBACK = (170, 94, 58)
BASALT_DARK = np.array((95, 51, 34), float)
DUST_BRIGHT = np.array((217, 148, 95), float)
ICE_WHITE = np.array((244, 246, 248), float)
VEGETATION_DEEP = np.array((47, 107, 58), float)
HABITABLE_TINT = np.array((96, 148, 112), float)
ICE_FULL_COVER = 8.0
STRUTTURE = {"greenhouse": "#4ade80", "habitat": "#f5f5f4", "solar_array": "#60a5fa",
             "oxygen_plant": "#7dd3fc", "weather_station": "#fbbf24", "storage_depot": "#c084fc"}
STRUTTURA_ALTRA = "#9ca3af"

#: Scala del rumore: l'interfaccia lo calcola sui pixel di una texture a 4 pixel
#: per cella (nx = px * 0.11); qui lo stesso rumore e' espresso in celle.
RUMORE_PER_CELLA = 4 * 0.11


# --- Rumore deterministico, come hash2 / valueNoise2 / fbm2 dell'interfaccia --------
def _i32(v):
    return ((v + 2**31) % 2**32) - 2**31


def _hash2(x, y):
    h = _i32(x.astype(np.int64) * 374761393 + y.astype(np.int64) * 668265263)
    h = h ^ (h >> 13)
    h = _i32(h * 1274126177)
    h = h ^ (h >> 16)
    return (h & 0xFFFFFFFF).astype(np.float64) / 4294967295.0


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _value_noise(x, y):
    x0 = np.floor(x)
    y0 = np.floor(y)
    tx = _smooth(x - x0)
    ty = _smooth(y - y0)
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    a, b = _hash2(x0, y0), _hash2(x0 + 1, y0)
    c, d = _hash2(x0, y0 + 1), _hash2(x0 + 1, y0 + 1)
    top = a + (b - a) * tx
    bot = c + (d - c) * tx
    return top + (bot - top) * ty


def fbm2(x, y, ottave):
    tot, amp, freq, norm = 0.0, 0.5, 1.0, 0.0
    for _ in range(ottave):
        tot = tot + _value_noise(x * freq, y * freq) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.13
    return tot / max(1e-6, norm)


# --- Il pianeta ---------------------------------------------------------------------
class Pianeta:
    """Stato statico del mondo e cottura delle texture di superficie."""

    def __init__(self, statico: Path):
        w = json.loads(Path(statico).read_text(encoding="utf-8"))
        self.W, self.H = int(w["width"]), int(w["height"])
        n = self.W * self.H
        self.terreno = np.zeros((n, 3))
        self.quota = np.zeros(n)
        self.polare = np.zeros(n)
        self.campi = {k: np.zeros(n) for k in ("dust", "water_ice", "habitability", "vegetation", "explored")}
        for c in w["cells"]:
            i = c["y"] * self.W + c["x"]
            self.terreno[i] = TERRAIN_ALBEDO.get(c.get("terrain"), FALLBACK)
            self.quota[i] = c.get("elevation") or 0.0
            self.polare[i] = c.get("polar_severity") or 0.0
            self._scrivi(i, c)
        xs, ys = np.meshgrid(np.arange(self.W), np.arange(self.H))
        regione = fbm2(xs.ravel() * 0.035, ys.ravel() * 0.035, 3)
        self.regione = np.clip(regione - 0.42, 0, 1) * 0.55
        self._fattori = {}

    def _scrivi(self, i, c):
        self.campi["dust"][i] = c.get("dust") or 0.0
        self.campi["water_ice"][i] = c.get("water_ice") or 0.0
        self.campi["habitability"][i] = c.get("habitability") or 0.0
        self.campi["vegetation"][i] = c.get("vegetation_biomass") or 0.0
        self.campi["explored"][i] = 1.0 if c.get("explored") else 0.0

    def applica(self, snap: dict) -> None:
        """Porta sul pianeta lo stato dinamico delle celle di un'istantanea."""
        for c in snap["cells"]:
            self._scrivi(c["y"] * self.W + c["x"], c)

    def albedo(self) -> np.ndarray:
        """Albedo per cella, (H, W, 3), come cellAlbedo con tutti i livelli accesi."""
        f = self.campi

        def mix(a, b, t):
            t = np.clip(t, 0, 1)[:, None]
            return a + (b - a) * t

        col = self.terreno.copy()
        col = mix(col, BASALT_DARK, self.regione)
        col = mix(col, DUST_BRIGHT, np.clip(f["dust"], 0, 1) * 0.30)
        # Il ghiaccio del suolo non schiarisce la superficie, solo le calotte polari
        # la schiariscono. Nel replay da cui viene la figura del globo in tesi il
        # campo water_ice arriva soltanto per le celle delle istantanee e altrove
        # vale zero: il pianeta appare rosso con le calotte bianche. Applicarlo qui
        # ovunque (mediana 6,1 su 8 di copertura piena) sbiancherebbe tre celle su
        # quattro e il video non somiglierebbe piu' al pianeta della tesi.
        col = mix(col, ICE_WHITE, np.clip(self.polare, 0, 1) * 0.9)
        col = mix(col, HABITABLE_TINT, np.clip(f["habitability"], 0, 1) * 0.45)
        col = mix(col, VEGETATION_DEEP, np.clip(f["vegetation"] / 1.5, 0, 1) * 0.85)
        e = f["explored"][:, None] > 0
        col = np.where(e, np.minimum(255, col * np.array([1.04, 1.07, 1.10]) + np.array([4, 6, 10])), col)
        return col.reshape(self.H, self.W, 3)

    def _coordinate(self, finestra, scala):
        x0, y0, x1, y1 = finestra
        w = (x1 - x0) * scala
        h = (y1 - y0) * scala
        gx = x0 + (np.arange(w) + 0.5) / scala - 0.5
        gy = y0 + (np.arange(h) + 0.5) / scala - 0.5
        return np.meshgrid(gx, gy)

    def fattori(self, finestra, scala):
        """Rilievo ombreggiato + grana (reliefFactors), statici: calcolati una volta."""
        chiave = (finestra, scala)
        if chiave in self._fattori:
            return self._fattori[chiave]
        gx, gy = self._coordinate(finestra, scala)
        nx, ny = gx * RUMORE_PER_CELLA, gy * RUMORE_PER_CELLA

        quota = self.quota.reshape(self.H, self.W)

        def altezza(ax, ay, bx, by):
            # L'interfaccia legge la quota della cella piu' vicina: a 4 pixel per
            # cella non si vede, ingrandita da' un'ombra a gradini dentro la cella.
            # Qui la quota e' interpolata fra i centri, e l'ombra segue il pendio.
            xf, yf = np.floor(ax), np.floor(ay)
            tx, ty = _smooth(ax - xf), _smooth(ay - yf)
            x0 = xf.astype(int) % self.W
            x1 = (x0 + 1) % self.W
            y0 = np.clip(yf.astype(int), 0, self.H - 1)
            y1 = np.clip(y0 + 1, 0, self.H - 1)
            top = quota[y0, x0] + (quota[y0, x1] - quota[y0, x0]) * tx
            bot = quota[y1, x0] + (quota[y1, x1] - quota[y1, x0]) * tx
            return fbm2(bx, by, 3) * 0.55 + (top + (bot - top) * ty) * 0.10

        qui = altezza(gx, gy, nx, ny)
        dx = altezza(gx + 0.6, gy, nx + 0.9, ny)
        dy = altezza(gx, gy + 0.6, nx, ny + 0.9)
        ombra = np.clip(1 + (dx - qui) * -0.62 + (dy - qui) * -0.55, 0.78, 1.22)
        grana = 0.94 + fbm2(nx * 0.5, ny * 0.5, 2) * 0.12
        # Dettaglio sotto la cella, che si legge solo da vicino (convenzione di
        # resa: l'interfaccia non scende mai sotto la cella). Un micro-rilievo
        # ombreggiato con la stessa luce, e una variazione di tinta a media scala.
        h = fbm2(nx * 6.0, ny * 6.0, 3)
        hx = fbm2(nx * 6.0 + 0.35, ny * 6.0, 3)
        hy = fbm2(nx * 6.0, ny * 6.0 + 0.35, 3)
        micro = np.clip(1 + (hx - h) * -1.5 + (hy - h) * -1.3, 0.74, 1.26)
        media = 0.91 + fbm2(nx * 2.2 + 17.0, ny * 2.2 + 5.0, 3) * 0.18
        f = ombra * grana * micro * media
        if scala >= 64:
            # solo nelle texture ravvicinate: sassi e grana del regolite
            hf = fbm2(nx * 34.0, ny * 34.0, 3)
            hfx = fbm2(nx * 34.0 + 0.3, ny * 34.0, 3)
            f = f * np.clip(1 + (hfx - hf) * -1.1, 0.8, 1.2) * (0.94 + fbm2(nx * 90.0, ny * 90.0, 2) * 0.12)
        f = f.astype(np.float32)
        self._fattori[chiave] = f
        return f

    def cuoci(self, albedo, finestra, scala) -> np.ndarray:
        """Texture equirettangolare della finestra (celle x0..x1, y0..y1), uint8."""
        gx, gy = self._coordinate(finestra, scala)
        xf = np.floor(gx)
        yf = np.floor(gy)
        tx = _smooth(gx - xf)
        ty = _smooth(gy - yf)
        x0 = xf.astype(int) % self.W
        x1 = (x0 + 1) % self.W
        y0 = np.clip(yf.astype(int), 0, self.H - 1)
        y1 = np.clip(y0 + 1, 0, self.H - 1)
        a, b = albedo[y0, x0], albedo[y0, x1]
        c, d = albedo[y1, x0], albedo[y1, x1]
        tx, ty = tx[..., None], ty[..., None]
        col = (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty
        col = col * self.fattori(finestra, scala)[..., None]
        return np.clip(col, 0, 255).astype(np.uint8)


# --- Proiezione -----------------------------------------------------------------------
def cella_a_latlon(x, y):
    """Coordinate di griglia (anche frazionarie, centro cella = intero) -> gradi."""
    return 89.5 - np.asarray(y, float), -179.5 + np.asarray(x, float)


def latlon_a_cella(lat, lon):
    return (np.asarray(lon) + 179.5) % 360.0, 89.5 - np.asarray(lat)


def inversa(lato, R, lat0, lon0):
    """Per ogni pixel di un riquadro lato x lato centrato: (dentro, lat, lon, z)."""
    c = (np.arange(lato) + 0.5 - lato / 2) / R
    x, y = np.meshgrid(c, -c)
    rho2 = x * x + y * y
    dentro = rho2 < 1.0
    z = np.sqrt(np.clip(1 - rho2, 0, 1))
    p0, l0 = math.radians(lat0), math.radians(lon0)
    lat = np.degrees(np.arcsin(np.clip(z * math.sin(p0) + y * math.cos(p0), -1, 1)))
    lon = lon0 + np.degrees(np.arctan2(x, z * math.cos(p0) - y * math.sin(p0)))
    return dentro, lat, lon, x, y, z


def diretta(lat, lon, R, lat0, lon0, lato):
    """(lat, lon) -> pixel del riquadro; visibile se sull'emisfero verso la camera."""
    p, l = np.radians(lat), np.radians(lon)
    p0, l0 = math.radians(lat0), math.radians(lon0)
    cosc = math.sin(p0) * np.sin(p) + math.cos(p0) * np.cos(p) * np.cos(l - l0)
    x = np.cos(p) * np.sin(l - l0)
    y = math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(l - l0)
    return lato / 2 + R * x, lato / 2 - R * y, cosc > 0.02


LUCE = np.array([-0.45, 0.40, 0.80])
LUCE = LUCE / np.linalg.norm(LUCE)


def disegna_globo(tex_globale, scala_g, tex_zona, finestra_zona, scala_z, lato, R, lat0, lon0,
                  fondo=(255, 255, 255)):
    """Il disco del pianeta visto da (lat0, lon0), raggio R, su fondo `fondo`."""
    dentro, lat, lon, x, y, z = inversa(lato, R, lat0, lon0)
    gx, gy = latlon_a_cella(lat, lon)
    img = np.empty((lato, lato, 3), np.float32)
    # texture globale
    Hg, Wg = tex_globale.shape[:2]
    u = np.clip(((gx + 0.5) * scala_g).astype(int), 0, Wg - 1)
    v = np.clip(((gy + 0.5) * scala_g).astype(int), 0, Hg - 1)
    img[:] = tex_globale[v, u]
    # texture ad alta risoluzione dove la finestra copre
    if tex_zona is not None:
        x0, y0, x1, y1 = finestra_zona
        gxz = gx.copy()
        gxz = np.where(gxz < x0 - 180, gxz + 360, np.where(gxz > x1 + 180, gxz - 360, gxz))
        uz = (gxz - x0 + 0.5) * scala_z
        vz = (gy - y0 + 0.5) * scala_z
        Hz, Wz = tex_zona.shape[:2]
        mz = (uz >= 0) & (uz < Wz - 1) & (vz >= 0) & (vz < Hz - 1) & dentro
        ui, vi = uz[mz].astype(int), vz[mz].astype(int)
        img[mz] = tex_zona[vi, ui]
    # luce: diffusa + ambiente, lieve oscuramento al bordo
    nl = np.clip(x * LUCE[0] + y * LUCE[1] + z * LUCE[2], 0, 1)
    luce = 0.50 + 0.62 * nl
    img *= luce[..., None]
    out = np.empty_like(img)
    out[:] = fondo
    out[dentro] = img[dentro]
    # alone dell'atmosfera appena fuori dal bordo
    rho = np.sqrt(x * x + y * y)
    anello = (rho >= 1.0) & (rho < 1.035)
    t = (1 - (rho[anello] - 1.0) / 0.035)[:, None] ** 2
    out[anello] = out[anello] * (1 - 0.55 * t) + np.array([224, 150, 110]) * 0.55 * t
    return np.clip(out, 0, 255).astype(np.uint8), dentro


def coloni_e_strutture(snap):
    """Posizioni frazionarie in coordinate di cella delle strutture (con colore)."""
    strutture = []
    for c in snap["cells"]:
        geo = c.get("geometry") or {}
        wm = geo.get("width_m") or 59000.0
        hm = geo.get("height_m") or 59000.0
        for s in c.get("structures") or []:
            fx = (s.get("local_x_m") or wm / 2) / wm - 0.5
            fy = (s.get("local_y_m") or hm / 2) / hm - 0.5
            strutture.append((c["x"] + fx, c["y"] + fy, STRUTTURE.get(s.get("type"), STRUTTURA_ALTRA)))
    return strutture


def celle_occupate(snap):
    """(x, y) -> coloni, per le celle con coloni o opere (come il conteggio della tesi)."""
    out = {}
    for c in snap["cells"]:
        n = len(c.get("agents_present") or [])
        if n or c.get("structures"):
            out[(c["x"], c["y"])] = n
    return out


# --- Camera obliqua ("da drone") e strutture in rilievo ------------------------------
#
# Convenzioni di disegno, dichiarate: le strutture sono parallelepipedi di 1,6 km
# di lato (i pannelli solari piu' larghi e bassi), con altezze per tipo molto
# esagerate rispetto al vero, perche' a scala reale un habitat alto dieci metri
# su una cella di 59 km non si vedrebbe. Posizione dentro la cella, tipo e
# colore vengono dall'istantanea e dall'interfaccia.
RAGGIO_KM = 3390.0
ALTEZZE_KM = {"habitat": 2.2, "greenhouse": 1.3, "solar_array": 0.35, "oxygen_plant": 1.8,
              "weather_station": 3.0, "storage_depot": 1.1, "shelter": 1.0, "infirmary": 1.6,
              "water_extractor": 1.4, "heater": 1.2, "laboratory": 1.9}
LATO_KM = {"solar_array": 2.4, "weather_station": 0.9}


def _vettore(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)], axis=-1)


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


class CameraObliqua:
    """Camera prospettica sopra il punto (lat, lon), inclinata di `incl` gradi
    dalla verticale, a `distanza` raggi planetari dal suolo; l'azimut 180 mette
    la camera a sud del bersaglio, rivolta a nord."""

    def __init__(self, lat, lon, distanza, incl, W, H, fov=38.0, azimut=180.0):
        T = _vettore(lat, lon)
        l, p = math.radians(lon), math.radians(lat)
        E = np.array([-math.sin(l), math.cos(l), 0.0])
        N = np.array([-math.sin(p) * math.cos(l), -math.sin(p) * math.sin(l), math.cos(p)])
        a, i = math.radians(azimut), math.radians(incl)
        dietro = math.cos(a) * N + math.sin(a) * E
        self.C = T * (1.0) + distanza * (math.cos(i) * T + math.sin(i) * dietro)
        f = T - self.C
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, T)
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.W, self.H = W, H
        self.th = math.tan(math.radians(fov) / 2)
        self.asp = W / H

    def raggi(self):
        x = ((np.arange(self.W) + 0.5) / self.W * 2 - 1) * self.th * self.asp
        y = (1 - (np.arange(self.H) + 0.5) / self.H * 2) * self.th
        X, Y = np.meshgrid(x, y)
        d = self.f[None, None, :] + X[..., None] * self.r + Y[..., None] * self.u
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def proietta(self, P):
        """Punti 3D (n, 3) -> pixel (n, 2), profondita' (n,), visibili (n,)."""
        v = P - self.C
        z = v @ self.f
        x = (v @ self.r) / (np.maximum(z, 1e-9) * self.th * self.asp)
        y = (v @ self.u) / (np.maximum(z, 1e-9) * self.th)
        px = (x + 1) / 2 * self.W
        py = (1 - y) / 2 * self.H
        n = P / np.linalg.norm(P, axis=-1, keepdims=True)
        verso = np.einsum("ij,ij->i", n, self.C - P) > 0
        return np.stack([px, py], axis=-1), z, (z > 0) & verso


def _bilineare(tex, u, v):
    H, W = tex.shape[:2]
    u = np.clip(u - 0.5, 0, W - 1.001)
    v = np.clip(v - 0.5, 0, H - 1.001)
    x0, y0 = u.astype(int), v.astype(int)
    tx, ty = (u - x0)[..., None], (v - y0)[..., None]
    a, b = tex[y0, x0].astype(np.float32), tex[y0, x0 + 1].astype(np.float32)
    c, d = tex[y0 + 1, x0].astype(np.float32), tex[y0 + 1, x0 + 1].astype(np.float32)
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty


def disegna_obliquo(cam: CameraObliqua, tex_g, scala_g, tex_z, zona, scala_z, fondo=(255, 255, 255),
                    livelli_fini=()):
    """Il pianeta visto dalla camera obliqua: suolo, foschia in lontananza, alone
    all'orizzonte, cielo del colore del fondo."""
    d = cam.raggi()
    C = cam.C
    b = d @ C
    c = C @ C - 1.0
    disc = b * b - c
    colpito = disc > 0
    t = np.where(colpito, -b - np.sqrt(np.clip(disc, 0, None)), 0)
    colpito &= t > 0
    P = C + t[..., None] * d
    lat = np.degrees(np.arcsin(np.clip(P[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(P[..., 1], P[..., 0]))
    gx, gy = latlon_a_cella(lat, lon)
    Hg, Wg = tex_g.shape[:2]
    img = tex_g[np.clip(((gy + 0.5) * scala_g).astype(int), 0, Hg - 1),
                np.clip(((gx + 0.5) * scala_g).astype(int), 0, Wg - 1)].astype(np.float32)
    if tex_z is not None:
        x0, y0, x1, y1 = zona
        gxz = np.where(gx < x0 - 180, gx + 360, np.where(gx > x1 + 180, gx - 360, gx))
        uz = (gxz - x0 + 0.5) * scala_z
        vz = (gy - y0 + 0.5) * scala_z
        Hz, Wz = tex_z.shape[:2]
        mz = (uz >= 0) & (uz < Wz - 1) & (vz >= 0) & (vz < Hz - 1) & colpito
        img[mz] = _bilineare(tex_z, uz[mz], vz[mz])
    for tex_f, (fx0, fy0, fx1, fy1), scala_f in livelli_fini:   # dal piu' largo al piu' fine
        uf = (gx - fx0 + 0.5) * scala_f
        vf = (gy - fy0 + 0.5) * scala_f
        Hf, Wf = tex_f.shape[:2]
        mf = (uf >= 1) & (uf < Wf - 2) & (vf >= 1) & (vf < Hf - 2) & colpito
        img[mf] = _bilineare(tex_f, uf[mf], vf[mf])
    L = -0.35 * cam.r + 0.55 * cam.u - 0.45 * cam.f
    L = L / np.linalg.norm(L)
    nl = np.clip(P @ L, 0, 1)
    img *= (0.62 + 0.48 * nl)[..., None]
    # foschia con la distanza, verso un rosa polveroso
    nebbia = np.clip(1 - np.exp(-(t - t[colpito].min() if colpito.any() else 0) / 0.35), 0, 0.55)[..., None]
    img = img * (1 - nebbia) + np.array([226, 176, 150]) * nebbia
    out = np.empty_like(img)
    out[:] = fondo
    out[colpito] = img[colpito]
    # alone all'orizzonte: dove il raggio passa appena sopra il suolo
    quota = np.sqrt(np.clip(c - b * b + 1.0, 0, None)) - 1.0      # distanza minima dal centro - 1
    alone = (~colpito) & (b < 0)
    a = np.clip(np.exp(-quota / 0.012), 0, 1)[..., None] * 0.75
    out = np.where(alone[..., None], out * (1 - a) + np.array([222, 150, 112]) * a, out)
    return np.clip(out, 0, 255).astype(np.uint8), colpito


def punto_superficie(x, y, fx=0.0, fy=0.0, quota_km=0.0):
    """Cella (x, y) piu' offset frazionario -> punto 3D, a quota_km dal suolo."""
    lat, lon = cella_a_latlon(x + fx, y + fy)
    return _vettore(lat, lon) * (1 + quota_km / RAGGIO_KM)


def strutture_3d(im: Image.Image, cam: CameraObliqua, snap: dict, scala: float = 1.0):
    """Le strutture dell'istantanea come parallelepipedi, dal piu' lontano.
    `scala` ingrandisce pianta e altezza (per le viste da lontano)."""
    d = ImageDraw.Draw(im)
    scatole = []
    for c in snap["cells"]:
        geo = c.get("geometry") or {}
        wm = geo.get("width_m") or 59000.0
        hm = geo.get("height_m") or 59000.0
        for s in c.get("structures") or []:
            fx = (s.get("local_x_m") or wm / 2) / wm - 0.5
            fy = (s.get("local_y_m") or hm / 2) / hm - 0.5
            tipo = s.get("type")
            scatole.append((c["x"] + fx, c["y"] + fy, tipo, wm, hm))
    if not scatole:
        return
    centri = np.array([punto_superficie(x, y) for x, y, *_ in scatole])
    _, prof, vis = cam.proietta(centri)
    L = -0.35 * cam.r + 0.55 * cam.u - 0.45 * cam.f
    L = L / np.linalg.norm(L)
    for k in np.argsort(-prof):
        if not vis[k]:
            continue
        x, y, tipo, wm, hm = scatole[k]
        lato = LATO_KM.get(tipo, 1.6) * scala
        h = ALTEZZE_KM.get(tipo, 1.0) * scala
        dx, dy = lato * 1000 / wm / 2, lato * 1000 / hm / 2
        base = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        giu = np.array([punto_superficie(x, y, a, b) for a, b in base])
        su = np.array([punto_superficie(x, y, a, b, h) for a, b in base])
        pg, _, _ = cam.proietta(giu)
        ps, _, _ = cam.proietta(su)
        col = np.array(_hex(STRUTTURE.get(tipo, STRUTTURA_ALTRA)), float)
        n0 = centri[k] / np.linalg.norm(centri[k])
        facce = []
        for i in range(4):
            j = (i + 1) % 4
            mezzo = (giu[i] + giu[j]) / 2
            fuori = mezzo - centri[k]
            fuori = fuori / (np.linalg.norm(fuori) + 1e-12)
            verso_cam = cam.C - mezzo
            if fuori @ verso_cam > 0:
                luce = 0.55 + 0.45 * max(0.0, float(fuori @ L))
                facce.append(([tuple(pg[i]), tuple(pg[j]), tuple(ps[j]), tuple(ps[i])], luce))
        for poli, luce in facce:
            d.polygon(poli, fill=tuple(int(v) for v in np.clip(col * luce * 0.82, 0, 255)))
        luce = 0.75 + 0.35 * max(0.0, float(n0 @ L))
        grande = np.ptp(ps[:, 0]) > 7
        d.polygon([tuple(p) for p in ps], fill=tuple(int(v) for v in np.clip(col * luce, 0, 255)),
                  outline=(60, 58, 60) if grande else None)
