# -*- coding: utf-8 -*-
"""Edifici dettagliati della colonia, per le viste oblique di `globo_marte.py`.

Un modello per tipo, fatto di primitive (cupole, cilindri, tunnel, scatole,
tetti, pannelli), con l'ombra portata sul suolo:

- rifugio (shelter): cupola gonfiabile;
- habitat: cilindro bianco con una fascia di finestre e tetto a cupola;
- serra: tunnel di vetro verde con le costole degli archi;
- pannello solare: pannello blu inclinato su due sostegni, con la griglia;
- estrattore d'acqua: serbatoio con tetto conico;
- impianto a ossigeno: basamento e due serbatoi azzurri;
- deposito: capannone con tetto a capanna viola;
- stazione meteo: antenna con lo strumento giallo in cima;
- riscaldatore: blocco scuro con la piastra arancione;
- infermeria: edificio bianco con la croce rossa sul tetto;
- laboratorio: edificio con cupola turchese.

Posizione nella cella e tipo vengono dall'istantanea; forme, misure (in km,
molto esagerate: un habitat vero alto dieci metri su una cella di 59 km non si
vedrebbe) e altezze sono convenzioni di disegno. I colori sono quelli
dell'interfaccia dove li ha (serre verdi, pannelli blu, ossigeno azzurro, meteo
giallo, depositi viola, habitat bianchi).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

import globo_marte as G


def _base_locale(P):
    n = P / np.linalg.norm(P)
    e = np.array([-P[1], P[0], 0.0])
    e = e / (np.linalg.norm(e) + 1e-12)
    m = np.cross(n, e)
    return n, e, m


class _Modello:
    """Raccoglie le facce di un edificio in coordinate locali (km)."""

    def __init__(self, P0, k, fine=True):
        self.P0, self.k, self.fine = P0, k, fine
        self.n, self.e, self.m = _base_locale(P0)
        self.facce = []          # (punti 3D, colore, normale, alfa, linee)
        self.impronta = []       # punti (a, b) della pianta, per l'ombra
        self.altezza = 0.0

    def p(self, a, b, h):
        return self.P0 + (a * self.e + b * self.m + h * self.n) * self.k

    def faccia(self, pts, colore, normale, alfa=255, linee=None):
        P = np.array([self.p(*q) for q in pts])
        nl = np.asarray(normale, float)
        N = nl[0] * self.e + nl[1] * self.m + nl[2] * self.n
        self.facce.append((P, colore, N / (np.linalg.norm(N) + 1e-12), alfa,
                           [np.array([self.p(*q) for q in L]) for L in (linee or [])]))

    def scatola(self, a, b, dx, dy, h0, h1, colore):
        c = [(a - dx, b - dy), (a + dx, b - dy), (a + dx, b + dy), (a - dx, b + dy)]
        normali = [(0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]
        for i in range(4):
            j = (i + 1) % 4
            self.faccia([(*c[i], h0), (*c[j], h0), (*c[j], h1), (*c[i], h1)], colore, normali[i])
        self.faccia([(*q, h1) for q in c], colore, (0, 0, 1))
        self.impronta += c
        self.altezza = max(self.altezza, h1)

    def cilindro(self, a, b, r, h0, h1, colore, lati=12, tetto=True):
        lati = lati if self.fine else 6
        ang = np.linspace(0, 2 * np.pi, lati + 1)
        for i in range(lati):
            a0, a1 = ang[i], ang[i + 1]
            q0 = (a + r * math.cos(a0), b + r * math.sin(a0))
            q1 = (a + r * math.cos(a1), b + r * math.sin(a1))
            mid = (a0 + a1) / 2
            self.faccia([(*q0, h0), (*q1, h0), (*q1, h1), (*q0, h1)], colore, (math.cos(mid), math.sin(mid), 0))
        if tetto:
            self.faccia([(a + r * math.cos(t), b + r * math.sin(t), h1) for t in ang[:-1]], colore, (0, 0, 1))
        self.impronta += [(a + r * math.cos(t), b + r * math.sin(t)) for t in ang[:-1]]
        self.altezza = max(self.altezza, h1)

    def cupola(self, a, b, r, h0, hr, colore, bande=4, lati=14, alfa=255):
        if not self.fine:
            bande, lati = 2, 7
        phi = np.linspace(0, np.pi / 2, bande + 1)
        ang = np.linspace(0, 2 * np.pi, lati + 1)
        for i in range(bande):
            p0, p1 = phi[i], phi[i + 1]
            for j in range(lati):
                t0, t1 = ang[j], ang[j + 1]
                pts = [(a + r * math.cos(p0) * math.cos(t0), b + r * math.cos(p0) * math.sin(t0), h0 + hr * math.sin(p0)),
                       (a + r * math.cos(p0) * math.cos(t1), b + r * math.cos(p0) * math.sin(t1), h0 + hr * math.sin(p0)),
                       (a + r * math.cos(p1) * math.cos(t1), b + r * math.cos(p1) * math.sin(t1), h0 + hr * math.sin(p1)),
                       (a + r * math.cos(p1) * math.cos(t0), b + r * math.cos(p1) * math.sin(t0), h0 + hr * math.sin(p1))]
                pm, tm = (p0 + p1) / 2, (t0 + t1) / 2
                self.faccia(pts, colore, (math.cos(pm) * math.cos(tm) / r, math.cos(pm) * math.sin(tm) / r,
                                          math.sin(pm) / max(hr, 1e-6)), alfa)
        self.impronta += [(a + r * math.cos(t), b + r * math.sin(t)) for t in ang[:-1]]
        self.altezza = max(self.altezza, h0 + hr)

    def tunnel(self, a, b, lung, r, colore, archi=6, alfa=225):
        archi = archi if self.fine else 3
        th = np.linspace(0, np.pi, archi + 1)
        x0, x1 = a - lung / 2, a + lung / 2
        for i in range(archi):
            t0, t1 = th[i], th[i + 1]
            q = [(x0, b + r * math.cos(t0), r * math.sin(t0)), (x1, b + r * math.cos(t0), r * math.sin(t0)),
                 (x1, b + r * math.cos(t1), r * math.sin(t1)), (x0, b + r * math.cos(t1), r * math.sin(t1))]
            tm = (t0 + t1) / 2
            costole = [[(x, b + r * math.cos(t0), r * math.sin(t0) + 0.01), (x, b + r * math.cos(t1), r * math.sin(t1) + 0.01)]
                       for x in np.linspace(x0, x1, 5)]
            self.faccia(q, colore, (0, math.cos(tm), math.sin(tm)), alfa, costole)
        for xe, s in ((x0, -1), (x1, 1)):
            self.faccia([(xe, b + r * math.cos(t), r * math.sin(t)) for t in th], colore, (s, 0, 0), alfa)
        self.impronta += [(x0, b - r), (x1, b - r), (x1, b + r), (x0, b + r)]
        self.altezza = max(self.altezza, r)

    def tetto(self, a, b, dx, dy, h0, hr, colore):
        c0, c1 = (a - dx, b, h0 + hr), (a + dx, b, h0 + hr)
        self.faccia([(a - dx, b - dy, h0), (a + dx, b - dy, h0), c1, c0], colore, (0, -hr, dy))
        self.faccia([(a + dx, b + dy, h0), (a - dx, b + dy, h0), c0, c1], colore, (0, hr, dy))
        self.altezza = max(self.altezza, h0 + hr)

    def pannello(self, a, b, w, d, h, incl, colore):
        s, c = math.sin(math.radians(incl)), math.cos(math.radians(incl))
        q = [(a - w / 2, b - d / 2 * c, h - d / 2 * s), (a + w / 2, b - d / 2 * c, h - d / 2 * s),
             (a + w / 2, b + d / 2 * c, h + d / 2 * s), (a - w / 2, b + d / 2 * c, h + d / 2 * s)]
        linee = [[(a - w / 2 + w * f, b - d / 2 * c, h - d / 2 * s), (a - w / 2 + w * f, b + d / 2 * c, h + d / 2 * s)]
                 for f in (0.25, 0.5, 0.75)]
        linee.append([(a - w / 2, b, h), (a + w / 2, b, h)])
        for x in (a - w / 3, a + w / 3):
            self.faccia([(x - 0.03, b, 0), (x + 0.03, b, 0), (x + 0.03, b, h), (x - 0.03, b, h)], (90, 90, 96), (0, -1, 0))
        self.faccia(q, colore, (0, -s, c), 255, linee)
        self.impronta += [(p[0], p[1]) for p in q]
        self.altezza = max(self.altezza, h + d / 2 * s)


def _modello(tipo, M):
    bianco, grigio = (238, 236, 232), (168, 170, 176)
    if tipo == "shelter":
        M.cupola(0, 0, 0.55, 0, 0.42, (214, 210, 204))
    elif tipo == "habitat":
        M.cilindro(0, 0, 0.75, 0, 1.0, bianco)
        M.cilindro(0, 0, 0.765, 0.55, 0.72, (62, 74, 92), tetto=False)
        M.cupola(0, 0, 0.75, 1.0, 0.32, bianco)
    elif tipo == "greenhouse":
        M.tunnel(0, 0, 2.0, 0.55, (110, 214, 146))
    elif tipo == "solar_array":
        M.pannello(0, 0, 2.2, 1.1, 0.45, 28, (30, 64, 150))
    elif tipo == "water_extractor":
        M.cilindro(0, 0, 0.42, 0, 0.85, grigio)
        M.cupola(0, 0, 0.42, 0.85, 0.3, (125, 190, 235), bande=2)
    elif tipo == "oxygen_plant":
        M.scatola(0, 0, 0.75, 0.45, 0, 0.25, grigio)
        for a in (-0.38, 0.38):
            M.cilindro(a, 0, 0.3, 0.25, 1.35, (125, 211, 252))
            M.cupola(a, 0, 0.3, 1.35, 0.18, (125, 211, 252), bande=2)
    elif tipo == "storage_depot":
        M.scatola(0, 0, 0.7, 0.45, 0, 0.55, grigio)
        M.tetto(0, 0, 0.7, 0.45, 0.55, 0.32, (192, 132, 252))
    elif tipo == "weather_station":
        M.scatola(0, 0, 0.05, 0.05, 0, 2.2, (150, 150, 156))
        M.scatola(0, 0, 0.18, 0.18, 2.2, 2.45, (251, 191, 36))
    elif tipo == "heater":
        M.scatola(0, 0, 0.45, 0.45, 0, 0.45, (96, 96, 104))
        M.scatola(0, 0, 0.32, 0.32, 0.45, 0.5, (251, 146, 60))
    elif tipo == "infirmary":
        M.scatola(0, 0, 0.6, 0.45, 0, 0.7, bianco)
        M.scatola(0, 0, 0.3, 0.07, 0.7, 0.73, (220, 38, 38))
        M.scatola(0, 0, 0.07, 0.3, 0.7, 0.73, (220, 38, 38))
    elif tipo == "research_lab":
        M.scatola(0, 0, 0.6, 0.5, 0, 0.7, (226, 240, 238))
        M.cupola(0.15, 0, 0.35, 0.7, 0.3, (45, 190, 175))
    else:
        M.scatola(0, 0, 0.5, 0.5, 0, 0.6, grigio)


def _inviluppo(punti):
    pts = sorted(set(punti))
    if len(pts) < 3:
        return pts

    def croce(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    basso, alto = [], []
    for q in pts:
        while len(basso) >= 2 and croce(basso[-2], basso[-1], q) <= 0:
            basso.pop()
        basso.append(q)
    for q in reversed(pts):
        while len(alto) >= 2 and croce(alto[-2], alto[-1], q) <= 0:
            alto.pop()
        alto.append(q)
    return basso[:-1] + alto[:-1]


def luce_da_camera(cam):
    """La luce di scena: dall'alto a sinistra, alle spalle di chi guarda."""
    L = -0.35 * cam.r + 0.55 * cam.u - 0.45 * cam.f
    return L / np.linalg.norm(L)


_MODELLI = {}


def _stampo(tipo, fine):
    """Il modello del tipo in coordinate locali (km), costruito una volta sola."""
    chiave = (tipo, fine)
    if chiave not in _MODELLI:
        M = _Modello(np.array([0.0, 0.0, 1.0]), 1.0, fine=fine)
        M.P0 = np.zeros(3)
        M.n, M.e, M.m = np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
        _modello(tipo, M)
        vert = np.concatenate([P for P, *_ in M.facce])
        conte = np.array([len(P) for P, *_ in M.facce])
        _MODELLI[chiave] = dict(
            vert=vert, conte=conte, normali=np.array([N for _, _, N, _, _ in M.facce]),
            colori=np.array([c for _, c, _, _, _ in M.facce], float), alfa=np.array([a for *_, a, _ in M.facce]),
            linee=[L for *_, L in M.facce], impronta=M.impronta, altezza=M.altezza)
    return _MODELLI[chiave]


# --- Disposizione senza sovrapposizioni ----------------------------------------------
#
# Come `layoutStructureMarkers` dell'interfaccia (frontend/src/rendering/
# structureLayout.ts): la cella e' divisa in una griglia di posti e ogni edificio
# va nel posto libero piu' vicino alla sua posizione vera, in un ordine fisso
# (per y, poi x, poi tipo). La griglia e' dimensionata sulla cella piu' affollata
# della run, e tutti i modelli sono scalati perche' il piu' ingombrante stia nel
# suo posto con un margine: nessun edificio ne tocca un altro, per costruzione,
# e i rapporti di grandezza fra i tipi restano quelli dei modelli.
CELLA_KM = 59.2
MARGINE_KM = 0.22


def _ingombro_massimo():
    tipi = ("shelter", "habitat", "greenhouse", "solar_array", "water_extractor", "oxygen_plant",
            "storage_depot", "weather_station", "heater", "infirmary", "research_lab", None)
    return max(2 * max(max(abs(a), abs(b)) for a, b in _stampo(t, True)["impronta"]) for t in tipi)


def passo_posti(*snapshots) -> float:
    """Il passo della griglia (km) che fa stare la cella piu' affollata."""
    n = max((len(c.get("structures") or []) for s in snapshots for c in s["cells"]), default=1)
    lato = max(1, math.ceil(math.sqrt(n * 1.06)))
    return min(3.2, CELLA_KM / lato)


def disponi(snap: dict, passo_km: float):
    """[(x + fx, y + fy, tipo)]: ogni edificio nel posto libero piu' vicino."""
    lato = max(1, int(CELLA_KM // passo_km))
    centri = (np.arange(lato) + 0.5) * (CELLA_KM / lato) * 1000        # m, dentro la cella
    gx, gy = np.meshgrid(centri, centri)
    gx, gy = gx.ravel(), gy.ravel()
    out = []
    for c in snap["cells"]:
        strutture = c.get("structures") or []
        if not strutture:
            continue
        geo = c.get("geometry") or {}
        wm = geo.get("width_m") or 59000.0
        hm = geo.get("height_m") or 59000.0
        libero = np.ones(len(gx), bool)
        voci = sorted(((s.get("local_y_m") if s.get("local_y_m") is not None else hm / 2,
                        s.get("local_x_m") if s.get("local_x_m") is not None else wm / 2,
                        s.get("type") or "", i) for i, s in enumerate(strutture)))
        for ly, lx, tipo, _ in voci:
            tx, ty = lx / wm * CELLA_KM * 1000, ly / hm * CELLA_KM * 1000
            d2 = np.where(libero, (gx - tx) ** 2 + (gy - ty) ** 2, np.inf)
            k = int(np.argmin(d2))
            if not np.isfinite(d2[k]):
                continue                        # cella piena: l'edificio non si disegna
            libero[k] = False
            out.append((c["x"] + gx[k] / (CELLA_KM * 1000) - 0.5, c["y"] + gy[k] / (CELLA_KM * 1000) - 0.5, tipo or None))
    return out


class Scena:
    """Gli edifici di un'istantanea, pronti per essere disegnati da qualunque
    camera: la geometria non dipende dalla camera e si costruisce una volta.
    `passo_km` e' il passo della griglia dei posti (vedi `passo_posti`); i
    modelli sono scalati per starci dentro."""

    def __init__(self, snap: dict, luce, passo_km: float | None = None, fine: bool = True):
        self.luce = luce
        passo = passo_km if passo_km is not None else passo_posti(snap)
        scala = (passo - MARGINE_KM) / _ingombro_massimo()
        k = scala / G.RAGGIO_KM
        per_tipo = {}
        for x, y, tipo in disponi(snap, passo):
            per_tipo.setdefault(tipo, []).append((x, y))
        V, facce_v, conte, colori, alfa, normali, centri, istanza, linee = [], [], [], [], [], [], [], [], []
        self.P0, self.ombre = [], []
        base = 0
        n_ist = 0
        for tipo, posti in per_tipo.items():
            st = _stampo(tipo, fine)
            xy = np.array(posti)
            # un po' di naturalezza senza toccarsi: spostamento dentro il margine
            # libero del proprio posto, e rotazioni di 90 gradi (ingombro invariato)
            semi = (np.round(xy[:, 0] * 1e4).astype(np.int64) * 73856093) ^ (np.round(xy[:, 1] * 1e4).astype(np.int64) * 19349663)
            rng = np.random.default_rng(np.abs(semi) % (2 ** 32))
            ingombro = 2 * max(max(abs(a), abs(b)) for a, b in st["impronta"]) * scala
            gioco = max(0.0, (passo - ingombro) / 2 - 0.03) / CELLA_KM
            xy = xy + rng.uniform(-gioco, gioco, size=xy.shape)
            ruota = rng.integers(0, 2, size=len(xy)) if tipo not in ("solar_array",) else np.zeros(len(xy), int)
            lat, lon = G.cella_a_latlon(xy[:, 0], xy[:, 1])
            P0 = G._vettore(lat, lon)
            n = P0 / np.linalg.norm(P0, axis=1, keepdims=True)
            e = np.stack([-P0[:, 1], P0[:, 0], np.zeros(len(P0))], axis=1)
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
            m = np.cross(n, e)
            e, m = np.where(ruota[:, None] == 1, m, e), np.where(ruota[:, None] == 1, -e, m)
            B = np.stack([e, m, n], axis=1)                         # (N, 3, 3): righe e, m, n
            W = P0[:, None, :] + k * np.einsum("vj,njk->nvk", st["vert"], B)
            Nw = np.einsum("fj,njk->nfk", st["normali"], B)
            luce_f = 0.5 + 0.62 * np.clip(np.einsum("nfk,k->nf", Nw, luce), 0, None)
            inizi = np.concatenate([[0], np.cumsum(st["conte"])[:-1]])
            for i in range(len(P0)):
                V.append(W[i])
                for f in range(len(st["conte"])):
                    a0 = base + inizi[f]
                    facce_v.append(a0)
                    conte.append(st["conte"][f])
                    colori.append(np.clip(st["colori"][f] * luce_f[i, f], 0, 255))
                    alfa.append(st["alfa"][f])
                    normali.append(Nw[i, f])
                    istanza.append(n_ist + i)
                    linee.append([P0[i] + k * (np.asarray(L) @ B[i]) for L in st["linee"][f]] if st["linee"][f] else None)
                base += len(W[i])
                # ombra: la pianta spostata lungo la luce, di quanto l'edificio e' alto
                Lt = np.array([luce @ e[i], luce @ m[i]])
                elev = max(0.25, float(luce @ n[i]))
                off = -Lt / (np.linalg.norm(Lt) + 1e-9) * st["altezza"] * 0.9 / elev
                pts = st["impronta"] + [(a + off[0], b + off[1]) for a, b in st["impronta"]]
                guscio = _inviluppo([(round(a, 4), round(b, 4)) for a, b in pts])
                if len(guscio) >= 3:
                    g = np.array([(a, b, 0.01) for a, b in guscio])
                    self.ombre.append(P0[i] + k * (g @ B[i]))
            self.P0.append(P0)
            n_ist += len(P0)
        self.V = np.concatenate(V) if V else np.zeros((0, 3))
        self.inizio = np.array(facce_v, int)
        self.conte = np.array(conte, int)
        self.colori = np.array(colori).astype(int) if colori else np.zeros((0, 3), int)
        self.alfa = np.array(alfa, int)
        self.normali = np.array(normali) if normali else np.zeros((0, 3))
        self.istanza = np.array(istanza, int)
        self.linee = linee
        self.P0 = np.concatenate(self.P0) if self.P0 else np.zeros((0, 3))
        # centro di ogni faccia
        if len(self.conte):
            somme = np.add.reduceat(self.V, self.inizio, axis=0) if np.all(np.diff(self.inizio) > 0) else \
                np.array([self.V[a:a + c].sum(axis=0) for a, c in zip(self.inizio, self.conte)])
            self.centri = somme / self.conte[:, None]
        else:
            self.centri = np.zeros((0, 3))

    def disegna(self, im: Image.Image, cam, ombre: bool = True):
        if not len(self.conte):
            return
        p0, _, vis_ist = cam.proietta(self.P0)
        W_, H_ = im.size
        margine = 0.12 * max(W_, H_)            # fuori inquadratura: non si disegna
        vis_ist &= (p0[:, 0] > -margine) & (p0[:, 0] < W_ + margine) & (p0[:, 1] > -margine) & (p0[:, 1] < H_ + margine)
        if ombre and self.ombre:
            tutte = np.concatenate(self.ombre)
            px, _, v = cam.proietta(tutte)
            strato = Image.new("RGBA", im.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(strato)
            a = 0
            for poli in self.ombre:
                b = a + len(poli)
                if v[a:b].all():
                    d.polygon([tuple(q) for q in px[a:b]], fill=(35, 20, 12, 95))
                a = b
            im.alpha_composite(strato)
        px, _, _ = cam.proietta(self.V)
        verso = np.einsum("fk,fk->f", self.normali, cam.C[None, :] - self.centri) > 0
        ok = vis_ist[self.istanza] & (verso | (self.alfa < 255))
        prof = np.linalg.norm(self.centri - cam.C[None, :], axis=1)
        ordine = np.where(ok)[0]
        ordine = ordine[np.argsort(-prof[ordine])]
        strato = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(strato)
        for f in ordine:
            a, c = self.inizio[f], self.conte[f]
            q = px[a:a + c]
            grande = np.ptp(q[:, 0]) + np.ptp(q[:, 1]) > 14
            d.polygon([tuple(t) for t in q], fill=tuple(self.colori[f]) + (int(self.alfa[f]),),
                      outline=(40, 38, 40, 110) if grande else None)
            if grande and self.linee[f]:
                for Lp in self.linee[f]:
                    lx, _, lv = cam.proietta(Lp)
                    if lv.all():
                        d.line([tuple(t) for t in lx], fill=(235, 245, 255, 160), width=1)
        im.alpha_composite(strato)


def edifici(im: Image.Image, cam, snap: dict, passo_km: float | None = None, ombre: bool = True, luce=None):
    """Gli edifici dell'istantanea, un modello per tipo, con le ombre portate,
    disposti senza sovrapposizioni."""
    Scena(snap, luce if luce is not None else luce_da_camera(cam), passo_km).disegna(im, cam, ombre)
