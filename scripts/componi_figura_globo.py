# -*- coding: utf-8 -*-
"""Compone la figura del globo dai fotogrammi catturati dall'interfaccia di analisi.

**Che cosa mostra.** Due righe, tre momenti ciascuna, sullo *stesso mondo e lo
stesso seme*: sopra la colonia senza governo, sotto la stessa colonia con
governatore e amministratori linguistici. L'unica cosa che cambia fra le due
righe e' chi governa. A fine run la riga di sotto ha piu' coloni e meno della
meta' del territorio: e' l'affermazione centrale del lavoro in una figura.

**Da dove vengono le immagini.** Non sono ridisegnate: sono fotogrammi del globo
tridimensionale con cui le run si osservano davvero (`analysis-frontend`,
componente `ReplayGlobe3D`, scena `marsGlobeScene.ts`), presi dal replay mentre
cambia soltanto il passo. Superficie, quota, vegetazione, strutture, coloni e
perimetro della colonia sono quelli che la scena disegna dai registri.

**Quale run.** `runs/base_pulita`, seme 7, i due bracci `ctrl_none` (baseline) e
`variante_D` (governatore + amministratori), che e' il braccio di riferimento
della campagna a protocollo intero. Il seme non e' stato scelto per l'esito: e'
quello gia' usato nella versione precedente della figura, perche' ha la colonia
non governata piu' estesa fra quelle che conservano le istantanee. Su questo
seme la differenza e' la piu' ampia della campagna, 98 celle; la media sui
cinque semi e' 63,6 e tutti e cinque hanno lo stesso segno. La didascalia in
tesi lo dichiara.

**Perche' i dischi sono portati alla stessa dimensione.** La camera del replay
non si puo' rimettere due volte esattamente alla stessa distanza: la vista si
aggancia alla colonia e poi la si allontana a mano. Fissare la distanza a occhio
lascerebbe due dischi di raggio diverso, e un pianeta piu' grande farebbe
sembrare piu' grande anche la colonia. Qui il raggio del disco si misura su ogni
fotogramma e tutti i dischi vengono portati allo stesso raggio: cio' che si
confronta e' la *frazione di pianeta* che la colonia copre, che e' esattamente
cio' che misura la grandezza «celle occupate».

**Il fondo.** Il replay disegna il pianeta su un cielo nero stellato. In pagina
un rettangolo nero pesa piu' del disegno e lo chiude: qui il disco viene
ritagliato e posato su bianco.

**Come si rifanno.** Si avvia l'interfaccia di analisi con il suo backend
(`npm run dev` in `analysis-frontend/`), si sceglie la run, si passa alla vista
3D, si porta il cursore del passo a fondo corsa, si gira il globo finche' la
colonia guarda la camera, si regola la distanza e si salva il canvas alle tre
fermate; poi si cambia braccio e si ripete. I sei fotogrammi sono archiviati in
`docs/figure_globo/` perche' rimettere la camera dov'era non e' riproducibile a
comando, e la figura deve restare rifacibile anche senza.

Uso:
    python scripts/componi_figura_globo.py
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

#: (file, conteggi). I conteggi vengono dalle istantanee della run e non
#: dall'immagine: li stampa `scripts/conta_globo.py` sulle stesse due run.
PASSI = ("passo 25", "passo 500", "passo 1000")
RIGHE = [
    ("Senza governo",
     [("base_ini.png", "300 coloni, 1 cella"),
      ("base_med.png", "833 coloni, 96 celle"),
      ("base_fin.png", "1621 coloni, 175 celle")]),
    ("Governatore + amministratori (LLM)",
     [("llm_ini.png", "300 coloni, 1 cella"),
      ("llm_med.png", "922 coloni, 44 celle"),
      ("llm_fin.png", "1877 coloni, 77 celle")]),
]

#: Sopra questa luminanza un pixel appartiene al pianeta e non al cielo. Il cielo
#: e' quasi nero, il lembo piu' scuro del pianeta sta ben sopra: la soglia non e'
#: delicata, e le stelle sono puntiformi e vengono tolte dal filtro di maggioranza.
SOGLIA = 42

#: Quanto si taglia dentro il bordo trovato dalla soglia, in frazione di raggio.
#: Il lembo del pianeta e' gia' quasi nero, e una maschera presa sul bordo
#: lascerebbe sul bianco un anello scuro.
RIENTRO = 0.026


def disco(im, margine: int = 0):
    """Centro e raggio del disco del pianeta, misurati sul fotogramma."""
    import numpy as np

    a = np.asarray(im.convert("L"), dtype=np.int16)
    pianeta = a > SOGLIA
    # Una riga o una colonna appartiene al disco se ha abbastanza pixel chiari:
    # una stella isolata non basta a spostare il bordo.
    righe = np.where(pianeta.sum(axis=1) > 12)[0]
    colonne = np.where(pianeta.sum(axis=0) > 12)[0]
    if righe.size == 0 or colonne.size == 0:
        raise SystemExit("non trovo il disco del pianeta nel fotogramma")
    y0, y1 = int(righe[0]), int(righe[-1])
    x0, x1 = int(colonne[0]), int(colonne[-1])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) / 2 + margine
    return cx, cy, r


def su_bianco(im, lato: int, raggio_finale: int):
    """Il disco, ritagliato, portato al raggio comune e posato su un quadrato bianco."""
    from PIL import Image, ImageDraw

    cx, cy, r = disco(im)
    r = r * (1.0 - RIENTRO)
    lato_sorgente = int(round(2 * r))
    ritaglio = im.convert("RGB").crop(
        (round(cx - r), round(cy - r), round(cx - r) + lato_sorgente,
         round(cy - r) + lato_sorgente))
    # Tutti i dischi allo stesso raggio: senza questo passaggio un pianeta piu'
    # grande farebbe sembrare piu' grande anche la colonia che vi sta sopra.
    d = 2 * raggio_finale
    ritaglio = ritaglio.resize((d, d), Image.LANCZOS)

    # Maschera circolare disegnata al quadruplo e rimpicciolita: il bordo del
    # pianeta contro il bianco e' la cosa che si guarda per prima, e a piena
    # risoluzione risulterebbe seghettato.
    s = 4
    maschera = Image.new("L", (d * s, d * s), 0)
    ImageDraw.Draw(maschera).ellipse([0, 0, d * s - 1, d * s - 1], fill=255)
    maschera = maschera.resize((d, d), Image.LANCZOS)

    tela = Image.new("RGB", (lato, lato), "white")
    tela.paste(ritaglio, ((lato - d) // 2, (lato - d) // 2), maschera)
    return tela


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--da", type=Path, default=RADICE / "docs/figure_globo")
    ap.add_argument("--out", type=Path, default=RADICE / "Tesi_LaTex/figures/mondo3d")
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    def carattere(dim: int, grassetto: bool = False):
        nomi = ("arialbd.ttf", "DejaVuSans-Bold.ttf", "segoeuib.ttf") if grassetto \
            else ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf")
        for nome in nomi:
            try:
                return ImageFont.truetype(nome, dim)
            except OSError:
                continue
        return ImageFont.load_default()

    f_passo, f_riga, f_conto = carattere(34), carattere(32, True), carattere(28)

    immagini = {}
    for _, celle in RIGHE:
        for nome, _ in celle:
            immagini[nome] = Image.open(args.da / nome).convert("RGB")

    # Il raggio comune e' il piu' piccolo fra quelli misurati: ingrandire un
    # fotogramma oltre la sua risoluzione lo ammorbidirebbe.
    raggio = int(min(disco(im)[2] * (1.0 - RIENTRO) for im in immagini.values()))
    lato = 2 * raggio + 14

    margine, spazio_col, testo_passo = 6, 34, 46
    riga_h, conto_h, spazio_riga = 46, 40, 26
    larghezza = margine * 2 + lato * 3 + spazio_col * 2
    altezza = (margine + testo_passo
               + len(RIGHE) * (riga_h + lato + conto_h)
               + (len(RIGHE) - 1) * spazio_riga + margine)
    tela = Image.new("RGB", (larghezza, altezza), "white")
    disegna = ImageDraw.Draw(tela)

    def centrato(testo, font, x, larg, y, tinta):
        w = disegna.textlength(testo, font=font)
        disegna.text((x + (larg - w) / 2, y), testo, fill=tinta, font=font)

    for i, passo in enumerate(PASSI):
        centrato(passo, f_passo, margine + i * (lato + spazio_col), lato,
                 margine, "#1a1a1a")

    y = margine + testo_passo
    for etichetta, celle in RIGHE:
        disegna.text((margine, y + 6), etichetta, fill="#7A4A10", font=f_riga)
        y += riga_h
        for i, (nome, conto) in enumerate(celle):
            x = margine + i * (lato + spazio_col)
            tela.paste(su_bianco(immagini[nome], lato, raggio), (x, y))
            centrato(conto, f_conto, x, lato, y + lato + 6, "#5a5a5a")
        y += lato + conto_h + spazio_riga

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tela.save(args.out.with_suffix(".png"))
    tela.save(args.out.with_suffix(".pdf"), "PDF", resolution=300.0)
    print(f"scritto {args.out}.png / .pdf  ({larghezza}x{altezza}, disco r={raggio} px)")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
