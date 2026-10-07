# -*- coding: utf-8 -*-
"""Estrae dalla tesi ogni frase che contiene un numero, per la verifica a mano.

La verifica dei numeri contro i registri si fa a mano, frase per frase, ma
trovare le frasi non deve essere fatto a mano: questo script le tira fuori,
divise per capitolo e per sezione, saltando le zone in cui un numero non e'
un'affermazione empirica (preamboli, verbatim, bibliografia) e le forme che
sono riferimenti e non misure (numeri di equazione, di figura, di tabella,
citazioni, anni).

Uso:
    python scripts/estrai_affermazioni.py --capitolo 3
    python scripts/estrai_affermazioni.py --tutte > C:/tmp/affermazioni.txt
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
TEX = RADICE / "Tesi_LaTex" / "Tesi_Terraforming_LLMs.tex"

#: Un numero conta come affermazione se non e' uno di questi.
RUMORE = re.compile(
    r"\\(ref|cref|Cref|label|cite|parencite|textcite|includegraphics|input|"
    r"includegraphics|url|href|footnote)\b"
)
NUMERO = re.compile(r"(?<![A-Za-z\\])\d")


def sezioni(righe: list[str]) -> list[tuple[int, str, str]]:
    """(riga, livello, titolo) per capitoli, sezioni e sottosezioni."""
    fuori = []
    for i, l in enumerate(righe):
        m = re.match(r"\\(chapter|section|subsection)\*?\{(.*?)\}", l)
        if m:
            fuori.append((i, m.group(1), m.group(2)))
    return fuori


def zone_da_saltare(righe: list[str]) -> list[tuple[int, int]]:
    """Intervalli di righe in cui i numeri non sono affermazioni."""
    fuori = []
    apri = None
    for i, l in enumerate(righe):
        if re.search(r"\\begin\{(lstlisting|verbatim|tikzpicture|equation|align)", l):
            apri = i
        elif apri is not None and re.search(r"\\end\{(lstlisting|verbatim|tikzpicture|equation|align)", l):
            fuori.append((apri, i))
            apri = None
    # il preambolo
    for i, l in enumerate(righe):
        if l.startswith(r"\begin{document}"):
            fuori.append((0, i))
            break
    return fuori


def frasi(testo: str) -> list[str]:
    """Spezza in frasi senza rompere i numeri decimali scritti alla italiana."""
    pezzi = re.split(r"(?<=[.;:])\s+(?=[A-ZÈÉÀÌÒÙ\\])", testo)
    return [p.strip() for p in pezzi if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capitolo", type=int, default=None)
    ap.add_argument("--tutte", action="store_true")
    ap.add_argument("--da", type=str, default=None, help="titolo di sezione da cui partire")
    args = ap.parse_args()

    righe = TEX.read_text(encoding="utf-8").split("\n")
    salta = zone_da_saltare(righe)
    dentro = lambda i: any(a <= i <= b for a, b in salta)  # noqa: E731

    strutt = sezioni(righe)
    cap = 0
    corrente = ("", "")
    n = 0
    for i, l in enumerate(righe):
        for riga, livello, titolo in strutt:
            if riga == i:
                if livello == "chapter":
                    cap += 1
                    corrente = (f"cap. {cap}", titolo)
                else:
                    corrente = (corrente[0], titolo)
                if args.tutte or args.capitolo == cap:
                    print(f"\n{'=' * 78}\n{corrente[0]} | {livello}: {titolo}\n{'=' * 78}")
        if args.capitolo is not None and cap != args.capitolo:
            continue
        if not (args.tutte or args.capitolo is not None):
            continue
        if dentro(i) or l.lstrip().startswith("%") or not l.strip():
            continue
        for f in frasi(l):
            if not NUMERO.search(RUMORE.sub("", f)):
                continue
            pulita = re.sub(r"\s+", " ", f)
            n += 1
            print(f"\n[L{i + 1}] {pulita}")
    print(f"\n\n--- {n} frasi con numeri ---", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
