# -*- coding: utf-8 -*-
"""Un punteggio di difficolta' della configurazione, calcolabile PRIMA della run.

**Perche'.** Confrontare governi su mondi diversi ha senso solo se si sa
quanto ciascun mondo e' difficile, e «difficile» deve essere un numero che si
calcola dal mondo e dalla configurazione, non dall'esito. Altrimenti si legge
il risultato due volte: una come misura, una come spiegazione.

**Che cosa entra.** Tutto cio' che e' fissato prima del primo passo e che la
simulazione stessa usa come vincolo: la base statica del mondo
(`world_static_base.json`) e la dotazione iniziale (`dotazione` nella
configurazione). La colonia nasce nella cella madre --- il centro dichiarato
nei metadati della base --- e nelle prime migliaia di giorni non esce da un
intorno di poche celle: il punteggio si calcola sulla FINESTRA di raggio `R`
(Chebyshev) attorno alla madre, che a raggio 5 sono 121 celle, l'ordine di
grandezza che una colonia di questo disegno arriva a occupare.

**Le quattro componenti**, ciascuna in [0, 1] con 1 = piu' difficile, e la
loro media come punteggio composito `D`:

1. `pericolo`  = media di radiazione/2, polvere, rischio del terreno, rischio
   di attraversamento (le ultime tre sono gia' in [0, 1]; la radiazione della
   generazione sta in [0.4, 1.8] e si riporta in [0, 1] dividendo per 2);
2. `scarsita`  = media dei deficit di ghiaccio, minerali e materiale da
   costruzione rispetto alle medie GLOBALI del profilo di riferimento
   (`balanced`, misurate sulla sua base statica: sono le costanti sotto),
   `1 - min(1, valore / riferimento)`;
3. `sito`      = 1 - media del `colony_site_score` delle celle, l'indice di
   qualita' del sito che la simulazione stessa calcola;
4. `dotazione` = 1 - moltiplicatore della dotazione iniziale (0.6 -> 0.4).

Pesi uguali, dichiarati: non c'e' una teoria che ne giustifichi altri, e un
peso stimato sugli esiti renderebbe il punteggio circolare. La validazione e'
a parte: la correlazione di rango fra `D` e gli esiti della BASELINE (vivi,
morti, celle) sulle configurazioni eseguite. Se il punteggio e' buono, la
baseline fa peggio dove `D` e' alto; se non lo e', si vede qui e non in tesi.

Uso:
    python scripts/score_difficolta.py <cartella-braccio-baseline> [...] [--raggio 5] [--out file.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analisi_decentramento import _cartelle_run, _risultati  # noqa: E402

#: Medie globali del profilo `balanced` (seme 3, 360x180), lette dalla sua base
#: statica il 2026-09-07. Sono il metro della scarsita': un mondo con meta' dei
#: minerali del riferimento ha deficit 0.5. Si ricalcolano con
#: `riferimento_da_base(base)` se il generatore cambia.
RIFERIMENTO = {"water_ice": 8.124, "minerals": 7.56, "construction_material": 12.61}
#: La radiazione della generazione sta in [0.4, 1.8]: /2 la porta in [0, 1].
RADIAZIONE_SCALA = 2.0
RAGGIO_DEFAULT = 5


def cella_madre(base: dict) -> tuple[int, int]:
    """(y, x) della cella madre dal centro dichiarato nei metadati."""
    md = base.get("metadata") or {}
    lat, lon = float(md["center_latitude_deg"]), float(md["center_longitude_deg"])
    w, h = int(base["width"]), int(base["height"])
    x = min(w - 1, max(0, int((lon + 180.0) / 360.0 * w)))
    y = min(h - 1, max(0, int((90.0 - lat) / 180.0 * h)))
    return y, x


def finestra(base: dict, raggio: int) -> list[dict]:
    y0, x0 = cella_madre(base)
    w = int(base["width"])
    fuori = []
    for c in base["cells"]:
        dy = abs(int(c["y"]) - y0)
        dx = abs(int(c["x"]) - x0)
        dx = min(dx, w - dx)  # la longitudine si chiude
        if dy <= raggio and dx <= raggio:
            fuori.append(c)
    return fuori


def _media(celle: list[dict], chiave: str, default: float = 0.0) -> float:
    v = [float(c[chiave]) for c in celle if isinstance(c.get(chiave), (int, float))]
    return statistics.fmean(v) if v else default


def _media_risorsa(celle: list[dict], chiave: str) -> float:
    v = [float((c.get("resources") or {}).get(chiave, 0.0)) for c in celle if isinstance(c.get("resources"), dict)]
    return statistics.fmean(v) if v else 0.0


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def componenti(base: dict, dotazione: float, raggio: int = RAGGIO_DEFAULT, riferimento: dict | None = None) -> dict:
    rif = riferimento or RIFERIMENTO
    W = finestra(base, raggio)
    pericolo = statistics.fmean([
        _clip01(_media(W, "radiation") / RADIAZIONE_SCALA),
        _clip01(_media(W, "dust")),
        _clip01(_media(W, "terrain_base_risk")),
        _clip01(_media(W, "traversal_risk")),
    ])
    scarsita = statistics.fmean([
        1.0 - _clip01(_media(W, "water_ice") / rif["water_ice"]),
        1.0 - _clip01(_media_risorsa(W, "minerals") / rif["minerals"]),
        1.0 - _clip01(_media_risorsa(W, "construction_material") / rif["construction_material"]),
    ])
    sito = 1.0 - _clip01(_media(W, "colony_site_score"))
    dot = 1.0 - _clip01(float(dotazione))
    comp = {"pericolo": pericolo, "scarsita": scarsita, "sito": sito, "dotazione": dot}
    comp["D"] = statistics.fmean(comp.values())
    comp["celle_finestra"] = len(W)
    comp["madre"] = list(cella_madre(base))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}


def riferimento_da_base(base: dict) -> dict:
    """Le medie globali di una base: per ricalcolare `RIFERIMENTO`."""
    celle = base["cells"]
    return {
        "water_ice": round(_media(celle, "water_ice"), 3),
        "minerals": round(_media_risorsa(celle, "minerals"), 3),
        "construction_material": round(_media_risorsa(celle, "construction_material"), 3),
    }


# --- validazione: correlazione di rango con gli esiti della baseline ------------------------

def _ranghi(valori: list[float]) -> list[float]:
    ordine = sorted(range(len(valori)), key=lambda i: valori[i])
    ranghi = [0.0] * len(valori)
    i = 0
    while i < len(ordine):
        j = i
        while j + 1 < len(ordine) and valori[ordine[j + 1]] == valori[ordine[i]]:
            j += 1
        medio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranghi[ordine[k]] = medio
        i = j + 1
    return ranghi


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = _ranghi(a), _ranghi(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 3) if den else None


def analizza_braccio(braccio: Path, raggio: int) -> list[dict]:
    risultati = _risultati(braccio)
    righe = []
    for seme, cartella in sorted(_cartelle_run(braccio).items()):
        base_path = cartella / "world_static_base.json"
        if not base_path.exists():
            continue
        base = json.loads(base_path.read_text(encoding="utf-8", errors="replace"))
        riga = risultati.get(seme, {})
        dotazione = float((riga.get("config") or {}).get("dotazione", 1.0) or 1.0)
        comp = componenti(base, dotazione, raggio)
        comp.update({
            "mondo": str((base.get("metadata") or {}).get("map_profile", "?")),
            "seme": seme,
            "braccio": braccio.name,
            "vivi": int(riga.get("population", 0) or 0),
            "morti": int(riga.get("deaths", 0) or 0),
            "celle": int((riga.get("espansione") or {}).get("celle_totali", 0) or 0),
        })
        righe.append(comp)
    return righe


def rapporto(bracci: list[Path], raggio: int) -> list[str]:
    righe_dati: list[dict] = []
    for b in bracci:
        righe_dati.extend(analizza_braccio(b, raggio))
    out = [f"# Punteggio di difficolta' della configurazione (finestra di raggio {raggio} attorno alla madre)", ""]
    out.append(
        "D = media di quattro componenti in [0,1], 1 = piu' difficile: pericolo (radiazione/2, polvere, "
        "rischi), scarsita' (deficit di ghiaccio, minerali, materiale rispetto al profilo balanced), "
        "sito (1 - qualita' del sito), dotazione (1 - moltiplicatore). Esiti della BASELINE accanto, per la validazione."
    )
    out.append("")
    out.append("| mondo | seme | madre | pericolo | scarsita' | sito | dotazione | **D** | vivi | morti | celle |")
    out.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(righe_dati, key=lambda r: (r["D"], r["mondo"], r["seme"])):
        out.append(
            f"| {r['mondo']} | {r['seme']} | {tuple(r['madre'])} | {r['pericolo']:.3f} | {r['scarsita']:.3f} | "
            f"{r['sito']:.3f} | {r['dotazione']:.2f} | **{r['D']:.3f}** | {r['vivi']} | {r['morti']} | {r['celle']} |"
        )
    out.append("")
    per_mondo: dict[str, list[dict]] = {}
    for r in righe_dati:
        per_mondo.setdefault(r["mondo"], []).append(r)
    out.append("| mondo | D medio | pericolo | scarsita' | sito | vivi medi | morti medi | celle medie |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mondo, rs in sorted(per_mondo.items(), key=lambda kv: statistics.fmean(r["D"] for r in kv[1])):
        f = lambda k: statistics.fmean(r[k] for r in rs)  # noqa: E731
        out.append(f"| {mondo} | **{f('D'):.3f}** | {f('pericolo'):.3f} | {f('scarsita'):.3f} | {f('sito'):.3f} | {f('vivi'):.0f} | {f('morti'):.0f} | {f('celle'):.0f} |")
    out.append("")
    if len(righe_dati) >= 3:
        D = [r["D"] for r in righe_dati]
        out.append("Correlazione di rango (Spearman) fra D e gli esiti della baseline, su tutte le run:")
        for k in ("vivi", "morti", "celle"):
            out.append(f"- D contro {k}: {spearman(D, [r[k] for r in righe_dati])}")
        for comp in ("pericolo", "scarsita", "sito"):
            out.append(f"- {comp} contro vivi: {spearman([r[comp] for r in righe_dati], [r['vivi'] for r in righe_dati])}; contro morti: {spearman([r[comp] for r in righe_dati], [r['morti'] for r in righe_dati])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bracci", nargs="+", type=Path, help="cartelle di braccio (baseline) con le run <braccio>_seed<n>/")
    ap.add_argument("--raggio", type=int, default=RAGGIO_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    testo = "\n".join(rapporto(args.bracci, args.raggio)) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(testo, encoding="utf-8")
        print(f"scritto {args.out}")
    else:
        sys.stdout.write(testo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
