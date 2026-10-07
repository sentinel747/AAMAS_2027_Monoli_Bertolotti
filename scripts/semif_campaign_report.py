# -*- coding: utf-8 -*-
"""Esiti della campagna System 1 v2h, ogni effetto accanto al rumore che puo' imitarlo.

Segue le convenzioni di `scripts/esiti_campagna.py` (quelle della tesi):

- **territorio** = `celle_totali` (celle con infrastruttura), **celle abitate** =
  `celle_insediate`: sono grandezze diverse e si riportano entrambe;
- **nascite** = popolazione finale - 300 fondatori + morti (identita' della
  popolazione, il registro riassuntivo non le conta);
- **il metro e' la ripetizione**: le tre esecuzioni Jev del seme 3 misurano
  quanto il risultato cambia da se'; il rumore e' la media delle differenze
  ASSOLUTE fra le coppie di esecuzioni identiche (la media con segno si annulla
  per compensazione);
- un effetto contro `none` si dichiara solo se e' **piu' grande del rumore e
  unanime nel segno** su tutti i semi disponibili.

Le run hanno cartelle `<braccio>_s<seme>` (ripetizioni `jev_s3_rep2`, ...).

Uso:
    python scripts/semif_campaign_report.py runs/jev_semif_experiments/campaign_v2h_20260923
    python scripts/semif_campaign_report.py <campagna> --json esiti.json
    python scripts/semif_campaign_report.py <campagna v4> --riusa <campagna v2h>
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic_governance.artifacts import governance_summary  # noqa: E402

FONDATORI = 300
GRANDEZZE = ("vivi", "morti", "nascite", "morti/nascita", "territorio", "celle abitate")
BRACCI = ("none", "scripted", "jev", "qwen", "laya")
NOMI = {
    "none": "nessun governo",
    "scripted": "costituzione scripted + amm. scripted",
    "jev": "Jev (TypeSafe) + amm. System 1",
    "qwen": "Qwen (farm) + amm. System 1",
    "laya": "Laya distillato + amm. System 1",
}
CARTELLA = re.compile(r"^(?P<braccio>[a-z]+)_s(?P<seme>\d+)(?:_rep(?P<rep>\d+))?$")


def _risultato(cartella: Path) -> dict | None:
    f = cartella / "results.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    r = r[0] if isinstance(r, list) else r
    esp = r.get("espansione") or {}
    vivi, morti = int(r["population"]), int(r["deaths"])
    nascite = vivi - FONDATORI + morti
    esito = {
        "vivi": vivi,
        "morti": morti,
        "nascite": nascite,
        "morti/nascita": round(morti / nascite, 3) if nascite else None,
        "territorio": esp.get("celle_totali"),
        "celle abitate": esp.get("celle_insediate"),
        "durata_min": round(float(r.get("wall_clock_s", 0.0)) / 60, 1),
    }
    interne = [d for d in cartella.iterdir() if (d / "governor_decisions.jsonl").exists()]
    if interne:
        esito["governo"] = governance_summary(interne[0])
    return esito


def raccogli(campagna: Path) -> dict:
    runs: dict = {}
    for d in sorted(campagna.iterdir()):
        m = CARTELLA.match(d.name) if d.is_dir() else None
        if not m:
            continue
        esito = _risultato(d)
        if esito is None:
            continue
        rep = int(m.group("rep") or 1)
        runs.setdefault(m.group("braccio"), {}).setdefault(int(m.group("seme")), {})[rep] = esito
    return runs


def rumore(runs: dict) -> dict:
    """Per ogni grandezza: differenze assolute fra esecuzioni identiche (tutte le coppie)."""
    out = {}
    for g in GRANDEZZE:
        diffs = []
        for semi in runs.values():
            for esecuzioni in semi.values():
                if len(esecuzioni) < 2:
                    continue
                for a, b in itertools.combinations(sorted(esecuzioni), 2):
                    va, vb = esecuzioni[a].get(g), esecuzioni[b].get(g)
                    if va is not None and vb is not None:
                        diffs.append(abs(va - vb))
        if diffs:
            out[g] = {"coppie": len(diffs), "medio": round(sum(diffs) / len(diffs), 3),
                      "massimo": round(max(diffs), 3)}
    return out


def rumore_per_braccio(runs: dict) -> dict:
    """Il metro di ciascun braccio, solo dalle sue ripetizioni (2026-09-24).

    Con le ripetizioni di piu' bracci il metro unico le mescolerebbe: ogni
    braccio va letto contro il PROPRIO rumore quando lo ha.
    """
    return {b: m for b, semi in runs.items() if (m := rumore({b: semi}))}


def contrasti(runs: dict, metro: dict, metri: dict | None = None) -> dict:
    """Braccio contro `none`, appaiato per seme, prima esecuzione di ciascun seme.

    La soglia e' il rumore del braccio stesso se ha ripetizioni (`metri`),
    altrimenti `metro` (il metro preregistrato: le ripetizioni di Jev).
    """
    base = runs.get("none", {})
    out = {}
    for braccio in BRACCI:
        if braccio == "none" or braccio not in runs:
            continue
        metro_braccio = (metri or {}).get(braccio) or metro
        righe = {}
        for g in GRANDEZZE:
            diffs = {}
            for seme, esecuzioni in sorted(runs[braccio].items()):
                if seme in base and 1 in esecuzioni:
                    a, b = esecuzioni[1].get(g), base[seme][1].get(g)
                    if a is not None and b is not None:
                        diffs[seme] = round(a - b, 3)
            if not diffs:
                continue
            media = sum(diffs.values()) / len(diffs)
            positivi = sum(1 for v in diffs.values() if v > 0)
            negativi = sum(1 for v in diffs.values() if v < 0)
            concordi = max(positivi, negativi)
            soglia = metro_braccio.get(g, {}).get("medio")
            unanime = concordi == len(diffs)
            sopra = soglia is not None and abs(media) > soglia
            righe[g] = {
                "per_seme": diffs, "media": round(media, 3),
                "segno": f"{concordi}/{len(diffs)} {'+' if positivi >= negativi else '-'}",
                "sopra_rumore": sopra, "unanime": unanime,
                "verdetto": "effetto" if (sopra and unanime) else "non distinguibile dal rumore",
            }
        out[braccio] = righe
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def markdown(runs: dict, metro: dict, contr: dict, metri: dict | None = None) -> str:
    L = ["## Esiti per run", "",
         "| braccio | seme | es. | vivi | morti | nascite | morti/nascita | territorio | celle abitate | min |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for braccio in BRACCI:
        for seme, esecuzioni in sorted(runs.get(braccio, {}).items()):
            for rep, e in sorted(esecuzioni.items()):
                L.append(f"| {braccio} | {seme} | {rep} | " + " | ".join(
                    _fmt(e.get(g)) for g in GRANDEZZE) + f" | {e['durata_min']} |")
    L += ["", "## Il metro: esecuzioni identiche dello stesso seme, braccio per braccio", "",
          "| braccio | grandezza | coppie | scarto assoluto medio | massimo |", "|---|---|---:|---:|---:|"]
    for braccio, mb in (metri or {"tutti": metro}).items():
        for g, m in mb.items():
            L.append(f"| {braccio} | {g} | {m['coppie']} | {_fmt(m['medio'])} | {_fmt(m['massimo'])} |")
    L += ["", "## Contro `none`, appaiato per seme", ""]
    for braccio, righe in contr.items():
        L += [f"### {NOMI[braccio]}", "",
              "| grandezza | per seme | media | segno | verdetto |", "|---|---|---:|---|---|"]
        for g, r in righe.items():
            per = ", ".join(f"s{s}: {_fmt(v)}" for s, v in r["per_seme"].items())
            L.append(f"| {g} | {per} | {_fmt(r['media'])} | {r['segno']} | {r['verdetto']} |")
        L.append("")
    L += ["## Come governa", "",
          "| braccio | seme | es. | cambi | riaff. | attese | abrog. | tasso int. | distr. | riscr. | acc. | senza legge |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for braccio in BRACCI:
        for seme, esecuzioni in sorted(runs.get(braccio, {}).items()):
            for rep, e in sorted(esecuzioni.items()):
                gov = e.get("governo") or {}
                g, a = gov.get("governor") or {}, gov.get("administrators") or {}
                if not g:
                    continue
                tasso = (g["law_changes"] + g["reaffirmations"]) / g["rounds"] if g.get("rounds") else None
                L.append(f"| {braccio} | {seme} | {rep} | {g.get('law_changes')} | {g.get('reaffirmations')} | "
                         f"{g.get('holds')} | {g.get('abrogations')} | {_fmt(tasso)} | {a.get('district_rounds')} | "
                         f"{a.get('rewrites')} | {a.get('accepts')} | {a.get('no_governor_law')} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campagna", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--riusa", type=Path, action="append", default=[],
        help="altra campagna da cui prendere i bracci che qui mancano (es. none e "
             "scripted di una campagna precedente, se il simulatore e' identico)",
    )
    parser.add_argument(
        "--bracci-riusati", nargs="+", default=["none", "scripted"],
        help="quali bracci prendere dalle campagne di --riusa (default: none scripted)",
    )
    args = parser.parse_args(argv)
    runs = raccogli(args.campagna)
    for altra in args.riusa:
        for braccio, semi in raccogli(altra).items():
            if braccio in args.bracci_riusati:
                runs.setdefault(braccio, semi)
    metri = rumore_per_braccio(runs)
    # Il metro preregistrato e' quello di Jev; senza ripetizioni di Jev, tutte.
    metro = metri.get("jev") or rumore(runs)
    contr = contrasti(runs, metro, metri)
    sys.stdout.reconfigure(encoding="utf-8")
    print(markdown(runs, metro, contr, metri))
    if args.json:
        args.json.write_text(json.dumps({"runs": runs, "rumore": metro, "rumore_per_braccio": metri,
                                         "contrasti": contr},
                                        indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
