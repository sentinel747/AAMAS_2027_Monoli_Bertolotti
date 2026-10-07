# -*- coding: utf-8 -*-
"""Le due risposte del pilota, lette dai registri e non a occhio.

**Prima domanda: ogni braccio regge?** Per ognuno dei dieci bracci corti dice se
ha scritto il risultato, quante tornate ha perso il governatore, quante chiamate
sono fallite, quanti guasti e quante risposte non interpretabili hanno avuto gli
amministratori, e quante politiche vuote sono state adottate. Un braccio che non
compare e' un braccio che non e' partito, ed e' la cosa piu' importante da
vedere: in campagna costerebbe due giorni.

**Seconda domanda: quanto vale il pavimento dei cantieri?** Tre valori sugli
stessi tre semi, senza modello. Si confrontano coloni vivi, celle occupate e
morti, e si guarda che cosa i coloni hanno DAVVERO fatto: se il difetto e' quello
descritto --- il pavimento espelle dal menu manutenzione, ghiaccio e foraggio ---
allora abbassandolo quelle azioni devono tornare a comparire nel conteggio.

Uso:  python scripts/esito_pilota.py [--fuori runs/pilota]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _righe(p: Path):
    if not p.exists():
        return []
    fuori = []
    for riga in p.read_text(encoding="utf-8", errors="replace").splitlines():
        riga = riga.strip()
        if not riga:
            continue
        try:
            fuori.append(json.loads(riga))
        except json.JSONDecodeError:
            continue
    return fuori


def _politiche_vuote(cartella: Path) -> int:
    quante = 0
    for reg in cartella.rglob("governor_decisions.jsonl"):
        for d in _righe(reg):
            if not d.get("missed") and not ((d.get("policy") or {}).get("rules")):
                quante += 1
    return quante


def _azioni(cartella: Path, seme: int) -> Counter:
    """Che cosa hanno fatto i coloni, dal riassunto delle azioni."""
    conta: Counter = Counter()
    for p in sorted(cartella.glob(f"*seed{seme}/action_summary.json")):
        try:
            dati = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sorgente = dati.get("counts") if isinstance(dati, dict) else None
        for azione, quante in (sorgente or dati or {}).items():
            if isinstance(quante, (int, float)):
                conta[str(azione)] += int(quante)
    return conta


def tenuta(base: Path) -> None:
    print("== I bracci reggono? ==")
    print("%-26s %5s %6s %7s %8s %10s %7s" %
          ("braccio", "run", "perse", "fallite", "guasti", "malformate", "vuote"))
    for cartella in sorted(base.iterdir()):
        if not cartella.is_dir() or cartella.name.startswith("pavimento"):
            continue
        righe = _righe(cartella / "results.jsonl")
        if not righe:
            print("%-26s  NESSUN RISULTATO: il braccio non e' partito" % cartella.name)
            continue
        for r in righe:
            api = r.get("api") or {}
            amm = r.get("amministrazione") or {}
            print("%-26s %5s %6s %7s %8s %10s %7s" % (
                cartella.name, r.get("seed"),
                r.get("governor_misses", "-"),
                api.get("failed_calls", "-"),
                amm.get("provider_failures", "assente" if amm else "-"),
                amm.get("malformed", "assente" if amm else "-"),
                _politiche_vuote(cartella),
            ))


def pavimento(base: Path) -> None:
    print()
    print("== Il pavimento dei cantieri ==")
    cartelle = sorted(p for p in base.iterdir()
                      if p.is_dir() and p.name.startswith("pavimento"))
    if not cartelle:
        print("nessuna run del pavimento: il pilota non e' arrivato in fondo.")
        return
    print("%-18s %5s %8s %7s %8s" % ("valore", "seme", "vivi", "celle", "morti"))
    medie: dict[str, list[tuple[float, float, float]]] = {}
    for cartella in cartelle:
        valore = cartella.name.replace("pavimento_", "").replace("_", ".")
        for r in _righe(cartella / "results.jsonl"):
            vivi = float(r.get("population") or 0)
            celle = float((r.get("espansione") or {}).get("celle_totali") or 0)
            morti = float(r.get("deaths") or 0)
            print("%-18s %5s %8.0f %7.0f %8.0f" %
                  (valore, r.get("seed"), vivi, celle, morti))
            medie.setdefault(valore, []).append((vivi, celle, morti))

    print()
    print("%-18s %8s %7s %8s" % ("media", "vivi", "celle", "morti"))
    for valore, valori in sorted(medie.items()):
        n = len(valori)
        print("%-18s %8.0f %7.0f %8.0f" % (
            valore,
            sum(v for v, _, _ in valori) / n,
            sum(c for _, c, _ in valori) / n,
            sum(m for _, _, m in valori) / n,
        ))

    print()
    print("== Che cosa hanno fatto i coloni (seme 3) ==")
    interessanti = ("maintain_structure", "collect_ice", "forage",
                    "collect_materials", "collect_minerals")
    for cartella in cartelle:
        valore = cartella.name.replace("pavimento_", "").replace("_", ".")
        conta = _azioni(cartella, 3)
        if not conta:
            print("%-18s (nessun riassunto delle azioni)" % valore)
            continue
        costruzioni = sum(q for a, q in conta.items() if a.startswith("build_"))
        pezzi = ", ".join(f"{a}={conta.get(a, 0)}" for a in interessanti)
        print("%-18s build=%-7d %s" % (valore, costruzioni, pezzi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fuori", default="runs/pilota")
    args = ap.parse_args()
    base = Path(args.fuori)
    if not base.exists():
        print("non trovo", base)
        return 1
    tenuta(base)
    pavimento(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
