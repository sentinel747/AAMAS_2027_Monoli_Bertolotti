# -*- coding: utf-8 -*-
"""Esperimento di controllo sulla copertura elettrica (2026-09-26).

Domanda: la compressione territoriale del braccio principale della tesi
(governatore LLM Qwen, prompt D, con amministratori, mondo balanced, 1000
passi) regge quando l'indicatore `power_coverage` dice il vero?

Confronta, seme per seme (3-7):
- `ctrl_none`         (tesi, `runs/base_qwen/ctrl_none`);
- `variante_D`        (tesi, prima esecuzione);
- `variante_D_rep2`   (tesi, seconda esecuzione: il metro del rumore);
- `variante_D_vera`   (oggi, identico piu' `--copertura-elettrica-vera`).

Convenzioni di `scripts/esiti_campagna.py`: territorio = `celle_totali`, celle
abitate = `celle_insediate`, nascite = vivi - 300 + morti. Riporta anche la
quota di celle-passo catturate da regole su `power_coverage`.

Uso:
    python scripts/controllo_copertura_report.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONDATORI = 300
BRACCI = {
    "ctrl_none": ROOT / "runs/base_qwen/ctrl_none",
    "variante_D": ROOT / "runs/base_qwen/variante_D",
    "variante_D_rep2": ROOT / "runs/base_qwen/variante_D_rep2",
    "variante_D_vera": ROOT / "runs/controllo_copertura_20260926",
    "variante_D_vera_rep2": ROOT / "runs/controllo_copertura_20260926",
}
GRANDEZZE = ("vivi", "morti", "nascite", "territorio", "celle abitate")


def _esito(r: dict) -> dict:
    vivi, morti = int(r["population"]), int(r["deaths"])
    esp = r.get("espansione") or {}
    return {"vivi": vivi, "morti": morti, "nascite": vivi - FONDATORI + morti,
            "territorio": esp.get("celle_totali"), "celle abitate": esp.get("celle_insediate")}


def _quota_power(cartella: Path) -> float | None:
    file = list(cartella.glob("**/governor_policy_hits.json"))
    tot = pw = 0
    for f in file:
        h = json.loads(f.read_text(encoding="utf-8"))
        for k, v in h.items():
            if isinstance(v, (int, float)):
                tot += v
                pw += v if "power_coverage" in k else 0
    return pw / tot if tot else None


def raccogli() -> dict:
    out: dict = {}
    for nome, base in BRACCI.items():
        if nome.startswith("variante_D_vera"):
            for d in sorted(base.glob(f"{nome}_s*")):
                f = d / "results.json"
                if f.exists():
                    r = json.loads(f.read_text(encoding="utf-8"))
                    r = r[0] if isinstance(r, list) else r
                    out.setdefault(nome, {})[int(r["seed"])] = {**_esito(r), "quota_power": _quota_power(d)}
            continue
        f = base / "results.json"
        if not f.exists():
            continue
        rs = json.loads(f.read_text(encoding="utf-8"))
        for r in rs if isinstance(rs, list) else [rs]:
            sub = [p for p in base.glob(f"*seed{r['seed']}") if p.is_dir()]
            out.setdefault(nome, {})[int(r["seed"])] = {
                **_esito(r), "quota_power": _quota_power(sub[0]) if sub else None}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    dati = raccogli()
    semi = sorted(dati.get("ctrl_none", {}))
    print("| braccio | seme | " + " | ".join(GRANDEZZE) + " | quota power_coverage |")
    print("|---|---:|" + "---:|" * (len(GRANDEZZE) + 1))
    for nome in BRACCI:
        for s in semi:
            e = dati.get(nome, {}).get(s)
            if e:
                q = e.get("quota_power")
                print(f"| {nome} | {s} | " + " | ".join(str(e[g]) for g in GRANDEZZE)
                      + f" | {'-' if q is None else f'{100 * q:.1f}%'} |")
    print()
    print("Differenza dal `ctrl_none` dello stesso seme (media, segno):")
    for nome in ("variante_D", "variante_D_rep2", "variante_D_vera", "variante_D_vera_rep2"):
        riga = []
        for g in GRANDEZZE:
            diffs = [dati[nome][s][g] - dati["ctrl_none"][s][g]
                     for s in semi if s in dati.get(nome, {})]
            if not diffs:
                continue
            neg = sum(d < 0 for d in diffs)
            pos = sum(d > 0 for d in diffs)
            riga.append(f"{g} {sum(diffs) / len(diffs):+.0f} ({max(neg, pos)}/{len(diffs)} {'-' if neg >= pos else '+'})")
        print(f"  {nome:16s} " + "; ".join(riga))
    for a, b in (("variante_D", "variante_D_rep2"), ("variante_D_vera", "variante_D_vera_rep2")):
        ok = [s for s in semi if s in dati.get(a, {}) and s in dati.get(b, {})]
        if ok:
            print()
            print(f"Rumore (|{a} - {b}|, media sui semi):")
            print("  " + "; ".join(
                f"{g} {sum(abs(dati[a][s][g] - dati[b][s][g]) for s in ok) / len(ok):.0f}"
                for g in GRANDEZZE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
