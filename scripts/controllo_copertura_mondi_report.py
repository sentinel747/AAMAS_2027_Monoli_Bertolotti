# -*- coding: utf-8 -*-
"""Controllo della copertura elettrica sugli altri mondi (2026-09-26).

Per ogni mondo confronta, seme per seme, contro `runs/mondi/<mondo>/ctrl_none`:
- le esecuzioni della tesi (`llm_completo_amm`, `_rep2`, `_rep3` se presenti;
  per `scarce_resources` la campagna corretta `llm_completo_amm_v2` e `_v2_rep2`);
- le run con `--copertura-elettrica-vera`
  (`runs/controllo_copertura_mondi/<mondo>/llm_amm_vera_s<seme>`).

Territorio = `celle_totali`, come in tesi. Stampa la differenza media per
esecuzione, i segni concordi e il rumore fra le esecuzioni della tesi.

**Baseline di oggi.** La `ctrl_none` del 7 settembre NON si riproduce bit per
bit col codice attuale (ice_rich seme 3 diverge dal passo ~198): le run
corrette si confrontano con `ctrl_none_oggi_s<seme>`, calcolata oggi nella
stessa configurazione; quelle della tesi con la propria `ctrl_none`. Ogni
effetto resta appaiato dentro la propria versione del codice.

Uso:
    python scripts/controllo_copertura_mondi_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONDI = ("ice_rich", "fragmented", "high_hazard", "scarce_resources")
TESI = {
    "scarce_resources": ("llm_completo_amm_v2", "llm_completo_amm_v2_rep2"),
}
TESI_DEFAULT = ("llm_completo_amm", "llm_completo_amm_rep2", "llm_completo_amm_rep3")


def _territorio(results: Path) -> dict[int, int]:
    rs = json.loads(results.read_text(encoding="utf-8"))
    rs = rs if isinstance(rs, list) else [rs]
    return {int(r["seed"]): int(r["espansione"]["celle_totali"]) for r in rs}


def _riga(nome: str, diffs: dict[int, int]) -> str:
    if not diffs:
        return f"  {nome:28s} -"
    media = sum(diffs.values()) / len(diffs)
    neg = sum(d < 0 for d in diffs.values())
    per = ", ".join(f"s{s}: {d:+d}" for s, d in sorted(diffs.items()))
    return f"  {nome:28s} media {media:+6.1f}  ({neg}/{len(diffs)} sotto)   {per}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for mondo in MONDI:
        base_dir = ROOT / "runs/mondi" / mondo
        base = _territorio(base_dir / "ctrl_none/results.json")
        print(f"== {mondo}  (senza governo: {', '.join(f's{s} {v}' for s, v in sorted(base.items()))})")
        esecuzioni = []
        for braccio in TESI.get(mondo, TESI_DEFAULT):
            f = base_dir / braccio / "results.json"
            if f.exists():
                t = _territorio(f)
                esecuzioni.append(t)
                print(_riga(f"tesi {braccio}", {s: t[s] - base[s] for s in t if s in base}))
        vera, oggi = {}, {}
        for s in sorted(base):
            cartella = ROOT / "runs/controllo_copertura_mondi" / mondo
            f = cartella / f"llm_amm_vera_s{s}" / "results.json"
            if f.exists():
                vera[s] = _territorio(f)[s]
            g = cartella / f"ctrl_none_oggi_s{s}" / "results.json"
            if g.exists():
                oggi[s] = _territorio(g)[s]
        if oggi:
            print(f"  senza governo, oggi: {', '.join(f's{s} {v}' for s, v in sorted(oggi.items()))}")
        print(_riga("indicatore corretto (vs oggi)", {s: vera[s] - oggi[s] for s in vera if s in oggi}))
        if len(esecuzioni) >= 2:
            a, b = esecuzioni[0], esecuzioni[1]
            comuni = [s for s in a if s in b]
            rumore = sum(abs(a[s] - b[s]) for s in comuni) / len(comuni)
            print(f"  rumore fra le prime due esecuzioni della tesi: {rumore:.1f} celle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
