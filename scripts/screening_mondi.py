# -*- coding: utf-8 -*-
"""Screening sui cinque mondi, con il metro con cui la tesi giudica le celle.

Il compagno `screening_metriche.py` guarda il solo mondo di riferimento e giudica
ogni grandezza contro il massimo scarto fra due esecuzioni identiche. E' il metro
piu' severo che i dati permettano, e nessuna delle centoventisette grandezze lo
supera. Questo script usa invece il metro con cui la tesi giudica davvero la
compressione del territorio, cioe' la concordanza sui confronti appaiati di tutta
la campagna dei mondi:

- in quanti mondi su cinque la differenza media ha lo stesso segno;
- in quante delle circa sessanta esecuzioni la differenza appaiata, rispetto alla
  baseline del proprio seme, ha quel segno;
- in quanti semi su venticinque il segno tiene in OGNI esecuzione di quel seme.

Serve a rispondere a una domanda sola: fra le grandezze mai riportate ce n'e'
qualcuna che, con lo stesso criterio delle celle occupate, si sarebbe dovuta
riportare?

Uso:
    python scripts/screening_mondi.py
    python scripts/screening_mondi.py --minimo 50   # soglia sulle esecuzioni concordi
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

MONDI = [
    ("riferimento", "runs/campagna_scarsa/ctrl_none",
     ["runs/campagna_scarsa_v3/llm_completo_amm",
      "runs/campagna_scarsa_v3/llm_completo_amm_rep2",
      "runs/campagna_scarsa_v3/llm_completo_amm_rep3"]),
    ("ricco di ghiaccio", "runs/mondi/ice_rich/ctrl_none",
     ["runs/mondi/ice_rich/llm_completo_amm",
      "runs/mondi/ice_rich/llm_completo_amm_rep2",
      "runs/mondi/ice_rich/llm_completo_amm_rep3"]),
    ("frammentato", "runs/mondi/fragmented/ctrl_none",
     ["runs/mondi/fragmented/llm_completo_amm",
      "runs/mondi/fragmented/llm_completo_amm_rep2"]),
    ("rischio alto", "runs/mondi/high_hazard/ctrl_none",
     ["runs/mondi/high_hazard/llm_completo_amm",
      "runs/mondi/high_hazard/llm_completo_amm_rep2"]),
    ("risorse scarse", "runs/mondi/scarce_resources/ctrl_none",
     ["runs/mondi/scarce_resources/llm_completo_amm_v2",
      "runs/mondi/scarce_resources/llm_completo_amm_v2_rep2"]),
]

ESCLUSE = {
    "run_id", "step", "simulation_step", "day", "simulated_day", "simulated_year",
    "days_per_step", "years_per_step", "initial_agent_count",
    "mars_sol_hours", "mars_year_earth_days", "mars_surface_gravity_m_s2",
    "surface_pressure_pa", "earth_mars_delay_minutes",
    "environmental_layer_enabled", "environmental_layer_mode",
    "automatic_logistics_enabled", "social_graph_active",
    "structure_cache_hit_rate", "structure_cached_cell_count",
    "structure_empty_fast_paths", "structure_invalidated_entries",
    "structure_position_hits", "structure_position_misses",
    "structure_view_hits", "structure_view_misses",
    "best_colony_site_x", "best_colony_site_y",
}


def seme_di(nome: str) -> int:
    return int(nome.rsplit("seed", 1)[1])


def valori(braccio: Path) -> dict[int, dict[str, float]]:
    fuori: dict[int, dict[str, float]] = {}
    for cartella in sorted(braccio.glob("*seed*")):
        f = cartella / "final_metrics.json"
        if not f.exists():
            continue
        v = {k: float(x) for k, x in json.loads(f.read_text(encoding="utf-8")).items()
             if k not in ESCLUSE and not isinstance(x, bool) and isinstance(x, (int, float))}
        m = cartella / "dead_agents.jsonl"
        if m.exists():
            cause: Counter = Counter()
            tot = 0
            for riga in io.open(m, encoding="utf-8", errors="replace"):
                riga = riga.strip()
                if not riga:
                    continue
                try:
                    cause[json.loads(riga).get("cause") or "?"] += 1
                    tot += 1
                except json.JSONDecodeError:
                    pass
            for c, n in cause.items():
                v["morti:" + c] = float(n)
                if tot:
                    v["quota_morti:" + c] = n / tot
        fuori[seme_di(cartella.name)] = v
    return fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minimo", type=int, default=45,
                    help="esecuzioni concordi minime per comparire (le celle ne fanno 50)")
    args = ap.parse_args()

    dati = []
    for nome, base_rel, rip_rel in MONDI:
        base = valori(RADICE / base_rel)
        rip = [valori(RADICE / r) for r in rip_rel]
        dati.append((nome, base, rip))

    chiavi = set.intersection(*[
        set(b[next(iter(b))]) & set(r[0][next(iter(r[0]))])
        for _, b, r in dati if b and r and r[0]
    ])

    righe = []
    for k in sorted(chiavi):
        mondi_segno = []
        esec_pos = esec_tot = 0
        semi_pieni = semi_tot = 0
        ok = True
        for nome, base, rip in dati:
            diff_mondo = []
            for s in sorted(base):
                v = [r[s][k] for r in rip if s in r and k in r[s]]
                if not v or k not in base[s]:
                    continue
                d = [x - base[s][k] for x in v]
                diff_mondo += d
                esec_tot += len(d)
                semi_tot += 1
            if not diff_mondo:
                ok = False
                break
            mondi_segno.append(statistics.fmean(diff_mondo))
        if not ok or not mondi_segno:
            continue
        verso = 1 if statistics.fmean(mondi_segno) > 0 else -1
        if all(abs(m) < 1e-12 for m in mondi_segno):
            continue
        mondi_conc = sum(1 for m in mondi_segno if (m > 0) == (verso > 0))
        # **Le differenze nulle non sono concordanze.** Su una grandezza intera,
        # per esempio il numero di azioni distinte usate, quasi ogni confronto
        # da' zero: contare gli zeri come concordi le faceva risultare 59/59 e
        # mettere in cima allo screening una grandezza che non si muove.
        esec_pos = esec_util = semi_pieni = semi_util = 0
        for nome, base, rip in dati:
            for s in sorted(base):
                v = [r[s][k] for r in rip if s in r and k in r[s]]
                if not v or k not in base[s]:
                    continue
                d = [x - base[s][k] for x in v]
                nz = [x for x in d if x != 0]
                esec_util += len(nz)
                esec_pos += sum(1 for x in nz if (x > 0) == (verso > 0))
                if nz:
                    semi_util += 1
                    if all((x > 0) == (verso > 0) for x in nz):
                        semi_pieni += 1
        if esec_util < esec_tot * 0.8:
            continue    # grandezza quasi sempre ferma: non e' un esito che si muove
        righe.append((mondi_conc, esec_pos, esec_util, semi_pieni, semi_util, k, verso,
                      statistics.fmean(mondi_segno)))

    righe.sort(key=lambda r: (-r[0], -r[1] / r[2]))
    print(f"Screening su {len(righe)} grandezze, cinque mondi, criterio delle celle occupate.")
    print("Riferimento: le celle occupate fanno 5 mondi, 50/59 esecuzioni, 18/25 semi pieni.\n")
    print(f"{'grandezza':42s} {'mondi':>6s} {'esecuzioni':>11s} {'semi pieni':>11s} {'verso':>6s} {'effetto medio':>14s}")
    print("-" * 100)
    for mc, ep, et, sp, st, k, verso, media in righe:
        if ep < args.minimo:
            continue
        print(f"{k[:42]:42s} {mc:3d}/5  {ep:5d}/{et:<5d} {sp:5d}/{st:<5d} "
              f"{'sale' if verso > 0 else 'scende':>6s} {media:14.4g}")
    print("-" * 100)
    print(f"Mostrate le grandezze con almeno {args.minimo} esecuzioni concordi su {et}.")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
