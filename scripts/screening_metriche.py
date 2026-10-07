# -*- coding: utf-8 -*-
"""Screening sistematico: tutto cio' che i registri di esito contengono.

**Perche'.** La tesi riporta tre grandezze di esito su centoquarantanove che
`final_metrics.json` registra a fine run, piu' i conteggi di `action_summary.json`
e le cause di morte di `dead_agents.jsonl`. Le altre non sono state scartate:
non sono mai state guardate. Prima di chiudere il lavoro serve sapere se fra
esse ci sia un effetto del governo linguistico che e' stato perso.

**Come.** Per ogni grandezza numerica si calcola la differenza appaiata per seme
fra LLM (Gov+Amm) e la colonia senza governo sul mondo di riferimento, e la si
legge contro il rumore del modello, misurato dalle tre esecuzioni ripetute dello
stesso braccio sugli stessi semi. Una grandezza passa lo screening se la
differenza media supera quel rumore e conserva il segno su cinque semi su cinque.

**Il problema che questo screening ha, e che va dichiarato.** Con centocinquanta
grandezze e la regola dei cinque segni concordi, il caso da solo produce circa
nove passaggi (150 x 2/32). Il primo stadio quindi non dimostra niente: serve
solo a restringere. Il secondo stadio chiede che l'effetto conservi il segno
anche con gli altri tre modelli, che sono esecuzioni indipendenti con lo stesso
mondo e gli stessi semi. Una grandezza che passa il primo stadio e regge su
quattro modelli su quattro e' un candidato serio; una che passa solo il primo
e' rumore fino a prova contraria.

Uso:
    python scripts/screening_metriche.py
    python scripts/screening_metriche.py --tutte      # stampa anche le bocciate
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

BASELINE = "runs/campagna_scarsa/ctrl_none"
RIPETUTE = [
    "runs/campagna_scarsa_v3/llm_completo_amm",
    "runs/campagna_scarsa_v3/llm_completo_amm_rep2",
    "runs/campagna_scarsa_v3/llm_completo_amm_rep3",
]
ALTRI_MODELLI = [
    ("gpt-oss 120B", "runs/modelli/gptoss120b/llm_completo_amm"),
    ("Qwen3.8 27B", "runs/modelli/qwen38_27b/llm_completo_amm"),
    ("Qwen3.6 27B", "runs/modelli/qwen36_27b/llm_completo_amm"),
]

#: Chiavi che non sono esiti: costanti del pianeta, contatori di cache,
#: identificativi, o parametri di configurazione. Escluderle qui, una volta e
#: dichiarandolo, e' piu' onesto che vederle passare lo screening e spiegare
#: dopo perche' non contano.
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


def metriche(braccio: Path) -> dict[int, dict[str, float]]:
    """seme -> {grandezza: valore}, da final_metrics, action_summary, morti."""
    fuori: dict[int, dict[str, float]] = {}
    for cartella in sorted(braccio.glob("*seed*")):
        f = cartella / "final_metrics.json"
        if not f.exists():
            continue
        v: dict[str, float] = {}
        for k, x in json.loads(f.read_text(encoding="utf-8")).items():
            if k in ESCLUSE or isinstance(x, bool):
                continue
            if isinstance(x, (int, float)):
                v[k] = float(x)

        a = cartella / "action_summary.json"
        if a.exists():
            d = json.loads(a.read_text(encoding="utf-8"))
            for k in ("attempted", "accepted", "rejected", "productive_actions_accepted",
                      "productive_action_rate", "passive_action_rate"):
                if k in d and isinstance(d[k], (int, float)):
                    v["azioni:" + k] = float(d[k])
            tot_ric = 0
            per_motivo: Counter = Counter()
            for azione, dd in (d.get("by_action") or {}).items():
                ric = dd.get("rejected", 0)
                tot_ric += ric
                if ric:
                    v[f"rifiuti:{azione}"] = float(ric)
                    tent = dd.get("attempted", 0)
                    if tent:
                        v[f"quota_rifiuti:{azione}"] = ric / tent
                for motivo, n in (dd.get("rejection_reasons") or {}).items():
                    per_motivo[motivo] += n
            for motivo, n in per_motivo.items():
                v["motivo:" + motivo[:48]] = float(n)

        m = cartella / "dead_agents.jsonl"
        if m.exists():
            cause: Counter = Counter()
            tot = 0
            for riga in io.open(m, encoding="utf-8", errors="replace"):
                riga = riga.strip()
                if not riga:
                    continue
                try:
                    r = json.loads(riga)
                except json.JSONDecodeError:
                    continue
                cause[r.get("cause") or "?"] += 1
                tot += 1
            for c, n in cause.items():
                v["morti:" + c] = float(n)
                if tot:
                    v["quota_morti:" + c] = n / tot

        fuori[seme_di(cartella.name)] = v
    return fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tutte", action="store_true")
    args = ap.parse_args()

    base = metriche(RADICE / BASELINE)
    rip = [metriche(RADICE / r) for r in RIPETUTE]
    altri = [(n, metriche(RADICE / r)) for n, r in ALTRI_MODELLI]

    semi = sorted(set(base) & set(rip[0]))
    chiavi = sorted(set(base[semi[0]]) & set(rip[0][semi[0]]))

    righe = []
    for k in chiavi:
        try:
            eff, rumori = [], []
            for s in semi:
                v = [r[s][k] for r in rip if s in r and k in r[s]]
                if len(v) < 2 or k not in base[s]:
                    raise KeyError
                eff.append(statistics.fmean(v) - base[s][k])
                rumori.append(max(v) - min(v))
        except (KeyError, IndexError):
            continue
        if all(abs(e) < 1e-12 for e in eff):
            continue
        rumore = max(rumori)
        media = statistics.fmean(eff)
        segni = sum(1 for e in eff if e > 0)
        concordi = max(segni, len(eff) - segni)
        passa1 = concordi == len(eff) and abs(media) > rumore

        # secondo stadio: lo stesso segno con gli altri tre modelli
        d_altri = []
        for nome, mm in altri:
            e = [mm[s][k] - base[s][k] for s in semi if s in mm and k in mm[s] and k in base[s]]
            d_altri.append((nome, statistics.fmean(e) if e else None))
        stesso = sum(1 for _, x in d_altri
                     if x is not None and x != 0 and (x > 0) == (media > 0))
        disponibili = sum(1 for _, x in d_altri if x is not None)
        righe.append((passa1, stesso, disponibili, k, media, rumore, concordi, len(eff), d_altri))

    righe.sort(key=lambda r: (-r[0], -r[1], -abs(r[4]) / (r[5] or 1)))
    print(f"Screening su {len(righe)} grandezze. Mondo di riferimento, semi {semi}.")
    print("Rumore = massimo scarto fra due esecuzioni identiche sullo stesso seme.\n")
    print(f"{'grandezza':44s} {'effetto':>12s} {'rumore':>11s} {'segni':>6s}  altri modelli")
    print("-" * 110)
    n1 = 0
    for passa1, stesso, disponibili, k, media, rumore, concordi, n, d_altri in righe:
        if not passa1 and not args.tutte:
            continue
        n1 += 1
        conc = " ".join(f"{nome.split()[0]}:{'+' if x and x > 0 else '-' if x else '0'}"
                        for nome, x in d_altri)
        marchio = "***" if stesso == disponibili else ("*" if stesso >= 2 else "")
        print(f"{k[:44]:44s} {media:12.4g} {rumore:11.4g} {concordi:3d}/{n}  {conc} {marchio}")
    print("-" * 110)
    print(f"{n1} grandezze passano il primo stadio (sopra il rumore, segni concordi).")
    print("*** = tutti e tre gli altri modelli hanno lo stesso segno; * = almeno due.")
    print("Attesa per solo caso al primo stadio: circa "
          f"{len(righe) * 2 / 2 ** len(semi):.0f} grandezze.")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
