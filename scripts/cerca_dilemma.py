# -*- coding: utf-8 -*-
"""Dove la colonia deve SCEGLIERE: la ricerca di uno scenario con un dilemma.

**Perche' serve.** Un confronto fra bracci ha oggetto solo dove il menu di
lavoro offre un'alternativa reale. Misurato il 2026-08-25: in una crisi da
distruzione ricostruire e' ovviamente giusto e ogni braccio lo fa, quindi una
costituzione che pesa `build x4` non cambia nulla — il peso promuove cio' che
gia' vince. Un pareggio ottenuto li' misurerebbe l'apparato, non il governatore.

**Cosa cerca.** Due proprieta' insieme, e nessuna delle due basta da sola:

- **pressione**: la colonia non ottiene tutto cio' che chiede — azioni
  rifiutate, morti, integrita' che scende. E' il criterio gia' fissato in
  `scripts/valuta_scenari.py`;
- **varianza di esito fra semi**: a parita' di configurazione l'esito cambia.
  Dove tutti i semi finiscono uguali non c'e' spazio perche' una politica
  sposti qualcosa; dove cambiano, quello spazio esiste ed e' misurabile.

La seconda e' il punto. Uno scenario duro ma con esito unico (tutti muoiono,
o tutti sopravvivono allo stesso modo) e' inutile quanto uno inerte.

Uso:
    python scripts/cerca_dilemma.py [--passi 200] [--agenti 120] [--semi 0,1,2]
"""
from __future__ import annotations

import argparse
import statistics
import sys

sys.path.insert(0, ".")

import numpy as np

from src.core import constants as C
from src.world.structures import StructureType

#: Le configurazioni in gara. Ciascuna cambia UNA cosa rispetto alla base,
#: cosi' che l'effetto sia attribuibile: profilo di mappa (dove stanno le
#: risorse), dotazione iniziale (quanto si parte forniti), ISRU (se il
#: materiale si rinnova).
SCENARI: dict[str, dict] = {
    "base": {},
    "mappa scarsa": {"map_profile": "scarce_resources"},
    "mappa a chiazze": {"map_profile": "fragmented"},
    "mappa ricca di minerali": {"map_profile": "mineral_rich"},
    "dotazione dimezzata": {"dotazione": 0.5},
    "dotazione a un quarto": {"dotazione": 0.25},
    "ISRU acceso": {"isru": 0.10},
}


def _misura(agenti: int, passi: int, seme: int, opzioni: dict) -> dict:
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(agenti, passi, seme, **opzioni)
    config["headless"]["progress_every"] = 10**9
    runner = AgentCoupledRunner(config)
    runner.run(days=passi, output_dir=None)

    cells, agents = runner.core.cells, runner.core.agents
    vivi = agents.alive_rows()
    abitate = cells.occupancy > 0
    conteggi = cells.struct_count.astype(np.float64)
    integrita = np.divide(
        cells.struct_integrity, conteggi,
        out=np.ones_like(cells.struct_integrity), where=conteggi > 0,
    )
    con_strutture = conteggi.sum(axis=-1) > 0
    metriche = runner.global_metrics[-1] if runner.global_metrics else {}
    return {
        "pop": int(vivi.size),
        "morti": len(runner.dead_agents),
        "celle": int(abitate.sum()),
        "accettazione": float(metriche.get("acceptance_rate", 1.0)),
        "integrita_min": float(integrita[con_strutture].min()) if con_strutture.any() else 1.0,
        "minerali_in_casa": float(cells.cell_res[abitate, C.R["minerals"]].sum()),
        "minerali_fuori": float(cells.cell_res[..., C.R["minerals"]].sum())
        - float(cells.cell_res[abitate, C.R["minerals"]].sum()),
        "strutture": int(cells.struct_count.sum()),
    }


def _pressione(esiti: list[dict]) -> list[str]:
    """Le condizioni gia' fissate in `valuta_scenari.py`, viste sull'insieme."""
    acceso = []
    if any(e["morti"] > 0 for e in esiti):
        acceso.append("morti")
    if any(e["accettazione"] < 1.0 for e in esiti):
        acceso.append("azioni rifiutate")
    if any(e["integrita_min"] < 0.90 for e in esiti):
        acceso.append("integrita' in calo")
    return acceso


def _varianza(esiti: list[dict]) -> dict[str, float]:
    """Quanto l'esito cambia al variare del solo seme."""
    def spread(chiave: str) -> float:
        valori = [float(e[chiave]) for e in esiti]
        return max(valori) - min(valori)

    return {
        "pop": spread("pop"),
        "celle": spread("celle"),
        "strutture": spread("strutture"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agenti", type=int, default=120)
    p.add_argument("--passi", type=int, default=200)
    p.add_argument("--semi", type=str, default="0,1,2")
    a = p.parse_args()
    semi = [int(s) for s in a.semi.split(",") if s.strip()]

    print(f"Braccio `none`, {a.agenti} coloni, {a.passi} passi, semi {semi}.\n")
    print(f"{'scenario':>24s} {'pop (per seme)':>18s} {'morti':>7s} {'celle':>9s} "
          f"{'strutt':>8s} {'min. in casa':>13s}  pressione")
    riepilogo = {}
    for nome, opzioni in SCENARI.items():
        esiti = [_misura(a.agenti, a.passi, s, opzioni) for s in semi]
        riepilogo[nome] = esiti
        pop = "/".join(str(e["pop"]) for e in esiti)
        celle = "/".join(str(e["celle"]) for e in esiti)
        morti = sum(e["morti"] for e in esiti)
        strut = "/".join(str(e["strutture"]) for e in esiti)
        casa = statistics.mean(e["minerali_in_casa"] for e in esiti)
        print(f"{nome:>24s} {pop:>18s} {morti:>7d} {celle:>9s} {strut:>8s} "
              f"{casa:13.1f}  {', '.join(_pressione(esiti)) or '—'}", flush=True)

    print("\n" + "=" * 78)
    print("VARIANZA DI ESITO FRA SEMI  (spazio in cui una politica puo' muovere)")
    print("=" * 78)
    print(f"  {'scenario':>24s} {'pop':>8s} {'celle':>8s} {'strutture':>11s}")
    for nome, esiti in riepilogo.items():
        v = _varianza(esiti)
        print(f"  {nome:>24s} {v['pop']:8.0f} {v['celle']:8.0f} {v['strutture']:11.0f}")
    print("\n  Uno scenario e' ammissibile per il confronto fra bracci se ha")
    print("  PRESSIONE e VARIANZA insieme: senza la prima nessuna scelta e'")
    print("  sbagliata, senza la seconda nessuna scelta cambia l'esito.")


if __name__ == "__main__":
    main()
