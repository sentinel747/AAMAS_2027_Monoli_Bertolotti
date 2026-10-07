# -*- coding: utf-8 -*-
"""Riesegue le run del 26-27 agosto per verificare la correzione della manutenzione.

**Perche' esiste.** Sette run rule-based da 200 coloni si sono estinte (sei su
sette) per disidratazione fra il passo 99 e il 140, senza che una sola
manutenzione venisse mai proposta: l'integrita' delle strutture decadeva
linearmente, la produzione d'acqua scala con l'integrita', e il bisogno di
manutenzione restava zero finche' non era troppo tardi.

Questo script NON ricostruisce una configurazione somigliante: carica quella
davvero salvata dalla run originale (`config.yaml` nella cartella della run),
cambia il solo numero di passi e la rilancia. E' l'unico modo perche' il
confronto prima/dopo sia un confronto e non un aneddoto.

Uso:
    python scripts/verifica_manutenzione.py [--passi 300] [--run 260826_test1_seed0 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

RADICE = Path("outputs/runs")

#: Le sei run estinte piu' la sopravvissuta. L'ordine e' quello dei semi.
PREDEFINITE = (
    "260826_test1_seed0",
    "260826_test3_seed2",
    "270826_test6_seed5",
)


def _carica_config(nome: str, passi: int) -> dict:
    percorso = RADICE / nome / "config.yaml"
    config = json.loads(percorso.read_text(encoding="utf-8"))
    config["days"] = passi
    config["name"] = f"verifica_{nome}"
    # Silenzia la barra di avanzamento: qui interessa solo l'esito.
    config.setdefault("headless", {})["progress_every"] = 10**9
    return config


def _esegui(config: dict, passi: int) -> dict:
    from src.agents import pillars
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    runner = AgentCoupledRunner(config)
    runner.run(days=passi, output_dir=None)

    core = runner.core
    vivi = core.agents.alive_rows()
    conteggi = core.cells.struct_count.astype(float)
    totale = conteggi.sum()
    integrita = core.cells.struct_integrity.sum() / totale if totale > 0 else 1.0

    # Le manutenzioni eseguite, dal contatore d'azione indipendente dai log.
    eseguite = getattr(runner, "action_counts", {}) or {}
    manutenzioni = int(eseguite.get("maintain_structure", 0))
    rifiuti = getattr(runner, "rejection_counts", {}) or {}
    respinte = sum(
        int(v) for k, v in rifiuti.items() if "maintain" in str(k)
    ) if isinstance(rifiuti, dict) else 0

    return {
        "vivi": int(vivi.size),
        "morti": len(runner.dead_agents),
        "integrita": float(integrita),
        "strutture": int(totale),
        "manutenzioni": manutenzioni,
        "respinte": respinte,
        "passi_fatti": int(getattr(core, "step_index", passi)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--passi", type=int, default=300,
                   help="l'estinzione originale cadeva fra 99 e 140: 300 la supera con margine")
    p.add_argument("--run", nargs="*", default=list(PREDEFINITE))
    a = p.parse_args()

    print(f"Riesecuzione con la configurazione ORIGINALE, {a.passi} passi.\n")
    print(f"{'run':24s} {'vivi':>6s} {'morti':>6s} {'integrita':>10s} "
          f"{'strutt':>7s} {'manut.':>7s} {'respinte':>9s}")
    for nome in a.run:
        esito = _esegui(_carica_config(nome, a.passi), a.passi)
        print(f"{nome:24s} {esito['vivi']:6d} {esito['morti']:6d} "
              f"{esito['integrita']:10.4f} {esito['strutture']:7d} "
              f"{esito['manutenzioni']:7d} {esito['respinte']:9d}", flush=True)

    print("\n  Prima della correzione, alle stesse configurazioni: 200 morti su 200,")
    print("  integrita' 0,547 e ZERO manutenzioni, con la run interrotta per")
    print("  estinzione fra il passo 99 e il 140.")


if __name__ == "__main__":
    main()
