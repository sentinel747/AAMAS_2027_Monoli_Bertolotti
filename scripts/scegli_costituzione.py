# -*- coding: utf-8 -*-
"""Quale costituzione, sui dati: lo sweep che sceglie `REFERENCE_RULES`.

La costituzione del braccio `scripted` — che e' anche il mandato che il prompt
del braccio `llm` riporta parola per parola — non si sceglie a tavolino. Questo
script la sceglie confrontando candidate contro la baseline **appaiate per
seme**, perche' la varianza fra semi domina largamente l'effetto atteso di una
politica: una media presa fra semi diversi misurerebbe il seme.

**Va rieseguito ogni volta che il mondo cambia.** La costituzione in vigore fino
al 2026-08-25 era stata scelta su un'economia che non conservava la massa; dopo
la ritaratura non solo era obsoleta, ma in crisi puntava nella direzione
sbagliata — mandava a foraggiare un magazzino vuoto.

Uso:
    python scripts/scegli_costituzione.py [--passi 200] [--agenti 120]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

import numpy as np

from src.core import constants as C

#: Le candidate, ciascuna un'ipotesi su cosa una colonia dovrebbe fare quando
#: una statistica pubblica esce dalla banda. Sono scritte nello stile in cui un
#: governo scrive: una condizione su una statistica, una priorita' che cambia.
CANDIDATE: dict[str, tuple] = {
    "attuale (cibo + affollamento)": (
        ("food_per_occupant", "<", 2.0, "sustenance", 3.0),
        ("occupants", ">", 150.0, "explore", 2.0),
    ),
    "corrente sotto il fabbisogno": (
        ("power_coverage", "<", 1.0, "build", 3.0),
    ),
    "manutenzione arretrata": (
        ("structure_integrity", "<", 0.70, "build", 2.5),
    ),
    "riserva idrica bassa": (
        ("water_per_occupant", "<", 1.0, "sustenance", 2.5),
    ),
    "giacimento esaurito -> espandere": (
        ("minerals_per_occupant", "<", 1.0, "explore", 2.5),
    ),
    "corrente + giacimento": (
        ("power_coverage", "<", 1.0, "build", 3.0),
        ("minerals_per_occupant", "<", 1.0, "explore", 2.5),
    ),
    # --- le candidate a VETTORE di pesi (2026-09-02) -----------------------
    # La regola che scatta porta con se' anche cio' che va protetto. Misurate
    # su tre semi a 400 passi: la prima riporta l'espansione da 3/1/1 celle a
    # 44/101/97 cambiando SOLO i pesi rispetto a "corrente + giacimento".
    "corrente + giacimento, explore protetto": (
        ("power_coverage", "<", 1.0, {"build": 3.0, "explore": 3.0}),
        ("minerals_per_occupant", "<", 1.0, {"explore": 2.5}),
    ),
    "corrente, con explore e cibo protetti": (
        ("power_coverage", "<", 1.0, {"build": 3.0, "explore": 3.0, "sustenance": 2.0}),
    ),
    "manutenzione, con explore protetto": (
        ("structure_integrity", "<", 0.70, {"build": 2.5, "explore": 2.5}),
        ("minerals_per_occupant", "<", 1.0, {"explore": 2.5}),
    ),
    "acqua, con explore protetto": (
        ("water_per_occupant", "<", 1.0, {"sustenance": 2.5, "explore": 2.0}),
        ("minerals_per_occupant", "<", 1.0, {"explore": 2.5}),
    ),
}


def _pesi(voce) -> dict:
    """I pesi di una regola, dal formato corto o da quello a vettore.

    `(..., "build", 3.0)` vale `{"build": 3.0}`; `(..., {"build": 3.0,
    "explore": 3.0})` porta il vettore. Il formato corto resta perche' le
    candidate storiche sono scritte cosi' e vanno confrontate senza riscriverle.
    """
    coda = voce[3]
    return dict(coda) if isinstance(coda, dict) else {coda: voce[4]}


def _policy(regole):
    from src.governors.policy import Condition, PILLAR_BY_NAME, Policy, Rule

    per_cond, ordine = {}, []
    for voce in regole:
        ind, op, soglia = voce[0], voce[1], voce[2]
        chiave = (ind, op, soglia)
        if chiave not in per_cond:
            per_cond[chiave] = {}
            ordine.append(chiave)
        for pilastro, peso in _pesi(voce).items():
            per_cond[chiave][PILLAR_BY_NAME[pilastro]] = peso
    return Policy(
        rules=tuple(Rule(condition=Condition(*k), weights=per_cond[k]) for k in ordine),
        rationale="candidata",
    )


def _run(agenti: int, passi: int, seme: int, regole, opzioni: dict) -> dict:
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(agenti, passi, seme, **opzioni)
    config["headless"]["progress_every"] = 10**9
    gov = config.setdefault("governors", {})
    gov["arm"] = "none" if regole is None else "scripted"
    gov["enabled"] = regole is not None
    gov["cadence_steps"] = 10

    import src.governors.arms as arms
    originale = arms.reference_policy
    if regole is not None:
        arms.reference_policy = lambda bounds: _policy(regole)
    try:
        runner = AgentCoupledRunner(config)
        runner.run(days=passi, output_dir=None)
    finally:
        arms.reference_policy = originale

    cells, agents = runner.core.cells, runner.core.agents
    vivi = agents.alive_rows()
    colpi = dict(runner.core.governor_policy_hits or {})
    scattate = sum(v for k, v in colpi.items() if not k.startswith("- nessuna"))
    inerti = colpi.get("- nessuna regola (preferenze pure)", 0)
    # Il conteggio PER REGOLA, nell'ordine in cui sono scritte: e' cio' che
    # distingue una costituzione che discrimina da una che e' un moltiplicatore
    # globale travestito. Le chiavi sono il testo della regola, quindi si
    # ricostruiscono dalle regole stesse e non dall'ordine del dizionario.
    per_regola = []
    if regole is not None:
        from src.governors.policy import rule_text

        for regola in _policy(regole).rules:
            per_regola.append(int(colpi.get(rule_text(regola), 0)))
    totale = sum(per_regola) + int(inerti)
    return {
        "pop": int(vivi.size),
        "morti": len(runner.dead_agents),
        "celle": int((cells.occupancy > 0).sum()),
        "scattate": int(scattate),
        "inerti": int(inerti),
        "per_regola": per_regola,
        "totale_celle_passo": int(totale),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agenti", type=int, default=120)
    p.add_argument("--passi", type=int, default=200)
    p.add_argument("--semi", type=str, default="0,1,2")
    p.add_argument("--isru", type=float, default=0.10,
                   help="scenario ammesso dal criterio di `cerca_dilemma.py`")
    a = p.parse_args()
    semi = [int(s) for s in a.semi.split(",") if s.strip()]
    opzioni = {"isru": a.isru} if a.isru else {}

    base = [_run(a.agenti, a.passi, s, None, opzioni) for s in semi]
    print(f"Scenario ISRU={a.isru}, {a.agenti} coloni, {a.passi} passi, semi {semi}.")
    print(f"Baseline: pop {'/'.join(str(b['pop']) for b in base)}, "
          f"celle {'/'.join(str(b['celle']) for b in base)}, "
          f"morti {sum(b['morti'] for b in base)}\n")

    print(f"{'costituzione':>34s} {'pop (delta)':>16s} {'celle (delta)':>16s} "
          f"{'scatti':>8s} {'concordanza':>12s} {'discrimina':>12s} {'ammessa':>9s}")
    for nome, regole in CANDIDATE.items():
        esiti = [_run(a.agenti, a.passi, s, regole, opzioni) for s in semi]
        dpop = [e["pop"] - b["pop"] for e, b in zip(esiti, base)]
        dcelle = [e["celle"] - b["celle"] for e, b in zip(esiti, base)]
        scatti = sum(e["scattate"] for e in esiti)
        # Concordanza di segno: l'unica lettura difendibile con pochi semi.
        segni = [np.sign(d) for d in dpop if d != 0]
        concordanza = (
            "—" if not segni
            else ("tutti +" if all(s > 0 for s in segni)
                  else "tutti -" if all(s < 0 for s in segni) else "discorde")
        )
        # Discriminazione: sommata sui semi, ogni regola deve prendere
        # qualcosa e nessuna deve prendere tutto.
        quante = len(esiti[0]["per_regola"])
        somme = [sum(e["per_regola"][i] for e in esiti) for i in range(quante)]
        totale = sum(e["totale_celle_passo"] for e in esiti) or 1
        in_ombra = sum(1 for v in somme if v == 0)
        costante = any(v >= 0.995 * totale for v in somme)
        if not somme:
            discrimina = "—"
        elif costante:
            discrimina = "COSTANTE"
        elif in_ombra:
            discrimina = f"{in_ombra} in ombra"
        else:
            discrimina = "si"
        ammessa = (
            "no" if (scatti == 0 or costante or in_ombra or concordanza == "tutti -")
            else "SI"
        )
        print(f"{nome:>34s} {'/'.join(f'{d:+d}' for d in dpop):>16s} "
              f"{'/'.join(f'{d:+d}' for d in dcelle):>16s} {scatti:8d} {concordanza:>12s} "
              f"{discrimina:>12s} {ammessa:>9s}",
              flush=True)
        if somme:
            quota = "  ".join(f"r{i+1}={100*v/totale:.0f}%" for i, v in enumerate(somme))
            print(f"{'':>34s}   {quota}")

    print("\n  `scatti` = quante volte una regola ha catturato una cella. Zero")
    print("  significa costituzione inerte, e va letto PRIMA dei delta: una")
    print("  candidata che non scatta non ha perso il confronto, non l'ha fatto.")
    print()
    print("  `discrimina` guarda la ripartizione fra le regole, non il totale.")
    print("  COSTANTE = una regola prende praticamente tutte le celle-passo: la")
    print("  sua condizione e' vera ovunque, quindi non e' una condizione ma un")
    print("  moltiplicatore globale, e su pesi relativi cio' che non nomina lo")
    print("  penalizza. `in ombra` = una regola non scatta mai perche' una")
    print("  precedente le porta via tutte le celle: e' scritta e non applicata.")
    print()
    print("  `ammessa` = scatta, discrimina, e non peggiora su tutti i semi. Un")
    print("  braccio di controllo competente puo' perdere, ma se perde sempre e'")
    print("  un handicap, e il confronto fra bracci misurerebbe quello.")


if __name__ == "__main__":
    main()
