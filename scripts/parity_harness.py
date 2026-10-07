from __future__ import annotations

"""Harness di parita': esegue una config golden e registra un digest per step.

Serve al Task 0 del piano di migrazione Rust
(``docs/superpowers/plans/2026-08-02-rust-kernel-migration.md``). Due usi:

1. **Passo bloccante 0.0** - verificare che due *processi distinti*, stesso commit
   e stesso seed, producano la stessa sequenza di digest. Finche' questo non e'
   vero non esiste un oracolo e ogni confronto Python/Rust misura rumore.
2. **Confronto fra backend** - la stessa sequenza, prodotta da backend diversi,
   individua lo step e il campo della prima divergenza.

L'harness non modifica il simulatore: intercetta ``kernel.step`` con un wrapper
che calcola il digest dopo ogni chiamata. Il comportamento della run resta quello
del backend sotto misura.

Esempi::

    python scripts/parity_harness.py --seed 0 --steps 20 --out a.jsonl
    python scripts/parity_harness.py --seed 0 --steps 20 --out b.jsonl
    python scripts/parity_harness.py --compare a.jsonl b.jsonl
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.role_profiles import DEFAULT_ROLE_DISTRIBUTION  # noqa: E402
from src.core import state_digest  # noqa: E402

# Config golden della migrazione. Deliberatamente piccola e con ambiente,
# crescita di popolazione e artifact disattivati: il digest deve misurare il
# kernel, non le sorgenti di variabilita' che gli stanno intorno.
GOLDEN_SEEDS = (0, 1, 2)


def golden_config(agents: int, steps: int, seed: int, sampling: str = "softmax") -> dict:
    return {
        "name": f"parity_seed{seed}",
        "seed": seed,
        "days": steps,
        "simulation": {"days_per_step": 1, "max_days": steps},
        "world": {"width": 36, "height": 24, "map_profile": "balanced"},
        "agents": {
            "count": agents,
            "llm_count": 0,
            "decision_mode": "preferences",
            "decision_sampling": sampling,
            "role_distribution": dict(DEFAULT_ROLE_DISTRIBUTION),
            "role_preference_randomness": 0.25,
            "initial_inventory": {
                "food": 10, "water": 10, "construction_material": 8, "tools": 2,
                "oxygen": 4, "energy": 2, "minerals": 4, "med_kits": 1,
            },
        },
        "colony": {
            "initial_structures": {
                "habitat": 1, "greenhouse": 1, "solar_array": 1, "oxygen_plant": 1,
            }
        },
        "population": {"enabled": False},
        "environmental_layer": {"enabled": False},
        "model": {"psychosocial_enabled": True},
        "snapshot_interval": 0,
        "headless": {
            "fast_observation": True,
            "store_memory_logs": False,
            "log_interval_steps": steps + 1,
            "max_action_log_rows": 0,
        },
    }


def realistic_options(
    agents: int,
    steps: int,
    run_name: str = "bench",
    map_profile: str = "balanced",
    dotazione: float = 1.0,
    isru: float = 0.0,
    sviluppo: float = 0.30,
) -> dict:
    """Kwargs di ``ManualConfigOptions`` per la configurazione REALE.

    La golden sopra e' un oracolo, non un carico rappresentativo: ha 4 strutture,
    ambiente e popolazione spenti e osservazione veloce. Ogni percentuale presa
    su di essa sovrastima la fase di decisione, perche' l'osservazione completa e
    la redistribuzione — che nella golden non esistono — spostano altrove il
    costo. Questa funzione produce invece la stessa configurazione che generano
    GUI, wizard e CLI, cioe' quella su cui valgono le conclusioni.

    Restituisce i *kwargs* e non la config gia' costruita perche' ``run_sweep``
    deve poterli serializzare verso processi figli, dove la config si costruisce
    dopo il fork/spawn.
    """
    # I default di ManualConfigOptions sono tarati su 50 agenti; la regola di
    # scaling dei totali di colonia sta in docs/MARS_ABM_NOTE_CONFIGURAZIONE.md.
    scale = max(1, agents // 50)
    # `dotazione` scala la dotazione iniziale di strutture. A 1,0 non tocca
    # niente, quindi la baseline bit-exact resta tale. Serve a creare per
    # configurazione la scarsita' che i profili di mappa non producono: vedi
    # docs/benchmarks/2026-08-21-scenari-esito.md. Il moltiplicatore agisce anche
    # sui DEPOSITI, ed e' li' che morde davvero — i depositi fabbricano
    # `construction_material` nella cella a ogni passo (kernel_biology) e la
    # colonia non ne costruisce mai di nuovi (`build_storage_depot` non e' fra le
    # azioni), quindi quanti se ne danno all'inizio e' quanti ne avra' per sempre.
    def dotata(base: int) -> int:
        return max(0, round(base * dotazione))

    return {
        "run_name": run_name,
        # "balanced" e' il default di ManualConfigOptions: passarlo esplicitamente
        # non cambia la config, quindi la baseline bit-exact resta tale.
        "map_profile": map_profile,
        # Zero = spenta, cioe' il comportamento di sempre.
        "isru_material_rate": isru,
        "development_build_priority": sviluppo,
        "mode": "rule",
        "agent_count": agents,
        "llm_count": 0,
        "max_steps": steps,
        # L'inventario iniziale e' PER AGENTE, ed e' li' che stanno davvero le
        # scorte: a 300 agenti, `initial_materials = 5` fa 1500 unita' di
        # `construction_material`, mentre i dodici depositi ne fabbricano ~64 in
        # ottocento passi. Scalare le sole strutture lasciava quindi intatto il
        # 96% della scorta, ed e' la ragione per cui a dotazione 0,1
        # `material_margin` restava a 1,96.
        "initial_food": 5 * dotazione,
        "initial_water": 5 * dotazione,
        "initial_materials": 5 * dotazione,
        "initial_tools": 1 * dotazione,
        "initial_habitats": dotata(10 * scale),
        "initial_greenhouses": dotata(7 * scale),
        "initial_solar_arrays": dotata(5 * scale),
        "initial_oxygen_plants": dotata(5 * scale),
        "initial_storage_depots": dotata(2 * scale),
        "initial_weather_stations": dotata(1 * scale),
    }


def realistic_config(
    agents: int,
    steps: int,
    seed: int,
    sampling: str = "softmax",
    map_profile: str = "balanced",
    dotazione: float = 1.0,
    isru: float | None = None,
    sviluppo: float = 0.30,
) -> dict:
    """Config reale pronta per il runner, con gli artifact tenuti in memoria.

    **`isru=None` significa «il default del prodotto» (2026-08-31).** Prima era
    scritto `0.0` qui dentro mentre `ManualConfigOptions` lo tiene a 1,0: questa
    funzione si dichiara «config reale» ed e' la sorgente di TUTTI i benchmark,
    dell'audit di conservazione della massa e dell'arnese di parita', quindi
    ogni verifica di questo progetto girava su una configurazione che nessuna
    run reale usa — e proprio sul ramo che decide se la colonia sopravvive.
    Due default per una quantita' sola: il secondo si legge dal primo.
    """
    from src.experiments.manual_config import ManualConfigOptions, build_manual_config

    if isru is None:
        isru = float(ManualConfigOptions.isru_material_rate)

    kwargs = realistic_options(
        agents,
        steps,
        run_name=f"bench_seed{seed}",
        map_profile=map_profile,
        dotazione=dotazione,
        isru=isru,
        sviluppo=sviluppo,
    )
    kwargs["seed"] = seed
    kwargs["decision_sampling"] = sampling
    config = build_manual_config(ManualConfigOptions(**kwargs))
    # Il layer ambientale richiede il runtime Fortran del MCD, che su questa
    # macchina non e' disponibile: resta l'unica differenza dichiarata rispetto a
    # una run di tesi, e vale ~3% del passo (misurato in docs/benchmarks).
    config.setdefault("environmental_layer", {})["enabled"] = False
    config["days"] = steps
    return config


def build_config(kind: str, agents: int, steps: int, seed: int, sampling: str) -> dict:
    if kind == "realistic":
        return realistic_config(agents, steps, seed, sampling)
    return golden_config(agents, steps, seed, sampling)


def run_digests(
    agents: int, steps: int, seed: int, sampling: str, config_kind: str = "golden"
) -> list[dict]:
    """Esegue la run e restituisce un record di digest per step."""
    import src.simulation.agent_coupled_runner as runner_module
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    records: list[dict] = []
    inner_step = runner_module._core_step

    def recording_step(state, step_index, dt_days, config, rng):
        outcome = inner_step(state, step_index, dt_days, config, rng)
        record = state_digest.step_digest(state, outcome)
        record["step"] = int(step_index)
        records.append(record)
        return outcome

    runner_module._core_step = recording_step
    try:
        runner = AgentCoupledRunner(build_config(config_kind, agents, steps, seed, sampling))
        runner.run(days=steps, output_dir=None)
    finally:
        runner_module._core_step = inner_step
    return records


def write_digests(path: Path, records: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"meta": meta}, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_digests(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "meta" in payload and "step" not in payload:
                meta = payload["meta"]
            else:
                records.append(payload)
    return meta, records


def compare(left_path: Path, right_path: Path) -> int:
    """Confronta due file di digest. Ritorna 0 se identici, 1 altrimenti."""
    left_meta, left = read_digests(left_path)
    right_meta, right = read_digests(right_path)

    if len(left) != len(right):
        print(
            f"DIVERGE: numero di step diverso ({len(left)} contro {len(right)}). "
            "Una run si e' fermata prima: estinzione o eccezione in un solo backend."
        )
        return 1

    for lhs, rhs in zip(left, right):
        if lhs["overall"] == rhs["overall"]:
            continue
        step = lhs["step"]
        divergence = state_digest.first_divergence(lhs["fields"], rhs["fields"])
        print(f"DIVERGE allo step {step}")
        if divergence is None:
            print(
                "  digest complessivo diverso ma nessun campo diverge: "
                "incoerenza dello schema di digest, non della simulazione."
            )
        else:
            name, lval, rval = divergence
            print(f"  primo campo divergente: {name}")
            print(f"    {left_path.name}: {lval}")
            print(f"    {right_path.name}: {rval}")
            diverging = [
                key
                for key in sorted(set(lhs["fields"]) | set(rhs["fields"]))
                if lhs["fields"].get(key) != rhs["fields"].get(key)
            ]
            print(f"  campi divergenti a questo step: {len(diverging)}")
            for key in diverging[:12]:
                print(f"    - {key}")
            if len(diverging) > 12:
                print(f"    ... e altri {len(diverging) - 12}")
        print(f"  meta sinistra: {left_meta}")
        print(f"  meta destra:   {right_meta}")
        return 1

    print(f"IDENTICI: {len(left)} step, digest finale {left[-1]['overall'] if left else 'n/d'}")
    return 0


def self_check(
    agents: int, steps: int, seeds: tuple[int, ...], sampling: str,
    config_kind: str = "golden",
) -> int:
    """Passo 0.0: due processi figli per seed, confronto delle sequenze.

    Deve girare in sottoprocessi e non in-process: il non-determinismo che questo
    passo cerca (ordine di iterazione dipendente da ``PYTHONHASHSEED``, indirizzi
    di oggetti nei tie-break) si manifesta *fra* interpreti, non dentro lo stesso.
    """
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for seed in seeds:
            paths = []
            for replica in ("a", "b"):
                out = tmp_path / f"seed{seed}_{replica}.jsonl"
                env = dict(os.environ)
                # Hash seed DIVERSI di proposito: se un digest dipende
                # dall'ordinamento di un set/dict su chiavi non ordinate, questo
                # e' cio' che lo rivela.
                env["PYTHONHASHSEED"] = "0" if replica == "a" else "12345"
                proc = subprocess.run(
                    [
                        sys.executable, str(Path(__file__).resolve()),
                        "--seed", str(seed), "--steps", str(steps),
                        "--agents", str(agents), "--sampling", sampling,
                        "--config", config_kind,
                        "--out", str(out), "--quiet",
                    ],
                    env=env, capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    print(f"seed {seed} replica {replica}: run fallita")
                    print(proc.stdout[-2000:])
                    print(proc.stderr[-2000:])
                    return 1
                paths.append(out)
            print(f"--- seed {seed} (PYTHONHASHSEED 0 contro 12345) ---")
            failures += compare(paths[0], paths[1])
    if failures:
        print(f"\nDETERMINISMO CROSS-PROCESSO: FALLITO su {failures} seed")
    else:
        print("\nDETERMINISMO CROSS-PROCESSO: OK su tutti i seed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harness di parita': digest per step di una config golden."
    )
    parser.add_argument("--agents", type=int, default=60)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("softmax", "greedy"), default="softmax")
    parser.add_argument(
        "--config", choices=("golden", "realistic"), default="golden",
        help="carico su cui verificare la parita'; 'realistic' e' l'unico che "
             "esercita espansione, redistribuzione e osservazione completa",
    )
    parser.add_argument("--out", type=Path, help="file JSONL di destinazione")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--compare", nargs=2, type=Path, metavar=("A", "B"),
        help="confronta due file di digest gia' prodotti",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="passo 0.0: verifica il determinismo fra processi distinti",
    )
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])

    if args.self_check:
        return self_check(args.agents, args.steps, GOLDEN_SEEDS, args.sampling, args.config)

    if args.quiet:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    records = run_digests(args.agents, args.steps, args.seed, args.sampling, args.config)
    if args.quiet:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    meta = {
        "config": args.config,
        "agents": args.agents,
        "steps": args.steps,
        "seed": args.seed,
        "sampling": args.sampling,
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "digest_schema": state_digest.SCHEMA_VERSION,
    }
    if args.out:
        write_digests(args.out, records, meta)
        print(f"scritti {len(records)} step in {args.out}")
    else:
        for record in records:
            print(f"step {record['step']:>5}  {record['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
