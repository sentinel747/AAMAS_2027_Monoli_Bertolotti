from __future__ import annotations

"""Esegue piu' seed in parallelo, un processo per run.

Il disegno sperimentale della tesi richiede comunque piu' seed per la stessa
configurazione: senza repliche una traiettoria non e' distinguibile dal rumore.
Finora quelle repliche si eseguivano una dopo l'altra, su una macchina con 16
core logici.

**Perche' questa e' la leva piu' grande disponibile.** Il tetto di Amdahl del port
in Rust della sola fase di decisione e' 3,79x, e il tentativo di port misurato
end-to-end e' stato una regressione. Il parallelismo fra seed non ha tetto di
Amdahl: le run sono completamente indipendenti, quindi il guadagno e' limitato
solo dai core e dalla memoria.

**Ed e' bit-exact per costruzione, non per verifica.** Ogni seed gira in un
processo separato che esegue esattamente cio' che eseguirebbe da solo: stessa
configurazione, stesso `SimulationController`, stessi artifact. Non c'e' stato
condiviso fra le run, quindi non esiste un meccanismo per cui il parallelismo
possa cambiare un risultato. `--verify` lo dimostra comunque confrontando i
digest di stato finali fra esecuzione parallela e sequenziale.

Esempi::

    python scripts/run_sweep.py --run-name tesi_base --seeds 0-4 --agents 300 --steps 500
    python scripts/run_sweep.py --run-name prova --seeds 0 1 2 --steps 60 --verify
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_seeds(tokens: list[str]) -> list[int]:
    """Accetta ``0 1 2`` oppure ``0-4`` oppure una combinazione dei due."""
    seeds: list[int] = []
    for token in tokens:
        token = token.strip()
        if "-" in token and not token.startswith("-"):
            start, _, end = token.partition("-")
            first, last = int(start), int(end)
            if last < first:
                raise ValueError(f"intervallo di seed non valido: {token}")
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(int(token))
    # `dict.fromkeys` invece di `set`: preserva l'ordine richiesto dall'utente,
    # che e' l'ordine in cui compariranno i risultati.
    return list(dict.fromkeys(seeds))


def build_config(options_kwargs: dict, seed: int) -> dict:
    """Config identica a quella che produrrebbe headless_runner per quel seed."""
    from src.experiments.manual_config import ManualConfigOptions, build_manual_config

    kwargs = dict(options_kwargs)
    kwargs["seed"] = seed
    base_name = kwargs.get("run_name") or "sweep"
    kwargs["run_name"] = f"{base_name}_seed{seed}"
    return build_manual_config(ManualConfigOptions(**kwargs))


def run_one(payload: tuple[dict, int, bool]) -> dict:
    """Worker: una run completa in questo processo.

    Deve stare a livello di modulo e non catturare nulla: su Windows i processi
    figli nascono con `spawn` e reimportano il modulo, quindi una chiusura o un
    oggetto non serializzabile non arriverebbe dall'altra parte.
    """
    options_kwargs, seed, want_digest = payload
    from src.api.state_store import SimulationController

    config = build_config(options_kwargs, seed)
    started = time.perf_counter()
    controller = SimulationController(config)
    controller.created_by = "sweep"

    while True:
        controller.step()
        if controller.stop_reason:
            break

    elapsed = time.perf_counter() - started
    result = {
        "seed": seed,
        "run_id": controller.run_id,
        "output_dir": str(controller.output_dir) if controller.output_dir else "",
        "steps": int(controller.step_index),
        "stop_reason": controller.stop_reason,
        "population": len(controller.agents),
        "deaths": len(getattr(controller, "dead_agents", [])),
        "elapsed_s": elapsed,
        "pid": os.getpid(),
    }
    if want_digest:
        # Il digest di stato e' l'unico confronto che dimostra davvero
        # l'identita' fra esecuzione parallela e sequenziale: run_id e cartelle
        # di output contengono timestamp e uuid, quindi differiscono sempre.
        from src.core import state_digest

        result["digest"] = state_digest.combine(state_digest.digest_state(controller.core))
    return result


def _options_kwargs(args: argparse.Namespace) -> dict:
    # Unica definizione della config reale, condivisa con i benchmark: se lo sweep
    # e il profilo per sottostadio girassero su carichi diversi, ogni bersaglio
    # scelto in base al profilo varrebbe per una simulazione che nessuno esegue.
    from scripts.parity_harness import realistic_options

    return realistic_options(args.agents, args.steps, run_name=args.run_name)


def main() -> int:
    # Meta' dei core LOGICI, non "i core meno due". Su questa macchina (16
    # logici, 8 fisici) 8 processi rendono 3,36x: l'efficienza e' gia' al 42%
    # perche' ogni run passa da ~60 s a ~140 s sotto contesa, segno che il limite
    # e' la banda di memoria e non il numero di thread hardware. Spingere fino a
    # 14 processi aggiungerebbe contesa senza aggiungere core fisici.
    default_workers = max(1, (os.cpu_count() or 2) // 2)
    parser = argparse.ArgumentParser(
        description="Esegue piu' seed in parallelo, un processo per run."
    )
    parser.add_argument("--run-name", required=True, help="prefisso del nome run")
    parser.add_argument(
        "--seeds", nargs="+", required=True,
        help="elenco di seed: '0 1 2', oppure un intervallo '0-4', o entrambi",
    )
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--workers", type=int, default=default_workers,
        help=f"processi paralleli (default {default_workers}: meta dei core logici, "
             "misurato come il punto in cui la contesa di memoria satura)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="esegue in un solo processo, per confronto",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="esegue anche in sequenziale e confronta i digest di stato finali",
    )
    parser.add_argument("--json", type=Path, help="scrive il riepilogo come JSON")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if not seeds:
        parser.error("nessun seed indicato")
    options_kwargs = _options_kwargs(args)
    want_digest = args.verify

    workers = 1 if args.sequential else max(1, min(args.workers, len(seeds)))
    print(
        f"sweep '{args.run_name}': {len(seeds)} seed {seeds}, "
        f"{args.agents} agenti x {args.steps} step, {workers} processi "
        f"(core disponibili: {os.cpu_count()})",
        flush=True,
    )

    started = time.perf_counter()
    results: list[dict] = []
    payloads = [(options_kwargs, seed, want_digest) for seed in seeds]

    if workers == 1:
        for payload in payloads:
            result = run_one(payload)
            results.append(result)
            print(
                f"  seed {result['seed']:>3}: {result['steps']} step, "
                f"pop {result['population']}, {result['elapsed_s']:.1f} s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, payload): payload[1] for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"  seed {result['seed']:>3}: {result['steps']} step, "
                    f"pop {result['population']}, {result['elapsed_s']:.1f} s "
                    f"(pid {result['pid']})",
                    flush=True,
                )
    wall = time.perf_counter() - started
    results.sort(key=lambda row: row["seed"])

    cpu_time = sum(row["elapsed_s"] for row in results)
    summary = {
        "run_name": args.run_name,
        "seeds": seeds,
        "agents": args.agents,
        "steps": args.steps,
        "workers": workers,
        "wall_s": wall,
        "cpu_s": cpu_time,
        # ATTENZIONE: questo rapporto SOVRASTIMA il guadagno reale. Sotto contesa
        # ogni run rallenta, quindi il tempo di CPU sommato in parallelo e'
        # maggiore del tempo che le stesse run impiegherebbero da sole. E' un
        # limite superiore utile a colpo d'occhio; il numero onesto e'
        # `speedup_vs_sequential`, che richiede `--verify`.
        "cpu_over_wall_upper_bound": cpu_time / wall if wall > 0 else 0.0,
        "runs": results,
    }

    print(f"\ntempo di parete: {wall:.1f} s")
    print(f"tempo di CPU sommato: {cpu_time:.1f} s")
    print(
        f"limite superiore del guadagno: {summary['cpu_over_wall_upper_bound']:.2f}x "
        f"su {workers} processi (il numero onesto richiede --verify)"
    )

    if args.verify:
        print(
            "\nverifica: rieseguo gli stessi seed in sequenziale "
            "(serve sia alla parita' sia al guadagno reale)...",
            flush=True,
        )
        sequential_started = time.perf_counter()
        sequential = [run_one((options_kwargs, seed, True)) for seed in seeds]
        sequential_wall = time.perf_counter() - sequential_started
        summary["sequential_wall_s"] = sequential_wall
        summary["speedup_vs_sequential"] = sequential_wall / wall if wall > 0 else 0.0
        print(f"  tempo di parete sequenziale: {sequential_wall:.1f} s")
        print(
            f"  GUADAGNO REALE: {summary['speedup_vs_sequential']:.2f}x "
            f"({sequential_wall:.1f} s -> {wall:.1f} s con {workers} processi)"
        )
        mismatches = []
        for parallel_run, serial_run in zip(results, sequential):
            if parallel_run["digest"] != serial_run["digest"]:
                mismatches.append(
                    f"seed {parallel_run['seed']}: parallelo {parallel_run['digest']} "
                    f"contro sequenziale {serial_run['digest']}"
                )
        summary["verified"] = not mismatches
        if mismatches:
            print("DIVERGENZE fra parallelo e sequenziale:")
            for line in mismatches:
                print(f"  {line}")
            return 1
        print(
            f"IDENTICI: {len(seeds)} seed, digest di stato finale invariato fra "
            "esecuzione parallela e sequenziale."
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"riepilogo scritto in {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
