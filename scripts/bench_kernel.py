from __future__ import annotations

"""Benchmark riproducibile del kernel, con repliche e ripartizione per sottostadio.

Implementa il protocollo del Task 0 del piano di migrazione Rust
(``docs/superpowers/plans/2026-08-02-rust-kernel-migration.md``, sezione 8):
warm-up separato, almeno cinque repliche, mediana e p95, contesto della macchina,
memoria di picco e tempo per sottostadio.

**Perche' non ``cProfile``.** La revisione 2 del piano (punto R2.2) ha misurato che
il profiler deterministico aggiunge un costo per-chiamata che gonfia di circa 2,4x
la run e, cosa peggiore, la gonfia *in modo non uniforme*: le funzioni con
centinaia di migliaia di chiamate risultano molto piu' pesanti di quanto siano in
una run reale, e sono proprio quelle che il piano promuoveva a bersaglio primario.
Qui la ripartizione si ottiene avvolgendo i punti di ingresso dei sottostadi con
``perf_counter``: poche decine di misure per step invece di decine di milioni, con
un overhead trascurabile e non correlato al numero di chiamate interne.

Esempi::

    python scripts/bench_kernel.py --agents 300 --steps 30 --replicas 5
    python scripts/bench_kernel.py --agents 300 --steps 30 --stages
    python scripts/bench_kernel.py --agents 300 --steps 30 --json out.json
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.parity_harness import build_config  # noqa: E402

# Nome del sottostadio -> attributo del modulo `src.core.kernel` da avvolgere.
# I nomi sono quelli importati NEL namespace di kernel.py: `step()` li risolve
# come globali del modulo, quindi riassegnarli li' intercetta le chiamate senza
# toccare il codice del kernel ne' i moduli di definizione.
STAGE_TARGETS = {
    "osservazione": ("observe", "observe_fast", "observe_rule_based"),
    "maschere_celle": ("compute_cell_masks", "compute_cell_proposals"),
    "decisione": ("decide_batch",),
    "azioni": ("execute_action",),
    "vitali": ("tick_vitals", "refresh_cell_alerts"),
    "biologia": ("update_cells",),
    "redistribuzione": ("compute_needs", "redistribute"),
}


@contextmanager
def stage_instrumentation(totals: dict[str, float], counts: dict[str, int]):
    """Avvolge i sottostadi del kernel con timer ``perf_counter``.

    Ripristina sempre gli attributi originali all'uscita: un benchmark che lascia
    il modulo strumentato falsificherebbe ogni misura successiva nello stesso
    processo.
    """
    import src.core.kernel as kernel

    originals: dict[str, object] = {}

    def wrap(stage: str, func):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                totals[stage] += time.perf_counter() - start
                counts[stage] += 1
        return timed

    for stage, names in STAGE_TARGETS.items():
        totals.setdefault(stage, 0.0)
        counts.setdefault(stage, 0)
        for name in names:
            target = getattr(kernel, name, None)
            if target is None:
                continue
            originals[name] = target
            setattr(kernel, name, wrap(stage, target))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(kernel, name, original)


def _one_run(agents: int, steps: int, seed: int, sampling: str, config_kind: str) -> float:
    """Una replica: costruisce il runner fuori dalla misura, cronometra il loop."""
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    runner = AgentCoupledRunner(build_config(config_kind, agents, steps, seed, sampling))
    start = time.perf_counter()
    runner.run(days=steps, output_dir=None)
    return time.perf_counter() - start


def _machine_context() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:  # pragma: no cover - contesto diagnostico, non logica
        commit = "sconosciuto"
    return {
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "sconosciuto",
        "machine": platform.machine(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark del kernel con repliche e ripartizione per sottostadio."
    )
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("softmax", "greedy"), default="softmax")
    parser.add_argument(
        "--config", choices=("golden", "realistic"), default="golden",
        help="golden = oracolo minimo del Task 0; realistic = ManualConfigOptions, "
             "la stessa config di GUI/wizard/CLI. Le percentuali per sottostadio "
             "vanno lette solo su 'realistic'",
    )
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="repliche scartate prima di misurare (import, cache, JIT del SO)",
    )
    parser.add_argument(
        "--stages", action="store_true",
        help="misura anche la ripartizione per sottostadio (replica separata)",
    )
    parser.add_argument(
        "--memory", action="store_true",
        help="misura la memoria di picco con tracemalloc (replica separata: rallenta)",
    )
    parser.add_argument("--json", type=Path, help="scrive il risultato come JSON")
    parser.add_argument(
        "--label", default="", help="etichetta del braccio, es. 'python-post-0b'"
    )
    args = parser.parse_args()

    print(
        f"warm-up: {args.warmup} replica/e (config {args.config}: "
        f"{args.agents} agenti, {args.steps} step, seed {args.seed})",
        flush=True,
    )
    for _ in range(args.warmup):
        _one_run(args.agents, args.steps, args.seed, args.sampling, args.config)

    timings: list[float] = []
    for replica in range(args.replicas):
        elapsed = _one_run(args.agents, args.steps, args.seed, args.sampling, args.config)
        timings.append(elapsed)
        print(f"  replica {replica + 1}/{args.replicas}: {elapsed:.3f} s", flush=True)

    agent_steps = args.agents * args.steps
    median = statistics.median(timings)
    # p95 su cinque repliche coincide con il massimo: e' dichiarato cosi'
    # esplicitamente invece di far credere a una stima piu' fine di quella che i
    # dati permettono.
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    result = {
        "label": args.label or "python",
        "config": {
            "kind": args.config,
            "agents": args.agents, "steps": args.steps, "seed": args.seed,
            "sampling": args.sampling,
            # La golden spegne ambiente e popolazione e osserva in modo veloce;
            # la realistic li accende. Dichiararlo nel JSON evita di confrontare
            # per sbaglio numeri presi su carichi diversi.
            "artifacts": False,
            "observation": "fast" if args.config == "golden" else "full",
        },
        "replicas": args.replicas,
        "warmup": args.warmup,
        "timings_s": timings,
        "median_s": median,
        "p95_s": ordered[p95_index],
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "spread_pct": (ordered[-1] - ordered[0]) / median * 100.0 if median else 0.0,
        "median_ms_per_step": median / args.steps * 1000.0,
        "agent_steps_per_s": agent_steps / median if median else 0.0,
        "machine": _machine_context(),
    }

    if args.stages:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        print("misura per sottostadio (replica dedicata)...", flush=True)
        with stage_instrumentation(totals, counts):
            instrumented = _one_run(args.agents, args.steps, args.seed, args.sampling, args.config)
        accounted = sum(totals.values())
        result["stages"] = {
            "instrumented_total_s": instrumented,
            "accounted_s": accounted,
            "unaccounted_s": instrumented - accounted,
            "overhead_vs_median_pct": (instrumented - median) / median * 100.0
            if median else 0.0,
            "per_stage": {
                stage: {
                    "seconds": seconds,
                    "pct_of_step": seconds / instrumented * 100.0 if instrumented else 0.0,
                    "calls": counts[stage],
                }
                for stage, seconds in sorted(
                    totals.items(), key=lambda kv: kv[1], reverse=True
                )
            },
        }

    if args.memory:
        print("misura della memoria di picco (replica dedicata)...", flush=True)
        tracemalloc.start()
        _one_run(args.agents, args.steps, args.seed, args.sampling, args.config)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["memory"] = {"peak_mb": peak / 1024 / 1024, "end_mb": current / 1024 / 1024}

    print(f"\n=== bench_kernel [{result['label']}] ===")
    print(f"commit               : {result['machine']['commit']}")
    print(f"config               : {args.config}, {args.agents} agenti x {args.steps} step, seed {args.seed}, {args.sampling}")
    print(f"repliche             : {args.replicas} (warm-up {args.warmup})")
    print(f"mediana              : {median:8.3f} s")
    print(f"p95                  : {result['p95_s']:8.3f} s")
    print(f"dispersione max-min  : {result['spread_pct']:8.2f} %")
    print(f"ms/step (mediana)    : {result['median_ms_per_step']:8.3f} ms")
    print(f"agent-step/s         : {result['agent_steps_per_s']:8.0f}")

    if args.stages:
        stages = result["stages"]
        print(
            f"\n--- ripartizione per sottostadio "
            f"(overhead della strumentazione: {stages['overhead_vs_median_pct']:+.1f}%) ---"
        )
        for stage, data in stages["per_stage"].items():
            print(
                f"  {stage:18s} {data['seconds']:8.3f} s  {data['pct_of_step']:5.1f}%  "
                f"({data['calls']} chiamate)"
            )
        print(
            f"  {'non attribuito':18s} {stages['unaccounted_s']:8.3f} s  "
            f"{stages['unaccounted_s'] / stages['instrumented_total_s'] * 100.0:5.1f}%"
        )

    if args.memory:
        print(f"\nmemoria di picco     : {result['memory']['peak_mb']:8.1f} MB")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nrisultato scritto in {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
