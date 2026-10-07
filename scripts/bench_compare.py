from __future__ import annotations

"""Confronto A/B interlacciato fra due bracci di benchmark.

Perche' non basta ``bench_kernel.py`` eseguito due volte: su questa macchina il
rumore run-to-run misurato e' arrivato al ~25%, con derive lente dovute a carico e
condizioni termiche. Misurare cinque repliche del braccio A e poi cinque del
braccio B rende quella deriva indistinguibile dall'effetto cercato: se la macchina
rallenta fra il primo e il secondo blocco, il braccio misurato per secondo sembra
peggiore di quanto sia. Con un gate a 1,2x l'errore e' dello stesso ordine
dell'effetto.

La misura e' quindi **interlacciata e appaiata**: A B A B A B..., ogni esecuzione
in un processo separato, e il rapporto si calcola sulle coppie adiacenti. Una
deriva lenta agisce quasi ugualmente sui due membri di ogni coppia e si cancella
nel rapporto, invece di sommarsi a esso.

I bracci sono definiti da variabili d'ambiente, cosi' lo stesso strumento serve al
Task 0b (precompute Python acceso/spento) e ai Task successivi (backend
python/rust/shadow) senza modifiche.

Esempi::

    python scripts/bench_compare.py \\
        --arm pre:MARSABM_PRECOMPUTE=0 --arm post:MARSABM_PRECOMPUTE=1 \\
        --agents 300 --steps 30 --replicas 5
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "scripts" / "bench_kernel.py"


def parse_arm(spec: str) -> tuple[str, dict[str, str]]:
    """``nome:VAR=val,VAR2=val2`` -> ``(nome, {VAR: val, ...})``."""
    if ":" not in spec:
        return spec, {}
    name, _, assignments = spec.partition(":")
    env: dict[str, str] = {}
    for chunk in assignments.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        env[key.strip()] = value.strip()
    return name, env


def run_once(env_overrides: dict[str, str], args, tmp: Path) -> float:
    out = tmp / "one.json"
    proc = subprocess.run(
        [
            sys.executable, str(BENCH),
            "--agents", str(args.agents), "--steps", str(args.steps),
            "--seed", str(args.seed), "--sampling", args.sampling,
            "--config", args.config,
            "--replicas", "1", "--warmup", "0", "--json", str(out),
        ],
        cwd=str(REPO_ROOT),
        env=os.environ | env_overrides,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"braccio fallito (env={env_overrides}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return float(json.loads(out.read_text(encoding="utf-8"))["median_s"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Confronto A/B interlacciato di due bracci di benchmark."
    )
    parser.add_argument(
        "--arm", action="append", required=True, metavar="NOME:VAR=VAL[,VAR=VAL]",
        help="braccio da confrontare; sono richiesti esattamente due",
    )
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("softmax", "greedy"), default="softmax")
    parser.add_argument(
        "--config", choices=("golden", "realistic"), default="golden",
        help="carico su cui misurare; un guadagno dichiarato per le run di tesi "
             "va preso su 'realistic'",
    )
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="coppie scartate all'inizio")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if len(args.arm) != 2:
        parser.error("servono esattamente due bracci")
    arms = [parse_arm(spec) for spec in args.arm]
    (name_a, env_a), (name_b, env_b) = arms

    print(f"braccio A = {name_a} {env_a or '(ambiente invariato)'}")
    print(f"braccio B = {name_b} {env_b or '(ambiente invariato)'}")
    print(
        f"config: {args.config}, {args.agents} agenti x {args.steps} step, "
        f"seed {args.seed}, {args.sampling}"
    )
    print(f"coppie interlacciate: {args.replicas} (warm-up {args.warmup})\n", flush=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for _ in range(args.warmup):
            run_once(env_a, args, tmp)
            run_once(env_b, args, tmp)

        pairs: list[tuple[float, float]] = []
        for index in range(args.replicas):
            # L'ordine si alterna fra coppie: se A fosse sempre primo, un costo
            # sistematico legato alla posizione (cache del filesystem ancora
            # fredda, turbo del processore ancora disponibile) finirebbe sempre
            # sullo stesso braccio.
            if index % 2 == 0:
                time_a = run_once(env_a, args, tmp)
                time_b = run_once(env_b, args, tmp)
            else:
                time_b = run_once(env_b, args, tmp)
                time_a = run_once(env_a, args, tmp)
            pairs.append((time_a, time_b))
            print(
                f"  coppia {index + 1}/{args.replicas}: "
                f"{name_a}={time_a:.3f}s  {name_b}={time_b:.3f}s  "
                f"rapporto={time_a / time_b:.3f}x",
                flush=True,
            )

    times_a = [pair[0] for pair in pairs]
    times_b = [pair[1] for pair in pairs]
    paired_ratios = [a / b for a, b in pairs]

    median_a = statistics.median(times_a)
    median_b = statistics.median(times_b)
    result = {
        "arm_a": {"name": name_a, "env": env_a, "timings_s": times_a, "median_s": median_a},
        "arm_b": {"name": name_b, "env": env_b, "timings_s": times_b, "median_s": median_b},
        "config": {
            "kind": args.config,
            "agents": args.agents, "steps": args.steps,
            "seed": args.seed, "sampling": args.sampling, "replicas": args.replicas,
        },
        "paired_ratios": paired_ratios,
        "median_paired_ratio": statistics.median(paired_ratios),
        "min_paired_ratio": min(paired_ratios),
        "max_paired_ratio": max(paired_ratios),
        "ratio_of_medians": median_a / median_b if median_b else 0.0,
    }

    print(f"\n=== {name_a} contro {name_b} ===")
    print(f"mediana {name_a:<14s}: {median_a:8.3f} s")
    print(f"mediana {name_b:<14s}: {median_b:8.3f} s")
    print(f"rapporto appaiato mediano : {result['median_paired_ratio']:8.3f}x")
    print(
        f"intervallo dei rapporti   : "
        f"{result['min_paired_ratio']:.3f}x - {result['max_paired_ratio']:.3f}x"
    )
    print(f"rapporto delle mediane    : {result['ratio_of_medians']:8.3f}x")
    if result["min_paired_ratio"] < 1.0 < result["max_paired_ratio"]:
        print(
            "\nATTENZIONE: i rapporti appaiati attraversano 1.0. Il segno dell'effetto "
            "non e' stabile su questa macchina: aumentare le repliche o la dimensione "
            "della configurazione prima di concludere."
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nrisultato scritto in {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
