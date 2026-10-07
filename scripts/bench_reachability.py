from __future__ import annotations

"""Gate prestazionale del Task 4: scalar contro numpy contro rust.

Misura il batch di raggiungibilita' nelle tre implementazioni di
``src/core/reachability_batch.py`` su un carico dimensionato come uno step reale:
due campi di distanza sulla griglia piu' la valutazione delle coppie
(agente, cella adiacente).

**Il numero che conta e' `numpy / rust`.** E' il rapporto fra due implementazioni
dello *stesso* algoritmo, e quindi l'unico che quantifica il cambio di linguaggio.
Il rapporto `scalar / rust` misura insieme il riordino algoritmico e il
linguaggio: e' il guadagno che si otterrebbe nel simulatore, ma attribuirlo a
Rust sarebbe l'errore che la revisione 2 del piano (punto R2.1) esiste per
impedire.

Le misure sono **interlacciate**: i bracci si alternano a ogni ripetizione invece
di essere misurati in blocchi. Su questa macchina il rumore run-to-run raggiunge
il ~25% con derive lente, e in blocchi la deriva diventa indistinguibile
dall'effetto.

Esempio::

    python scripts/bench_reachability.py --agents 300 --repeats 30
"""

import argparse
import json
import statistics
import time
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core import native  # noqa: E402
from src.core.reachability_batch import (  # noqa: E402
    reachability_batch_numpy,
    reachability_batch_rust,
    reachability_scalar,
    support_distance_field_numpy,
    support_distance_field_rust,
)


def build_workload(agents: int, width: int, height: int, seed: int = 0) -> dict:
    """Carico dimensionato come uno step reale della configurazione golden.

    Ogni agente valuta le 8 celle adiacenti: e' il numero di coppie che il
    profilo post-0b misura (70.032 valutazioni in 30 step con 300 agenti, cioe'
    circa 2.334 per step).
    """
    rng = np.random.default_rng(seed)
    cells = width * height
    agent_x = rng.integers(0, width, size=agents)
    agent_y = rng.integers(0, height, size=agents)

    pair_agent, pair_cell = [], []
    for index in range(agents):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor_x = (int(agent_x[index]) + dx) % width
                neighbor_y = min(max(int(agent_y[index]) + dy, 0), height - 1)
                pair_agent.append(index)
                pair_cell.append(neighbor_y * width + neighbor_x)

    return {
        "width": width,
        "height": height,
        "cells": cells,
        # Numero di strutture di supporto tipico di una colonia avviata.
        "food_targets": (
            rng.integers(0, width, size=4).astype(np.int64),
            rng.integers(0, height, size=4).astype(np.int64),
        ),
        "water_targets": (
            rng.integers(0, width, size=9).astype(np.int64),
            rng.integers(0, height, size=9).astype(np.int64),
        ),
        "pair_agent": np.asarray(pair_agent, dtype=np.int64),
        "pair_cell": np.asarray(pair_cell, dtype=np.int64),
        "agent_food": rng.uniform(0.0, 3.0, size=agents),
        "agent_water": rng.uniform(0.0, 3.0, size=agents),
        "agent_steps_per_cell": rng.integers(1, 4, size=agents).astype(np.int64),
    }


def run_scalar(work: dict) -> np.ndarray:
    return reachability_scalar(
        work["pair_agent"], work["pair_cell"], work["width"], work["height"],
        work["food_targets"], work["water_targets"],
        work["agent_food"], work["agent_water"], work["agent_steps_per_cell"],
    )


def run_numpy(work: dict) -> np.ndarray:
    food = support_distance_field_numpy(work["width"], work["height"], *work["food_targets"])
    water = support_distance_field_numpy(work["width"], work["height"], *work["water_targets"])
    return reachability_batch_numpy(
        work["pair_agent"], work["pair_cell"], food, water,
        work["agent_food"], work["agent_water"], work["agent_steps_per_cell"],
    )


def run_rust(work: dict) -> np.ndarray:
    food = support_distance_field_rust(work["width"], work["height"], *work["food_targets"])
    water = support_distance_field_rust(work["width"], work["height"], *work["water_targets"])
    return reachability_batch_rust(
        work["pair_agent"], work["pair_cell"], food, water,
        work["agent_food"], work["agent_water"], work["agent_steps_per_cell"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate del Task 4: scalar contro numpy contro rust."
    )
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--width", type=int, default=36)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    work = build_workload(args.agents, args.width, args.height)
    arms: dict[str, callable] = {"scalar": run_scalar, "numpy": run_numpy}
    if native.is_available():
        arms["rust"] = run_rust
    else:
        print(
            f"ATTENZIONE: braccio rust assente ({native.unavailable_reason()}). "
            "Eseguire `python scripts/build_rust.py`.\n"
        )

    # L'equivalenza e' il prerequisito del gate: confrontare la velocita' di
    # implementazioni che calcolano cose diverse non significherebbe nulla.
    reference = run_scalar(work)
    for name, function in arms.items():
        if not np.array_equal(np.asarray(function(work)), reference):
            raise SystemExit(
                f"il braccio `{name}` non concorda con l'oracolo scalare: "
                "gate interrotto prima della misura."
            )
    print(
        f"equivalenza verificata sui tre bracci "
        f"({len(work['pair_agent'])} coppie, {reference.sum()} verdetti positivi)\n"
    )

    for _ in range(args.warmup):
        for function in arms.values():
            function(work)

    samples: dict[str, list[float]] = {name: [] for name in arms}
    for _ in range(args.repeats):
        # Interlacciamento: un giro completo dei bracci per ripetizione, cosi'
        # una deriva lenta della macchina colpisce tutti i bracci allo stesso
        # modo invece di concentrarsi su quello misurato per ultimo.
        for name, function in arms.items():
            start = time.perf_counter()
            function(work)
            samples[name].append(time.perf_counter() - start)

    medians = {name: statistics.median(values) for name, values in samples.items()}
    result = {
        "config": {
            "agents": args.agents, "width": args.width, "height": args.height,
            "pairs": int(len(work["pair_agent"])), "cells": int(work["cells"]),
            "repeats": args.repeats,
        },
        "median_s": medians,
        "p95_s": {
            name: sorted(values)[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
            for name, values in samples.items()
        },
    }

    print(f"{'braccio':10s} {'mediana':>12s} {'p95':>12s} {'us/coppia':>12s}")
    pairs = len(work["pair_agent"])
    for name in arms:
        print(
            f"{name:10s} {medians[name] * 1e3:9.3f} ms "
            f"{result['p95_s'][name] * 1e3:9.3f} ms "
            f"{medians[name] / pairs * 1e6:9.3f} us"
        )

    if "rust" in medians:
        language_gain = medians["numpy"] / medians["rust"]
        combined_gain = medians["scalar"] / medians["rust"]
        algorithmic_gain = medians["scalar"] / medians["numpy"]
        result["speedup_vs_controllo_numpy"] = language_gain
        result["speedup_vs_scalar"] = combined_gain
        result["algorithmic_gain_scalar_over_numpy"] = algorithmic_gain

        print(f"\nriordino algoritmico (scalar / numpy) : {algorithmic_gain:8.2f}x")
        print(f"cambio di linguaggio  (numpy / rust)  : {language_gain:8.2f}x   <-- il numero di Rust")
        print(f"effetto combinato     (scalar / rust) : {combined_gain:8.2f}x")
        print(
            "\nIl gate del Task 4 chiede almeno 3x sul batch mirato rispetto al "
            "Python precomputato equivalente, cioe' sul rapporto numpy / rust."
        )
        print(
            f"Esito: {'SUPERATO' if language_gain >= 3.0 else 'NON SUPERATO'} "
            f"({language_gain:.2f}x contro 3,00x richiesti)"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nrisultato scritto in {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
