from __future__ import annotations

"""Benchmark del solo loop di step del kernel SoA (v2).

Separa la worldgen (una-tantum, in __init__) dal loop di step, cosi' ms/step
riflette il costo per-step effettivo e non l'IO iniziale. Con --profile aggancia
cProfile e ripartisce il tempo tra le fasi del kernel (decisioni, azioni, vitali,
biologia, eventi, snapshot) leggendo il modulo di ciascuna funzione campionata.

Il confronto v1(oggetti) vs v2(kernel) a parita' di seed e' gia' coperto da
scripts/compare_v1_v2.py (che stampa anche i tempi wall-clock dei due motori);
questo script isola il costo per-step della v2 e dice DOVE lo spende.
"""

import argparse
import cProfile
import pstats
from collections import defaultdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.agent_coupled_runner import AgentCoupledRunner
from src.agents.role_profiles import DEFAULT_ROLE_DISTRIBUTION

# Mappa (frammento di path del modulo) -> etichetta di fase, in ordine di priorita'.
PHASE_MODULES = [
    ("core/kernel_vitals", "vitali"),
    ("world/vitals", "vitali"),
    ("core/kernel_biology", "biologia"),
    ("world/colony_dynamics", "biologia"),
    ("world/step_effects", "biologia"),
    ("core/needs", "biologia"),
    ("core/redistribution", "redistribuzione"),
    ("core/cell_action_mask", "decisioni"),
    ("core/kernel_decision", "decisioni"),
    ("agents/preference_agent", "decisioni"),
    ("agents/rule_based_agent", "decisioni"),
    ("world/perception", "decisioni"),
    ("agents/action_space", "azioni"),
    ("agents/build_policy", "azioni"),
    ("world/extreme_events", "eventi"),
    ("core/snapshot", "snapshot"),
    ("core/views", "facciate(views)"),
    ("core/arrays", "arrays"),
    ("core/kernel", "kernel(orchestraz.)"),
]


def _build_config(agents: int, steps: int, seed: int, decision_mode: str) -> dict:
    return {
        "name": "benchmark_step",
        "seed": seed,
        "days": steps,
        "simulation": {"days_per_step": 1, "max_days": steps},
        "world": {"width": 36, "height": 24, "map_profile": "balanced"},
        "agents": {
            "count": agents,
            "llm_count": 0,
            "decision_mode": decision_mode,
            "decision_sampling": "softmax",
            **(
                {
                    "role_distribution": dict(DEFAULT_ROLE_DISTRIBUTION),
                    "role_preference_randomness": 0.25,
                }
                if decision_mode == "preferences"
                else {}
            ),
            "initial_inventory": {
                "food": 10, "water": 10, "construction_material": 8, "tools": 2,
                "oxygen": 4, "energy": 2, "minerals": 4, "med_kits": 1,
            },
        },
        "colony": {"initial_structures": {"habitat": 1, "greenhouse": 1, "solar_array": 1, "oxygen_plant": 1}},
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


def _phase_for(filename: str) -> str:
    norm = filename.replace("\\", "/")
    for fragment, label in PHASE_MODULES:
        if fragment in norm:
            return label
    return "altro"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ms/step del kernel SoA (v2).")
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decision-mode", choices=("tree", "preferences"), default="tree"
    )
    parser.add_argument("--profile", action="store_true", help="Profila con cProfile e ripartisce per fase")
    args = parser.parse_args()

    config = _build_config(args.agents, args.steps, args.seed, args.decision_mode)

    t0 = time.perf_counter()
    runner = AgentCoupledRunner(config)
    worldgen_s = time.perf_counter() - t0
    live_agents = len(runner.agents)

    if args.profile:
        profiler = cProfile.Profile()
        t1 = time.perf_counter()
        profiler.enable()
        runner.run(days=args.steps, output_dir=None)
        profiler.disable()
        loop_s = time.perf_counter() - t1
    else:
        t1 = time.perf_counter()
        runner.run(days=args.steps, output_dir=None)
        loop_s = time.perf_counter() - t1

    ms_per_step = loop_s / max(1, args.steps) * 1000.0
    agent_steps = live_agents * args.steps
    agent_steps_per_s = agent_steps / max(loop_s, 1e-9)

    print("\n=== benchmark_step (v2 kernel SoA) ===")
    print(
        f"agenti(vivi all'avvio)={live_agents}  step={args.steps}  "
        f"seed={args.seed}  decision_mode={args.decision_mode}"
    )
    print(f"worldgen (una-tantum) : {worldgen_s:8.3f} s")
    print(f"loop di step          : {loop_s:8.3f} s")
    print(f"ms/step               : {ms_per_step:8.3f} ms")
    print(f"agent-step/s          : {agent_steps_per_s:8.0f}")

    if args.profile:
        stats = pstats.Stats(profiler)
        by_phase: dict[str, float] = defaultdict(float)
        total_tottime = 0.0
        for (filename, _lineno, _func), (_cc, _nc, tottime, _ct, _callers) in stats.stats.items():
            by_phase[_phase_for(filename)] += tottime
            total_tottime += tottime
        print("\n--- ripartizione per fase (tottime, self-time) ---")
        for label, secs in sorted(by_phase.items(), key=lambda kv: kv[1], reverse=True):
            pct = secs / max(total_tottime, 1e-9) * 100.0
            print(f"  {label:22s} {secs:8.3f} s  {pct:5.1f}%")
        print(f"  {'TOTALE (self-time)':22s} {total_tottime:8.3f} s")
        print("\n--- top 15 funzioni per tempo cumulativo ---")
        stats.sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
