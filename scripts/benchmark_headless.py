from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.agent_coupled_runner import AgentCoupledRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MarsABM headless throughput.")
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--estimate-agents", type=int, default=100_000)
    parser.add_argument("--estimate-steps", type=int, default=10_000)
    args = parser.parse_args()

    base_config = {
        "name": "headless_benchmark",
        "seed": 42,
        "days": args.steps,
        "simulation": {"days_per_step": 1, "max_days": args.steps},
        "world": {"width": 36, "height": 24, "map_profile": "balanced"},
        "agents": {
            "count": args.agents,
            "llm_count": 0,
            "initial_inventory": {
                "food": 10,
                "water": 10,
                "construction_material": 8,
                "tools": 2,
                "oxygen": 4,
                "energy": 2,
                "minerals": 4,
                "med_kits": 1,
            },
        },
        "colony": {"initial_structures": {"habitat": 1, "greenhouse": 1, "solar_array": 1, "oxygen_plant": 1}},
        "population": {"enabled": False},
        "environmental_layer": {"enabled": False},
        "snapshot_interval": 0,
    }

    v1_like = copy.deepcopy(base_config)
    v1_like["model"] = {"psychosocial_enabled": True}
    v1_like["headless"] = {
        "fast_observation": False,
        "store_memory_logs": True,
        "log_interval_steps": max(1, args.steps),
        "max_action_log_rows": args.agents * args.steps + 1,
    }

    v2 = copy.deepcopy(base_config)
    v2["model"] = {"psychosocial_enabled": False}
    v2["headless"] = {
        "fast_observation": True,
        "store_memory_logs": False,
        "log_interval_steps": max(1, args.steps),
        "max_action_log_rows": 0,
    }

    aggregate = copy.deepcopy(base_config)
    aggregate["agents"]["count"] = args.estimate_agents
    aggregate["days"] = args.estimate_steps
    aggregate["simulation"] = {"days_per_step": 1, "max_days": args.estimate_steps}
    aggregate["model"] = {"psychosocial_enabled": False}
    aggregate["headless"] = {
        "aggregate_mode": True,
        "aggregate_sample_agents": 50,
        "log_interval_steps": max(1, args.estimate_steps),
    }

    results = [run_case("v1_like", v1_like, args.steps), run_case("v2_headless", v2, args.steps)]
    print("\nagent_based_case,seconds,agent_steps_per_second,estimated_extreme_seconds,estimated_extreme_hours")
    extreme_agent_steps = args.estimate_agents * args.estimate_steps
    for result in results:
        estimated_seconds = extreme_agent_steps / max(1.0, result["agent_steps_per_second"])
        print(
            f"{result['case']},{result['seconds']:.3f},{result['agent_steps_per_second']:.0f},"
            f"{estimated_seconds:.0f},{estimated_seconds / 3600.0:.2f}"
        )
    aggregate_result = run_case("v2_macro_aggregate_estimate", aggregate, args.estimate_steps)
    print("\nmacro_case,seconds,macro_updates,estimated_agents,estimated_steps,note")
    print(
        f"{aggregate_result['case']},{aggregate_result['seconds']:.3f},{args.estimate_steps},"
        f"{args.estimate_agents},{args.estimate_steps},"
        "not_agent_based_no_individual_decisions"
    )


def run_case(name: str, config: dict, steps: int) -> dict:
    runner = AgentCoupledRunner(config)
    started = time.perf_counter()
    runner.run(days=steps, output_dir=None)
    seconds = time.perf_counter() - started
    agent_steps = max(1, config["agents"]["count"] * steps)
    rate = agent_steps / max(seconds, 1.0e-9)
    return {"case": name, "seconds": seconds, "agent_steps_per_second": rate}


if __name__ == "__main__":
    main()
