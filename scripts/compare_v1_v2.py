"""Task 14 Step 1-2: equivalence comparison between the pre-refactor object
engine (v1, commit 8ac9339 - see .superpowers/sdd/task-14-brief.md) and the
current kernel-driven engine (v2, this HEAD, `redistribution.enabled=false`).

Both engines are driven through the exact same public entry point
(`src.simulation.agent_coupled_runner.AgentCoupledRunner(config).run(days=...,
output_dir=...)`), which both v1 and v2 still expose byte-identically (only
its INTERNALS differ - object loop vs `src.core.kernel.step`). This script
never imports both versions of the `src` package in one process (heavy risk
of stale/duplicate module state); instead it launches ONE subprocess per
engine, each with only that engine's checkout on `sys.path[0]`, and compares
the artifacts left on disk - exactly the "practical mode" the task brief
recommends.

Usage (from the repo root, main worktree):

    python scripts/compare_v1_v2.py --preset small
    python scripts/compare_v1_v2.py --preset medium
    python scripts/compare_v1_v2.py --preset both        (default)
    python scripts/compare_v1_v2.py --agents 50 --steps 100 --control-interval 25

    # Step 4 (exploratory, NOT an equivalence requirement): v2 vs v2 with
    # redistribution flipped on, instead of v1 vs v2:
    python scripts/compare_v1_v2.py --against redistribution --preset small

Internal worker mode (spawned by the orchestrator above, not meant to be
called directly):

    python scripts/compare_v1_v2.py --worker --repo-root R --config-json C
        --output-dir O --days N
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_ROOT = REPO_ROOT.parent / "mars-v1-ref"


# ---------------------------------------------------------------------------
# Shared config builder - ONE config dict fed to both engines verbatim. Only
# knobs that are new since v1 (e.g. "redistribution") are simply ignored by
# the v1 object engine, which never reads that key.
# ---------------------------------------------------------------------------

def build_config(
    *,
    agents: int,
    steps: int,
    seed: int,
    world_width: int,
    world_height: int,
    snapshot_interval: int,
    redistribution_enabled: bool,
    days_per_step: int = 7,
) -> dict:
    # Structure scaling per docs/MARS_ABM_NOTE_CONFIGURAZIONE.md ("Come
    # scalare i valori quando cambio il numero di agenti iniziali"): ~linear
    # in N off the documented N=50 baseline.
    scale = agents / 50.0
    structures = {
        "habitat": max(1, round(10 * scale)),
        "greenhouse": max(1, round(7 * scale)),
        "solar_array": max(1, round(5 * scale)),
        "oxygen_plant": max(1, round(5 * scale)),
        "storage_depot": max(1, round(2 * scale)),
        "weather_station": max(1, round(1 * scale)),
    }
    return {
        "name": f"equivalence_{agents}a_{steps}s_seed{seed}",
        "seed": seed,
        "days": steps,
        "simulation": {"days_per_step": days_per_step, "max_days": steps * days_per_step},
        "world": {"width": world_width, "height": world_height, "map_profile": "balanced", "cell_degradation": True},
        "agents": {
            "count": agents,
            "llm_count": 0,
            # The frozen v1 baseline has no preference policy; equivalence is
            # therefore defined exclusively for the historical decision tree.
            "decision_mode": "tree",
            "initial_inventory": {"food": 5, "water": 5, "construction_material": 5, "tools": 1},
        },
        "colony": {"initial_structures": structures},
        # Task 13 review caveat, load-bearing for a clean comparison: growth
        # OFF avoids the documented maybe_spawn_agent day_start/day_end RNG
        # shift between the pre-migration CLI and the unified shells.
        "population": {"enabled": False},
        "environmental_layer": {"enabled": False},
        "extreme_events": {"enabled": True, "chance": 0.05},
        "model": {"psychosocial_enabled": False},
        "headless": {
            "fast_observation": True,
            "rule_based_observation": True,
            "log_interval_steps": max(1, steps),
            "store_memory_logs": False,
            "max_action_log_rows": 0,
            "max_replay_rows": 0,
        },
        "snapshot_interval": snapshot_interval,
        "redistribution": {"enabled": redistribution_enabled},
    }


# ---------------------------------------------------------------------------
# Worker: runs ONE engine, in its own process, rooted at --repo-root.
# ---------------------------------------------------------------------------

def _run_worker(args: argparse.Namespace) -> None:
    repo_root = str(Path(args.repo_root).resolve())
    # Make sure THIS engine's src/ wins over anything else importable -
    # insert first, and do not rely on cwd/site-packages.
    sys.path.insert(0, repo_root)
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))

    # Pre-existing, PROTECTED-module non-determinism (found during the Task
    # 14 equivalence audit, NOT a kernel/refactor bug): src/agents/rule_based_
    # agent.py's own general-exploration fallback ("should_move = random.
    # random() < ...", one call site) reads the process-wide stdlib `random`
    # module directly, not an injected/seeded generator - true in BOTH v1 and
    # v2 (the module is byte-identical, never modified). Neither engine's own
    # runner (agent_coupled_runner.py/state_store.py) ever calls
    # `random.seed(config["seed"])`, so that ONE branch's outcome depends on
    # whatever ambient global `random` state each separate OS process happens
    # to start with - genuinely non-reproducible ACROSS processes, by design
    # of the original (pre-refactor) code, unrelated to SoA vectorization.
    # Seeding the global module here, identically in both engines' worker
    # process, controls for that ambient state WITHOUT touching the
    # protected module - this is the comparison harness's own determinism
    # knob, not a behavior change to either engine.
    import random as _random
    _random.seed(int(config.get("seed", 0)))

    from src.simulation.agent_coupled_runner import AgentCoupledRunner  # noqa: E402  (path set above)

    runner = AgentCoupledRunner(config)
    started = time.perf_counter()
    runner.run(days=args.days, output_dir=args.output_dir)
    elapsed = time.perf_counter() - started
    (Path(args.output_dir) / "_worker_timing.json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "steps": args.days}), encoding="utf-8"
    )


def _run_engine(repo_root: Path, config: dict, output_dir: Path, days: int, label: str) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_json = output_dir / "_config.json"
    config_json.write_text(json.dumps(config), encoding="utf-8")
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--repo-root", str(repo_root),
        "--config-json", str(config_json),
        "--output-dir", str(output_dir),
        "--days", str(days),
    ]
    print(f"[compare_v1_v2] running {label} engine (root={repo_root}) -> {output_dir}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"{label} engine run failed (exit {result.returncode})")
    timing_path = output_dir / "_worker_timing.json"
    elapsed = json.loads(timing_path.read_text(encoding="utf-8"))["elapsed_seconds"] if timing_path.exists() else float("nan")
    print(f"[compare_v1_v2] {label} done in {elapsed:.2f}s", flush=True)
    return elapsed


# ---------------------------------------------------------------------------
# Artifact loading + comparison
# ---------------------------------------------------------------------------

def _load_snapshot_steps(output_dir: Path) -> dict[int, dict]:
    snap_dir = output_dir / "world_snapshots"
    out: dict[int, dict] = {}
    if not snap_dir.exists():
        return out
    for path in sorted(snap_dir.glob("step_*.json")):
        step_num = int(path.name.split("_")[1])
        out[step_num] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _struct_counts(snapshot: dict) -> dict[tuple[int, int, str], int]:
    counts: dict[tuple[int, int, str], int] = {}
    for cell in snapshot.get("cells", []):
        x, y = cell["x"], cell["y"]
        for structure in cell.get("structures", []):
            key = (x, y, structure["type"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def _agent_positions(snapshot: dict) -> list[tuple[str, float, float]]:
    return sorted((a["agent_id"], a["x"], a["y"]) for a in snapshot.get("agents", []))


_KNOWN_ACCEPTED_METRIC_KEYS = {
    # Task 3/8's documented sacrifice (see src/core/kernel_biology.py's
    # module docstring): the vectorized cell only stores struct_count +
    # SUM(integrity) per (cell, type), not each structure's own integrity.
    # StructureView therefore hands every instance of a type in a cell the
    # SAME mean integrity, so any metric that re-derives per-structure
    # efficiency (clipped at the 0.4 cliff, src/world/structures.py
    # `Structure.efficiency`) via `world.cells[...].structures` can diverge
    # from the object engine's true per-instance sum whenever individual
    # integrities straddle that threshold. Flagged, not failed, here.
    "colony_built_area_m2",
}


def _compare_metrics(step: int, v1_metrics: dict, v2_metrics: dict, tolerance: float, divergences: list, accepted: list) -> None:
    keys = sorted(set(v1_metrics) & set(v2_metrics))
    for key in keys:
        a, b = v1_metrics[key], v2_metrics[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = abs(float(a) - float(b))
            if delta > tolerance:
                record = {"step": step, "field": f"metrics.{key}", "v1": a, "v2": b, "abs_delta": delta}
                if key in _KNOWN_ACCEPTED_METRIC_KEYS:
                    accepted.append(record)
                else:
                    divergences.append(record)
        elif a != b:
            divergences.append({"step": step, "field": f"metrics.{key}", "v1": a, "v2": b, "abs_delta": None})


def compare_runs(v1_dir: Path, v2_dir: Path, tolerance: float) -> dict:
    v1_snaps = _load_snapshot_steps(v1_dir)
    v2_snaps = _load_snapshot_steps(v2_dir)
    steps = sorted(set(v1_snaps) & set(v2_snaps))
    missing_v1 = sorted(set(v2_snaps) - set(v1_snaps))
    missing_v2 = sorted(set(v1_snaps) - set(v2_snaps))

    divergences: list[dict] = []
    accepted: list[dict] = []

    for step in steps:
        s1, s2 = v1_snaps[step], v2_snaps[step]

        pop1, pop2 = len(s1.get("agents", [])), len(s2.get("agents", []))
        if pop1 != pop2:
            divergences.append({"step": step, "field": "population", "v1": pop1, "v2": pop2, "abs_delta": abs(pop1 - pop2)})

        pos1, pos2 = _agent_positions(s1), _agent_positions(s2)
        if pos1 != pos2:
            for (id1, x1, y1), (id2, x2, y2) in zip(pos1, pos2):
                if (id1, x1, y1) != (id2, x2, y2):
                    divergences.append({
                        "step": step, "field": "agent_position",
                        "v1": (id1, x1, y1), "v2": (id2, x2, y2), "abs_delta": None,
                    })
                    break
            else:
                divergences.append({"step": step, "field": "agent_position_count", "v1": len(pos1), "v2": len(pos2), "abs_delta": None})

        sc1, sc2 = _struct_counts(s1), _struct_counts(s2)
        for key in sorted(set(sc1) | set(sc2)):
            c1, c2 = sc1.get(key, 0), sc2.get(key, 0)
            if c1 != c2:
                divergences.append({"step": step, "field": f"struct_count{key}", "v1": c1, "v2": c2, "abs_delta": abs(c1 - c2)})

        _compare_metrics(step, s1.get("metrics", {}), s2.get("metrics", {}), tolerance, divergences, accepted)

    return {
        "control_steps_compared": steps,
        "missing_snapshots_in_v1": missing_v1,
        "missing_snapshots_in_v2": missing_v2,
        "divergences": divergences,
        "accepted_divergences": accepted,
    }


def _summarize(result: dict, tolerance: float) -> None:
    steps = result["control_steps_compared"]
    print(f"\ncontrol steps compared: {steps}")
    if result["missing_snapshots_in_v1"] or result["missing_snapshots_in_v2"]:
        print(f"WARNING missing snapshots - only in v2: {result['missing_snapshots_in_v1']}, only in v1: {result['missing_snapshots_in_v2']}")

    divergences = result["divergences"]
    accepted = result["accepted_divergences"]

    if not divergences:
        print(f"EQUIVALENCE: no divergence above tolerance={tolerance} on any control step.")
    else:
        first = divergences[0]
        print(f"FIRST DIVERGENCE: step={first['step']} field={first['field']} v1={first['v1']!r} v2={first['v2']!r} abs_delta={first['abs_delta']!r}")
        print(f"total divergent (step, field) pairs: {len(divergences)}")
        by_field: dict[str, float] = {}
        for d in divergences:
            if d["abs_delta"] is not None:
                by_field[d["field"]] = max(by_field.get(d["field"], 0.0), d["abs_delta"])
        if by_field:
            print("max abs delta per numeric field (top 10):")
            for field, delta in sorted(by_field.items(), key=lambda kv: -kv[1])[:10]:
                print(f"  {field}: {delta:.6g}")

    if accepted:
        by_field = {}
        for d in accepted:
            if d["abs_delta"] is not None:
                by_field[d["field"]] = max(by_field.get(d["field"], 0.0), d["abs_delta"])
        print("\nACCEPTED divergence (documented Task-3/8 mean-integrity-vs-per-instance sacrifice):")
        for field, delta in by_field.items():
            n = sum(1 for d in accepted if d["field"] == field)
            print(f"  {field}: max abs delta={delta:.6g} over {n} control-step observation(s)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

PRESETS = {
    "small": dict(agents=50, steps=100, world_width=48, world_height=30),
    "medium": dict(agents=200, steps=400, world_width=64, world_height=40),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", help=argparse.SUPPRESS)
    parser.add_argument("--config-json", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--days", type=int, help=argparse.SUPPRESS)

    parser.add_argument("--preset", choices=["small", "medium", "both"], default="both")
    parser.add_argument("--agents", type=int, default=None, help="override agent count (disables --preset)")
    parser.add_argument("--steps", type=int, default=None, help="override step count (disables --preset)")
    parser.add_argument("--world-width", type=int, default=None)
    parser.add_argument("--world-height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-interval", type=int, default=25)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--v1-root", type=str, default=str(DEFAULT_V1_ROOT))
    parser.add_argument("--against", choices=["v1", "redistribution"], default="v1",
                         help="'v1': v1-worktree object engine vs v2 HEAD (redistribution OFF) - the equivalence "
                              "requirement. 'redistribution': v2 HEAD OFF vs v2 HEAD ON (Step 4, exploratory only).")
    parser.add_argument("--keep-artifacts", action="store_true", help="do not delete the temp output dirs afterwards")
    args = parser.parse_args()

    if args.worker:
        _run_worker(args)
        return

    if args.agents is not None or args.steps is not None:
        scenarios = [{
            "label": "custom",
            "agents": args.agents or 50,
            "steps": args.steps or 100,
            "world_width": args.world_width or 48,
            "world_height": args.world_height or 30,
        }]
    else:
        names = ["small", "medium"] if args.preset == "both" else [args.preset]
        scenarios = [{"label": name, **PRESETS[name]} for name in names]

    v1_root = Path(args.v1_root).resolve()
    if args.against == "v1" and not v1_root.exists():
        raise SystemExit(f"v1 worktree not found at {v1_root} - run: git worktree add {v1_root} 8ac9339")

    overall_ok = True
    with tempfile.TemporaryDirectory(prefix="mars_equivalence_") as tmp:
        tmp_path = Path(tmp) if not args.keep_artifacts else Path(tempfile.mkdtemp(prefix="mars_equivalence_"))
        for scenario in scenarios:
            print(f"\n=== scenario={scenario['label']} agents={scenario['agents']} steps={scenario['steps']} "
                  f"against={args.against} ===")
            config_a = build_config(
                agents=scenario["agents"], steps=scenario["steps"], seed=args.seed,
                world_width=scenario["world_width"], world_height=scenario["world_height"],
                snapshot_interval=args.control_interval, redistribution_enabled=False,
            )
            dir_a = tmp_path / f"{scenario['label']}_a"
            dir_b = tmp_path / f"{scenario['label']}_b"

            if args.against == "v1":
                config_b = config_a
                label_a, root_a = "v1(object)", v1_root
                label_b, root_b = "v2(kernel,redistribution=off)", REPO_ROOT
            else:
                config_b = build_config(
                    agents=scenario["agents"], steps=scenario["steps"], seed=args.seed,
                    world_width=scenario["world_width"], world_height=scenario["world_height"],
                    snapshot_interval=args.control_interval, redistribution_enabled=True,
                )
                label_a, root_a = "v2(kernel,redistribution=off)", REPO_ROOT
                label_b, root_b = "v2(kernel,redistribution=on)", REPO_ROOT

            elapsed_a = _run_engine(root_a, config_a, dir_a, scenario["steps"], label_a)
            elapsed_b = _run_engine(root_b, config_b, dir_b, scenario["steps"], label_b)

            result = compare_runs(dir_a, dir_b, args.tolerance)
            _summarize(result, args.tolerance)
            print(f"timing: {label_a}={elapsed_a:.2f}s  {label_b}={elapsed_b:.2f}s")
            if result["divergences"]:
                overall_ok = False
        if args.keep_artifacts:
            print(f"\nartifacts kept at: {tmp_path}")

    print("\n=== RESULT:", "EQUIVALENT" if overall_ok else "DIVERGENCE FOUND", "===")
    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
