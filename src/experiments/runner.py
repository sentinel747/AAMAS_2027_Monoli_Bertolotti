from __future__ import annotations

from pathlib import Path
import asyncio
from datetime import datetime
import json
from uuid import uuid4

import yaml

from src.simulation.agent_coupled_runner import AgentCoupledRunner
from src.visualization.plots import plot_metric_csv

from .metrics import agent_metrics, composite_agent_score, emergence_indicators, planetary_metrics
from .report_generator import generate_report


class ExperimentRunner:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.scenario_name = self.config.get("name", self.config_path.stem)

    def run(self, output_root: str | Path = "outputs/runs") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        steps = int(self.config.get("days", 30))
        dps = int((self.config.get("simulation") or {}).get("days_per_step", 1))
        seed = int(self.config.get("seed", 0))
        world_cfg = self.config.get("world") or {}
        map_profile = str(world_cfg.get("map_profile") or self.config.get("map_profile") or "balanced")
        run_id = f"{timestamp}_{self.scenario_name}_{map_profile}_seed{seed}_s{steps}_dps{dps}_{uuid4().hex[:6]}"
        output_dir = Path(output_root) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        runner = AgentCoupledRunner(self.config)
        runner.run(days=steps, output_dir=output_dir)
        if not runner.global_metrics:
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
            return output_dir
        last = runner.global_metrics[-1]
        initial_agent_count = int((self.config.get("agents") or {}).get("count", len(runner.agents) or 1))
        metrics = {
            "run_id": run_id,
            "seed": seed,
            "map_profile": runner.world.metadata.get("map_profile", map_profile),
            "requested_map_profile": runner.world.metadata.get("requested_map_profile", map_profile),
            **last,
            **planetary_metrics(last),
            **agent_metrics(
                runner.agents,
                runner.validated_actions,
                initial_agent_count=initial_agent_count,
                initial_agent_ids=getattr(runner, "initial_agent_ids", None),
            ),
        }
        metrics.update(composite_agent_score(metrics))
        indicators = emergence_indicators(metrics)
        (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output_dir / "model_version.json").write_text(json.dumps({"version": "translated-stgraph-plus-agent-world-mvp"}, indent=2), encoding="utf-8")
        generate_report(output_dir, self.scenario_name, metrics, indicators)
        charts = output_dir / "charts"
        charts.mkdir(exist_ok=True)
        plot_metric_csv(output_dir / "state_timeseries.csv", charts)
        return output_dir
