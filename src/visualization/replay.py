from __future__ import annotations

from pathlib import Path
import json


def load_replay_manifest(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "replay_manifest.json"
    if not path.exists():
        return {"snapshots": [], "events": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    snapshots_dir = Path(run_dir) / data.get("snapshots", "world_snapshots")
    data["snapshot_files"] = sorted(str(p) for p in snapshots_dir.glob("*.json"))
    return data
