from pathlib import Path

from src.visualization.replay import load_replay_manifest


def test_replay_manifest_loads_existing_run():
    runs = [p for p in Path("outputs/runs").iterdir() if p.is_dir() and (p / "replay_manifest.json").exists()]
    assert runs
    manifest = load_replay_manifest(runs[0])
    assert "snapshot_files" in manifest
