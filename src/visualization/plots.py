from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = [
    "average_habitability",
    "exploration_coverage",
    "population",
    "structures_built",
    "cooperation_index",
    "conflict_intensity",
]


def plot_metric_csv(csv_path: str | Path, output_dir: str | Path, metrics: list[str] | None = None) -> list[Path]:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    df = pd.read_csv(csv_path)
    created: list[Path] = []
    x = df["day"] if "day" in df else range(len(df))
    for metric in metrics or DEFAULT_METRICS:
        if metric not in df:
            continue
        plt.figure(figsize=(6, 3))
        plt.plot(x, df[metric])
        plt.title(metric)
        plt.xlabel("day")
        plt.tight_layout()
        out = output_dir / f"{metric}.png"
        plt.savefig(out)
        plt.close()
        created.append(out)
    return created
