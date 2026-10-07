from __future__ import annotations

from pathlib import Path


def generate_report(output_dir: str | Path, scenario_name: str, metrics: dict, indicators: dict) -> Path:
    output_dir = Path(output_dir)
    text = [
        f"# Experiment Summary: {scenario_name}",
        "",
        "## Final Metrics",
        "",
    ]
    for key, value in metrics.items():
        text.append(f"- {key}: {value}")
    text.extend(["", "## Emergence Indicators", ""])
    for key, value in indicators.items():
        text.append(f"- {key}: {value}")
    path = output_dir / "summary.md"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path
