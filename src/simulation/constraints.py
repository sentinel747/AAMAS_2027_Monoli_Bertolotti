from __future__ import annotations

from typing import Any
import math

import numpy as np


def sanitize_value(name: str, value: Any) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    try:
        if isinstance(value, np.ndarray):
            value = np.nan_to_num(value, nan=0.0, posinf=1e12, neginf=-1e12)
            return value, warnings
        if isinstance(value, (int, float)):
            if math.isnan(float(value)) or math.isinf(float(value)):
                warnings.append(f"{name} produced non-finite value; replaced with 0")
                return 0.0, warnings
    except TypeError:
        pass
    return value, warnings


def scalarize(value: Any) -> float:
    try:
        if isinstance(value, np.ndarray):
            return float(np.nanmean(value))
        if isinstance(value, list):
            return float(np.nanmean(np.array(value, dtype=float)))
        return float(value)
    except Exception:
        return 0.0
