from __future__ import annotations

"""Distance helpers for the cylindrical Mars grid.

Longitude (``x``) wraps at the antimeridian while latitude (``y``) remains
bounded.  Agent routing must therefore use the shortest wrapped horizontal
distance instead of a plain ``abs(x1 - x2)``.
"""


def wrapped_axis_distance(first: int, second: int, period: int) -> int:
    """Return the shortest unsigned distance on a periodic integer axis."""
    if period <= 0:
        raise ValueError("period must be positive")
    direct = abs(int(first) - int(second)) % period
    return min(direct, period - direct)


def wrapped_manhattan_distance(
    first_x: int,
    first_y: int,
    second_x: int,
    second_y: int,
    width: int,
) -> int:
    """Manhattan distance on a grid whose horizontal axis wraps."""
    return wrapped_axis_distance(first_x, second_x, width) + abs(
        int(first_y) - int(second_y)
    )


def wrapped_chebyshev_distance(
    first_x: int,
    first_y: int,
    second_x: int,
    second_y: int,
    width: int,
) -> int:
    """Chebyshev distance on a grid whose horizontal axis wraps."""
    return max(
        wrapped_axis_distance(first_x, second_x, width),
        abs(int(first_y) - int(second_y)),
    )
