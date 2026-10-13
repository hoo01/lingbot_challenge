"""Learning-rate schedules shared by formal and continuation training."""

from __future__ import annotations

import math


def schedule_multiplier(
    step: int,
    max_steps: int,
    warmup: int,
    minimum: float = 0.1,
    *,
    mode: str = "cosine",
    constant: float | None = None,
) -> float:
    """Return the LR multiplier for a zero-based scheduler step.

    ``constant`` is deliberately explicit for continuation runs: it prevents a
    changed ``max_steps`` value from reinterpreting the completed H0-A cosine
    curve and raising the learning rate after resume.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if mode == "constant":
        if warmup != 0:
            raise ValueError("constant continuation schedule must not use warmup")
        value = minimum if constant is None else constant
        if not 0 < value <= 1:
            raise ValueError("constant LR multiplier must be in (0, 1]")
        return float(value)
    if mode != "cosine":
        raise ValueError("lr_schedule must be 'cosine' or 'constant'")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum LR multiplier must be in [0, 1]")
    if warmup and step < warmup:
        return max(1, step + 1) / warmup
    progress = min(1, max(0, (step - warmup) / max(1, max_steps - warmup)))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
