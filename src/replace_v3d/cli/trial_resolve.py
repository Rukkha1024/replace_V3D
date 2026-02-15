from __future__ import annotations

from typing import Callable


def resolve_velocity_trial(
    *,
    c3d_name: str,
    velocity_arg: float | None,
    trial_arg: int | None,
    parse_fn: Callable[[str], tuple[float, int]],
) -> tuple[float, int]:
    """Resolve (velocity, trial) from args with filename fallback.

    Matches the common pattern used in the runnable scripts:
    - If either arg is missing, parse both from the filename
    - Override whichever arg(s) were explicitly provided
    """

    if velocity_arg is None or trial_arg is None:
        vel, tr = parse_fn(c3d_name)
        velocity = vel if velocity_arg is None else float(velocity_arg)
        trial = tr if trial_arg is None else int(trial_arg)
        return float(velocity), int(trial)

    return float(velocity_arg), int(trial_arg)

