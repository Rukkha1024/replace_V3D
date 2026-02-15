from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_velocity(velocity: float) -> str:
    if float(velocity).is_integer():
        return str(int(velocity))
    return str(float(velocity))


def build_trial_key(subject: str, velocity: float, trial: int) -> str:
    return f"{subject}-{format_velocity(velocity)}-{int(trial)}"


def iter_c3d_files(c3d_dir: Path) -> list[Path]:
    c3d_dir = Path(c3d_dir)
    return sorted([path for path in c3d_dir.rglob("*.c3d") if path.is_file()])


def append_rows_to_csv(
    out_csv: Path,
    df: pd.DataFrame,
    *,
    header_written: bool,
    encoding: str,
) -> bool:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, mode="a", index=False, header=not header_written, encoding=encoding)
    return True

