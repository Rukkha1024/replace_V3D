from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass
class TrialEvents:
    subject: str
    velocity: float
    trial: int

    platform_onset_original: int
    platform_offset_original: int
    step_onset_original: Optional[int]

    trim_start_original: int
    platform_onset_local: int
    platform_offset_local: int
    step_onset_local: Optional[int]


def load_trial_events(
    event_xlsm: str | Path,
    subject: str,
    velocity: float,
    trial: int,
    pre_frames: int = 100,
    sheet_name: str = "platform",
) -> TrialEvents:
    """Read platform/step events from `perturb_inform.xlsm` (sheet `platform`).

    Assumes mocap is trimmed to: [platform_onset - pre_frames, platform_offset + pre_frames]
    so that `platform_onset_local == pre_frames + 1` (normally 101).
    """
    event_xlsm = Path(event_xlsm)

    df = pd.read_excel(event_xlsm, sheet_name=sheet_name)

    row = df[
        (df["subject"].astype(str) == str(subject))
        & (df["velocity"].astype(float) == float(velocity))
        & (df["trial"].astype(int) == int(trial))
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly 1 matching row in '{sheet_name}' for "
            f"subject={subject}, velocity={velocity}, trial={trial}. "
            f"Got {len(row)} rows."
        )

    r = row.iloc[0]

    platform_onset = int(r["platform_onset"])
    platform_offset = int(r["platform_offset"])
    step_onset = None if pd.isna(r.get("step_onset")) else int(r["step_onset"])

    trim_start = platform_onset - pre_frames

    def to_local(original_frame: int) -> int:
        return int(original_frame - trim_start + 1)

    platform_onset_local = to_local(platform_onset)
    platform_offset_local = to_local(platform_offset)
    step_onset_local = None if step_onset is None else to_local(step_onset)

    return TrialEvents(
        subject=str(subject),
        velocity=float(velocity),
        trial=int(trial),
        platform_onset_original=platform_onset,
        platform_offset_original=platform_offset,
        step_onset_original=step_onset,
        trim_start_original=trim_start,
        platform_onset_local=platform_onset_local,
        platform_offset_local=platform_offset_local,
        step_onset_local=step_onset_local,
    )


def parse_trial_from_filename(c3d_name: str) -> Tuple[float, int]:
    """Parse velocity + trial number from file name rule:
    `{date}_{name_initial}_perturb_{velocity}_{trial}.c3d`

    Example: `251112_KUO_perturb_60_001.c3d` -> velocity=60.0, trial=1
    """
    stem = Path(c3d_name).name
    if stem.lower().endswith(".c3d"):
        stem = stem[:-4]
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected c3d filename format: {c3d_name}")
    # parts: [date, initials, perturb, velocity, trial]
    velocity = float(parts[3])
    trial = int(parts[4])
    return velocity, trial


def load_subject_body_mass_kg(
    event_xlsm: str | Path,
    subject: str,
    sheet_name: str = "meta",
    row_key: str = "몸무게",
) -> Optional[float]:
    """Load subject body mass (kg) from `perturb_inform.xlsm`.

    The repo's `perturb_inform.xlsm` uses a wide format `meta` sheet:
      - Column `subject`: metadata key labels (e.g., 성별/나이/키/몸무게)
      - Each remaining column is a subject name (e.g., 김우연)

    Returns None if the sheet/row/subject column is missing.
    """

    event_xlsm = Path(event_xlsm)
    df = pd.read_excel(event_xlsm, sheet_name=sheet_name)
    if "subject" not in df.columns:
        return None
    if subject not in df.columns:
        return None

    key_series = df["subject"].astype(str).str.strip()
    row = df[key_series == str(row_key).strip()]
    if len(row) < 1:
        return None

    val = row.iloc[0][subject]
    if pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        return None

