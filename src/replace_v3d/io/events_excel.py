from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
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


@lru_cache(maxsize=16)
def _read_excel_cached(path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(Path(path), sheet_name=sheet_name)


def _normalize_token(value: object) -> str:
    token = str(value).strip()
    token = re.sub(r"\s+", "", token)
    token = token.replace("_", "").replace("-", "")
    return token.upper()


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


def parse_subject_velocity_trial_from_filename(c3d_name: str) -> Tuple[str, float, int]:
    """Parse subject token + velocity + trial from C3D filename.

    Expected default convention:
    `{date}_{subject_token}_perturb_{velocity}_{trial}.c3d`
    """
    stem = Path(c3d_name).name
    if stem.lower().endswith(".c3d"):
        stem = stem[:-4]
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected c3d filename format: {c3d_name}")

    perturb_idx: Optional[int] = None
    for i, value in enumerate(parts):
        if str(value).lower() == "perturb":
            perturb_idx = i
            break

    if perturb_idx is not None and perturb_idx >= 2 and perturb_idx + 2 < len(parts):
        subject_token = "_".join(parts[1:perturb_idx]).strip()
        if not subject_token:
            subject_token = str(parts[1]).strip()
        velocity = float(parts[perturb_idx + 1])
        trial = int(parts[perturb_idx + 2])
        return subject_token, velocity, trial

    # Fallback for strict fixed-index format
    subject_token = str(parts[1]).strip()
    velocity = float(parts[3])
    trial = int(parts[4])
    return subject_token, velocity, trial


def parse_trial_from_filename(c3d_name: str) -> Tuple[float, int]:
    """Parse velocity + trial number from file name rule:
    `{date}_{name_initial}_perturb_{velocity}_{trial}.c3d`

    Example: `251112_KUO_perturb_60_001.c3d` -> velocity=60.0, trial=1
    """
    _subject_token, velocity, trial = parse_subject_velocity_trial_from_filename(c3d_name)
    return velocity, trial


def resolve_subject_from_token(
    event_xlsm: str | Path,
    subject_token: str,
    *,
    platform_sheet: str = "platform",
    meta_sheet: str = "meta",
    transpose_sheet: str = "transpose_meta",
) -> str:
    """Resolve filename token to canonical subject name used in Excel sheets.

    Resolution order:
    1) token matches `platform.subject`
    2) token matches a subject column name in `meta`
    3) `meta` alias row (row key includes `이니셜|initial|alias|code|id`) matches token
    4) `transpose_meta` alias column (column name includes above keywords) matches token
    """
    token = str(subject_token).strip()
    if not token:
        raise ValueError("Empty subject token")
    token_norm = _normalize_token(token)
    xlsm_path = str(Path(event_xlsm).resolve())

    # 1) direct match in platform.subject
    try:
        df_platform = _read_excel_cached(xlsm_path, platform_sheet)
        if "subject" in df_platform.columns:
            subjects = df_platform["subject"].astype(str).str.strip()
            hit = subjects[subjects == token]
            if len(hit) > 0:
                return token
    except Exception:
        pass

    # 2) meta subject columns / 3) alias rows in meta
    try:
        df_meta = _read_excel_cached(xlsm_path, meta_sheet)
        meta_subject_cols = [str(col) for col in df_meta.columns if str(col) != "subject"]
        if token in meta_subject_cols:
            return token

        if "subject" in df_meta.columns:
            keys = df_meta["subject"].astype(str).str.strip()
            alias_mask = keys.str.contains(
                r"(?:이니셜|initial|alias|code|\bid\b)",
                case=False,
                regex=True,
                na=False,
            )
            alias_rows = df_meta[alias_mask]
            for _, row in alias_rows.iterrows():
                for col in meta_subject_cols:
                    value = row.get(col)
                    if pd.isna(value):
                        continue
                    if _normalize_token(value) == token_norm:
                        return col
    except Exception:
        pass

    # 4) transpose_meta alias columns
    try:
        df_transposed = _read_excel_cached(xlsm_path, transpose_sheet)
        if "subject" in df_transposed.columns:
            subjects = df_transposed["subject"].astype(str).str.strip()
            hit = subjects[subjects == token]
            if len(hit) > 0:
                return token

            for col in df_transposed.columns:
                col_str = str(col)
                if col_str == "subject":
                    continue
                if not re.search(r"(?:이니셜|initial|alias|code|\bid\b)", col_str, flags=re.IGNORECASE):
                    continue
                alias_values = df_transposed[col].astype(str).map(_normalize_token)
                matched = df_transposed[alias_values == token_norm]
                if len(matched) == 1:
                    return str(matched.iloc[0]["subject"]).strip()
    except Exception:
        pass

    raise ValueError(
        f"Could not resolve subject for token={token!r}. "
        "Add token mapping in sheet 'meta' (alias row) or 'transpose_meta' (alias column)."
    )


def load_subject_leg_length_cm(
    event_xlsm: str | Path,
    subject: str,
    *,
    meta_sheet: str = "meta",
    transpose_sheet: str = "transpose_meta",
    row_key: str = "다리길이",
) -> Optional[float]:
    """Load leg length in cm used for xCOM calculation."""
    xlsm_path = str(Path(event_xlsm).resolve())

    # Wide meta sheet: keys in `subject`, subject names as columns
    try:
        df_meta = _read_excel_cached(xlsm_path, meta_sheet)
        if "subject" in df_meta.columns and subject in df_meta.columns:
            keys = df_meta["subject"].astype(str).str.strip()
            row = df_meta[keys == str(row_key).strip()]
            if len(row) >= 1:
                value = row.iloc[0][subject]
                if not pd.isna(value):
                    return float(value)
    except Exception:
        pass

    # Transpose meta: each row is subject
    try:
        df_transposed = _read_excel_cached(xlsm_path, transpose_sheet)
        if "subject" in df_transposed.columns and row_key in df_transposed.columns:
            hit = df_transposed[df_transposed["subject"].astype(str).str.strip() == str(subject).strip()]
            if len(hit) == 1:
                value = hit.iloc[0][row_key]
                if not pd.isna(value):
                    return float(value)
    except Exception:
        pass

    return None


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
