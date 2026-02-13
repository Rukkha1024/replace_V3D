#!/usr/bin/env python
"""Run COM / XCoM / BoS / MoS pipeline for perturbation trials.

This script is intended to be called by an agent (Codex CLI) or directly by a user.

Key features:
- Reads marker trajectories from C3D (no external C3D libs)
- Computes a Visual3D-like COM using a segment-centroid proxy
- Computes XCoM (Hof) using subject leg length
- Builds marker-based BoS polygons from foot landmarks (virtual medial/lateral edges)
- Computes signed MoS as shortest distance from XCoM to BoS polygon boundary
- Optionally validates COM vs a Visual3D COM export (xlsx)

Outputs are saved as CSV files in the requested output directory.

Examples:
    python scripts/run_mos_pipeline.py \
      --c3d /path/to/251112_KUO_perturb_60_001.c3d \
      --events /path/to/perturb_inform.xlsm \
      --subject "김우연" \
      --anthro ./assets/example_anthro_kimwooyeon.yaml \
      --out_dir ./outputs \
      --v3d_com_ref /path/to/251112_KUO_perturb_60_001_COM.xlsx
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Local imports (work both when run from repo root or from scripts/)
import os, sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from c3d_reader import read_c3d_points, detect_common_prefix, strip_prefix
from com_proxy import compute_com_proxy
from mos_metrics import compute_foot_landmarks, compute_mos
from events import (
    load_platform_sheet,
    find_event_row,
    infer_step_side,
    infer_trial_velocity_from_filename,
    map_events_into_c3d_frames,
)


def load_anthropometrics(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Anthropometrics file must be a YAML mapping")
    return data


def require_fields(d: Dict[str, Any], fields: List[str]) -> None:
    missing = [k for k in fields if k not in d or d[k] is None]
    if missing:
        raise ValueError(
            "Missing required anthropometrics fields: " + ", ".join(missing) +
            ". Provide them in meters."
        )


def load_v3d_com_xlsx(path: str | Path) -> np.ndarray:
    """Load Visual3D exported COM xlsx (LINK_MODEL_BASED::ORIGINAL::COM style)."""
    df_raw = pd.read_excel(path, header=None)
    # Data start at row index 2; columns: File | Frame | X | Y | Z
    df = df_raw.iloc[2:, 1:5].copy()
    df.columns = ["Frame", "X", "Y", "Z"]
    df = df.apply(pd.to_numeric, errors='coerce')
    arr = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    return arr


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return float('nan')
    if np.allclose(np.std(a), 0) or np.allclose(np.std(b), 0):
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def save_bos_hulls_csv(hulls: List[Optional[np.ndarray]], out_path: Path, plane_axes: Tuple[int, int]) -> None:
    rows = []
    for frame, hull in enumerate(hulls):
        if hull is None:
            continue
        for vi, (x, y) in enumerate(hull):
            rows.append({
                "frame": frame,
                "vertex": vi,
                f"axis{plane_axes[0]}": x,
                f"axis{plane_axes[1]}": y,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_one_trial(
    c3d_path: Path,
    out_dir: Path,
    anthro: Dict[str, Any],
    events_path: Optional[Path] = None,
    subject: Optional[str] = None,
    velocity: Optional[float] = None,
    trial: Optional[int] = None,
    v3d_com_ref: Optional[Path] = None,
    vertical_axis: int = 2,
    plane_axes: Tuple[int, int] = (0, 1),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    header, labels, xyz = read_c3d_points(c3d_path)
    fs = float(header.frame_rate_hz)
    n_frames = xyz.shape[0]

    # Prefix handling
    prefix = detect_common_prefix(labels)
    base_labels = strip_prefix(labels, prefix)

    # COM proxy
    com, com_info = compute_com_proxy(xyz, labels, prefix=prefix)

    # Optional V3D COM validation
    validation_text = None
    if v3d_com_ref is not None:
        v3d_arr = load_v3d_com_xlsx(v3d_com_ref)
        # Align lengths (common in trimmed exports)
        m = min(len(v3d_arr), len(com))
        corrs = [corrcoef_safe(com[:m, i], v3d_arr[:m, i]) for i in range(3)]
        validation_text = (
            f"COM validation vs V3D ({v3d_com_ref.name})\n"
            f"- Corr X: {corrs[0]:.6f}\n"
            f"- Corr Y: {corrs[1]:.6f}\n"
            f"- Corr Z: {corrs[2]:.6f}\n"
            f"- Prefix detected: '{prefix}'\n"
            f"- Missing segments: {com_info.missing_segments}\n"
        )
        (out_dir / f"{c3d_path.stem}_COM_validation.txt").write_text(validation_text, encoding='utf-8')

    # Anthropometrics required fields
    require_fields(anthro, ["leg_length_m", "foot_length_m", "foot_width_m", "ankle_width_m", "knee_width_m"])
    leg_length_m = float(anthro["leg_length_m"])
    foot_width_m = float(anthro["foot_width_m"])
    ankle_width_m = float(anthro["ankle_width_m"])

    # Determine up axis vector
    up_axis = np.zeros(3, dtype=float)
    up_axis[vertical_axis] = 1.0

    # Marker lookup by base names (with prefix in stored labels)
    label_to_idx = {lab: i for i, lab in enumerate(base_labels)}

    def must_get(name: str) -> np.ndarray:
        if name not in label_to_idx:
            raise ValueError(f"Missing required marker '{name}' in {c3d_path.name}")
        return xyz[:, label_to_idx[name], :]

    # Feet
    LHEE = must_get("LHEE")
    LTOE = must_get("LTOE")
    LANK = must_get("LANK")
    LMed = must_get("LFoot_3")

    RHEE = must_get("RHEE")
    RTOE = must_get("RTOE")
    RANK = must_get("RANK")
    RMed = must_get("RFoot_3")

    left_foot = compute_foot_landmarks(
        heel=LHEE,
        toe=LTOE,
        lat_ankle=LANK,
        med_ankle=LMed,
        forefoot_width_m=foot_width_m,
        rearfoot_width_m=ankle_width_m,
        up_axis=up_axis,
    )

    right_foot = compute_foot_landmarks(
        heel=RHEE,
        toe=RTOE,
        lat_ankle=RANK,
        med_ankle=RMed,
        forefoot_width_m=foot_width_m,
        rearfoot_width_m=ankle_width_m,
        up_axis=up_axis,
    )

    # Events mapping
    platform_onset_frame = None
    platform_offset_frame = None
    step_onset_frame = None
    step_side = None

    if events_path is not None:
        if subject is None:
            raise ValueError("--subject is required when --events is provided")

        if velocity is None or trial is None:
            vel2, tr2 = infer_trial_velocity_from_filename(c3d_path.name)
            velocity = velocity if velocity is not None else vel2
            trial = trial if trial is not None else tr2

        if velocity is None or trial is None:
            raise ValueError("Could not infer velocity/trial from filename; provide --velocity and --trial")

        df_events = load_platform_sheet(events_path)
        ev = find_event_row(df_events, subject=subject, velocity=float(velocity), trial=int(trial))

        step_side = infer_step_side(ev.state)

        po, pf, so = map_events_into_c3d_frames(
            ev,
            c3d_first_frame=int(header.first_frame),
            n_frames=n_frames,
            assume_trimmed_rule=True,
        )
        platform_onset_frame, platform_offset_frame, step_onset_frame = po, pf, so

    # Compute MoS
    mos_res = compute_mos(
        com=com,
        fs_hz=fs,
        leg_length_m=leg_length_m,
        left_foot=left_foot,
        right_foot=right_foot,
        plane_axes=plane_axes,
        step_onset_frame=step_onset_frame,
        step_side=step_side,
    )

    # Save outputs
    t = np.arange(n_frames) / fs

    com_df = pd.DataFrame({
        "frame": np.arange(n_frames),
        "time_s": t,
        "COM_X": com[:, 0],
        "COM_Y": com[:, 1],
        "COM_Z": com[:, 2],
    })
    com_df.to_csv(out_dir / f"{c3d_path.stem}_COM.csv", index=False)

    xcom_df = pd.DataFrame({
        "frame": np.arange(n_frames),
        "time_s": t,
        "XCOM_X": mos_res.xcom[:, 0],
        "XCOM_Y": mos_res.xcom[:, 1],
        "XCOM_Z": mos_res.xcom[:, 2],
        "vCOM_X": mos_res.com_vel[:, 0],
        "vCOM_Y": mos_res.com_vel[:, 1],
        "vCOM_Z": mos_res.com_vel[:, 2],
        "omega0": mos_res.omega0,
    })
    xcom_df.to_csv(out_dir / f"{c3d_path.stem}_XCOM.csv", index=False)

    mos_df = pd.DataFrame({
        "frame": np.arange(n_frames),
        "time_s": t,
        "MOS": mos_res.mos,
        "plane_axes": [f"{plane_axes[0]},{plane_axes[1]}"] * n_frames,
    })
    # Add events as metadata columns (constant)
    mos_df["platform_onset_frame"] = platform_onset_frame
    mos_df["platform_offset_frame"] = platform_offset_frame
    mos_df["step_onset_frame"] = step_onset_frame
    mos_df["step_side"] = step_side
    mos_df.to_csv(out_dir / f"{c3d_path.stem}_MOS.csv", index=False)

    save_bos_hulls_csv(mos_res.bos_hulls_2d, out_dir / f"{c3d_path.stem}_BOS_vertices.csv", plane_axes)

    # Save a compact JSON summary
    summary = {
        "c3d": str(c3d_path),
        "n_frames": int(n_frames),
        "fs_hz": float(fs),
        "prefix": prefix,
        "anthro_subject": anthro.get("subject"),
        "leg_length_m": leg_length_m,
        "plane_axes": list(plane_axes),
        "vertical_axis": int(vertical_axis),
        "events": {
            "platform_onset_frame": platform_onset_frame,
            "platform_offset_frame": platform_offset_frame,
            "step_onset_frame": step_onset_frame,
            "step_side": step_side,
        },
        "com_proxy": {
            "weights_sum": com_info.weights_sum,
            "missing_segments": com_info.missing_segments,
        },
    }
    (out_dir / f"{c3d_path.stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_plane_axes(s: str) -> Tuple[int, int]:
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError("--plane_axes must be like '0,1'")
    return int(parts[0]), int(parts[1])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--c3d", type=str, help="Path to a single C3D file")
    ap.add_argument("--c3d_dir", type=str, help="Directory containing C3D files")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")

    ap.add_argument("--anthro", type=str, required=True, help="Anthropometrics YAML (meters)")

    ap.add_argument("--events", type=str, help="perturb_inform.xlsm path")
    ap.add_argument("--subject", type=str, help="Subject name (must match events sheet)")
    ap.add_argument("--velocity", type=float, help="Trial velocity (optional; inferred from filename if absent)")
    ap.add_argument("--trial", type=int, help="Trial index (optional; inferred from filename if absent)")

    ap.add_argument("--v3d_com_ref", type=str, help="Optional V3D COM export xlsx for validation")

    ap.add_argument("--vertical_axis", type=int, default=2, help="Index of global up axis (0=X,1=Y,2=Z)")
    ap.add_argument("--plane_axes", type=str, default="0,1", help="Ground plane axes, e.g. '0,1' for X-Y")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    anthro = load_anthropometrics(args.anthro)

    plane_axes = parse_plane_axes(args.plane_axes)

    events_path = Path(args.events) if args.events else None
    v3d_ref = Path(args.v3d_com_ref) if args.v3d_com_ref else None

    c3d_files: List[Path] = []
    if args.c3d:
        c3d_files.append(Path(args.c3d))
    if args.c3d_dir:
        c3d_files.extend(sorted(Path(args.c3d_dir).glob("*.c3d")))

    if not c3d_files:
        raise SystemExit("Provide --c3d or --c3d_dir")

    for p in c3d_files:
        run_one_trial(
            c3d_path=p,
            out_dir=out_dir,
            anthro=anthro,
            events_path=events_path,
            subject=args.subject,
            velocity=args.velocity,
            trial=args.trial,
            v3d_com_ref=v3d_ref,
            vertical_axis=int(args.vertical_axis),
            plane_axes=plane_axes,
        )


if __name__ == "__main__":
    main()
