"""QA script: verify forceplate inertial subtraction against reference files.

Loads NPZ templates and compares velocity selection, interpolation weights,
and corrected values against shared reference data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

import _bootstrap

_bootstrap.ensure_src_on_path()
_REPO_ROOT = _bootstrap.REPO_ROOT

from replace_v3d.torque.forceplate_inertial import ForceplateInertialTemplate, load_forceplate_inertial_templates


_CHANNELS: Tuple[str, ...] = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def _pad_edge(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if int(arr.size) >= int(target_len):
        return arr
    if arr.size == 0:
        return np.zeros(int(target_len), dtype=float)
    return np.pad(arr, (0, int(target_len) - int(arr.size)), mode="edge")


def _template_matrix_from_policy(
    *,
    templates: Dict[int, ForceplateInertialTemplate],
    policy: str,
    velocity_int_used: int | None,
    velocity_int_lo: int | None,
    velocity_int_hi: int | None,
    interp_weight: float | None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "template_policy": policy,
        "template_velocity_int_used": velocity_int_used,
        "template_velocity_int_lo": velocity_int_lo,
        "template_velocity_int_hi": velocity_int_hi,
        "template_interp_weight": interp_weight,
    }

    if policy in ("exact", "nearest"):
        if velocity_int_used is None:
            raise ValueError(f"template_policy={policy} but template_velocity_int_used is null")
        if int(velocity_int_used) not in templates:
            raise ValueError(f"Template not found: velocity_int={velocity_int_used}")
        tmpl = templates[int(velocity_int_used)]
        mat = tmpl.as_matrix()
        info["template_len"] = int(mat.shape[0])
        info["template_n_trials"] = int(tmpl.n_trials)
        return mat, info

    if policy == "interpolate":
        if velocity_int_lo is None or velocity_int_hi is None or interp_weight is None:
            raise ValueError("template_policy=interpolate but lo/hi/w is missing")
        if int(velocity_int_lo) not in templates or int(velocity_int_hi) not in templates:
            raise ValueError(f"Template missing for interpolate: lo={velocity_int_lo}, hi={velocity_int_hi}")
        lo = templates[int(velocity_int_lo)]
        hi = templates[int(velocity_int_hi)]
        w = float(interp_weight)
        target_len = int(max(lo.length(), hi.length()))
        lo_mat = lo.as_matrix()
        hi_mat = hi.as_matrix()

        out = np.empty((target_len, 6), dtype=float)
        for j in range(6):
            out[:, j] = (1.0 - w) * _pad_edge(lo_mat[:, j], target_len) + w * _pad_edge(hi_mat[:, j], target_len)
        info["template_len"] = int(target_len)
        info["template_n_trials"] = int(lo.n_trials + hi.n_trials)
        return out, info

    if policy in ("skip", "none", ""):
        info["template_len"] = 0
        info["template_n_trials"] = 0
        return np.zeros((0, 6), dtype=float), info

    raise ValueError(f"Unknown template_policy: {policy!r}")


def _build_unloaded(
    *,
    n_frames: int,
    onset0: int,
    offset0: int,
    template_mat: np.ndarray,
) -> np.ndarray:
    """Match Stage01 diagnostics semantics:
    - unloaded is NaN before onset
    - onset..offset: template aligned to onset
    - after offset: hold last template value to the end
    """

    n_frames = int(n_frames)
    onset0_i = int(onset0)
    offset0_i = int(offset0)
    if onset0_i < 0:
        onset0_i = 0
    if offset0_i < 0:
        offset0_i = 0
    if onset0_i >= n_frames:
        onset0_i = n_frames - 1
    if offset0_i >= n_frames:
        offset0_i = n_frames - 1
    if offset0_i < onset0_i:
        onset0_i, offset0_i = offset0_i, onset0_i

    unloaded = np.full((n_frames, 6), np.nan, dtype=float)
    template_mat = np.asarray(template_mat, dtype=float)
    template_len = int(template_mat.shape[0])
    if template_len <= 0:
        return unloaded

    human_duration = int(offset0_i - onset0_i + 1)
    head_len = int(min(human_duration, template_len))
    if head_len > 0:
        unloaded[onset0_i : onset0_i + head_len, :] = template_mat[:head_len, :]

    if human_duration > template_len:
        tail_start = onset0_i + template_len
        tail_end = offset0_i + 1
        if tail_start < tail_end:
            unloaded[tail_start:tail_end, :] = template_mat[template_len - 1, :][None, :]

    if (offset0_i + 1) < n_frames:
        unloaded[offset0_i + 1 :, :] = template_mat[template_len - 1, :][None, :]

    return unloaded


def _build_corrected(measured: np.ndarray, unloaded: np.ndarray) -> np.ndarray:
    measured = np.asarray(measured, dtype=float)
    unloaded = np.asarray(unloaded, dtype=float)
    corrected = measured.copy()
    mask = np.isfinite(unloaded)
    corrected[mask] = measured[mask] - unloaded[mask]
    return corrected


def _equal_with_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def _first_mismatch(eq: np.ndarray) -> Tuple[int, int] | None:
    """Return (flat_index, ...) is inconvenient; return (row, col) for 2D eq mask."""

    eq = np.asarray(eq, dtype=bool)
    if bool(eq.all()):
        return None
    idx = int(np.argmin(eq.ravel()))  # first False
    row = idx // int(eq.shape[1])
    col = idx % int(eq.shape[1])
    return row, col


def _mdf(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    # Use polars first, then pandas (AGENTS rule).
    if not rows:
        return pl.DataFrame([]).to_pandas()
    return pl.DataFrame(rows).to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--diagnostics_parquet",
        default=str(
            Path("/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/shared_files/output/01_dataset")
            / "forceplate_subtract_diagnostics.parquet"
        ),
        help="shared_files Stage01 output parquet (forceplate_subtract_diagnostics.parquet)",
    )
    ap.add_argument(
        "--templates_npz",
        default="src/replace_v3d/torque/assets/fp_inertial_templates.npz",
        help="Templates NPZ built by scripts/torque_build_fp_inertial_templates.py",
    )
    ap.add_argument(
        "--out_report_csv",
        default="output/fp_inertial_subtract_diff_report.csv",
        help="Per-unit summary report CSV",
    )
    ap.add_argument(
        "--out_fail_examples_csv",
        default="output/fp_inertial_subtract_fail_examples.csv",
        help="First-mismatch examples CSV (only failing units)",
    )
    ap.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding (default utf-8-sig for Korean text in Excel)",
    )
    args = ap.parse_args()

    diagnostics_path = Path(args.diagnostics_parquet)
    templates_path = Path(args.templates_npz)
    if not templates_path.is_absolute():
        templates_path = _REPO_ROOT / templates_path
    out_report_csv = Path(args.out_report_csv)
    out_fail_csv = Path(args.out_fail_examples_csv)

    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Diagnostics parquet not found: {diagnostics_path}")
    if not templates_path.exists():
        raise FileNotFoundError(
            f"Templates NPZ not found: {templates_path}. Build it first with scripts/torque_build_fp_inertial_templates.py"
        )

    templates = load_forceplate_inertial_templates(templates_path)
    if not templates:
        raise RuntimeError(f"No templates loaded from: {templates_path}")

    cols: List[str] = [
        "subject",
        "velocity",
        "trial_num",
        "mocap_idx_local",
        "onset_local_100hz",
        "offset_local_100hz",
        "template_policy",
        "template_velocity_int_used",
        "template_velocity_int_lo",
        "template_velocity_int_hi",
        "template_interp_weight",
    ]
    for ch in _CHANNELS:
        cols.append(f"{ch}_measured_100hz")
        cols.append(f"{ch}_unloaded_100hz")
        cols.append(f"{ch}_corrected_100hz")

    df = pl.read_parquet(diagnostics_path, columns=cols).sort(
        ["subject", "velocity", "trial_num", "mocap_idx_local"]
    )

    groups = df.partition_by(["subject", "velocity", "trial_num"], as_dict=True)
    report_rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []

    for (subject, velocity, trial_num), g in groups.items():
        g = g.sort("mocap_idx_local")
        idx = g["mocap_idx_local"].to_numpy()
        n_frames = int(np.max(idx)) + 1 if idx.size else int(g.height)

        onset_vals = g["onset_local_100hz"].unique().to_list()
        offset_vals = g["offset_local_100hz"].unique().to_list()
        if len(onset_vals) != 1 or len(offset_vals) != 1:
            raise ValueError(f"Non-unique onset/offset for {subject}-{velocity}-{trial_num}: {onset_vals}, {offset_vals}")
        onset0 = int(onset_vals[0])
        offset0 = int(offset_vals[0])

        policy_vals = g["template_policy"].unique().to_list()
        if len(policy_vals) != 1:
            raise ValueError(f"Non-unique template_policy for {subject}-{velocity}-{trial_num}: {policy_vals}")
        policy = str(policy_vals[0])

        # Build measured/unloaded/corrected arrays from Stage01 outputs
        measured = np.full((n_frames, 6), np.nan, dtype=float)
        stage_unloaded = np.full((n_frames, 6), np.nan, dtype=float)
        stage_corrected = np.full((n_frames, 6), np.nan, dtype=float)
        for j, ch in enumerate(_CHANNELS):
            measured[idx, j] = g[f"{ch}_measured_100hz"].to_numpy()
            stage_unloaded[idx, j] = g[f"{ch}_unloaded_100hz"].to_numpy()
            stage_corrected[idx, j] = g[f"{ch}_corrected_100hz"].to_numpy()

        # Template selection mirrors diagnostics columns (ground truth for policy)
        v_used = g["template_velocity_int_used"].unique().to_list()
        v_lo = g["template_velocity_int_lo"].unique().to_list()
        v_hi = g["template_velocity_int_hi"].unique().to_list()
        w = g["template_interp_weight"].unique().to_list()
        template_mat, template_info = _template_matrix_from_policy(
            templates=templates,
            policy=policy,
            velocity_int_used=None if v_used == [None] else (None if len(v_used) != 1 else int(v_used[0])),
            velocity_int_lo=None if v_lo == [None] else (None if len(v_lo) != 1 else int(v_lo[0])),
            velocity_int_hi=None if v_hi == [None] else (None if len(v_hi) != 1 else int(v_hi[0])),
            interp_weight=None if w == [None] else (None if len(w) != 1 else float(w[0])),
        )

        pred_unloaded = _build_unloaded(n_frames=n_frames, onset0=onset0, offset0=offset0, template_mat=template_mat)
        pred_corrected = _build_corrected(measured, pred_unloaded)

        eq_unloaded = _equal_with_nan(pred_unloaded, stage_unloaded)
        eq_corrected = _equal_with_nan(pred_corrected, stage_corrected)

        unloaded_ok = bool(eq_unloaded.all())
        corrected_ok = bool(eq_corrected.all())

        # Summaries
        unloaded_mismatch_n = int(np.count_nonzero(~eq_unloaded))
        corrected_mismatch_n = int(np.count_nonzero(~eq_corrected))

        report = {
            "subject": subject,
            "velocity": float(velocity),
            "trial_num": int(trial_num),
            "n_frames_100hz": int(n_frames),
            "onset_local_100hz": int(onset0),
            "offset_local_100hz": int(offset0),
            "unloaded_ok": bool(unloaded_ok),
            "corrected_ok": bool(corrected_ok),
            "unloaded_mismatch_n": int(unloaded_mismatch_n),
            "corrected_mismatch_n": int(corrected_mismatch_n),
            **template_info,
        }

        report_rows.append(report)

        if not unloaded_ok or not corrected_ok:
            # record first mismatch for unloaded and/or corrected
            if not unloaded_ok:
                loc = _first_mismatch(eq_unloaded)
                assert loc is not None
                r, c = loc
                fail_rows.append(
                    {
                        "subject": subject,
                        "velocity": float(velocity),
                        "trial_num": int(trial_num),
                        "kind": "unloaded",
                        "channel": _CHANNELS[int(c)],
                        "mocap_idx_local": int(r),
                        "expected_stage01": float(stage_unloaded[r, c]),
                        "actual_replace_v3d": float(pred_unloaded[r, c]),
                        **template_info,
                    }
                )
            if not corrected_ok:
                loc = _first_mismatch(eq_corrected)
                assert loc is not None
                r, c = loc
                fail_rows.append(
                    {
                        "subject": subject,
                        "velocity": float(velocity),
                        "trial_num": int(trial_num),
                        "kind": "corrected",
                        "channel": _CHANNELS[int(c)],
                        "mocap_idx_local": int(r),
                        "expected_stage01": float(stage_corrected[r, c]),
                        "actual_replace_v3d": float(pred_corrected[r, c]),
                        **template_info,
                    }
                )

    rep_pd = _mdf(report_rows)
    fail_pd = _mdf(fail_rows)

    out_report_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fail_csv.parent.mkdir(parents=True, exist_ok=True)
    rep_pd.to_csv(out_report_csv, index=False, encoding=str(args.encoding))
    fail_pd.to_csv(out_fail_csv, index=False, encoding=str(args.encoding))

    total_units = int(len(report_rows))
    failed_units = int(rep_pd.loc[~(rep_pd["unloaded_ok"] & rep_pd["corrected_ok"])].shape[0]) if total_units else 0
    print("[OK] Forceplate inertial subtract verification vs shared_files Stage01")
    print(f"     diagnostics: {diagnostics_path}")
    print(f"     templates:   {templates_path}")
    print(f"     units:       {total_units}")
    print(f"     failed:      {failed_units}")
    print(f"     report_csv:  {out_report_csv}")
    print(f"     fail_csv:    {out_fail_csv}")

    if failed_units > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
