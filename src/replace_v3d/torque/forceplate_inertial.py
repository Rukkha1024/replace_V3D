from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .forceplate import ForcePlatform


MissingVelocityPolicy = Literal["skip", "nearest", "interpolate"]


@dataclass(frozen=True)
class ForceplateInertialTemplate:
    """Per-velocity inertial (unloaded) template at mocap rate (typically 100 Hz).

    This mirrors Stage01's direct-subtract logic in `shared_files/stages/01_build_dataset.py`:
    - A per-velocity template is built from unloaded trials (forceplate_3.csv)
    - Each channel is baseline-shifted so template[0] == 0
    - At application time, the template is aligned to platform onset, then the last
      value is held constant after the platform offset.
    """

    velocity_int: int
    unload_range_frames: int
    n_trials: int
    fx: np.ndarray  # (L,)
    fy: np.ndarray  # (L,)
    fz: np.ndarray  # (L,)
    mx: np.ndarray  # (L,)
    my: np.ndarray  # (L,)
    mz: np.ndarray  # (L,)
    meta: Dict[str, Any]

    def length(self) -> int:
        return int(self.fx.size)

    def as_matrix(self) -> np.ndarray:
        """Return (L,6) matrix in [Fx,Fy,Fz,Mx,My,Mz] order."""

        return np.column_stack(
            [
                np.asarray(self.fx, dtype=float),
                np.asarray(self.fy, dtype=float),
                np.asarray(self.fz, dtype=float),
                np.asarray(self.mx, dtype=float),
                np.asarray(self.my, dtype=float),
                np.asarray(self.mz, dtype=float),
            ]
        )


def load_forceplate_inertial_templates(npz_path: str | Path) -> Dict[int, ForceplateInertialTemplate]:
    """Load templates saved by `scripts/torque_build_fp_inertial_templates.py`.

    The file format is a simple NPZ with keys:
      - velocity_ints: (K,)
      - v{V}_fx, v{V}_fy, v{V}_fz, v{V}_mx, v{V}_my, v{V}_mz
      - v{V}_unload_range_frames (scalar)
      - v{V}_n_trials (scalar)
      - v{V}_meta_json (string)
    """

    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    if "velocity_ints" not in data:
        raise ValueError(f"Invalid templates npz (missing 'velocity_ints'): {npz_path}")

    velocity_ints = [int(v) for v in np.asarray(data["velocity_ints"]).tolist()]
    templates: Dict[int, ForceplateInertialTemplate] = {}
    for v in velocity_ints:
        key_prefix = f"v{v}_"
        fx = np.asarray(data[f"{key_prefix}fx"], dtype=float)
        fy = np.asarray(data[f"{key_prefix}fy"], dtype=float)
        fz = np.asarray(data[f"{key_prefix}fz"], dtype=float)
        mx = np.asarray(data[f"{key_prefix}mx"], dtype=float)
        my = np.asarray(data[f"{key_prefix}my"], dtype=float)
        mz = np.asarray(data[f"{key_prefix}mz"], dtype=float)

        unload_range_frames = int(np.asarray(data[f"{key_prefix}unload_range_frames"]).item())
        n_trials = int(np.asarray(data[f"{key_prefix}n_trials"]).item())

        meta_json = None
        meta_key = f"{key_prefix}meta_json"
        if meta_key in data:
            meta_json = str(np.asarray(data[meta_key]).item())
        meta: Dict[str, Any]
        if meta_json:
            try:
                meta = dict(json.loads(meta_json))
            except Exception:
                meta = {"meta_json": meta_json}
        else:
            meta = {}

        templates[int(v)] = ForceplateInertialTemplate(
            velocity_int=int(v),
            unload_range_frames=int(unload_range_frames),
            n_trials=int(n_trials),
            fx=fx,
            fy=fy,
            fz=fz,
            mx=mx,
            my=my,
            mz=mz,
            meta=meta,
        )

    return templates


def _pad_edge(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if int(arr.size) >= int(target_len):
        return arr
    if arr.size == 0:
        return np.zeros(int(target_len), dtype=float)
    return np.pad(arr, (0, int(target_len) - int(arr.size)), mode="edge")


def _select_template(
    templates: Dict[int, ForceplateInertialTemplate],
    *,
    velocity: float,
    missing_policy: MissingVelocityPolicy,
) -> Tuple[Optional[ForceplateInertialTemplate], Dict[str, Any]]:
    """Select or build a template given a (possibly non-integer) velocity."""

    v_int = int(round(float(velocity)))
    keys = sorted(int(k) for k in templates.keys())

    info: Dict[str, Any] = {
        "velocity": float(velocity),
        "velocity_int": int(v_int),
        "template_policy": "skip",
        "template_velocity_int_used": None,
        "template_velocity_int_lo": None,
        "template_velocity_int_hi": None,
        "template_interp_weight": None,
    }

    if v_int in templates:
        info.update(
            {
                "template_policy": "exact",
                "template_velocity_int_used": int(v_int),
            }
        )
        return templates[int(v_int)], info

    if not keys or missing_policy == "skip":
        return None, info

    v_val = float(velocity)
    if missing_policy == "nearest":
        used = min(keys, key=lambda k: (abs(float(k) - v_val), k))
        info.update(
            {
                "template_policy": "nearest",
                "template_velocity_int_used": int(used),
            }
        )
        return templates[int(used)], info

    # interpolate
    lower = [k for k in keys if float(k) <= v_val]
    upper = [k for k in keys if float(k) >= v_val]
    if not lower or not upper:
        # out of range -> nearest
        used = min(keys, key=lambda k: (abs(float(k) - v_val), k))
        info.update(
            {
                "template_policy": "nearest",
                "template_velocity_int_used": int(used),
            }
        )
        return templates[int(used)], info

    k_lo = int(lower[-1])
    k_hi = int(upper[0])
    if k_lo == k_hi:
        info.update(
            {
                "template_policy": "nearest",
                "template_velocity_int_used": int(k_lo),
            }
        )
        return templates[int(k_lo)], info

    w = float((v_val - float(k_lo)) / float(k_hi - k_lo))

    lo = templates[int(k_lo)]
    hi = templates[int(k_hi)]

    # Stage01 behavior: pad to max length (edge) then interpolate.
    target_len = int(
        max(
            lo.fx.size,
            lo.fy.size,
            lo.fz.size,
            lo.mx.size,
            lo.my.size,
            lo.mz.size,
            hi.fx.size,
            hi.fy.size,
            hi.fz.size,
            hi.mx.size,
            hi.my.size,
            hi.mz.size,
        )
    )

    fx = (1.0 - w) * _pad_edge(lo.fx, target_len) + w * _pad_edge(hi.fx, target_len)
    fy = (1.0 - w) * _pad_edge(lo.fy, target_len) + w * _pad_edge(hi.fy, target_len)
    fz = (1.0 - w) * _pad_edge(lo.fz, target_len) + w * _pad_edge(hi.fz, target_len)
    mx = (1.0 - w) * _pad_edge(lo.mx, target_len) + w * _pad_edge(hi.mx, target_len)
    my = (1.0 - w) * _pad_edge(lo.my, target_len) + w * _pad_edge(hi.my, target_len)
    mz = (1.0 - w) * _pad_edge(lo.mz, target_len) + w * _pad_edge(hi.mz, target_len)

    merged = ForceplateInertialTemplate(
        velocity_int=int(v_int),
        unload_range_frames=int(max(0, target_len - 1)),
        n_trials=int(lo.n_trials + hi.n_trials),
        fx=fx,
        fy=fy,
        fz=fz,
        mx=mx,
        my=my,
        mz=mz,
        meta={
            "interpolated": True,
            "lo": int(k_lo),
            "hi": int(k_hi),
            "w": float(w),
            "lo_meta": lo.meta,
            "hi_meta": hi.meta,
        },
    )

    info.update(
        {
            "template_policy": "interpolate",
            "template_velocity_int_used": None,
            "template_velocity_int_lo": int(k_lo),
            "template_velocity_int_hi": int(k_hi),
            "template_interp_weight": float(w),
        }
    )
    return merged, info


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den2 = np.where(np.abs(den) < eps, np.nan, den)
    return num / den2


def _cop_plate_xy(F_plate: np.ndarray, M_plate: np.ndarray) -> np.ndarray:
    F_plate = np.asarray(F_plate, dtype=float)
    M_plate = np.asarray(M_plate, dtype=float)
    Fz = F_plate[:, 2]
    cop_x = _safe_div(-M_plate[:, 1], Fz)
    cop_y = _safe_div(M_plate[:, 0], Fz)
    return np.column_stack([cop_x, cop_y])


def _corners_plate_xy(fp: ForcePlatform) -> np.ndarray:
    # row-vector convention used in forceplate.py: v_lab = v_plate @ R.T
    # so v_plate = v_lab @ R (for orthonormal R).
    corners_rel_lab = np.asarray(fp.corners_lab, dtype=float) - np.asarray(fp.origin_lab, dtype=float)[None, :]
    corners_plate = corners_rel_lab @ np.asarray(fp.R_pl2lab, dtype=float)
    return corners_plate[:, 0:2]


def _cop_in_bounds_mask(cop_xy: np.ndarray, corners_xy: np.ndarray, *, margin_m: float = 0.0) -> np.ndarray:
    cop_xy = np.asarray(cop_xy, dtype=float)
    corners_xy = np.asarray(corners_xy, dtype=float)
    if corners_xy.shape != (4, 2):
        raise ValueError(f"Expected corners_xy shape (4,2), got {corners_xy.shape}")
    x_min = float(np.nanmin(corners_xy[:, 0])) - float(margin_m)
    x_max = float(np.nanmax(corners_xy[:, 0])) + float(margin_m)
    y_min = float(np.nanmin(corners_xy[:, 1])) - float(margin_m)
    y_max = float(np.nanmax(corners_xy[:, 1])) + float(margin_m)
    x = cop_xy[:, 0]
    y = cop_xy[:, 1]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


def apply_forceplate_inertial_subtract(
    analog_avg: np.ndarray,
    fp: ForcePlatform,
    *,
    velocity: float,
    onset0: int,
    offset0: int,
    templates: Dict[int, ForceplateInertialTemplate],
    missing_policy: MissingVelocityPolicy = "skip",
    qc_fz_threshold_n: float = 20.0,
    qc_margin_m: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Subtract inertial template from the selected platform channels.

    Parameters
    ----------
    analog_avg:
        Averaged analog array, shape (n_frames, n_channels).
    fp:
        ForcePlatform mapping (channel_indices_0based points to Fx..Mz).
    velocity:
        Trial velocity (may be non-integer).
    onset0, offset0:
        Platform onset/offset frame indices *in the trimmed file* (0-based, mocap rate).
    templates:
        Dict of templates keyed by velocity_int.
    missing_policy:
        skip | nearest | interpolate

    Returns
    -------
    analog_corr:
        Copy of analog_avg with selected platform channels corrected.
    info:
        Dict with template selection + basic COP/Fz QC.
    """

    analog_avg = np.asarray(analog_avg, dtype=float)
    n_frames = int(analog_avg.shape[0])
    idx = np.asarray(fp.channel_indices_0based, dtype=int)
    if idx.size != 6:
        raise ValueError(f"ForcePlatform.channel_indices_0based must have 6 entries, got {idx}")

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

    tmpl, info = _select_template(templates, velocity=float(velocity), missing_policy=missing_policy)
    info.update(
        {
            "missing_policy": str(missing_policy),
            "platform_onset0": int(onset0_i),
            "platform_offset0": int(offset0_i),
        }
    )

    # Prepare QC values even if we skip
    def _qc_for(analog: np.ndarray) -> Dict[str, Any]:
        F_plate = analog[:, idx[0:3]]
        M_plate = analog[:, idx[3:6]]
        cop_xy = _cop_plate_xy(F_plate, M_plate)
        corners_xy = _corners_plate_xy(fp)
        in_bounds = _cop_in_bounds_mask(cop_xy, corners_xy, margin_m=float(qc_margin_m))
        win = np.zeros(n_frames, dtype=bool)
        win[onset0_i : offset0_i + 1] = True
        valid = (
            win
            & (np.abs(F_plate[:, 2]) >= float(qc_fz_threshold_n))
            & np.isfinite(cop_xy[:, 0])
            & np.isfinite(cop_xy[:, 1])
        )
        denom = int(np.count_nonzero(valid))
        if denom <= 0:
            return {
                "qc_valid_n": 0,
                "qc_cop_in_bounds_frac": float("nan"),
                "qc_fz_positive_frac": float("nan"),
            }
        return {
            "qc_valid_n": denom,
            "qc_cop_in_bounds_frac": float(np.count_nonzero(in_bounds & valid) / denom),
            "qc_fz_positive_frac": float(np.count_nonzero((F_plate[:, 2] > 0) & valid) / denom),
        }

    qc_before = _qc_for(analog_avg)
    info.update({f"before_{k}": v for k, v in qc_before.items()})

    if tmpl is None:
        info["applied"] = False
        info["reason"] = "no_template"
        # still include QC thresholds
        info["qc_fz_threshold_n"] = float(qc_fz_threshold_n)
        info["qc_margin_m"] = float(qc_margin_m)
        return analog_avg, info

    template_mat = tmpl.as_matrix()
    template_len = int(template_mat.shape[0])
    human_duration = int(offset0_i - onset0_i + 1)

    unloaded = np.zeros((n_frames, 6), dtype=float)
    head_len = int(min(human_duration, template_len))
    if head_len > 0:
        unloaded[onset0_i : onset0_i + head_len, :] = template_mat[:head_len, :]

    # If the human duration exceeds template length, hold the last template value until offset.
    if human_duration > template_len:
        tail_start = onset0_i + template_len
        tail_end = offset0_i + 1
        if tail_start < tail_end:
            unloaded[tail_start:tail_end, :] = template_mat[template_len - 1, :][None, :]

    # Stage01 behavior: always hold the last template value after offset.
    if (offset0_i + 1) < n_frames:
        unloaded[offset0_i + 1 :, :] = template_mat[template_len - 1, :][None, :]

    analog_corr = analog_avg.copy()
    analog_corr[:, idx] = analog_corr[:, idx] - unloaded

    qc_after = _qc_for(analog_corr)
    info.update({f"after_{k}": v for k, v in qc_after.items()})

    info.update(
        {
            "applied": True,
            "template_len": int(template_len),
            "unload_range_frames": int(max(0, template_len - 1)),
            "template_n_trials": int(tmpl.n_trials),
            "qc_fz_threshold_n": float(qc_fz_threshold_n),
            "qc_margin_m": float(qc_margin_m),
        }
    )

    # Simple QC failure heuristic (for warnings/strict mode)
    before_frac = float(info.get("before_qc_cop_in_bounds_frac", float("nan")))
    after_frac = float(info.get("after_qc_cop_in_bounds_frac", float("nan")))
    qc_failed = False
    if np.isfinite(after_frac) and after_frac < 0.5:
        qc_failed = True
    if np.isfinite(before_frac) and np.isfinite(after_frac) and (after_frac + 1e-9) < (before_frac - 0.2):
        qc_failed = True
    info["qc_failed"] = bool(qc_failed)

    return analog_corr, info
