"""Relative joint angular velocity from segment rotation matrices.

Implements w = vee(Rdot * R^T), then builds joint-relative velocity as:
  w_rel_lab = w_moving_lab - w_reference_lab
and resolves it in reference and moving frames (ref/mov). Export is deg/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replace_v3d.joint_angles.v3d_joint_angles import SegmentFrames


def _gradient(x: np.ndarray, dt: float) -> np.ndarray:
    if x.shape[0] < 3:
        return np.gradient(x, dt, axis=0)
    return np.gradient(x, dt, axis=0, edge_order=2)


def _skew_to_vec(S: np.ndarray) -> np.ndarray:
    """Convert skew-symmetric matrix to vector w (so that S @ v == w x v)."""

    if S.ndim != 3 or S.shape[1:] != (3, 3):
        raise ValueError(f"S must have shape (T,3,3). Got {S.shape}")
    return np.column_stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]])


def segment_angular_velocity_lab(R: np.ndarray, rate_hz: float) -> np.ndarray:
    """Segment angular velocity vector resolved in lab coordinates (rad/s)."""

    R_arr = np.asarray(R, dtype=float)
    if R_arr.ndim != 3 or R_arr.shape[1:] != (3, 3):
        raise ValueError(f"R must have shape (T,3,3). Got {R_arr.shape}")

    dt = 1.0 / float(rate_hz)
    Rdot = _gradient(R_arr, dt)
    Omega = np.matmul(Rdot, np.transpose(R_arr, (0, 2, 1)))
    return _skew_to_vec(Omega)


def segment_angular_velocity_segment(R: np.ndarray, rate_hz: float) -> np.ndarray:
    """Segment angular velocity vector resolved in the segment frame (rad/s)."""

    w_lab = segment_angular_velocity_lab(R, rate_hz=rate_hz)
    R_arr = np.asarray(R, dtype=float)
    return np.einsum("tji,tj->ti", R_arr, w_lab)


@dataclass(frozen=True)
class RelativeAngularVelocity:
    lab: np.ndarray
    reference: np.ndarray
    moving: np.ndarray


def relative_angular_velocity(ref: np.ndarray, moving: np.ndarray, rate_hz: float) -> RelativeAngularVelocity:
    """Relative angular velocity between reference and moving segments (rad/s)."""

    w_ref_lab = segment_angular_velocity_lab(ref, rate_hz=rate_hz)
    w_mov_lab = segment_angular_velocity_lab(moving, rate_hz=rate_hz)
    w_rel_lab = w_mov_lab - w_ref_lab

    ref_arr = np.asarray(ref, dtype=float)
    mov_arr = np.asarray(moving, dtype=float)
    w_rel_ref = np.einsum("tji,tj->ti", ref_arr, w_rel_lab)
    w_rel_mov = np.einsum("tji,tj->ti", mov_arr, w_rel_lab)
    return RelativeAngularVelocity(lab=w_rel_lab, reference=w_rel_ref, moving=w_rel_mov)


def compute_joint_angular_velocity_columns(frames: SegmentFrames, rate_hz: float) -> dict[str, np.ndarray]:
    """Compute strict V3D-style joint angular velocity columns (deg/s).

    Column naming contract:
    - <Joint>_<Side>_ref_[X|Y|Z]_deg_s
    - <Joint>_<Side>_mov_[X|Y|Z]_deg_s
    - Trunk/Neck omit side.
    """

    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Hip_L": (frames.pelvis, frames.thigh_L),
        "Hip_R": (frames.pelvis, frames.thigh_R),
        "Knee_L": (frames.thigh_L, frames.shank_L),
        "Knee_R": (frames.thigh_R, frames.shank_R),
        "Ankle_L": (frames.shank_L, frames.foot_L),
        "Ankle_R": (frames.shank_R, frames.foot_R),
        "Trunk": (frames.pelvis, frames.thorax),
        "Neck": (frames.thorax, frames.head),
    }

    out: dict[str, np.ndarray] = {}
    for joint, (ref, mov) in pairs.items():
        rel = relative_angular_velocity(ref, mov, rate_hz=rate_hz)
        ref_deg_s = np.degrees(rel.reference)
        mov_deg_s = np.degrees(rel.moving)
        for axis, j in zip(("X", "Y", "Z"), range(3)):
            out[f"{joint}_ref_{axis}_deg_s"] = ref_deg_s[:, j]
            out[f"{joint}_mov_{axis}_deg_s"] = mov_deg_s[:, j]
    return out

