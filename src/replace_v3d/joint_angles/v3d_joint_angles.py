from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from replace_v3d.com import compute_joint_centers


@dataclass(frozen=True)
class V3DJointAngles3D:
    """3D joint angles (deg) in Visual3D-like intrinsic XYZ order.

    Arrays are shape (T,), where T is the number of frames analyzed (end_frame).
    """

    hip_L_X: np.ndarray
    hip_L_Y: np.ndarray
    hip_L_Z: np.ndarray

    hip_R_X: np.ndarray
    hip_R_Y: np.ndarray
    hip_R_Z: np.ndarray

    knee_L_X: np.ndarray
    knee_L_Y: np.ndarray
    knee_L_Z: np.ndarray

    knee_R_X: np.ndarray
    knee_R_Y: np.ndarray
    knee_R_Z: np.ndarray

    ankle_L_X: np.ndarray
    ankle_L_Y: np.ndarray
    ankle_L_Z: np.ndarray

    ankle_R_X: np.ndarray
    ankle_R_Y: np.ndarray
    ankle_R_Z: np.ndarray

    trunk_X: np.ndarray
    trunk_Y: np.ndarray
    trunk_Z: np.ndarray

    neck_X: np.ndarray
    neck_Y: np.ndarray
    neck_Z: np.ndarray


@dataclass(frozen=True)
class SegmentFrames:
    """Segment coordinate systems; columns are [X, Y, Z] unit vectors in global."""

    pelvis: np.ndarray  # (T,3,3)
    thorax: np.ndarray  # (T,3,3)
    head: np.ndarray  # (T,3,3)
    thigh_L: np.ndarray  # (T,3,3)
    thigh_R: np.ndarray  # (T,3,3)
    shank_L: np.ndarray  # (T,3,3)
    shank_R: np.ndarray  # (T,3,3)
    foot_L: np.ndarray  # (T,3,3)
    foot_R: np.ndarray  # (T,3,3)


def _norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / _norm(v, eps=eps)


def _project_to_plane(v: np.ndarray, n_unit: np.ndarray) -> np.ndarray:
    """Project vector v onto plane with normal n_unit (both shape (...,3))."""

    return v - np.sum(v * n_unit, axis=-1, keepdims=True) * n_unit


def _maybe_flip_xy(x: np.ndarray, y: np.ndarray, right_hint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure X axis points to 'right_hint' direction by flipping (x,y) together."""

    dot = np.sum(x * right_hint, axis=-1)
    flip = dot < 0
    if np.any(flip):
        x = x.copy()
        y = y.copy()
        x[flip] *= -1.0
        y[flip] *= -1.0
    return x, y


def _frame_from_xz(x0: np.ndarray, z0: np.ndarray, right_hint: Optional[np.ndarray] = None) -> np.ndarray:
    """Build right-handed SCS from approximate X and Z."""

    x = _normalize(x0)
    z = _normalize(z0)
    y = _normalize(np.cross(z, x))
    x = _normalize(np.cross(y, z))

    if right_hint is not None:
        x, y = _maybe_flip_xy(x, y, right_hint=right_hint)

    return np.stack([x, y, z], axis=-1)


def _frame_from_xy(x0: np.ndarray, y0: np.ndarray, right_hint: Optional[np.ndarray] = None) -> np.ndarray:
    """Build right-handed SCS from approximate X and Y."""

    x = _normalize(x0)
    y = _normalize(y0)
    z = _normalize(np.cross(x, y))
    y = _normalize(np.cross(z, x))
    x = _normalize(x)

    if right_hint is not None:
        x, y = _maybe_flip_xy(x, y, right_hint=right_hint)

    return np.stack([x, y, z], axis=-1)


def _frame_from_yz(y0: np.ndarray, z0: np.ndarray, right_hint: Optional[np.ndarray] = None) -> np.ndarray:
    """Build right-handed SCS from approximate Y and Z."""

    z = _normalize(z0)
    y = _project_to_plane(y0, z)
    y = _normalize(y)
    x = _normalize(np.cross(y, z))  # y × z = x
    y = _normalize(np.cross(z, x))  # z × x = y

    if right_hint is not None:
        x, y = _maybe_flip_xy(x, y, right_hint=right_hint)

    return np.stack([x, y, z], axis=-1)


def _m(points: np.ndarray, labels: list[str], name: str) -> np.ndarray:
    try:
        idx = labels.index(name)
    except ValueError as e:
        raise KeyError(f"Required marker not found: {name}") from e
    return points[:, idx, :]


def euler_intrinsic_xyz_from_matrix(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intrinsic XYZ Euler angles from rotation matrices.

    Parameters
    ----------
    R : (T,3,3)
        Rotation matrix for distal relative to reference (R = ref^T * dist)

    Returns
    -------
    ax, ay, az : (T,)
        Angles in radians (X, Y, Z) using intrinsic XYZ sequence.

    Notes
    -----
    intrinsic XYZ == extrinsic ZYX
    """

    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R must have shape (T,3,3). Got {R.shape}")

    r20 = R[:, 2, 0]
    ay = np.arcsin(np.clip(-r20, -1.0, 1.0))

    cos_ay = np.cos(ay)
    singular = np.abs(cos_ay) < 1e-8

    ax = np.empty(R.shape[0], dtype=float)
    az = np.empty(R.shape[0], dtype=float)

    ns = ~singular
    ax[ns] = np.arctan2(R[ns, 2, 1], R[ns, 2, 2])
    az[ns] = np.arctan2(R[ns, 1, 0], R[ns, 0, 0])

    if np.any(singular):
        s = singular
        ax[s] = np.arctan2(-R[s, 0, 1], R[s, 1, 1])
        az[s] = 0.0

    return ax, ay, az


def _angles_deg_from_frames(ref: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X,Y,Z) joint angles in degrees for intrinsic XYZ sequence."""

    Rrel = np.matmul(np.transpose(ref, (0, 2, 1)), dist)  # ref^T * dist
    ax, ay, az = euler_intrinsic_xyz_from_matrix(Rrel)
    return np.degrees(ax), np.degrees(ay), np.degrees(az)


def build_segment_frames(
    points: np.ndarray,
    labels: list[str],
    end_frame: Optional[int] = None,
) -> SegmentFrames:
    """Build segment frames for pelvis/thigh/shank/foot + thorax/head.

    Segment axes are defined as:
    - X = +Right
    - Y = +Anterior
    - Z = +Up/+Proximal
    """

    T = points.shape[0]
    end = T if end_frame is None else max(1, min(int(end_frame), T))
    pts = points[:end]

    jc = compute_joint_centers(pts, labels)

    # Pelvis (Visual3D-like): X=Right, Y=Anterior, Z=Up
    LASI = _m(pts, labels, "LASI")
    RASI = _m(pts, labels, "RASI")
    LPSI = _m(pts, labels, "LPSI")
    RPSI = _m(pts, labels, "RPSI")
    origin = (LASI + RASI) / 2.0
    midPSIS = (LPSI + RPSI) / 2.0
    x0_pelvis = RASI - LASI
    y0_pelvis = origin - midPSIS
    pelvis = _frame_from_xy(x0_pelvis, y0_pelvis, right_hint=x0_pelvis)

    # Thigh (L/R): Z=hip->knee (proximal), X=+Right (from knee medial/lateral)
    LKNE = _m(pts, labels, "LKNE")
    RKNE = _m(pts, labels, "RKNE")
    LShin = _m(pts, labels, "LShin_3")
    RShin = _m(pts, labels, "RShin_3")

    hip_L = jc["hip_L"][:end]
    hip_R = jc["hip_R"][:end]
    knee_L = jc["knee_L"][:end]
    knee_R = jc["knee_R"][:end]

    z0_thigh_L = hip_L - knee_L
    z0_thigh_R = hip_R - knee_R

    xhint_thigh_L = LShin - LKNE  # medial - lateral (left): points to right
    xhint_thigh_R = RKNE - RShin  # lateral - medial (right): points to right
    thigh_L = _frame_from_xz(xhint_thigh_L, z0_thigh_L, right_hint=xhint_thigh_L)
    thigh_R = _frame_from_xz(xhint_thigh_R, z0_thigh_R, right_hint=xhint_thigh_R)

    # Shank (L/R): Z=knee->ankle (proximal), X=+Right (ankle medial/lateral)
    LANK = _m(pts, labels, "LANK")
    RANK = _m(pts, labels, "RANK")
    LFoot = _m(pts, labels, "LFoot_3")
    RFoot = _m(pts, labels, "RFoot_3")

    ankle_L = jc["ankle_L"][:end]
    ankle_R = jc["ankle_R"][:end]

    z0_shank_L = knee_L - ankle_L
    z0_shank_R = knee_R - ankle_R
    xhint_shank_L = LFoot - LANK  # medial - lateral (left): to right
    xhint_shank_R = RANK - RFoot  # lateral - medial (right): to right
    shank_L = _frame_from_xz(xhint_shank_L, z0_shank_L, right_hint=xhint_shank_L)
    shank_R = _frame_from_xz(xhint_shank_R, z0_shank_R, right_hint=xhint_shank_R)

    # Foot (L/R): X=+Right (ankle axis), Y=heel->toe (anterior), Z=Up
    LTOE = _m(pts, labels, "LTOE")
    RTOE = _m(pts, labels, "RTOE")
    LHEE = _m(pts, labels, "LHEE")
    RHEE = _m(pts, labels, "RHEE")

    xhint_foot_L = LFoot - LANK
    xhint_foot_R = RANK - RFoot
    y0_foot_L = LTOE - LHEE
    y0_foot_R = RTOE - RHEE
    foot_L = _frame_from_xy(xhint_foot_L, y0_foot_L, right_hint=xhint_foot_L)
    foot_R = _frame_from_xy(xhint_foot_R, y0_foot_R, right_hint=xhint_foot_R)

    # Thorax: use C7/T10 + CLAV/STRN to define Y/Z, enforce X with shoulders
    C7 = _m(pts, labels, "C7")
    T10 = _m(pts, labels, "T10")
    CLAV = _m(pts, labels, "CLAV")
    STRN = _m(pts, labels, "STRN")
    LSHO = _m(pts, labels, "LSHO")
    RSHO = _m(pts, labels, "RSHO")

    z0_thorax = C7 - T10
    front = (CLAV + STRN) / 2.0
    back = (C7 + T10) / 2.0
    y0_thorax = front - back
    right_hint_thorax = RSHO - LSHO
    thorax = _frame_from_yz(y0_thorax, z0_thorax, right_hint=right_hint_thorax)

    # Head: use head markers; enforce X with RFHD-LFHD
    LFHD = _m(pts, labels, "LFHD")
    RFHD = _m(pts, labels, "RFHD")
    LBHD = _m(pts, labels, "LBHD")
    RBHD = _m(pts, labels, "RBHD")

    x0_head = RFHD - LFHD
    y0_head = (LFHD + RFHD) / 2.0 - (LBHD + RBHD) / 2.0
    head = _frame_from_xy(x0_head, y0_head, right_hint=x0_head)

    return SegmentFrames(
        pelvis=pelvis,
        thorax=thorax,
        head=head,
        thigh_L=thigh_L,
        thigh_R=thigh_R,
        shank_L=shank_L,
        shank_R=shank_R,
        foot_L=foot_L,
        foot_R=foot_R,
    )


def compute_v3d_joint_angles_3d(
    points: np.ndarray,
    labels: list[str],
    end_frame: Optional[int] = None,
) -> V3DJointAngles3D:
    """Compute 3D joint angles (ankle/knee/hip/trunk/neck) in Visual3D-like XYZ."""

    frames = build_segment_frames(points, labels, end_frame=end_frame)

    # hip: pelvis -> thigh
    hip_L = _angles_deg_from_frames(frames.pelvis, frames.thigh_L)
    hip_R = _angles_deg_from_frames(frames.pelvis, frames.thigh_R)

    # knee: thigh -> shank
    knee_L = _angles_deg_from_frames(frames.thigh_L, frames.shank_L)
    knee_R = _angles_deg_from_frames(frames.thigh_R, frames.shank_R)

    # ankle: shank -> foot
    ankle_L = _angles_deg_from_frames(frames.shank_L, frames.foot_L)
    ankle_R = _angles_deg_from_frames(frames.shank_R, frames.foot_R)

    # trunk: pelvis -> thorax
    trunk = _angles_deg_from_frames(frames.pelvis, frames.thorax)

    # neck: thorax -> head
    neck = _angles_deg_from_frames(frames.thorax, frames.head)

    return V3DJointAngles3D(
        hip_L_X=hip_L[0],
        hip_L_Y=hip_L[1],
        hip_L_Z=hip_L[2],
        hip_R_X=hip_R[0],
        hip_R_Y=hip_R[1],
        hip_R_Z=hip_R[2],
        knee_L_X=knee_L[0],
        knee_L_Y=knee_L[1],
        knee_L_Z=knee_L[2],
        knee_R_X=knee_R[0],
        knee_R_Y=knee_R[1],
        knee_R_Z=knee_R[2],
        ankle_L_X=ankle_L[0],
        ankle_L_Y=ankle_L[1],
        ankle_L_Z=ankle_L[2],
        ankle_R_X=ankle_R[0],
        ankle_R_Y=ankle_R[1],
        ankle_R_Z=ankle_R[2],
        trunk_X=trunk[0],
        trunk_Y=trunk[1],
        trunk_Z=trunk[2],
        neck_X=neck[0],
        neck_Y=neck[1],
        neck_Z=neck[2],
    )

