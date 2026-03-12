"""Inverse dynamics: internal proximal joint moments from kinematics + GRF.

This implementation is strict and transparent: it computes moments for a minimal
segment chain (foot→shank→thigh→pelvis, plus trunk/head) and resolves each joint
moment in the stated reference segment coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replace_v3d.joint_angles.v3d_joint_angles import SegmentFrames

from .angular_velocity import _gradient, segment_angular_velocity_lab
from .anthropometrics import BodySegmentParams, get_body_segment_params


@dataclass(frozen=True)
class TimeVaryingWrench:
    point: np.ndarray  # (T,3)
    force: np.ndarray  # (T,3)
    moment: np.ndarray  # (T,3)

    def negate(self) -> "TimeVaryingWrench":
        return TimeVaryingWrench(point=self.point, force=-self.force, moment=-self.moment)


@dataclass(frozen=True)
class ForceplateWrenchSeries:
    plate_index_1based: int
    fp_origin_lab: np.ndarray  # (3,)
    grf_lab: np.ndarray  # (T,3)
    grm_lab_at_fp_origin: np.ndarray  # (T,3)
    cop_x_m: np.ndarray  # (T,)
    cop_y_m: np.ndarray  # (T,)
    valid_contact_mask: np.ndarray  # (T,)
    corners_lab: np.ndarray | None = None  # (4,3)


@dataclass(frozen=True)
class ForceAssignmentConfig:
    """Config for V3D-style contact-block assignment in multi-plate mode."""

    cop_distance_threshold_m: float = 0.2
    remove_incomplete_assignments: bool = True
    require_segment_projection_on_plate: bool = True
    log_assignment_summary: bool = True


@dataclass(frozen=True)
class AssignedBlock:
    plate_index_1based: int
    start: int
    end: int
    assigned_side: int | None  # 0=L, 1=R, None=invalid/unassigned
    mean_distance_L_m: float
    mean_distance_R_m: float
    invalid_reason: str | None


@dataclass(frozen=True)
class AssignmentResult:
    left_mask_per_plate: list[np.ndarray]
    right_mask_per_plate: list[np.ndarray]
    invalid_mask: np.ndarray
    assigned_blocks: list[AssignedBlock]


@dataclass(frozen=True)
class SegmentState:
    prox_pos: np.ndarray  # (T,3)
    com_pos: np.ndarray  # (T,3)
    dist_pos: np.ndarray  # (T,3)
    frame: np.ndarray  # (T,3,3) local->lab
    mass_kg: float
    com_acc_lab: np.ndarray  # (T,3)
    omega_lab: np.ndarray  # (T,3)
    alpha_lab: np.ndarray  # (T,3)
    inertia_lab: np.ndarray  # (T,3,3)


def _resolve(vec_lab: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Resolve lab vector into the given segment frame (local coords)."""

    return np.einsum("tji,tj->ti", frame, vec_lab)


def _moment_at_point(M_at_a: np.ndarray, a: np.ndarray, b: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Shift moment from point a to point b: M_b = M_a + (a-b) x F."""

    return M_at_a + np.cross(a - b, F)


def _estimate_cop_lab(
    *,
    cop_x_m: np.ndarray,
    cop_y_m: np.ndarray,
    fp_origin_lab: np.ndarray,
) -> np.ndarray:
    z = float(fp_origin_lab[2])
    return np.column_stack([cop_x_m, cop_y_m, np.full_like(cop_x_m, z, dtype=float)])


def _assign_force_to_side_by_cop(
    *,
    cop_lab: np.ndarray,
    ankle_L: np.ndarray,
    ankle_R: np.ndarray,
) -> np.ndarray:
    """Return side assignment: 0=left, 1=right, -1=unassigned (NaN COP)."""

    cop = np.asarray(cop_lab, dtype=float)
    L = np.asarray(ankle_L, dtype=float)
    R = np.asarray(ankle_R, dtype=float)
    valid = np.all(np.isfinite(cop[:, 0:2]), axis=1) & np.all(np.isfinite(L[:, 0:2]), axis=1) & np.all(
        np.isfinite(R[:, 0:2]), axis=1
    )
    out = np.full(cop.shape[0], -1, dtype=int)
    if not np.any(valid):
        return out
    dL = np.sum((cop[valid, 0:2] - L[valid, 0:2]) ** 2, axis=1)
    dR = np.sum((cop[valid, 0:2] - R[valid, 0:2]) ** 2, axis=1)
    out_valid = np.where(dR < dL, 1, 0)
    out[valid] = out_valid.astype(int)
    return out


def _find_contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool).ravel()
    if arr.size == 0:
        return []
    idx = np.flatnonzero(arr)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for raw in idx[1:]:
        cur = int(raw)
        if cur != prev + 1:
            runs.append((start, prev + 1))
            start = cur
        prev = cur
    runs.append((start, prev + 1))
    return runs


def _segment_distance_to_cop_block(
    *,
    segment_com: np.ndarray,
    cop_lab: np.ndarray,
    start: int,
    end: int,
) -> float:
    seg = np.asarray(segment_com[start:end], dtype=float)
    cop = np.asarray(cop_lab[start:end], dtype=float)
    if seg.size == 0 or cop.size == 0:
        return float("inf")
    finite = np.all(np.isfinite(seg), axis=1) & np.all(np.isfinite(cop), axis=1)
    if not np.any(finite):
        return float("inf")
    dist = np.linalg.norm(seg[finite] - cop[finite], axis=1)
    return float(np.nanmean(dist))


def _project_points_to_plate_xy_mask(
    *,
    points_lab: np.ndarray,
    corners_lab: np.ndarray | None,
) -> np.ndarray:
    pts = np.asarray(points_lab, dtype=float)
    T = int(pts.shape[0])
    if corners_lab is None:
        return np.zeros(T, dtype=bool)
    corners = np.asarray(corners_lab, dtype=float)
    if corners.shape != (4, 3):
        return np.zeros(T, dtype=bool)

    x = pts[:, 0]
    y = pts[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    inside = np.zeros(T, dtype=bool)
    if not np.any(valid):
        return inside

    px = corners[:, 0]
    py = corners[:, 1]
    n = int(corners.shape[0])
    xv = x[valid]
    yv = y[valid]
    in_poly = np.zeros(xv.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        xi = float(px[i])
        yi = float(py[i])
        xj = float(px[j])
        yj = float(py[j])
        denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
        intersects = ((yi > yv) != (yj > yv)) & (xv < ((xj - xi) * (yv - yi) / denom + xi))
        in_poly ^= intersects
        j = i

    # Treat points on polygon edges as inside to avoid boundary asymmetry.
    on_edge = np.zeros(xv.shape[0], dtype=bool)
    edge_tol = 1e-6
    j = n - 1
    for i in range(n):
        xi = float(px[i])
        yi = float(py[i])
        xj = float(px[j])
        yj = float(py[j])
        dx = xj - xi
        dy = yj - yi
        cross = (xv - xi) * dy - (yv - yi) * dx
        within_seg = ((xv - xi) * (xv - xj) + (yv - yi) * (yv - yj)) <= edge_tol
        on_edge |= (np.abs(cross) <= edge_tol) & within_seg
        j = i

    inside[valid] = in_poly | on_edge
    return inside


def _format_assigned_block_summary(block: AssignedBlock) -> str:
    side_text = {0: "L", 1: "R"}.get(block.assigned_side, "invalid")
    details = (
        f"FP{block.plate_index_1based} block {block.start}-{block.end} "
        f"-> {side_text}, mean_dist_L={block.mean_distance_L_m:.3f}m, "
        f"mean_dist_R={block.mean_distance_R_m:.3f}m"
    )
    if block.invalid_reason is not None:
        details = f"{details} ({block.invalid_reason})"
    return details


def _assign_contact_blocks_v3d_style(
    *,
    forceplates: list[ForceplateWrenchSeries],
    foot_L_com: np.ndarray,
    foot_R_com: np.ndarray,
    foot_L_prox: np.ndarray,
    foot_L_dist: np.ndarray,
    foot_R_prox: np.ndarray,
    foot_R_dist: np.ndarray,
    threshold_m: float,
    remove_incomplete_assignments: bool,
    require_segment_projection_on_plate: bool,
) -> AssignmentResult:
    T = int(foot_L_com.shape[0])
    hard_invalid_mask = np.zeros(T, dtype=bool)
    contact_any_mask = np.zeros(T, dtype=bool)
    left_mask_per_plate: list[np.ndarray] = []
    right_mask_per_plate: list[np.ndarray] = []
    assigned_blocks: list[AssignedBlock] = []

    for fp_series in forceplates:
        raw_contact_mask = np.asarray(fp_series.valid_contact_mask[:T], dtype=bool).copy()
        raw_contact_runs = _find_contiguous_true_runs(raw_contact_mask)

        grf_lab = np.asarray(fp_series.grf_lab[:T], dtype=float)
        grm_lab = np.asarray(fp_series.grm_lab_at_fp_origin[:T], dtype=float)
        contact_mask = raw_contact_mask.copy()
        contact_mask &= np.all(np.isfinite(grf_lab), axis=1)
        contact_mask &= np.all(np.isfinite(grm_lab), axis=1)

        cop_lab = _estimate_cop_lab(
            cop_x_m=np.asarray(fp_series.cop_x_m[:T], dtype=float),
            cop_y_m=np.asarray(fp_series.cop_y_m[:T], dtype=float),
            fp_origin_lab=np.asarray(fp_series.fp_origin_lab, dtype=float),
        )
        contact_mask &= np.all(np.isfinite(cop_lab[:, 0:2]), axis=1)
        contact_any_mask |= contact_mask

        left_mask = np.zeros(T, dtype=bool)
        right_mask = np.zeros(T, dtype=bool)

        has_corners = fp_series.corners_lab is not None and np.asarray(fp_series.corners_lab).shape == (4, 3)
        proj_L_prox = _project_points_to_plate_xy_mask(points_lab=foot_L_prox, corners_lab=fp_series.corners_lab)
        proj_L_dist = _project_points_to_plate_xy_mask(points_lab=foot_L_dist, corners_lab=fp_series.corners_lab)
        proj_R_prox = _project_points_to_plate_xy_mask(points_lab=foot_R_prox, corners_lab=fp_series.corners_lab)
        proj_R_dist = _project_points_to_plate_xy_mask(points_lab=foot_R_dist, corners_lab=fp_series.corners_lab)
        proj_L_overlap = proj_L_prox & proj_L_dist
        proj_R_overlap = proj_R_prox & proj_R_dist

        for start, end in _find_contiguous_true_runs(contact_mask):
            block_slice = slice(start, end)
            dL = _segment_distance_to_cop_block(segment_com=foot_L_com, cop_lab=cop_lab, start=start, end=end)
            dR = _segment_distance_to_cop_block(segment_com=foot_R_com, cop_lab=cop_lab, start=start, end=end)

            assigned_side: int | None = None
            invalid_reason: str | None = None

            if remove_incomplete_assignments:
                touches_edge_raw_run = False
                for raw_start, raw_end in raw_contact_runs:
                    if not (end <= raw_start or start >= raw_end):
                        if raw_start == 0 or raw_end == T:
                            touches_edge_raw_run = True
                            break
                if touches_edge_raw_run:
                    invalid_reason = "incomplete_contact_block"

            if invalid_reason is None and has_corners:
                left_proj_ratio = float(np.nanmean(proj_L_overlap[block_slice].astype(float)))
                right_proj_ratio = float(np.nanmean(proj_R_overlap[block_slice].astype(float)))
                both_threshold = dL <= float(threshold_m) and dR <= float(threshold_m)
                very_close = abs(dL - dR) <= 0.02
                two_feet_projection = left_proj_ratio >= 0.35 and right_proj_ratio >= 0.35
                if both_threshold and (two_feet_projection or very_close):
                    invalid_reason = "two_feet_same_plate"

            if invalid_reason is None:
                left_ok = dL <= float(threshold_m)
                right_ok = dR <= float(threshold_m)
                if not left_ok and not right_ok:
                    invalid_reason = "distance_threshold_exceeded"
                elif left_ok and not right_ok:
                    assigned_side = 0
                elif right_ok and not left_ok:
                    assigned_side = 1
                else:
                    assigned_side = 0 if dL <= dR else 1

            if (
                invalid_reason is None
                and assigned_side is not None
                and require_segment_projection_on_plate
                and has_corners
            ):
                if assigned_side == 0:
                    proj_pass = bool(np.any(proj_L_prox[block_slice] & proj_L_dist[block_slice]))
                else:
                    proj_pass = bool(np.any(proj_R_prox[block_slice] & proj_R_dist[block_slice]))
                if not proj_pass:
                    assigned_side = None
                    invalid_reason = "segment_projection_failed"

            if assigned_side == 0:
                left_mask[block_slice] = True
            elif assigned_side == 1:
                right_mask[block_slice] = True
            else:
                if invalid_reason in {"incomplete_contact_block", "two_feet_same_plate", "segment_projection_failed"}:
                    hard_invalid_mask[block_slice] = True

            assigned_blocks.append(
                AssignedBlock(
                    plate_index_1based=int(fp_series.plate_index_1based),
                    start=int(start),
                    end=int(end),
                    assigned_side=assigned_side,
                    mean_distance_L_m=float(dL),
                    mean_distance_R_m=float(dR),
                    invalid_reason=invalid_reason,
                )
            )

        left_mask_per_plate.append(left_mask)
        right_mask_per_plate.append(right_mask)

    assigned_any_mask = np.zeros(T, dtype=bool)
    for left_mask in left_mask_per_plate:
        assigned_any_mask |= left_mask
    for right_mask in right_mask_per_plate:
        assigned_any_mask |= right_mask
    invalid_mask = hard_invalid_mask | (contact_any_mask & ~assigned_any_mask)

    return AssignmentResult(
        left_mask_per_plate=left_mask_per_plate,
        right_mask_per_plate=right_mask_per_plate,
        invalid_mask=invalid_mask,
        assigned_blocks=assigned_blocks,
    )


def _make_gravity_wrench(com_pos: np.ndarray, mass_kg: float, g: float = 9.81) -> TimeVaryingWrench:
    T = com_pos.shape[0]
    force = np.zeros((T, 3), dtype=float)
    force[:, 2] = -float(mass_kg) * float(g)
    moment = np.zeros((T, 3), dtype=float)
    return TimeVaryingWrench(point=com_pos, force=force, moment=moment)


def _segment_inertia_lab(
    *,
    frame: np.ndarray,
    mass_kg: float,
    length_m: np.ndarray,
    rog_fraction_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Diagonal inertia using per-segment radius-of-gyration fractions."""

    kx, ky, kz = (float(rog_fraction_xyz[0]), float(rog_fraction_xyz[1]), float(rog_fraction_xyz[2]))
    L2 = np.clip(np.asarray(length_m, dtype=float) ** 2, 0.0, None)
    # TODO: future geometry-based inertia from proximal/distal radii and segment geometry.
    Ixx = float(mass_kg) * (kx * kx) * L2
    Iyy = float(mass_kg) * (ky * ky) * L2
    Izz = float(mass_kg) * (kz * kz) * L2
    I_local = np.zeros((frame.shape[0], 3, 3), dtype=float)
    I_local[:, 0, 0] = Ixx
    I_local[:, 1, 1] = Iyy
    I_local[:, 2, 2] = Izz
    R = np.asarray(frame, dtype=float)
    return np.einsum("tij,tjk,tlk->til", R, I_local, R)


def _make_segment_state(
    *,
    prox_pos: np.ndarray,
    dist_pos: np.ndarray,
    com_pos: np.ndarray,
    frame: np.ndarray,
    mass_kg: float,
    rog_fraction_xyz: tuple[float, float, float],
    rate_hz: float,
) -> SegmentState:
    dt = 1.0 / float(rate_hz)
    com_vel = _gradient(np.asarray(com_pos, dtype=float), dt)
    com_acc = _gradient(com_vel, dt)

    omega_lab = segment_angular_velocity_lab(frame, rate_hz=rate_hz)
    alpha_lab = _gradient(omega_lab, dt)
    seg_len = np.linalg.norm(np.asarray(dist_pos, dtype=float) - np.asarray(prox_pos, dtype=float), axis=1)
    inertia_lab = _segment_inertia_lab(
        frame=frame,
        mass_kg=float(mass_kg),
        length_m=seg_len,
        rog_fraction_xyz=rog_fraction_xyz,
    )

    return SegmentState(
        prox_pos=np.asarray(prox_pos, dtype=float),
        com_pos=np.asarray(com_pos, dtype=float),
        dist_pos=np.asarray(dist_pos, dtype=float),
        frame=np.asarray(frame, dtype=float),
        mass_kg=float(mass_kg),
        com_acc_lab=com_acc,
        omega_lab=omega_lab,
        alpha_lab=alpha_lab,
        inertia_lab=inertia_lab,
    )


def _proximal_wrench(
    *,
    state: SegmentState,
    distal_wrenches: list[TimeVaryingWrench],
    external_wrenches: list[TimeVaryingWrench],
) -> TimeVaryingWrench:
    T = state.com_pos.shape[0]
    F_known = np.zeros((T, 3), dtype=float)
    M_known_about_com = np.zeros((T, 3), dtype=float)

    def add_wrench(w: TimeVaryingWrench) -> None:
        nonlocal F_known, M_known_about_com
        F_known = F_known + w.force
        M_known_about_com = M_known_about_com + (w.moment + np.cross(w.point - state.com_pos, w.force))

    for w in distal_wrenches:
        add_wrench(w)
    for w in external_wrenches:
        add_wrench(w)

    F_prox = float(state.mass_kg) * state.com_acc_lab - F_known

    Iw = np.einsum("tij,tj->ti", state.inertia_lab, state.omega_lab)
    dyn = np.einsum("tij,tj->ti", state.inertia_lab, state.alpha_lab) + np.cross(state.omega_lab, Iw)

    M_prox = dyn - np.cross(state.prox_pos - state.com_pos, F_prox) - M_known_about_com
    return TimeVaryingWrench(point=state.prox_pos, force=F_prox, moment=M_prox)


def _nan_moment_cols(joint_prefix: str, T: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for axis in ("X", "Y", "Z"):
        out[f"{joint_prefix}_{axis}_Nm"] = np.full(T, np.nan, dtype=float)
    return out


def _nan_all_joint_moment_cols(T: int) -> dict[str, np.ndarray]:
    return {
        **_nan_moment_cols("Hip_L_ref", T),
        **_nan_moment_cols("Hip_R_ref", T),
        **_nan_moment_cols("Knee_L_ref", T),
        **_nan_moment_cols("Knee_R_ref", T),
        **_nan_moment_cols("Ankle_L_ref", T),
        **_nan_moment_cols("Ankle_R_ref", T),
        **_nan_moment_cols("Trunk_ref", T),
        **_nan_moment_cols("Neck_ref", T),
    }


def compute_joint_moment_columns(
    *,
    points: np.ndarray,
    labels: list[str],
    frames: SegmentFrames,
    joint_centers: dict[str, np.ndarray],
    rate_hz: float,
    body_mass_kg: float | None,
    fp_origin_lab: np.ndarray,
    grf_lab: np.ndarray,
    grm_lab_at_fp_origin: np.ndarray,
    cop_x_m: np.ndarray,
    cop_y_m: np.ndarray,
    segment_params: BodySegmentParams | None = None,
) -> dict[str, np.ndarray]:
    """Compute internal joint moment columns resolved in reference segment frames.

    Output naming contract (Nm):
    - Hip/Knee/Ankle: <Joint>_<Side>_ref_[X|Y|Z]_Nm
    - Trunk/Neck: <Joint>_ref_[X|Y|Z]_Nm
    """

    T = int(frames.pelvis.shape[0])
    nan_all = _nan_all_joint_moment_cols(T)
    if body_mass_kg is None or not np.isfinite(float(body_mass_kg)) or float(body_mass_kg) <= 0:
        return nan_all

    try:
        return _compute_joint_moment_columns_impl(
            points=points,
            labels=labels,
            frames=frames,
            joint_centers=joint_centers,
            rate_hz=rate_hz,
            body_mass_kg=float(body_mass_kg),
            fp_origin_lab=fp_origin_lab,
            grf_lab=grf_lab,
            grm_lab_at_fp_origin=grm_lab_at_fp_origin,
            cop_x_m=cop_x_m,
            cop_y_m=cop_y_m,
            segment_params=get_body_segment_params() if segment_params is None else segment_params,
        )
    except Exception:
        return nan_all


def compute_joint_moment_columns_multi(
    *,
    points: np.ndarray,
    labels: list[str],
    frames: SegmentFrames,
    joint_centers: dict[str, np.ndarray],
    rate_hz: float,
    body_mass_kg: float | None,
    forceplates: list[ForceplateWrenchSeries],
    segment_params: BodySegmentParams | None = None,
    assignment_config: ForceAssignmentConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute lower-limb joint moments from a selected multi-plate forceplate set."""

    T = int(frames.pelvis.shape[0])
    nan_all = _nan_all_joint_moment_cols(T)
    if body_mass_kg is None or not np.isfinite(float(body_mass_kg)) or float(body_mass_kg) <= 0:
        return nan_all
    if not forceplates:
        return nan_all

    try:
        return _compute_joint_moment_columns_multi_impl(
            points=points,
            labels=labels,
            frames=frames,
            joint_centers=joint_centers,
            rate_hz=rate_hz,
            body_mass_kg=float(body_mass_kg),
            forceplates=forceplates,
            segment_params=get_body_segment_params() if segment_params is None else segment_params,
            assignment_config=ForceAssignmentConfig() if assignment_config is None else assignment_config,
        )
    except Exception:
        return nan_all


def _plate_xy_bounds(corners_lab: np.ndarray | None) -> tuple[float, float, float, float] | None:
    if corners_lab is None:
        return None
    arr = np.asarray(corners_lab, dtype=float)
    if arr.shape != (4, 3):
        return None
    return (
        float(np.nanmin(arr[:, 0])),
        float(np.nanmax(arr[:, 0])),
        float(np.nanmin(arr[:, 1])),
        float(np.nanmax(arr[:, 1])),
    )


def _ankles_share_plate_xy(
    *,
    bounds_xy: tuple[float, float, float, float] | None,
    ankle_L: np.ndarray,
    ankle_R: np.ndarray,
) -> np.ndarray:
    if bounds_xy is None:
        return np.zeros(ankle_L.shape[0], dtype=bool)
    min_x, max_x, min_y, max_y = bounds_xy
    L_inside = (
        np.isfinite(ankle_L[:, 0])
        & np.isfinite(ankle_L[:, 1])
        & (ankle_L[:, 0] >= min_x)
        & (ankle_L[:, 0] <= max_x)
        & (ankle_L[:, 1] >= min_y)
        & (ankle_L[:, 1] <= max_y)
    )
    R_inside = (
        np.isfinite(ankle_R[:, 0])
        & np.isfinite(ankle_R[:, 1])
        & (ankle_R[:, 0] >= min_x)
        & (ankle_R[:, 0] <= max_x)
        & (ankle_R[:, 1] >= min_y)
        & (ankle_R[:, 1] <= max_y)
    )
    return L_inside & R_inside


def _compute_joint_moment_columns_multi_impl(
    *,
    points: np.ndarray,
    labels: list[str],
    frames: SegmentFrames,
    joint_centers: dict[str, np.ndarray],
    rate_hz: float,
    body_mass_kg: float,
    forceplates: list[ForceplateWrenchSeries],
    segment_params: BodySegmentParams,
    assignment_config: ForceAssignmentConfig,
) -> dict[str, np.ndarray]:
    T = int(frames.pelvis.shape[0])

    seg = segment_params
    mass = float(body_mass_kg)

    jc = joint_centers
    hip_L = jc["hip_L"][:T]
    hip_R = jc["hip_R"][:T]
    knee_L = jc["knee_L"][:T]
    knee_R = jc["knee_R"][:T]
    ankle_L = jc["ankle_L"][:T]
    ankle_R = jc["ankle_R"][:T]
    pelvis_origin = jc["pelvis_origin"][:T]
    C7 = jc["C7"][:T]
    head_center = jc["head_center"][:T]

    def m(name: str) -> np.ndarray:
        idx = labels.index(name)
        return points[:T, idx, :]

    LTOE, RTOE = m("LTOE"), m("RTOE")
    LHEE, RHEE = m("LHEE"), m("RHEE")

    foot_L_prox = ankle_L
    foot_L_dist = LTOE
    foot_L_com = LHEE + seg.lower.foot.com_fraction_from_prox * (LTOE - LHEE)

    foot_R_prox = ankle_R
    foot_R_dist = RTOE
    foot_R_com = RHEE + seg.lower.foot.com_fraction_from_prox * (RTOE - RHEE)

    shank_L_prox = knee_L
    shank_L_dist = ankle_L
    shank_L_com = knee_L + seg.lower.shank.com_fraction_from_prox * (ankle_L - knee_L)

    shank_R_prox = knee_R
    shank_R_dist = ankle_R
    shank_R_com = knee_R + seg.lower.shank.com_fraction_from_prox * (ankle_R - knee_R)

    thigh_L_prox = hip_L
    thigh_L_dist = knee_L
    thigh_L_com = hip_L + seg.lower.thigh.com_fraction_from_prox * (knee_L - hip_L)

    thigh_R_prox = hip_R
    thigh_R_dist = knee_R
    thigh_R_com = hip_R + seg.lower.thigh.com_fraction_from_prox * (knee_R - hip_R)

    try:
        STRN = m("STRN")
    except Exception:
        STRN = None
    thorax_ref = 0.56 * C7 + 0.44 * STRN if STRN is not None else C7

    trunk_prox = pelvis_origin
    trunk_dist = C7
    trunk_com = pelvis_origin + seg.upper.trunk.com_fraction_from_prox * (thorax_ref - pelvis_origin)

    head_prox = C7
    head_dist = head_center
    head_com = C7 + seg.upper.head.com_fraction_from_prox * (head_center - C7)

    m_foot = mass * float(seg.lower.foot.mass_fraction)
    m_shank = mass * float(seg.lower.shank.mass_fraction)
    m_thigh = mass * float(seg.lower.thigh.mass_fraction)
    m_trunk = mass * float(seg.upper.trunk.mass_fraction)
    m_head = mass * float(seg.upper.head.mass_fraction)

    state_foot_L = _make_segment_state(
        prox_pos=foot_L_prox,
        dist_pos=foot_L_dist,
        com_pos=foot_L_com,
        frame=frames.foot_L,
        mass_kg=m_foot,
        rog_fraction_xyz=seg.lower.foot.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_shank_L = _make_segment_state(
        prox_pos=shank_L_prox,
        dist_pos=shank_L_dist,
        com_pos=shank_L_com,
        frame=frames.shank_L,
        mass_kg=m_shank,
        rog_fraction_xyz=seg.lower.shank.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_thigh_L = _make_segment_state(
        prox_pos=thigh_L_prox,
        dist_pos=thigh_L_dist,
        com_pos=thigh_L_com,
        frame=frames.thigh_L,
        mass_kg=m_thigh,
        rog_fraction_xyz=seg.lower.thigh.rog_fraction_xyz,
        rate_hz=rate_hz,
    )

    state_foot_R = _make_segment_state(
        prox_pos=foot_R_prox,
        dist_pos=foot_R_dist,
        com_pos=foot_R_com,
        frame=frames.foot_R,
        mass_kg=m_foot,
        rog_fraction_xyz=seg.lower.foot.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_shank_R = _make_segment_state(
        prox_pos=shank_R_prox,
        dist_pos=shank_R_dist,
        com_pos=shank_R_com,
        frame=frames.shank_R,
        mass_kg=m_shank,
        rog_fraction_xyz=seg.lower.shank.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_thigh_R = _make_segment_state(
        prox_pos=thigh_R_prox,
        dist_pos=thigh_R_dist,
        com_pos=thigh_R_com,
        frame=frames.thigh_R,
        mass_kg=m_thigh,
        rog_fraction_xyz=seg.lower.thigh.rog_fraction_xyz,
        rate_hz=rate_hz,
    )

    state_trunk = _make_segment_state(
        prox_pos=trunk_prox,
        dist_pos=trunk_dist,
        com_pos=trunk_com,
        frame=frames.thorax,
        mass_kg=m_trunk,
        rog_fraction_xyz=seg.upper.trunk.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_head = _make_segment_state(
        prox_pos=head_prox,
        dist_pos=head_dist,
        com_pos=head_com,
        frame=frames.head,
        mass_kg=m_head,
        rog_fraction_xyz=seg.upper.head.rog_fraction_xyz,
        rate_hz=rate_hz,
    )

    assignment_result = _assign_contact_blocks_v3d_style(
        forceplates=forceplates,
        foot_L_com=foot_L_com,
        foot_R_com=foot_R_com,
        foot_L_prox=foot_L_prox,
        foot_L_dist=foot_L_dist,
        foot_R_prox=foot_R_prox,
        foot_R_dist=foot_R_dist,
        threshold_m=float(assignment_config.cop_distance_threshold_m),
        remove_incomplete_assignments=bool(assignment_config.remove_incomplete_assignments),
        require_segment_projection_on_plate=bool(assignment_config.require_segment_projection_on_plate),
    )

    if bool(assignment_config.log_assignment_summary):
        print("[INFO] force assignment summary:")
        if assignment_result.assigned_blocks:
            for block in assignment_result.assigned_blocks:
                print(f"    {_format_assigned_block_summary(block)}")
        else:
            print("    no contact blocks detected")

    left_has_contact = np.zeros(T, dtype=bool)
    right_has_contact = np.zeros(T, dtype=bool)
    plate_rows: list[dict[str, np.ndarray]] = []
    for fp_series, left_mask, right_mask in zip(
        forceplates,
        assignment_result.left_mask_per_plate,
        assignment_result.right_mask_per_plate,
    ):
        grf_lab = np.asarray(fp_series.grf_lab[:T], dtype=float)
        grm_lab = np.asarray(fp_series.grm_lab_at_fp_origin[:T], dtype=float)
        cop_lab = _estimate_cop_lab(
            cop_x_m=np.asarray(fp_series.cop_x_m[:T], dtype=float),
            cop_y_m=np.asarray(fp_series.cop_y_m[:T], dtype=float),
            fp_origin_lab=np.asarray(fp_series.fp_origin_lab, dtype=float),
        )
        M_at_cop = _moment_at_point(
            grm_lab,
            np.asarray(fp_series.fp_origin_lab, dtype=float)[None, :],
            cop_lab,
            grf_lab,
        )

        left_has_contact |= np.asarray(left_mask, dtype=bool)
        right_has_contact |= np.asarray(right_mask, dtype=bool)
        plate_rows.append(
            {
                "grf_lab": grf_lab,
                "moment_at_cop": M_at_cop,
                "cop_lab": cop_lab,
                "left_mask": np.asarray(left_mask, dtype=bool),
                "right_mask": np.asarray(right_mask, dtype=bool),
            }
        )

    left_valid = left_has_contact & ~assignment_result.invalid_mask
    right_valid = right_has_contact & ~assignment_result.invalid_mask

    def masked_wrench(
        *,
        grf_lab: np.ndarray,
        moment_at_cop: np.ndarray,
        cop_lab: np.ndarray,
        mask: np.ndarray,
    ) -> TimeVaryingWrench:
        force = np.zeros_like(grf_lab)
        moment = np.zeros_like(moment_at_cop)
        point = np.asarray(cop_lab, dtype=float).copy()
        force[mask] = grf_lab[mask]
        moment[mask] = moment_at_cop[mask]
        force[assignment_result.invalid_mask] = np.nan
        moment[assignment_result.invalid_mask] = np.nan
        point[assignment_result.invalid_mask] = np.nan
        return TimeVaryingWrench(point=point, force=force, moment=moment)

    left_wrenches = [
        masked_wrench(
            grf_lab=row["grf_lab"],
            moment_at_cop=row["moment_at_cop"],
            cop_lab=row["cop_lab"],
            mask=row["left_mask"],
        )
        for row in plate_rows
    ]
    right_wrenches = [
        masked_wrench(
            grf_lab=row["grf_lab"],
            moment_at_cop=row["moment_at_cop"],
            cop_lab=row["cop_lab"],
            mask=row["right_mask"],
        )
        for row in plate_rows
    ]

    foot_L_prox_w = _proximal_wrench(
        state=state_foot_L,
        distal_wrenches=[],
        external_wrenches=[*left_wrenches, _make_gravity_wrench(state_foot_L.com_pos, state_foot_L.mass_kg)],
    )
    shank_L_prox_w = _proximal_wrench(
        state=state_shank_L,
        distal_wrenches=[foot_L_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_shank_L.com_pos, state_shank_L.mass_kg)],
    )
    thigh_L_prox_w = _proximal_wrench(
        state=state_thigh_L,
        distal_wrenches=[shank_L_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_thigh_L.com_pos, state_thigh_L.mass_kg)],
    )

    foot_R_prox_w = _proximal_wrench(
        state=state_foot_R,
        distal_wrenches=[],
        external_wrenches=[*right_wrenches, _make_gravity_wrench(state_foot_R.com_pos, state_foot_R.mass_kg)],
    )
    shank_R_prox_w = _proximal_wrench(
        state=state_shank_R,
        distal_wrenches=[foot_R_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_shank_R.com_pos, state_shank_R.mass_kg)],
    )
    thigh_R_prox_w = _proximal_wrench(
        state=state_thigh_R,
        distal_wrenches=[shank_R_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_thigh_R.com_pos, state_thigh_R.mass_kg)],
    )

    head_prox_w = _proximal_wrench(
        state=state_head,
        distal_wrenches=[],
        external_wrenches=[_make_gravity_wrench(state_head.com_pos, state_head.mass_kg)],
    )
    trunk_prox_w = _proximal_wrench(
        state=state_trunk,
        distal_wrenches=[head_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_trunk.com_pos, state_trunk.mass_kg)],
    )

    ankle_L_ref = _resolve(foot_L_prox_w.moment, frames.shank_L)
    ankle_R_ref = _resolve(foot_R_prox_w.moment, frames.shank_R)
    knee_L_ref = _resolve(shank_L_prox_w.moment, frames.thigh_L)
    knee_R_ref = _resolve(shank_R_prox_w.moment, frames.thigh_R)
    hip_L_ref = _resolve(thigh_L_prox_w.moment, frames.pelvis)
    hip_R_ref = _resolve(thigh_R_prox_w.moment, frames.pelvis)
    trunk_ref = _resolve(trunk_prox_w.moment, frames.pelvis)
    neck_ref = _resolve(head_prox_w.moment, frames.thorax)

    for arr, mask in (
        (ankle_L_ref, left_valid),
        (knee_L_ref, left_valid),
        (hip_L_ref, left_valid),
        (ankle_R_ref, right_valid),
        (knee_R_ref, right_valid),
        (hip_R_ref, right_valid),
    ):
        arr[~mask] = np.nan

    out: dict[str, np.ndarray] = {}
    for axis, j in zip(("X", "Y", "Z"), range(3)):
        out[f"Ankle_L_ref_{axis}_Nm"] = ankle_L_ref[:, j]
        out[f"Ankle_R_ref_{axis}_Nm"] = ankle_R_ref[:, j]
        out[f"Knee_L_ref_{axis}_Nm"] = knee_L_ref[:, j]
        out[f"Knee_R_ref_{axis}_Nm"] = knee_R_ref[:, j]
        out[f"Hip_L_ref_{axis}_Nm"] = hip_L_ref[:, j]
        out[f"Hip_R_ref_{axis}_Nm"] = hip_R_ref[:, j]
        out[f"Trunk_ref_{axis}_Nm"] = trunk_ref[:, j]
        out[f"Neck_ref_{axis}_Nm"] = neck_ref[:, j]

    return out


def _compute_joint_moment_columns_impl(
    *,
    points: np.ndarray,
    labels: list[str],
    frames: SegmentFrames,
    joint_centers: dict[str, np.ndarray],
    rate_hz: float,
    body_mass_kg: float,
    fp_origin_lab: np.ndarray,
    grf_lab: np.ndarray,
    grm_lab_at_fp_origin: np.ndarray,
    cop_x_m: np.ndarray,
    cop_y_m: np.ndarray,
    segment_params: BodySegmentParams,
) -> dict[str, np.ndarray]:
    T = int(frames.pelvis.shape[0])

    seg = segment_params
    mass = float(body_mass_kg)

    jc = joint_centers
    hip_L = jc["hip_L"][:T]
    hip_R = jc["hip_R"][:T]
    knee_L = jc["knee_L"][:T]
    knee_R = jc["knee_R"][:T]
    ankle_L = jc["ankle_L"][:T]
    ankle_R = jc["ankle_R"][:T]
    pelvis_origin = jc["pelvis_origin"][:T]
    C7 = jc["C7"][:T]
    head_center = jc["head_center"][:T]

    def m(name: str) -> np.ndarray:
        idx = labels.index(name)
        return points[:T, idx, :]

    # Foot endpoints and COMs (use heel->toe line as in whole_body COM model)
    LTOE, RTOE = m("LTOE"), m("RTOE")
    LHEE, RHEE = m("LHEE"), m("RHEE")

    foot_L_prox = ankle_L
    foot_L_dist = LTOE
    foot_L_com = LHEE + seg.lower.foot.com_fraction_from_prox * (LTOE - LHEE)

    foot_R_prox = ankle_R
    foot_R_dist = RTOE
    foot_R_com = RHEE + seg.lower.foot.com_fraction_from_prox * (RTOE - RHEE)

    shank_L_prox = knee_L
    shank_L_dist = ankle_L
    shank_L_com = knee_L + seg.lower.shank.com_fraction_from_prox * (ankle_L - knee_L)

    shank_R_prox = knee_R
    shank_R_dist = ankle_R
    shank_R_com = knee_R + seg.lower.shank.com_fraction_from_prox * (ankle_R - knee_R)

    thigh_L_prox = hip_L
    thigh_L_dist = knee_L
    thigh_L_com = hip_L + seg.lower.thigh.com_fraction_from_prox * (knee_L - hip_L)

    thigh_R_prox = hip_R
    thigh_R_dist = knee_R
    thigh_R_com = hip_R + seg.lower.thigh.com_fraction_from_prox * (knee_R - hip_R)

    # Trunk/head COMs consistent with repo COM model
    try:
        STRN = m("STRN")
    except Exception:
        STRN = None
    thorax_ref = 0.56 * C7 + 0.44 * STRN if STRN is not None else C7

    trunk_prox = pelvis_origin
    trunk_dist = C7
    trunk_com = pelvis_origin + seg.upper.trunk.com_fraction_from_prox * (thorax_ref - pelvis_origin)

    head_prox = C7
    head_dist = head_center
    head_com = C7 + seg.upper.head.com_fraction_from_prox * (head_center - C7)

    m_foot = mass * float(seg.lower.foot.mass_fraction)
    m_shank = mass * float(seg.lower.shank.mass_fraction)
    m_thigh = mass * float(seg.lower.thigh.mass_fraction)
    m_trunk = mass * float(seg.upper.trunk.mass_fraction)
    m_head = mass * float(seg.upper.head.mass_fraction)

    state_foot_L = _make_segment_state(
        prox_pos=foot_L_prox,
        dist_pos=foot_L_dist,
        com_pos=foot_L_com,
        frame=frames.foot_L,
        mass_kg=m_foot,
        rog_fraction_xyz=seg.lower.foot.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_shank_L = _make_segment_state(
        prox_pos=shank_L_prox,
        dist_pos=shank_L_dist,
        com_pos=shank_L_com,
        frame=frames.shank_L,
        mass_kg=m_shank,
        rog_fraction_xyz=seg.lower.shank.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_thigh_L = _make_segment_state(
        prox_pos=thigh_L_prox,
        dist_pos=thigh_L_dist,
        com_pos=thigh_L_com,
        frame=frames.thigh_L,
        mass_kg=m_thigh,
        rog_fraction_xyz=seg.lower.thigh.rog_fraction_xyz,
        rate_hz=rate_hz,
    )

    state_foot_R = _make_segment_state(
        prox_pos=foot_R_prox,
        dist_pos=foot_R_dist,
        com_pos=foot_R_com,
        frame=frames.foot_R,
        mass_kg=m_foot,
        rog_fraction_xyz=seg.lower.foot.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_shank_R = _make_segment_state(
        prox_pos=shank_R_prox,
        dist_pos=shank_R_dist,
        com_pos=shank_R_com,
        frame=frames.shank_R,
        mass_kg=m_shank,
        rog_fraction_xyz=seg.lower.shank.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_thigh_R = _make_segment_state(
        prox_pos=thigh_R_prox,
        dist_pos=thigh_R_dist,
        com_pos=thigh_R_com,
        frame=frames.thigh_R,
        mass_kg=m_thigh,
        rog_fraction_xyz=seg.lower.thigh.rog_fraction_xyz,
        rate_hz=rate_hz,
    )

    state_trunk = _make_segment_state(
        prox_pos=trunk_prox,
        dist_pos=trunk_dist,
        com_pos=trunk_com,
        frame=frames.thorax,
        mass_kg=m_trunk,
        rog_fraction_xyz=seg.upper.trunk.rog_fraction_xyz,
        rate_hz=rate_hz,
    )
    state_head = _make_segment_state(
        prox_pos=head_prox,
        dist_pos=head_dist,
        com_pos=head_com,
        frame=frames.head,
        mass_kg=m_head,
        rog_fraction_xyz=seg.upper.head.rog_fraction_xyz,
        rate_hz=rate_hz,
    )

    cop_lab = _estimate_cop_lab(cop_x_m=cop_x_m[:T], cop_y_m=cop_y_m[:T], fp_origin_lab=np.asarray(fp_origin_lab))
    M_at_cop = _moment_at_point(grm_lab_at_fp_origin[:T], np.asarray(fp_origin_lab)[None, :], cop_lab, grf_lab[:T])
    grf_wrench = TimeVaryingWrench(point=cop_lab, force=grf_lab[:T], moment=M_at_cop)

    assign = _assign_force_to_side_by_cop(cop_lab=cop_lab, ankle_L=ankle_L, ankle_R=ankle_R)
    mask_L = assign == 0
    mask_R = assign == 1
    mask_unassigned = assign == -1

    def masked_wrench(mask: np.ndarray) -> TimeVaryingWrench:
        force = np.zeros_like(grf_wrench.force)
        moment = np.zeros_like(grf_wrench.moment)
        point = np.asarray(grf_wrench.point, dtype=float).copy()

        if np.any(mask_unassigned):
            force[mask_unassigned] = np.nan
            moment[mask_unassigned] = np.nan
            point[mask_unassigned] = np.nan
        if np.any(mask):
            force[mask] = grf_wrench.force[mask]
            moment[mask] = grf_wrench.moment[mask]
        return TimeVaryingWrench(point=point, force=force, moment=moment)

    wrench_L = masked_wrench(mask_L)
    wrench_R = masked_wrench(mask_R)

    # Lower limbs (distal -> proximal)
    foot_L_prox_w = _proximal_wrench(
        state=state_foot_L,
        distal_wrenches=[],
        external_wrenches=[wrench_L, _make_gravity_wrench(state_foot_L.com_pos, state_foot_L.mass_kg)],
    )
    shank_L_prox_w = _proximal_wrench(
        state=state_shank_L,
        distal_wrenches=[foot_L_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_shank_L.com_pos, state_shank_L.mass_kg)],
    )
    thigh_L_prox_w = _proximal_wrench(
        state=state_thigh_L,
        distal_wrenches=[shank_L_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_thigh_L.com_pos, state_thigh_L.mass_kg)],
    )

    foot_R_prox_w = _proximal_wrench(
        state=state_foot_R,
        distal_wrenches=[],
        external_wrenches=[wrench_R, _make_gravity_wrench(state_foot_R.com_pos, state_foot_R.mass_kg)],
    )
    shank_R_prox_w = _proximal_wrench(
        state=state_shank_R,
        distal_wrenches=[foot_R_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_shank_R.com_pos, state_shank_R.mass_kg)],
    )
    thigh_R_prox_w = _proximal_wrench(
        state=state_thigh_R,
        distal_wrenches=[shank_R_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_thigh_R.com_pos, state_thigh_R.mass_kg)],
    )

    # Neck/trunk (head child -> trunk)
    head_prox_w = _proximal_wrench(
        state=state_head,
        distal_wrenches=[],
        external_wrenches=[_make_gravity_wrench(state_head.com_pos, state_head.mass_kg)],
    )
    trunk_prox_w = _proximal_wrench(
        state=state_trunk,
        distal_wrenches=[head_prox_w.negate()],
        external_wrenches=[_make_gravity_wrench(state_trunk.com_pos, state_trunk.mass_kg)],
    )

    out: dict[str, np.ndarray] = {}

    # Ankle moments: resolve foot proximal moment in shank frame
    ankle_L_ref = _resolve(foot_L_prox_w.moment, frames.shank_L)
    ankle_R_ref = _resolve(foot_R_prox_w.moment, frames.shank_R)
    knee_L_ref = _resolve(shank_L_prox_w.moment, frames.thigh_L)
    knee_R_ref = _resolve(shank_R_prox_w.moment, frames.thigh_R)
    hip_L_ref = _resolve(thigh_L_prox_w.moment, frames.pelvis)
    hip_R_ref = _resolve(thigh_R_prox_w.moment, frames.pelvis)
    trunk_ref = _resolve(trunk_prox_w.moment, frames.pelvis)
    neck_ref = _resolve(head_prox_w.moment, frames.thorax)

    for axis, j in zip(("X", "Y", "Z"), range(3)):
        out[f"Ankle_L_ref_{axis}_Nm"] = ankle_L_ref[:, j]
        out[f"Ankle_R_ref_{axis}_Nm"] = ankle_R_ref[:, j]
        out[f"Knee_L_ref_{axis}_Nm"] = knee_L_ref[:, j]
        out[f"Knee_R_ref_{axis}_Nm"] = knee_R_ref[:, j]
        out[f"Hip_L_ref_{axis}_Nm"] = hip_L_ref[:, j]
        out[f"Hip_R_ref_{axis}_Nm"] = hip_R_ref[:, j]
        out[f"Trunk_ref_{axis}_Nm"] = trunk_ref[:, j]
        out[f"Neck_ref_{axis}_Nm"] = neck_ref[:, j]

    return out
