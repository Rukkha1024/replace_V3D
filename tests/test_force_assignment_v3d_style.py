"""Tests for V3D-style multi-plate block assignment."""

from __future__ import annotations

import numpy as np

from replace_v3d.joint_dynamics.inverse_dynamics import (
    ForceplateWrenchSeries,
    _assign_contact_blocks_v3d_style,
)


def _make_fp(
    *,
    plate_index_1based: int,
    T: int,
    contact_mask: np.ndarray,
    cop_x: np.ndarray,
    corners_xyxy: tuple[float, float, float, float],
) -> ForceplateWrenchSeries:
    min_x, max_x, min_y, max_y = corners_xyxy
    corners = np.asarray(
        [
            [min_x, min_y, 0.0],
            [max_x, min_y, 0.0],
            [max_x, max_y, 0.0],
            [min_x, max_y, 0.0],
        ],
        dtype=float,
    )
    return ForceplateWrenchSeries(
        plate_index_1based=plate_index_1based,
        fp_origin_lab=np.asarray([0.0, 0.0, 0.0], dtype=float),
        grf_lab=np.tile(np.asarray([0.0, 0.0, 700.0], dtype=float), (T, 1)),
        grm_lab_at_fp_origin=np.zeros((T, 3), dtype=float),
        cop_x_m=np.asarray(cop_x, dtype=float),
        cop_y_m=np.zeros(T, dtype=float),
        valid_contact_mask=np.asarray(contact_mask, dtype=bool),
        corners_lab=corners,
    )


def _fixed_point(T: int, x: float, y: float, z: float) -> np.ndarray:
    return np.tile(np.asarray([x, y, z], dtype=float), (T, 1))


def test_block_assigns_to_left_when_left_distance_is_smaller() -> None:
    T = 12
    contact = np.zeros(T, dtype=bool)
    contact[3:7] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.3, -0.2, 0.2),
    )

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.left_mask_per_plate[0][3:7])
    assert not np.any(out.right_mask_per_plate[0][3:7])
    assert not np.any(out.invalid_mask[3:7])
    assert out.assigned_blocks[0].assigned_side == 0


def test_two_feet_same_plate_block_becomes_invalid() -> None:
    T = 10
    contact = np.zeros(T, dtype=bool)
    contact[2:6] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.full(T, 0.5, dtype=float),
        corners_xyxy=(-0.2, 1.2, -0.2, 0.2),
    )

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=1.0,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.invalid_mask[2:6])
    assert out.assigned_blocks[0].assigned_side is None
    assert out.assigned_blocks[0].invalid_reason == "two_feet_same_plate"


def test_incomplete_block_is_invalid_when_option_is_enabled() -> None:
    T = 8
    contact = np.zeros(T, dtype=bool)
    contact[0:3] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.3, -0.2, 0.2),
    )

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.invalid_mask[0:3])
    assert out.assigned_blocks[0].invalid_reason == "incomplete_contact_block"


def test_same_foot_can_be_assigned_to_two_plates() -> None:
    T = 10
    contact = np.zeros(T, dtype=bool)
    contact[3:7] = True
    fp1 = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.3, -0.2, 0.2),
    )
    fp2 = _make_fp(
        plate_index_1based=3,
        T=T,
        contact_mask=contact,
        cop_x=np.full(T, 0.05, dtype=float),
        corners_xyxy=(-0.1, 0.4, -0.2, 0.2),
    )

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp1, fp2],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.left_mask_per_plate[0][3:7])
    assert np.all(out.left_mask_per_plate[1][3:7])
    assert not np.any(out.invalid_mask[3:7])


def test_valid_plate_assignment_is_not_wiped_by_other_plate_threshold_reject() -> None:
    T = 10
    contact = np.zeros(T, dtype=bool)
    contact[2:6] = True
    fp_valid = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.3, -0.2, 0.2),
    )
    fp_far = _make_fp(
        plate_index_1based=3,
        T=T,
        contact_mask=contact,
        cop_x=np.full(T, 5.0, dtype=float),
        corners_xyxy=(4.7, 5.3, -0.2, 0.2),
    )

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp_valid, fp_far],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.left_mask_per_plate[0][2:6])
    assert not np.any(out.invalid_mask[2:6])


def test_projection_accepts_points_on_plate_edge() -> None:
    T = 10
    contact = np.zeros(T, dtype=bool)
    contact[2:6] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.3, -0.2, 0.2),
    )

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.3, 0.0, 0.0),  # exact edge
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.left_mask_per_plate[0][2:6])
    assert not np.any(out.invalid_mask[2:6])


def test_two_feet_projection_without_distance_overlap_keeps_valid_side() -> None:
    T = 10
    contact = np.zeros(T, dtype=bool)
    contact[2:6] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 1.8, -0.2, 0.2),  # both feet project onto same plate
    )

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.5, 0.0, 0.0),  # far from COP block
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=_fixed_point(T, 1.5, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.6, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.left_mask_per_plate[0][2:6])
    assert not np.any(out.invalid_mask[2:6])


def test_projection_requires_prox_and_dist_in_same_frames() -> None:
    T = 10
    contact = np.zeros(T, dtype=bool)
    contact[2:6] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.3, -0.2, 0.2),
    )

    foot_L_prox = _fixed_point(T, 0.0, 0.0, 0.0)
    foot_L_dist = _fixed_point(T, 0.6, 0.0, 0.0)
    foot_L_dist[4:6, 0] = 0.1
    foot_L_prox[2:4, 0] = 0.0
    foot_L_prox[4:6, 0] = 0.6

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=foot_L_prox,
        foot_L_dist=foot_L_dist,
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.invalid_mask[2:6])
    assert out.assigned_blocks[0].invalid_reason == "segment_projection_failed"


def test_two_feet_projection_uses_overlap_not_union() -> None:
    T = 10
    contact = np.zeros(T, dtype=bool)
    contact[2:6] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.8, -0.2, 0.2),
    )

    foot_R_prox = _fixed_point(T, 1.2, 0.0, 0.0)  # outside
    foot_R_dist = _fixed_point(T, 1.2, 0.0, 0.0)  # outside
    foot_R_dist[2:4, 0] = 0.2  # dist inside, prox outside
    foot_R_prox[4:6, 0] = 0.2  # prox inside, dist outside

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 0.3, 0.0, 0.0),  # within threshold but not very_close
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=foot_R_prox,
        foot_R_dist=foot_R_dist,
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.all(out.left_mask_per_plate[0][2:6])
    assert not np.any(out.invalid_mask[2:6])


def test_incomplete_detection_uses_raw_contact_edge_even_with_nonfinite_start_frame() -> None:
    T = 8
    contact = np.zeros(T, dtype=bool)
    contact[0:4] = True
    fp = _make_fp(
        plate_index_1based=1,
        T=T,
        contact_mask=contact,
        cop_x=np.zeros(T, dtype=float),
        corners_xyxy=(-0.3, 0.3, -0.2, 0.2),
    )
    fp = ForceplateWrenchSeries(
        plate_index_1based=fp.plate_index_1based,
        fp_origin_lab=fp.fp_origin_lab,
        grf_lab=fp.grf_lab.copy(),
        grm_lab_at_fp_origin=fp.grm_lab_at_fp_origin.copy(),
        cop_x_m=fp.cop_x_m.copy(),
        cop_y_m=fp.cop_y_m.copy(),
        valid_contact_mask=fp.valid_contact_mask.copy(),
        corners_lab=fp.corners_lab.copy(),
    )
    fp.grf_lab[0, :] = np.nan  # drop cleaned contact frame 0

    out = _assign_contact_blocks_v3d_style(
        forceplates=[fp],
        foot_L_com=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_R_com=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_L_prox=_fixed_point(T, 0.0, 0.0, 0.0),
        foot_L_dist=_fixed_point(T, 0.1, 0.0, 0.0),
        foot_R_prox=_fixed_point(T, 1.0, 0.0, 0.0),
        foot_R_dist=_fixed_point(T, 1.1, 0.0, 0.0),
        threshold_m=0.5,
        remove_incomplete_assignments=True,
        require_segment_projection_on_plate=True,
    )

    assert np.any(out.invalid_mask[1:4])
    assert out.assigned_blocks[0].invalid_reason == "incomplete_contact_block"
