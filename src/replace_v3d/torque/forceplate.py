from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..io.c3d_reader import _parse_parameters


@dataclass(frozen=True)
class C3DAnalogAvg:
    """Per-frame averaged analog channels.

    Notes
    -----
    - C3D analog channels are often sampled faster than mocap points.
      This loader averages the sub-samples within each mocap frame.
    - Returned `values` is shape (n_frames, n_channels).
    """

    labels: List[str]
    rate_hz: float
    samples_per_frame: int
    values: np.ndarray  # (n_frames, n_channels)


@dataclass(frozen=True)
class ForcePlatform:
    """Force platform geometry + channel mapping."""

    index_1based: int
    fp_type: int
    corners_lab: np.ndarray  # (4,3)
    origin_plate: np.ndarray  # (3,) in plate coordinates
    origin_lab: np.ndarray  # (3,) in lab coordinates
    R_pl2lab: np.ndarray  # (3,3)
    channel_numbers_1based: np.ndarray  # (6,) [Fx,Fy,Fz,Mx,My,Mz]
    channel_indices_0based: np.ndarray  # (6,) zero-based indices into ANALOG channels


@dataclass(frozen=True)
class ForcePlatformCollection:
    point_rate_hz: float
    analog: C3DAnalogAvg
    platforms: List[ForcePlatform]


def _rotation_from_corners(corners_lab: np.ndarray) -> np.ndarray:
    """Build a plate->lab rotation matrix using FORCE_PLATFORM:CORNERS.

    Corner order is assumed to follow the C3D / Visual3D convention:
    1=(+x,+y), 2=(-x,+y), 3=(-x,-y), 4=(+x,-y) in the *plate* coordinate system,
    while `corners_lab` are provided in lab coordinates.

    Returns
    -------
    R : (3,3)
        Columns are the plate basis vectors expressed in lab coordinates.
        Thus: v_lab = R @ v_plate.
    """

    c1, c2, c3, c4 = np.asarray(corners_lab, dtype=float)
    # Use averaged opposite edges (more robust than a single edge).
    x_vec = 0.5 * ((c1 - c2) + (c4 - c3))
    y_vec = 0.5 * ((c2 - c3) + (c1 - c4))

    def _safe_unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            raise ValueError("Degenerate forceplate corner geometry: zero-length axis")
        return v / n

    x_hat = _safe_unit(x_vec)
    y_hat = _safe_unit(y_vec)
    z_hat = np.cross(x_hat, y_hat)
    z_hat = _safe_unit(z_hat)

    # Re-orthogonalize Y to guarantee orthonormal basis
    y_hat = np.cross(z_hat, x_hat)
    y_hat = _safe_unit(y_hat)

    return np.column_stack([x_hat, y_hat, z_hat])


def _read_c3d_header(data: bytes) -> Tuple[int, int, int, int, float, int, float]:
    """Return essential C3D header fields.

    Returns
    -------
    (param_block, n_points, first_frame, last_frame, scale, data_start_block, point_rate_hz)
    """

    hdr = data[:512]
    param_block = hdr[0]
    key = hdr[1]
    if key != 80:
        raise ValueError(f"Not a C3D file (header key={key}, expected 80).")

    n_points = struct.unpack("<H", hdr[2:4])[0]
    first_frame = struct.unpack("<H", hdr[6:8])[0]
    last_frame = struct.unpack("<H", hdr[8:10])[0]
    scale = struct.unpack("<f", hdr[12:16])[0]
    data_start_block = struct.unpack("<H", hdr[16:18])[0]
    point_rate_hz = struct.unpack("<f", hdr[20:24])[0]
    return (
        int(param_block),
        int(n_points),
        int(first_frame),
        int(last_frame),
        float(scale),
        int(data_start_block),
        float(point_rate_hz),
    )


def read_c3d_analog_avg(path: str | Path) -> Tuple[float, C3DAnalogAvg]:
    """Read per-frame averaged analog data.

    Returns
    -------
    point_rate_hz : float
    analog : C3DAnalogAvg
    """

    path = Path(path)
    data = path.read_bytes()

    param_block, n_points, first_frame, last_frame, scale, data_start_block, point_rate_hz = _read_c3d_header(data)
    if scale >= 0:
        raise NotImplementedError("Integer-scaled C3D is not supported yet (scale>=0).")

    groups = _parse_parameters(data, param_block)

    # ANALOG group
    analog_group = None
    for g in groups.values():
        if g["name"].upper() == "ANALOG":
            analog_group = g
            break
    if analog_group is None:
        raise ValueError("ANALOG group not found in C3D parameters.")
    aparams = analog_group["params"]
    analog_used = int(aparams["USED"]["value"])
    analog_rate_hz = float(aparams["RATE"]["value"])
    analog_labels = list(aparams["LABELS"]["value"])
    if len(analog_labels) != analog_used:
        analog_labels = analog_labels[:analog_used]

    samples_per_frame = int(round(analog_rate_hz / point_rate_hz))
    if samples_per_frame <= 0:
        raise ValueError(f"Invalid analog sampling ratio: analog_rate={analog_rate_hz}, point_rate={point_rate_hz}")

    n_frames = int(last_frame - first_frame + 1)
    data_start = int((data_start_block - 1) * 512)

    total_data_bytes = len(data) - data_start
    frame_bytes = total_data_bytes // n_frames
    point_bytes = n_points * 16  # float32 points: (x,y,z,residual) per marker
    if point_bytes > frame_bytes:
        raise ValueError("Unexpected C3D frame layout: point_bytes > frame_bytes")

    analog_bytes = frame_bytes - point_bytes
    expected_float = analog_used * samples_per_frame * 4
    expected_int16 = analog_used * samples_per_frame * 2
    if analog_bytes == expected_float:
        analog_dtype = "<f4"
        scale_mode = "float"
    elif analog_bytes == expected_int16:
        analog_dtype = "<i2"
        scale_mode = "int16"
    else:
        raise ValueError(
            "Unexpected analog byte size per frame. "
            f"analog_bytes={analog_bytes}, expected_float={expected_float}, expected_int16={expected_int16}"
        )

    values = np.empty((n_frames, analog_used), dtype=float)

    # Optional scaling params for int16 mode
    gen_scale = float(aparams.get("GEN_SCALE", {}).get("value", 1.0))
    scale_vec = aparams.get("SCALE", {}).get("value", None)
    offset_vec = aparams.get("OFFSET", {}).get("value", None)
    if scale_mode == "int16":
        if scale_vec is None or offset_vec is None:
            raise ValueError("ANALOG:SCALE / ANALOG:OFFSET required for int16 analog data")
        scale_vec = np.asarray(scale_vec, dtype=float)[:analog_used]
        offset_vec = np.asarray(offset_vec, dtype=float)[:analog_used]
    else:
        scale_vec = None
        offset_vec = None

    offset = data_start
    for i in range(n_frames):
        frame = data[offset : offset + frame_bytes]
        analog_block = frame[point_bytes : point_bytes + analog_bytes]
        arr = np.frombuffer(analog_block, dtype=analog_dtype)
        arr = arr.reshape((samples_per_frame, analog_used))
        if scale_mode == "float":
            values[i, :] = arr.mean(axis=0)
        else:
            # Classic C3D scaling: (raw - offset) * scale * gen_scale
            raw = arr.astype(float)
            scaled = (raw - offset_vec[None, :]) * scale_vec[None, :] * gen_scale
            values[i, :] = scaled.mean(axis=0)
        offset += frame_bytes

    return point_rate_hz, C3DAnalogAvg(
        labels=analog_labels,
        rate_hz=analog_rate_hz,
        samples_per_frame=samples_per_frame,
        values=values,
    )


def read_force_platforms(path: str | Path) -> ForcePlatformCollection:
    """Read FORCE_PLATFORM metadata + averaged analog signals."""

    path = Path(path)
    data = path.read_bytes()

    param_block, _, _, _, _, _, point_rate_hz = _read_c3d_header(data)
    groups = _parse_parameters(data, param_block)

    # ANALOG avg
    point_rate_hz2, analog = read_c3d_analog_avg(path)
    if abs(point_rate_hz2 - point_rate_hz) > 1e-6:
        point_rate_hz = point_rate_hz2

    # FORCE_PLATFORM group
    fp_group = None
    for g in groups.values():
        if g["name"].upper() == "FORCE_PLATFORM":
            fp_group = g
            break
    if fp_group is None:
        raise ValueError("FORCE_PLATFORM group not found in C3D parameters.")

    fpp = fp_group["params"]
    n_fp = int(fpp["USED"]["value"])
    fp_types = np.asarray(fpp.get("TYPE", {}).get("value", [0] * n_fp), dtype=int)

    # CORNERS dims typically [3,4,n_fp]
    corners_flat = np.asarray(fpp["CORNERS"]["value"], dtype=float)
    corners_dims = fpp["CORNERS"]["dims"]
    if corners_dims != [3, 4, n_fp]:
        raise ValueError(f"Unexpected FORCE_PLATFORM:CORNERS dims={corners_dims}, expected [3,4,{n_fp}]")
    corners = corners_flat.reshape((3, 4, n_fp), order="F")  # column-major per C3D spec

    # ORIGIN dims typically [3,n_fp]
    origin_flat = np.asarray(fpp["ORIGIN"]["value"], dtype=float)
    origin_dims = fpp["ORIGIN"]["dims"]
    if origin_dims != [3, n_fp]:
        raise ValueError(f"Unexpected FORCE_PLATFORM:ORIGIN dims={origin_dims}, expected [3,{n_fp}]")
    origin_plate = origin_flat.reshape((3, n_fp), order="F")

    # CHANNEL dims typically [6,n_fp]
    ch_flat = np.asarray(fpp["CHANNEL"]["value"], dtype=int)
    ch_dims = fpp["CHANNEL"]["dims"]
    if ch_dims != [6, n_fp]:
        raise ValueError(f"Unexpected FORCE_PLATFORM:CHANNEL dims={ch_dims}, expected [6,{n_fp}]")
    ch = ch_flat.reshape((6, n_fp), order="F")

    platforms: List[ForcePlatform] = []
    for i in range(n_fp):
        corners_lab = corners[:, :, i].T  # (4,3)
        R = _rotation_from_corners(corners_lab)
        origin_lab = corners_lab.mean(axis=0) + (R @ origin_plate[:, i])
        ch_num_1 = ch[:, i].astype(int)
        ch_idx_0 = ch_num_1 - 1
        platforms.append(
            ForcePlatform(
                index_1based=i + 1,
                fp_type=int(fp_types[i]) if len(fp_types) > i else 0,
                corners_lab=corners_lab,
                origin_plate=origin_plate[:, i].astype(float),
                origin_lab=origin_lab.astype(float),
                R_pl2lab=R.astype(float),
                channel_numbers_1based=ch_num_1,
                channel_indices_0based=ch_idx_0,
            )
        )

    return ForcePlatformCollection(point_rate_hz=float(point_rate_hz), analog=analog, platforms=platforms)


def choose_active_force_platform(
    analog_avg: np.ndarray,
    platforms: List[ForcePlatform],
) -> ForcePlatform:
    """Pick the platform with the largest |Fz| across the trial."""

    best: ForcePlatform | None = None
    best_score = -float("inf")
    for fp in platforms:
        fz_idx = int(fp.channel_indices_0based[2])
        if fz_idx < 0 or fz_idx >= analog_avg.shape[1]:
            continue
        score = float(np.nanmax(np.abs(analog_avg[:, fz_idx])))
        if score > best_score:
            best_score = score
            best = fp
    if best is None:
        raise ValueError("Could not select an active force platform (no valid Fz channels)")
    return best


def extract_platform_wrenches_lab(
    analog_avg: np.ndarray,
    fp: ForcePlatform,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (F_lab, M_lab) for the selected force platform.

    Parameters
    ----------
    analog_avg
        (n_frames, n_analog) averaged analog channels.
    fp
        ForcePlatform mapping with channel_indices_0based = [Fx,Fy,Fz,Mx,My,Mz].
    """

    idx = fp.channel_indices_0based.astype(int)
    F_plate = analog_avg[:, idx[0:3]]
    M_plate = analog_avg[:, idx[3:6]]
    # row-vector transform: v_lab = v_plate @ R^T
    F_lab = F_plate @ fp.R_pl2lab.T
    M_lab = M_plate @ fp.R_pl2lab.T
    return F_lab, M_lab
