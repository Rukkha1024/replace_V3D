from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class C3DPoints:
    """Minimal C3D container for marker trajectories."""

    labels: List[str]  # stripped labels (e.g., "LASI")
    labels_raw: List[str]  # raw labels from file (e.g., "251112_KUO_LASI")
    points: np.ndarray  # (n_frames, n_points, 3) float32, meters
    rate_hz: float
    first_frame: int
    last_frame: int


def _parse_parameters(data: bytes, param_block: int) -> Dict[int, dict]:
    """Parse C3D parameter section into groups->params dict.

    This is intentionally minimal: we only need POINT:LABELS and POINT:UNITS.
    """
    block_size = 512
    start = (param_block - 1) * block_size
    header = data[start : start + 4]
    n_blocks = header[2]
    p = start + 4

    groups: Dict[int, dict] = {}

    while p < start + n_blocks * block_size:
        name_len = struct.unpack("<b", data[p : p + 1])[0]
        group_id = struct.unpack("<b", data[p + 1 : p + 2])[0]
        if name_len == 0:
            break

        name = data[p + 2 : p + 2 + name_len].decode("latin-1")
        offset_field_pos = p + 2 + name_len
        offset_to_next = struct.unpack("<h", data[offset_field_pos : offset_field_pos + 2])[0]
        next_pos = offset_field_pos + offset_to_next

        q = offset_field_pos + 2

        if group_id < 0:
            gid = -group_id
            desc_len = data[q]
            q += 1
            desc = data[q : q + desc_len].decode("latin-1") if desc_len else ""
            groups[gid] = {"name": name, "desc": desc, "params": {}}
        else:
            gid = group_id
            if gid not in groups:
                groups[gid] = {"name": f"GROUP{gid}", "desc": "", "params": {}}

            if q >= next_pos:
                p = next_pos
                continue

            ptype = struct.unpack("<b", data[q : q + 1])[0]
            q += 1
            dim_count = data[q]
            q += 1
            dims = [data[q + i] for i in range(dim_count)]
            q += dim_count

            n_elem = 1
            for d in dims:
                n_elem *= d

            elem_size = abs(ptype)
            data_bytes_len = n_elem * elem_size if ptype != 0 else 0
            raw = data[q : q + data_bytes_len]
            q += data_bytes_len

            desc_len = data[q] if q < next_pos else 0
            q += 1
            desc = data[q : q + desc_len].decode("latin-1") if desc_len else ""

            # decode
            if ptype == 1:
                arr = list(struct.unpack("<" + "b" * n_elem, raw))
                value = arr[0] if dim_count == 0 else arr
            elif ptype == 2:
                arr = list(struct.unpack("<" + "h" * n_elem, raw))
                value = arr[0] if dim_count == 0 else arr
            elif ptype == 4:
                arr = list(struct.unpack("<" + "f" * n_elem, raw))
                value = arr[0] if dim_count == 0 else arr
            elif ptype == -1:
                # char
                if dim_count == 2:
                    strlen, nstr = dims[0], dims[1]
                    value = [
                        raw[i * strlen : (i + 1) * strlen].decode("latin-1").rstrip(" \x00")
                        for i in range(nstr)
                    ]
                else:
                    value = raw.decode("latin-1").rstrip(" \x00")
            else:
                value = raw

            groups[gid]["params"][name] = {"type": ptype, "dims": dims, "value": value, "desc": desc}

        p = next_pos

    return groups


def read_c3d_points(path: str | Path) -> C3DPoints:
    """Read marker trajectories from a C3D file.

    - Supports float-based C3D (scale factor < 0), typical for OptiTrack exports.
    - Returns points in **meters** (expects POINT:UNITS == 'm').
    """
    path = Path(path)
    data = path.read_bytes()

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
    rate_hz = struct.unpack("<f", hdr[20:24])[0]

    if scale >= 0:
        raise NotImplementedError("Integer-scaled C3D is not supported yet (scale>=0).")

    groups = _parse_parameters(data, param_block)

    point_group = None
    for g in groups.values():
        if g["name"].upper() == "POINT":
            point_group = g
            break
    if point_group is None:
        raise ValueError("POINT group not found in C3D parameters.")

    labels_raw = point_group["params"]["LABELS"]["value"]
    units = point_group["params"].get("UNITS", {}).get("value", None)
    if units not in (None, "m"):
        raise ValueError(f"Expected POINT:UNITS == 'm', got {units!r}")

    # Strip leading date/initial prefix like "251112_KUO_"
    labels = [re.sub(r"^\d+_[A-Z]+_", "", lab) for lab in labels_raw]

    n_frames = last_frame - first_frame + 1
    data_start = (data_start_block - 1) * 512

    total_data_bytes = len(data) - data_start
    frame_bytes = total_data_bytes // n_frames
    point_bytes = n_points * 16  # 4 floats per marker (x,y,z,residual)
    if point_bytes > frame_bytes:
        raise ValueError("Unexpected C3D frame layout: point_bytes > frame_bytes")

    pts = np.empty((n_frames, n_points, 3), dtype=np.float32)
    offset = data_start
    for i in range(n_frames):
        frame = data[offset : offset + point_bytes]
        arr = np.frombuffer(frame, dtype="<f4")  # little-endian float32
        xyzr = arr.reshape(n_points, 4)
        pts[i, :, :] = xyzr[:, :3]
        offset += frame_bytes

    return C3DPoints(
        labels=labels,
        labels_raw=labels_raw,
        points=pts,
        rate_hz=float(rate_hz),
        first_frame=int(first_frame),
        last_frame=int(last_frame),
    )
