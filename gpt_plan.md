좋아. **1번(Visual3D의 `Resolve_Discontinuity`) 방식으로 “내가 직접 실행해보고”, 그 적용 전/후 변화를 플롯으로 보여주고**, 그 다음에 **repo에 적용할 패치**까지 같이 줄게.

`Resolve_Discontinuity`의 핵심은 Visual3D 문서 그대로야:

* **Euler/Cardan joint angle은 ±180°에서 불연속(랩)이 생길 수 있고**
* 그 불연속 지점에서 **range(보통 360°)를 더/빼서 연속 신호로 만든다**
* event_label 없이 쓰면 “첫 유효 프레임”을 기준으로 시작한다
  → 우리가 구현하는 오프라인 CSV 후처리와 딱 맞음 ([HAS-Motion][1])

---

## 1번 적용 전/후 플롯 (내가 직접 실행한 결과)

### A) Baseline(1..11)에서 mean이 왜 깨지는지 + Resolve_Discontinuity가 어떻게 고치는지

* 파란 점: baseline Z (wrap된 -180..180)
* 주황 점: baseline Z (Resolve_Discontinuity 적용 후; 연속)
* 점선: baseline mean(선형 mean)

**wrap된 상태의 baseline mean = -16.0° (엉뚱함)**
**Resolve_Discontinuity 후 baseline mean = -179.6° (의도대로)**

[Download plot: resolve_discontinuity_baseline.png](sandbox:/mnt/data/resolve_discontinuity_baseline.png)

---

### B) 전체 시계열에서 “현재 ana0” vs “1번 적용 ana0” 비교

* 파란색: Z extracted (wrapped -180..180)
* 주황색: **현재 ana0** = wrap된 값을 그대로 baseline mean 차감 → 스파이크/점프가 심해짐
* 초록색: **1번 적용 ana0** = Resolve_Discontinuity로 연속화 후 baseline mean 차감 → 안정적(0 근처)

[Download plot: resolve_discontinuity_timeseries.png](sandbox:/mnt/data/resolve_discontinuity_timeseries.png)

---

## 왜 이게 Visual3D 방식이냐?

Visual3D 문서에서 `Resolve_Discontinuity(signal, range, ...)`는 **불연속 지점에서 range를 add/subtract 해서 연속화**한다고 명시합니다. 그리고 Euler로 각을 파싱할 때 생기는 ±180° 점프는 “자연스러운 결과”라며, 이걸 제거하는 “convenience function”으로 제공한다고도 적혀 있어요. ([HAS-Motion][1])
즉, 지금 repo의 “Z가 이상하게 보임”은 **각도 계산이 틀린 게 아니라, wrap된 각도에 baseline(mean)을 선형으로 빼서 생긴 후처리 부작용**이고, Visual3D도 그 상황을 전제로 해결책을 제공하는 겁니다. ([HAS-Motion][1])

---

# PATCH (1번: Resolve_Discontinuity 방식 추가)

아래 패치는 repo의 `postprocess_joint_angles()`에 **Visual3D Resolve_Discontinuity 스타일의 불연속 해소(=±360 보정)**를 추가하고, **baseline subtraction 전에 자동 적용**하도록 합니다.

* 기본값: `resolve_discontinuity_range_deg=360.0` (활성)
* 컬럼별로 “정말 불연속이 있는 경우(연속 프레임 점프 > 180°)”에만 적용(불필요한 컬럼은 그대로 유지)
* 원하면 끌 수 있음: `resolve_discontinuity_range_deg=None`

```diff
*** Begin Patch
*** Delete File: src/replace_v3d/joint_angles/postprocess.py
*** End Patch
*** Begin Patch
*** Add File: src/replace_v3d/joint_angles/postprocess.py
+"""Post-processing helpers for Visual3D-like joint angle outputs.
+
+This module intentionally operates *after* joint angles are computed. It does not change
+segment definitions or the joint-angle math.
+
+It provides:
+1) Anatomical "presentation" convention:
+   Visual3D-style right-hand-rule joint angles often produce opposite sign meanings for
+   LEFT vs RIGHT in Y/Z (ab/adduction and internal/external rotation). A common practice
+   is to negate LEFT Y/Z so that the sign meaning matches the RIGHT side.
+
+2) Discontinuity resolution (Visual3D Resolve_Discontinuity-style):
+   Euler/Cardan angles can show ±180° wrap discontinuities. Visual3D provides a convenience
+   function that removes these by adding/subtracting a range (typically 360°) at the point
+   of discontinuity. We implement the same idea here for offline CSV outputs.
+
+3) Baseline-normalized convention:
+   Subtract the mean of a quiet-standing baseline window to remove static offsets.
+"""
+
+from __future__ import annotations
+
+from dataclasses import dataclass
+from typing import Iterable, Sequence
+
+import numpy as np
+import polars as pl
+
+
+@dataclass(frozen=True)
+class JointAnglePostprocessMeta:
+    """Metadata returned by :func:`postprocess_joint_angles`."""
+
+    frame_col: str
+    baseline_frame_start: int
+    baseline_frame_end: int
+    flipped_columns: tuple[str, ...]
+    baseline_values: dict[str, float]
+    resolved_columns: tuple[str, ...] = ()
+    resolve_discontinuity_range_deg: float | None = None
+
+
+def _default_flip_columns(columns: Sequence[str]) -> list[str]:
+    """Default 'anatomical sign unification' columns.
+
+    Convention used here:
+    - keep X (flex/ext) as-is
+    - flip LEFT Y/Z so that Y/Z have the same sign meaning as the RIGHT side
+
+    Targets Hip/Knee/Ankle only; only flips columns that exist.
+    """
+    candidates: list[str] = []
+    for joint in ("Hip", "Knee", "Ankle"):
+        for axis in ("Y", "Z"):
+            candidates.append(f"{joint}_L_{axis}_deg")
+    return [c for c in candidates if c in columns]
+
+
+def _needs_resolve_discontinuity(x_deg: np.ndarray, range_deg: float) -> bool:
+    """Heuristic: if any finite consecutive jump exceeds range/2, we likely crossed the Euler wrap."""
+    idx = np.flatnonzero(np.isfinite(x_deg))
+    if idx.size < 2:
+        return False
+    diffs = np.diff(x_deg[idx])
+    if diffs.size == 0:
+        return False
+    return bool(np.nanmax(np.abs(diffs)) > (range_deg / 2.0))
+
+
+def _resolve_discontinuity_deg(x_deg: np.ndarray, range_deg: float = 360.0) -> tuple[np.ndarray, bool]:
+    """Resolve Euler wrap discontinuities by adding/subtracting `range_deg`.
+
+    This mirrors Visual3D's Resolve_Discontinuity concept: when a discontinuity occurs,
+    add/subtract the expected range (commonly 360°) so the signal becomes continuous.
+
+    Returns
+    -------
+    x_resolved, changed
+      `changed` indicates whether any correction was applied.
+
+    Notes
+    -----
+    - Output may exceed [-180, 180]; this is expected for a "continuous" representation.
+    - NaNs are preserved; processing starts from the first finite sample.
+    """
+    x = np.array(x_deg, dtype=float, copy=True)
+
+    idx = np.flatnonzero(np.isfinite(x))
+    if idx.size == 0:
+        return x, False
+
+    start = idx[0]
+    offset = 0.0
+    prev = x[start]
+    changed = False
+
+    for i in range(start + 1, len(x)):
+        if not np.isfinite(x[i]):
+            continue
+        cur = x[i] + offset
+        diff = cur - prev
+
+        if diff > range_deg / 2.0:
+            offset -= range_deg
+            cur -= range_deg
+            changed = True
+        elif diff < -range_deg / 2.0:
+            offset += range_deg
+            cur += range_deg
+            changed = True
+
+        x[i] = cur
+        prev = cur
+
+    return x, changed
+
+
+def postprocess_joint_angles(
+    df: pl.DataFrame,
+    *,
+    frame_col: str = "Frame",
+    unify_lr_sign: bool = True,
+    baseline_frames: tuple[int, int] | None = (1, 11),
+    flip_columns: Iterable[str] | None = None,
+    resolve_discontinuity_range_deg: float | None = 360.0,
+    resolve_discontinuity_columns: Iterable[str] | None = None,
+) -> tuple[pl.DataFrame, JointAnglePostprocessMeta]:
+    """Apply analysis-friendly post-processing to a joint-angle time series.
+
+    Parameters
+    ----------
+    df:
+        Polars DataFrame containing a frame column and angle columns.
+    frame_col:
+        Name of the frame index column (e.g., ``"Frame"`` or ``"MocapFrame"``).
+    unify_lr_sign:
+        If True, flips selected LEFT Y/Z columns (multiplies by -1) so that left/right
+        have consistent sign meaning.
+    baseline_frames:
+        If provided, subtract the mean of each angle column over this inclusive frame window
+        ``(start, end)``.
+        The default (1, 11) corresponds to python indices [0..10] often used for quiet standing.
+    flip_columns:
+        Optional explicit list of columns to flip. If None, uses the default set (Hip/Knee/Ankle
+        LEFT Y/Z).
+    resolve_discontinuity_range_deg:
+        If not None, apply Visual3D Resolve_Discontinuity-style unwrapping to designated angle
+        columns prior to baseline subtraction. Typical value is 360 degrees.
+        Set to None to disable.
+    resolve_discontinuity_columns:
+        Optional explicit list of angle columns to apply discontinuity resolution to.
+        If None, considers all ``*_deg`` columns and auto-applies only if a wrap jump is detected.
+
+    Returns
+    -------
+    df_out, meta
+        Postprocessed DataFrame (same columns) and metadata.
+    """
+    if frame_col not in df.columns:
+        raise KeyError(f"Frame column not found: {frame_col!r}")
+
+    angle_cols = [c for c in df.columns if c.endswith("_deg")]
+    flipped: list[str] = []
+    resolved: list[str] = []
+
+    out = df
+
+    # 1) Sign unification (LEFT Y/Z)
+    if unify_lr_sign:
+        flip_cols = list(flip_columns) if flip_columns is not None else _default_flip_columns(df.columns)
+        if flip_cols:
+            out = out.with_columns([(-pl.col(c)).alias(c) for c in flip_cols])
+            flipped = flip_cols
+
+    # 2) Resolve Euler wrap discontinuities (Visual3D Resolve_Discontinuity-style)
+    if resolve_discontinuity_range_deg is not None:
+        range_deg = float(resolve_discontinuity_range_deg)
+        if range_deg <= 0:
+            raise ValueError(f"resolve_discontinuity_range_deg must be > 0. Got {resolve_discontinuity_range_deg!r}")
+
+        cols = (
+            [c for c in resolve_discontinuity_columns]  # type: ignore[arg-type]
+            if resolve_discontinuity_columns is not None
+            else angle_cols
+        )
+        cols = [c for c in cols if c in out.columns]
+
+        repl: list[pl.Series] = []
+        for c in cols:
+            x = out.get_column(c).to_numpy()
+            if not _needs_resolve_discontinuity(x, range_deg=range_deg):
+                continue
+            x2, changed = _resolve_discontinuity_deg(x, range_deg=range_deg)
+            if changed:
+                repl.append(pl.Series(c, x2))
+                resolved.append(c)
+
+        if repl:
+            out = out.with_columns(repl)
+
+    # 3) Baseline subtraction (quiet standing mean)
+    baseline_values: dict[str, float] = {}
+    if baseline_frames is not None:
+        b0, b1 = int(baseline_frames[0]), int(baseline_frames[1])
+        if b1 < b0:
+            raise ValueError(f"baseline_frames must satisfy end>=start. Got {baseline_frames!r}")
+
+        base_df = out.filter(pl.col(frame_col).is_between(b0, b1, closed="both"))
+        if base_df.height == 0:
+            raise ValueError(
+                f"No rows found for baseline window {baseline_frames!r} using frame_col={frame_col!r}."
+            )
+
+        base_row = base_df.select([pl.col(c).mean().alias(c) for c in angle_cols]).row(0)
+        baseline_values = {c: float(v) for c, v in zip(angle_cols, base_row)}
+        out = out.with_columns([(pl.col(c) - pl.lit(baseline_values[c])).alias(c) for c in angle_cols])
+
+    meta = JointAnglePostprocessMeta(
+        frame_col=str(frame_col),
+        baseline_frame_start=int(baseline_frames[0]) if baseline_frames is not None else -1,
+        baseline_frame_end=int(baseline_frames[1]) if baseline_frames is not None else -1,
+        flipped_columns=tuple(flipped),
+        baseline_values=baseline_values,
+        resolved_columns=tuple(resolved),
+        resolve_discontinuity_range_deg=resolve_discontinuity_range_deg,
+    )
+
+    return out, meta
*** End Patch
```

선택(권장)으로, 문서도 ana0 정의에 “불연속 해소” 단계를 명시하려면 아래도 같이 적용하세요:

```diff
*** Begin Patch
*** Delete File: JOINT_ANGLE_CONVENTIONS.md
*** End Patch
*** Begin Patch
*** Add File: JOINT_ANGLE_CONVENTIONS.md
+## 관절각 출력 컨벤션 (표준 출력 = ana0)
+
+본 저장소는 마커 기반 세그먼트 좌표계로부터 Visual3D 방식의 3D 관절각(내재적 XYZ)을 계산합니다.
+이제 **관절각의 표준 출력은 ana0 하나만** 사용합니다.
+
+- 저장 파일: `*_JOINT_ANGLES_preStep.csv`
+- 의미: **ana0 = (좌우 부호 통일) + (Resolve_Discontinuity) + (quiet standing 기저선 차감)**
+
+---
+
+## ana0 정의 (이 저장소의 표준)
+
+ana0는 아래 단계를 순서대로 적용한 관절각 시계열입니다.
+
+### 1) 좌우 부호 통일 (LEFT만 적용)
+Hip/Knee/Ankle의 **LEFT** Y/Z 성분만 부호를 반전합니다.
+- `*_L_Y_deg = - *_L_Y_deg`
+- `*_L_Z_deg = - *_L_Z_deg`
+목적: 좌/우 비교 시 Y/Z 부호의 의미를 LEFT와 RIGHT 간에 일관되게 맞춤.
+
+### 2) Resolve_Discontinuity (Euler wrap 불연속 해소)
+Euler/Cardan 각은 ±180° 경계에서 값이 갑자기 -180↔+180으로 “점프”할 수 있습니다.
+Visual3D의 `Resolve_Discontinuity(signal, range, ...)`와 같은 아이디어로,
+불연속 지점에서 **range(기본 360°)** 를 더/빼서 신호를 연속화합니다.
+
+> 구현 기본값: `resolve_discontinuity_range_deg=360.0`
+> 비활성화: `resolve_discontinuity_range_deg=None`
+
+### 3) quiet standing 기저선 차감
+정적 기립 구간 평균을 빼서 정적 오프셋을 제거합니다.
+- 기저선 구간: **프레임 1..11** (양 끝 포함)
+- 모든 `*_deg` 열에 대해:
+  - `angle = angle - mean(angle[1..11])`
+
+---
+
+## 구현 위치
+- 후처리 로직: `src/replace_v3d/joint_angles/postprocess.py`
+- 단일 시행 내보내기(ana0만 저장): `scripts/run_joint_angles_pipeline.py`
+- 배치 통합 CSV 내 관절각(ana0 값 사용): `scripts/run_batch_all_timeseries_csv.py`
*** End Patch
```

---

# VSCode Codex CLI에 붙여 넣을 프롬프트

> **Codex Prompt**
>
> Apply the patch as follows:
>
> 1. Replace `src/replace_v3d/joint_angles/postprocess.py` with the new version in the patch. This adds a Visual3D Resolve_Discontinuity-style step (range=360°) before baseline subtraction, applied per-column only when a wrap jump (>180°) is detected.
> 2. (Optional but recommended) Replace `JOINT_ANGLE_CONVENTIONS.md` with the patched version documenting the new ana0 steps.
>
> After applying:
>
> * Run the usual pipeline that generates `*_JOINT_ANGLES_preStep.csv`.
> * Plot `Ankle_*_Z_deg` before/after the change (or compare against a previous run) and verify that:
>
>   * The baseline window frames 1..11 no longer mixes +179/-179 in a way that breaks the baseline mean.
>   * The “ana0” output no longer exhibits ±180/±360 spike artifacts in ankle Z.

