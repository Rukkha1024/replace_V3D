선택지 **B(세그먼트 정의를 바꿔서 Ankle_*_Z_deg 자체를 “정상 복원”)** 가 “근본 해결”인 건 맞아요. 지금 repo에서 Ankle Z가 거의 0(마이크로 degree)로 **구조적으로 눌리는 이유**가 “후처리/플로팅”이 아니라 **Shank(하퇴) SCS 정의가 Foot SCS와 같은 축을 공유하도록 만들어져 있어서**이기 때문입니다.

그리고 “V3D(Visual3D)와 동일”하게 가려면, Visual3D가 세그먼트 좌표계를 만드는 방식(특히 **Method 2: proximal/distal 각각 medial+lat border 4점으로 plane LS-fit**)에 맞춰서 shank 좌표계를 만들어야 합니다. Visual3D 문서에서도 4개의 border target을 쓰는 경우 **최소자승(least squares)로 anatomical/frontal plane을 맞추고**, **inferior/superior 축(기본 Z)을 distal→proximal joint center 벡터로 정렬**한 다음, **plane과 Z에 수직인 축을 만들고 오른손좌표계로 X를 완성**한다고 명시합니다. ([HAS-Motion][1])
또 Visual3D의 기본 joint angle은 **Cardan X-Y-Z**(z-up, y-anterior 세그먼트 좌표계에서 JCS와 동치 취급)이며, 좌/우는 오른손 법칙 때문에 부호 해석이 달라질 수 있음을 문서가 설명합니다. ([HAS-Motion][2])

---

## 왜 지금 정의가 Ankle Z를 “0으로 고정”시키나 (핵심)

현재 `src/replace_v3d/joint_angles/v3d_joint_angles.py`에서

* Shank X-hint: `LFoot_3 - LANK` (발목 내/외측 마커쌍)
* Foot X-hint: `LFoot_3 - LANK` (똑같은 벡터)

즉, **Shank와 Foot의 X축이 같은 정보로 만들어져서** 상대회전에서 “yaw(=Z)” 성분이 거의 사라집니다. (dist의 x축이 ref의 x축에 대해 회전하지 못하게 잠금)

이건 Visual3D 방식과도 어긋납니다. Visual3D에서 **Method 2로 shank를 만들면** knee medial/lateral + ankle medial/lateral 4점으로 plane을 만들기 때문에, shank의 frontal plane(=XZ)와 그로부터 나오는 축이 **발목쌍 하나만으로 결정되지 않습니다**. ([HAS-Motion][1])

---

## “V3D 동일” 옵션 B의 구현 방향

당신 repo 마커셋은 이미 **knee/ankle 양쪽에 medial+lateral 쌍**을 갖고 있어요(문서/코드 주석 기준: Knee = LKNE/RKNE + LShin_3/RShin_3, Ankle = LANK/RANK + LFoot_3/RFoot_3).
따라서 shank는 Visual3D **Method 2**를 그대로 구현할 수 있습니다:

* border targets (4개): knee(lat,med) + ankle(lat,med)
* per-frame LS-fit plane normal 계산
* Z: kneeJC → ankleJC (distal→proximal)
* Y: plane ⟂ & Z ⟂
* X: 오른손좌표계로 완성 + “우측(+Right)” 방향으로 부호 고정

이렇게 하면 **Foot는 기존대로(ankle axis + heel→toe)** 두더라도, shank X가 foot X와 더 이상 동일하게 잠기지 않아 **Ankle_*_Z_deg가 구조적으로 0에 수렴하는 문제는 해소**됩니다.

> 참고: Visual3D는 원칙적으로 static trial을 통해 SCS를 정의하는 워크플로를 전제로 설명합니다. ([HAS-Motion][3])
> 하지만 지금 repo처럼 “프레임마다 마커로 SCS를 직접 재구성”하는 접근에서도, **Method 2 plane-fit로 shank를 만들면** 최소한 “ankle axis로 shank X를 고정”하는 현재 문제는 제거됩니다.

---

## 적용 patch (unified diff)

```diff
diff --git a/src/replace_v3d/joint_angles/v3d_joint_angles.py b/src/replace_v3d/joint_angles/v3d_joint_angles.py
index 6c2d8e0..c1a4d73 100644
--- a/src/replace_v3d/joint_angles/v3d_joint_angles.py
+++ b/src/replace_v3d/joint_angles/v3d_joint_angles.py
@@ -132,6 +132,31 @@ def _frame_from_yz(y0: np.ndarray, z0: np.ndarray, right_hint: Optional[np.ndarr
 
     return np.stack([x, y, z], axis=-1)
 
+
+def _fit_plane_normal_ls(P: np.ndarray) -> np.ndarray:
+    """Least-squares plane normal for point sets (per frame).
+
+    Parameters
+    ----------
+    P : (T,K,3)
+        K>=3 points defining a plane (K=4 for Visual3D Method 2 shank example).
+
+    Returns
+    -------
+    n_unit : (T,3)
+        Unit normal vectors of the best-fit plane for each frame.
+    """
+    if P.ndim != 3 or P.shape[-1] != 3:
+        raise ValueError(f"P must have shape (T,K,3). Got {P.shape}")
+
+    # Center points per frame then take the smallest singular vector.
+    c = np.mean(P, axis=1, keepdims=True)
+    A = P - c
+    _u, _s, vh = np.linalg.svd(A, full_matrices=False)
+    n = vh[:, -1, :]
+    return _normalize(n)
+
 
 def _m(points: np.ndarray, labels: list[str], name: str) -> np.ndarray:
     try:
@@ -241,7 +266,12 @@ def build_segment_frames(
     thigh_L = _frame_from_xz(xhint_thigh_L, z0_thigh_L, right_hint=xhint_thigh_L)
     thigh_R = _frame_from_xz(xhint_thigh_R, z0_thigh_R, right_hint=xhint_thigh_R)
 
-    # Shank (L/R): Z=knee->ankle (proximal), X=+Right (ankle medial/lateral)
+    # Shank (L/R): Visual3D-style Method 2 (4 border targets)
+    # - Fit the anatomical/frontal plane from medial+lateral markers at the knee and ankle
+    # - Align Z (inferior/superior) with knee->ankle (proximal)
+    # - Define Y as perpendicular to both the plane and Z, then complete X by right-hand rule
+    #
+    # This avoids the previous "X-axis lock" (shank X == foot X) that collapses Ankle_*_Z.
     LANK = _m(pts, labels, "LANK")
     RANK = _m(pts, labels, "RANK")
     LFoot = _m(pts, labels, "LFoot_3")
@@ -252,10 +282,17 @@ def build_segment_frames(
 
     z0_shank_L = knee_L - ankle_L
     z0_shank_R = knee_R - ankle_R
-    xhint_shank_L = LFoot - LANK  # medial - lateral (left): to right
-    xhint_shank_R = RANK - RFoot  # lateral - medial (right): to right
-    shank_L = _frame_from_xz(xhint_shank_L, z0_shank_L, right_hint=xhint_shank_L)
-    shank_R = _frame_from_xz(xhint_shank_R, z0_shank_R, right_hint=xhint_shank_R)
+    # Best-fit plane normals (per frame) using 4 border targets (Method 2).
+    # Order doesn't matter for least-squares plane fit.
+    y0_shank_L = _fit_plane_normal_ls(np.stack([LKNE, LShin, LANK, LFoot], axis=1))
+    y0_shank_R = _fit_plane_normal_ls(np.stack([RKNE, RShin, RANK, RFoot], axis=1))
+
+    right_hint_shank_L = LShin - LKNE  # medial - lateral (left): points to right
+    right_hint_shank_R = RKNE - RShin  # lateral - medial (right): points to right
+
+    shank_L = _frame_from_yz(y0_shank_L, z0_shank_L, right_hint=right_hint_shank_L)
+    shank_R = _frame_from_yz(y0_shank_R, z0_shank_R, right_hint=right_hint_shank_R)
 
     # Foot (L/R): X=+Right (ankle axis), Y=heel->toe (anterior), Z=Up
     LTOE = _m(pts, labels, "LTOE")
```

---

## Codex CLI에게 보낼 프롬프트 (그대로 복붙)

아래 프롬프트만 전달하면 됩니다.

> **PROMPT (for Codex CLI in VSCode)**
>
> repo `replace_V3D`에서 `Ankle_L_Z_deg`, `Ankle_R_Z_deg`가 거의 0(마이크로-degree)로 붕괴되는 문제가 있다. 원인은 `src/replace_v3d/joint_angles/v3d_joint_angles.py`의 `build_segment_frames()`에서 shank X축을 발목 내/외측 마커쌍(`LFoot_3-LANK`, `RANK-RFoot_3`)으로 정의해서 foot X축과 동일한 정보로 “잠금(lock)”되기 때문이다. 이 때문에 `R_rel = shank^T @ foot`에서 yaw(=intrinsic XYZ의 Z) 성분이 구조적으로 0에 수렴한다.
>
> 해결은 **선택지 B**: 세그먼트 정의를 Visual3D 방식으로 고쳐서 `Ankle_*_Z_deg` 자체를 정상 복원해야 한다. Visual3D의 Segment Coordinate System 정의에서 **Method 2(4 border targets)**는 knee/ankle의 medial+lateral 4점을 사용해 least-squares로 frontal(anatomical) plane을 맞추고, Z축은 distal→proximal joint center 벡터로 정렬하며, Y축은 plane과 Z에 수직이 되도록 만들고 X축은 오른손 규칙으로 완성한다.
>
> 구현 요구사항:
>
> 1. `v3d_joint_angles.py`에 per-frame LS plane normal을 구하는 helper를 추가한다. (입력 (T,K,3), SVD로 smallest singular vector)
> 2. shank frame 생성부를 교체한다.
>
>    * `y0_shank_L = plane_normal( [LKNE, LShin_3, LANK, LFoot_3] )`
>    * `y0_shank_R = plane_normal( [RKNE, RShin_3, RANK, RFoot_3] )`
>    * `z0_shank_* = kneeJC - ankleJC`
>    * `shank_* = _frame_from_yz(y0_shank_*, z0_shank_*, right_hint = (LShin_3-LKNE)/(RKNE-RShin_3))`로 +Right 방향을 강제
> 3. foot frame은 일단 기존 정의(ankle axis + heel→toe)를 유지한다.
> 4. 변경 후 아래 검증을 자동으로 수행하는 간단한 스크립트/커맨드도 함께 제공해라:
>
>    * `output/all_trials_timeseries.csv` 또는 단일 trial 결과에서 `Ankle_*_Z_deg`의 range가 더 이상 거의 0이 아닌지 확인
>    * `scripts/plot_grid_timeseries.py --sample --only_subjects 강비은 --only_velocities 30`로 생성되는 `output/figures/grid_timeseries/joint_angles_lower__subject-강비은__velocity-30__sample.png`에서 ankle Z가 꺼져 있지 않은지 확인
>
> 위 요구를 만족하도록 **unified diff patch**를 작성해서 적용해줘.

---
