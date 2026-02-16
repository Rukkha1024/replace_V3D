
너는 Git repo(Rukkha1024/replace_V3D)의 코드를 수정한다.

목표:
- 단일 시행 관절각 파이프라인(scripts/run_joint_angles_pipeline.py)은 이제부터 출력 파일을 딱 1개만 저장한다.
- 그 파일은 항상 ana0이며 파일명은 기존 표준 이름을 유지해서:
  <trial>_JOINT_ANGLES_preStep.csv 로 저장한다.
- raw/anat/ana0 별도 파일 저장, Excel(xlsx) 저장, MD5 검증용 스키마 보존 로직은 전부 삭제한다(유지/백업 금지).
- ana0 정의는 그대로 유지한다:
  1) 좌우 부호 통일(LEFT Hip/Knee/Ankle Y/Z만 negate)
  2) quiet standing baseline mean subtraction (frames 1..11 inclusive)
  => postprocess_joint_angles(df_raw, unify_lr_sign=True, baseline_frames=(1,11)) 결과를 저장한다.

수정 범위:
1) scripts/run_joint_angles_pipeline.py
   - pandas import 제거
   - df_raw 생성 후 postprocess_joint_angles 1번만 호출해 df_ana0 생성
   - df_ana0만 <trial>_JOINT_ANGLES_preStep.csv 로 저장(polars write_csv 사용)
   - 다른 저장/메타/Excel/print 로직 삭제

2) README.md
   - single-trial outputs에서 JOINT_ANGLES 관련 설명을 csv 1개(ana0)로 수정
   - joint angles 섹션 outputs에서도 xlsx/MD5 언급 제거하고 csv(ana0)만 남김

3) JOINT_ANGLE_CONVENTIONS.md
   - “raw vs anat vs ana0 출력파일 여러 개” 서술 제거
   - 표준 출력은 ana0 1개 파일(<trial>_JOINT_ANGLES_preStep.csv)이라는 점을 명확히 기술
   - ana0의 2단계 정의(부호 통일 + frames 1..11 baseline subtraction)만 남김

4) main.py
   - _collect_outputs에서 angles 단계 산출물로 <trial>_JOINT_ANGLES_preStep.csv만 수집하도록 수정
     (xlsx/anat/ana0 파일명 제거)

제약:
- 동작/옵션/엔드프레임(스텝온셋 직전까지) 로직은 유지
- 코드 스타일은 기존 파일 톤 유지
- 불필요한 기능 추가 금지(새 옵션 만들지 말 것)
```

---

## 2) 패치 (git diff)

아래 패치를 그대로 적용하면 됩니다.

```diff
diff --git a/JOINT_ANGLE_CONVENTIONS.md b/JOINT_ANGLE_CONVENTIONS.md
index a6467fc..e83e0d8 100644
--- a/JOINT_ANGLE_CONVENTIONS.md
+++ b/JOINT_ANGLE_CONVENTIONS.md
@@ -1,73 +1,40 @@
-## 관절각 출력 컨벤션 (raw vs anat vs ana0)
+## 관절각 출력 컨벤션 (표준 출력 = ana0)
 
-본 저장소는 마커 기반 세그먼트 좌표계로부터 Visual3D 방식의 3D 관절각(내재적 XYZ)을
-계산합니다.
+본 저장소는 마커 기반 세그먼트 좌표계로부터 Visual3D 방식의 3D 관절각(내재적 XYZ)을 계산합니다.
 
-중요: **raw 관절각 계산 자체는 아래 컨벤션에 의해 변경되지 않습니다.**
-아래 컨벤션은 내보낸 시계열에 적용되는 *후처리*입니다.
+이제 **단일 시행(scripts/run_joint_angles_pipeline.py)의 최종 저장 파일은 1개만** 생성하며,
+그 파일은 **항상 ana0**입니다.
 
----
-
-## Raw 컨벤션 (`*_JOINT_ANGLES_preStep.csv`)
-
-Raw 출력은 다음 과정의 직접적인 결과입니다:
-
-- 세그먼트 프레임 구성
-- 상대 회전 (근위 → 원위)
-- 내재적 XYZ 오일러 분해
-
-Raw 출력은 재현성과 MD5 검증을 위해 변경 없이 유지됩니다.
-
-각도 열은 `*_deg`로 끝나며, 다음을 포함합니다:
-
-- Hip / Knee / Ankle: 좌우 구분 (예: `Hip_L_Y_deg`, `Hip_R_Y_deg`)
-- Trunk / Neck: 좌우 구분 없음 (예: `Trunk_Y_deg`)
+- 저장 파일: `*_JOINT_ANGLES_preStep.csv`
+- 의미: **ana0 = (좌우 부호 통일) + (quiet standing 기저선 차감)**
 
 ---
 
-## 해부학적 표현 컨벤션 (`*_anat.csv`)
+## ana0 정의 (이 저장소의 표준)
 
-`*_anat.csv`는 raw 관절각의 후처리 사본으로, 단일 목표를 가집니다:
+ana0는 아래 2단계를 순서대로 적용한 관절각 시계열입니다.
 
-좌우 비교 시 Y/Z 부호의 의미를 LEFT와 RIGHT 간에 일관되게 만드는 것.
+### 1) 좌우 부호 통일 (LEFT만 적용)
 
-**Hip/Knee/Ankle** (왼쪽만 해당):
+Hip/Knee/Ankle의 **LEFT** Y/Z 성분만 부호를 반전합니다.
 
 - `*_L_Y_deg = - *_L_Y_deg`
 - `*_L_Z_deg = - *_L_Z_deg`
 
-기저선 차감은 수행하지 않습니다.
+목적: 좌/우 비교 시 Y/Z 부호의 의미를 LEFT와 RIGHT 간에 일관되게 맞춤.
 
-### `_anat` 적용 후 실질적 해석
+### 2) quiet standing 기저선 차감
 
-부호 반전 후 (RIGHT를 기준 의미로 사용):
-
-- **Y 양수:** 내전(adduction) (좌우 동일)
-- **Z 양수:** 내회전(internal rotation) (좌우 동일)
-
-X는 변경 없음.
-
----
-
-## 기저선 정규화 컨벤션 (`*_ana0.csv`)
-
-`*_ana0.csv`는 `_anat`과 동일한 부호 반전 각도에서 시작한 뒤, 정적 기립
-기저선을 차감하여 정적 오프셋을 제거합니다.
+정적 기립 구간 평균을 빼서 정적 오프셋을 제거합니다.
 
 - 기저선 구간: **프레임 1..11** (양 끝 포함)
 - 모든 `*_deg` 열에 대해:
   - `angle = angle - mean(angle[1..11])`
 
-이는 **Δ각도** 비교 및 세그먼트 좌표계 정렬 오차로 인한 작은 정적 오프셋
-제거에 유용합니다.
-
 ---
 
 ## 구현 위치
 
 - 후처리 로직: `src/replace_v3d/joint_angles/postprocess.py`
-- 단일 시행 내보내기 (raw + `_anat` + `_ana0`): `scripts/run_joint_angles_pipeline.py`
-- 배치 통합 CSV 선택적 접미사 열:
-  - `--angles_anat` → `*_deg_anat` 추가
-  - `--angles_ana0` → `*_deg_ana0` 추가
-  - 스크립트: `scripts/run_batch_all_timeseries_csv.py`
+- 단일 시행 내보내기(표준=ana0만 저장): `scripts/run_joint_angles_pipeline.py`
+- 배치 통합 CSV(옵션): `scripts/run_batch_all_timeseries_csv.py`
diff --git a/README.md b/README.md
index f3f39cc..b290a83 100644
--- a/README.md
+++ b/README.md
@@ -33,7 +33,7 @@ conda run -n module python main.py \
 
 Outputs:
 - `<trial>_MOS_preStep.xlsx` (timeseries + summary + event mapping + COM validation)
-- `<trial>_JOINT_ANGLES_preStep.xlsx` / `.csv` (+ `_anat`, `_ana0` CSVs)
+- `<trial>_JOINT_ANGLES_preStep.csv` (ana0: sign-unified + quiet-standing baseline-subtracted)
 - `<trial>_ankle_torque.xlsx` (if forceplate/analog is present)
 
 Single-trial options:
@@ -92,5 +92,4 @@ conda run -n module python main.py \
```

Outputs:
-- `<trial>_JOINT_ANGLES_preStep.xlsx`
-- `<trial>_JOINT_ANGLES_preStep.csv` (for MD5 validation)
+- `<trial>_JOINT_ANGLES_preStep.csv` (ana0: sign-unified + quiet-standing baseline-subtracted)
diff --git a/main.py b/main.py
index f40e36e..4cd3d1c 100644
--- a/main.py
+++ b/main.py
@@ -51,9 +51,6 @@ def _collect_outputs(*, out_dir: Path, c3d_stem: str, steps: set[str]) -> list[P
out = []
if "angles" in steps:
out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep.csv")

* ```
     out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep.xlsx")
  ```
* ```
     out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep_anat.csv")
  ```
* ```
     out.append(out_dir / f"{c3d_stem}_JOINT_ANGLES_preStep_ana0.csv")
  ```

  if "mos" in steps:
  out.append(out_dir / f"{c3d_stem}_MOS_preStep.xlsx")
  if "torque" in steps:
  diff --git a/scripts/run_joint_angles_pipeline.py b/scripts/run_joint_angles_pipeline.py
  index 989b10e..8550577 100644
  --- a/scripts/run_joint_angles_pipeline.py
  +++ b/scripts/run_joint_angles_pipeline.py
  @@ -9,7 +9,6 @@ _bootstrap.ensure_src_on_path()
  import argparse

import numpy as np
-import pandas as pd
import polars as pl

from replace_v3d.c3d_reader import read_c3d_points
@@ -62,8 +61,7 @@ def main() -> None:
frames = np.arange(1, end_frame + 1)
times = (frames - 1) / rate

* # IMPORTANT: keep CSV schema identical to reference output for MD5 validation.
* df_pl = pl.DataFrame(

- df_raw = pl.DataFrame(
  {
  "Frame": frames,
  "Time_s": times,
  @@ -98,77 +96,20 @@ def main() -> None:
  }
  )

* # ---------------------------------------------------------------------
* # Post-processed outputs for analysis / presentation (raw output unchanged).
* #
* # `_anat`: sign-unified (LEFT Hip/Knee/Ankle Y/Z negated), no baseline.
* # `_ana0`: `_anat` + quiet-standing baseline subtraction (frames 1..11).
* # ---------------------------------------------------------------------
* df_pl_anat, meta_pp_anat = postprocess_joint_angles(
* ```
     df_pl,
  ```
* ```
     frame_col="Frame",
  ```
* ```
     unify_lr_sign=True,
  ```
* ```
     baseline_frames=None,
  ```
* )
* df_pl_ana0, meta_pp_ana0 = postprocess_joint_angles(
* ```
     df_pl,
  ```

- # Standard output = ana0 (analysis-friendly):
- # - unify L/R sign meaning (LEFT Hip/Knee/Ankle Y/Z negated)
- # - subtract quiet-standing baseline mean (frames 1..11, inclusive)
- df_ana0, _meta_pp = postprocess_joint_angles(
- ```
     df_raw,
     frame_col="Frame",
     unify_lr_sign=True,
     baseline_frames=(1, 11),
  ```

  )

  out_csv = out_dir / f"{c3d_path.stem}_JOINT_ANGLES_preStep.csv"

* out_xlsx = out_dir / f"{c3d_path.stem}_JOINT_ANGLES_preStep.xlsx"
*
* out_csv_anat = out_dir / f"{c3d_path.stem}_JOINT_ANGLES_preStep_anat.csv"
* out_csv_ana0 = out_dir / f"{c3d_path.stem}_JOINT_ANGLES_preStep_ana0.csv"
*
* # CSV first (stable for MD5)
* # NOTE: Use pandas formatting for stable exponent padding (e.g., e-07),
* # matching the provided reference CSV used for MD5 validation.
* df = df_pl.to_pandas()
* df.to_csv(out_csv, index=False)
*
* # Post-processed CSVs (extra; safe for analysis but not for MD5 validation)
* df_anat = df_pl_anat.to_pandas()
* df_ana0 = df_pl_ana0.to_pandas()
* df_anat.to_csv(out_csv_anat, index=False)
* df_ana0.to_csv(out_csv_ana0, index=False)
*
* # Excel (for inspection)
* meta = {
* ```
     "Trial": c3d_path.stem,
  ```
* ```
     "subject": events.subject,
  ```
* ```
     "velocity": events.velocity,
  ```
* ```
     "trial": events.trial,
  ```
* ```
     "rate_hz": rate,
  ```
* ```
     "platform_onset_local": events.platform_onset_local,
  ```
* ```
     "step_onset_local": events.step_onset_local,
  ```
* ```
     "analysis_end_local": end_frame,
  ```
* ```
     "angle_sequence": "Intrinsic XYZ (Visual3D-like: reference X, floating Y, non-reference Z)",
  ```
* ```
     "segment_axes": "X=+Right, Y=+Anterior, Z=+Up/Proximal",
  ```
* ```
     "knee_medial_marker": "LShin_3 / RShin_3",
  ```
* ```
     "ankle_medial_marker": "LFoot_3 / RFoot_3",
  ```
* ```
     "hip_center_method": "Harrington (via replace_v3d.com.compute_joint_centers)",
  ```
* }
*
* with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
* ```
     df.to_excel(writer, sheet_name="timeseries_preStep", index=False)
  ```
* ```
     df_anat.to_excel(writer, sheet_name="timeseries_preStep_anat", index=False)
  ```
* ```
     df_ana0.to_excel(writer, sheet_name="timeseries_preStep_ana0", index=False)
  ```
* ```
     pl.DataFrame([meta]).to_pandas().to_excel(writer, sheet_name="meta", index=False)
  ```
* ```
     pl.DataFrame([meta_pp_anat.__dict__]).to_pandas().to_excel(
  ```
* ```
         writer, sheet_name="meta_postprocess_anat", index=False
  ```
* ```
     )
  ```
* ```
     pl.DataFrame([meta_pp_ana0.__dict__]).to_pandas().to_excel(
  ```
* ```
         writer, sheet_name="meta_postprocess_ana0", index=False
  ```
* ```
     )
  ```
* ```
     pl.DataFrame([vars(events)]).to_pandas().to_excel(writer, sheet_name="events", index=False)
  ```

- df_ana0.write_csv(out_csv)

  print(f"[OK] Saved: {out_csv}")

* print(f"[OK] Saved: {out_csv_anat}")

* print(f"[OK] Saved: {out_csv_ana0}")

* print(f"[OK] Saved: {out_xlsx}")

if **name** == "**main**":


---

### 패치 적용 후 동작 요약
- `scripts/run_joint_angles_pipeline.py` 실행 시:
  - 출력은 **오직 1개**: `<trial>_JOINT_ANGLES_preStep.csv`
  - 내용은 **ana0(부호통일 + frame 1..11 baseline subtraction)** 결과
  - raw/anat/별도 ana0 파일/xlsx **전부 생성 안 함**
- 문서(README, JOINT_ANGLE_CONVENTIONS)와 `main.py`의 출력 수집도 그 기준으로 정리됨.

원하면, 다음 단계로 **batch CSV(run_batch_all_timeseries_csv.py)에서도 raw 각도 컬럼 자체를 없애고 ana0만 넣도록**(스키마 변경)까지 같이 “완전히 ana0-only”로 더 밀어붙이는 패치도 바로 만들어줄게.

raw 각도 컬럼 자체를 없애고 ana0만 넣도록**(스키마 변경)까지 같이 “완전히 ana0-only”로만 할 수 있도록 코드 수정해라. 