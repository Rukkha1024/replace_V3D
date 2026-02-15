ㅇㅋ. **1) 좌우 sign 통일 + 2) quiet standing baseline(0–10) 제거**를 **둘 다 한 번에** 적용하는 “후처리(ana0)”를 내가 정해서 **repo에 코드로 박아두는 형태**로 패치 만들어놨어.

핵심은 **원래 joint angle 계산(세그먼트→상대회전→intrinsic XYZ)** 은 손대지 않고,
**출력된 각도 time-series를 post-process** 해서 “좌/우 비교용”으로 하나 더 뽑게 만든 거야.

---

## 내가 “알아서” 정한 규칙 (코드에 그대로 남김)

### (1) 좌우 sign 통일

* **Hip/Knee/Ankle의 LEFT 쪽 Y, Z만 -1 곱해서 뒤집음**

  * X(flex/ext)은 그대로 둠
* 즉 아래 컬럼들만 flip:

  * `Hip_L_Y_deg`, `Hip_L_Z_deg`
  * `Knee_L_Y_deg`, `Knee_L_Z_deg`
  * `Ankle_L_Y_deg`, `Ankle_L_Z_deg`

### (2) quiet standing baseline 제거

* 네가 말한 **data[0..10]** 을 frame 번호로 해석하면,

  * script 출력 기준 `Frame = 1..11` (inclusive)
* 그래서 **baseline window를 (1, 11)** 로 두고,
* 모든 `*_deg` 컬럼(hip/knee/ankle/trunk/neck 포함)에 대해

  * baseline 평균을 빼서 **quiet standing에서 0° 기준**으로 맞춤.

---

## 네 데이터에서 “quiet standing(1..11프레임)” L/R 차이 실제로 얼마나 줄어드냐 (요약)

(내가 가진 `251112_KUO_perturb_60_001_angles_replaceV3D.xlsx` 기준으로 확인)

* **적용 전(raw)**: quiet standing인데도 L–R RMSE가

  * `Ankle Y` 약 **16°**
  * `Knee Z` 약 **11°**
  * 이런 식으로 크게 보였고

* **적용 후(ana0)**: quiet standing(1..11) 구간에서 L–R RMSE가 전부

  * 대략 **0.003° ~ 0.044°** 수준으로 떨어짐
    (즉 “정적 구간에서 좌우 차이가 없어야 한다”는 조건에 거의 맞게 됨)

이건 네가 문제 삼은 “nonstep인데도 좌우 차이 심함”의 **대표 원인(좌우 sign + static offset)** 을 딱 겨냥해서 제거한 결과야.

---

## 패치 파일(= 네 repo에 적용할 파일)

여기 있어.

[Download patch](sandbox:/mnt/data/replace_V3D_jointangles_ana0.patch)

---

## 적용 방법 (너 repo에서 그대로)

repo 루트에서:

```bash
git apply /path/to/replace_V3D_jointangles_ana0.patch
# 또는 충돌 나면
git apply --3way /path/to/replace_V3D_jointangles_ana0.patch
```

---

## 패치가 바꾸는 것들 (코드에 남겨둔 위치)

### 1) 새 파일 추가

* `src/replace_v3d/joint_angles/postprocess.py`

  * `postprocess_joint_angles()` 함수
  * flip 대상 컬럼 규칙은 `_default_flip_columns()`에 박아둠

### 2) 단일 trial joint angle script는 “ana0 출력”을 자동으로 추가 생성

* `scripts/run_joint_angles_pipeline.py`

  * 기존 raw:

    * `*_JOINT_ANGLES_preStep.csv`  ✅ 그대로 유지 (MD5 검증용)
  * 추가 ana0:

    * `*_JOINT_ANGLES_preStep_ana0.csv` ✅ 새로 생성 (분석용)
  * 엑셀도 시트 추가:

    * `timeseries_preStep_ana0`
    * `meta_postprocess`

### 3) batch unified CSV는 옵션으로 ana0 컬럼 추가 가능

* `scripts/run_batch_all_timeseries_csv.py`

  * 새 옵션:

    * `--angles_ana0`
  * 켜면 기존 컬럼은 그대로 두고,

    * `Hip_L_X_deg_ana0` 같은 식으로 **suffix `_ana0` 붙은 컬럼들**이 추가됨
      (스키마 깨지는 거 싫어서 “덮어쓰기”는 안 했음)

---

## 너가 나중에 바꾸고 싶을 때(딱 두 군데만 보면 됨)

* **flip 규칙 바꾸고 싶다** → `src/replace_v3d/joint_angles/postprocess.py`의 `_default_flip_columns()`
* **baseline 프레임 바꾸고 싶다** →

  * 단일 trial: `scripts/run_joint_angles_pipeline.py`에서 `baseline_frames=(1, 11)`
  * 배치: `scripts/run_batch_all_timeseries_csv.py`에서 `baseline_frames=(1, 11)`

---

원하면 다음 단계로는,

* ana0가 아니라 **“V3D에서 흔히 말하는 ‘anatomical convention’(예: abd/add, int/ext의 +방향 통일)”** 기준으로 축/부호를 좀 더 엄밀히 맞춘 버전(= DOF별 정의서까지 포함)을 추가로 만들어줄 수도 있어.
  근데 지금 네가 겪는 “quiet standing에서 좌우 차이 너무 큼” 문제는 이 패치(ana0)로 바로 잡힐 가능성이 높아.


==========
V3D 쪽에서 말하는 “anatomical convention(= L/R에서 abd/add, int/ext의 +방향을 **같은 의미**로 맞추는 것)”을 **따로 export**하는 버전을 추가해뒀어. 핵심은 Visual3D에서 기본 RHR(오른손법칙) 때문에 **Left/Right의 Y, Z 부호 의미가 서로 반대**로 나오는 걸(특히 hip/knee) “표현용(presentation)”으로 **LEFT의 Y/Z만 negate**해서 통일하는 방식이야. ([HAS-Motion][1])

---

## 이번 “anatomical convention”이 정확히 뭐를 하냐 (엄밀 버전)

Visual3D 기본(기본 SCS: X=Right, Y=Anterior, Z=Up + intrinsic XYZ)에서는 XYZ가 아래 DOF 의미로 매핑되고, RHR 때문에 **Y/Z는 L/R에서 부호 의미가 달라짐**이 문서에 명확히 나와있어. ([HAS-Motion][1])

* X: flex/ext
* Y: abd/add
* Z: axial rotation (int/ext rot) ([HAS-Motion][1])

Visual3D의 기본 “positive” 예시는 (Hip/Knee) 이렇게 정리되어 있고(Left는 +Y=abduction, +Z=external / Right는 +Y=adduction, +Z=internal), 그래서 L/R 비교할 때 Y/Z가 뒤집혀 보이는 게 정상 동작이야. ([HAS-Motion][1])

그리고 Visual3D 튜토리얼에서도 “**표현할 때 left knee angle의 y,z를 negate하는 게 흔하다**”고 직접 말해. ([yumpu.com][2])

### 그래서 내가 추가한 `_anat` 정의(= anatomical presentation)

**Hip/Knee/Ankle에 대해 LEFT만:**

* `*_L_Y_deg *= -1`
* `*_L_Z_deg *= -1`

이렇게 하면 결과적으로:

* **Y(+) = adduction (L/R 동일 의미)**
* **Z(+) = internal rotation (L/R 동일 의미)**
  가 되도록 **오른쪽 다리의 부호 의미를 기준으로 좌측을 통일**해주는 형태야. ([HAS-Motion][1])

> 참고: X는 원칙적으로 L/R에서 sign 의미가 같다고 Visual3D가 설명하고 있어서(그래서 “abd/add, int/ext”만 통일 이슈) `_anat`에서는 X는 건드리지 않았어. ([HAS-Motion][1])

---

## 패치에서 실제로 추가된 것들

### 1) 단일 trial joint angle 스크립트 출력 1개 추가

`scripts/run_joint_angles_pipeline.py`에 아래 파일을 추가 저장하도록 했어:

* 기존(raw): `*_JOINT_ANGLES_preStep.csv`
* 기존(ana0): `*_JOINT_ANGLES_preStep_ana0.csv`  (L Y/Z 통일 + **quiet standing baseline 제거**)
* ✅ 신규(anat): `*_JOINT_ANGLES_preStep_anat.csv`  (**L Y/Z 통일만**, baseline 없음)

엑셀(`*_JOINT_ANGLES_preStep.xlsx`)에도 시트가 추가돼:

* `timeseries_preStep_anat`
* `meta_postprocess_anat`

### 2) batch 통합 타임시리즈 CSV에도 `_anat` 컬럼 옵션 추가

`scripts/run_batch_all_timeseries_csv.py`에 옵션 추가:

* `--angles_anat`
  → 기존 raw 컬럼은 그대로 두고, 추가 컬럼을 `*_deg_anat` suffix로 더 붙여서 내보냄
  (스키마 깨짐 방지)

예: `Hip_L_Y_deg_anat`, `Knee_L_Z_deg_anat`, …

---

## DOF 정의서(문서)도 같이 추가함

repo root에 아래 파일이 생겨:

* `JOINT_ANGLE_CONVENTIONS.md`

내용은:

* 세그먼트 축 정의(이 repo의 frame 구성 방식)
* joint별 proximal/distal 정의(hip/knee/ankle/trunk/neck)
* raw vs anat vs ana0의 차이
* Visual3D 근거 링크(공식 문서/튜토리얼) 포함

---

## 적용용 패치 파일

이 패치는 **이전에 준 `ana0` 패치가 적용된 상태**를 기준으로 만들어졌어(= `postprocess_joint_angles`가 이미 존재해야 함).

1. (아직 ana0 안 했으면 먼저)
   [Download ana0 patch](sandbox:/mnt/data/replace_V3D_jointangles_ana0.patch)

2. 그 다음 anat 추가 패치
   [Download anatomical convention patch](sandbox:/mnt/data/replace_V3D_jointangles_anat.patch)

적용은 보통 이렇게:

```bash
git apply replace_V3D_jointangles_ana0.patch
git apply replace_V3D_jointangles_anat.patch
```

---

## 사용법 요약

### 단일 trial

기존대로 실행하면 이제 `_anat.csv`도 같이 나와.

```bash
conda run -n module python scripts/run_joint_angles_pipeline.py \
  --c3d data/...c3d \
  --event_xlsm data/perturb_inform.xlsm \
  --subject "김우연" \
  --out_dir output
```

### batch 통합 타임시리즈

`_anat` 컬럼을 추가로 뽑고 싶으면:

```bash
conda run -n module python scripts/run_batch_all_timeseries_csv.py \
  --c3d_dir data/all_data \
  --event_xlsm data/perturb_inform.xlsm \
  --out_csv output/all_trials_timeseries.csv \
  --overwrite \
  --angles_anat
```

`--angles_ana0`랑 같이 써도 돼(둘 다 export).

