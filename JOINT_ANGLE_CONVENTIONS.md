# 관절각 출력 컨벤션 (표준 출력 = ana0)

본 저장소는 OptiTrack Motive의 Conventional 39 마커셋 + 사용자 내측 마커로
촬영된 C3D 데이터로부터 **Visual3D 방식의 3D 관절각**을 계산합니다.

**표준 출력은 `ana0` 하나만** 사용합니다.

- 저장 파일: `*_JOINT_ANGLES_preStep.csv`
- 의미: **ana0 = (좌우 부호 통일) + (Resolve_Discontinuity: Ankle Z) + (quiet standing 기저선 차감)**

---

## 1. 좌표계 및 분해 방식

### 글로벌 좌표계

| 축 | 방향 |
|----|------|
| X  | +Right (우측) |
| Y  | +Anterior (전방) |
| Z  | +Up (상방) |

### 오일러 분해

관절각은 **내재적(intrinsic) XYZ 오일러 각도**로 분해됩니다.

1. 근위(proximal) 세그먼트 좌표계를 기준(reference)으로 설정
2. 원위(distal) 세그먼트 좌표계와의 상대 회전 행렬 계산: `R_rel = R_ref^T @ R_dist`
3. `R_rel`에서 intrinsic XYZ 순서로 3개의 각도 추출:
   - `ay = arcsin(-R[2,0])`
   - `ax = arctan2(R[2,1], R[2,2])`
   - `az = arctan2(R[1,0], R[0,0])`
4. 짐벌 락(|cos(ay)| < 1e-8) 발생 시 fallback 적용

결과 단위는 **도(degree)** 입니다.

---

## 2. 계산 대상 관절 및 세그먼트 정의

### 2.1 관절중심(Joint Center) 계산

| 관절 | 방법 | 사용 마커 |
|------|------|----------|
| Hip  | Harrington(2007) 회귀식 | LASI, RASI, LPSI, RPSI (골반 너비·깊이로 추정) |
| Knee | 내외측 마커 중점 | LKNE/RKNE (외측) + LShin_3/RShin_3 (내측) |
| Ankle| 내외측 마커 중점 | LANK/RANK (외측) + LFoot_3/RFoot_3 (내측) |

### 2.2 세그먼트 좌표계

각 세그먼트는 마커 위치로부터 직교 정규화된 3×3 회전 행렬(오른손 좌표계)을 구성합니다.

| 세그먼트 | 주요 마커 | 축 정의 요약 |
|----------|----------|-------------|
| **Pelvis** | LASI, RASI, LPSI, RPSI | X=우측(RASI→LASI), Y=전방(PSI중점→원점), Z=상방 |
| **Thigh (L/R)** | Hip JC, Knee JC, 내외측 무릎 마커 | Z=근위(Hip→Knee), X=우측(내→외측) |
| **Shank (L/R)** | Knee JC, Ankle JC, 내외측 발목 마커 | Z=근위(Knee→Ankle), X=우측(내→외측) |
| **Foot (L/R)** | Ankle JC, TOE, HEE, 내외측 발 마커 | X=우측(내→외측), Y=전방(HEE→TOE), Z=상방 |
| **Thorax** | C7, T10, CLAV, STRN, LSHO, RSHO | Y=전방, Z=상방(C7→T10), X=우측(어깨 힌트) |
| **Head** | LFHD, RFHD, LBHD, RBHD | X=우측(RFHD→LFHD), Y=전방(전두→후두), Z=상방 |

### 2.3 관절 = 근위→원위 세그먼트 쌍

| 관절 | 근위(Reference) | 원위(Distal) | 좌우 구분 |
|------|----------------|-------------|----------|
| **Hip** | Pelvis | Thigh | L / R |
| **Knee** | Thigh | Shank | L / R |
| **Ankle** | Shank | Foot | L / R |
| **Trunk** | Pelvis | Thorax | 없음 (단일) |
| **Neck** | Thorax | Head | 없음 (단일) |

---

## 3. ana0 후처리 (3단계)

`ana0`는 raw 관절각에 아래 3단계를 **순서대로** 적용한 결과입니다.

### 3.1 좌우 부호 통일 (Step 1)

**대상:** Hip / Knee / Ankle의 **LEFT** 측 Y, Z 성분만

**적용 수식:**

```
*_L_Y_deg = - *_L_Y_deg
*_L_Z_deg = - *_L_Z_deg
```

**반전 대상 열 (총 6개):**

- `Hip_L_Y_deg`, `Hip_L_Z_deg`
- `Knee_L_Y_deg`, `Knee_L_Z_deg`
- `Ankle_L_Y_deg`, `Ankle_L_Z_deg`

**X축은 반전하지 않습니다.** Trunk, Neck은 좌우 구분이 없으므로 반전 대상이 아닙니다.

**목적:** 좌측과 우측의 Y/Z 부호가 동일한 해부학적 의미를 갖도록 통일합니다.

**반전 후 부호 해석 (RIGHT 기준으로 통일):**

| 축 | 양수(+) 의미 | 비고 |
|----|-------------|------|
| X  | (변경 없음 — 굴곡/신전 등 관절별 고유) | 좌우 동일 |
| Y  | 내전(adduction) | 좌우 동일 |
| Z  | 내회전(internal rotation) | 좌우 동일 |

### 3.2 Resolve_Discontinuity (Ankle Z only) (Step 2)

Euler/Cardan 관절각은 ±180° 경계에서 값이 갑자기 -180↔+180으로 “점프(wrap)”할 수 있습니다.
이 저장소에서는 Visual3D의 `Resolve_Discontinuity(signal, range, ...)`와 같은 아이디어로,
불연속 지점에서 **range(기본 360°)** 를 더/빼서 신호를 연속화합니다.

**적용 대상:** `Ankle_L_Z_deg`, `Ankle_R_Z_deg` (발목 Z축만)

**목적:** Ankle Z에서 wrap로 인해 baseline mean이 깨지거나, ana0 결과에 큰 스파이크가 생기는 현상을 방지합니다.

> 비고: 다른 축(X/Y)이나 다른 관절(Z 포함)은 값이 변하지 않도록 본 단계 적용 대상에서 제외합니다.

### 3.3 quiet standing 기저선 차감 (Step 3)

정적 기립(quiet standing) 구간의 평균값을 빼서 정적 오프셋을 제거합니다.

**기저선 구간:** Frame 1 ~ 11 (양 끝 포함, 총 11프레임)

**적용 수식:** 모든 `*_deg` 열에 대해:

```
angle[i] = angle[i] - mean(angle[Frame 1..11])
```

**목적:**
- 세그먼트 좌표계 정렬 오차로 인한 작은 정적 오프셋 제거
- 결과값은 quiet standing 대비 **Δ각도(변화량)** 로 해석

---

## 4. 출력 열 명명 규칙

### 패턴

```
{Joint}_{Side}_{Axis}_deg      — 좌우 구분 관절 (Hip, Knee, Ankle)
{Joint}_{Axis}_deg             — 단일 관절 (Trunk, Neck)
```

### 전체 출력 열 목록

```
Hip_L_X_deg,  Hip_L_Y_deg,  Hip_L_Z_deg
Hip_R_X_deg,  Hip_R_Y_deg,  Hip_R_Z_deg
Knee_L_X_deg, Knee_L_Y_deg, Knee_L_Z_deg
Knee_R_X_deg, Knee_R_Y_deg, Knee_R_Z_deg
Ankle_L_X_deg, Ankle_L_Y_deg, Ankle_L_Z_deg
Ankle_R_X_deg, Ankle_R_Y_deg, Ankle_R_Z_deg
Trunk_X_deg,  Trunk_Y_deg,  Trunk_Z_deg
Neck_X_deg,   Neck_Y_deg,   Neck_Z_deg
```

총 **24개** 각도 열 (좌우 3관절 × 2측 × 3축 = 18 + 단일 2관절 × 3축 = 6).

---

## 5. 출력 파일

### 단일 시행 (`run_joint_angles_pipeline.py`)

- 파일명: `{c3d_stem}_JOINT_ANGLES_preStep.csv`
- 프레임 열: `Frame` (1-indexed), `Time_s`
- 분석 구간: C3D 시작 ~ `step_onset_local - 1` (step onset 이전까지)
- 관절각 열: 위 24개 열 (ana0 적용 완료)

### 배치 통합 (`run_batch_all_timeseries_csv.py`)

- 모든 시행을 하나의 CSV로 통합
- 프레임 열: `MocapFrame` (100Hz 기준)
- 메타데이터 열: `subject`, `velocity`, `trial` 등
- 관절각은 동일한 ana0 값 사용 (24개 열)
- COM, MOS, Torque, GRF 등 다른 변수와 함께 출력

---

## 6. 구현 위치

| 역할 | 파일 |
|------|------|
| 세그먼트 좌표계 구성 | `src/replace_v3d/joint_angles/v3d_joint_angles.py` |
| 관절중심 계산 (Harrington 등) | `src/replace_v3d/com/joint_centers.py` |
| 오일러 분해 | `src/replace_v3d/joint_angles/v3d_joint_angles.py` |
| ana0 후처리 (부호 통일 + 기저선 차감) | `src/replace_v3d/joint_angles/postprocess.py` |
| 단일 시행 파이프라인 | `scripts/run_joint_angles_pipeline.py` |
| 배치 통합 CSV | `scripts/run_batch_all_timeseries_csv.py` |
| 시상면 각도 (KneeFlex, AnkleDorsi) | `src/replace_v3d/joint_angles/sagittal.py` |
