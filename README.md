# replace_V3D

모션 캡처 및 포스 플레이트 데이터로부터 신체중심(COM), 외삽 신체중심(xCOM), 지지면(BoS), 안정성 여유(MoS), 3차원 관절 각도, 지면반력(GRF), 압력중심(COP), 발목 토크를 산출하는 순수 Python 생체역학 분석 파이프라인이다. 본 파이프라인은 Visual3D를 대체하기 위해 개발되었으며, 후방 지지면 이동 섭동(posterior support-surface translation perturbation) 과제 중 수집된 OptiTrack Conventional 39-marker set 데이터를 처리한다.

## 1. 참가자 및 실험 프로토콜

24명의 건강한 젊은 성인이 연구에 참여하였다. 각 참가자는 이동 가능한 플랫폼 위에 서서 60~135 cm/s 범위의 속도로 예기치 않은 후방 이동 섭동을 받았으며, 속도는 혼합 순서(mixed order)로 제시되었다. 자세 반응은 회복 전략에 따라 두 범주로 분류되었다: 보상적 발디딤(compensatory step)이 나타난 **step** 시행(53회)과 발디딤 없이 균형을 회복한 **non-step** 시행(72회). 다음의 포함 기준을 적용한 결과 총 125회 시행이 분석에 포함되었다:

- 혼합 속도 조건만 포함 (`mixed = 1`)
- 젊은 성인 참가자만 포함 (`age_group = young`)
- Step 시행은 동측 발디딤(ipsilateral stepping)만 포함 (오른손잡이는 오른발, 왼손잡이는 왼발)
- Non-step 시행은 우세측에 관계없이 모두 포함

시행 메타데이터(섭동 속도, 반응 분류, 참가자 인구통계학적 정보)는 `perturb_inform.xlsm`에 기록되었다.

## 2. 데이터 수집

전신 운동학 데이터는 OptiTrack 모션 캡처 시스템과 **Conventional 39-marker set**를 사용하여 100 Hz 샘플링 속도로 수집되었다. 아날로그 신호(힘 및 모멘트)는 내장 포스 플레이트를 통해 1000 Hz로 동시 기록되었다. 모든 데이터는 C3D 형식으로 저장되었다.

각 C3D 파일은 처리 전에 100 Hz 모캡 시간 기준으로 `[platform_onset - 100, platform_offset + 100]` 프레임 구간으로 사전 트리밍되었다. 이벤트 타이밍(platform onset, platform offset, step onset)은 수동으로 표기되어 Excel 워크북(`perturb_inform.xlsm`)에 저장되었다.

## 3. 데이터 처리 파이프라인

### 3.1 C3D 읽기 및 마커 추출

마커 궤적은 Python 기반 리더(`src/replace_v3d/io/c3d_reader.py`)를 사용하여 C3D 파일에서 추출되었다. 원본 OptiTrack 라벨(예: `251112_KUO_LASI`)과 단순화된 라벨(예: `LASI`) 모두 자동 라벨 정규화를 통해 지원되었다.

### 3.2 관절 중심 추정

관절 중심은 마커 위치로부터 다음과 같이 추정되었다:

| 관절 | 방법 |
|------|------|
| 고관절(Hip) | Harrington et al. (2007) 회귀식 — 골반 너비 및 깊이 기반 |
| 슬관절(Knee) | 외측(LKNE/RKNE)과 내측(LShin_3/RShin_3) 마커의 중간점 |
| 족관절(Ankle) | 외측(LANK/RANK)과 내측(LFoot_3/RFoot_3) 마커의 중간점 |
| 주관절(Elbow) | 외측(LELB/RELB)과 내측(LUArm_3/RUArm_3) 마커의 중간점 |
| 완관절(Wrist) | LWRA/RWRA와 LWRB/RWRB의 중간점 |

고관절 중심은 Harrington (2007) 회귀 방정식을 사용하여 추정되었으며, 골반 너비(PW, RASI와 LASI 사이의 거리)와 골반 깊이(PD, 골반 원점과 후상장골극 중간점 사이의 거리)가 예측 변수로 사용되었다.

### 3.3 전신 신체중심

전신 COM은 De Leva (1996)의 질량 분율 및 COM 배치 비율을 사용하여 14개 분절 COM의 가중합으로 산출되었다:

`COM = sum(m_i * COM_i)`

여기서 `m_i`는 질량 분율, `COM_i`는 각 분절의 근위-원위 관절 중심 사이에서 분절별 비율로 보간된 점으로 추정된 신체중심 위치를 나타낸다.

| 분절 | 질량 분율 | COM 비율 (근위→원위) |
|------|----------|---------------------|
| 머리(Head) | 0.0694 | (C7 + 머리 중심 경유) |
| 몸통(Trunk) | 0.4346 | 0.797 (골반 원점 → 흉부 기준점) |
| 상완(Upper arm, ×2) | 0.0271 | 0.436 |
| 전완(Forearm, ×2) | 0.0162 | 0.430 |
| 손(Hand, ×2) | 0.0061 | 0.506 |
| 대퇴(Thigh, ×2) | 0.1416 | 0.433 |
| 하퇴(Shank, ×2) | 0.0433 | 0.433 |
| 발(Foot, ×2) | 0.0137 | 0.500 |

흉부 기준점은 C7과 흉골 마커의 가중 조합으로 정의되었다: `thorax_ref = 0.56 * C7 + 0.44 * STRN`. 몸통 COM은 골반 원점에서 이 흉부 기준점까지 거리의 79.7% 지점에 배치되었다. 머리 COM은 C7에서 머리 중심(LFHD, RFHD, LBHD, RBHD의 평균)까지 거리의 100% 지점에 위치하였다.

산출된 COM 궤적은 차단 주파수 6 Hz의 4차 영위상(zero-phase) Butterworth 저역통과 필터로 필터링되었다.

### 3.4 COM 속도 및 외삽 신체중심

COM 속도(vCOM)는 필터링된 COM 위치의 중앙차분 미분으로 산출되었다:

`vCOM = d(COM) / dt`

외삽 신체중심(xCOM)은 Hof et al. (2005)의 방법에 따라 산출되었다:

`xCOM = COM + vCOM / sqrt(g / l)`

여기서 `g = 9.81 m/s^2`이며, `l`은 메타데이터 파일에서 얻은 참가자의 다리 길이를 나타낸다.

### 3.5 지지면

지지면(BoS)은 각 프레임에서 8개의 발 랜드마크 마커를 지면(XY 평면)에 투영한 볼록 껍질(convex hull)로 정의되었다:

`LHEE, LTOE, LANK, LFoot_3, RHEE, RTOE, RANK, RFoot_3`

인체측정학적 발 확장(foot expansion)은 적용하지 않았으며, BoS 경계는 마커 위치만으로 결정되었다. 각 프레임에서 축 정렬 경계(minX, maxX, minY, maxY)와 다각형 면적이 볼록 껍질로부터 추출되었다.

### 3.6 안정성 여유

세 가지 MoS 정의가 산출되었다:

| 변수 | 정의 |
|------|------|
| `MOS_minDist_signed` | xCOM에서 BoS 다각형 경계까지의 부호 있는 최소 거리. 양수는 xCOM이 BoS 내부, 음수는 외부를 나타냄 |
| `MOS_AP_v3d` | 가장 가까운 전후방(AP) BoS 경계까지의 거리: `min(xCOM_X - minX, maxX - xCOM_X)` |
| `MOS_ML_v3d` | 가장 가까운 내외측(ML) BoS 경계까지의 거리: `min(xCOM_Y - minY, maxY - xCOM_Y)` |

`MOS_AP_v3d` 및 `MOS_ML_v3d` 정의는 Visual3D 문서에 기술된 최근접 경계(closest-bound) 접근법을 따랐다. 전체 최근접 경계 변수(`MOS_v3d = min(MOS_AP_v3d, MOS_ML_v3d)`)도 함께 산출되었다.

### 3.7 관절 각도

3차원 관절 각도는 5개 관절(고관절, 슬관절, 족관절, 몸통, 목)에 대해, 해당되는 경우 양측으로 산출되었으며, 내재적(intrinsic) XYZ Cardan 분해를 사용하였다. 분절 좌표계는 해부학적 마커 클러스터로부터 오른손 좌표계 규칙에 따라 구성되었다.

모든 관절 각도 시계열은 platform onset 시점 값으로부터의 변화량(onset-zeroed)으로 표현되었다.

추가로, 관절 각속도(joint angular velocity)는 관절각을 프레임 미분하여 근사하지 않고, **기준 분절(reference segment) 대비 이동 분절(moving segment)의 상대 각속도 벡터**로 계산되었다. 동일한 각속도 벡터를 두 가지 분절 좌표계에서 읽을 수 있으므로, 본 파이프라인은 다음 두 계열을 모두 출력한다:

- `*_ref_*_deg_s`: 상대 각속도 벡터를 reference segment 좌표계에서 분해한 성분
- `*_mov_*_deg_s`: 상대 각속도 벡터를 moving segment 좌표계에서 분해한 성분

각속도 컬럼은 `Hip_L_ref_X_deg_s`와 같이 `ref/mov` 구분이 포함되며, onset-zeroing은 적용하지 않았다.

### 3.8 운동역학 변수

지면반력(GRF)과 지면반력 모멘트(GRM)는 C3D 포스 플레이트 메타데이터 및 아날로그 채널에서 추출되었다. inverse dynamics에 사용할 포스 플레이트는 `config.yaml > forceplate.analysis.use_for_inverse_dynamics`에서 명시적으로 지정되며, `[fp1, fp3]`처럼 비연속 부분집합도 허용된다. 힘 및 모멘트 데이터는 Stage01 생체역학 좌표계(Z축 상방, 피험자 중심 수평축)로 변환되었다. 이 파이프라인은 단일 레거시 컬럼(`GRF_*`, `GRM_*`, `COP_*`)을 더 이상 출력하지 않고, 선택된 plate별 `FP{n}_*` 컬럼으로만 raw forceplate 시계열을 내보낸다.

**압력중심(COP)**은 변환된 힘 및 모멘트 데이터로부터 산출되었다:

`COP_X = -M_Y / F_Z` , `COP_Y = M_X / F_Z`

본 파이프라인은 **발목 토크(AnkleTorque*) 계열을 출력하지 않는다.**

추가로, 관절 모멘트(joint moment)는 **internal/proximal joint moment**로 정의되었으며, inverse dynamics(분절별 힘/모멘트 평형 방정식)를 통해 산출되었다. `use_for_inverse_dynamics`에 plate가 1개만 지정된 경우에는 `single_plate_strict` 모드로 동작하며, 선택된 plate의 raw forceplate 시계열(`FP{n}_*`)만 내보내고, `*_ref_*_Nm` 계열(Hip/Knee/Ankle/Trunk/Neck)은 전부 NaN으로 남긴다(계산 자체를 수행하지 않음). 선택된 plate가 2개 이상이면 `multi_plate_v3d` 모드로 동작하며, 지정된 plate 집합 안에서만 COP 기반 좌/우 할당(nearest-COP, V3D-style contact block assignment)을 수행한 뒤, 발→하퇴→대퇴 방향의 재귀 계산으로 하지 관절 모멘트를 구한다. COP를 계산할 수 없거나, 한 plate 위에 양측 족관절이 동시에 올라가 분리가 불가능한 프레임은 해당 프레임의 하지 관절 모멘트를 NaN으로 남긴다. 출력 모멘트 성분은 `*_ref_*_Nm` 형태로, 각 관절에서 지정된 reference segment 좌표계로 분해하여 제공한다(예: `Knee_L_ref_Y_Nm`). onset-zeroing은 적용하지 않았다.

관성 보정(inertial correction) 절차가 적용되어, 사전 산출된 정적 기립 템플릿(quiet-standing template)을 이용하여 측정된 힘과 모멘트에서 플랫폼 가속도 아티팩트를 제거하였다.

선택된 plate의 raw forceplate 시계열은 plate별 컬럼(`FP{n}_*`)으로 출력된다. GRF/GRM(`FP{n}_GRF_*`, `FP{n}_GRM_*`)에는 platform onset 기준 baseline subtraction이 적용되며, COP(`FP{n}_COP_*`)는 절대 좌표로 유지된다.

### 3.9 신호처리 요약

| 파라미터 | 값 |
|---------|-----|
| 모캡 샘플링 속도 | 100 Hz |
| 아날로그 샘플링 속도 | 1000 Hz |
| COM 저역통과 필터 | 6 Hz, 4차 Butterworth, 영위상 |
| 미분 방법 | 중앙차분(Central difference) |
| 관절 각도 분해 | 내재적 XYZ Cardan |
| Onset zeroing | Platform onset 프레임에서의 값을 차감 (관절 각도 등 일부 시계열). Forceplate GRF/GRM은 extraction 단계에서 baseline subtraction이 적용됨 |

## 4. 좌표계 및 부호 규약

모든 변수는 다음의 실험실 좌표계로 표현되었다:

| 축 | 방향 | 양수 (+) | 음수 (-) |
|----|------|---------|---------|
| X | 전후방(AP) | 전방(Anterior) | 후방(Posterior) |
| Y | 내외측(ML) | 우측(Right) | 좌측(Left) |
| Z | 수직 | 상방(Upward) | 하방(Downward) |

관절 각도 부호 규약:

| 관절 | X축 (+/-) | Y축 (+/-) | Z축 (+/-) |
|------|----------|----------|----------|
| 고관절(Hip) | 굴곡 / 신전 | 내전 / 외전 | 내회전 / 외회전 |
| 슬관절(Knee) | 굴곡 / 신전 | 내전 / 외전 | 내회전 / 외회전 |
| 족관절(Ankle) | 배굴 / 저굴 | 내전 / 외전 | 내회전 / 외회전 |
| 몸통(Trunk) | 굴곡 / 신전 | 내전 / 외전 | 내회전 / 외회전 |
| 목(Neck) | 굴곡 / 신전 | 내전 / 외전 | 내회전 / 외회전 |

## 5. 시간축 및 정규화

두 가지 시간 표현이 유지되었다:

- **MocapFrame** (100 Hz): 트리밍된 C3D 파일 내의 순차적 프레임 인덱스
- **original_DeviceFrame** (1000 Hz): 절대 장치 프레임 번호, 출처 추적용으로 보존

출력 CSV 파일은 원시 시간축(`MocapFrame`, `time_from_platform_onset_s`)을 유지한다. 구간별 시간 정규화(piecewise time normalization)는 시각화(grid plot)에만 적용되었으며, onset 전·후 구간이 각각 고정된 프레임 수로 선형 워핑되었다. 이 정규화는 출력 데이터에 영향을 미치지 않았다.

## 6. 통계 분석

### 6.1 Step vs. Non-Step 비교 (발디딤 전 구간)

발디딤 전 생체역학적 반응이 step과 non-step 전략 간에 차이가 있는지 검정하기 위하여, 세 범주에 걸쳐 34개 종속변수에 대해 선형 혼합 모형(LMM)을 적합하였다:

**균형 및 안정성 (17개 종속변수):** COM 범위 및 경로 길이(AP, ML), vCOM 최댓값(AP, ML), COP 범위·경로 길이·최대 속도(AP, ML), MoS 최솟값(`MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`), platform onset 및 step onset 시점의 xCOM-BoS 거리.

**관절 각도 (10개 종속변수):** 고관절, 슬관절, 족관절(지지측), 몸통, 목의 시상면 운동 범위 및 최댓값.

**힘 및 토크 (7개 종속변수):** GRF 최댓값 및 범위(AP, ML, 수직), 시상면 발목 토크 최댓값.

분석 구간은 `[platform_onset, step_onset]`으로 정의되었다. Step 시행의 경우 실제 step onset 프레임이 사용되었고, non-step 시행의 경우 해당 피험자-속도 그룹의 평균 step onset이 대입되었다.

각 모형은 `DV ~ step_TF + (1 | subject)` 형태로, REML 추정을 사용하였다. 다중 비교 보정은 Benjamini-Hochberg FDR(BH-FDR) 방법으로 유의수준 alpha = 0.05에서 수행되었다.

### 6.2 초기 자세 전략 분석 (Onset 프레임)

섭동 onset 시점의 신체 배치(body configuration)가 이후의 균형 회복 전략을 예측하는지 평가하기 위하여, platform onset 단일 프레임에서 추출된 19개 종속변수에 대해 LMM을 적합하였다:

**균형 (8개 종속변수):** COM 위치(AP, ML), vCOM(AP, ML), MoS(`MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`), 정규화된 xCOM-BoS 거리.

**Onset 시점 관절 각도 (5개 종속변수):** 고관절, 슬관절, 족관절(지지측), 몸통, 목의 절대값(non-onset-zeroed) 시상면 각도.

**Onset 시점 힘 변수 (6개 종속변수):** 절대 COP 위치(AP, ML), GRF(AP, ML, 수직), 체질량 정규화 시상면 발목 토크.

각 모형은 동일한 형태(`DV ~ step_TF + (1 | subject)`)로, BH-FDR 보정(alpha = 0.05)을 적용하였다.

## 7. 파이프라인 실행

모든 스크립트는 `module` conda 환경에서 실행되었다:

```bash
conda run -n module python <script>
```

### 7.1 일괄 내보내기

전체 파이프라인(마커 추출, 관절 중심 추정, COM/xCOM/BoS/MoS 산출, 관절 각도, forceplate raw(FP{n}_*), 관절 모멘트(다중 plate 모드))이 단일 일괄 프로세스로 실행되었다:

```bash
conda run -n module python main.py --overwrite
```

`config.yaml`의 `forceplate.analysis.use_for_inverse_dynamics`가 inverse dynamics용 plate 선택의 단일 기준이다. 예를 들어 `[fp3]`은 strict single-plate mode, `[fp1, fp3]`은 multi-plate mode로 해석된다.

| 인자 | 기본값 | 설명 |
|------|-------|------|
| `--c3d_dir` | `data/all_data` | 트리밍된 C3D 파일 디렉토리 |
| `--event_xlsm` | `data/perturb_inform.xlsm` | 이벤트 메타데이터 Excel 워크북 |
| `--out_csv` | `output/all_trials_timeseries.csv` | 출력 CSV 경로 |
| `--overwrite` | (플래그) | 기존 출력 CSV 덮어쓰기 |
| `--skip_unmatched` | (플래그) | 피험자/이벤트 매핑 미해결 시행 건너뛰기 |
| `--pre_frames` | `100` | local-absolute 변환을 위한 프레임 버퍼 |
| `--encoding` | `utf-8-sig` | CSV 인코딩 (한국어 Excel 호환 BOM) |
| `--on_error` | `continue` | 오류 처리 방식: `continue` 또는 `abort` |
| `--md5_reference_dir` | (없음) | MD5 체크섬 검증용 참조 디렉토리 |

### 7.2 격자 시각화

내보낸 CSV로부터 시계열 격자 플롯(피험자 × 속도 × 변수 범주)이 생성되었다:

```bash
# 샘플 미리보기 (3명 피험자, 2개 속도)
conda run -n module python scripts/plot_grid_timeseries.py --sample

# 전체 피험자-속도 그룹
conda run -n module python scripts/plot_grid_timeseries.py --group_by subject_velocity

# 피험자별 오버레이
conda run -n module python scripts/plot_grid_timeseries.py --group_by subject

# 필터링된 하위 집합
conda run -n module python scripts/plot_grid_timeseries.py \
  --only_subjects subject1,subject2 \
  --only_velocities 60,70
```

| 인자 | 기본값 | 설명 |
|------|-------|------|
| `--group_by` | `subject_velocity` | 그룹화 모드: `subject_velocity`, `subject`, 또는 `total_mean` |
| `--sample` | (플래그) | 미리보기만 생성 (3명 피험자, 2개 속도) |
| `--dpi` | `300` | 그림 해상도 |
| `--segment_frames` | `100` | 이벤트 기준 구간 크기 (프레임) |
| `--x_piecewise` | 활성화 | 표시를 위한 구간별 시간 정규화 |
| `--y_zero_onset` | 활성화 | 시행별 platform onset 시점 값 차감 |
| `--separate_step_nonstep` | 비활성화 | Step vs. non-step 별도 그림 생성 |

이벤트 기준선이 중첩 표시되었다: platform onset(빨간색), platform offset(초록색), step onset(파란색 점선). `FP{n}_*` raw forceplate 계열은 이미 extraction 단계에서 GRF/GRM이 onset 기준으로 보정되므로, plot 단계의 `--y_zero_onset` 적용 대상에서 제외된다.

## 8. 출력 설명

### 8.1 시계열 CSV

주요 출력 파일(`output/all_trials_timeseries.csv`)은 시행별·모캡 프레임별 한 행으로 구성된 장형(long format) 구조이다. 컬럼 군은 다음과 같이 요약된다:

| 군 | 컬럼 | 단위 | Onset-Zeroed |
|----|------|------|-------------|
| 식별자 | `subject`, `velocity`, `trial` | — | — |
| 시간 | `MocapFrame`, `time_from_platform_onset_s` | 프레임, 초 | — |
| 이벤트 | `platform_onset_local`, `platform_offset_local`, `step_onset_local` | 프레임 | — |
| COM | `COM_X`, `COM_Y`, `COM_Z` | m | 아니오 |
| vCOM | `vCOM_X`, `vCOM_Y`, `vCOM_Z` | m/s | 아니오 |
| xCOM | `xCOM_X`, `xCOM_Y`, `xCOM_Z` | m | 아니오 |
| BoS | `BOS_area`, `BOS_minX`, `BOS_maxX`, `BOS_minY`, `BOS_maxY` | m, m² | 아니오 |
| MoS | `MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`, `MOS_v3d` | m | 아니오 |
| 관절 각도 | `Hip_L_X_deg`, ..., `Neck_Z_deg` | deg | 예 |
| 관절 각속도 | `Hip_L_ref_X_deg_s`, ..., `Neck_mov_Z_deg_s` | deg/s | 아니오 |
| inverse dynamics 메타 | `inverse_dynamics_mode`, `inverse_dynamics_forceplates` | — | — |
| Forceplate raw | `FP{n}_GRF_*_N`, `FP{n}_GRM_*_Nm_at_FPorigin`, `FP{n}_COP_*_m`, `FP{n}_ContactValid` | N, Nm, m | GRF/GRM: 예 / COP·ContactValid: 아니오 |
| 관절 모멘트 | `Hip_L_ref_X_Nm`, ..., `Neck_ref_Z_Nm` | Nm | 아니오 |

### 8.2 시각화

격자 플롯은 `output/figures/grid_timeseries/`에 저장되며, 변수 범주는 `config.yaml > plot_grid_timeseries.categories`에 정의된다. 기본 구성은 MoS/BoS, COM 계열, 하지 관절 각도, 상체 관절 각도, forceplate raw(FP{n}_*), 관절 모멘트(hip/knee/ankle)이다.

## 9. 참고문헌

- De Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223-1230.
- Harrington, M. E., Zavatsky, A. B., Lawson, S. E. M., Yuan, Z., & Theologis, T. N. (2007). Prediction of the hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance imaging. *Journal of Biomechanics*, 40(3), 595-602.
- Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). The condition for dynamic stability. *Journal of Biomechanics*, 38(1), 1-8.
- Van Wouwe, T., Afschrift, M., De Groote, F., & Vanwanseele, B. (2021). Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance. *Journal of Neurophysiology*, 125(5), 1983-1998.
