# Initial Posture Strategy LMM (Baseline Mean Before Platform Onset)

## Research Question

**"Van Wouwe et al. (2021) 관점에서, platform onset 직전 300 ms baseline posture 평균이 step/nonstep 전략 차이를 설명한다면 baseline 변수에서 step/nonstep 차이가 광범위하게 유의한가?"**

이번 보고서는 `platform_onset_local` 단일 프레임 대신 onset 전 `[-0.30, 0.00] s` **구간 평균(range mean)** 을 사용해 초기 자세를 baseline posture로 요약한다.
`95% CI`는 이 baseline range mean 비교에서만 제시하며, single-frame 결과는 `report.md`에서 별도로 다룬다.

## Prior Studies

### Van Wouwe et al. (2021) — Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance

- **Methodology**: 예측 시뮬레이션 + 실험 데이터 결합. 초기 자세(COM 위치)와 task-level goal(노력-안정성 우선순위)의 상호작용으로 전략 variability를 설명.
- **Experimental design**: 10명의 젊은 성인, 예측 불가능한 backward support-surface translation, stepping/nonstepping 반응 기록.
- **Key results**:
  - 최대 trunk lean 변동성: within-subject mean range 약 `28.3°`, across-subject mean range 약 `39.9°`
  - initial COM position과 maximal trunk lean 관계는 subject-specific (`R^2 = 0.29–0.82`)
  - `xCOM/BOS_onset`, `xCOM/BOS_300ms`를 안정성 지표로 사용
- **Conclusions**: 초기 자세는 intra-subject variability에, task-level goal은 inter-subject 차이에 기여하며, 두 요인 상호작용이 전략 차이를 설명한다.

## Methodological Adaptation

| Prior Method | Current Implementation | Deviation Rationale |
|---|---|---|
| 단일 시점 또는 특정 초기 posture 지표로 전략 variability를 해석 | `time_from_platform_onset_s ∈ [-0.30, 0.00]` 구간 평균을 사용 | 단일 프레임 노이즈를 줄이고 onset 직전 자세를 baseline posture로 요약하기 위해 |
| 초기 posture와 안정성 지표를 함께 해석 | COM/MOS/xCOM-BOS + joint angle + force/torque를 같은 LMM 틀에서 비교 | 동일 조건 step/nonstep 차이를 여러 biomechanical domain에서 동시에 비교하기 위해 |
| 실험/시뮬레이션 기반 posture 정의 | `output/all_trials_timeseries.csv`의 export 값을 frame-wise 평균 | 현재 저장소의 재현 가능한 분석 입력을 유지하고 새 파일 export를 만들지 않기 위해 |

This analysis adopts the prior study's focus on initial posture and stability metrics, but changes the operational definition from a single onset frame to a 300 ms pre-onset range mean. In this project, the baseline report is also the only report that presents Wald `95% CI`.

## Data Summary

- Trials used: **126** (`step=53`, `nonstep=73`), subjects=24
- Baseline window: **`[-0.30, 0.00] s`**
- Baseline frames per trial: min=`31`, max=`31`, mean=`31.00`
- Excluded trials without baseline frames: **0**
- Input:
  - `output/all_trials_timeseries.csv`
  - `data/perturb_inform.xlsm`
- 분석 변수:
  - baseline 후보 총 **74개**
  - 검정 가능(testable) **74개**
  - 검정 불가(untestable) **0개**

## Analysis Methodology

- **Analysis window**: `time_from_platform_onset_s ∈ [-0.30, 0.00]`
- **Statistical model**: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- **Outlier rule**: 각 변수별로 `step`/`nonstep` 그룹 내부에서 `1.5×IQR` 밖 trial 제거
- **Confidence interval**: `step_TFstep` 계수의 Wald `95% CI`를 baseline range mean 비교에 한해 함께 보고
- **Multiple comparison correction**: BH-FDR (74개 baseline 변수 전체 1회)
- **Significance reporting**: `Sig` only (`***`, `**`, `*`, `n.s.`), `alpha=0.05`
- **Displayed result policy**: Results 표에는 **FDR 유의 변수만** 표시

### Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|---|---|---|---|
| X | exported X-axis positive direction | exported X-axis opposite direction | `COM_X_baseline`, `GRF_X_baseline`, `Hip_stance_X_baseline` |
| Y | exported Y-axis positive direction | exported Y-axis opposite direction | `COM_Y_baseline`, `GRF_Y_baseline`, `Hip_stance_Y_baseline` |
| Z | exported Z-axis positive direction | exported Z-axis opposite direction | `GRF_Z_baseline`, `Trunk_Z_baseline` |

### Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|---|---|---|---|
| `MOS_minDist_signed_baseline` | BOS 안쪽 여유가 큰 방향 | BOS 경계 밖 또는 여유 감소 방향 | exported `MOS_minDist_signed` baseline 평균 |
| `MOS_AP_v3d_baseline` | AP margin이 더 큰 상태 | AP margin이 더 작은 상태 | exported `MOS_AP_v3d` baseline 평균 |
| `MOS_ML_v3d_baseline` | ML margin이 더 큰 상태 | ML margin이 더 작은 상태 | exported `MOS_ML_v3d` baseline 평균 |
| `xCOM_BOS_norm_baseline` | `BOS_minX`에서 `BOS_maxX` 쪽으로 더 큰 상대 위치 | `BOS_minX` 쪽으로 더 작은 상대 위치 | `(xCOM_X - BOS_minX) / (BOS_maxX - BOS_minX)` frame 평균 |

### Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|---|---|---|
| Joint angle (`*_baseline`) | exported joint-angle axis의 양/음 방향 | frame-wise stance 선택 후 baseline 평균, 추가 C3D 재계산 없음 |
| Joint angular velocity (`*_ref_*_baseline`, `*_mov_*_baseline`) | exported angular-velocity axis의 양/음 방향 | Hip/Knee/Ankle은 frame-wise stance 선택 후 baseline 평균 |
| Segment moment (`*_ref_*_baseline`) | exported internal moment axis의 양/음 방향 | Hip/Knee/Ankle은 frame-wise stance 선택 후 baseline 평균 |
| COP / COM / GRF | exported CSV 축 방향의 양/음 값 | onset-aligned timeseries CSV 값을 직접 평균 |
| `AnkleTorqueMid_Y_perkg_baseline` | exported internal ankle torque Y축 양/음 방향 | `AnkleTorqueMid_int_Y_Nm_per_kg` baseline 평균 |

### Stance-Leg Selection Rule

- `step_r` trial: left leg angle을 stance로 사용
- `step_l` trial: right leg angle을 stance로 사용
- `nonstep` trial: 해당 subject의 step trial 분포(`major_step_side`)로 stance를 선택
- `tie` (`step_r_count == step_l_count`): left/right 평균값 사용
- Subject summary: `step_r_major=9`, `step_l_major=10`, `tie=5`
- Tie subjects: `강비은, 김서하, 김유민, 안지연, 유재원`

### Analyzed Variables (Full Set, n=74)

| Variable | Family | Testability at baseline | Result status |
|---|---|---|---|
| `COM_X_baseline` | Balance | testable | n.s. |
| `COM_Y_baseline` | Balance | testable | n.s. |
| `vCOM_X_baseline` | Balance | testable | ** |
| `vCOM_Y_baseline` | Balance | testable | n.s. |
| `MOS_minDist_signed_baseline` | Balance | testable | *** |
| `MOS_AP_v3d_baseline` | Balance | testable | *** |
| `MOS_ML_v3d_baseline` | Balance | testable | n.s. |
| `xCOM_BOS_norm_baseline` | Balance | testable | *** |
| `Hip_stance_X_baseline` | Joint_baseline | testable | n.s. |
| `Hip_stance_Y_baseline` | Joint_baseline | testable | n.s. |
| `Hip_stance_Z_baseline` | Joint_baseline | testable | n.s. |
| `Knee_stance_X_baseline` | Joint_baseline | testable | ** |
| `Knee_stance_Y_baseline` | Joint_baseline | testable | n.s. |
| `Knee_stance_Z_baseline` | Joint_baseline | testable | n.s. |
| `Ankle_stance_X_baseline` | Joint_baseline | testable | n.s. |
| `Ankle_stance_Y_baseline` | Joint_baseline | testable | n.s. |
| `Ankle_stance_Z_baseline` | Joint_baseline | testable | n.s. |
| `Trunk_X_baseline` | Joint_baseline | testable | n.s. |
| `Trunk_Y_baseline` | Joint_baseline | testable | n.s. |
| `Trunk_Z_baseline` | Joint_baseline | testable | n.s. |
| `Neck_X_baseline` | Joint_baseline | testable | n.s. |
| `Neck_Y_baseline` | Joint_baseline | testable | n.s. |
| `Neck_Z_baseline` | Joint_baseline | testable | n.s. |
| `Hip_stance_ref_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Hip_stance_ref_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Hip_stance_ref_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Hip_stance_mov_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Hip_stance_mov_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Hip_stance_mov_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Hip_stance_ref_X_Nm_baseline` | Moment_baseline | testable | ** |
| `Hip_stance_ref_Y_Nm_baseline` | Moment_baseline | testable | *** |
| `Hip_stance_ref_Z_Nm_baseline` | Moment_baseline | testable | *** |
| `Knee_stance_ref_X_deg_s_baseline` | Velocity_baseline | testable | * |
| `Knee_stance_ref_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Knee_stance_ref_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Knee_stance_mov_X_deg_s_baseline` | Velocity_baseline | testable | * |
| `Knee_stance_mov_Y_deg_s_baseline` | Velocity_baseline | testable | ** |
| `Knee_stance_mov_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Knee_stance_ref_X_Nm_baseline` | Moment_baseline | testable | *** |
| `Knee_stance_ref_Y_Nm_baseline` | Moment_baseline | testable | *** |
| `Knee_stance_ref_Z_Nm_baseline` | Moment_baseline | testable | *** |
| `Ankle_stance_ref_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Ankle_stance_ref_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Ankle_stance_ref_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Ankle_stance_mov_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Ankle_stance_mov_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Ankle_stance_mov_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Ankle_stance_ref_X_Nm_baseline` | Moment_baseline | testable | *** |
| `Ankle_stance_ref_Y_Nm_baseline` | Moment_baseline | testable | *** |
| `Ankle_stance_ref_Z_Nm_baseline` | Moment_baseline | testable | *** |
| `Trunk_ref_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Trunk_ref_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Trunk_ref_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Trunk_mov_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Trunk_mov_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Trunk_mov_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Trunk_ref_X_Nm_baseline` | Moment_baseline | testable | n.s. |
| `Trunk_ref_Y_Nm_baseline` | Moment_baseline | testable | n.s. |
| `Trunk_ref_Z_Nm_baseline` | Moment_baseline | testable | n.s. |
| `Neck_ref_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Neck_ref_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Neck_ref_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Neck_mov_X_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Neck_mov_Y_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Neck_mov_Z_deg_s_baseline` | Velocity_baseline | testable | n.s. |
| `Neck_ref_X_Nm_baseline` | Moment_baseline | testable | n.s. |
| `Neck_ref_Y_Nm_baseline` | Moment_baseline | testable | n.s. |
| `Neck_ref_Z_Nm_baseline` | Moment_baseline | testable | n.s. |
| `COP_X_baseline` | Force_baseline | testable | n.s. |
| `COP_Y_baseline` | Force_baseline | testable | n.s. |
| `GRF_X_baseline` | Force_baseline | testable | n.s. |
| `GRF_Y_baseline` | Force_baseline | testable | n.s. |
| `GRF_Z_baseline` | Force_baseline | testable | n.s. |
| `AnkleTorqueMid_Y_perkg_baseline` | Force_baseline | testable | n.s. |

## Results

### Hypothesis Verdict (strict)

- **Rule**: testable baseline 변수 전부가 FDR 유의여야 PASS
- **Observed**: testable significant ratio = `17/74`, untestable=`0`
- **Verdict**: **FAIL**

### Significant Variables Only (BH-FDR < 0.05)

| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | 95% CI | Sig |
|---|---|---:|---:|---:|---:|---|
| `Hip_stance_ref_Y_Nm_baseline` | Moment_baseline | -0.54±1.24 | 21.06±20.96 | -22.28 | `[-26.61, -17.95]` | *** |
| `Ankle_stance_ref_Y_Nm_baseline` | Moment_baseline | 0.02±0.07 | 22.75±28.02 | -25.06 | `[-30.13, -19.98]` | *** |
| `Knee_stance_ref_Y_Nm_baseline` | Moment_baseline | 0.24±0.28 | 16.89±17.09 | -18.55 | `[-22.30, -14.80]` | *** |
| `Ankle_stance_ref_X_Nm_baseline` | Moment_baseline | 0.32±0.08 | 22.62±20.38 | -22.77 | `[-27.59, -17.95]` | *** |
| `Hip_stance_ref_Z_Nm_baseline` | Moment_baseline | -0.14±0.35 | 5.94±6.21 | -6.63 | `[-8.04, -5.21]` | *** |
| `Knee_stance_ref_X_Nm_baseline` | Moment_baseline | -0.09±0.35 | 10.06±11.27 | -11.97 | `[-14.59, -9.35]` | *** |
| `MOS_minDist_signed_baseline` | Balance | 0.07±0.01 | 0.06±0.01 | 0.01 | `[0.01, 0.01]` | *** |
| `Knee_stance_ref_Z_Nm_baseline` | Moment_baseline | 0.00±0.03 | 1.32±1.68 | -1.44 | `[-1.84, -1.05]` | *** |
| `xCOM_BOS_norm_baseline` | Balance | 0.64±0.07 | 0.70±0.07 | -0.05 | `[-0.06, -0.03]` | *** |
| `MOS_AP_v3d_baseline` | Balance | 0.07±0.01 | 0.06±0.02 | 0.01 | `[0.01, 0.01]` | *** |
| `Ankle_stance_ref_Z_Nm_baseline` | Moment_baseline | 0.01±0.01 | 1.01±1.58 | -1.01 | `[-1.46, -0.57]` | *** |
| `Knee_stance_mov_Y_deg_s_baseline` | Velocity_baseline | -0.07±0.25 | 0.11±0.29 | -0.19 | `[-0.29, -0.09]` | ** |
| `Hip_stance_ref_X_Nm_baseline` | Moment_baseline | -0.29±2.03 | 1.87±5.14 | -1.72 | `[-2.63, -0.81]` | ** |
| `Knee_stance_X_baseline` | Joint_baseline | -0.26±0.18 | -0.21±0.16 | -0.09 | `[-0.14, -0.04]` | ** |
| `vCOM_X_baseline` | Balance | 0.00±0.00 | 0.00±0.00 | -0.00 | `[-0.00, -0.00]` | ** |
| `Knee_stance_mov_X_deg_s_baseline` | Velocity_baseline | 1.28±1.06 | 1.02±0.76 | 0.39 | `[0.14, 0.64]` | * |
| `Knee_stance_ref_X_deg_s_baseline` | Velocity_baseline | 1.28±1.06 | 1.06±0.72 | 0.36 | `[0.11, 0.61]` | * |

### Outlier Exclusion Summary

| Variable | Step raw | Step outliers | Step kept | Nonstep raw | Nonstep outliers | Nonstep kept |
|---|---:|---:|---:|---:|---:|---:|
| `AnkleTorqueMid_Y_perkg_baseline` | 53 | 1 | 52 | 73 | 4 | 69 |
| `Ankle_stance_X_baseline` | 53 | 4 | 49 | 72 | 5 | 67 |
| `Ankle_stance_Y_baseline` | 53 | 13 | 40 | 72 | 8 | 64 |
| `Ankle_stance_Z_baseline` | 53 | 6 | 47 | 72 | 6 | 66 |
| `Ankle_stance_mov_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 1 | 71 |
| `Ankle_stance_mov_Y_deg_s_baseline` | 53 | 11 | 42 | 72 | 6 | 66 |
| `Ankle_stance_mov_Z_deg_s_baseline` | 53 | 7 | 46 | 72 | 6 | 66 |
| `Ankle_stance_ref_X_Nm_baseline` | 53 | 11 | 42 | 72 | 0 | 72 |
| `Ankle_stance_ref_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 4 | 68 |
| `Ankle_stance_ref_Y_Nm_baseline` | 53 | 11 | 42 | 72 | 1 | 71 |
| `Ankle_stance_ref_Y_deg_s_baseline` | 53 | 9 | 44 | 72 | 4 | 68 |
| `Ankle_stance_ref_Z_Nm_baseline` | 53 | 11 | 42 | 72 | 13 | 59 |
| `Ankle_stance_ref_Z_deg_s_baseline` | 53 | 7 | 46 | 72 | 7 | 65 |
| `COM_X_baseline` | 53 | 0 | 53 | 73 | 0 | 73 |
| `COM_Y_baseline` | 53 | 4 | 49 | 73 | 0 | 73 |
| `COP_X_baseline` | 53 | 0 | 53 | 73 | 1 | 72 |
| `COP_Y_baseline` | 53 | 4 | 49 | 73 | 0 | 73 |
| `GRF_X_baseline` | 53 | 9 | 44 | 73 | 12 | 61 |
| `GRF_Y_baseline` | 53 | 3 | 50 | 73 | 3 | 70 |
| `GRF_Z_baseline` | 53 | 4 | 49 | 73 | 7 | 66 |
| `Hip_stance_X_baseline` | 53 | 6 | 47 | 72 | 7 | 65 |
| `Hip_stance_Y_baseline` | 53 | 7 | 46 | 72 | 6 | 66 |
| `Hip_stance_Z_baseline` | 53 | 7 | 46 | 72 | 9 | 63 |
| `Hip_stance_mov_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 3 | 69 |
| `Hip_stance_mov_Y_deg_s_baseline` | 53 | 5 | 48 | 72 | 7 | 65 |
| `Hip_stance_mov_Z_deg_s_baseline` | 53 | 5 | 48 | 72 | 5 | 67 |
| `Hip_stance_ref_X_Nm_baseline` | 53 | 4 | 49 | 72 | 5 | 67 |
| `Hip_stance_ref_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 5 | 67 |
| `Hip_stance_ref_Y_Nm_baseline` | 53 | 12 | 41 | 72 | 3 | 69 |
| `Hip_stance_ref_Y_deg_s_baseline` | 53 | 9 | 44 | 72 | 7 | 65 |
| `Hip_stance_ref_Z_Nm_baseline` | 53 | 11 | 42 | 72 | 4 | 68 |
| `Hip_stance_ref_Z_deg_s_baseline` | 53 | 3 | 50 | 72 | 5 | 67 |
| `Knee_stance_X_baseline` | 53 | 6 | 47 | 72 | 8 | 64 |
| `Knee_stance_Y_baseline` | 53 | 6 | 47 | 72 | 3 | 69 |
| `Knee_stance_Z_baseline` | 53 | 11 | 42 | 72 | 9 | 63 |
| `Knee_stance_mov_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 8 | 64 |
| `Knee_stance_mov_Y_deg_s_baseline` | 53 | 7 | 46 | 72 | 5 | 67 |
| `Knee_stance_mov_Z_deg_s_baseline` | 53 | 8 | 45 | 72 | 10 | 62 |
| `Knee_stance_ref_X_Nm_baseline` | 53 | 14 | 39 | 72 | 3 | 69 |
| `Knee_stance_ref_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 9 | 63 |
| `Knee_stance_ref_Y_Nm_baseline` | 53 | 12 | 41 | 72 | 5 | 67 |
| `Knee_stance_ref_Y_deg_s_baseline` | 53 | 7 | 46 | 72 | 6 | 66 |
| `Knee_stance_ref_Z_Nm_baseline` | 53 | 11 | 42 | 72 | 6 | 66 |
| `Knee_stance_ref_Z_deg_s_baseline` | 53 | 9 | 44 | 72 | 11 | 61 |
| `MOS_AP_v3d_baseline` | 53 | 0 | 53 | 73 | 0 | 73 |
| `MOS_ML_v3d_baseline` | 53 | 0 | 53 | 73 | 0 | 73 |
| `MOS_minDist_signed_baseline` | 53 | 0 | 53 | 73 | 1 | 72 |
| `Neck_X_baseline` | 53 | 5 | 48 | 72 | 5 | 67 |
| `Neck_Y_baseline` | 53 | 2 | 51 | 72 | 4 | 68 |
| `Neck_Z_baseline` | 53 | 5 | 48 | 72 | 5 | 67 |
| `Neck_mov_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 9 | 63 |
| `Neck_mov_Y_deg_s_baseline` | 53 | 2 | 51 | 72 | 7 | 65 |
| `Neck_mov_Z_deg_s_baseline` | 53 | 5 | 48 | 72 | 5 | 67 |
| `Neck_ref_X_Nm_baseline` | 53 | 4 | 49 | 72 | 3 | 69 |
| `Neck_ref_X_deg_s_baseline` | 53 | 5 | 48 | 72 | 9 | 63 |
| `Neck_ref_Y_Nm_baseline` | 53 | 0 | 53 | 72 | 0 | 72 |
| `Neck_ref_Y_deg_s_baseline` | 53 | 3 | 50 | 72 | 5 | 67 |
| `Neck_ref_Z_Nm_baseline` | 53 | 1 | 52 | 72 | 0 | 72 |
| `Neck_ref_Z_deg_s_baseline` | 53 | 6 | 47 | 72 | 6 | 66 |
| `Trunk_X_baseline` | 53 | 7 | 46 | 72 | 2 | 70 |
| `Trunk_Y_baseline` | 53 | 5 | 48 | 72 | 3 | 69 |
| `Trunk_Z_baseline` | 53 | 2 | 51 | 72 | 4 | 68 |
| `Trunk_mov_X_deg_s_baseline` | 53 | 4 | 49 | 72 | 2 | 70 |
| `Trunk_mov_Y_deg_s_baseline` | 53 | 3 | 50 | 72 | 6 | 66 |
| `Trunk_mov_Z_deg_s_baseline` | 53 | 4 | 49 | 72 | 5 | 67 |
| `Trunk_ref_X_Nm_baseline` | 53 | 0 | 53 | 72 | 0 | 72 |
| `Trunk_ref_X_deg_s_baseline` | 53 | 2 | 51 | 72 | 2 | 70 |
| `Trunk_ref_Y_Nm_baseline` | 53 | 1 | 52 | 72 | 4 | 68 |
| `Trunk_ref_Y_deg_s_baseline` | 53 | 4 | 49 | 72 | 5 | 67 |
| `Trunk_ref_Z_Nm_baseline` | 53 | 0 | 53 | 72 | 2 | 70 |
| `Trunk_ref_Z_deg_s_baseline` | 53 | 3 | 50 | 72 | 5 | 67 |
| `vCOM_X_baseline` | 53 | 3 | 50 | 73 | 6 | 67 |
| `vCOM_Y_baseline` | 53 | 1 | 52 | 73 | 2 | 71 |
| `xCOM_BOS_norm_baseline` | 53 | 0 | 53 | 73 | 0 | 73 |

## Comparison with Prior Studies

| Comparison Item | Prior Study Result | Current Result | Verdict |
|---|---|---|---|
| Initial posture operationalization | onset/posture-related 초기 상태가 전략 variability를 설명 | onset 직전 300 ms baseline 평균으로 posture를 정의 | Partially consistent |
| Broad posture separation by kinematic variables | initial posture effect가 존재하지만 subject/task interaction이 중요 | baseline joint-angle 유의 변수 `1/15` (`Knee_stance_X_baseline`) | Consistent |
| Stability metric relevance | `xCOM/BOS`와 onset stability interpretation 제시 | baseline 유의 변수 `vCOM_X_baseline, MOS_minDist_signed_baseline, MOS_AP_v3d_baseline, xCOM_BOS_norm_baseline, Knee_stance_X_baseline, Hip_stance_ref_X_Nm_baseline, Hip_stance_ref_Y_Nm_baseline, Hip_stance_ref_Z_Nm_baseline, Knee_stance_ref_X_deg_s_baseline, Knee_stance_mov_X_deg_s_baseline, Knee_stance_mov_Y_deg_s_baseline, Knee_stance_ref_X_Nm_baseline, Knee_stance_ref_Y_Nm_baseline, Knee_stance_ref_Z_Nm_baseline, Ankle_stance_ref_X_Nm_baseline, Ankle_stance_ref_Y_Nm_baseline, Ankle_stance_ref_Z_Nm_baseline` | Consistent |

## Interpretation & Conclusion

1. onset 전 `[-0.30, 0.00] s` 구간 평균으로 초기 자세를 요약해도 strict 기준 가설은 **FAIL**였다.
2. baseline joint-angle 변수는 총 `15`개 중 `1`개가 FDR 유의였고, 유의 변수는 `Knee_stance_X_baseline`였다.
3. 반면 balance 계열에서는 `MOS_minDist_signed_baseline`, `MOS_AP_v3d_baseline`, `xCOM_BOS_norm_baseline`, `vCOM_X_baseline`가 유의해, baseline posture 차이가 전혀 없다고 보기는 어렵다.
4. 따라서 본 데이터에서는 onset 직전 baseline posture 차이가 step/nonstep 전략 차이를 완전히 설명한다고 단정하기 어렵고, prior study가 제시한 posture-goal interaction의 일부만 현재 집단 비교에서 포착된 것으로 해석하는 것이 안전하다.

## Limitations

1. 본 분석은 `output/all_trials_timeseries.csv`의 export 값을 평균한 결과로, C3D 재계산 기반 onset absolute 변수와 직접 동일하지 않다.
2. task-level goal 파라미터를 직접 모델링하지 않았다.
3. baseline 평균은 초기 자세를 안정적으로 요약하지만, onset 직전 순간적인 준비 동작은 희석할 수 있다.
4. single-frame 결과와 baseline mean 결과는 질문 자체가 다르므로, 두 보고서의 유의 변수 수를 직접 같은 의미로 해석하면 안 된다.

## Reproduction

```bash
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_baseline_lmm.py --dry-run
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_baseline_lmm.py
```

- Output: 콘솔 통계 결과 + 자동 갱신 `report_baseline.md`, `결과) 주제2-Segement Angle_baseline.md`

## Figures

| File | Description |
|---|---|
| (none) | This baseline analysis does not generate figures. |

---
Auto-generated by analyze_initial_posture_strategy_baseline_lmm.py.
