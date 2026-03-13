# Initial Posture Strategy LMM (Single-Frame Comparison)

## Research Question

**"Van Wouwe et al. (2021) 관점에서, single-frame posture 지표가 step/nonstep 전략 차이를 설명한다면 platform onset과 step onset 중 어떤 시점에서 분화가 더 뚜렷한가?"**

이번 보고서는 `platform_onset_local`과 `step_onset_target_local`의 **단일 프레임 LMM** 결과를 다룬다. baseline range mean 결과는 별도 문서인 `report_baseline.md`에서 다루며, 본 문서에는 `95% CI`를 포함하지 않는다.

## Prior Studies

### Van Wouwe et al. (2021) — Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance

- **Methodology**: 예측 시뮬레이션 + 실험 데이터 결합. 초기 자세(COM 위치)와 task-level goal(노력-안정성 우선순위)의 상호작용으로 전략 variability를 설명.
- **Experimental design**: 10명의 젊은 성인, 예측 불가능한 backward support-surface translation, stepping/nonstepping 반응 기록.
- **Key results**:
  - 최대 trunk lean 변동성: within-subject mean range 약 `28.3°`, across-subject mean range 약 `39.9°`
  - initial COM position과 maximal trunk lean 관계는 subject-specific (`R^2 = 0.29–0.82`)
  - `xCOM/BOS_onset`, `xCOM/BOS_300ms`를 안정성 지표로 사용
- **Conclusions**: 초기 자세는 intra-subject variability에, task-level goal은 inter-subject 차이에 기여하며, 두 요인 상호작용이 전략 차이를 설명.

## Data Summary

- Trials: **125** (`step=53`, `nonstep=72`), subjects=24
- Input:
  - `output/all_trials_timeseries.csv`
  - `data/perturb_inform.xlsm`
  - `data/all_data/*.c3d`
  - `src/replace_v3d/torque/assets/fp_inertial_templates.npz`
- 분석 변수:
  - onset 후보 총 **29개**
  - 검정 가능(testable) **29개**
  - 검정 불가(untestable) **0개**
- Force inertial QC mode: **non-strict**

## Analysis Methodology

이 보고서의 질문은 onset 전 평균 posture가 아니라 **특정 단일 프레임에서 step/nonstep 차이가 얼마나 드러나는지**를 보는 것이다. 따라서 baseline 평균 분석과는 목적이 다르며, `report_baseline.md`와 직접 같은 질문으로 읽으면 안 된다.

- **Analysis point**: `platform_onset_local` 단일 프레임
- **Statistical model**: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- **Outlier rule**: 각 변수별 `step/nonstep` 그룹 내부에서 `1.5×IQR` 밖 trial 제거
- **Confidence interval policy**: 본 단일 프레임 보고서에서는 `Estimate`와 `Sig`만 보고하고, `95% CI`는 baseline range mean 보고서에서만 제시
- **Multiple comparison correction**: BH-FDR (29개 onset 변수 전체 1회)
- **Significance reporting**: `Sig` only (`***`, `**`, `*`, `n.s.`), `alpha=0.05`
- **Displayed result policy**: Results 표에는 **FDR 유의 변수만** 표시

### Coordinate Definition (Joint Angle)

- Joint angle는 `compute_v3d_joint_angles_3d` 기준의 **intrinsic XYZ Euler sequence**를 사용한다.
- Segment 좌표계는 전역 기준으로 `X=+Right`, `Y=+Anterior`, `Z=+Up/+Proximal`로 구성된다.
- 따라서 `*_X/*_Y/*_Z`는 각각 해당 축 회전 성분이며, 단순히 sagittal/frontal/transverse와 1:1로 고정 해석하면 안 된다.

### Stance-Leg Selection Rule

- `step_r` trial: left leg angle을 stance로 사용
- `step_l` trial: right leg angle을 stance로 사용
- `nonstep` trial: 해당 subject의 step trial 분포(`major_step_side`)로 stance를 선택
- `tie` (`step_r_count == step_l_count`): left/right 평균값 사용
- Subject summary: `step_r_major=9`, `step_l_major=10`, `tie=5`
- Tie subjects: `강비은, 김서하, 김유민, 안지연, 유재원`

### Analyzed Variables (Full Set, n=29)

| Variable | Family | Testability at onset | Result status |
|---|---|---|---|
| `COM_X` | Balance | testable | n.s. |
| `COM_Y` | Balance | testable | n.s. |
| `vCOM_X` | Balance | testable | ** |
| `vCOM_Y` | Balance | testable | * |
| `MOS_minDist_signed` | Balance | testable | *** |
| `MOS_AP_v3d` | Balance | testable | *** |
| `MOS_ML_v3d` | Balance | testable | n.s. |
| `xCOM_BOS_norm_onset` | Balance | testable | *** |
| `Hip_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Hip_stance_Y_abs_onset` | Joint_absolute | testable | * |
| `Hip_stance_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `Knee_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Knee_stance_Y_abs_onset` | Joint_absolute | testable | n.s. |
| `Knee_stance_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `Ankle_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Ankle_stance_Y_abs_onset` | Joint_absolute | testable | n.s. |
| `Ankle_stance_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `Trunk_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Trunk_Y_abs_onset` | Joint_absolute | testable | * |
| `Trunk_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `Neck_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Neck_Y_abs_onset` | Joint_absolute | testable | n.s. |
| `Neck_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `COP_X_abs_onset` | Force_absolute | testable | n.s. |
| `COP_Y_abs_onset` | Force_absolute | testable | n.s. |
| `GRF_X_abs_onset` | Force_absolute | testable | * |
| `GRF_Y_abs_onset` | Force_absolute | testable | ** |
| `GRF_Z_abs_onset` | Force_absolute | testable | n.s. |
| `AnkleTorqueMid_Y_perkg_abs_onset` | Force_absolute | testable | *** |

## Results

### Hypothesis Verdict (strict)

- **Rule**: testable onset 변수 전부가 FDR 유의여야 PASS
- **Observed**: testable significant ratio = `10/29`, untestable=`0`
- **Verdict**: **FAIL**

### Significant Variables Only (BH-FDR < 0.05)

| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---|---:|---:|---:|---|
| `MOS_minDist_signed` | Balance | 0.07±0.01 | 0.06±0.01 | 0.01 | *** |
| `MOS_AP_v3d` | Balance | 0.08±0.01 | 0.06±0.02 | 0.01 | *** |
| `xCOM_BOS_norm_onset` | Balance | 0.63±0.07 | 0.70±0.07 | -0.05 | *** |
| `AnkleTorqueMid_Y_perkg_abs_onset` | Force_absolute | -2.42±0.11 | -2.55±0.13 | 0.10 | *** |
| `GRF_Y_abs_onset` | Force_absolute | 4.22±2.64 | 2.53±0.66 | 1.27 | ** |
| `vCOM_X` | Balance | 0.01±0.01 | 0.02±0.01 | -0.00 | ** |
| `GRF_X_abs_onset` | Force_absolute | -1.27±3.33 | -2.79±0.90 | 1.20 | * |
| `Hip_stance_Y_abs_onset` | Joint_absolute | 1.54±4.18 | -0.09±3.50 | 1.55 | * |
| `Trunk_Y_abs_onset` | Joint_absolute | -1.01±3.52 | 0.74±2.96 | -0.69 | * |
| `vCOM_Y` | Balance | -0.00±0.00 | 0.00±0.00 | -0.00 | * |

### Outlier Exclusion Summary (Platform Onset)

| Variable | Step raw | Step outliers | Step kept | Nonstep raw | Nonstep outliers | Nonstep kept |
|---|---:|---:|---:|---:|---:|---:|
| `AnkleTorqueMid_Y_perkg_abs_onset` | 53 | 5 | 48 | 72 | 3 | 69 |
| `Ankle_stance_X_abs_onset` | 53 | 0 | 53 | 72 | 0 | 72 |
| `Ankle_stance_Y_abs_onset` | 53 | 1 | 52 | 72 | 7 | 65 |
| `Ankle_stance_Z_abs_onset` | 53 | 0 | 53 | 72 | 0 | 72 |
| `COM_X` | 53 | 0 | 53 | 72 | 0 | 72 |
| `COM_Y` | 53 | 5 | 48 | 72 | 0 | 72 |
| `COP_X_abs_onset` | 53 | 0 | 53 | 72 | 1 | 71 |
| `COP_Y_abs_onset` | 53 | 3 | 50 | 72 | 0 | 72 |
| `GRF_X_abs_onset` | 53 | 3 | 50 | 72 | 22 | 50 |
| `GRF_Y_abs_onset` | 53 | 1 | 52 | 72 | 14 | 58 |
| `GRF_Z_abs_onset` | 53 | 4 | 49 | 72 | 3 | 69 |
| `Hip_stance_X_abs_onset` | 53 | 2 | 51 | 72 | 8 | 64 |
| `Hip_stance_Y_abs_onset` | 53 | 0 | 53 | 72 | 0 | 72 |
| `Hip_stance_Z_abs_onset` | 53 | 0 | 53 | 72 | 3 | 69 |
| `Knee_stance_X_abs_onset` | 53 | 0 | 53 | 72 | 0 | 72 |
| `Knee_stance_Y_abs_onset` | 53 | 1 | 52 | 72 | 7 | 65 |
| `Knee_stance_Z_abs_onset` | 53 | 0 | 53 | 72 | 0 | 72 |
| `MOS_AP_v3d` | 53 | 0 | 53 | 72 | 1 | 71 |
| `MOS_ML_v3d` | 53 | 0 | 53 | 72 | 0 | 72 |
| `MOS_minDist_signed` | 53 | 0 | 53 | 72 | 3 | 69 |
| `Neck_X_abs_onset` | 53 | 1 | 52 | 72 | 1 | 71 |
| `Neck_Y_abs_onset` | 53 | 1 | 52 | 72 | 0 | 72 |
| `Neck_Z_abs_onset` | 53 | 1 | 52 | 72 | 0 | 72 |
| `Trunk_X_abs_onset` | 53 | 1 | 52 | 72 | 3 | 69 |
| `Trunk_Y_abs_onset` | 53 | 2 | 51 | 72 | 3 | 69 |
| `Trunk_Z_abs_onset` | 53 | 1 | 52 | 72 | 5 | 67 |
| `vCOM_X` | 53 | 5 | 48 | 72 | 1 | 71 |
| `vCOM_Y` | 53 | 1 | 52 | 72 | 5 | 67 |
| `xCOM_BOS_norm_onset` | 53 | 0 | 53 | 72 | 0 | 72 |

## Step-Onset Single-Frame Analysis

- **Analysis point**: `step_onset_target_local` 단일 프레임
- **Target rule**:
  - step trial: 해당 trial의 `step_onset_local` 사용
  - nonstep trial: 동일 subject의 step trial `step_onset_local` 평균값을 대입한 후 frame으로 반올림
- **Valid trials**: `119/126` (step=`52`, nonstep=`67`)
- **Excluded trials**: `7` (step_onset 결측 step=`1`, step 참조 부재 nonstep=`6`, frame 불일치=`0`)
- **nonstep step_onset 참조 부재 subject**: `권유영, 김종철, 방주원`
- **Observed**: testable significant ratio = `45/74`

### Step-Onset Variables (Full Set, n=74)

| Variable | Family | Testability at onset | Result status |
|---|---|---|---|
| `COM_X_step_onset` | Balance_step_onset | testable | *** |
| `COM_Y_step_onset` | Balance_step_onset | testable | n.s. |
| `vCOM_X_step_onset` | Balance_step_onset | testable | *** |
| `vCOM_Y_step_onset` | Balance_step_onset | testable | *** |
| `MOS_minDist_signed_step_onset` | Balance_step_onset | testable | *** |
| `MOS_AP_v3d_step_onset` | Balance_step_onset | testable | *** |
| `MOS_ML_v3d_step_onset` | Balance_step_onset | testable | n.s. |
| `xCOM_BOS_norm_step_onset` | Balance_step_onset | testable | *** |
| `Hip_stance_X_step_onset` | Joint_step_onset | testable | *** |
| `Hip_stance_Y_step_onset` | Joint_step_onset | testable | * |
| `Hip_stance_Z_step_onset` | Joint_step_onset | testable | * |
| `Knee_stance_X_step_onset` | Joint_step_onset | testable | n.s. |
| `Knee_stance_Y_step_onset` | Joint_step_onset | testable | n.s. |
| `Knee_stance_Z_step_onset` | Joint_step_onset | testable | *** |
| `Ankle_stance_X_step_onset` | Joint_step_onset | testable | n.s. |
| `Ankle_stance_Y_step_onset` | Joint_step_onset | testable | n.s. |
| `Ankle_stance_Z_step_onset` | Joint_step_onset | testable | *** |
| `Trunk_X_step_onset` | Joint_step_onset | testable | *** |
| `Trunk_Y_step_onset` | Joint_step_onset | testable | * |
| `Trunk_Z_step_onset` | Joint_step_onset | testable | n.s. |
| `Neck_X_step_onset` | Joint_step_onset | testable | n.s. |
| `Neck_Y_step_onset` | Joint_step_onset | testable | n.s. |
| `Neck_Z_step_onset` | Joint_step_onset | testable | n.s. |
| `Hip_stance_ref_X_deg_s_step_onset` | Velocity_step_onset | testable | *** |
| `Hip_stance_ref_Y_deg_s_step_onset` | Velocity_step_onset | testable | * |
| `Hip_stance_ref_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Hip_stance_mov_X_deg_s_step_onset` | Velocity_step_onset | testable | *** |
| `Hip_stance_mov_Y_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Hip_stance_mov_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Hip_stance_ref_X_Nm_step_onset` | Moment_step_onset | testable | * |
| `Hip_stance_ref_Y_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Hip_stance_ref_Z_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Knee_stance_ref_X_deg_s_step_onset` | Velocity_step_onset | testable | *** |
| `Knee_stance_ref_Y_deg_s_step_onset` | Velocity_step_onset | testable | * |
| `Knee_stance_ref_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Knee_stance_mov_X_deg_s_step_onset` | Velocity_step_onset | testable | *** |
| `Knee_stance_mov_Y_deg_s_step_onset` | Velocity_step_onset | testable | * |
| `Knee_stance_mov_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Knee_stance_ref_X_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Knee_stance_ref_Y_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Knee_stance_ref_Z_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Ankle_stance_ref_X_deg_s_step_onset` | Velocity_step_onset | testable | *** |
| `Ankle_stance_ref_Y_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Ankle_stance_ref_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Ankle_stance_mov_X_deg_s_step_onset` | Velocity_step_onset | testable | *** |
| `Ankle_stance_mov_Y_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Ankle_stance_mov_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Ankle_stance_ref_X_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Ankle_stance_ref_Y_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Ankle_stance_ref_Z_Nm_step_onset` | Moment_step_onset | testable | ** |
| `Trunk_ref_X_deg_s_step_onset` | Velocity_step_onset | testable | * |
| `Trunk_ref_Y_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Trunk_ref_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Trunk_mov_X_deg_s_step_onset` | Velocity_step_onset | testable | * |
| `Trunk_mov_Y_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Trunk_mov_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Trunk_ref_X_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Trunk_ref_Y_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Trunk_ref_Z_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Neck_ref_X_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Neck_ref_Y_deg_s_step_onset` | Velocity_step_onset | testable | * |
| `Neck_ref_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Neck_mov_X_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Neck_mov_Y_deg_s_step_onset` | Velocity_step_onset | testable | *** |
| `Neck_mov_Z_deg_s_step_onset` | Velocity_step_onset | testable | n.s. |
| `Neck_ref_X_Nm_step_onset` | Moment_step_onset | testable | n.s. |
| `Neck_ref_Y_Nm_step_onset` | Moment_step_onset | testable | *** |
| `Neck_ref_Z_Nm_step_onset` | Moment_step_onset | testable | *** |
| `COP_X_step_onset` | Force_step_onset | testable | n.s. |
| `COP_Y_step_onset` | Force_step_onset | testable | *** |
| `GRF_X_step_onset` | Force_step_onset | testable | ** |
| `GRF_Y_step_onset` | Force_step_onset | testable | *** |
| `GRF_Z_step_onset` | Force_step_onset | testable | * |
| `AnkleTorqueMid_Y_perkg_step_onset` | Force_step_onset | testable | *** |

### Significant Step-Onset Variables Only (BH-FDR < 0.05)

| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---|---:|---:|---:|---|
| `Hip_stance_ref_Y_Nm_step_onset` | Moment_step_onset | -72.94±26.95 | 10.29±17.33 | -83.73 | *** |
| `COP_Y_step_onset` | Force_step_onset | 0.56±0.03 | 0.48±0.02 | 0.08 | *** |
| `Knee_stance_ref_Z_Nm_step_onset` | Moment_step_onset | 17.31±8.14 | -1.14±2.10 | 18.54 | *** |
| `GRF_Y_step_onset` | Force_step_onset | 44.47±21.60 | -2.51±11.29 | 46.35 | *** |
| `Knee_stance_ref_Y_Nm_step_onset` | Moment_step_onset | -38.45±20.81 | 14.09±15.72 | -52.54 | *** |
| `Trunk_ref_Z_Nm_step_onset` | Moment_step_onset | 6.79±3.47 | 0.62±2.92 | 5.85 | *** |
| `vCOM_X_step_onset` | Balance_step_onset | -0.02±0.05 | 0.03±0.03 | -0.05 | *** |
| `AnkleTorqueMid_Y_perkg_step_onset` | Force_step_onset | 1.43±0.23 | 1.74±0.09 | -0.23 | *** |
| `Trunk_ref_Y_Nm_step_onset` | Moment_step_onset | 12.10±8.16 | 0.50±4.80 | 10.69 | *** |
| `Ankle_stance_ref_Y_Nm_step_onset` | Moment_step_onset | -19.31±15.63 | 8.66±18.50 | -28.32 | *** |
| `Ankle_stance_ref_X_Nm_step_onset` | Moment_step_onset | -69.18±23.12 | -31.15±28.94 | -35.62 | *** |
| `Hip_stance_mov_X_deg_s_step_onset` | Velocity_step_onset | 7.03±10.49 | -8.15±12.44 | 13.61 | *** |
| `Hip_stance_ref_X_deg_s_step_onset` | Velocity_step_onset | 6.75±10.05 | -7.98±12.36 | 13.11 | *** |
| `vCOM_Y_step_onset` | Balance_step_onset | -0.03±0.04 | 0.01±0.03 | -0.04 | *** |
| `MOS_minDist_signed_step_onset` | Balance_step_onset | 0.04±0.03 | 0.07±0.01 | -0.02 | *** |
| `xCOM_BOS_norm_step_onset` | Balance_step_onset | 0.28±0.21 | 0.41±0.12 | -0.13 | *** |
| `Hip_stance_X_step_onset` | Joint_step_onset | 6.70±2.89 | 4.49±2.20 | 2.55 | *** |
| `Knee_stance_ref_X_Nm_step_onset` | Moment_step_onset | -86.65±34.38 | -42.16±38.25 | -42.18 | *** |
| `Neck_ref_Z_Nm_step_onset` | Moment_step_onset | -0.26±0.38 | 0.08±0.35 | -0.29 | *** |
| `COM_X_step_onset` | Balance_step_onset | -1.51±0.03 | -1.50±0.03 | -0.02 | *** |
| `MOS_AP_v3d_step_onset` | Balance_step_onset | 0.06±0.03 | 0.08±0.02 | -0.02 | *** |
| `Trunk_ref_X_Nm_step_onset` | Moment_step_onset | -27.92±12.79 | -34.92±9.80 | 6.99 | *** |
| `Hip_stance_ref_Z_Nm_step_onset` | Moment_step_onset | -5.58±9.75 | 1.96±2.83 | -7.36 | *** |
| `Neck_ref_Y_Nm_step_onset` | Moment_step_onset | 0.51±0.80 | -0.01±0.63 | 0.51 | *** |
| `Trunk_X_step_onset` | Joint_step_onset | 2.15±2.82 | 0.04±1.97 | 2.00 | *** |
| `Ankle_stance_mov_X_deg_s_step_onset` | Velocity_step_onset | -0.59±13.42 | 10.54±16.45 | -9.89 | *** |
| `Ankle_stance_ref_X_deg_s_step_onset` | Velocity_step_onset | -1.02±13.59 | 9.84±16.54 | -9.25 | *** |
| `Knee_stance_mov_X_deg_s_step_onset` | Velocity_step_onset | -3.85±13.90 | 2.70±10.97 | -7.98 | *** |
| `Ankle_stance_Z_step_onset` | Joint_step_onset | -0.49±2.27 | 0.99±1.85 | -1.03 | *** |
| `Knee_stance_ref_X_deg_s_step_onset` | Velocity_step_onset | -3.80±14.27 | 2.63±11.16 | -7.89 | *** |
| `Knee_stance_Z_step_onset` | Joint_step_onset | -1.25±1.57 | -0.75±1.15 | -0.58 | *** |
| `Neck_mov_Y_deg_s_step_onset` | Velocity_step_onset | -7.38±8.34 | -1.40±9.25 | -6.17 | *** |
| `Ankle_stance_ref_Z_Nm_step_onset` | Moment_step_onset | 5.05±5.60 | 2.08±4.38 | 2.81 | ** |
| `GRF_X_step_onset` | Force_step_onset | 6.69±22.36 | -2.28±16.73 | 9.76 | ** |
| `Knee_stance_ref_Y_deg_s_step_onset` | Velocity_step_onset | -1.52±4.03 | -0.09±2.16 | -1.74 | * |
| `Hip_stance_ref_Y_deg_s_step_onset` | Velocity_step_onset | 2.16±9.26 | -1.68±4.90 | 3.65 | * |
| `Knee_stance_mov_Y_deg_s_step_onset` | Velocity_step_onset | -1.82±4.11 | 0.20±3.40 | -1.95 | * |
| `Hip_stance_Z_step_onset` | Joint_step_onset | -0.06±2.55 | 0.78±1.60 | -0.93 | * |
| `Neck_ref_Y_deg_s_step_onset` | Velocity_step_onset | -8.05±9.90 | -3.81±9.78 | -5.00 | * |
| `Hip_stance_ref_X_Nm_step_onset` | Moment_step_onset | -41.30±25.21 | -28.89±28.75 | -11.44 | * |
| `Trunk_ref_X_deg_s_step_onset` | Velocity_step_onset | 1.74±11.87 | 6.28±11.14 | -4.53 | * |
| `Trunk_mov_X_deg_s_step_onset` | Velocity_step_onset | 1.49±12.85 | 6.67±11.30 | -4.57 | * |
| `GRF_Z_step_onset` | Force_step_onset | 6.23±34.30 | -8.49±31.54 | 11.29 | * |
| `Hip_stance_Y_step_onset` | Joint_step_onset | -0.67±1.62 | 0.04±1.03 | -0.49 | * |
| `Trunk_Y_step_onset` | Joint_step_onset | 0.80±1.71 | 0.46±0.95 | 0.51 | * |

### Outlier Exclusion Summary (Step Onset)

| Variable | Step raw | Step outliers | Step kept | Nonstep raw | Nonstep outliers | Nonstep kept |
|---|---:|---:|---:|---:|---:|---:|
| `AnkleTorqueMid_Y_perkg_step_onset` | 52 | 0 | 52 | 67 | 13 | 54 |
| `Ankle_stance_X_step_onset` | 52 | 0 | 52 | 67 | 0 | 67 |
| `Ankle_stance_Y_step_onset` | 52 | 3 | 49 | 67 | 1 | 66 |
| `Ankle_stance_Z_step_onset` | 52 | 2 | 50 | 67 | 1 | 66 |
| `Ankle_stance_mov_X_deg_s_step_onset` | 52 | 1 | 51 | 67 | 3 | 64 |
| `Ankle_stance_mov_Y_deg_s_step_onset` | 52 | 6 | 46 | 67 | 6 | 61 |
| `Ankle_stance_mov_Z_deg_s_step_onset` | 52 | 5 | 47 | 67 | 7 | 60 |
| `Ankle_stance_ref_X_Nm_step_onset` | 52 | 0 | 52 | 67 | 0 | 67 |
| `Ankle_stance_ref_X_deg_s_step_onset` | 52 | 2 | 50 | 67 | 3 | 64 |
| `Ankle_stance_ref_Y_Nm_step_onset` | 52 | 3 | 49 | 67 | 10 | 57 |
| `Ankle_stance_ref_Y_deg_s_step_onset` | 52 | 4 | 48 | 67 | 5 | 62 |
| `Ankle_stance_ref_Z_Nm_step_onset` | 52 | 3 | 49 | 67 | 6 | 61 |
| `Ankle_stance_ref_Z_deg_s_step_onset` | 52 | 4 | 48 | 67 | 8 | 59 |
| `COM_X_step_onset` | 52 | 1 | 51 | 67 | 0 | 67 |
| `COM_Y_step_onset` | 52 | 4 | 48 | 67 | 0 | 67 |
| `COP_X_step_onset` | 52 | 0 | 52 | 67 | 1 | 66 |
| `COP_Y_step_onset` | 52 | 5 | 47 | 67 | 2 | 65 |
| `GRF_X_step_onset` | 52 | 4 | 48 | 67 | 11 | 56 |
| `GRF_Y_step_onset` | 52 | 6 | 46 | 67 | 6 | 61 |
| `GRF_Z_step_onset` | 52 | 6 | 46 | 67 | 7 | 60 |
| `Hip_stance_X_step_onset` | 52 | 3 | 49 | 67 | 8 | 59 |
| `Hip_stance_Y_step_onset` | 52 | 1 | 51 | 67 | 1 | 66 |
| `Hip_stance_Z_step_onset` | 52 | 4 | 48 | 67 | 5 | 62 |
| `Hip_stance_mov_X_deg_s_step_onset` | 52 | 2 | 50 | 67 | 1 | 66 |
| `Hip_stance_mov_Y_deg_s_step_onset` | 52 | 2 | 50 | 67 | 3 | 64 |
| `Hip_stance_mov_Z_deg_s_step_onset` | 52 | 8 | 44 | 67 | 6 | 61 |
| `Hip_stance_ref_X_Nm_step_onset` | 52 | 3 | 49 | 67 | 2 | 65 |
| `Hip_stance_ref_X_deg_s_step_onset` | 52 | 2 | 50 | 67 | 1 | 66 |
| `Hip_stance_ref_Y_Nm_step_onset` | 52 | 7 | 45 | 67 | 12 | 55 |
| `Hip_stance_ref_Y_deg_s_step_onset` | 52 | 3 | 49 | 67 | 3 | 64 |
| `Hip_stance_ref_Z_Nm_step_onset` | 52 | 3 | 49 | 67 | 12 | 55 |
| `Hip_stance_ref_Z_deg_s_step_onset` | 52 | 8 | 44 | 67 | 5 | 62 |
| `Knee_stance_X_step_onset` | 52 | 0 | 52 | 67 | 3 | 64 |
| `Knee_stance_Y_step_onset` | 52 | 1 | 51 | 67 | 0 | 67 |
| `Knee_stance_Z_step_onset` | 52 | 1 | 51 | 67 | 0 | 67 |
| `Knee_stance_mov_X_deg_s_step_onset` | 52 | 2 | 50 | 67 | 3 | 64 |
| `Knee_stance_mov_Y_deg_s_step_onset` | 52 | 6 | 46 | 67 | 5 | 62 |
| `Knee_stance_mov_Z_deg_s_step_onset` | 52 | 2 | 50 | 67 | 5 | 62 |
| `Knee_stance_ref_X_Nm_step_onset` | 52 | 0 | 52 | 67 | 1 | 66 |
| `Knee_stance_ref_X_deg_s_step_onset` | 52 | 2 | 50 | 67 | 3 | 64 |
| `Knee_stance_ref_Y_Nm_step_onset` | 52 | 3 | 49 | 67 | 14 | 53 |
| `Knee_stance_ref_Y_deg_s_step_onset` | 52 | 4 | 48 | 67 | 10 | 57 |
| `Knee_stance_ref_Z_Nm_step_onset` | 52 | 2 | 50 | 67 | 16 | 51 |
| `Knee_stance_ref_Z_deg_s_step_onset` | 52 | 2 | 50 | 67 | 4 | 63 |
| `MOS_AP_v3d_step_onset` | 52 | 2 | 50 | 67 | 2 | 65 |
| `MOS_ML_v3d_step_onset` | 52 | 2 | 50 | 67 | 3 | 64 |
| `MOS_minDist_signed_step_onset` | 52 | 2 | 50 | 67 | 2 | 65 |
| `Neck_X_step_onset` | 52 | 5 | 47 | 67 | 6 | 61 |
| `Neck_Y_step_onset` | 52 | 2 | 50 | 67 | 0 | 67 |
| `Neck_Z_step_onset` | 52 | 3 | 49 | 67 | 4 | 63 |
| `Neck_mov_X_deg_s_step_onset` | 52 | 5 | 47 | 67 | 1 | 66 |
| `Neck_mov_Y_deg_s_step_onset` | 52 | 8 | 44 | 67 | 6 | 61 |
| `Neck_mov_Z_deg_s_step_onset` | 52 | 3 | 49 | 67 | 4 | 63 |
| `Neck_ref_X_Nm_step_onset` | 52 | 3 | 49 | 67 | 6 | 61 |
| `Neck_ref_X_deg_s_step_onset` | 52 | 6 | 46 | 67 | 1 | 66 |
| `Neck_ref_Y_Nm_step_onset` | 52 | 1 | 51 | 67 | 0 | 67 |
| `Neck_ref_Y_deg_s_step_onset` | 52 | 7 | 45 | 67 | 4 | 63 |
| `Neck_ref_Z_Nm_step_onset` | 52 | 3 | 49 | 67 | 0 | 67 |
| `Neck_ref_Z_deg_s_step_onset` | 52 | 1 | 51 | 67 | 3 | 64 |
| `Trunk_X_step_onset` | 52 | 4 | 48 | 67 | 8 | 59 |
| `Trunk_Y_step_onset` | 52 | 3 | 49 | 67 | 8 | 59 |
| `Trunk_Z_step_onset` | 52 | 0 | 52 | 67 | 3 | 64 |
| `Trunk_mov_X_deg_s_step_onset` | 52 | 2 | 50 | 67 | 2 | 65 |
| `Trunk_mov_Y_deg_s_step_onset` | 52 | 2 | 50 | 67 | 9 | 58 |
| `Trunk_mov_Z_deg_s_step_onset` | 52 | 6 | 46 | 67 | 3 | 64 |
| `Trunk_ref_X_Nm_step_onset` | 52 | 1 | 51 | 67 | 1 | 66 |
| `Trunk_ref_X_deg_s_step_onset` | 52 | 3 | 49 | 67 | 3 | 64 |
| `Trunk_ref_Y_Nm_step_onset` | 52 | 5 | 47 | 67 | 2 | 65 |
| `Trunk_ref_Y_deg_s_step_onset` | 52 | 3 | 49 | 67 | 8 | 59 |
| `Trunk_ref_Z_Nm_step_onset` | 52 | 7 | 45 | 67 | 4 | 63 |
| `Trunk_ref_Z_deg_s_step_onset` | 52 | 6 | 46 | 67 | 3 | 64 |
| `vCOM_X_step_onset` | 52 | 3 | 49 | 67 | 4 | 63 |
| `vCOM_Y_step_onset` | 52 | 1 | 51 | 67 | 2 | 65 |
| `xCOM_BOS_norm_step_onset` | 52 | 0 | 52 | 67 | 0 | 67 |

## Interpretation & Conclusion

1. platform onset 단일 프레임에서는 29개 변수 중 `10`개만 유의해 strict 기준 가설은 **FAIL**였다. 즉, 섭동 직후 posture snapshot만으로 전략 분화를 광범위하게 설명하기는 어려웠다.
2. platform onset에서는 관절 각도 변수는 총 15개 중 2개가 유의했고 (`Hip_stance_Y_abs_onset, Trunk_Y_abs_onset`), 나머지는 `n.s.`였다. 유의 변수는 COM/MOS와 일부 force/torque 변수에 제한적으로 나타났다.
3. step onset 단일 프레임에서는 총 `45`개가 FDR 유의였고, joint-angle 15개 중 `7`개가 유의했다 (`Hip_stance_X_step_onset, Trunk_X_step_onset, Ankle_stance_Z_step_onset, Knee_stance_Z_step_onset, Hip_stance_Z_step_onset, Hip_stance_Y_step_onset, Trunk_Y_step_onset`). 본 데이터에서는 전략 분화가 섭동 직후보다 발 들기 직전 프레임에서 더 강하게 관찰됐다.
4. 따라서 single-frame 비교만 놓고 보면, step/nonstep 전략 차이는 `platform onset`의 초기 snapshot보다 `step onset` 직전의 준비 자세에서 더 뚜렷하다. 다만 `platform onset` 29개 변수와 `step onset` 74개 변수 전체가 일관되게 유의하지는 않으므로, 단일 프레임 변수만으로 전략 차이를 완전히 설명한다고 단정할 수는 없다.

## Limitations

1. 원문의 task-level goal 파라미터를 직접 모델링하지 않았다.
2. 본 분석은 Van Wouwe 2021의 simulation 기반 인과 프레임을 1:1 재현한 결과가 아니다.
3. step onset은 nonstep trial에서 subject 평균 `step_onset_local`을 참조하므로, step trial의 실제 발 들기 순간과 완전히 같은 관측점은 아니다.
4. 본 문서는 single-frame 분석만 다루며, onset 전 구간 평균과 `95% CI` 해석은 `report_baseline.md`를 따로 봐야 한다.

## Reproduction

```bash
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

- Output: 콘솔 통계 결과 + 자동 갱신 `report.md`, `결과) 주제2-Segement Angle.md`

---
Auto-generated by analyze_initial_posture_strategy_lmm.py.
