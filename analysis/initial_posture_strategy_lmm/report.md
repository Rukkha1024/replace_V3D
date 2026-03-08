# Initial Posture Strategy LMM (Platform Onset + Step Onset)

## Research Question

**"Van Wouwe et al. (2021) 관점에서, 초기 자세(initial posture)가 step/nonstep 전략 차이를 설명한다면 platform onset과 step onset 단일 프레임 변수에서 step/nonstep 차이가 광범위하게 유의한가?"**

이번 버전은 platform onset 29개 변수와 step onset 29개 변수를 각각 비교하되, 각 변수별 `step/nonstep` 그룹 내부 `1.5×IQR` 이상치를 제외한 단일 프레임 LMM 결과를 보고한다.

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

- **Analysis point**: `platform_onset_local` 단일 프레임
- **Statistical model**: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- **Outlier rule**: 각 변수별 `step/nonstep` 그룹 내부에서 `1.5×IQR` 밖 trial 제거
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
- **Observed**: testable significant ratio = `18/29`

### Step-Onset Variables (Full Set, n=29)

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
| `COP_X_step_onset` | Force_step_onset | testable | n.s. |
| `COP_Y_step_onset` | Force_step_onset | testable | *** |
| `GRF_X_step_onset` | Force_step_onset | testable | ** |
| `GRF_Y_step_onset` | Force_step_onset | testable | *** |
| `GRF_Z_step_onset` | Force_step_onset | testable | * |
| `AnkleTorqueMid_Y_perkg_step_onset` | Force_step_onset | testable | *** |

### Significant Step-Onset Variables Only (BH-FDR < 0.05)

| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---|---:|---:|---:|---|
| `COP_Y_step_onset` | Force_step_onset | 0.56±0.03 | 0.48±0.02 | 0.08 | *** |
| `GRF_Y_step_onset` | Force_step_onset | 44.47±21.60 | -2.51±11.29 | 46.35 | *** |
| `vCOM_X_step_onset` | Balance_step_onset | -0.02±0.05 | 0.03±0.03 | -0.05 | *** |
| `AnkleTorqueMid_Y_perkg_step_onset` | Force_step_onset | 1.43±0.23 | 1.74±0.09 | -0.23 | *** |
| `vCOM_Y_step_onset` | Balance_step_onset | -0.03±0.04 | 0.01±0.03 | -0.04 | *** |
| `MOS_minDist_signed_step_onset` | Balance_step_onset | 0.04±0.03 | 0.07±0.01 | -0.02 | *** |
| `xCOM_BOS_norm_step_onset` | Balance_step_onset | 0.28±0.21 | 0.41±0.12 | -0.13 | *** |
| `Hip_stance_X_step_onset` | Joint_step_onset | 6.70±2.89 | 4.49±2.20 | 2.55 | *** |
| `COM_X_step_onset` | Balance_step_onset | -1.51±0.03 | -1.50±0.03 | -0.02 | *** |
| `MOS_AP_v3d_step_onset` | Balance_step_onset | 0.06±0.03 | 0.08±0.02 | -0.02 | *** |
| `Trunk_X_step_onset` | Joint_step_onset | 2.15±2.82 | 0.04±1.97 | 2.00 | *** |
| `Ankle_stance_Z_step_onset` | Joint_step_onset | -0.49±2.27 | 0.99±1.85 | -1.03 | *** |
| `Knee_stance_Z_step_onset` | Joint_step_onset | -1.25±1.57 | -0.75±1.15 | -0.58 | *** |
| `GRF_X_step_onset` | Force_step_onset | 6.69±22.36 | -2.28±16.73 | 9.76 | ** |
| `Hip_stance_Z_step_onset` | Joint_step_onset | -0.06±2.55 | 0.78±1.60 | -0.93 | * |
| `Hip_stance_Y_step_onset` | Joint_step_onset | -0.67±1.62 | 0.04±1.03 | -0.49 | * |
| `GRF_Z_step_onset` | Force_step_onset | 6.23±34.30 | -8.49±31.54 | 11.29 | * |
| `Trunk_Y_step_onset` | Joint_step_onset | 0.80±1.71 | 0.46±0.95 | 0.51 | * |

### Outlier Exclusion Summary (Step Onset)

| Variable | Step raw | Step outliers | Step kept | Nonstep raw | Nonstep outliers | Nonstep kept |
|---|---:|---:|---:|---:|---:|---:|
| `AnkleTorqueMid_Y_perkg_step_onset` | 52 | 0 | 52 | 67 | 13 | 54 |
| `Ankle_stance_X_step_onset` | 52 | 0 | 52 | 67 | 0 | 67 |
| `Ankle_stance_Y_step_onset` | 52 | 3 | 49 | 67 | 1 | 66 |
| `Ankle_stance_Z_step_onset` | 52 | 2 | 50 | 67 | 1 | 66 |
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
| `Knee_stance_X_step_onset` | 52 | 0 | 52 | 67 | 3 | 64 |
| `Knee_stance_Y_step_onset` | 52 | 1 | 51 | 67 | 0 | 67 |
| `Knee_stance_Z_step_onset` | 52 | 1 | 51 | 67 | 0 | 67 |
| `MOS_AP_v3d_step_onset` | 52 | 2 | 50 | 67 | 2 | 65 |
| `MOS_ML_v3d_step_onset` | 52 | 2 | 50 | 67 | 3 | 64 |
| `MOS_minDist_signed_step_onset` | 52 | 2 | 50 | 67 | 2 | 65 |
| `Neck_X_step_onset` | 52 | 5 | 47 | 67 | 6 | 61 |
| `Neck_Y_step_onset` | 52 | 2 | 50 | 67 | 0 | 67 |
| `Neck_Z_step_onset` | 52 | 3 | 49 | 67 | 4 | 63 |
| `Trunk_X_step_onset` | 52 | 4 | 48 | 67 | 8 | 59 |
| `Trunk_Y_step_onset` | 52 | 3 | 49 | 67 | 8 | 59 |
| `Trunk_Z_step_onset` | 52 | 0 | 52 | 67 | 3 | 64 |
| `vCOM_X_step_onset` | 52 | 3 | 49 | 67 | 4 | 63 |
| `vCOM_Y_step_onset` | 52 | 1 | 51 | 67 | 2 | 65 |
| `xCOM_BOS_norm_step_onset` | 52 | 0 | 52 | 67 | 0 | 67 |

## Interpretation & Conclusion

1. 각 변수별 이상치를 제외하고 보아도 platform onset 29개 변수의 strict 기준 가설은 **FAIL**였다.
2. platform onset에서는 관절 각도 변수는 총 15개 중 2개가 유의했고 (`Hip_stance_Y_abs_onset, Trunk_Y_abs_onset`), 나머지는 `n.s.`였다. 유의 변수는 COM/MOS 및 ankle torque 일부에도 관찰되었다.
3. step onset 29개 변수에서는 총 `18`개가 FDR 유의였고, joint-angle 15개 중 `7`개가 유의했다 (`Hip_stance_X_step_onset, Trunk_X_step_onset, Ankle_stance_Z_step_onset, Knee_stance_Z_step_onset, Hip_stance_Z_step_onset, Hip_stance_Y_step_onset, Trunk_Y_step_onset`).
4. 따라서 본 데이터에서는 초기 자세 차이가 시점에 따라 다르게 나타날 수 있지만, 단일 시점 변수만으로 step/nonstep 전략 차이를 광범위하게 설명한다고 단정하기는 어렵다.

## Limitations

1. 원문의 task-level goal 파라미터를 직접 모델링하지 않았다.
2. 본 분석은 Van Wouwe 2021의 simulation 기반 인과 프레임을 1:1 재현한 결과가 아니다.
3. step onset은 nonstep trial에서 subject 평균 `step_onset_local`을 참조하므로, 직접 관측 step 시점과 완전히 같지는 않다.

## Reproduction

```bash
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

- Output: 콘솔 통계 결과 + 자동 갱신 `report.md`, `결과) 주제2-Segement Angle.md`

---
Auto-generated by analyze_initial_posture_strategy_lmm.py.
