# Initial Posture Strategy LMM (Platform Onset)

## Research Question

**"Van Wouwe et al. (2021) 관점에서, 초기 자세(initial posture)가 step/nonstep 전략 차이를 설명한다면 platform onset 시점 변수에서 step/nonstep 차이가 광범위하게 유의한가?"**

이번 버전은 onset-zero 변수(기존 export 각도/힘) 대신 C3D에서 재계산한 **absolute onset 변수**를 각도와 force에 모두 적용해, "검정불가" 문제를 최소화한 모델이다.

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
| `vCOM_Y` | Balance | testable | n.s. |
| `MOS_minDist_signed` | Balance | testable | *** |
| `MOS_AP_v3d` | Balance | testable | *** |
| `MOS_ML_v3d` | Balance | testable | n.s. |
| `xCOM_BOS_norm_onset` | Balance | testable | *** |
| `Hip_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Hip_stance_Y_abs_onset` | Joint_absolute | testable | n.s. |
| `Hip_stance_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `Knee_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Knee_stance_Y_abs_onset` | Joint_absolute | testable | *** |
| `Knee_stance_Z_abs_onset` | Joint_absolute | testable | *** |
| `Ankle_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Ankle_stance_Y_abs_onset` | Joint_absolute | testable | ** |
| `Ankle_stance_Z_abs_onset` | Joint_absolute | testable | *** |
| `Trunk_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Trunk_Y_abs_onset` | Joint_absolute | testable | n.s. |
| `Trunk_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `Neck_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Neck_Y_abs_onset` | Joint_absolute | testable | n.s. |
| `Neck_Z_abs_onset` | Joint_absolute | testable | n.s. |
| `COP_X_abs_onset` | Force_absolute | testable | n.s. |
| `COP_Y_abs_onset` | Force_absolute | testable | n.s. |
| `GRF_X_abs_onset` | Force_absolute | testable | n.s. |
| `GRF_Y_abs_onset` | Force_absolute | testable | n.s. |
| `GRF_Z_abs_onset` | Force_absolute | testable | n.s. |
| `AnkleTorqueMid_Y_perkg_abs_onset` | Force_absolute | testable | *** |

## Results

### Hypothesis Verdict (strict)

- **Rule**: testable onset 변수 전부가 FDR 유의여야 PASS
- **Observed**: testable significant ratio = `9/29`, untestable=`0`
- **Verdict**: **FAIL**

### Significant Variables Only (BH-FDR < 0.05)

| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---|---:|---:|---:|---|
| `MOS_minDist_signed` | Balance | 0.07±0.01 | 0.06±0.02 | 0.01 | *** |
| `xCOM_BOS_norm_onset` | Balance | 0.63±0.07 | 0.70±0.07 | -0.05 | *** |
| `MOS_AP_v3d` | Balance | 0.08±0.01 | 0.06±0.02 | 0.01 | *** |
| `Knee_stance_Y_abs_onset` | Joint_absolute | 2.73±3.74 | -0.40±4.14 | 2.49 | *** |
| `AnkleTorqueMid_Y_perkg_abs_onset` | Force_absolute | -2.45±0.16 | -2.58±0.17 | 0.09 | *** |
| `Ankle_stance_Z_abs_onset` | Joint_absolute | 6.37±8.00 | -0.04±7.28 | 4.58 | *** |
| `Knee_stance_Z_abs_onset` | Joint_absolute | 3.09±4.28 | 0.01±3.28 | 2.32 | *** |
| `vCOM_X` | Balance | 0.01±0.01 | 0.02±0.01 | -0.00 | ** |
| `Ankle_stance_Y_abs_onset` | Joint_absolute | -8.58±7.85 | -2.04±10.64 | -4.80 | ** |

## Interpretation & Conclusion

1. 각도와 force를 absolute onset으로 전환해도 모든 onset 변수가 유의하지는 않았고, strict 기준 가설은 **FAIL**였다.
2. 관절 각도 변수는 총 15개 중 4개가 유의했고 (`Knee_stance_Y_abs_onset, Knee_stance_Z_abs_onset, Ankle_stance_Z_abs_onset, Ankle_stance_Y_abs_onset`), 나머지는 `n.s.`였다. 유의 변수는 COM/MOS 및 ankle torque 일부에도 관찰되었다.
3. 따라서 본 데이터에서는 onset 시점의 광범위한 초기 자세 차이가 step/nonstep 전략 차이를 직접 설명한다고 단정하기 어렵다.

## Limitations

1. 원문의 task-level goal 파라미터를 직접 모델링하지 않았다.
2. 본 분석은 Van Wouwe 2021의 simulation 기반 인과 프레임을 1:1 재현한 결과가 아니다.
3. 관절 각속도와 BOS 기하학 파생 지표 분해 검정은 본 버전에 포함하지 않았다.

## Reproduction

```bash
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

- Output: 콘솔 통계 결과 + 자동 갱신 `report.md`, `결과) 주제2-Segement Angle.md`

---
Auto-generated by analyze_initial_posture_strategy_lmm.py.
