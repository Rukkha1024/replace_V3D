# Initial Posture Strategy LMM (Platform Onset)

## Research Question

**"Van Wouwe et al. (2021) 관점에서, 초기 자세(initial posture)가 step/nonstep 전략 차이를 설명한다면 platform onset 시점 변수에서 step/nonstep 차이가 광범위하게 유의한가?"**

본 분석은 `analysis/step_vs_nonstep_lmm`의 기존 변수 체계를 유지하면서, onset-zero 구조 때문에 onset 시점에서 검정이 불가능한 각도/힘 변수 문제를 분리해 점검하기 위해 수행했다. 특히 각도는 C3D에서 Visual3D-like raw angle을 재계산하여 onset 절대값을 추가 검정했다.

## Prior Studies

### Van Wouwe et al. (2021) — Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance

- **Methodology**: 예측 시뮬레이션 + 실험 데이터 결합. 초기 자세(COM 위치)와 task-level goal(노력-안정성 우선순위)의 상호작용을 통해 postural strategy variability를 설명.
- **Experimental design**: 10명의 젊은 성인에게 예측 불가능한 backward support-surface translation 적용, stepping/nonstepping 반응 기록.
- **Key results**:
  - 최대 trunk lean의 변동성이 큼: within-subject mean range 약 `28.3°`, across-subject mean range 약 `39.9°`.
  - initial COM position과 maximal trunk lean 관계는 subject-specific하며 `R^2 = 0.29–0.82` 범위를 보임.
  - `xCOM/BOS_onset`, `xCOM/BOS_300ms`를 안정성 지표로 정의해 초기/초기반응 말기 상태를 비교.
- **Conclusions**: 초기 자세는 intra-subject variability를 설명하고, task-level goal 차이는 inter-subject 차이를 설명한다. 두 요인의 상호작용이 step incidence 및 전략 variability를 함께 설명한다.

## Methodological Adaptation

| Prior Study Method | Current Implementation | Deviation Rationale |
|---|---|---|
| Initial posture와 strategy 관계 분석 | `DV ~ step_TF + (1|subject)` LMM으로 step/nonstep 집단차 검정 | 원문은 simulation + subject-specific relation 중심, 본 분석은 현재 데이터 구조에 맞춰 집단 비교 모델 사용 |
| `xCOM/BOS_onset`, `xCOM/BOS_300ms` 지표 사용 | `xCOM_BOS_norm_onset`를 onset에서 계산 | 300ms 지점은 이번 분석 목적(엄격 onset 단일 시점)에서 제외 |
| Trunk/angle variability 해석 | onset-zero export와 분리하여 C3D 재계산 absolute angle onset 변수를 추가 | CSV의 onset-zero 구조에서는 onset 각도가 상수(0)여서 검정 불가 |
| 광범위 변수 비교 | 기존 34DV 원천 신호 기반 onset 변수 + absolute angle onset 추가 | 원문 1:1 재현이 아닌, 사용자 데이터에서 가설의 검정 가능성 점검 목적 |

**Summary**: 본 분석은 Van Wouwe 2021의 "초기 자세 + 전략 variability" 문제의식을 채택했지만, 원문 simulation 프레임 대신 현재 저장소 데이터 구조에 맞는 onset 단일시점 LMM으로 구현했다.

## Data Summary

- Trials: **125** (`step=53`, `nonstep=72`), subjects=24
- Input:
  - `output/all_trials_timeseries.csv`
  - `data/perturb_inform.xlsm`
  - `data/all_data/*.c3d` (absolute angle 재계산용)
- 분석 변수:
  - onset 후보 총 **24개**
  - 검정 가능(testable) **13개**
  - 검정 불가(untestable) **11개** (`constant_zero`)

## Analysis Methodology

- **Analysis point**: `platform_onset_local` 단일 프레임
- **Statistical model**: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- **Multiple comparison correction**: BH-FDR (testable onset 변수 전체에 1회 적용)
- **Significance reporting**: `Sig` only (`***`, `**`, `*`, `n.s.`), `alpha=0.05`
- **Displayed result policy**: Results 표에는 **FDR 유의 변수만** 표시
- **Absolute angle handling**:
  - CSV onset-zero 각도(`*_deg`)는 onset에서 구조적으로 0이므로 검정불가로 분류
  - C3D에서 `compute_v3d_joint_angles_3d`로 raw angle을 재계산해 onset 절대각(`*_abs_onset`)을 별도 검정

### Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|---|---|---|---|
| X (AP) | 전방(anterior) | 후방(posterior) | `COM_X`, `vCOM_X`, `MOS_AP_v3d`, `xCOM_BOS_norm_onset`, `Trunk_X_abs_onset` |
| Y (ML) | 우측(right/lateral) | 좌측(left/medial) | `COM_Y`, `vCOM_Y`, `MOS_ML_v3d` |
| Z (Vertical) | 상방(up) | 하방(down) | 본 onset 결과표에는 직접 사용 안 함 |

### Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|---|---|---|---|
| `MOS_minDist_signed` | 안정 여유가 상대적으로 큼(경계로부터 안쪽 여유 증가) | 안정 여유가 상대적으로 작음(경계 근접/감소) | step−nonstep estimate 부호 해석 |
| `MOS_AP_v3d` | AP 방향 안정 여유 증가 | AP 방향 안정 여유 감소 | step−nonstep estimate 부호 해석 |
| `MOS_ML_v3d` | ML 방향 안정 여유 증가 | ML 방향 안정 여유 감소 | step−nonstep estimate 부호 해석 |
| `xCOM_BOS_norm_onset` | BOS 내 상대 위치가 더 posterior 쪽 | BOS 내 상대 위치가 더 anterior 쪽 | `(xCOM_X-BOS_minX)/(BOS_maxX-BOS_minX)` |

### Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|---|---|---|
| Joint angle (`*_deg`) | 축 방향 회전 부호 | 본 CSV는 onset-zero export라 onset 프레임 값은 0으로 고정 가능 |
| Joint absolute onset (`*_abs_onset`) | raw Visual3D-like angle 부호 | C3D 재계산값 사용 (onset-zero 미적용) |
| Force/Torque (`GRF_*`, `AnkleTorque*`) | 축 방향 힘/토크 부호 | 본 CSV는 onset-zero export라 onset 프레임 값은 0으로 고정 가능 |
| COP onset0 | onset 대비 변위 | 정의상 onset 프레임에서 0 |

## Results

### Hypothesis Verdict (strict)

- **Rule**: testable onset 변수 전부가 FDR 유의이고, untestable 핵심 변수군이 없어야 PASS
- **Observed**: testable significant ratio = `4/13`, untestable=`11`
- **Verdict**: **FAIL**

### Significant Variables Only (BH-FDR < 0.05)

| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---|---:|---:|---:|---|
| `MOS_minDist_signed` | Balance | 0.0718±0.0143 | 0.0573±0.0154 | 0.0109 | *** |
| `xCOM_BOS_norm_onset` | Balance | 0.6318±0.0681 | 0.7005±0.0727 | -0.0513 | *** |
| `MOS_AP_v3d` | Balance | 0.0753±0.0147 | 0.0619±0.0159 | 0.0102 | *** |
| `vCOM_X` | Balance | 0.0097±0.0088 | 0.0161±0.0100 | -0.0037 | ** |

### Variable Classification (for interpretation)

- **Untestable at onset (constant_zero)**: `Hip_stance_X_deg`, `Knee_stance_X_deg`, `Ankle_stance_X_deg`, `Trunk_X_deg`, `Neck_X_deg`, `COP_X_m_onset0`, `COP_Y_m_onset0`, `GRF_X_N`, `GRF_Y_N`, `GRF_Z_N`, `AnkleTorqueMid_int_Y_Nm_per_kg`
- 이유: 배치 export가 onset-zero를 적용해 onset 프레임 값이 구조적으로 0이 됨.

## Comparison with Prior Studies

| Comparison Item | Prior Study Result | Current Result | Verdict |
|---|---|---|---|
| 초기자세가 전략 variability를 설명 | 초기 COM 위치가 trunk lean과 subject-specific하게 연관 (`R^2=0.29–0.82`) | onset 변수 중 4개만 유의, 11개는 구조적으로 검정불가 | Partially consistent |
| `xCOM/BOS_onset` 중요성 | onset 안정성 지표로 사용 | `xCOM_BOS_norm_onset` 유의 (`step < nonstep`) | Consistent (direction tested) |
| trunk 전략 지표 | trunk lean variability가 핵심 | onset-zero trunk는 검정불가, raw absolute trunk onset은 비유의 | Not tested (onset-zero confound) |
| 전반적 결론 수준 | 초기자세 + task-level goal 상호작용 강조 | step/nonstep 집단 평균 차 중심, goal 파라미터 직접 모델링 없음 | Inconsistent (scope mismatch) |

## Interpretation & Conclusion

1. 엄격한 onset 단일시점 기준에서 Van Wouwe식 "광범위 onset 차이" 가설은 본 데이터에서 **FAIL**이다.
2. 다만 balance 계열에서 `MOS_minDist_signed`, `MOS_AP_v3d`, `xCOM_BOS_norm_onset`, `vCOM_X`는 유의하여 초기 자세 관련 신호가 일부 존재한다.
3. 각도/힘/토크의 상당수는 "비유의"가 아니라 "검정불가(구조적 상수)"이며, 이는 통계 결론 이전에 데이터 표현 방식(onset-zero)의 영향이다.

## Limitations

1. 원문의 task-level goal 파라미터를 직접 모델링하지 않았고, step/nonstep 집단 비교로 단순화했다.
2. onset-zero export 구조 때문에 onset 단일시점에서 많은 변수(특히 각도/힘/토크)가 구조적으로 상수가 된다.
3. 본 비교는 Van Wouwe 2021의 simulation 기반 인과 프레임을 1:1 재현한 결과가 아니라, 사용자 데이터에 맞춘 operational test이다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
conda run --no-capture-output -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`, `data/all_data/*.c3d`
- Output: 콘솔 통계 결과(유의 변수만 표시), `report.md`

## Figures

- 이번 분석은 사용자 요청에 따라 **figure를 생성하지 않음**.

