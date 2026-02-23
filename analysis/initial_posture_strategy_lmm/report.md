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

## Methodological Adaptation

| Prior Study Method | Current Implementation | Deviation Rationale |
|---|---|---|
| 초기자세-전략 관계 분석 | `DV ~ step_TF + (1|subject)` LMM | 원문은 simulation + subject-specific relation 중심, 본 분석은 현재 데이터 구조의 집단 비교 모델 |
| `xCOM/BOS_onset`, `xCOM/BOS_300ms` | `xCOM_BOS_norm_onset`만 onset 단일시점 적용 | 이번 검정 목표를 onset 단일시점으로 고정 |
| 각도/힘 초기값 해석 | C3D에서 absolute onset 각도 + absolute onset force 재계산 | 기존 CSV onset-zero값은 onset에서 상수화될 수 있어 검정 왜곡 우려 |
| 광범위 변수 비교 | Balance + Joint absolute + Force absolute의 onset 변수 19개 | 원문 1:1 재현이 아닌 사용자 데이터 기반 가설 검정 |

**Summary**: Van Wouwe의 문제의식을 채택하되, 본 데이터 구조에 맞춰 onset 단일시점 LMM으로 운영화했다.

## Data Summary

- Trials: **125** (`step=53`, `nonstep=72`), subjects=24
- Input:
  - `output/all_trials_timeseries.csv`
  - `data/perturb_inform.xlsm`
  - `data/all_data/*.c3d`
  - `src/replace_v3d/torque/assets/fp_inertial_templates.npz`
- 분석 변수:
  - onset 후보 총 **19개**
  - 검정 가능(testable) **19개**
  - 검정 불가(untestable) **0개**

## Analysis Methodology

- **Analysis point**: `platform_onset_local` 단일 프레임
- **Statistical model**: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- **Multiple comparison correction**: BH-FDR (19개 onset 변수 전체 1회)
- **Significance reporting**: `Sig` only (`***`, `**`, `*`, `n.s.`), `alpha=0.05`
- **Displayed result policy**: Results 표에는 **FDR 유의 변수만** 표시
- **Absolute joint handling**:
  - `compute_v3d_joint_angles_3d`로 onset raw angle 추출
  - stance-equivalent 매핑(`step_r->L`, `step_l->R`, nonstep은 subject major step side, tie는 L/R 평균)
- **Absolute force handling**:
  - forceplate 채널을 Stage01 변환 + inertial subtraction 후 onset 절대값 사용
  - 사용 변수: `GRF_*_abs_onset`, `COP_*_abs_onset`, `AnkleTorqueMid_Y_perkg_abs_onset`
  - 비엄격 모드에서 inertial QC 경고 trial 4건 확인(분석은 계속 수행)

### Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|---|---|---|---|
| X (AP) | 전방(anterior) | 후방(posterior) | `COM_X`, `vCOM_X`, `MOS_AP_v3d`, `xCOM_BOS_norm_onset`, `GRF_X_abs_onset`, `COP_X_abs_onset` |
| Y (ML) | 우측(right/lateral) | 좌측(left/medial) | `COM_Y`, `vCOM_Y`, `MOS_ML_v3d`, `GRF_Y_abs_onset`, `COP_Y_abs_onset`, `AnkleTorqueMid_Y_perkg_abs_onset` |
| Z (Vertical) | 상방(up) | 하방(down) | `GRF_Z_abs_onset` |

### Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|---|---|---|---|
| `MOS_minDist_signed` | 안정 여유 증가 | 안정 여유 감소 | step−nonstep estimate 부호 해석 |
| `MOS_AP_v3d` | AP 안정 여유 증가 | AP 안정 여유 감소 | step−nonstep estimate 부호 해석 |
| `MOS_ML_v3d` | ML 안정 여유 증가 | ML 안정 여유 감소 | step−nonstep estimate 부호 해석 |
| `xCOM_BOS_norm_onset` | BOS 내 상대 위치가 더 posterior | BOS 내 상대 위치가 더 anterior | `(xCOM_X-BOS_minX)/(BOS_maxX-BOS_minX)` |

### Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|---|---|---|
| Joint absolute onset (`*_abs_onset`) | raw Visual3D-like angle 부호 | onset-zero 미적용 값 사용 |
| Force absolute onset (`GRF_*_abs_onset`) | Stage01 축 기준 힘 부호 | inertial subtraction 적용 후 onset 절대값 |
| COP absolute onset (`COP_*_abs_onset`) | Stage01 Cx/Cy 부호 | `compute_cop_stage01_xy` 결과 |
| Ankle torque absolute onset | internal torque Y 부호 | `Nm/kg` 정규화 값 사용 |

### Analyzed Variables (Full Set, n=19)

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
| `Knee_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Ankle_stance_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Trunk_X_abs_onset` | Joint_absolute | testable | n.s. |
| `Neck_X_abs_onset` | Joint_absolute | testable | n.s. |
| `COP_X_abs_onset` | Force_absolute | testable | n.s. |
| `COP_Y_abs_onset` | Force_absolute | testable | n.s. |
| `GRF_X_abs_onset` | Force_absolute | testable | n.s. |
| `GRF_Y_abs_onset` | Force_absolute | testable | n.s. |
| `GRF_Z_abs_onset` | Force_absolute | testable | n.s. |
| `AnkleTorqueMid_Y_perkg_abs_onset` | Force_absolute | testable | *** |

## Results

### Hypothesis Verdict (strict)

- **Rule**: testable onset 변수 전부가 FDR 유의여야 PASS
- **Observed**: testable significant ratio = `5/19`, untestable=`0`
- **Verdict**: **FAIL**

### Significant Variables Only (BH-FDR < 0.05)

| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---|---:|---:|---:|---|
| `MOS_minDist_signed` | Balance | 0.0718±0.0143 | 0.0573±0.0154 | 0.0109 | *** |
| `xCOM_BOS_norm_onset` | Balance | 0.6318±0.0681 | 0.7005±0.0727 | -0.0513 | *** |
| `MOS_AP_v3d` | Balance | 0.0753±0.0147 | 0.0619±0.0159 | 0.0102 | *** |
| `AnkleTorqueMid_Y_perkg_abs_onset` | Force_absolute | -2.4516±0.1632 | -2.5763±0.1679 | 0.0931 | *** |
| `vCOM_X` | Balance | 0.0097±0.0088 | 0.0161±0.0100 | -0.0037 | ** |

## Comparison with Prior Studies

| Comparison Item | Prior Study Result | Current Result | Verdict |
|---|---|---|---|
| 초기자세가 전략 variability를 설명 | 초기 COM 위치가 trunk lean과 subject-specific하게 연관 (`R^2=0.29–0.82`) | onset 변수 19개 중 5개만 유의 | Partially consistent |
| `xCOM/BOS_onset` 중요성 | onset 안정성 지표로 사용 | `xCOM_BOS_norm_onset` 유의 (`step < nonstep`) | Consistent |
| trunk 전략 지표 | trunk lean variability 핵심 | `Trunk_X_abs_onset`는 비유의 | Inconsistent |
| force 계열 초기값 영향 | 원문에서 간접적으로 전략 반응과 연계 | absolute force 중 `AnkleTorqueMid_Y_perkg_abs_onset` 유의 | Partially consistent |
| 결론 수준 | 초기자세 + task-level goal 상호작용 강조 | goal 파라미터 직접 모델링 없음 | Inconsistent (scope mismatch) |

## Interpretation & Conclusion

1. 각도와 force를 absolute onset으로 전환해도, 엄격 기준에서는 `5/19`만 유의하여 Van Wouwe식 "광범위 onset 차이" 가설은 **FAIL**이다.
2. 유의 변수는 `MOS_minDist_signed`, `MOS_AP_v3d`, `xCOM_BOS_norm_onset`, `vCOM_X`, `AnkleTorqueMid_Y_perkg_abs_onset`로, 초기 안정성/전후 방향 상태 신호가 일부 전략 차이를 반영한다.
3. 핵심은 "검정불가" 문제가 아니라, absolute 변환 이후에도 대부분 변수는 통계적으로 분리되지 않는다는 점이다.

## Limitations

1. 원문의 task-level goal 파라미터를 직접 모델링하지 않았다.
2. inertial subtraction QC 경고가 4 trial에서 관찰되었고(non-strict), 이 선택이 force 절대값 해석에 영향을 줄 수 있다.
3. 본 분석은 Van Wouwe 2021의 simulation 기반 인과 프레임을 1:1 재현한 결과가 아니다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
conda run --no-capture-output -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`, `data/all_data/*.c3d`, `src/replace_v3d/torque/assets/fp_inertial_templates.npz`
- Output: 콘솔 통계 결과(유의 변수만 표시), `report.md`

## Figures

- 이번 분석은 사용자 요청에 따라 **figure를 생성하지 않음**.
