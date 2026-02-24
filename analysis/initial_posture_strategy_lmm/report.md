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
| 초기자세가 전략 variability를 설명 | 초기 COM 위치가 trunk lean과 subject-specific하게 연관 (`R^2=0.29–0.82`) | onset 변수 19개 중 5개만 유의; 관절 각도 전부 비유의 | Weakly consistent |
| `xCOM/BOS_onset` 중요성 | onset 안정성 지표로 사용 | `xCOM_BOS_norm_onset` 유의 (`step < nonstep`), 단 `COM_X` 비유의 → BOS 정규화 기여 의심 | Conditionally consistent |
| trunk 전략 지표 | trunk lean variability 핵심 | `Trunk_X_abs_onset`는 비유의 | Inconsistent |
| force 계열 초기값 영향 | 원문에서 간접적으로 전략 반응과 연계 | absolute force 중 `AnkleTorqueMid_Y_perkg_abs_onset` 유의 | Partially consistent |
| 결론 수준 | 초기자세 + task-level goal 상호작용 강조 | goal 파라미터 직접 모델링 없음 | Inconsistent (scope mismatch) |

## Interpretation & Conclusion

### 1. 엄격 기준 결과

각도와 force를 absolute onset으로 전환해도, `5/19`만 FDR 유의하여 "onset에서 광범위한 step/nonstep 차이"라는 가설은 **FAIL**이다.

### 2. 관절 각도 비유의 vs COM 파생 유의 — 생체역학적 비일관성

본 분석의 **가장 큰 해석적 문제**는, COM 파생 변수(`xCOM_BOS_norm_onset`, `MOS_AP_v3d`, `MOS_minDist_signed`, `vCOM_X`)는 유의한데, 그 원천(source)인 **관절 각도 5개가 전부 비유의**라는 점이다.

COM 위치는 관절 각도의 함수이다. 각 segment의 위치는 관절 각도 chain으로 결정되고, COM은 각 segment 위치의 mass-weighted average이다:

```
관절 각도 (ankle, knee, hip, trunk, neck)
  → segment positions
    → COM = Σ(m_i × p_i) / Σ(m_i)
      → xCOM = COM_X + vCOM_X / sqrt(g/l)
        → xCOM_BOS_norm = (xCOM_X - BOS_minX) / (BOS_maxX - BOS_minX)
```

이 chain에서 입력(관절 각도) 전부가 step/nonstep 간 차이가 없다면, 출력(COM)에서 유의한 차이가 나타나는 것은 논리적으로 설명이 필요하다. 가능한 경로는 다음과 같다:

| 가능한 설명 | 평가 |
|---|---|
| 개별 관절의 미세 차이가 누적되어 COM에서 유의해짐 | 5개 관절 **모두** n.s.이므로, 누적으로도 유의한 차이를 만들기 어려움 |
| segment mass 분포의 개인차가 COM에 영향 | inter-subject 차이이며, LMM의 `(1\|subject)` random intercept에 이미 흡수됨 |
| 관절 각속도 차이가 vCOM_X를 통해 xCOM에 기여 | 각속도는 검정하지 않아 배제 불가. 단, `vCOM_X` 단독으로는 전체 유의 패턴을 설명하기 부족 |
| **BOS 정규화(분모)가 유의성을 주도** | 가장 유력한 설명. 아래 상세 |

### 3. xCOM_BOS_norm_onset 유의성의 원천 분석

`xCOM_BOS_norm_onset`은 `(xCOM_X - BOS_minX) / (BOS_maxX - BOS_minX)`로 계산된다.

- **분자**: `xCOM_X - BOS_minX`. `COM_X` 자체가 n.s.이므로, xCOM의 유의성은 주로 `vCOM_X` 기여분(`vCOM_X / sqrt(g/l)`)과 `BOS_minX`(뒤꿈치 위치) 차이에 의존한다.
- **분모**: `BOS_maxX - BOS_minX` (발 길이에 해당하는 BOS 범위). step/nonstep 간 **발 위치(foot placement)**가 다르면, 분모가 달라져 정규화 결과가 유의해질 수 있다.

즉, `xCOM_BOS_norm_onset`의 유의성이 **신체 자세(posture)의 실질적 차이**를 반영하는 것인지, 아니면 **발 위치/BOS 범위의 차이**가 정규화를 통해 만들어낸 산물인지 분리되지 않는다. `MOS_AP_v3d`와 `MOS_minDist_signed`도 BOS polygon에 의존하므로, 같은 문제가 적용된다.

### 4. AnkleTorqueMid_Y_perkg_abs_onset의 해석

유일하게 유의한 force 변수인 ankle torque는 onset에서 step trial의 plantarflexion torque가 nonstep보다 약간 작다(절대값 기준). 이것은 onset 전 quiet standing 중 ankle stiffness 또는 weight distribution의 미세한 차이를 반영할 수 있으나, 관절 각도(`Ankle_stance_X_abs_onset`)가 비유의인 상태에서 torque만 유의한 것은 **co-contraction 또는 muscle tone 차이**를 시사할 가능성이 있다. 다만 단일 변수로 강한 해석을 내리기는 어렵다.

### 5. 종합 결론

**onset 시점에서 step/nonstep 간 신체 자세(posture)의 실질적 차이를 확인하기 어렵다.** 관절 각도 5개 (hip, knee, ankle, trunk, neck) 전부가 비유의이므로, "초기 자세가 다르기 때문에 전략이 달라진다"는 Van Wouwe식 해석은 본 데이터에서 **강하게 지지되지 않는다.**

유의한 COM 파생 변수(`xCOM_BOS_norm_onset`, `MOS_AP_v3d`, `MOS_minDist_signed`)는 BOS 정규화에 의존하며, 그 원천인 관절 각도에서 차이가 없으므로, **발 위치(foot placement) 또는 BOS 기하학의 차이**가 유의성을 만들어냈을 가능성을 배제할 수 없다. `vCOM_X`의 약한 유의성(**)은 onset 직전의 미세한 동적 상태 차이를 시사하지만, 이것만으로 "초기 자세 → 전략 결정"이라는 인과적 해석을 뒷받침하기에는 불충분하다.

Van Wouwe (2021)의 원문은 **개인 내(within-subject) 연속 상관**(COM 위치 → trunk lean, R²=0.29–0.82)과 **시뮬레이션 기반 인과 추론**으로 가설을 지지했으나, 본 분석은 **집단 간 평균 비교** 프레임이므로 직접적 대조에 한계가 있다. 그럼에도, 관절 각도 전부 비유의라는 결과는 본 데이터에서 "초기 자세의 전략 예측력"이 제한적임을 보여준다.

## Limitations

1. 원문의 task-level goal 파라미터를 직접 모델링하지 않았다.
2. inertial subtraction QC 경고가 4 trial에서 관찰되었고(non-strict), 이 선택이 force 절대값 해석에 영향을 줄 수 있다.
3. 본 분석은 Van Wouwe 2021의 simulation 기반 인과 프레임을 1:1 재현한 결과가 아니다.
4. 관절 각속도를 검정하지 않았다. vCOM은 관절 각속도의 함수이므로, 각속도 검정을 추가하면 COM 유의성의 원천을 더 명확히 분리할 수 있다.
5. BOS 변수(BOS_minX, BOS_maxX, BOS polygon area 등)를 독립적으로 검정하지 않았다. xCOM_BOS_norm과 MOS의 유의성이 자세 차이인지 BOS 기하학 차이인지 분리하려면 BOS 자체의 step/nonstep 비교가 필요하다.
6. 원문은 개인 내 연속 상관(within-subject correlation) 분석인 반면, 본 분석은 집단 평균 비교(LMM with binary predictor)이므로 분석 프레임이 근본적으로 다르다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
conda run --no-capture-output -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`, `data/all_data/*.c3d`, `src/replace_v3d/torque/assets/fp_inertial_templates.npz`
- Output: 콘솔 통계 결과(유의 변수만 표시), `report.md`

## Figures

- 이번 분석은 사용자 요청에 따라 **figure를 생성하지 않음**.
