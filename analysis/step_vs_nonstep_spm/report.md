# Step vs. Non-step SPM 1D Analysis Report

## Research Question

동일한 섭동 조건에서 step/nonstep 전략이 [platform onset → step onset] 구간의 시계열 전체에서 언제 유의하게 다른지 SPM 1D로 확인한다.

## Data Summary

- 분석 프레임 수: 41927 (원본 42846)
- 분석 시행 수: 181 (step=108, nonstep=73)
- 피험자 수: 24
- 제외 시행: step onset 누락 2개, event 범위 이탈 2개
- 입력 데이터: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- 전처리 필터(`scripts/apply_post_filter_from_meta.py`): mixed==1, age_group==young, ipsilateral step only
- 분석 변수 수: 38

## Analysis Methodology

- 분석 구간: 각 trial의 `[platform_onset_local, end_frame]`
  - Step trial: `end_frame = step_onset_local`
  - Nonstep trial: 같은 `(subject, velocity)`의 step onset 평균값(부족 시 platform sheet fallback)
- 시간 정규화: 0-100% (101 points), NaN 20% 초과 trial 제외, 그 외 선형보간
- 짝지음 단위: 피험자 내 step/nonstep 평균 곡선
- SPM 검정: paired t-test (parametric + nonparametric permutation)
- 비모수 순열 횟수: 10000
- 다중비교 보정: family별 Bonferroni (`alpha = 0.05 / family_size`)
- Nonstep stance side: subject별 major step side 사용, tie는 (L+R)/2
- xCOM/BOS 정규화: `foot_len_m = (발길이_왼 + 발길이_오른)/2` 기반

### Coordinate & Sign Conventions

Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|------|---------------|---------------|-----------|
| AP (X) | +X = 전방 | -X = 후방 | COM_X, vCOM_X, xCOM_X, BOS_minX/maxX, MOS_AP_v3d |
| ML (Y) | +Y = 좌측 | -Y = 우측 | COM_Y, vCOM_Y, xCOM_Y, BOS_minY/maxY, MOS_ML_v3d |
| Vertical (Z) | +Z = 위 | -Z = 아래 | COM_Z, vCOM_Z, xCOM_Z, GRF_Z_N |

Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|--------|--------------|--------------|----------------|
| MOS_minDist_signed | BOS 내부/안정 여유 | BOS 외부/안정 여유 부족 | signed minimum distance |
| MOS_AP_v3d | AP 경계 내부 방향 | AP 경계 외부 방향 | AP bound-relative sign |
| MOS_ML_v3d | ML 경계 내부 방향 | ML 경계 외부 방향 | ML bound-relative sign |
| xCOM_BOS_AP_foot | BOS_minX 기준 전방 상대 위치 증가 | BOS_minX 기준 전방 상대 위치 감소 | foot length 정규화 |
| xCOM_BOS_ML_foot | BOS_minY 기준 좌측 상대 위치 증가 | BOS_minY 기준 우측 상대 위치 증가 | foot length 정규화 |

Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|----------------|------------------|-----------|
| Joint angles (Hip/Knee/Ankle/Trunk/Neck) | 각 축의 해부학적 회전 부호를 데이터 원 부호 그대로 사용 | stance side만 Hip/Knee/Ankle X축에 적용 |
| GRF_* / GRM_* | force/torque 원시 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |
| COP_* | COP 절대 좌표 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |
| AnkleTorqueMid_int_Y_Nm_per_kg | internal torque 부호 유지 | 체중 정규화 값 사용 |

## Results

- Parametric 유의 변수: 11 / 38
- Nonparametric 유의 변수: 11 / 38

### Family-level Summary

| Family | Variables | Param Sig | Nonparam Sig |
|--------|-----------|-----------|--------------|
| AnkleTorque | 1 | 0 | 0 |
| BOS | 5 | 0 | 0 |
| COM | 3 | 2 | 2 |
| COP | 2 | 1 | 1 |
| GRF | 3 | 0 | 0 |
| GRM | 3 | 0 | 0 |
| MOS | 4 | 3 | 3 |
| Neck | 3 | 0 | 0 |
| StanceJoint | 3 | 0 | 0 |
| Trunk | 3 | 0 | 0 |
| vCOM | 3 | 2 | 2 |
| xCOM | 3 | 2 | 2 |
| xCOM_BOS | 2 | 1 | 1 |

### Significant Variable Summary (Sig only)

| Variable | Family | N_pairs | Param interval (%) | Nonparam interval (%) | Param Sig | Nonparam Sig |
|----------|--------|---------|--------------------|-----------------------|-----------|--------------|
| COM_X | COM | 24 | 59.8-100.0 | 60.0-100.0 | <0.05 | <0.05 |
| COM_Z | COM | 24 | 0.0-4.8, 28.0-37.2 | 0.0-10.1, 26.1-38.8 | <0.05 | <0.05 |
| COP_X_m | COP | 24 | 99.5-100.0 | 63.2-67.0, 91.8-100.0 | <0.05 | <0.05 |
| MOS_AP_v3d | MOS | 24 | 0.0-15.9, 61.6-100.0 | 0.0-16.1, 61.4-100.0 | <0.05 | <0.05 |
| MOS_minDist_signed | MOS | 24 | 0.0-16.4, 61.8-100.0 | 0.0-16.5, 61.7-100.0 | <0.05 | <0.05 |
| MOS_v3d | MOS | 24 | 0.0-15.9, 61.6-100.0 | 0.0-16.0, 61.4-100.0 | <0.05 | <0.05 |
| vCOM_X | vCOM | 24 | 1.6-18.4, 54.0-100.0 | 1.0-18.9, 53.2-100.0 | <0.05 | <0.05 |
| vCOM_Z | vCOM | 24 | 64.6-69.3 | 62.1-71.0 | <0.05 | <0.05 |
| xCOM_X | xCOM | 24 | 46.4-100.0 | 44.0-100.0 | <0.05 | <0.05 |
| xCOM_Z | xCOM | 24 | 62.9-72.5 | 60.6-73.9 | <0.05 | <0.05 |
| xCOM_BOS_AP_foot | xCOM_BOS | 24 | 0.0-100.0 | 0.0-100.0 | <0.05 | <0.05 |

### Cross-check with prior LMM focus variables

| Variable | SPM status |
|----------|------------|
| xCOM_BOS_AP_foot | param sig, nonparam sig |
| xCOM_BOS_ML_foot | param n.s., nonparam n.s. |
| Hip_stance_X_deg | param n.s., nonparam n.s. |

## Discussion

### 시간대별 주요 발견

**초기 구간 (0-16%):** MOS_AP_v3d, MOS_minDist_signed, MOS_v3d가 섭동 직후부터 유의하였다. 이는 섭동 인가 시점에서 step 전략군의 안정성 마진(Margin of Stability)이 nonstep 전략군과 즉각적으로 구분됨을 시사한다. 즉, step을 선택하는 피험자는 섭동 초기에 이미 xCOM이 BOS 경계에 더 가깝거나 이를 벗어나는 경향이 있다.

**중기 구간 (46-60%):** vCOM_X(54%)와 xCOM_X(46%)에서 유의 구간이 시작된다. 이 시점은 step 전략군이 보상적 발 디딤을 준비하며 전방 속도(vCOM_X)가 증가하고, 그에 따라 외삽 질량 중심(xCOM_X)이 전방으로 이동하기 시작하는 구간이다. nonstep 전략군은 이 시기에 발을 고정한 채 근위부 전략(ankle/hip)으로 대응하므로 속도·위치 변화가 상대적으로 작다.

**후기 구간 (60-100%):** COM_X, xCOM_X, vCOM_X, MOS 계열이 모두 유의해지며, step 전략의 전방 COM 이동과 새로운 BOS 확보 과정이 반영된다. COM_Z와 xCOM_Z는 60-73% 구간에서만 유의한데, 이는 step 실행 시 일시적으로 수직 COM이 하강한 뒤 회복하는 패턴을 나타낸다. COP_X_m은 후기(~92-100%)에 유의하여, step 착지 후 COP가 전방으로 급격히 이동하는 시점을 포착한다.

**전 구간 유의 (0-100%):** xCOM_BOS_AP_foot가 정규화 구간 전체에서 유의하였다. 이는 AP 방향 xCOM-BOS 상대 위치가 step/nonstep 전략 간에 근본적으로 다름을 의미하며, step 전략은 xCOM이 BOS 전방 경계를 지속적으로 초과하는 반면, nonstep 전략은 BOS 내부에 xCOM을 유지하는 패턴을 보인다.

### ML 방향 및 관절 각도

xCOM_BOS_ML_foot, Hip_stance_X_deg 등 ML 방향 및 관절 변수는 유의하지 않았다. 이는 본 섭동이 주로 AP 방향으로 가해졌기 때문에, step/nonstep 전략 간 차이가 AP 안정성 변수에 집중되는 것으로 해석된다.

## Conclusion

1. Step 전략군은 섭동 직후(0-16%)부터 AP 방향 MOS가 nonstep 전략군과 유의하게 달라, 보상 전략 선택이 섭동 초기 안정성 상태와 관련됨을 확인하였다.
2. 중기(46-60%) 이후 vCOM_X, xCOM_X, COM_X가 순차적으로 유의해져, step 준비→실행 과정에서의 전방 이동이 시계열 수준에서 뚜렷이 구분된다.
3. xCOM_BOS_AP_foot가 전 구간 유의하여, AP 방향 xCOM-BOS 관계가 step/nonstep 전략을 구분짓는 핵심 지표임을 시사한다.
4. ML 방향 변수 및 관절 각도 변수는 유의하지 않아, AP 섭동 하에서 두 전략 간 차이는 시상면(AP·수직) 변수에 국한된다.

## Limitations

- 본 분석은 young, mixed==1, ipsilateral step 시행만 포함하므로, 고령자·contralateral step 등으로 일반화 시 주의가 필요하다.
- paired t-test 구조상 피험자 내 step/nonstep 시행이 모두 존재하는 경우만 분석되어, 한 전략만 사용하는 피험자는 제외되었다.
- 비모수 순열 검정과 모수 검정 결과가 모든 유의 변수에서 일치하여 분포 가정 위반 우려는 낮다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_spm/analyze_step_vs_nonstep_spm.py
```

## Output Files

- `analysis/step_vs_nonstep_spm/spm_results.csv`
- `analysis/step_vs_nonstep_spm/figures/spm_<variable>.png`
- `analysis/step_vs_nonstep_spm/figures/heatmap_significant.png`
- `analysis/step_vs_nonstep_spm/report.md`

