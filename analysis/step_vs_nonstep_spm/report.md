# Step vs. Non-step SPM 1D Analysis Report

## Research Question

동일한 섭동 조건에서 step/nonstep 전략이 [platform onset → step onset] 구간의 시계열 전체에서 언제 유의하게 다른지 SPM 1D로 확인한다.

## Data Summary

- 분석 프레임 수: 41927 (원본 42846)
- 분석 시행 수: 181 (step=108, nonstep=73)
- 피험자 수: 24
- 제외 시행: step onset 누락 2개, event 범위 이탈 2개
- 입력 데이터: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
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

## Interpretation

유의 구간은 시간 정규화 축(0-100%)에서 step/nonstep 차이가 집중되는 시점을 의미한다.
Parametric과 nonparametric 결과가 동시에 유의한 변수는 분포 가정에 덜 민감한 차이로 해석할 수 있다.
단, 본 분석은 mixed==1 시행과 paired subject만 포함하므로 일반화 시 포함 기준을 함께 제시해야 한다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_spm/analyze_step_vs_nonstep_spm.py
```

## Output Files

- `analysis/step_vs_nonstep_spm/spm_results.csv`
- `analysis/step_vs_nonstep_spm/figures/spm_<variable>.png`
- `analysis/step_vs_nonstep_spm/figures/heatmap_significant.png`
- `analysis/step_vs_nonstep_spm/report.md`

