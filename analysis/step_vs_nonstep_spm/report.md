# Step vs. Non-step SPM 1D Analysis Report

## Research Question

동일한 섭동 조건에서 step/nonstep 전략이 [platform onset → step onset] 구간의 시계열 전체에서 언제 유의하게 다른지 SPM 1D로 확인한다.

## Data Summary

- 분석 프레임 수: 29005 (원본 29249)
- 분석 시행 수: 125 (step=52, nonstep=73)
- 피험자 수: 24
- 제외 시행: step onset 누락 1개, event 범위 이탈 0개
- 입력 데이터: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- 전처리 필터(`scripts/apply_post_filter_from_meta.py`): mixed==1, age_group==young, ipsilateral step only
- 분석 변수 수: 83

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
| Joint angular velocity (`*_ref_*_deg_s`, `*_mov_*_deg_s`) | 각속도 축 성분의 원 부호 유지 | Hip/Knee/Ankle은 stance side로 변환 후 비교 |
| Segment moment (`*_ref_*_Nm`) | internal moment 축 성분의 원 부호 유지 | Hip/Knee/Ankle은 stance side로 변환 후 비교 |
| GRF_* / GRM_* | force/torque 원시 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |
| COP_* | COP 절대 좌표 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |
| AnkleTorqueMid_int_Y_Nm_per_kg | internal torque 부호 유지 | 체중 정규화 값 사용 |

## Results

- Parametric 유의 변수: 14 / 83
- Nonparametric 유의 변수: 15 / 83

### Family-level Summary

| Family | Variables | Param Sig | Nonparam Sig |
|--------|-----------|-----------|--------------|
| AnkleTorque | 1 | 0 | 0 |
| BOS | 5 | 0 | 0 |
| COM | 3 | 1 | 1 |
| COP | 2 | 1 | 1 |
| GRF | 3 | 0 | 0 |
| GRM | 3 | 0 | 0 |
| JointVelocity | 30 | 2 | 3 |
| MOS | 4 | 3 | 3 |
| Neck | 3 | 0 | 0 |
| SegmentMoment | 15 | 4 | 4 |
| StanceJoint | 3 | 0 | 0 |
| Trunk | 3 | 0 | 0 |
| vCOM | 3 | 1 | 1 |
| xCOM | 3 | 1 | 1 |
| xCOM_BOS | 2 | 1 | 1 |

### Significant Variable Summary (Sig only)

| Variable | Family | N_pairs | Param interval (%) | Nonparam interval (%) | Direction | Mean diff |
|----------|--------|---------|--------------------|-----------------------|-----------|-----------|
| COM_X | COM | 21 | 71.1-100.0 | 71.3-100.0 | step < nonstep | -0.0137 |
| COP_Y_m | COP | 21 | 36.7-38.6, 95.9-100.0 | 36.4-39.3, 93.8-100.0 | direction changes | 0.0270 |
| Ankle_stance_mov_Y_deg_s | JointVelocity | 21 | - | 12.9-14.1, 17.5-18.7 | step < nonstep | -6.2989 |
| Hip_stance_mov_X_deg_s | JointVelocity | 21 | 82.2-100.0 | 79.3-100.0 | step > nonstep | 15.0279 |
| Hip_stance_ref_X_deg_s | JointVelocity | 21 | 83.8-99.6 | 81.0-100.0 | step > nonstep | 15.0461 |
| MOS_AP_v3d | MOS | 21 | 0.0-11.9, 62.6-100.0 | 0.0-12.5, 60.5-100.0 | direction changes | -0.0140 |
| MOS_minDist_signed | MOS | 21 | 0.0-13.2, 64.9-100.0 | 0.0-13.3, 64.1-100.0 | direction changes | -0.0133 |
| MOS_v3d | MOS | 21 | 0.0-11.9, 62.6-100.0 | 0.0-12.9, 59.1-100.0 | direction changes | -0.0140 |
| Ankle_stance_ref_X_Nm | SegmentMoment | 21 | 98.2-100.0 | 97.5-100.0 | step < nonstep | -32.0446 |
| Ankle_stance_ref_Y_Nm | SegmentMoment | 21 | 96.0-97.4 | 95.5-98.4 | step < nonstep | -33.5853 |
| Hip_stance_ref_Y_Nm | SegmentMoment | 21 | 94.5-100.0 | 94.0-100.0 | step < nonstep | -63.3652 |
| Knee_stance_ref_Z_Nm | SegmentMoment | 21 | 93.3-100.0 | 91.9-100.0 | step > nonstep | 13.4456 |
| vCOM_X | vCOM | 21 | 44.5-100.0 | 43.3-100.0 | step < nonstep | -0.0314 |
| xCOM_X | xCOM | 21 | 51.7-100.0 | 50.1-100.0 | step < nonstep | -0.0216 |
| xCOM_BOS_AP_foot | xCOM_BOS | 21 | 0.0-100.0 | 0.0-100.0 | step < nonstep | -0.0734 |

### Cross-check with prior LMM focus variables

| Variable | SPM status |
|----------|------------|
| xCOM_BOS_AP_foot | param sig, nonparam sig |
| xCOM_BOS_ML_foot | param n.s., nonparam n.s. |
| Hip_stance_X_deg | param failed: Zero variance detected at the following nodes:, nonparam n.s. |

### Test Execution Notes

| Variable | Parametric status | Nonparametric status |
|----------|-------------------|----------------------|
| AnkleTorqueMid_int_Y_Nm_per_kg | failed: Zero variance detected at the following nodes: | n.s. |
| Ankle_stance_X_deg | failed: Zero variance detected at the following nodes: | n.s. |
| GRF_X_N | failed: Zero variance detected at the following nodes: | n.s. |
| GRF_Y_N | failed: Zero variance detected at the following nodes: | n.s. |
| GRF_Z_N | failed: Zero variance detected at the following nodes: | n.s. |
| GRM_X_Nm_at_FPorigin | failed: Zero variance detected at the following nodes: | n.s. |
| GRM_Y_Nm_at_FPorigin | failed: Zero variance detected at the following nodes: | n.s. |
| GRM_Z_Nm_at_FPorigin | failed: Zero variance detected at the following nodes: | n.s. |
| Hip_stance_X_deg | failed: Zero variance detected at the following nodes: | n.s. |
| Knee_stance_X_deg | failed: Zero variance detected at the following nodes: | n.s. |
| Neck_X_deg | failed: Zero variance detected at the following nodes: | n.s. |
| Neck_Y_deg | failed: Zero variance detected at the following nodes: | n.s. |
| Neck_Z_deg | failed: Zero variance detected at the following nodes: | n.s. |
| Trunk_X_deg | failed: Zero variance detected at the following nodes: | n.s. |
| Trunk_Y_deg | failed: Zero variance detected at the following nodes: | n.s. |
| Trunk_Z_deg | failed: Zero variance detected at the following nodes: | n.s. |

## Discussion

### 결과 해석

유의 구간은 MOS 계열의 초기/후기 분리, vCOM_X의 초기+후기 두 구간, COM_Z의 초기/중기 구간, xCOM_X·COM_X의 중후기 구간, xCOM_BOS_AP_foot의 전 구간 유의처럼 변수별로 다른 시간 패턴을 보였다.

방향성은 자동으로 계산한 `step - nonstep` 평균 차이를 기준으로 확인하였다. COM_X, vCOM_X, xCOM_X, xCOM_BOS_AP_foot는 주된 유의 구간에서 `step < nonstep`이었고, COM_Z, xCOM_Z, COP_X_m은 `step > nonstep`이었다. MOS 계열은 초기에는 `step > nonstep`, 후기에는 `step < nonstep`으로 바뀌어 한 방향의 차이로 요약되지 않았다.

따라서 이 SPM 결과만으로 `step` 전략이 전 구간에서 더 전방으로 이동한다거나, BOS 경계를 지속적으로 넘는다고 단정할 수는 없다. 방향성 해석은 변수와 시간 구간별로 나누어 읽어야 하며, 기전 설명은 평균 곡선이나 추가 분석과 함께 제시하는 것이 안전하다.

### Direction Check

| Variable | Source | Direction | Mean diff |
|----------|--------|-----------|-----------|
| Ankle_stance_mov_Y_deg_s | nonparam | step < nonstep | -6.2989 |
| Ankle_stance_ref_X_Nm | param | step < nonstep | -32.0446 |
| Ankle_stance_ref_Y_Nm | param | step < nonstep | -33.5853 |
| COM_X | param | step < nonstep | -0.0137 |
| COP_Y_m | param | direction changes | 0.0270 |
| Hip_stance_mov_X_deg_s | param | step > nonstep | 15.0279 |
| Hip_stance_ref_X_deg_s | param | step > nonstep | 15.0461 |
| Hip_stance_ref_Y_Nm | param | step < nonstep | -63.3652 |
| Knee_stance_ref_Z_Nm | param | step > nonstep | 13.4456 |
| MOS_AP_v3d | param | direction changes | -0.0140 |
| MOS_minDist_signed | param | direction changes | -0.0133 |
| MOS_v3d | param | direction changes | -0.0140 |
| vCOM_X | param | step < nonstep | -0.0314 |
| xCOM_BOS_AP_foot | param | step < nonstep | -0.0734 |
| xCOM_X | param | step < nonstep | -0.0216 |

## Conclusion

1. Step/nonstep 차이는 변수마다 다른 시간 구간에서 나타났고, 일부 변수는 두 개 이상의 분리된 유의 구간을 보였다.
2. xCOM_BOS_AP_foot는 전 구간 유의하여 가장 일관된 구분 지표였지만, 방향은 `step < nonstep`으로 요약되었다.
3. MOS 계열은 초기와 후기의 방향이 바뀌어, 동일 변수라도 시간 구간별 해석이 필요했다.
4. Parametric SPM이 zero variance로 실패한 변수들은 음성 결과가 아니라 미검정 항목으로 해석해야 한다.

## Limitations

- 본 분석은 young, mixed==1, ipsilateral step 시행만 포함하므로, 고령자·contralateral step 등으로 일반화 시 주의가 필요하다.
- paired t-test 구조상 피험자 내 step/nonstep 시행이 모두 존재하는 경우만 분석되어, 한 전략만 사용하는 피험자는 제외되었다.
- 여러 변수에서 parametric SPM이 zero variance로 실패했으므로, 해당 변수의 모수 결과는 미검정으로 남는다.
- 비모수 순열 검정과 모수 검정이 모두 성공한 유의 변수에서는 두 검정의 유의 여부가 일치하였다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_spm/analyze_step_vs_nonstep_spm.py
```

## Output Files

- `analysis/step_vs_nonstep_spm/spm_results.csv`
- `analysis/step_vs_nonstep_spm/figures/spm_<variable>.png`
- `analysis/step_vs_nonstep_spm/figures/heatmap_significant.png`
- `analysis/step_vs_nonstep_spm/report.md`

