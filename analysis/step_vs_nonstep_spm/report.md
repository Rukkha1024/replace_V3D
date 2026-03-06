# Step vs. Non-step SPM 1D Analysis Report

## Research Question

동일한 섭동 조건에서 step/nonstep 전략이 [platform onset → step onset] 구간의 시계열 전체에서 언제 유의하게 다른지 SPM 1D로 확인한다.

## Data Summary

- 분석 프레임 수: 41927 (원본 42846)
- 분석 시행 수: 181 (step=108, nonstep=73)
- 피험자 수: 24
- 제외 시행: step onset 누락 2개, event 범위 이탈 2개
- 입력 데이터: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- 전처리 필터(`scripts/apply_post_filter_from_meta.py`): `mixed==1`, `age_group==young`, `ipsilateral step only`
- 분석 변수 수: 38

## Analysis Methodology

- 분석 구간: 각 trial의 `[platform_onset_local, end_frame]`
  - Step trial: `end_frame = step_onset_local`
  - Nonstep trial: 같은 `(subject, velocity)`의 step onset 평균값(부족 시 platform sheet fallback)
- 시간 정규화: 0-100% (101 points), NaN 20% 초과 trial 제외, 그 외 선형보간
- 정규화 구간 역추적: 각 trial에서 `absolute frame = platform_onset_local + (pct/100) * (end_frame - platform_onset_local)`로 환산할 수 있다. 모캡 100 Hz 기준 `1 frame = 10 ms`이므로 `time_from_platform_onset_s`로 다시 표현 가능하다.
- 절대시점 해석 주의: `%`는 공통 축이지만 absolute time은 trial별 window 길이에 따라 달라진다. 아래 대표 absolute time은 분석에 포함된 trial의 window 길이 중앙값 0.460 s(IQR 0.330-0.610 s)를 기준으로 환산했다.
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

| Variable | Family | N_pairs | Param interval (%) | Nonparam interval (%) | Direction | Mean diff |
|----------|--------|---------|--------------------|-----------------------|-----------|-----------|
| COM_X | COM | 24 | 59.8-100.0 | 60.0-100.0 | step < nonstep | -0.0112 |
| COM_Z | COM | 24 | 0.0-4.8, 28.0-37.2 | 0.0-10.1, 26.1-38.8 | step > nonstep | 0.0011 |
| COP_X_m | COP | 24 | 99.5-100.0 | 63.2-67.0, 91.8-100.0 | step > nonstep | 0.0079 |
| MOS_AP_v3d | MOS | 24 | 0.0-15.9, 61.6-100.0 | 0.0-16.1, 61.4-100.0 | direction changes | -0.0114 |
| MOS_minDist_signed | MOS | 24 | 0.0-16.4, 61.8-100.0 | 0.0-16.5, 61.7-100.0 | direction changes | -0.0111 |
| MOS_v3d | MOS | 24 | 0.0-15.9, 61.6-100.0 | 0.0-16.0, 61.4-100.0 | direction changes | -0.0114 |
| vCOM_X | vCOM | 24 | 1.6-18.4, 54.0-100.0 | 1.0-18.9, 53.2-100.0 | step < nonstep | -0.0226 |
| vCOM_Z | vCOM | 24 | 64.6-69.3 | 62.1-71.0 | step > nonstep | 0.0100 |
| xCOM_X | xCOM | 24 | 46.4-100.0 | 44.0-100.0 | step < nonstep | -0.0182 |
| xCOM_Z | xCOM | 24 | 62.9-72.5 | 60.6-73.9 | step > nonstep | 0.0043 |
| xCOM_BOS_AP_foot | xCOM_BOS | 24 | 0.0-100.0 | 0.0-100.0 | step < nonstep | -0.0664 |

### Significant Interval Timing (Representative Absolute Time)

아래 absolute time은 parametric 유의 구간을 분석 window 길이의 중앙값 0.460 s에 역매핑한 대표값이다. Nonparametric 구간도 거의 동일했으므로, 본문 해석은 이 대표값을 기준으로 읽어도 무방하다.

| Variable | Param interval (%) | Representative time from platform onset (s) |
|----------|--------------------|---------------------------------------------|
| COM_X | 59.8-100.0 | 0.275-0.460 |
| COM_Z | 0.0-4.8, 28.0-37.2 | 0.000-0.022, 0.129-0.171 |
| COP_X_m | 99.5-100.0 | 0.458-0.460 |
| MOS_AP_v3d | 0.0-15.9, 61.6-100.0 | 0.000-0.073, 0.283-0.460 |
| MOS_minDist_signed | 0.0-16.4, 61.8-100.0 | 0.000-0.075, 0.284-0.460 |
| MOS_v3d | 0.0-15.9, 61.6-100.0 | 0.000-0.073, 0.283-0.460 |
| vCOM_X | 1.6-18.4, 54.0-100.0 | 0.008-0.085, 0.248-0.460 |
| vCOM_Z | 64.6-69.3 | 0.297-0.319 |
| xCOM_X | 46.4-100.0 | 0.213-0.460 |
| xCOM_Z | 62.9-72.5 | 0.289-0.334 |
| xCOM_BOS_AP_foot | 0.0-100.0 | 0.000-0.460 |

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

이를 대표 absolute time으로 다시 쓰면, `COM_X`는 platform onset 후 약 0.275-0.460 s에서, `vCOM_X`는 0.008-0.085 s와 0.248-0.460 s에서, `xCOM_X`는 0.213-0.460 s에서 유의했다. `xCOM_BOS_AP_foot`는 0.000-0.460 s 전 구간에서 유의했고, MOS 계열(`MOS_AP_v3d`, `MOS_minDist_signed`, `MOS_v3d`)은 공통적으로 초기 약 0.000-0.073/0.075 s와 후기 약 0.283/0.284-0.460 s에서 유의했다.

`COM_Z`는 onset 직후 0.000-0.022 s와 중기 0.129-0.171 s에서, `vCOM_Z`는 0.297-0.319 s에서, `xCOM_Z`는 0.289-0.334 s에서, `COP_X_m`는 거의 종점인 0.458-0.460 s에서 차이가 나타났다. 따라서 결과의 핵심은 단순히 "유의했다"가 아니라, 어떤 변수는 onset 직후에만, 어떤 변수는 중후기에만, 어떤 변수는 전 구간에서 전략 차이가 분리되었다는 점이다.

방향성은 자동으로 계산한 `step - nonstep` 평균 차이를 기준으로 확인하였다. COM_X, vCOM_X, xCOM_X, xCOM_BOS_AP_foot는 주된 유의 구간에서 `step < nonstep`이었고, COM_Z, xCOM_Z, COP_X_m은 `step > nonstep`이었다. MOS 계열은 초기에는 `step > nonstep`, 후기에는 `step < nonstep`으로 바뀌어 한 방향의 차이로 요약되지 않았다.

이 패턴은 `step` 전략의 차이가 반응 전 구간에 균일하게 나타난다기보다, step onset에 가까워질수록 AP 안정성 관련 변수에서 더 뚜렷하게 분리된다는 점을 보여준다. 특히 `xCOM_BOS_AP_foot`가 0-100% 전 구간에서 `step < nonstep`이었던 결과는, 동일한 perturbation 강도에서도 step trial이 nonstep보다 BOS 후방 경계에 더 가까운 상태로 회복 과정을 진행했음을 시사한다. 이는 step trial에서 전후방 안정 여유가 처음부터 끝까지 더 작았고, 고정 지지 전략만으로는 회복을 마무리하기 어려운 조건이 지속되었음을 의미한다.

같은 해석은 `COM_X`, `vCOM_X`, `xCOM_X`의 중후기 유의 구간과도 일치한다. 세 변수 모두 주로 후반부에서 `step < nonstep`이었으므로, 두 전략의 분리는 섭동 직후의 자동 반응보다 step 준비 구간에서 더 크게 나타난 것으로 해석할 수 있다. 즉 nonstep은 BOS 안에서 COM/xCOM을 다시 제어하는 방향으로 수렴한 반면, step은 후반부로 갈수록 BOS 내부 안정 여유를 유지하지 못하고 보상 스텝이 필요한 상태로 진행되었다고 볼 수 있다.

한편 MOS 계열은 초기에는 `step > nonstep`, 후기에는 `step < nonstep`으로 역전되었다. 이는 step trial이 섭동 직후에는 일시적으로 AP 안정 여유를 확보하는 보정을 보이더라도, 그 차이가 step onset 직전까지 유지되지는 않았음을 뜻한다. 다시 말해 step 전략은 초기 대응이 없는 반응이 아니라, 초기 보정 이후 후기 구간에서 안정 여유가 다시 줄어들며 최종적으로 BOS 재구성이 필요한 회복 경로로 전환된 패턴에 가깝다.

`COM_Z`, `xCOM_Z`, `COP_X_m`에서 관찰된 `step > nonstep` 결과도 이러한 해석을 보완한다. step trial에서는 step onset에 가까워질수록 COM 높이와 extrapolated COM의 수직 성분, 그리고 COP 조절이 함께 커졌으며, 이는 단순한 위치 차이만이 아니라 보상 스텝을 준비하는 체중 재배치가 동반되었을 가능성과 합치된다. 다만 이 차이만으로 전략 선택의 단일 원인을 직접 입증하는 것은 아니므로, 해석은 각 변수와 시간 구간에 한정해 읽는 것이 적절하다.

### Direction Check

| Variable | Source | Direction | Mean diff |
|----------|--------|-----------|-----------|
| COM_X | param | step < nonstep | -0.0112 |
| COM_Z | param | step > nonstep | 0.0011 |
| COP_X_m | param | step > nonstep | 0.0079 |
| MOS_AP_v3d | param | direction changes | -0.0114 |
| MOS_minDist_signed | param | direction changes | -0.0111 |
| MOS_v3d | param | direction changes | -0.0114 |
| vCOM_X | param | step < nonstep | -0.0226 |
| vCOM_Z | param | step > nonstep | 0.0100 |
| xCOM_BOS_AP_foot | param | step < nonstep | -0.0664 |
| xCOM_X | param | step < nonstep | -0.0182 |
| xCOM_Z | param | step > nonstep | 0.0043 |

## Conclusion

1. Step/nonstep 차이는 변수마다 다른 시간 구간에서 나타났고, 일부 변수는 두 개 이상의 분리된 유의 구간을 보였다.
2. `xCOM_BOS_AP_foot`는 전 구간에서 `step < nonstep`이었으며, step trial이 회복 과정 전반에서 BOS 후방 경계에 더 가까운 상태로 진행되었음을 보여주는 가장 일관된 구분 지표였다.
3. `COM_X`, `vCOM_X`, `xCOM_X`의 중후기 분리는 두 전략의 차이가 섭동 직후보다 step onset에 가까운 준비 구간에서 더 뚜렷해짐을 보여주며, step 전략이 BOS 내부 안정 여유를 끝까지 유지하지 못한 회복 경로였음을 시사했다.
4. MOS 계열은 초기의 일시적 보정 이후 후기에는 step 쪽 안정 여유가 더 작아지는 양상을 보여, step 반응이 단순한 초기 실패가 아니라 회복 과정 후반의 안정성 저하와 연결된 전략임을 나타냈다.
5. Parametric SPM이 zero variance로 실패한 변수들은 음성 결과가 아니라 미검정 항목으로 해석해야 한다.

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
