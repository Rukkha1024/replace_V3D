# Step vs. Non-step Biomechanical LMM Analysis

## Research Question

**"동일 perturbation 강도에서 step과 non-step 균형회복 전략 간 biomechanical 변수에 유의한 차이가 있는가?"**

Platform translation perturbation 실험에서 동일한 mixed velocity 조건 하에 step과 non-step이 공존하는 상황에서, COM, COP, MoS, 관절 각도, GRF, ankle torque 등 biomechanical 변수를 Linear Mixed Model(LMM)로 비교 분석하였다.

## Data Summary

- **184 trials** (step=112, nonstep=72) from 24 subjects
- Data source: `output/all_trials_timeseries.csv` (frame-level 100Hz timeseries) + `data/perturb_inform.xlsm` (step/nonstep classification)
- **32 dependent variables** across 3 variable families

### Analysis Window

분석 시간 구간은 trial마다 개별적으로 설정하였다:

- **분석 시작**: `platform_onset_local` — 각 trial에서 perturbation platform이 움직이기 시작한 MocapFrame
- **분석 종료**: `end_frame` — step onset 기반으로 trial별 산출
  - **Step trial**: 해당 trial의 실제 `step_onset_local` (발이 지면에서 처음 떨어지는 시점)
  - **Nonstep trial**: 동일 (subject, velocity) 내 step trial들의 `step_onset_local` 평균값을 대입

이 방식은 step 발생 직전까지의 CPA(Compensatory Postural Adjustment) 구간에 집중하기 위한 것이다. 고정 시간 창(예: 0–800ms)은 perturbation 강도와 피험자에 따라 step onset timing이 다르므로 적합하지 않다. Nonstep trial에 동일 (subject, velocity) 내 step trial의 평균 step onset을 부여함으로써, 두 조건 간 동일한 시간 구조에서의 비교가 가능하도록 하였다.

### Key Variables

Trial-level 집계 방식:
- **range** (max − min): COM, COP 변위 범위, GRF 범위
- **path_length** (Σ|Δ|): COM, COP 총 이동 거리
- **abs_peak** (`max(|x|)`): 분석 구간 내 절대최대값(absolute peak). `*_peak` 변수명은 모두 이 정의를 사용
- **min_val** (최소값): MoS 최소값 (가장 불안정한 순간)
- **abs_peak_velocity** (max|Δ/Δt|): COP 최대 속도

---

## Results

### 1. LMM Summary

Statistical method: Linear Mixed Model via R lmerTest
- Model: `DV ~ step_TF + (1|subject)`
- Estimation: REML (restricted maximum likelihood)
- Inference: lmerTest fixed-effect inference
- Multiple comparison: Benjamini-Hochberg FDR per variable family (`Sig` only, α=0.05)

**Overall: 13/32 variables showed FDR-significant differences (Sig at α=0.05)**

### 2. Balance/Stability Family (15 variables, 6 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|----------|-------------|-----------------|----------|-----|
| vCOM_Y_peak | 0.0587±0.0329 | 0.0390±0.0284 | 0.0232 | *** |
| COP_X_range | 0.0849±0.0145 | 0.0957±0.0133 | -0.0092 | *** |
| COP_Y_range | 0.1142±0.1082 | 0.0551±0.0353 | 0.0604 | *** |
| MOS_minDist_signed_min | 0.0373±0.0258 | 0.0485±0.0121 | -0.0119 | *** |
| MOS_AP_v3d_min | 0.0421±0.0257 | 0.0530±0.0124 | -0.0115 | *** |
| MOS_ML_v3d_min | 0.1291±0.0163 | 0.1329±0.0169 | -0.0059 | ** |
| COP_Y_path_length | 0.1888±0.2315 | 0.1215±0.0751 | 0.0742 | * |
| COM_X_range | — | — | -0.0015 | n.s. |
| COM_X_path_length | — | — | 0.0003 | n.s. |
| vCOM_X_peak | — | — | 0.0004 | n.s. |
| COM_Y_range | — | — | 0.0003 | n.s. |
| COM_Y_path_length | — | — | 0.0008 | n.s. |
| COP_X_path_length | — | — | -0.0035 | n.s. |
| COP_X_peak_velocity | — | — | 0.1541 | n.s. |
| COP_Y_peak_velocity | — | — | 1.6913 | n.s. |

### 3. Joint Angles Family (10 variables, 5 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|----------|-------------|-----------------|----------|-----|
| Hip_R_ROM | 8.8704±3.5363 | 8.2460±4.1579 | 1.5529 | *** |
| Hip_R_peak | 8.6901±3.4595 | 7.6392±3.0939 | 1.7377 | *** |
| Knee_R_ROM | 11.5689±6.4566 | 10.6368±5.2064 | 2.2142 | *** |
| Knee_R_peak | 10.6882±6.2259 | 9.8288±4.9566 | 2.0362 | *** |
| Neck_ROM | 10.2490±10.5563 | 10.6053±9.2047 | 2.1557 | * |
| Neck_peak | 9.9506±10.5394 | 10.2947±9.2522 | 2.1528 | * |
| Ankle_R_ROM | — | — | 0.6524 | n.s. |
| Ankle_R_peak | — | — | 0.5597 | n.s. |
| Trunk_ROM | — | — | 0.8867 | n.s. |
| Trunk_peak | — | — | 0.7644 | n.s. |

### 4. Force/Torque Family (7 variables, 0 significant)

| Variable | Estimate | Sig |
|----------|----------|-----|
| GRF_X_peak | -2.6569 | n.s. |
| GRF_X_range | -3.8233 | n.s. |
| GRF_Y_peak | 1.8912 | n.s. |
| GRF_Y_range | -6.0032 | n.s. |
| GRF_Z_peak | 11.1985 | n.s. |
| GRF_Z_range | 7.8412 | n.s. |
| AnkleTorqueMid_Y_peak | -0.0238 | n.s. |

---

## Interpretation

### Balance & Stability

platform onset ~ step onset 구간에서 step 시행은 ML(좌우) 방향 COM 속도(vCOM_Y_peak)와 COP 변위(COP_Y_range, COP_Y_path_length)가 유의하게 크다. 이는 stepping 직전까지의 CPA 구간에서 ML 방향 자세 동요가 step 유발의 핵심 특징임을 시사한다.

COP_X_range는 **nonstep에서 유의하게 크다** (estimate = -0.0092). 이는 nonstep 전략이 AP 방향 COP 이동을 통해 fixed-support strategy로 균형을 유지하는 반면, step 전략은 AP 방향 COP 보상이 부족하여 stepping으로 전환됨을 시사한다.

MoS 최소값은 step에서 유의하게 낮다 (MOS_minDist_signed_min: step=0.0373 vs nonstep=0.0485; MOS_AP_v3d_min: step=0.0421 vs nonstep=0.0530). 이는 step onset 직전까지의 구간에서 step 시행이 더 불안정한 상태에 도달함을 확인한다.

### Joint Angles

Hip과 Knee의 ROM 및 absolute peak(`abs_peak = max(|x|)`)가 step에서 유의하게 크다 (Hip ROM: 8.87° vs 8.25°, Knee ROM: 11.57° vs 10.64°). 이는 stepping 준비 과정에서의 하지 굴곡 증가를 반영한다. Neck 각도도 유의하며 상체 보상적 움직임을 나타낸다. 발목(Ankle)과 체간(Trunk)은 유의한 차이가 없어, CPA 구간에서는 ankle/trunk 전략이 두 조건에서 유사함을 시사한다.

### Force/Torque

[platform_onset, step_onset] 구간에서 GRF와 ankle torque는 **모두 비유의**하다. 이전 0-800ms 고정 구간 분석에서는 GRF_Z와 ankle torque가 유의했으나, step onset 이전 구간으로 제한하면 차이가 사라진다. 이는 GRF와 torque의 step-nonstep 차이가 주로 step onset 이후(발이 떨어지는 시점 이후)에 발생함을 의미한다.

### 시간 구간 변경의 영향

고정 0-800ms → [platform_onset, step_onset] 적응형 구간으로 변경한 결과:
- FDR 유의 변수: **22/32 → 13/32**로 감소
- Force/Torque family: **3/7 → 0/7** (step onset 이후 차이만 반영하던 것이 제거됨)
- COM range/path_length: 유의하지 않음 (0-800ms에서 유의했던 차이가 CPA 구간에서는 미미)
- **CPA 구간에 특이적인 차이**: vCOM_Y_peak, COP_Y_range, MoS 지표가 핵심 변별 요인으로 남음

### Conclusion

1. **13/32 biomechanical 변수가 [platform_onset, step_onset] 구간에서 FDR-유의.** CPA 구간에 집중할수록 변별력 있는 변수가 선별된다.
2. **ML 방향 COM 속도와 COP 변위가 step의 핵심 특징**이며, AP 방향 COP range는 nonstep에서 더 크다 (fixed-support strategy 반영).
3. **MoS(동적 안정성)가 step에서 유의하게 낮다.** Step onset 직전까지 더 불안정한 상태에 도달한다.
4. **Hip/Knee ROM이 step에서 유의하게 크다.** Stepping 준비를 위한 하지 굴곡이 CPA 구간에서 이미 시작된다.
5. **GRF와 ankle torque는 CPA 구간에서 비유의.** 이전 0-800ms 분석에서 유의했던 차이는 step onset 이후 구간의 영향이었다.

---

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py
```

**Input**: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
**Output**: fig1–fig3 PNG (generated in this folder), stdout statistics

## Figures

| File | Description |
|------|-------------|
| fig1_lmm_forest_plot.png | Forest plot: LMM estimate ± 95% CI for all 32 DVs, family별 색상, FDR 유의 항목 강조 |
| fig2_violin_significant.png | FDR-significant 변수의 violin + strip plot (step vs nonstep) |
| fig3_descriptive_heatmap.png | z-scored group mean heatmap with significance markers |
