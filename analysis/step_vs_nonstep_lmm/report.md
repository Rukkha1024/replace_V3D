# Step vs. Non-step Biomechanical LMM Analysis

## Research Question

**"동일 perturbation 강도에서 step과 non-step 균형회복 전략 간 biomechanical 변수에 유의한 차이가 있는가?"**

Platform translation perturbation 실험에서 동일한 mixed velocity 조건 하에 step과 non-step이 공존하는 상황에서, COM, COP, MoS, 관절 각도, GRF, ankle torque 등 biomechanical 변수를 Linear Mixed Model(LMM)로 비교 분석하였다.

## Data Summary

- **125 trials** (step=53, nonstep=72) from 24 subjects
- Data source: `output/all_trials_timeseries.csv` (frame-level 100Hz timeseries) + `data/perturb_inform.xlsm` (step/nonstep classification)
- **32 dependent variables** across 3 variable families
- `end_frame` 산출 규칙: `(subject, velocity)` step 평균 우선, 불가 시 global step 평균 fallback 적용 (이번 실행에서 5 nonstep trials 보정)

### Analysis Window

분석 시간 구간은 trial마다 개별적으로 설정하였다:

- **분석 시작**: `platform_onset_local` — 각 trial에서 perturbation platform이 움직이기 시작한 MocapFrame
- **분석 종료**: `end_frame` — step onset 기반으로 trial별 산출
  - **Step trial**: 해당 trial의 실제 `step_onset_local` (발이 지면에서 처음 떨어지는 시점)
  - **Nonstep trial**:
    1. 동일 `(subject, velocity)` 내 step trial들의 `step_onset_local` 평균값 대입
    2. 위 값이 없으면 전체 step trial의 global `step_onset_local` 평균값 대입

이 방식은 step 발생 직전까지의 CPA(Compensatory Postural Adjustment) 구간에 집중하기 위한 것이다. 고정 시간 창(예: 0–800ms)은 perturbation 강도와 피험자에 따라 step onset timing이 다르므로 적합하지 않다. Nonstep trial에 동일 `(subject, velocity)` 평균을 우선 적용하고, nonstep-only 그룹에는 global step 평균 fallback을 적용해 분석 제외를 최소화하면서 두 조건 간 시간 구조 비교 가능성을 유지하였다.

### Coordinate & Sign Conventions

방향성 해석을 위해 좌표축의 `(+)/(-)`를 아래 기준으로 고정하였다 (실데이터 QC: 185개 C3D).

#### Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|------|---------------|---------------|-----------|
| AP (X) | `+X = Anterior` | `-X = Posterior` | `COM_X`, `vCOM_X`, `xCOM_X`, `COP_X_*`, `BOS_minX/BOS_maxX`, `MOS_AP_v3d` |
| ML (Y) | `+Y = Left` | `-Y = Right` | `COM_Y`, `vCOM_Y`, `xCOM_Y`, `COP_Y_*`, `BOS_minY/BOS_maxY`, `MOS_ML_v3d` |
| Vertical (Z) | `+Z = Up` | `-Z = Down` | `COM_Z`, `vCOM_Z`, `xCOM_Z`, `GRF_Z` |

#### Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|--------|--------------|--------------|----------------|
| `MOS_minDist_signed` | `inside` | `outside` | convex hull 기반 signed min distance |
| `MOS_AP_v3d` | AP bound 내부 | AP bound 외부 | closest-bound (AP) |
| `MOS_ML_v3d` | ML bound 내부 | ML bound 외부 | closest-bound (ML) |

#### Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|----------------|------------------|-----------|
| Joint angles (X/Y/Z) | X: `+Flex / -Ext` (ankle X: `+Dorsi / -Plantar`), Y: `+Add / -Abd`, Z: `+IR / -ER` | Left Y/Z는 sign-unification(부호 반전) 적용 |
| `GRF_*`, `GRM_*`, `AnkleTorque*` | 플랫폼 onset 기준 Δ값으로 해석 | onset-zeroed 값 기준으로 비교 |
| `AnkleTorque*_int`, `AnkleTorque*_ext` | 내부토크는 외부토크와 반대 부호 | `AnkleTorque*_int = -AnkleTorque*_ext` |
| `COP_*_m`, `COP_*_m_onset0` | absolute 좌표 vs onset-zeroed 변위 | 두 표현을 구분해 해석 |

Note: 레거시 메모(`-X=AP`, `+Z=ML`, `+Y=UP`)와 충돌할 수 있으나, 본 리포트는 현재 배치 데이터의 실측 QC 기준을 우선 적용한다.

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

**Overall: 17/32 variables showed FDR-significant differences (Sig at α=0.05)**

### 2. Balance/Stability Family (15 variables, 7 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|----------|-------------|-----------------|----------|-----|
| vCOM_Y_peak | 0.0561±0.0294 | 0.0347±0.0236 | 0.0223 | *** |
| COP_Y_range | 0.1034±0.0409 | 0.0503±0.0365 | 0.0584 | *** |
| COP_Y_path_length | 0.1535±0.0709 | 0.1016±0.0724 | 0.0671 | *** |
| COP_Y_peak_velocity | 1.6746±0.9534 | 1.1995±1.0801 | 0.7084 | *** |
| MOS_minDist_signed_min | 0.0335±0.0282 | 0.0482±0.0119 | -0.0139 | *** |
| MOS_AP_v3d_min | 0.0383±0.0282 | 0.0525±0.0126 | -0.0134 | *** |
| MOS_ML_v3d_min | 0.1312±0.0171 | 0.1354±0.0153 | -0.0053 | * |
| COP_X_range | — | — | 0.0109 | n.s. |
| COM_X_range | — | — | -0.0016 | n.s. |
| COM_X_path_length | — | — | 0.0003 | n.s. |
| vCOM_X_peak | — | — | -0.0001 | n.s. |
| COM_Y_range | — | — | 0.0008 | n.s. |
| COM_Y_path_length | — | — | 0.0011 | n.s. |
| COP_X_path_length | — | — | 0.0334 | n.s. |
| COP_X_peak_velocity | — | — | 2.3034 | n.s. |

### 3. Joint Angles Family (10 variables, 8 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|----------|-------------|-----------------|----------|-----|
| Hip_R_ROM | 8.8818±3.6268 | 8.0920±4.0379 | 1.8969 | *** |
| Hip_R_peak | 8.7473±3.6535 | 7.6009±3.1116 | 2.0498 | *** |
| Knee_R_ROM | 12.1091±7.3416 | 10.7427±5.3134 | 3.4487 | *** |
| Knee_R_peak | 11.0859±7.1514 | 9.8867±5.0708 | 3.2359 | *** |
| Ankle_R_ROM | 9.9153±5.8061 | 9.4366±5.1941 | 1.3772 | * |
| Ankle_R_peak | 8.0364±4.9096 | 7.5008±4.4938 | 1.1587 | * |
| Neck_ROM | 10.1496±10.9637 | 10.4525±8.9101 | 2.4296 | * |
| Neck_peak | 9.8476±10.9712 | 10.2679±8.9096 | 2.3376 | * |
| Trunk_ROM | — | — | 0.6851 | n.s. |
| Trunk_peak | — | — | 0.4890 | n.s. |

### 4. Force/Torque Family (7 variables, 2 significant)

| Variable | Estimate | Sig |
|----------|----------|-----|
| GRF_X_peak | 1.8828 | n.s. |
| GRF_X_range | 13.8490 | n.s. |
| GRF_Y_peak | 25.6989 | *** |
| GRF_Y_range | 38.2639 | *** |
| GRF_Z_peak | 21.1873 | n.s. |
| GRF_Z_range | 22.2703 | n.s. |
| AnkleTorqueMid_Y_peak | -0.0690 | n.s. |

---

## Interpretation

해석 섹션의 AP/ML/inside-outside 방향 문구는 위 `Coordinate & Sign Conventions` 표의 `(+)/(-)` 정의를 따른다.

### Balance & Stability

platform onset ~ step onset 구간에서 step 시행은 ML(좌우) 방향 COM 속도(vCOM_Y_peak)와 COP 변위/속도(COP_Y_range, COP_Y_path_length, COP_Y_peak_velocity)가 유의하게 크다. 이는 stepping 직전까지의 CPA 구간에서 ML 방향 자세 동요가 step 유발의 핵심 특징임을 시사한다.

COP_X 계열(COP_X_range, COP_X_path_length, COP_X_peak_velocity)은 현재 데이터에서 유의하지 않았다. 따라서 AP 방향 COP 보상 차이는 본 분석 구간에서 일관된 분별 신호로 확인되지 않았다.

MoS 최소값은 step에서 유의하게 낮다 (MOS_minDist_signed_min: step=0.0335 vs nonstep=0.0482; MOS_AP_v3d_min: step=0.0383 vs nonstep=0.0525; MOS_ML_v3d_min: step=0.1312 vs nonstep=0.1354). 이는 step onset 직전까지의 구간에서 step 시행이 더 불안정한 상태에 도달함을 확인한다.

### Joint Angles

Hip과 Knee의 ROM 및 absolute peak(`abs_peak = max(|x|)`)가 step에서 유의하게 크다 (Hip ROM: 8.88° vs 8.09°, Knee ROM: 12.11° vs 10.74°). 이는 stepping 준비 과정에서의 하지 굴곡 증가를 반영한다. Ankle ROM/peak도 유의했고, Neck 변수는 LMM 고정효과 기준에서 유의하였다. Trunk는 유의한 차이가 없었다.

### Force/Torque

[platform_onset, step_onset] 구간에서 GRF_Y_peak와 GRF_Y_range는 step에서 유의하게 크다. 반면 GRF_X/GRF_Z 계열과 ankle torque는 유의하지 않았다. 즉, 수직축/전후축 지면반력보다는 ML 방향 지면반력 변동이 step 분별에 더 직접적으로 연결된다.

### Conclusion

1. **17/32 biomechanical 변수가 [platform_onset, step_onset] 구간에서 FDR-유의**하였다.
2. **ML 방향 COM/COP 지표(vCOM_Y_peak, COP_Y_range, COP_Y_path_length, COP_Y_peak_velocity)**가 step의 핵심 특징으로 확인되었다.
3. **MoS 최소값(MOS_minDist_signed_min, MOS_AP_v3d_min, MOS_ML_v3d_min)은 step에서 유의하게 낮아** step onset 직전 불안정성이 더 크다.
4. **Hip/Knee뿐 아니라 Ankle의 ROM/peak도 step에서 유의하게 크며**, Trunk는 유의하지 않았다.
5. **Force/Torque family에서는 GRF_Y_peak, GRF_Y_range만 유의**했고, GRF_X/GRF_Z 및 ankle torque는 비유의였다.

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
