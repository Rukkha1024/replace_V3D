# Step vs. Non-step Biomechanical LMM Analysis

## Research Question

**"동일 perturbation 강도에서 step과 non-step 균형회복 전략 간 biomechanical 변수에 유의한 차이가 있는가?"**

Platform translation perturbation 실험에서 동일한 mixed velocity 조건 하에 step과 non-step이 공존하는 상황에서, COM, COP, MoS, 관절 각도, GRF, ankle torque 등 biomechanical 변수를 Linear Mixed Model(LMM)로 비교 분석하였다.

## Data Summary

- **184 trials** (step=112, nonstep=72) from 24 subjects
- Data source: `output/all_trials_timeseries.csv` (frame-level 100Hz timeseries) + `data/perturb_inform.xlsm` (step/nonstep classification)
- Analysis window: platform onset (0ms) ~ 800ms
- Trial-level aggregation: range, path length, peak velocity, minimum MoS per trial
- **32 dependent variables** across 3 variable families

---

## Results

### 1. LMM Summary

Statistical method: Linear Mixed Model via R lmerTest (`DV ~ step_TF + (1|subject)`, REML, Satterthwaite df). Benjamini-Hochberg FDR correction applied within each variable family.

**Overall: 22/32 variables showed FDR-significant differences (p_FDR < 0.05)**

### 2. Balance/Stability Family (15 variables, 13 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | df | p_FDR | Sig |
|----------|-------------|-----------------|----------|-----|-------|-----|
| COM_X_range | 0.0298±0.0172 | 0.0366±0.0197 | -0.0073 | 163.9 | 0.0018 | ** |
| COM_Y_range | 0.0326±0.0195 | 0.0185±0.0147 | 0.0152 | 164.2 | <0.0001 | *** |
| COM_Y_path_length | 0.0367±0.0191 | 0.0198±0.0144 | 0.0179 | 163.2 | <0.0001 | *** |
| vCOM_Y_peak | 0.1182±0.0558 | 0.0625±0.0462 | 0.0602 | 162.4 | <0.0001 | *** |
| COP_X_range | 0.1165±0.0289 | 0.0984±0.0128 | 0.0186 | 163.7 | <0.0001 | *** |
| COP_X_path_length | 0.1616±0.0538 | 0.1347±0.0347 | 0.0286 | 162.1 | <0.0001 | *** |
| COP_X_peak_velocity | 1.2919±1.0309 | 1.0091±0.3901 | 0.2855 | 164.9 | 0.0199 | * |
| COP_Y_range | 0.1721±0.1173 | 0.0777±0.0478 | 0.0969 | 167.7 | <0.0001 | *** |
| COP_Y_path_length | 0.3467±0.2503 | 0.1767±0.1023 | 0.1796 | 166.0 | <0.0001 | *** |
| MOS_minDist_signed_min | 0.0275±0.0203 | 0.0442±0.0164 | -0.0165 | 166.8 | <0.0001 | *** |
| MOS_AP_v3d_min | 0.0393±0.0202 | 0.0489±0.0163 | -0.0098 | 164.6 | 0.0001 | *** |
| MOS_ML_v3d_min | 0.0925±0.0294 | 0.1202±0.0260 | -0.0294 | 162.9 | <0.0001 | *** |
| COM_X_path_length | — | — | -0.0028 | 164.8 | 0.1911 | n.s. |
| vCOM_X_peak | — | — | -0.0023 | 163.9 | 0.6689 | n.s. |
| COP_Y_peak_velocity | — | — | 2.5863 | 175.1 | 0.0531 | n.s. |

### 3. Joint Angles Family (10 variables, 7 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | df | p_FDR | Sig |
|----------|-------------|-----------------|----------|-----|-------|-----|
| Hip_R_ROM | 10.6548±4.6166 | 8.9880±4.5181 | 2.6918 | 161.1 | <0.0001 | *** |
| Hip_R_peak | 10.3887±4.6448 | 8.0723±3.0549 | 3.0357 | 161.7 | <0.0001 | *** |
| Knee_R_ROM | 16.2523±8.7516 | 11.4938±5.3884 | 6.1824 | 160.6 | <0.0001 | *** |
| Knee_R_peak | 15.3082±8.6136 | 10.6343±5.2147 | 6.0188 | 160.5 | <0.0001 | *** |
| Trunk_ROM | 6.9868±5.8001 | 5.7786±4.2952 | 1.3166 | 160.6 | 0.0307 | * |
| Neck_ROM | 15.9277±11.2654 | 14.5762±9.8620 | 3.6762 | 160.1 | 0.0002 | *** |
| Neck_peak | 15.3617±11.3329 | 13.9499±10.0291 | 3.6308 | 160.2 | 0.0002 | *** |
| Ankle_R_ROM | — | — | 0.6252 | 160.3 | 0.2544 | n.s. |
| Ankle_R_peak | — | — | 0.3444 | 160.3 | 0.4673 | n.s. |
| Trunk_peak | — | — | 0.9967 | 160.6 | 0.0915 | n.s. |

### 4. Force/Torque Family (7 variables, 3 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | df | p_FDR | Sig |
|----------|-------------|-----------------|----------|-----|-------|-----|
| GRF_Z_peak | 217.4900±135.0443 | 178.8896±125.0511 | 52.3015 | 159.5 | 0.0001 | *** |
| GRF_Z_range | 358.0233±199.1921 | 310.1817±198.5282 | 77.9869 | 159.3 | <0.0001 | *** |
| AnkleTorqueMid_Y_peak | 1.3169±0.7856 | 0.9591±0.5944 | 0.3908 | 160.7 | <0.0001 | *** |
| GRF_X_peak | — | — | 0.8837 | 159.0 | 0.7756 | n.s. |
| GRF_X_range | — | — | 3.7275 | 159.0 | 0.1490 | n.s. |
| GRF_Y_peak | — | — | 3.6216 | 159.0 | 0.0509 | n.s. |
| GRF_Y_range | — | — | -1.1537 | 159.0 | 0.7756 | n.s. |

---

## Interpretation

### Balance & Stability

Step 시행에서 COM과 COP의 Y축(ML, 좌우) 변위, 이동 거리, 최대 속도가 모두 유의하게 크다. 이는 step 발생 시 ML 방향으로 더 큰 자세 동요가 나타남을 의미한다. 반면 COM X축(AP) range는 nonstep에서 유의하게 더 크다 (estimate = -0.0073). 이는 nonstep 전략이 AP 방향에서 더 큰 COM 변위를 허용하면서도 stepping 없이 균형을 유지함을 시사한다.

MoS 최소값은 step에서 유의하게 낮다 (MOS_minDist_signed_min: step=0.0275 vs nonstep=0.0442). 이는 step 시행에서 동적 안정성이 더 낮은 순간이 존재함을 확인한다. 특히 ML 방향 MoS(MOS_ML_v3d_min)의 차이가 가장 크며 (estimate = -0.0294), ML 불안정성이 step 유발의 핵심 요인일 수 있음을 시사한다.

### Joint Angles

무릎(Knee)과 고관절(Hip)의 ROM 및 peak가 step에서 유의하게 크다. 특히 무릎 ROM이 약 5° 더 크며 (step: 16.3° vs nonstep: 11.5°), 이는 stepping 준비 과정에서의 하지 굴곡 증가를 반영한다. 경추(Neck) 각도도 step에서 유의하게 크며, 상체의 보상적 움직임을 나타낸다. 반면 발목(Ankle) 각도는 유의한 차이가 없어, 발목 전략이 두 조건에서 유사하게 작용함을 시사한다.

### Force/Torque

수직 GRF(Z축) peak와 range가 step에서 유의하게 크며 (peak: 217.5N vs 178.9N), ankle torque peak도 유의하게 크다 (1.32 vs 0.96 Nm/kg). 수평 GRF(X, Y)는 유의한 차이가 없어, step과 nonstep 간 차이가 주로 수직 방향의 하중 이동에서 나타남을 보여준다.

### Conclusion

1. **22/32 biomechanical 변수가 step vs. nonstep 간 FDR-유의한 차이를 보인다.** 동일 perturbation 강도에서도 두 전략은 명확히 구별되는 biomechanical 패턴을 나타낸다.
2. **ML 방향의 COM/COP 변위 및 MoS 감소가 step의 핵심 특징이다.** ML 불안정성이 stepping 유발의 주요 기전일 가능성이 있다.
3. **무릎·고관절 ROM이 step에서 유의하게 크며**, stepping 준비 과정의 하지 굴곡 증가를 반영한다.
4. **발목 각도는 차이가 없으나 ankle torque는 유의하게 크다.** 발목 전략의 크기(토크)가 다를 뿐 각도 변화는 유사하다.
5. **수직 GRF가 step에서 크며**, 수평 방향은 차이 없다.

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
