# Step vs. Non-step LMM Analysis (Re-run)

## Research Question

**동일 perturbation 강도(mixed velocity)에서 step vs non-step 균형회복 전략 간 biomechanical 변수 차이가 있는가?**

## Data & Window

- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- Trials: **125** (step=52, nonstep=73), subjects=24
- Window: `[platform_onset_local, step_onset_local]`
  - step: trial의 실제 `step_onset_local`
  - nonstep: 동일 (subject, velocity) step trial의 `step_onset_local` 평균을 대입
- Window duration (ms): mean=509.5, sd=246.0, range=[170.0, 1400.0]

## Model & Multiple Comparisons

- Model (DV별 독립): `DV ~ step_TF + (1|subject)` (REML, R `lmerTest`)
- Multiple comparison: Benjamini–Hochberg FDR (BH-FDR), family-wise
  - 운동역학: COM/COP/GRF/MoS/xCOM-BOS, ankle torque, segment moment
  - 운동학: Hip/Knee/Ankle/Trunk/Neck angle and angular velocity

## Coordinate Definition (Joint Angle)

- Joint angle는 Visual3D-like **intrinsic XYZ Euler sequence** 기준으로 계산한다.
- Segment 좌표계는 `X=+Right`, `Y=+Anterior`, `Z=+Up/+Proximal`를 사용한다.
- `*_X/*_Y/*_Z`는 축 회전 성분이며, 임상 평면(sagittal/frontal/transverse)과 완전한 1:1 대응으로 단정하지 않는다.
- 본 분석의 관절각 LMM 변수는 `X`축 성분(`Hip/Knee/Ankle stance`, `Trunk/Neck`)만 사용한다.

## Stance-Leg Selection Rule

- `step_r` trial: left 관절각을 stance 값으로 사용한다.
- `step_l` trial: right 관절각을 stance 값으로 사용한다.
- `nonstep` trial: subject별 `mixed==1` step trial의 `major_step_side`를 stance 기준으로 사용한다.
- `major_step_side`가 tie이면 `(left + right)/2` 평균값을 stance 값으로 사용한다.

## Results (BH-FDR, alpha=0.05)

- FDR significant: **41/83** (LMM DVs only; supplementary DVs excluded)

### Key Variables

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Trunk_peak` | 4.6569±5.3632 | 3.7877±4.2027 | 0.5459 | n.s. |
| `xCOM_BOS_platformonset` | 0.6329±0.0683 | 0.7004±0.0722 | -0.0519 | *** |
| `xCOM_BOS_steponset` | 0.2840±0.2087 | 0.4002±0.1163 | -0.1283 | *** |

### Supplementary (size interpretation; not used for significance)

| Variable | Step (M±SD, cm) | Nonstep (M±SD, cm) | Δ (step−nonstep, cm) |
|---|---:|---:|---:|
| `xCOM_BOS_cm_platformonset` | 12.92±1.63 | 14.43±1.61 | -1.51 |
| `xCOM_BOS_cm_steponset` | 6.00±4.48 | 8.15±2.60 | -2.15 |

- `xCOM_BOS_cm_*` = `(xCOM_hof - BOS_rear) × 100 [cm]`로 해석 가능한 보조 지표(거리 크기 해석용).

### Full LMM Table

| DV | Family | Estimate | SE | t | p_fdr | Sig |
|---|---|---:|---:|---:|---:|---|
| `COP_Y_range` | 운동역학 | 0.0600 | 0.0059 | 10.093 | 0.000000 | *** |
| `GRF_Y_range` | 운동역학 | 39.2969 | 4.4839 | 8.764 | 0.000000 | *** |
| `Hip_stance_ref_Y_Nm_peak` | 운동역학 | 41.5893 | 4.8301 | 8.610 | 0.000000 | *** |
| `COP_Y_mean_velocity` | 운동역학 | 0.1510 | 0.0192 | 7.868 | 0.000000 | *** |
| `COP_Y_path_length` | 운동역학 | 0.0695 | 0.0089 | 7.855 | 0.000000 | *** |
| `GRF_Y_peak` | 운동역학 | 26.5423 | 3.5842 | 7.405 | 0.000000 | *** |
| `xCOM_BOS_steponset` | 운동역학 | -0.1283 | 0.0177 | -7.241 | 0.000000 | *** |
| `Knee_stance_ref_Y_Nm_peak` | 운동역학 | 27.3602 | 3.8608 | 7.087 | 0.000000 | *** |
| `AnkleTorqueMid_Y_peak` | 운동역학 | -0.1423 | 0.0215 | -6.603 | 0.000000 | *** |
| `Knee_stance_ref_Z_Nm_peak` | 운동역학 | 8.6837 | 1.3170 | 6.594 | 0.000000 | *** |
| `xCOM_BOS_platformonset` | 운동역학 | -0.0519 | 0.0080 | -6.504 | 0.000000 | *** |
| `Ankle_stance_ref_X_Nm_peak` | 운동역학 | 26.3492 | 4.3622 | 6.040 | 0.000000 | *** |
| `MOS_minDist_signed_min` | 운동역학 | -0.0145 | 0.0026 | -5.552 | 0.000001 | *** |
| `COP_Y_peak_velocity` | 운동역학 | 0.7289 | 0.1340 | 5.442 | 0.000001 | *** |
| `MOS_AP_v3d_min` | 운동역학 | -0.0140 | 0.0026 | -5.301 | 0.000002 | *** |
| `Ankle_stance_ref_Y_Nm_peak` | 운동역학 | 20.7274 | 3.9845 | 5.202 | 0.000003 | *** |
| `vCOM_Y_peak` | 운동역학 | 0.0235 | 0.0047 | 5.013 | 0.000005 | *** |
| `Trunk_ref_Y_Nm_peak` | 운동역학 | 5.4257 | 1.1772 | 4.609 | 0.000027 | *** |
| `Neck_ref_Y_Nm_peak` | 운동역학 | 0.2630 | 0.0626 | 4.204 | 0.000118 | *** |
| `Hip_stance_ref_X_Nm_peak` | 운동역학 | -43.5678 | 15.1438 | -2.877 | 0.010576 | * |
| `Neck_ref_Z_Nm_peak` | 운동역학 | 0.0963 | 0.0385 | 2.503 | 0.028351 | * |
| `MOS_ML_v3d_min` | 운동역학 | -0.0054 | 0.0022 | -2.421 | 0.033541 | * |
| `Trunk_ref_Z_Nm_peak` | 운동역학 | 1.6216 | 0.8109 | 2.000 | 0.089748 | n.s. |
| `Neck_ref_X_Nm_peak` | 운동역학 | 0.2353 | 0.1499 | 1.570 | 0.214325 | n.s. |
| `vCOM_Y_mean_abs` | 운동역학 | 0.0030 | 0.0020 | 1.543 | 0.215906 | n.s. |
| `Ankle_stance_ref_Z_Nm_peak` | 운동역학 | -2.2919 | 1.7060 | -1.343 | 0.290075 | n.s. |
| `GRF_Z_peak` | 운동역학 | 23.3563 | 17.3892 | 1.343 | 0.290075 | n.s. |
| `GRF_Z_range` | 운동역학 | 25.3944 | 19.1977 | 1.323 | 0.290075 | n.s. |
| `Knee_stance_ref_X_Nm_peak` | 운동역학 | 11.2271 | 8.9046 | 1.261 | 0.311868 | n.s. |
| `COP_X_peak_velocity` | 운동역학 | 2.3394 | 2.0404 | 1.147 | 0.363800 | n.s. |
| `COM_Y_path_length` | 운동역학 | 0.0012 | 0.0011 | 1.036 | 0.409523 | n.s. |
| `vCOM_X_mean_abs` | 운동역학 | -0.0021 | 0.0020 | -1.031 | 0.409523 | n.s. |
| `COP_X_mean_velocity` | 운동역학 | 0.1783 | 0.1888 | 0.945 | 0.451904 | n.s. |
| `COP_X_path_length` | 운동역학 | 0.0347 | 0.0411 | 0.844 | 0.506146 | n.s. |
| `COM_X_range` | 운동역학 | -0.0012 | 0.0015 | -0.808 | 0.515088 | n.s. |
| `COM_Y_range` | 운동역학 | 0.0009 | 0.0011 | 0.769 | 0.515088 | n.s. |
| `Trunk_ref_X_Nm_peak` | 운동역학 | -1.2732 | 1.6176 | -0.787 | 0.515088 | n.s. |
| `GRF_X_range` | 운동역학 | 13.6140 | 20.7350 | 0.657 | 0.580448 | n.s. |
| `COP_X_range` | 운동역학 | 0.0122 | 0.0201 | 0.607 | 0.601250 | n.s. |
| `Hip_stance_ref_Z_Nm_peak` | 운동역학 | 1.1361 | 2.4198 | 0.469 | 0.687743 | n.s. |
| `COM_X_path_length` | 운동역학 | 0.0007 | 0.0016 | 0.446 | 0.688327 | n.s. |
| `GRF_X_peak` | 운동역학 | 1.7831 | 11.3183 | 0.158 | 0.895969 | n.s. |
| `vCOM_X_peak` | 운동역학 | 0.0005 | 0.0043 | 0.113 | 0.910357 | n.s. |
| `Hip_stance_mov_Y_deg_s_peak` | 운동학 | 4.1948 | 0.8856 | 4.737 | 0.000097 | *** |
| `Knee_stance_mov_Y_deg_s_peak` | 운동학 | 6.3235 | 1.3592 | 4.653 | 0.000097 | *** |
| `Knee_stance_mov_Z_deg_s_peak` | 운동학 | 7.5341 | 1.6069 | 4.689 | 0.000097 | *** |
| `Knee_stance_ref_Z_deg_s_peak` | 운동학 | 8.1856 | 1.6998 | 4.816 | 0.000097 | *** |
| `Hip_stance_ref_Y_deg_s_peak` | 운동학 | 5.0141 | 1.1058 | 4.534 | 0.000125 | *** |
| `Hip_stance_peak` | 운동학 | 1.2559 | 0.3320 | 3.783 | 0.001754 | ** |
| `Hip_stance_mov_Z_deg_s_peak` | 운동학 | 15.4661 | 4.2548 | 3.635 | 0.002109 | ** |
| `Hip_stance_ref_Z_deg_s_peak` | 운동학 | 15.0641 | 4.1373 | 3.641 | 0.002109 | ** |
| `Hip_stance_ROM` | 운동학 | 1.1573 | 0.3419 | 3.385 | 0.004509 | ** |
| `Ankle_stance_mov_Z_deg_s_peak` | 운동학 | 9.8918 | 3.0882 | 3.203 | 0.007111 | ** |
| `Ankle_stance_ref_Z_deg_s_peak` | 운동학 | 9.6284 | 3.1635 | 3.044 | 0.008975 | ** |
| `Neck_ref_X_deg_s_peak` | 운동학 | 12.7255 | 4.1227 | 3.087 | 0.008975 | ** |
| `Trunk_mov_Z_deg_s_peak` | 운동학 | 5.8957 | 1.9334 | 3.049 | 0.008975 | ** |
| `Neck_mov_X_deg_s_peak` | 운동학 | 12.4003 | 4.1081 | 3.018 | 0.009179 | ** |
| `Trunk_ref_Z_deg_s_peak` | 운동학 | 5.2563 | 1.9267 | 2.728 | 0.019747 | * |
| `Trunk_ref_Y_deg_s_peak` | 운동학 | 6.4352 | 2.4568 | 2.619 | 0.024943 | * |
| `Neck_ROM` | 운동학 | 2.6052 | 1.0521 | 2.476 | 0.035087 | * |
| `Neck_peak` | 운동학 | 2.5700 | 1.0550 | 2.436 | 0.036820 | * |
| `Neck_ref_Y_deg_s_peak` | 운동학 | 6.1003 | 2.6068 | 2.340 | 0.044047 | * |
| `Knee_stance_ref_Y_deg_s_peak` | 운동학 | 2.8649 | 1.2868 | 2.226 | 0.056166 | n.s. |
| `Neck_mov_Y_deg_s_peak` | 운동학 | 4.8045 | 2.2064 | 2.178 | 0.057081 | n.s. |
| `Trunk_mov_Y_deg_s_peak` | 운동학 | 4.9830 | 2.2841 | 2.182 | 0.057081 | n.s. |
| `Neck_mov_Z_deg_s_peak` | 운동학 | 4.4958 | 2.2964 | 1.958 | 0.091598 | n.s. |
| `Knee_stance_ROM` | 운동학 | 0.6134 | 0.3736 | 1.642 | 0.171090 | n.s. |
| `Trunk_mov_X_deg_s_peak` | 운동학 | 5.7233 | 3.5647 | 1.606 | 0.171090 | n.s. |
| `Trunk_ref_X_deg_s_peak` | 운동학 | 5.8192 | 3.5887 | 1.622 | 0.171090 | n.s. |
| `Hip_stance_mov_X_deg_s_peak` | 운동학 | 4.6008 | 2.9077 | 1.582 | 0.172939 | n.s. |
| `Hip_stance_ref_X_deg_s_peak` | 운동학 | 4.3304 | 2.8523 | 1.518 | 0.188716 | n.s. |
| `Knee_stance_peak` | 운동학 | 0.5318 | 0.3672 | 1.448 | 0.200932 | n.s. |
| `Trunk_ROM` | 운동학 | 0.7472 | 0.5131 | 1.456 | 0.200932 | n.s. |
| `Neck_ref_Z_deg_s_peak` | 운동학 | 2.0867 | 1.6872 | 1.237 | 0.282257 | n.s. |
| `Trunk_peak` | 운동학 | 0.5459 | 0.5045 | 1.082 | 0.352218 | n.s. |
| `Knee_stance_mov_X_deg_s_peak` | 운동학 | 4.2805 | 4.4135 | 0.970 | 0.393464 | n.s. |
| `Knee_stance_ref_X_deg_s_peak` | 운동학 | 4.3284 | 4.4349 | 0.976 | 0.393464 | n.s. |
| `Ankle_stance_ref_X_deg_s_peak` | 운동학 | 2.3399 | 2.7560 | 0.849 | 0.454689 | n.s. |
| `Ankle_stance_mov_Y_deg_s_peak` | 운동학 | 1.8636 | 2.4173 | 0.771 | 0.491646 | n.s. |
| `Ankle_stance_ref_Y_deg_s_peak` | 운동학 | 1.4483 | 2.1738 | 0.666 | 0.547805 | n.s. |
| `Ankle_stance_mov_X_deg_s_peak` | 운동학 | 1.4494 | 2.9949 | 0.484 | 0.662580 | n.s. |
| `Ankle_stance_peak` | 운동학 | 0.2023 | 0.4800 | 0.422 | 0.691572 | n.s. |
| `Ankle_stance_ROM` | 운동학 | 0.1804 | 0.5089 | 0.354 | 0.723764 | n.s. |

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py
```

## Figures

| File | Description |
|---|---|
| `fig1_lmm_forest_plot.png` | LMM estimate ± 95% CI |
| `fig2_violin_significant.png` | Significant DV violin/strip |
| `fig3_descriptive_heatmap.png` | z-scored group mean heatmap |
