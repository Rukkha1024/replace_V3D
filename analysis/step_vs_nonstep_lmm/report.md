# Step vs. Non-step LMM Analysis (Re-run)

## Research Question

**동일 perturbation 강도(mixed velocity)에서 step vs non-step 균형회복 전략 간 biomechanical 변수 차이가 있는가?**

## Data & Window

- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- Trials: **181** (step=108, nonstep=73), subjects=24
- Window: `[platform_onset_local, step_onset_local]`
  - step: trial의 실제 `step_onset_local`
  - nonstep: 동일 (subject, velocity) step trial의 `step_onset_local` 평균을 대입
- Window duration (ms): mean=502.9, sd=238.8, range=[150.0, 1400.0]

## Model & Multiple Comparisons

- Model (DV별 독립): `DV ~ step_TF + (1|subject)` (REML, R `lmerTest`)
- Multiple comparison: Benjamini–Hochberg FDR (BH-FDR), family-wise
  - 운동역학: COM/COP/GRF/MoS/xCOM-BOS 등
  - 운동학: Hip/Knee/Ankle/Trunk/Neck (ROM/peak)

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

- FDR significant: **15/38** (LMM DVs only; supplementary DVs excluded)

### Key Variables

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Trunk_peak` | 4.4225±5.6911 | 3.8789±4.2274 | 0.6542 | n.s. |
| `xCOM_BOS_platformonset` | 0.6325±0.0680 | 0.7004±0.0722 | -0.0574 | *** |
| `xCOM_BOS_steponset` | 0.2833±0.1739 | 0.4048±0.1088 | -0.1240 | *** |

### Supplementary (size interpretation; not used for significance)

| Variable | Step (M±SD, cm) | Nonstep (M±SD, cm) | Δ (step−nonstep, cm) |
|---|---:|---:|---:|
| `xCOM_BOS_cm_platformonset` | 12.91±1.64 | 14.43±1.61 | -1.51 |
| `xCOM_BOS_cm_steponset` | 5.92±3.74 | 8.21±2.39 | -2.29 |

- `xCOM_BOS_cm_*` = `(xCOM_hof - BOS_rear) × 100 [cm]`로 해석 가능한 보조 지표(거리 크기 해석용).

### Full LMM Table

| DV | Family | Estimate | SE | t | p_fdr | Sig |
|---|---|---:|---:|---:|---:|---|
| `COP_Y_range` | 운동역학 | 0.0555 | 0.0049 | 11.361 | 0.000000 | *** |
| `GRF_Y_range` | 운동역학 | 34.7922 | 3.6148 | 9.625 | 0.000000 | *** |
| `GRF_Y_peak` | 운동역학 | 25.3449 | 2.6827 | 9.447 | 0.000000 | *** |
| `xCOM_BOS_platformonset` | 운동역학 | -0.0574 | 0.0066 | -8.736 | 0.000000 | *** |
| `COP_Y_mean_velocity` | 운동역학 | 0.1454 | 0.0169 | 8.623 | 0.000000 | *** |
| `xCOM_BOS_steponset` | 운동역학 | -0.1240 | 0.0143 | -8.658 | 0.000000 | *** |
| `COP_Y_path_length` | 운동역학 | 0.0615 | 0.0083 | 7.398 | 0.000000 | *** |
| `COP_Y_peak_velocity` | 운동역학 | 0.7913 | 0.1175 | 6.736 | 0.000000 | *** |
| `vCOM_Y_peak` | 운동역학 | 0.0225 | 0.0036 | 6.203 | 0.000000 | *** |
| `MOS_minDist_signed_min` | 운동역학 | -0.0121 | 0.0023 | -5.148 | 0.000002 | *** |
| `MOS_AP_v3d_min` | 운동역학 | -0.0114 | 0.0023 | -4.938 | 0.000005 | *** |
| `AnkleTorqueMid_Y_peak` | 운동역학 | -0.1271 | 0.0319 | -3.986 | 0.000239 | *** |
| `MOS_ML_v3d_min` | 운동역학 | -0.0057 | 0.0019 | -3.065 | 0.005508 | ** |
| `vCOM_Y_mean_abs` | 운동역학 | 0.0022 | 0.0015 | 1.448 | 0.299310 | n.s. |
| `GRF_Z_peak` | 운동역학 | 14.0132 | 12.1393 | 1.154 | 0.466858 | n.s. |
| `COM_X_range` | 운동역학 | -0.0012 | 0.0012 | -0.987 | 0.568750 | n.s. |
| `GRF_Z_range` | 운동역학 | 13.4028 | 14.2731 | 0.939 | 0.575094 | n.s. |
| `COP_X_mean_velocity` | 운동역학 | 0.0992 | 0.1309 | 0.758 | 0.662470 | n.s. |
| `COP_X_peak_velocity` | 운동역학 | 1.0990 | 1.4167 | 0.776 | 0.662470 | n.s. |
| `COM_X_path_length` | 운동역학 | 0.0004 | 0.0012 | 0.329 | 0.885960 | n.s. |
| `COM_Y_path_length` | 운동역학 | 0.0003 | 0.0010 | 0.336 | 0.885960 | n.s. |
| `COP_X_path_length` | 운동역학 | 0.0123 | 0.0285 | 0.431 | 0.885960 | n.s. |
| `GRF_X_peak` | 운동역학 | 2.7440 | 8.6534 | 0.317 | 0.885960 | n.s. |
| `vCOM_X_peak` | 운동역학 | -0.0010 | 0.0033 | -0.307 | 0.885960 | n.s. |
| `vCOM_X_mean_abs` | 운동역학 | -0.0004 | 0.0016 | -0.223 | 0.922688 | n.s. |
| `GRF_X_range` | 운동역학 | 2.8538 | 16.3623 | 0.174 | 0.928055 | n.s. |
| `COM_Y_range` | 운동역학 | -0.0000 | 0.0010 | -0.035 | 0.976278 | n.s. |
| `COP_X_range` | 운동역학 | -0.0004 | 0.0139 | -0.030 | 0.976278 | n.s. |
| `Hip_stance_peak` | 운동학 | 1.1676 | 0.2852 | 4.094 | 0.000677 | *** |
| `Hip_stance_ROM` | 운동학 | 1.0357 | 0.3028 | 3.420 | 0.003996 | ** |
| `Neck_ROM` | 운동학 | 1.9024 | 0.9235 | 2.060 | 0.102657 | n.s. |
| `Neck_peak` | 운동학 | 1.9772 | 0.9220 | 2.145 | 0.102657 | n.s. |
| `Trunk_ROM` | 운동학 | 0.7528 | 0.5442 | 1.383 | 0.337054 | n.s. |
| `Trunk_peak` | 운동학 | 0.6542 | 0.5354 | 1.222 | 0.372705 | n.s. |
| `Ankle_stance_ROM` | 운동학 | -0.3758 | 0.5453 | -0.689 | 0.702431 | n.s. |
| `Knee_stance_peak` | 운동학 | 0.2143 | 0.3812 | 0.562 | 0.718428 | n.s. |
| `Ankle_stance_peak` | 운동학 | -0.1640 | 0.5072 | -0.323 | 0.746910 | n.s. |
| `Knee_stance_ROM` | 운동학 | 0.1488 | 0.3916 | 0.380 | 0.746910 | n.s. |

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
