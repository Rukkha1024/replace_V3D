# Step vs. Non-step Biomechanical LMM Analysis

## Research Question

**"동일 perturbation 강도에서 step과 non-step 균형회복 전략 간 biomechanical 변수 차이가 있는가?"**

이번 분석은 step/non-step가 함께 존재하는 mixed velocity trial에서, pre-step 구간의 biomechanical 반응을 LMM으로 비교하는 목적이다.

## Data & Model

- Trials: **125** (step=53, nonstep=72), subjects=24
- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- DVs: **36개** (Balance/Stability 19, Joint Angles 10, Force/Torque 7)
- Window: `[platform_onset_local, end_frame]`
  - step: `end_frame = actual step_onset_local`
  - nonstep: subject-velocity step 평균 onset, 불가 시 prefilter `platform` fallback
    - end_frame fill (subject-velocity mean): 68
    - end_frame fill (prefilter platform subject-velocity mean): 5
- Model: `DV ~ step_TF + (1|subject)`, REML (`lmerTest`)
- Multiple comparison: BH-FDR (family-wise, alpha=0.05)
- Stance-equivalent joint: major step side 기준 (step_r=14, step_l=14, tie=7)

요청 변수 정의:
- `θtrunk,max`는 `Trunk_peak = max(|Trunk_X_deg|)`로 대응
- `xCOM/BOS_platformonset = (xCOM_X - BOS_minX)/(BOS_maxX - BOS_minX)` at `platform_onset_local`
- `xCOM/BOS_steponset`
  - step: actual step onset
  - nonstep: `platform_onset_local + 300 ms (30 frames)`

## Results

### Overall

- **FDR significant: 14/36**

| Family | Total DVs | Significant DVs |
|---|---:|---:|
| Balance/Stability | 19 | 10 |
| Joint Angles | 10 | 2 |
| Force/Torque | 7 | 2 |

### Full LMM Results

| DV | Family | Estimate | SE | t | Sig |
|---|---|---:|---:|---:|---|
| `COM_X_range` | Balance/Stability | -0.0015 | 0.0015 | -0.972 | n.s. |
| `COM_X_path_length` | Balance/Stability | 0.0005 | 0.0016 | 0.297 | n.s. |
| `vCOM_X_peak` | Balance/Stability | -0.0001 | 0.0043 | -0.027 | n.s. |
| `COM_Y_range` | Balance/Stability | 0.0009 | 0.0011 | 0.826 | n.s. |
| `COM_Y_path_length` | Balance/Stability | 0.0012 | 0.0011 | 1.084 | n.s. |
| `vCOM_Y_peak` | Balance/Stability | 0.0230 | 0.0047 | 4.913 | *** |
| `COP_X_range` | Balance/Stability | 0.0119 | 0.0200 | 0.593 | n.s. |
| `COP_X_path_length` | Balance/Stability | 0.0349 | 0.0410 | 0.851 | n.s. |
| `COP_X_peak_velocity` | Balance/Stability | 2.3092 | 2.0353 | 1.135 | n.s. |
| `COP_X_mean_velocity` | Balance/Stability | 0.1742 | 0.1883 | 0.925 | n.s. |
| `COP_Y_range` | Balance/Stability | 0.0594 | 0.0060 | 9.971 | *** |
| `COP_Y_path_length` | Balance/Stability | 0.0685 | 0.0089 | 7.708 | *** |
| `COP_Y_peak_velocity` | Balance/Stability | 0.7236 | 0.1329 | 5.445 | *** |
| `COP_Y_mean_velocity` | Balance/Stability | 0.1494 | 0.0192 | 7.786 | *** |
| `MOS_minDist_signed_min` | Balance/Stability | -0.0141 | 0.0027 | -5.282 | *** |
| `MOS_AP_v3d_min` | Balance/Stability | -0.0136 | 0.0027 | -5.041 | *** |
| `MOS_ML_v3d_min` | Balance/Stability | -0.0053 | 0.0022 | -2.389 | * |
| `xCOM_BOS_platformonset` | Balance/Stability | -0.0513 | 0.0080 | -6.441 | *** |
| `xCOM_BOS_steponset` | Balance/Stability | -0.1008 | 0.0156 | -6.458 | *** |
| `Hip_stance_ROM` | Joint Angles | 1.1553 | 0.3391 | 3.407 | ** |
| `Hip_stance_peak` | Joint Angles | 1.2650 | 0.3295 | 3.839 | ** |
| `Knee_stance_ROM` | Joint Angles | 0.6453 | 0.3730 | 1.730 | n.s. |
| `Knee_stance_peak` | Joint Angles | 0.5509 | 0.3651 | 1.509 | n.s. |
| `Ankle_stance_ROM` | Joint Angles | 0.1893 | 0.5049 | 0.375 | n.s. |
| `Ankle_stance_peak` | Joint Angles | 0.1941 | 0.4763 | 0.407 | n.s. |
| `Trunk_ROM` | Joint Angles | 0.7324 | 0.5098 | 1.437 | n.s. |
| `Trunk_peak` | Joint Angles | 0.5205 | 0.5024 | 1.036 | n.s. |
| `Neck_ROM` | Joint Angles | 2.5034 | 1.0518 | 2.380 | n.s. |
| `Neck_peak` | Joint Angles | 2.4130 | 1.0639 | 2.268 | n.s. |
| `GRF_X_peak` | Force/Torque | 1.8828 | 10.9093 | 0.173 | n.s. |
| `GRF_X_range` | Force/Torque | 13.8490 | 20.6084 | 0.672 | n.s. |
| `GRF_Y_peak` | Force/Torque | 26.3340 | 3.5593 | 7.399 | *** |
| `GRF_Y_range` | Force/Torque | 39.0415 | 4.4292 | 8.815 | *** |
| `GRF_Z_peak` | Force/Torque | 21.3140 | 17.5549 | 1.214 | n.s. |
| `GRF_Z_range` | Force/Torque | 22.5322 | 19.4730 | 1.157 | n.s. |
| `AnkleTorqueMid_Y_peak` | Force/Torque | -0.0681 | 0.0339 | -2.005 | n.s. |

### Requested Variables (Step vs Nonstep)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `θtrunk,max` (`Trunk_peak`) | 8.2162±3.6414 | 7.8527±3.0502 | 0.5205 | n.s. |
| `xCOM/BOS_platformonset` | 0.6318±0.0681 | 0.7005±0.0727 | -0.0513 | *** |
| `xCOM/BOS_steponset` | 0.2817±0.2073 | 0.3668±0.1242 | -0.1008 | *** |

### FDR Significant Variables (p_fdr 순)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|---|---:|---:|---:|---|
| `COP_Y_range` | 0.1034±0.0409 | 0.0484±0.0354 | 0.0594 | *** |
| `GRF_Y_range` | 70.4857±38.3969 | 34.2181±19.4201 | 39.0415 | *** |
| `COP_Y_mean_velocity` | 0.3288±0.1648 | 0.2092±0.1430 | 0.1494 | *** |
| `COP_Y_path_length` | 0.1535±0.0709 | 0.0964±0.0700 | 0.0685 | *** |
| `GRF_Y_peak` | 51.0308±29.5295 | 25.8191±15.1209 | 26.3340 | *** |
| `xCOM_BOS_platformonset` | 0.6318±0.0681 | 0.7005±0.0727 | -0.0513 | *** |
| `xCOM_BOS_steponset` | 0.2817±0.2073 | 0.3668±0.1242 | -0.1008 | *** |
| `COP_Y_peak_velocity` | 1.6746±0.9534 | 1.1466±1.0364 | 0.7236 | *** |
| `MOS_minDist_signed_min` | 0.0335±0.0282 | 0.0489±0.0121 | -0.0141 | *** |
| `MOS_AP_v3d_min` | 0.0383±0.0282 | 0.0532±0.0126 | -0.0136 | *** |
| `vCOM_Y_peak` | 0.0561±0.0294 | 0.0340±0.0239 | 0.0230 | *** |
| `Hip_stance_peak` | 8.2162±3.6414 | 7.8527±3.0502 | 1.2650 | ** |
| `Hip_stance_ROM` | 8.3574±3.6196 | 8.3193±3.9006 | 1.1553 | ** |
| `MOS_ML_v3d_min` | 0.1312±0.0171 | 0.1354±0.0153 | -0.0053 | * |

### 결과 해석

**Balance/Stability (10/19 유의)**
- **AP 방향 (Y축) 변수 집중 유의**: COP_Y_range, COP_Y_path_length, COP_Y_peak_velocity, COP_Y_mean_velocity, vCOM_Y_peak 모두 step > nonstep. step 전략 시 전후 방향 균형 조절이 더 크다.
- **MoS 감소**: MOS_minDist_signed_min, MOS_AP_v3d_min, MOS_ML_v3d_min 모두 step < nonstep (음수 estimate). step 전략은 stability margin이 더 적다.
- **xCOM/BOS (AP)**: platform onset과 step onset 시점 모두 step < nonstep. step 전략에서 xCOM이 BOS 내에서 상대적으로 전방에 위치(값이 작음).
- **AP 방향 (X축) 변수**: COM_X, COP_X 관련 변수는 모두 비유의. 내외측 방향이 주요 차이 방향.

**Joint Angles (2/10 유의)**
- **Hip만 유의**: Hip_stance_ROM, Hip_stance_peak 모두 step > nonstep. Hip 전략 사용이 step에서 더 크다.
- Trunk_peak (θtrunk,max)는 step>nonstep 경향이나 FDR 비유의 (t=1.036).
- Ankle, Knee는 비유의.

**Force/Torque (2/7 유의)**
- **GRF_Y만 유의**: GRF_Y_peak(step=51.0N vs nonstep=25.8N), GRF_Y_range 모두 step >> nonstep. AP 방향 지면반력이 step에서 약 2배.
- GRF_X, GRF_Z, AnkleTorque는 비유의.

## Van Wouwe et al. - 2021 통합 비교

이번 비교는 1:1 재현이 아니라, 요청하신 step/non-step 분리 관점으로 재해석한 대응표다.

| 항목 | Van Wouwe et al. - 2021 | 현재 구현 | 현재 결과 | 판정 |
|---|---|---|---|---|
| 분석 단위 | subject별 variability + robust regression | pooled trial LMM (`DV ~ step_TF + (1|subject)`) | 집단 평균 차이 추정 | 구조 차이 |
| `θtrunk,max` | 전략 변화(ankle/hip/step)의 핵심 지표 | `Trunk_peak=max(\|Trunk_X_deg\|)`로 대응 | step>nonstep 경향이나 FDR 비유의 (t=1.036) | Partially consistent |
| `xCOM/BOS_onset` 대응 | anterior initial COM가 stepping 경향과 연관 | `xCOM_BOS_platformonset` | step(0.6318) < nonstep(0.7005), `***` | Inconsistent |
| `xCOM/BOS_300ms` 대응 | later response 전략 관련 핵심 변수 | step=actual onset, nonstep=onset+300ms (`xCOM_BOS_steponset`) | step(0.2817) < nonstep(0.3668), `***` | Partially consistent |
| 결론 레벨 | 초기 자세 + task-level goal 상호작용 강조 | goal parameter 직접 모델링 없음 | step/nonstep 집단 차이 중심 | 비교 범위 제한 |

## Conclusion

1. 본 데이터에서 step/non-step 분리는 **14/36** DV에서 FDR 유의했다.
2. 유의 변수는 AP 방향 COP/COM/GRF 및 MoS에 집중되어, step 전략이 전후 방향 균형 반응 크기와 stability margin 감소에서 특징적으로 구분됨을 확인했다. 새로 추가된 COP mean velocity (path_length / window_duration) 역시 AP 방향(Y축)에서 유의했다.
3. 요청 변수 3개 중 `xCOM/BOS_platformonset`, `xCOM/BOS_steponset`은 유의했고, `θtrunk,max(=Trunk_peak)`는 비유의였다.
4. Joint 관절 중 Hip만 유의하여, step 전략에서 hip strategy 사용이 더 크다는 해석이 가능하다.
5. Van Wouwe 2021과의 비교는 가능하지만, 통계 단위가 달라 완전한 1:1 대응은 아니다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py
```

- `--no-figures` 옵션으로 figure 없이 통계만 재현 가능.

## Figures

| File | Description |
|---|---|
| `fig1_lmm_forest_plot.png` | LMM estimate ± 95% CI |
| `fig2_violin_significant.png` | Significant DV violin/strip |
| `fig3_descriptive_heatmap.png` | z-scored group mean heatmap |
