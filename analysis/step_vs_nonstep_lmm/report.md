# Step vs. Non-step Biomechanical LMM Analysis

## Research Question

**"동일 perturbation 강도에서 step과 non-step 균형회복 전략 간 biomechanical 변수 차이가 있는가?"**

이번 분석은 step/non-step가 함께 존재하는 mixed velocity trial에서, pre-step 구간의 biomechanical 반응을 LMM으로 비교하는 목적이다.

## Data & Model

- Trials: **125** (step=53, nonstep=72), subjects=24
- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- DVs: **34개** (Balance/Stability 17, Joint Angles 10, Force/Torque 7)
- Window: `[platform_onset_local, end_frame]`
  - step: `end_frame = actual step_onset_local`
  - nonstep: subject-velocity step 평균 onset, 불가 시 prefilter `platform` fallback
- Model: `DV ~ step_TF + (1|subject)`, REML (`lmerTest`)
- Multiple comparison: BH-FDR (family-wise, alpha=0.05)

요청 변수 정의:
- `θtrunk,max`는 `Trunk_peak = max(|Trunk_X_deg|)`로 대응
- `xCOM/BOS_platformonset = (xCOM_X - BOS_minX)/(BOS_maxX - BOS_minX)` at `platform_onset_local`
- `xCOM/BOS_steponset`
  - step: actual step onset
  - nonstep: `platform_onset_local + 300 ms (30 frames)`

## Results

### Overall

- **FDR significant: 13/34**

| Family | Total DVs | Significant DVs |
|---|---:|---:|
| Balance/Stability | 17 | 9 |
| Joint Angles | 10 | 2 |
| Force/Torque | 7 | 2 |

### Requested Variables (Step vs Nonstep)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step-nonstep) | Sig |
|---|---:|---:|---:|---|
| `θtrunk,max` (`Trunk_peak`) | 8.2162±3.6414 | 7.8527±3.0502 | 0.5205 | n.s. |
| `xCOM/BOS_platformonset` (`xCOM_BOS_platformonset`) | 0.6318±0.0681 | 0.7005±0.0727 | -0.0513 | *** |
| `xCOM/BOS_steponset` (`xCOM_BOS_steponset`) | 0.2817±0.2073 | 0.3668±0.1242 | -0.1008 | *** |

### 주요 유의 변수 요약

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|---|---:|---:|---:|---|
| `vCOM_Y_peak` | 0.0561±0.0294 | 0.0340±0.0239 | 0.0230 | *** |
| `COP_Y_range` | 0.1034±0.0409 | 0.0484±0.0354 | 0.0594 | *** |
| `MOS_AP_v3d_min` | 0.0383±0.0282 | 0.0532±0.0126 | -0.0136 | *** |
| `GRF_Y_peak` | 51.0308±29.5295 | 25.8191±15.1209 | 26.3340 | *** |
| `Hip_stance_peak` | 8.2162±3.6414 | 7.8527±3.0502 | 1.2650 | ** |

## Van Wouwe et al. - 2021 통합 비교 (단일 섹션)

이번 비교는 1:1 재현이 아니라, 요청하신 step/non-step 분리 관점으로 재해석한 대응표다.

| 항목 | Van Wouwe et al. - 2021 | 현재 구현 | 현재 결과 | 판정 |
|---|---|---|---|---|
| 분석 단위 | subject별 variability + robust regression | pooled trial LMM (`DV ~ step_TF + (1|subject)`) | 집단 평균 차이 추정 | 구조 차이 |
| `θtrunk,max` | 전략 변화(ankle/hip/step)의 핵심 지표 | `Trunk_peak=max(|Trunk_X_deg|)`로 대응 | step>nonstep 경향이나 FDR 비유의 | Partially consistent |
| `xCOM/BOS_onset` 대응 | anterior initial COM가 stepping 경향과 연관 | `xCOM_BOS_platformonset` | step(0.6318) < nonstep(0.7005), `***` | Inconsistent |
| `xCOM/BOS_300ms` 대응 | later response 전략 관련 핵심 변수 | step=actual onset, nonstep=onset+300ms (`xCOM_BOS_steponset`) | step(0.2817) < nonstep(0.3668), `***` | Partially consistent |
| 결론 레벨 | 초기 자세 + task-level goal 상호작용 강조 | goal parameter 직접 모델링 없음 | step/nonstep 집단 차이 중심 | 비교 범위 제한 |

## Conclusion

1. 본 데이터에서 step/non-step 분리는 `13/34` DV에서 FDR 유의했다.
2. 요청 변수 3개 중 `xCOM/BOS_platformonset`, `xCOM/BOS_steponset`은 유의했고, `θtrunk,max(=Trunk_peak)`는 비유의였다.
3. Van Wouwe 2021과의 비교는 가능하지만, 통계 단위가 달라 완전한 1:1 대응은 아니다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py --no-figures
```

- `--no-figures` 기준 검증으로 기존 figure 파일은 갱신하지 않았다.

## Figures

| File | Description |
|---|---|
| `fig1_lmm_forest_plot.png` | LMM estimate ± 95% CI |
| `fig2_violin_significant.png` | Significant DV violin/strip |
| `fig3_descriptive_heatmap.png` | z-scored group mean heatmap |
