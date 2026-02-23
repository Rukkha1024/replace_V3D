# Step vs. Non-step Biomechanical LMM Analysis

## Research Question

**"동일 perturbation 강도에서 step과 non-step 균형회복 전략 간 biomechanical 변수에 유의한 차이가 있는가?"**

동일 mixed velocity 조건에서 step/non-step 전략이 공존하는 trial을 대상으로, pre-step 구간의 biomechanical 지표를 LMM으로 비교했다.

## Data Summary

- Trials: **125** (step=53, nonstep=72), subjects=24
- Input: `output/all_trials_timeseries.csv` + `data/perturb_inform.xlsm`
- DVs: **34** (Balance/Stability 17, Joint Angles 10, Force/Torque 7)
- Analysis window: `[platform_onset_local, end_frame]`
  - step: `end_frame = actual step_onset_local`
  - nonstep: `(subject, velocity)` step 평균 onset 우선, 불가 시 prefilter `platform` subject-velocity 평균 fallback

## Analysis Methodology

- Model: `DV ~ step_TF + (1|subject)`
- Estimation: REML (`lmerTest`)
- Multiple comparison: BH-FDR, family-wise, `alpha=0.05`
- Sig rule: `*** (<0.001)`, `** (<0.01)`, `* (<0.05)`, `n.s.`

추가 변수 정의(요청 반영):
- `θtrunk,max`는 본 파이프라인의 `Trunk_peak = max(|Trunk_X_deg|)`와 동일하게 사용
- `xCOM_BOS_platformonset = (xCOM_X - BOS_minX)/(BOS_maxX - BOS_minX)` at `platform_onset_local`
- `xCOM_BOS_steponset`:
  - step: actual `step_onset_local`
  - nonstep: `platform_onset_local + 300 ms (30 frames @100Hz)`

## Results

### 1) LMM Overall

- **FDR significant: 13/34**
- 신규 요청 변수 결과:
  - `xCOM_BOS_platformonset`: Estimate=-0.0513, `***`
  - `xCOM_BOS_steponset`: Estimate=-0.1008, `***`
  - `Trunk_peak(=θtrunk,max)`: Estimate=0.5205, `n.s.`

### 2) Family-wise Summary Table

| Family | Total DVs | Significant DVs | Key significant variables |
|---|---:|---:|---|
| Balance/Stability | 17 | 9 | `vCOM_Y_peak`, `COP_Y_range`, `COP_Y_path_length`, `COP_Y_peak_velocity`, `MOS_minDist_signed_min`, `MOS_AP_v3d_min`, `MOS_ML_v3d_min`, `xCOM_BOS_platformonset`, `xCOM_BOS_steponset` |
| Joint Angles | 10 | 2 | `Hip_stance_ROM`, `Hip_stance_peak` |
| Force/Torque | 7 | 2 | `GRF_Y_peak`, `GRF_Y_range` |

### 3) Requested Variable Table (Step vs Nonstep)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step-nonstep) | Sig |
|---|---:|---:|---:|---|
| `θtrunk,max` (`Trunk_peak`) | 8.2162±3.6414 | 7.8527±3.0502 | 0.5205 | n.s. |
| `xCOM/BOS_platformonset` (`xCOM_BOS_platformonset`) | 0.6318±0.0681 | 0.7005±0.0727 | -0.0513 | *** |
| `xCOM/BOS_steponset` (`xCOM_BOS_steponset`) | 0.2817±0.2073 | 0.3668±0.1242 | -0.1008 | *** |

## Van Wouwe et al. - 2021 통합 비교 (단일 섹션)

1:1 재현이 아니라, **step vs nonstep 분리 관점**으로 핵심 항목만 대응했다.

| 항목 | Van Wouwe et al. - 2021 | Current Implementation (this report) | Current Result | 판정 |
|---|---|---|---|---|
| 실험/모델 단위 | subject별 posture-variability + robust regression | pooled trial LMM (`DV ~ step_TF + (1|subject)`) | group 평균 차이 추정 | 구조 차이 있음 |
| `θtrunk,max` | 전략(ankle/hip/step) 변화의 핵심 지표 | `Trunk_peak=max(|Trunk_X_deg|)`로 대응 | step>nonstep 경향이나 FDR 비유의 (`n.s.`) | Partially consistent |
| `xCOM/BOS_onset` 대응 | anterior initial COM 증가 시 stepping 경향 증가 보고 | `xCOM_BOS_platformonset`으로 대응 | step(0.6318) < nonstep(0.7005), Estimate=-0.0513, `***` | Inconsistent |
| `xCOM/BOS_300ms` 대응 | later response 전략과 연계되는 핵심 변수 | step=actual onset, nonstep=onset+300ms로 대응(`xCOM_BOS_steponset`) | step(0.2817) < nonstep(0.3668), Estimate=-0.1008, `***` | Partially consistent |
| 원문 결론 대응 | 초기 자세 + task-level goal 상호작용이 개인차 설명 | 본 분석은 goal parameter 직접 모델링 없음 | step/nonstep 집단 차이만 직접 검정 | 비교 범위 제한 |

## Conclusion

1. 본 데이터에서 step/nonstep 분리는 `13/34` DV에서 FDR 유의했다.
2. 요청 변수 중 `xCOM_BOS_platformonset`, `xCOM_BOS_steponset`은 유의했고, `θtrunk,max(=Trunk_peak)`는 유의하지 않았다.
3. Van Wouwe 2021과 비교 시, 변수명 유사성은 있지만 통계 단위(개인별 기울기 vs 집단 평균 차이)가 달라 완전 대응은 불가능하다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py --no-figures
```

- `--no-figures` 기준으로 검증했으며 기존 fig 파일은 갱신하지 않았다.

## Figures

| File | Description |
|---|---|
| `fig1_lmm_forest_plot.png` | LMM estimate ± 95% CI |
| `fig2_violin_significant.png` | significant DV violin/strip |
| `fig3_descriptive_heatmap.png` | z-scored group mean heatmap |
