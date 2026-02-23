# Step vs. Non-step Biomechanical LMM Analysis

## Research Question

**"동일 perturbation 강도에서 step과 non-step 균형회복 전략 간 biomechanical 변수에 유의한 차이가 있는가?"**

Platform translation perturbation 실험에서 동일한 mixed velocity 조건 하에 step과 non-step이 공존하는 상황에서, COM, COP, MoS, 관절 각도, GRF, ankle torque 등 biomechanical 변수를 Linear Mixed Model(LMM)로 비교 분석하였다.

## key papers

### Van Wouwe et al. - 2021

- Methodology:
  - 10명 healthy young adults를 대상으로 backward support-surface translation perturbation 수행
  - perturbation onset은 platform velocity `2 cm/s` threshold로 정의
  - 핵심 변수는 `htrunk,max`(maximal trunk lean), `xCOM/BOS_onset`, `xCOM/BOS_300ms`
  - 분석은 subject별 robust regression 기반으로, `xCOM/BOS`와 trunk strategy 간 관계를 평가
- Key results:
  - trunk lean variability: within-subject range 평균 `28.3°`, across-subject mean range `39.9°`
  - early response에서 `xCOM/BOS_onset`와 `xCOM/BOS_300ms`는 유의한 양의 관계 (`R²=0.53–0.72`, `P<0.001`)
  - later response에서 `xCOM/BOS_300ms`와 `htrunk,max`의 관계는 subject-specific slope (`R²=0.29–0.81`, `P<0.01`)
- Authors' conclusion:
  - 초기 자세(특히 anterior COM)가 hip/stepping strategy 사용 증가와 연결되고,
  - task-level goal(stability vs effort)와의 상호작용이 개인 간 반응 차이를 설명한다.

## Data Summary

- **125 trials** (step=53, nonstep=72) from 24 subjects
- Data source: `output/all_trials_timeseries.csv` (frame-level 100Hz timeseries) + `data/perturb_inform.xlsm` (step/nonstep classification)
- **34 dependent variables** across 3 variable families
- `end_frame` 산출 규칙: `(subject, velocity)` step 평균 우선, 불가 시 필터링 전 `platform` step trial의 동일 `(subject, velocity)` 평균 fallback 적용 (이번 실행에서 5 nonstep trials 보정)
- Joint angle(X plane) side 규칙: `step_R/step_L`를 stepping foot로 해석하고, step trial은 반대측(stance), nonstep/footlift trial은 subject별 `major_step_side`(산출 기준: `mixed==1` step trial) 반대측을 사용한다. `major_step_side`가 tie면 `(L+R)/2` 평균을 사용한다.

### Analysis Window

분석 시간 구간은 trial마다 개별적으로 설정하였다:

- **분석 시작**: `platform_onset_local` — 각 trial에서 perturbation platform이 움직이기 시작한 MocapFrame
- **분석 종료**: `end_frame` — step onset 기반으로 trial별 산출
  - **Step trial**: 해당 trial의 실제 `step_onset_local` (발이 지면에서 처음 떨어지는 시점)
  - **Nonstep trial**:
    1. 동일 `(subject, velocity)` 내 step trial들의 `step_onset_local` 평균값 대입
    2. 위 값이 없으면 `platform` 시트(필터링 전 step trial 전체)의 동일 `(subject, velocity)` 평균 `step_onset_local` 대입

이 방식은 step 발생 직전까지의 CPA(Compensatory Postural Adjustment) 구간에 집중하기 위한 것이다. 고정 시간 창(예: 0–800ms)은 perturbation 강도와 피험자에 따라 step onset timing이 다르므로 적합하지 않다. Nonstep trial에 동일 `(subject, velocity)` 평균을 우선 적용하고, 해당 값이 없을 때도 필터링 전 `platform` step trial의 동일 `(subject, velocity)` 평균을 사용해 global 평균 유입 없이 시간 구조 비교 가능성을 유지하였다.

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
| Joint angles (X/Y/Z) | X: `+Flex / -Ext` (ankle X: `+Dorsi / -Plantar`), Y: `+Add / -Abd`, Z: `+IR / -ER` | Left Y/Z는 sign-unification(부호 반전) 적용, 본 분석의 Hip/Knee/Ankle X는 `stance-equivalent`로 재구성 |
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
- **value_at_event** (event frame value): 지정 이벤트 프레임 값 추출
  - `xCOM_BOS_platformonset`: `platform_onset_local` 시점
  - `xCOM_BOS_steponset`: step trial은 실제 step onset, nonstep trial은 `platform_onset_local + 300ms(30 frames)`
  - `xCOM/BOS = (xCOM_X - BOS_minX)/(BOS_maxX - BOS_minX)`

---

## Results

### 1. LMM Summary

Statistical method: Linear Mixed Model via R lmerTest
- Model: `DV ~ step_TF + (1|subject)`
- Estimation: REML (restricted maximum likelihood)
- Inference: lmerTest fixed-effect inference
- Multiple comparison: Benjamini-Hochberg FDR per variable family (`Sig` only, α=0.05)

**Overall: 13/34 variables showed FDR-significant differences (Sig at α=0.05)**

### 2. Balance/Stability Family (17 variables, 9 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|----------|-------------|-----------------|----------|-----|
| vCOM_Y_peak | 0.0561±0.0294 | 0.0340±0.0239 | 0.0230 | *** |
| COP_Y_range | 0.1034±0.0409 | 0.0484±0.0354 | 0.0594 | *** |
| COP_Y_path_length | 0.1535±0.0709 | 0.0964±0.0700 | 0.0685 | *** |
| COP_Y_peak_velocity | 1.6746±0.9534 | 1.1466±1.0364 | 0.7236 | *** |
| MOS_minDist_signed_min | 0.0335±0.0282 | 0.0489±0.0121 | -0.0141 | *** |
| MOS_AP_v3d_min | 0.0383±0.0282 | 0.0532±0.0126 | -0.0136 | *** |
| MOS_ML_v3d_min | 0.1312±0.0171 | 0.1354±0.0153 | -0.0053 | * |
| xCOM_BOS_platformonset | 0.6318±0.0681 | 0.7005±0.0727 | -0.0513 | *** |
| xCOM_BOS_steponset | 0.2817±0.2073 | 0.3668±0.1242 | -0.1008 | *** |
| COP_X_range | — | — | 0.0119 | n.s. |
| COM_X_range | — | — | -0.0015 | n.s. |
| COM_X_path_length | — | — | 0.0005 | n.s. |
| vCOM_X_peak | — | — | -0.0001 | n.s. |
| COM_Y_range | — | — | 0.0009 | n.s. |
| COM_Y_path_length | — | — | 0.0012 | n.s. |
| COP_X_path_length | — | — | 0.0349 | n.s. |
| COP_X_peak_velocity | — | — | 2.3092 | n.s. |

### 3. Joint Angles Family (10 variables, 2 significant)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|----------|-------------|-----------------|----------|-----|
| Hip_stance_ROM | 8.3574±3.6196 | 8.3193±3.9006 | 1.1553 | ** |
| Hip_stance_peak | 8.2162±3.6414 | 7.8527±3.0502 | 1.2650 | ** |
| Knee_stance_ROM | — | — | 0.6453 | n.s. |
| Knee_stance_peak | — | — | 0.5509 | n.s. |
| Ankle_stance_ROM | — | — | 0.1893 | n.s. |
| Ankle_stance_peak | — | — | 0.1941 | n.s. |
| Neck_ROM | — | — | 2.5034 | n.s. |
| Neck_peak | — | — | 2.4130 | n.s. |
| Trunk_ROM | — | — | 0.7324 | n.s. |
| Trunk_peak | — | — | 0.5205 | n.s. |

### 4. Force/Torque Family (7 variables, 2 significant)

| Variable | Estimate | Sig |
|----------|----------|-----|
| GRF_X_peak | 1.8828 | n.s. |
| GRF_X_range | 13.8490 | n.s. |
| GRF_Y_peak | 26.3340 | *** |
| GRF_Y_range | 39.0415 | *** |
| GRF_Z_peak | 21.3140 | n.s. |
| GRF_Z_range | 22.5322 | n.s. |
| AnkleTorqueMid_Y_peak | -0.0681 | n.s. |

---

## Interpretation

해석 섹션의 AP/ML/inside-outside 방향 문구는 위 `Coordinate & Sign Conventions` 표의 `(+)/(-)` 정의를 따른다.

### Balance & Stability

platform onset ~ step onset 구간에서 step 시행은 ML(좌우) 방향 COM 속도(vCOM_Y_peak)와 COP 변위/속도(COP_Y_range, COP_Y_path_length, COP_Y_peak_velocity)가 유의하게 크다. 이는 stepping 직전까지의 CPA 구간에서 ML 방향 자세 동요가 step 유발의 핵심 특징임을 시사한다.

COP_X 계열(COP_X_range, COP_X_path_length, COP_X_peak_velocity)은 현재 데이터에서 유의하지 않았다. 따라서 AP 방향 COP 보상 차이는 본 분석 구간에서 일관된 분별 신호로 확인되지 않았다.

MoS 최소값은 step에서 유의하게 낮다 (MOS_minDist_signed_min: step=0.0335 vs nonstep=0.0489; MOS_AP_v3d_min: step=0.0383 vs nonstep=0.0532; MOS_ML_v3d_min: step=0.1312 vs nonstep=0.1354). 이는 step onset 직전까지의 구간에서 step 시행이 더 불안정한 상태에 도달함을 확인한다.

이번 업데이트에서 추가한 `xCOM_BOS` 지표도 step/nonstep를 유의하게 구분했다. `xCOM_BOS_platformonset`(step=0.6318, nonstep=0.7005)과 `xCOM_BOS_steponset`(step=0.2817, nonstep=0.3668)은 모두 step에서 더 낮았다.

### Joint Angles

Stance-equivalent 기준으로는 Hip ROM/peak만 step에서 유의하게 컸다 (Hip_stance_ROM: 8.36° vs 8.32°, Hip_stance_peak: 8.22° vs 7.85°). Knee/Ankle 및 Trunk/Neck 변수는 FDR 기준에서 유의하지 않았다.

### Force/Torque

[platform_onset, step_onset] 구간에서 GRF_Y_peak와 GRF_Y_range는 step에서 유의하게 크다. 반면 GRF_X/GRF_Z 계열과 ankle torque는 유의하지 않았다. 즉, 수직축/전후축 지면반력보다는 ML 방향 지면반력 변동이 step 분별에 더 직접적으로 연결된다.

### Van Wouwe 2021 비교 (step vs nonstep 중심)

Van Wouwe et al. (2021)의 실험/모델 구조를 본 데이터에 1:1로 재현하기는 어렵다(perturbation protocol, 분석모델, 목표함수 설정 차이). 따라서 아래 비교는 핵심 변수의 방향성과 step vs nonstep 분리 관점에서만 수행했다.

| Comparison Item | Van Wouwe et al. - 2021 | Current step vs nonstep result | Verdict |
|-----------------|--------------------------|---------------------------------|---------|
| `θtrunk,max` (`htrunk,max`) | larger trunk lean은 hip/stepping strategy 증가와 연결 | `Trunk_peak(=max|Trunk_X_deg|)`: step=8.2162, nonstep=7.8527, Estimate=0.5205, n.s. | Partially consistent |
| `xCOM/BOS_platformonset` | anterior initial COM(`xCOM/BOS_onset`↑)는 stepping 경향 증가와 연관 | `xCOM_BOS_platformonset`: step=0.6318, nonstep=0.7005, Estimate=-0.0513, `***` | Inconsistent |
| `xCOM/BOS_steponset` 대응 항목 | 원문은 `xCOM/BOS_300ms`를 사용해 later response strategy와 연계 | 본 분석은 step=실제 step onset, nonstep=onset+300ms로 정의. `xCOM_BOS_steponset`: step=0.2817, nonstep=0.3668, Estimate=-0.1008, `***` | Partially consistent |

해석상 핵심은 다음과 같다. 원문은 subject별 회귀 기울기(continuous strategy) 중심이고, 본 분석은 pooled LMM의 집단 평균 차이(step vs nonstep) 중심이다. 따라서 동일 변수명이라도 통계적 의미가 완전히 같지는 않다.

### Conclusion

1. **13/34 biomechanical 변수가 [platform_onset, step_onset] 구간에서 FDR-유의**하였다.
2. **ML 방향 COM/COP 지표(vCOM_Y_peak, COP_Y_range, COP_Y_path_length, COP_Y_peak_velocity)**가 step의 핵심 특징으로 확인되었다.
3. **MoS 최소값(MOS_minDist_signed_min, MOS_AP_v3d_min, MOS_ML_v3d_min)은 step에서 유의하게 낮아** step onset 직전 불안정성이 더 크다.
4. **Joint family에서는 Hip_stance ROM/peak만 유의**했고, Knee/Ankle/Trunk/Neck는 FDR 기준 비유의였다.
5. **Force/Torque family에서는 GRF_Y_peak, GRF_Y_range만 유의**했고, GRF_X/GRF_Z 및 ankle torque는 비유의였다.
6. **요청 변수 중 `xCOM_BOS_platformonset`, `xCOM_BOS_steponset`은 유의했고, `θtrunk,max(=Trunk_peak)`는 비유의**였다.

---

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py --no-figures
```

**Input**: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
**Output**: stdout statistics (`--no-figures` 실행 시 figure 파일 미갱신)

## Figures

이번 업데이트에서는 `--no-figures`로 실행하여 기존 figure 파일을 재생성/갱신하지 않았다.

| File | Description |
|------|-------------|
| fig1_lmm_forest_plot.png | Forest plot: LMM estimate ± 95% CI for all DVs, family별 색상, FDR 유의 항목 강조 |
| fig2_violin_significant.png | FDR-significant 변수의 violin + strip plot (step vs nonstep) |
| fig3_descriptive_heatmap.png | z-scored group mean heatmap with significance markers |
