# xCOM/BOS Normalization 기반 Step vs Nonstep LMM 분석

## Research Question

**"methods_list.md의 xCOM/BOS 정규화 공식을 현재 데이터에 적용했을 때, step vs nonstep 전략 차이를 LMM으로 검증할 수 있는가?"**

본 분석은 `analysis/xCOM&BOS_normalization/methods_list.md`의 논문별 공식을 현재 데이터 구조에 맞춰 통합 구현하고, 같은 섭동 조건에서 step/nonstep 전략 차이를 통계적으로 검정하기 위해 수행되었다.

## Prior Studies

### Van Wouwe et al. (2021) — Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance

- **Methodology**: Hof 공식 `xCOM = COM + vCOM/sqrt(g/l)`을 사용하고 `xCOM/BOS`를 지표화하여 onset 및 300 ms 시점 안정성을 평가.
- **Experimental design**: standing backward perturbation 과제에서 trial-by-trial variability와 early response(0-300 ms) 안정성 비교.
- **Key results**: 개인별 초기 자세와 trunk lean 관계가 보고되었고, subject-specific 설명력은 `R² = 0.29-0.82` 범위.
- **Conclusions**: 초기 자세와 task-level goal의 상호작용이 전략 variability를 설명.

### Salot et al. (2016) — Reactive Balance in Individuals With Chronic Stroke

- **Methodology**: `XCOM = xCOM + vCOM/sqrt(g/l)` 기반으로 `XCOM/BOS`를 foot length로 정규화하고 step lift-off/touchdown 시점에서 비교.
- **Experimental design**: chronic stroke vs healthy의 **2그룹** 비교, backward loss-of-balance 상황에서 stepping response 평가.
- **Key results**: step 이벤트 시점에서 posterior 안정성 지표(`XCOM/BOS`)가 그룹 차이를 보였고, foot-length normalization을 통해 inter-subject scale 차이를 보정.
- **Conclusions**: foot-length normalized XCOM/BOS는 집단 간 반응 전략 차이를 비교하는 유용한 지표.

### Joshi et al. (2018) — Reactive balance to unanticipated trip-like perturbations

- **Methodology**: 위치 지표 `XCOM/BOS = (COM - BOS)/foot_length`와 속도 지표 `VCOM/BOS = (COM_vel - BOS_vel)/sqrt(g*height)`를 병행 분석.
- **Experimental design**: young control, older control, stroke의 **3그룹**에서 lift-off/touchdown 시점 반응을 비교.
- **Key results**: 위치와 속도 정규화 지표를 분리해 해석하며, 속도항 정규화에 `sqrt(g*h)`를 사용.
- **Conclusions**: 위치 기반과 속도 기반 정규화를 함께 보는 접근이 stepping 위험 해석에 유효.

## Methodological Adaptation

| Prior Study Method | Current Implementation | Deviation Rationale |
|---|---|---|
| Hof 기반 xCOM + BOS 정규화 (Van Wouwe) | `DV1 = (xCOM_hof - BOS_rear) / foot_length` | 현재 데이터에는 논문과 동일한 모델 기반 BOS 정의(OpenSim ankle-to-toe)가 없으므로, 동일 의미의 rear-BOS 대비 정규화로 통합 |
| foot length 정규화 (Salot/Patel/Bhatt) | `meta` 시트의 `발길이_왼/오른` 평균(mm→m) 사용 | `transpose_meta`는 누락/0값이 있어 제외하고 `meta` 행-기반 원본에서 복원 추출 |
| Joshi 위치 지표 `(COM-BOS)/foot_length` | `DV2 = (COM_X - BOS_minX)/foot_length` | XCOM 대신 COM 위치항만 분리해 위치항 기여를 명시적으로 확인 |
| Joshi 속도 지표 `(vCOM-vBOS)/sqrt(g*h)` | `DV3 = (vCOM_X - vBOSrear)/sqrt(g*height)` | CSV에 BOS rear 속도가 직접 없으므로 `BOS_minX` 차분(100 Hz)으로 계산 |
| 논문 이벤트(lift-off/touchdown, onset, 300ms) | `platform onset`, `step onset` 2개 이벤트로 고정 | 사용자 확정 요구사항에 따라 이벤트 수를 2개로 제한 |
| 통계 모델 | `DV ~ step_TF * velocity_c + (1|subject)` | 반복측정 구조(24명, 125 trial)를 반영하기 위해 LMM 유지 + velocity 중심화 |

**Summary**: 본 분석은 Van Wouwe/Salot/Joshi 계열 공식을 현재 데이터에 맞게 통합(`DV1~DV3`)하고, `platform onset`/`step onset` 이벤트에서 step/nonstep 차이를 상호작용 LMM으로 검정했다.

## Data Summary

- Trials: **125** (`step=53`, `nonstep=72`)
- Subjects: **24**
- Frames: **29,038**
- Input data:
  - `output/all_trials_timeseries.csv`
  - `data/perturb_inform.xlsm` (`platform`, `meta`)
- Velocity summary: mean `58.28`, SD `43.68`
- Final DVs: **6개** (3개 공식 × 2개 이벤트)

## Analysis Methodology

- **Analysis events**: `platform_onset_local`, `step_onset_eval`
- **step_onset_eval policy**:
  - step: `step_onset_local`
  - missing step onset: same `(subject, velocity)` mean
  - residual fallback: prefilter `platform` event 기반 subject-velocity mean
- **Statistical model**: `DV ~ step_TF * velocity_c + (1|subject)` (REML, `lmerTest`)
- **Multiple comparison correction**: BH-FDR, term별 분리 보정(`main_step_effect`, `interaction_step_x_velocity`)
- **Significance reporting**: `Sig` only (`***`, `**`, `*`, `n.s.`, alpha=0.05)
- **Normalization inputs**:
  - `foot_len_m = ((발길이_왼 + 발길이_오른)/2) / 1000`
  - `height_m = 키 / 100`
  - `leg_len_m = 다리길이 / 100`

### Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|---|---|---|---|
| X (AP) | 전방(anterior) | 후방(posterior) | `COM_X`, `vCOM_X`, `BOS_minX`, `xCOM_hof` |
| Y (ML) | 우측(lateral) | 좌측(medial) | (본 분석 직접 사용 없음) |
| Z (Vertical) | 상방(up) | 하방(down) | (본 분석 직접 사용 없음) |

### Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|---|---|---|---|
| `DV1=(xCOM_hof-BOS_rear)/foot_len` | xCOM가 BOS rear 기준 전방 | xCOM가 BOS rear 기준 후방 | step-nonstep LMM 계수 부호 |
| `DV2=(COM_X-BOS_rear)/foot_len` | COM이 BOS rear 기준 전방 | COM이 BOS rear 기준 후방 | step-nonstep LMM 계수 부호 |
| `DV3=(vCOM_X-vBOSrear)/sqrt(g*h)` | 상대 전방 속도 증가 | 상대 후방 속도 증가 | step-nonstep LMM 계수 부호 |

### Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|---|---|---|
| xCOM/BOS normalized metrics (`DV1~DV3`) | AP축 기반 부호 해석 | 모두 dimensionless 정규화 값 |
| Joint angles | 본 분석 미사용 | 해당 없음 |
| Force/Torque | 본 분석 미사용 | 해당 없음 |

### Analyzed Variables (Full Set)

| Variable | Formula family | Event | Result status (main / interaction) |
|---|---|---|---|
| `DV1_xcom_hof_rear_over_foot_platformonset` | VanWouwe+Salot+Patel+Bhatt | platform_onset | `*** / n.s.` |
| `DV1_xcom_hof_rear_over_foot_steponset` | VanWouwe+Salot+Patel+Bhatt | step_onset | `*** / n.s.` |
| `DV2_com_rear_over_foot_platformonset` | Joshi_position | platform_onset | `*** / n.s.` |
| `DV2_com_rear_over_foot_steponset` | Joshi_position | step_onset | `** / n.s.` |
| `DV3_vcom_rel_over_sqrtgh_platformonset` | Joshi_velocity | platform_onset | `n.s. / n.s.` |
| `DV3_vcom_rel_over_sqrtgh_steponset` | Joshi_velocity | step_onset | `*** / n.s.` |

## Results

### 1. Main Effect (`step_TFstep`) Results

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|---|---:|---:|---:|---|
| `DV1_xcom_hof_rear_over_foot_platformonset` | 0.5091±0.0596 | 0.5715±0.0654 | -0.0453 | *** |
| `DV1_xcom_hof_rear_over_foot_steponset` | 0.2354±0.1720 | 0.3232±0.0996 | -0.0945 | *** |
| `DV2_com_rear_over_foot_platformonset` | 0.4976±0.0583 | 0.5525±0.0627 | -0.0411 | *** |
| `DV2_com_rear_over_foot_steponset` | 0.2648±0.1109 | 0.2872±0.0745 | -0.0257 | ** |
| `DV3_vcom_rel_over_sqrtgh_platformonset` | -0.0245±0.0159 | -0.0291±0.0195 | -0.0002 | n.s. |
| `DV3_vcom_rel_over_sqrtgh_steponset` | 0.0684±0.0429 | -0.0039±0.0184 | 0.0712 | *** |

- Main effect 유의 계수: **5/6**

### 2. Interaction Effect (`step_TFstep:velocity_c`) Results

| Variable | Interaction Estimate | Sig |
|---|---:|---|
| `DV1_xcom_hof_rear_over_foot_platformonset` | -0.0004 | n.s. |
| `DV1_xcom_hof_rear_over_foot_steponset` | -0.0006 | n.s. |
| `DV2_com_rear_over_foot_platformonset` | -0.0004 | n.s. |
| `DV2_com_rear_over_foot_steponset` | -0.0005 | n.s. |
| `DV3_vcom_rel_over_sqrtgh_platformonset` | 0.0000 | n.s. |
| `DV3_vcom_rel_over_sqrtgh_steponset` | -0.0001 | n.s. |

- Interaction 유의 계수: **0/6**

### 3. Overall

- 전체 계수 기준 FDR 유의: **5/12**
- 유의성은 대부분 `main_step_effect`에서 관찰되었고, `step×velocity` 상호작용은 본 데이터에서 유의하지 않았다.

## Comparison with Prior Studies

| Comparison Item | Prior Study Result | Current Result | Verdict |
|---|---|---|---|
| Hof 기반 XCOM/BOS 지표의 전략 구분력 | onset/early 시점 안정성 지표로 유효 | DV1이 onset/step_onset 모두 유의(`***`) | Consistent |
| COM 위치항 정규화의 차이 | 위치항 정규화(`(COM-BOS)/foot`) 사용 보고 | DV2가 두 이벤트 모두 유의(`***`, `**`) | Consistent |
| 속도 정규화 항의 분리 해석 | `VCOM/BOS`를 별도 지표로 해석 | DV3는 step_onset에서만 유의(`***`), onset은 n.s. | Partially consistent |
| 속도 조건과의 상호작용 | 논문별로 task/intensity 영향 보고 | `step_TF*velocity_c`는 6개 모두 n.s. | Inconsistent |
| 이벤트 정의 일치성 | lift-off/touchdown, 300ms 등 다양 | 사용자 요구로 onset/step_onset 2개로 제한 | Not tested |

`Inconsistent`/`Not tested` 항목은 이벤트 정의와 실험 설계(원문 vs 현재 데이터)의 차이에서 발생했을 가능성이 높다.

## Interpretation & Conclusion

1. `methods_list` 기반 xCOM/BOS 계열 통합식은 본 데이터에서 step/nonstep 전략 차이를 통계적으로 구분했다. 특히 DV1, DV2는 두 이벤트 모두 유의했다.
2. 속도 정규화 지표(DV3)는 step onset에서만 유의하여, 전략 분리는 이벤트 시점 의존성이 크다.
3. 상호작용(`step×velocity`)은 유의하지 않아, 현재 표본에서는 속도 변화가 step/nonstep 차이를 추가적으로 증폭/완화한다는 근거가 부족했다.

## Limitations

1. 원문의 lift-off/touchdown/300ms 이벤트를 모두 재현하지 못하고 `platform onset`, `step onset`만 사용했다.
2. `methods_list`의 다중 논문식을 수학적으로 중복 통합했기 때문에 논문별 독립 지표를 1:1로 모두 유지하지는 않았다.
3. `step_onset` 결측 trial은 subject-velocity 평균으로 보완하여 이벤트를 구성했으므로, 개별 trial의 실제 step timing 변동은 일부 축약될 수 있다.
4. 본 분석은 xCOM/BOS 계열만 대상으로 하며 joint/force 계열과의 통합 모델은 포함하지 않았다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/xCOM&BOS_normalization/analyze_xcom_bos_normalization_lmm.py --dry-run
conda run --no-capture-output -n module python analysis/xCOM&BOS_normalization/analyze_xcom_bos_normalization_lmm.py
```

- Input: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- Output: figure 3개 + 콘솔 LMM 결과

## Figures

| File | Description |
|---|---|
| `fig1_main_effect_forest.png` | `step_TFstep` 주효과 계수 forest plot |
| `fig2_interaction_forest.png` | `step_TFstep:velocity_c` 상호작용 계수 forest plot |
| `fig3_violin_significant.png` | 유의(또는 top-p) DV의 step/nonstep 분포 violin/strip |
