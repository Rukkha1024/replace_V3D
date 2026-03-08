---
---
# 가설

1. initial phase에서 nonstep과 step의 관절 각도는 차이가 있을 것이다.

# results

## platform_onset 단일시점 LMM

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Hip_stance_X_abs_onset` | 11.52±4.45 | 11.91±3.54 | -0.32 | n.s. |
| `Hip_stance_Y_abs_onset` | 1.54±4.18 | -0.09±3.50 | 1.55 | * |
| `Hip_stance_Z_abs_onset` | -0.08±9.49 | 1.61±8.00 | -1.30 | n.s. |
| `Knee_stance_X_abs_onset` | -4.51±5.71 | -3.17±5.01 | -0.12 | n.s. |
| `Knee_stance_Y_abs_onset` | -2.65±3.34 | -2.97±2.38 | 0.26 | n.s. |
| `Knee_stance_Z_abs_onset` | -4.38±2.93 | -3.93±3.03 | -0.40 | n.s. |
| `Ankle_stance_X_abs_onset` | 4.29±3.03 | 4.04±3.08 | 0.01 | n.s. |
| `Ankle_stance_Y_abs_onset` | 10.45±4.02 | 10.80±2.85 | -0.43 | n.s. |
| `Ankle_stance_Z_abs_onset` | -8.49±5.65 | -8.00±5.81 | 0.14 | n.s. |
| `Trunk_X_abs_onset` | 3.17±5.91 | 3.25±5.34 | -0.19 | n.s. |
| `Trunk_Y_abs_onset` | -1.01±3.52 | 0.74±2.96 | -0.69 | * |
| `Trunk_Z_abs_onset` | 0.97±3.37 | 0.36±2.60 | 0.34 | n.s. |
| `Neck_X_abs_onset` | 21.22±8.16 | 23.36±6.98 | -0.10 | n.s. |
| `Neck_Y_abs_onset` | 2.76±5.48 | 1.83±5.05 | 0.18 | n.s. |
| `Neck_Z_abs_onset` | -0.79±5.10 | 0.31±4.99 | -0.08 | n.s. |

## step_onset 단일시점 LMM

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Hip_stance_X_step_onset` | 6.70±2.89 | 4.49±2.20 | 2.55 | *** |
| `Hip_stance_Y_step_onset` | -0.67±1.62 | 0.04±1.03 | -0.49 | * |
| `Hip_stance_Z_step_onset` | -0.06±2.55 | 0.78±1.60 | -0.93 | * |
| `Knee_stance_X_step_onset` | -7.16±6.12 | -8.20±4.34 | -0.16 | n.s. |
| `Knee_stance_Y_step_onset` | -0.54±1.14 | -1.02±1.09 | 0.17 | n.s. |
| `Knee_stance_Z_step_onset` | -1.25±1.57 | -0.75±1.15 | -0.58 | *** |
| `Ankle_stance_X_step_onset` | 6.72±5.50 | 6.81±4.59 | -0.46 | n.s. |
| `Ankle_stance_Y_step_onset` | 0.99±1.27 | 1.23±1.46 | -0.25 | n.s. |
| `Ankle_stance_Z_step_onset` | -0.49±2.27 | 0.99±1.85 | -1.03 | *** |
| `Trunk_X_step_onset` | 2.15±2.82 | 0.04±1.97 | 2.00 | *** |
| `Trunk_Y_step_onset` | 0.80±1.71 | 0.46±0.95 | 0.51 | * |
| `Trunk_Z_step_onset` | -0.46±2.16 | -0.25±1.52 | -0.06 | n.s. |
| `Neck_X_step_onset` | -5.81±5.20 | -6.87±4.67 | 0.10 | n.s. |
| `Neck_Y_step_onset` | -0.20±2.74 | -0.19±2.10 | -0.11 | n.s. |
| `Neck_Z_step_onset` | 1.14±1.82 | 0.84±1.76 | 0.31 | n.s. |

## coordinate 해석 기준

- 관절각 계산은 Visual3D-like intrinsic `XYZ` 순서를 사용한다.
- Segment 좌표계 기준은 `X=+Right`, `Y=+Anterior`, `Z=+Up/+Proximal`이다.
- Hip/Knee/Ankle의 `Y/Z`는 좌우(L/R) 해석 일관성을 위해 **LEFT side 값을 부호 반전**하여 RIGHT 의미와 통일한다.
- 따라서 `X/Y/Z`는 각 축 회전 성분이며, 임상적 평면(sagittal/frontal/transverse)과 완전한 1:1 대응으로 단정하지 않는다.

## stance 기준

- step trial은 `step_r -> 좌측 stance`, `step_l -> 우측 stance`로 계산한다.
- nonstep trial은 subject별 step trial의 `major_step_side`를 stance 기준으로 사용한다.
- `step_r_count == step_l_count`인 tie subject는 좌/우 평균으로 계산한다.
- 이번 실행 요약: `step_r_major=9`, `step_l_major=10`, `tie=5` (tie subjects: `강비은, 김서하, 김유민, 안지연, 유재원`)

- step_onset 비교 규칙:
  - step trial: 해당 trial의 `step_onset_local` 사용
  - nonstep trial: 동일 subject의 step trial `step_onset_local` 평균값을 대입한 후 frame으로 반올림
  - step_onset 기준 유효 trial: `119/126` (step=`52`, nonstep=`67`)
  - 제외 trial: `7` (step_onset 결측 step=`1`, step 참조 부재 nonstep=`6`, frame 불일치=`0`)
  - nonstep step_onset 참조 부재 subject: `권유영, 김종철, 방주원`

- 해석 노트:
  - platform_onset: 15개 segment angle 변수(X/Y/Z) 중 2개가 FDR 유의였다: `Hip_stance_Y_abs_onset, Trunk_Y_abs_onset`.
  - step_onset: 15개 step_onset segment angle 변수(X/Y/Z) 중 7개가 FDR 유의였다: `Hip_stance_X_step_onset, Trunk_X_step_onset, Ankle_stance_Z_step_onset, Knee_stance_Z_step_onset, Hip_stance_Z_step_onset, Hip_stance_Y_step_onset, Trunk_Y_step_onset`.
  - 두 시점 모두에서 `Estimate`와 `Sig`는 변수별 `step/nonstep` 그룹 내부 `1.5×IQR` 이상치 제외 후 계산했다.
  - `95% CI`는 single-frame 보고서에 포함하지 않으며, baseline range mean 보고서에서만 제시한다.
  - 두 시점 모두에서 전축이 일관되게 유의하지 않다면, 관절각만으로 전략 차이를 설명하는 근거는 제한적이다.

# 결과 해석

## platform_onset 해석

- platform onset joint-angle 15개 변수 중 `2`개가 FDR 유의였다: `Hip_stance_Y_abs_onset, Trunk_Y_abs_onset`.
- 이 시점은 baseline 평균이 아니라 섭동 직후의 posture snapshot에 가깝다.
- 따라서 platform onset에서는 지지다리 및 체간 정렬 차이가 일부 축에서만 관찰되며, 전략 분화의 출발점이라기보다 제한적인 초기 반응 차이로 해석하는 편이 안전하다.

## step_onset 해석

- step onset joint-angle 15개 변수 중 `7`개가 FDR 유의였다: `Hip_stance_X_step_onset, Trunk_X_step_onset, Ankle_stance_Z_step_onset, Knee_stance_Z_step_onset, Hip_stance_Z_step_onset, Hip_stance_Y_step_onset, Trunk_Y_step_onset`.
- step onset은 `step` trial의 실제 `step_onset_local`과, `nonstep` trial의 subject 평균 step onset 참조 frame을 사용한다.
- 따라서 이 결과는 평균적인 초기 자세라기보다 실제 발 들기 직전의 준비 자세 또는 전략 실행 직전 posture 차이에 더 가깝다.

## 종합 해석

- 같은 단일 프레임 이상치 제외 규칙에서 보면, platform onset보다 step onset에서 유의한 joint-angle 차이가 더 많이 관찰된다.
- 즉, 전략 차이는 섭동 직후 정적 snapshot보다 실제 발 들기 직전 single-frame에서 더 뚜렷하게 나타나는 경향이 있다.
- 다만 두 시점 모두 전축이 일관되게 유의하지는 않으므로, 관절각만으로 step/nonstep 전략 차이를 완전히 설명한다고 단정하기는 어렵다.
- onset 전 구간 평균과 `95% CI` 해석은 baseline 보고서와 분리해서 읽어야 한다.


# 결론

- 가설 1 결과: **FAIL**
- single-frame 비교에서는 step onset이 platform onset보다 더 강한 분화를 보였지만, 관절각만으로 전략 차이를 단정하기에는 근거가 제한적이다.

# keypapers

1. Van Wouwe et al. (2021): 초기 자세와 전략 variability의 상호작용을 제시했으며, 본 문서는 그 질문을 두 개의 single-frame 시점으로 나누어 비교한다.

---
Auto-generated by analyze_initial_posture_strategy_lmm.py.
