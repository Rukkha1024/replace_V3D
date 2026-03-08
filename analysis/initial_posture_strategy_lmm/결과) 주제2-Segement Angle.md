---
---
# 가설

1. initial phase에서 nonstep과 step의 segment angle은 차이가 있을 것이다.

# 결과 요약

- baseline mean(`[-0.30, 0.00] s`)에서는 joint-angle 15개 중 `Knee_stance_X_baseline` 1개만 FDR 유의했다. 즉, onset 전 평균 자세 차이만으로 전략 분화를 설명하는 근거는 제한적이었다.
- `platform_onset` 단일 프레임에서는 joint-angle 15개 중 `2`개가 유의했고, `step_onset` 단일 프레임에서는 `7`개가 유의했다.
- 따라서 segment angle 차이는 섭동 직후 snapshot보다 실제 발 들기 직전 자세에서 더 뚜렷하게 나타났다.

# results

## baseline mean LMM

- baseline 결과는 짧게 보면 충분하다. onset 전 `[-0.30, 0.00] s` 평균에서는 `Knee_stance_X_baseline`만 유의했고, 나머지 joint-angle 변수는 모두 `n.s.`였다.
- 따라서 baseline 평균 자세는 전략 차이의 출발점을 일부 보여줄 수는 있지만, segment angle 관점에서 step/nonstep을 강하게 분리하지는 못했다.

## platform_onset 단일시점 LMM

- `platform_onset_local`은 섭동 직후 posture snapshot으로 해석했다.
- joint-angle 15개 변수 중 유의 변수는 `2`개였다.

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Hip_stance_Y_abs_onset` | 1.54±4.18 | -0.09±3.50 | 1.55 | * |
| `Trunk_Y_abs_onset` | -1.01±3.52 | 0.74±2.96 | -0.69 | * |

- 나머지 `13`개 joint-angle 변수는 모두 `n.s.`였다.
- 즉, platform onset에서는 지지다리 hip과 trunk의 일부 축에서만 차이가 관찰됐고, 전반적인 segment angle 분화는 아직 제한적이었다.

## step_onset 단일시점 LMM

- `step` trial은 실제 `step_onset_local`을 사용했고, `nonstep` trial은 동일 subject의 평균 `step_onset_local`을 참조 frame으로 사용했다.
- joint-angle 15개 변수 중 유의 변수는 `7`개였다.

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Hip_stance_X_step_onset` | 6.70±2.89 | 4.49±2.20 | 2.55 | *** |
| `Hip_stance_Y_step_onset` | -0.67±1.62 | 0.04±1.03 | -0.49 | * |
| `Hip_stance_Z_step_onset` | -0.06±2.55 | 0.78±1.60 | -0.93 | * |
| `Knee_stance_Z_step_onset` | -1.25±1.57 | -0.75±1.15 | -0.58 | *** |
| `Ankle_stance_Z_step_onset` | -0.49±2.27 | 0.99±1.85 | -1.03 | *** |
| `Trunk_X_step_onset` | 2.15±2.82 | 0.04±1.97 | 2.00 | *** |
| `Trunk_Y_step_onset` | 0.80±1.71 | 0.46±0.95 | 0.51 | * |

- 가장 강한 분화는 `Hip_stance_X_step_onset`와 `Trunk_X_step_onset`에서 나타났고, distal segment에서는 `Knee_stance_Z_step_onset`, `Ankle_stance_Z_step_onset`도 뚜렷했다.
- 나머지 `8`개 joint-angle 변수는 모두 `n.s.`였다.
- 즉, 실제 발 들기 직전에는 stance hip, trunk, 그리고 일부 distal Z축 각도에서 step/nonstep 전략 차이가 더 명확하게 드러났다.

## 해석 기준

- 관절각 계산은 Visual3D-like intrinsic `XYZ` 순서를 사용했다.
- Segment 좌표계는 `X=+Right`, `Y=+Anterior`, `Z=+Up/+Proximal` 기준이다.
- 따라서 `X/Y/Z`는 각 축 회전 성분이며, sagittal/frontal/transverse와 완전한 1:1 대응으로 단정하지 않는다.
- step trial은 `step_r -> 좌측 stance`, `step_l -> 우측 stance`로 계산했고, nonstep trial은 subject별 `major_step_side`를 stance 기준으로 사용했다.

# 결과 해석

## baseline 해석

- baseline 평균에서는 joint-angle 차이가 거의 나타나지 않았다.
- 따라서 주제 2에서 baseline posture는 보조적 배경 정보로는 의미가 있지만, segment angle만으로 전략 차이를 미리 구분하는 지표라고 보기는 어렵다.

## platform_onset 해석

- platform onset에서는 유의한 축이 `Hip_stance_Y_abs_onset`, `Trunk_Y_abs_onset` 두 개에 그쳤다.
- 이는 섭동 직후 posture snapshot에서 일부 지지다리 및 trunk 정렬 차이는 보이지만, 아직 전략 전체가 관절각 수준에서 충분히 분리되지는 않았다는 뜻에 가깝다.

## step_onset 해석

- step onset에서는 유의 변수가 `7/15`로 늘어났고, hip과 trunk의 proximal 조절뿐 아니라 knee와 ankle의 distal 축 차이도 함께 나타났다.
- 특히 `Hip_stance_X_step_onset`와 `Trunk_X_step_onset`는 step onset 시점의 joint-angle 분화를 대표하는 변수로 볼 수 있다.
- 동시에 `Knee_stance_Z_step_onset`, `Ankle_stance_Z_step_onset`까지 유의했다는 점은, step onset 직전에는 proximal segment뿐 아니라 일부 distal segment에서도 차이가 함께 나타났다는 뜻으로 읽는 편이 안전하다.

## 종합 해석

- 같은 단일 프레임 LMM 기준에서 segment angle 차이는 `platform_onset`보다 `step_onset`에서 훨씬 더 뚜렷했다.
- 따라서 주제 2의 step/nonstep 분화는 섭동 직후의 정적 자세 차이라기보다, 실제 발 들기 직전에 형성되는 준비 자세와 실행 직전 조절 차이에서 더 강하게 드러난다고 해석하는 편이 자연스럽다.
- 다만 두 시점 모두 모든 축이 일관되게 유의한 것은 아니므로, segment angle만으로 전략 차이를 완전히 설명한다고 단정할 수는 없다.

# 결론

- source report의 strict overall verdict: **FAIL** (`29`개 전체 onset 변수 기준)
- baseline에서는 차이가 제한적이었고, 관절각 기반 전략 분화는 `step_onset` 직전 single-frame에서 가장 선명했다.
- 즉, 주제 2의 segment angle 결과는 "초기 평균 자세"보다 "실제 stepping 준비 자세"가 전략 차이를 더 잘 드러낸다는 쪽에 가깝다.

# keypapers

1. Van Wouwe et al. (2021): 초기 자세와 task-level goal의 상호작용이 전략 variability를 설명할 수 있음을 제시했고, 본 결과는 그 관점을 baseline, platform onset, step onset으로 나누어 비교한 요약이다.
