---
---
# 가설

1. initial phase에서 nonstep과 step의 관절 각도는 차이가 있을 것이다.

# results

## platform_onset 단일시점 LMM

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Hip_stance_X_abs_onset` | 10.75±5.87 | 9.94±6.99 | 0.12 | n.s. |
| `Hip_stance_Y_abs_onset` | 1.54±4.18 | -0.09±3.50 | 1.55 | * |
| `Hip_stance_Z_abs_onset` | -0.08±9.49 | 0.74±8.90 | -1.17 | n.s. |
| `Knee_stance_X_abs_onset` | -4.51±5.71 | -3.17±5.01 | -0.12 | n.s. |
| `Knee_stance_Y_abs_onset` | -2.85±3.64 | -2.96±3.71 | 0.36 | n.s. |
| `Knee_stance_Z_abs_onset` | -4.38±2.93 | -3.93±3.03 | -0.40 | n.s. |
| `Ankle_stance_X_abs_onset` | 4.29±3.03 | 4.04±3.08 | 0.01 | n.s. |
| `Ankle_stance_Y_abs_onset` | 10.72±4.42 | 11.21±5.25 | -0.25 | n.s. |
| `Ankle_stance_Z_abs_onset` | -8.49±5.65 | -8.00±5.81 | 0.14 | n.s. |
| `Trunk_X_abs_onset` | 2.79±6.49 | 2.49±6.40 | -0.18 | n.s. |
| `Trunk_Y_abs_onset` | -1.00±4.07 | 0.33±3.51 | -0.36 | n.s. |
| `Trunk_Z_abs_onset` | 1.15±3.58 | 0.79±3.59 | 0.24 | n.s. |
| `Neck_X_abs_onset` | 17.59±27.66 | 23.02±7.51 | -5.25 | n.s. |
| `Neck_Y_abs_onset` | 2.17±6.95 | 1.83±5.05 | -0.35 | n.s. |
| `Neck_Z_abs_onset` | 0.92±13.45 | 0.31±4.99 | 1.30 | n.s. |

## step_onset 단일시점 LMM

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |
|---|---:|---:|---:|---|
| `Hip_stance_X_step_onset` | 7.73±3.94 | 4.02±3.81 | 3.76 | *** |
| `Hip_stance_Y_step_onset` | -0.17±2.01 | -0.01±1.12 | 0.11 | n.s. |
| `Hip_stance_Z_step_onset` | -1.53±4.00 | 0.50±2.47 | -2.15 | *** |
| `Knee_stance_X_step_onset` | -9.86±7.45 | -8.72±4.87 | -3.22 | *** |
| `Knee_stance_Y_step_onset` | -0.58±1.65 | -1.02±1.09 | 0.07 | n.s. |
| `Knee_stance_Z_step_onset` | -1.24±1.98 | -0.75±1.15 | -0.52 | n.s. |
| `Ankle_stance_X_step_onset` | 8.15±5.21 | 6.81±4.59 | 1.19 | n.s. |
| `Ankle_stance_Y_step_onset` | 1.77±3.13 | 1.15±1.59 | 0.60 | n.s. |
| `Ankle_stance_Z_step_onset` | -0.07±4.61 | 0.88±2.04 | -0.52 | n.s. |
| `Trunk_X_step_onset` | 3.06±5.63 | 1.11±4.19 | 1.41 | n.s. |
| `Trunk_Y_step_onset` | 0.69±2.15 | 0.21±1.62 | 0.66 | n.s. |
| `Trunk_Z_step_onset` | -0.46±2.16 | -0.36±1.90 | -0.00 | n.s. |
| `Neck_X_step_onset` | -8.32±11.82 | -9.28±9.08 | -1.94 | n.s. |
| `Neck_Y_step_onset` | -0.21±3.08 | -0.19±2.10 | -0.10 | n.s. |
| `Neck_Z_step_onset` | 0.80±2.25 | 0.85±2.20 | -0.04 | n.s. |

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
  - step_onset 기준 유효 trial: `119/125` (step=`52`, nonstep=`67`)
  - 제외 trial: `6` (step_onset 결측 step=`1`, step 참조 부재 nonstep=`5`, frame 불일치=`0`)
  - nonstep step_onset 참조 부재 subject: `권유영, 김종철, 방주원`

- 해석 노트:
  - platform_onset: 15개 segment angle 변수(X/Y/Z) 중 1개가 FDR 유의였다: `Hip_stance_Y_abs_onset`.
  - step_onset: 15개 step_onset segment angle 변수(X/Y/Z) 중 3개가 FDR 유의였다: `Hip_stance_X_step_onset, Knee_stance_X_step_onset, Hip_stance_Z_step_onset`.
  - 두 시점 모두에서 전축이 일관되게 유의하지 않다면, 관절각만으로 전략 차이를 설명하는 근거는 제한적이다.

# 결과 해석

> **좌표 → 해부학 매핑 전제**
> 본 해석은 intrinsic XYZ 순서(X=+Right, Y=+Anterior, Z=+Up)를 기준으로 한다.
> 또한 Hip/Knee/Ankle의 **Y/Z는 좌우(L/R) 해석 일관성을 위해 LEFT side를 부호 반전하여 통일한 값**을 사용한다.
> - **X축 회전** → sagittal plane _(앞뒤 방향으로 자르는 면)_:
>   - Hip: flexion(+) / extension(−)
>   - Ankle: dorsiflexion(+) / plantar flexion(−)
>   - Knee: flexion(−) / extension(+) *(본 구현에서 Knee X 부호는 독립 sagittal proxy와의 비교로 확인됨)*
> - **Y축 회전** → frontal plane _(좌우 방향으로 자르는 면)_:
>   - Hip: abduction(+) / adduction(−)
>   - Knee: valgus(+) / varus(−)
>   - Ankle: eversion(+) / inversion(−)
> - **Z축 회전** → transverse plane _(수평으로 자르는 면)_: internal rotation(+) / external rotation(−)
>
> 단, intrinsic 순서 특성상 각도 크기가 클수록 축 간 crosstalk 영향이 커질 수 있으며, 임상 평면과 완전한 1:1 대응으로 단정할 수 없다.

 **용어 해설**

| 용어 | 설명 |
|---|---|
| Flexion (굴곡) | 관절을 구부리는 방향 (예: 무릎을 접는 것) |
| Extension (신전) | 관절을 펴는 방향 (예: 무릎을 쭉 펴는 것) |
| Knee valgus (무릎 외반) | 무릎이 안쪽으로 무너지는 방향 (흔히 "X자 다리" 방향) |
| Knee varus (무릎 내반) | 무릎이 바깥쪽으로 벌어지는 방향 (흔히 "O자 다리" 방향) |
| Inversion (내번) | 발 안쪽이 들리고 발바닥이 안쪽을 향하는 방향 (발목 삐는 방향) |
| Eversion (외번) | 발 바깥쪽이 들리고 발바닥이 바깥쪽을 향하는 방향 |
| Dorsiflexion (배측굴곡) | 발끝을 정강이 쪽으로 들어올리는 방향 |
| Plantar flexion (저측굴곡) | 발끝을 아래로 뻗는 방향 (까치발 서는 방향) |
| Internal rotation (내회전) | 뼈 또는 분절이 몸의 중심선 쪽으로 회전하는 것 |
| External rotation (외회전) | 뼈 또는 분절이 몸의 중심선 바깥쪽으로 회전하는 것 |
| Stance leg (지지 다리) | 발이 바닥을 딛고 있어 몸무게를 지지하는 쪽 다리 |

> **⚠ 어느 쪽 다리인가?**
> 이 결과에서 분석된 모든 관절(Hip, Knee, Ankle)은 **stance leg**, 즉 **발을 내딛지 않는 쪽(지지하는 쪽) 다리**의 값이다.
> - Step 그룹: 오른발로 step한 trial → **왼쪽** 다리가 stance / 왼발로 step한 trial → **오른쪽** 다리가 stance
> - 피험자마다 주 step 방향이 달라(step_r 우세 9명, step_l 우세 10명, tie 5명), stance 측은 고정된 좌/우가 아니다.
> - 각 피험자의 우세 step 방향을 기준으로 stance 측을 통일한 뒤 평균을 낸 값이므로, "step 반대편 다리"의 평균 자세로 이해하면 된다.

---

## platform_onset 해석 (섭동 시작 시점)

> 이 시점에서 분석되는 엉덩이·무릎·발목은 모두 **아직 발을 내딛지 않은 쪽(지지 다리)**의 관절이다. 즉, 곧 step을 하게 될 반대편 발이 들리기 전, 바닥을 딛고 있던 다리의 자세를 보는 것이다.

유의한 변수는 15개 중 1개로, **Hip의 frontal plane(Y)** 성분만 FDR 유의였다. Knee/Ankle의 Y/Z 및 모든 X 성분은 `n.s.`였다.

### Hip: frontal plane(Y) 차이

- **Hip_stance_Y** (Estimate = +1.55, *): Step(1.54°) > Nonstep(−0.09°)
  - Step 그룹은 섭동 시작 시점에서 지지 측 hip이 Nonstep보다 **더 abducted(바깥쪽으로 벌어진)** 정렬을 보였다.
  - 즉, platform_onset에서 관절각의 집단 차이는 knee/ankle보다 **hip의 frontal 정렬 차이**로 요약된다.

---

## step_onset 해석 (발 들기 시작 시점)

> 이 시점에서 분석되는 엉덩이·무릎은 모두 **반대편 발이 막 들리는 순간, 여전히 바닥을 짚고 있는 쪽(지지 다리)**의 관절이다. 즉, step 동작이 실제로 시작될 때 몸을 떠받치고 있는 다리의 자세를 보는 것이다.

유의한 3개 변수는 **sagittal plane(X)의 Hip/Knee와 transverse plane(Z)의 Hip**이었다. 발목 및 trunk는 유의하지 않았다.

### Hip: sagittal & transverse plane 차이

- **Hip_stance_X** (Estimate = +3.76, ***): Step(7.73°) > Nonstep(4.02°)
  - 발을 들기 시작하는 시점에서 지지 측 엉덩이는 Nonstep보다 유의하게 더 큰 **hip flexion** _(엉덩이 관절이 더 많이 구부러진, 즉 상체가 앞으로 더 기울어지거나 stance 측이 더 많이 체중을 받는)_ 상태였다.
- **Hip_stance_Z** (Estimate = −2.15, ***): Step(−1.53°) < Nonstep(0.50°)
  - Step 그룹은 지지 측 엉덩이가 **external rotation** _(바깥쪽으로 약간 비틀린)_, Nonstep은 상대적으로 **internal rotation** 방향이었다.
  - 즉, step 동작이 실제로 시작되는 시점에서는 hip의 transverse 성분(Z)에서 그룹 차이가 관찰되었다.

### Knee: sagittal plane 차이

- **Knee_stance_X** (Estimate = −3.22, ***): Step(−9.86°) < Nonstep(−8.72°)
  - 본 구현에서 Knee X는 **음수 방향이 flexion(굴곡)** 이므로, Step 그룹에서 무릎이 **더 굴곡된(더 bent)** 상태였다.
  - 반대편 발을 들어올리는 순간, 지지 다리가 약간 더 굴곡된 자세를 취하는 패턴으로 해석할 수 있다.

---

## 종합 해석

| 시점 | 유의 관절 | 평면 | 주요 차이 |
|---|---|---|---|
| platform_onset | Hip | Frontal | Step: 지지 측 hip abduction 증가 |
| step_onset | Hip, Knee | Sagittal + Transverse | Step: hip flexion 증가 + hip external rotation + knee flexion 증가 |

- **평면별 패턴**: platform_onset에서는 **frontal(Y)** 성분에서만 제한적으로 차이가 관찰되었고, step_onset에서는 **sagittal(X) 및 transverse(Z)** 성분 차이가 두드러졌다. 몸통과 목은 두 시점 모두 집단 간 차이가 없었다.
- **전략 구분 근거로서의 한계**: 유의한 변수가 15개 중 1개(platform_onset), 3개(step_onset)에 불과했다. 즉, 관절각만으로 step/nonstep 전략을 단정하기에는 근거가 제한적이다.
- **해석 주의점**: 유의한 frontal/transverse 차이가 전략 선택의 **원인**인지, 우연히 다른 초기 자세의 **결과**인지는 단면적 비교만으로 인과 관계를 확정할 수 없다.

---

# 결론

- 가설 1 결과: **FAIL**
- platform_onset 및 step_onset 관절각 비교를 함께 보아도, 전략 차이를 관절각만으로 단정하기에는 근거가 제한적이다.

# keypapers

1. Van Wouwe et al. (2021): 초기 자세와 전략 variability의 상호작용을 제시했지만, 본 onset 단일시점 집단 비교에서는 segment angle에서 유의 차이가 재현되지 않았다.

---
Auto-generated by analyze_initial_posture_strategy_lmm.py.
