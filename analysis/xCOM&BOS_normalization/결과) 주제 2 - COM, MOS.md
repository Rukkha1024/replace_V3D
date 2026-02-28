---
cssclass: clean-embeds
date created: 2026-01-19. 00.31
---

# results 


## plot으로 확인했을 때 

<!-- 단위 기준: 원 데이터(COM/xCOM/BOS/MOS)는 m, 본 문서 길이 해석은 cm(=m*100)로 표기 -->

### COM 
![](https://i.imgur.com/7d8Qh1M.png)

- COM X
	- step_onset 이후로 극명하게 나뉨. 
	- vCOM, xCOM은 그래도 onset - step_onset 사이에 살짝 차이가 있음. nonstep 값이 좀 더 높네. 
- COM Y: 애초에 stepping의 영역이다 보니깐 별로 의미가 없는 것 같네. 
- COM Z: 둘이 차이 못 느낌 

### MOS 

![](https://i.imgur.com/4clnWYI.png)

- MOS 부호 설명
	- 양수(+): xCOM이 BOS 안에 있음. 
	- 음수(-): xCOM이 BOS 밖으로 나감. 
- MOS AP: nonstep이 더 안정적이다? <!-- 근데 이게 끝? 해석을 못하겠음  -->
- MOS ML: 얘도 모르겠음. <!-- 선행연구나 읽자 -->


## 1차 스크리닝 (COM + xCOM + xCOM/BOS, LMM)

### 1차 스크리닝 목적/구간

- 목적: 2차 확인 분석 전에, `COM + xCOM + xCOM/BOS`에서 step/nonstep 차이가 나타나는 후보 변수를 먼저 탐색.
- 분석구간:
	- step: `[platform_onset_local, step_onset_local]`
	- nonstep: `[platform_onset_local, same (subject, velocity) stepping mean step_onset_local]`
	- fallback: prefilter 이벤트 기반 subject-velocity mean 보완.
- 모델: `DV ~ step_TF + (1|subject)`, REML.
- 다중비교: 1차 스크리닝 전체 DV(24개) 대상으로 BH-FDR 일괄 보정.
- 주의: `xCOM_BOS_ML_foot`는 이번 1차 스크리닝에서 **foot_length 공통 분모**를 적용한 사용자 지정 정의.
- 전체 결과 파일: `analysis/xCOM&BOS_normalization/com_xcom_screening_lmm_results.csv`

### COM+xCOM+xCOM/BOS 변수표

| Group | 변수 구성 |
|---|---|
| COM (X/Y/Z) | `max-min`, `mean_velocity`, `peak_velocity` |
| xCOM (X/Y/Z) | `max-min`, `mean_velocity`, `peak_velocity` |
| xCOM/BOS (AP/ML, foot 정규화) | `platform_onset`, `step_onset`, `window_mean` |

### LMM 결과 (1차 유의 변수 shortlist)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig. |
|---|---|---|---|---|
| `xCOM_Y_peak_velocity` | 0.2705±0.0940 | 0.1483±0.0703 | 0.1269 | *** |
| `xCOM_BOS_AP_foot_mean_window` | 0.3292±0.1256 | 0.4036±0.0996 | -0.0666 | *** |
| `xCOM_BOS_AP_foot_platformonset` | 0.5091±0.0596 | 0.5715±0.0654 | -0.0440 | *** |
| `xCOM_BOS_AP_foot_steponset` | 0.2354±0.1720 | 0.3232±0.0996 | -0.0893 | *** |
| `xCOM_Y_mean_velocity` | 0.0703±0.0320 | 0.0450±0.0233 | 0.0277 | *** |
| `COM_Y_peak_velocity` | 0.0561±0.0294 | 0.0340±0.0239 | 0.0230 | *** |
| `xCOM_Y_range` | 0.0235±0.0116 | 0.0161±0.0124 | 0.0078 | ** |
| `COM_Z_mean_velocity` | 0.0250±0.0118 | 0.0220±0.0082 | 0.0036 | * |
| `xCOM_X_mean_velocity` | 0.1286±0.0778 | 0.1347±0.0694 | 0.0133 | * |

### 1차 해석 (유의 변수 기준)

- 전체 24개 DV 중 9개가 FDR 유의.
- 유의 신호는 `xCOM` 계열, 특히 `Y축`(`xCOM_Y_peak_velocity`, `xCOM_Y_mean_velocity`, `xCOM_Y_range`)에 집중.
- `xCOM/BOS_AP_foot`는 이벤트 2개 + 구간평균 모두 유의(`step < nonstep`)라서 AP 안정성 차이 후보로 우선순위가 높음.
- `COM` 단독 변수는 `COM_Y_peak_velocity`, `COM_Z_mean_velocity`만 유의했고 나머지는 비유의.
- 따라서 2차 분석은 `xCOM_Y` 계열 + `xCOM_BOS_AP_foot` 계열을 우선 검증 대상으로 두는 것이 합리적.


## DV1 통계결과 (LMM)

### DV1 산출 방법

DV1은 **"특정 이벤트 시점에서 xCOM이 발 뒤꿈치(BOS 뒤 경계)보다 얼마나 앞에 있는가"**를 나타낸 값.

**재료 ①: `xCOM_hof` (Hof, 2005)**

- extrapolated COM(xCOM)의 AP 위치.
- 공식: `xCOM_hof = COM_X + vCOM_X / ω₀`
	- `COM_X`: 무게중심(COM)의 AP 위치 (m)
	- `vCOM_X`: COM의 AP 속도 (m/s)
	- `ω₀ = sqrt(g / 신장)`, g = 9.81 m/s²  →  신장이 클수록 ω₀가 작고, 같은 속도라도 xCOM이 더 앞으로 나감.

**재료 ②: `BOS_rear`**

- 지지면(BOS)의 AP 최솟값, 즉 발 뒤꿈치 위치 (m).
- 값이 클수록 발이 앞쪽에 있다는 뜻.

**계산 — 두 버전 병행**

```
Primary (통계 검정용):
  DV1_norm = (xCOM_hof - BOS_rear) / foot_len   [무차원]

Supplementary (크기 해석용):
  DV1_abs_cm = (xCOM_hof - BOS_rear) × 100      [cm]
```

- 분자(`xCOM − BOS_rear`)는 동일. 나누기(foot_len) vs 곱하기(×100)만 다름.
- **Primary (norm)**: 피험자 간 발 크기 차이를 보정 → 통계적으로 적절. 선행연구(Salot, Patel, Bhatt)와 동일한 정규화.
- **Supplementary (abs_cm)**: "step군이 nonstep보다 ~1.1 cm 더 뒤" 같은 직관적 크기 해석에 사용.
- 양수 → xCOM이 발 뒤꿈치보다 앞에 있음. 값이 작을수록 불안정 방향.

**값 추출 시점**

trial 전체 시계열 중에서 이벤트 프레임(1개) 시점의 값만 뽑아 trial당 1개 값으로 만든 뒤 LMM에 투입.

- `platform_onset` 버전: 플랫폼이 움직이기 시작한 프레임
- `step_onset` 버전: stepping이 시작된 프레임

---

### LMM 결과

- 모델: Primary `DV1_norm ~ step_TF + (1|subject)` / Supplementary `DV1_abs_cm ~ step_TF + (1|subject)`
- 참조: nonstep (`step_TFstep` Estimate = step − nonstep 차이)

| Event | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig. | Step (M±SD, cm) | Nonstep (M±SD, cm) | Estimate (cm) | Sig. |
|---|---|---|---|---|---|---|---|---|
| platform onset | 0.51±0.06 | 0.57±0.07 | −0.04 | *** | 12.91±1.62 | 14.44±1.62 | −1.10 | *** |
| step onset | 0.24±0.17 | 0.32±0.10 | −0.09 | *** | 5.95±4.45 | 8.19±2.61 | −2.29 | *** |

- 방향 해석
	- DV1이 작을수록 BOS rear 대비 xCOM이 더 후방.
	- 따라서 두 이벤트 모두 step군이 nonstep군보다 상대적으로 후방 위치.
	- 절대 크기로 보면 platform onset 시 약 1.1 cm, step onset 시 약 2.3 cm 차이. 

#### Additional Check (Raw, No Filtering)

- 별도 임시 실험에서 stance leg 상관 없이 step 표본을 그냥 mixed == 1 모두 해당하는 것으로 넓혀서 분석 진행. 
- 대상: `xCOM_BOS_norm_onset`, `xCOM_BOS_norm_300ms` (표본: trials=184, subjects=24, step=112, nonstep=72)

- DV1은 위에서 정의한 것처럼 분자(`xCOM_hof - BOS_rear`)는 동일하고,
	- 정규화 버전: `DV1_norm = (xCOM_hof - BOS_rear) / foot_len`
	- 절대 크기 버전: `DV1_abs_cm = (xCOM_hof - BOS_rear) × 100`
- raw 표본에서도 두 버전 모두 같은 방향(음수)으로 유지되는지 LMM으로 재확인.
- 이벤트: `platform onset`, `step onset`
- 표본: trials=184, subjects=24, step=112, nonstep=72  
	- 단, `DV1_*_step_onset`은 step onset 프레임이 없는 step trial 일부 때문에 step n이 `112 -> 110`으로 줄어듦.

| Event | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig. | Step (M±SD, cm) | Nonstep (M±SD, cm) | Estimate (cm) | Sig. |
|---|---|---|---|---|---|---|---|---|
| platform onset | 0.52±0.06 | 0.58±0.07 | −0.05 | *** | 13.08±1.65 | 14.63±1.64 | −1.19 | *** |
| step onset | 0.22±0.17 | 0.34±0.10 | −0.11 | *** | 5.63±4.32 | 8.65±2.67 | −2.90 | *** |

- 해석
	- `DV1_norm`과 `DV1_abs_cm`은 분자가 동일하므로 방향 해석은 같아야 한다.
	- raw 표본에서도 두 이벤트 모두 step군이 nonstep군보다 **더 후방(DV1이 더 작음)**.
	- cm로 보면 platform onset에서 약 `1.19 cm`, step onset에서 약 `2.90 cm` 차이로 관찰됨.


# 3. 결론 

- DV1 기준으로 보면, step이 nonstep보다 **더 앞**에 있는 게 아니라 오히려 **더 뒤(후방)**에 있음.
	- 통계 검정은 foot_len 정규화 값(`DV1_norm`) 기준으로 수행.
	- platform onset: step < nonstep (`0.51` vs `0.57`, Estimate `-0.04`, `***`; 절대 cm 환산 약 1.1 cm 차이)
	- step onset: step < nonstep (`0.24` vs `0.32`, Estimate `-0.09`, `***`; 절대 cm 환산 약 2.3 cm 차이)
- 즉 platform onset 시점의 초기 xCOM-BOS 상대 위치 차이가 strategy(step vs nonstep)와 연관되어 있음.



# 4. key papers와의 결과 일치도 

##  key papers

1. [[@Effects of the type and direction of support surface perturbation on postural responses|Chen et al., 2014]]

2. [[@Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance|Van Wouwe et al., 2021]]
	- 섭동 전 xCOM/BOS의 값이 strategy를 변경하는데 있어서 중요하다고 이야기함. 
	- 일치점: Van Wouwe et al. (2021)이 제시한 것처럼, 초기 자세/초기 xCOM-BOS 상태가 전략 차이와 연결된다는 해석과 맞음.
	- 차이점: Van Wouwe는 onset-early window(예: 0-300 ms) 중심이고, 본 분석은 `platform onset`, `step onset` 이벤트 기반이라는 점이 다름.
