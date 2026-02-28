---
cssclass: clean-embeds
date created: 2026-01-19. 00.31
date revised: 2026-03-01
---

# 1. 가설

동일 perturbation 강도(mixed velocity)에서 xCOM/BOS 정규화 지표(DV1–DV3)는 step 전략과 nonstep 전략 간 유의한 차이를 보일 것이다. Forward translation perturbation이므로 AP 방향(X축)에서 주된 차이가 관찰될 것이다.

**분석 변수 (DV1–DV3):**

| DV | 공식 | 선행연구 | 의미 |
|---|---|---|---|
| DV1 | `(xCOM_hof − BOS_rear) / foot_length` | Van Wouwe (2021), Salot (2016), Patel (2015), Bhatt group | xCOM의 BOS rear 대비 상대 위치 (무차원) |
| DV2 | `(COM_X − BOS_rear) / foot_length` | Joshi (2018) 위치항 | COM 위치항만 분리한 BOS rear 대비 상대 위치 (무차원) |
| DV3 | `(vCOM_X − vBOS_rear) / √(g × height)` | Joshi (2018) 속도항 | COM–BOS 상대 속도를 신장으로 정규화 (무차원) |

- DV1 Supplementary: `DV1_abs_cm = (xCOM_hof − BOS_rear) × 100 [cm]` — 직관적 크기 해석용 병행.

**분석 이벤트:** `platform_onset`, `step_onset` (2개)

**분석구간:**
- Step trial: `[platform_onset_local, step_onset_local]`
- Nonstep trial: `[platform_onset_local, same (subject, velocity) step trial의 step_onset_local 평균]`
- Fallback: prefilter 이벤트 기반 subject-velocity mean 보완.

**좌표계:** `X = +Anterior (AP)`, `Y = +Left (ML)`, `Z = +Up (Vertical)`

**통계 모델:** `DV ~ step_TF + (1|subject)`, REML (`lmerTest`). BH-FDR 일괄 보정.

---

# 2. 결과

## 2.1 1차 스크리닝 (COM + xCOM + xCOM/BOS, 24 DVs)

### 목적

DV1–DV3 확인 분석 전에, COM + xCOM + xCOM/BOS 계열에서 step/nonstep 차이가 나타나는 후보 변수를 탐색한다.

### 스크리닝 변수 구성

| Group | 변수 |
|---|---|
| COM (X/Y/Z) | `max-min`, `mean_velocity`, `peak_velocity` |
| xCOM (X/Y/Z) | `max-min`, `mean_velocity`, `peak_velocity` |
| xCOM/BOS (AP/ML, foot 정규화) | `platform_onset`, `step_onset`, `window_mean` |

- 다중비교: 24개 DV 전체 BH-FDR 보정.
- 전체 결과 파일: `com_xcom_screening_lmm_results.csv`

### FDR 유의 변수 (9/24개)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|---|---:|---:|---:|---|
| `xCOM_Y_peak_velocity` | 0.2705±0.0940 | 0.1483±0.0703 | 0.1269 | *** |
| `xCOM_BOS_AP_foot_mean_window` | 0.3292±0.1256 | 0.4036±0.0996 | −0.0666 | *** |
| `xCOM_BOS_AP_foot_platformonset` | 0.5091±0.0596 | 0.5715±0.0654 | −0.0440 | *** |
| `xCOM_BOS_AP_foot_steponset` | 0.2354±0.1720 | 0.3232±0.0996 | −0.0893 | *** |
| `xCOM_Y_mean_velocity` | 0.0703±0.0320 | 0.0450±0.0233 | 0.0277 | *** |
| `COM_Y_peak_velocity` | 0.0561±0.0294 | 0.0340±0.0239 | 0.0230 | *** |
| `xCOM_Y_max_min` | 0.0235±0.0116 | 0.0161±0.0124 | 0.0078 | ** |
| `COM_Z_mean_velocity` | 0.0250±0.0118 | 0.0220±0.0082 | 0.0036 | * |
| `xCOM_X_mean_velocity` | 0.1286±0.0778 | 0.1347±0.0694 | 0.0133 | * |

### 스크리닝 해석

1. 유의 신호는 xCOM 계열, 특히 Y축(`xCOM_Y_peak_velocity`, `xCOM_Y_mean_velocity`, `xCOM_Y_max_min`)에 집중되었다.
2. `xCOM_BOS_AP_foot`는 이벤트 2개 + 구간평균 모두 유의(`step < nonstep`)하여, AP 안정성 차이 후보로 우선순위가 가장 높았다.
3. COM 단독 변수는 `COM_Y_peak_velocity`, `COM_Z_mean_velocity`만 유의했고 나머지는 비유의였다.
4. 따라서 2차 확인 분석(DV1–DV3)은 `xCOM/BOS_AP_foot` 계열을 중심으로 위치항(DV2)과 속도항(DV3)을 분리 검증하도록 설계하였다.

---

## 2.2 DV1–DV3 Main Effect (LMM)

### 부호 해석 기준

| Metric | (+) 의미 | (−) 의미 | 판정 기준 |
|---|---|---|---|
| DV1 = (xCOM_hof − BOS_rear) / foot_len | xCOM이 BOS rear보다 전방 | xCOM이 BOS rear보다 후방 | `step_TFstep` 계수 부호 |
| DV2 = (COM_X − BOS_rear) / foot_len | COM이 BOS rear보다 전방 | COM이 BOS rear보다 후방 | `step_TFstep` 계수 부호 |
| DV3 = (vCOM_X − vBOS_rear) / √(g×h) | 상대 전방 속도 증가 | 상대 후방 속도 증가 | `step_TFstep` 계수 부호 |

- AP(X)축 부호: `+` 전방, `−` 후방.
- DV1, DV2: 값이 작을수록 xCOM/COM이 BOS rear에 가까움 → 후방 불안정.

### 모델 정보

- 모델: `DV ~ step_TF + (1|subject)`, REML
- 표본: trials = 125 (step = 53, nonstep = 72), subjects = 24
- 참조 수준: nonstep (Estimate = step − nonstep)
- 다중비교: 6개 DV 전체 BH-FDR 보정

### 주효과 결과 (6개 DV)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig |
|---|---:|---:|---:|---|
| `DV1_xcom_hof_rear_over_foot_platformonset` | 0.5091±0.0596 | 0.5715±0.0654 | −0.0440 | *** |
| `DV1_xcom_hof_rear_over_foot_steponset` | 0.2354±0.1720 | 0.3232±0.0996 | −0.0893 | *** |
| `DV2_com_rear_over_foot_platformonset` | 0.4976±0.0583 | 0.5525±0.0627 | −0.0395 | *** |
| `DV2_com_rear_over_foot_steponset` | 0.2648±0.1109 | 0.2872±0.0745 | −0.0225 | * |
| `DV3_vcom_rel_over_sqrtgh_platformonset` | −0.0245±0.0159 | −0.0291±0.0195 | −0.0001 | n.s. |
| `DV3_vcom_rel_over_sqrtgh_steponset` | 0.0684±0.0429 | −0.0039±0.0184 | 0.0715 | *** |

- FDR 유의: **5/6**

### DV1 해석 (위치 + 속도 통합항)

- Platform onset: step군의 DV1이 nonstep보다 0.04 낮았다(`***`). xCOM이 BOS rear에 상대적으로 더 가까운(후방) 위치에 있었다.
- Step onset: 차이가 0.09로 확대되었다(`***`). 섭동 진행 후 step군의 xCOM이 더 큰 폭으로 후방 이동하였다.

### DV2 해석 (위치항 분리)

- Platform onset: step군의 DV2가 nonstep보다 0.04 낮았다(`***`). COM 위치 자체가 이미 BOS rear에 더 가까웠다.
- Step onset: 차이가 0.02로 줄었으나 유의하였다(`*`). 위치항 기여는 onset 시점에서 더 두드러졌다.

### DV3 해석 (속도항 분리)

- Platform onset: 전략 간 상대 속도 차이는 없었다(n.s.). 섭동 직전 COM–BOS 상대 속도는 동등하였다.
- Step onset: step군에서 상대 전방 속도가 0.07 더 컸다(`***`). 섭동 진행 중 step군의 COM이 BOS 대비 전방으로 더 빠르게 이동하였다.

### DV1–DV3 종합 요약

| 이벤트 | DV1 (xCOM/BOS) | DV2 (위치항) | DV3 (속도항) |
|---|---|---|---|
| platform onset | *** (step < nonstep) | *** (step < nonstep) | n.s. |
| step onset | *** (step < nonstep) | * (step < nonstep) | *** (step > nonstep) |

- Platform onset 시점: 위치항(DV2)이 전략 차이를 주도하였고, 속도항(DV3)은 아직 분화되지 않았다.
- Step onset 시점: 위치항 차이는 축소된 반면 속도항이 크게 분화되어, xCOM 전체 차이(DV1)를 위치+속도 공동으로 설명하는 구조로 전환되었다.

---

## 2.3 DV1 Supplementary — 절대 크기 (cm) 및 확장 표본 검증

### Primary 표본 (125 trials) — norm + abs_cm 병행

| Event | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig | Step (M±SD, cm) | Nonstep (M±SD, cm) | Estimate (cm) | Sig |
|---|---:|---:|---:|---|---:|---:|---:|---|
| platform onset | 0.51±0.06 | 0.57±0.07 | −0.04 | *** | 12.91±1.62 | 14.44±1.62 | −1.10 | *** |
| step onset | 0.24±0.17 | 0.32±0.10 | −0.09 | *** | 5.95±4.45 | 8.19±2.61 | −2.29 | *** |

- 절대 크기: platform onset에서 약 1.1 cm, step onset에서 약 2.3 cm 차이.

### Additional Check — 확장 표본 (184 trials, Raw/No Filtering)

Stance leg 필터를 해제하고 전체 step trial(mixed == 1)을 포함하여 방향 일관성을 재확인하였다.

- 표본: trials = 184, subjects = 24 (step = 112, nonstep = 72)
- step_onset DVs: step onset 미보유 trial 제외로 step n = 110.

| Event | Step (M±SD) | Nonstep (M±SD) | Estimate | Sig | Step (M±SD, cm) | Nonstep (M±SD, cm) | Estimate (cm) | Sig |
|---|---:|---:|---:|---|---:|---:|---:|---|
| platform onset | 0.52±0.06 | 0.58±0.07 | −0.05 | *** | 13.08±1.65 | 14.63±1.64 | −1.19 | *** |
| step onset | 0.22±0.17 | 0.34±0.10 | −0.11 | *** | 5.63±4.32 | 8.65±2.67 | −2.90 | *** |

- 확장 표본에서도 두 이벤트 모두 step군이 nonstep보다 DV1이 낮았다(후방). 방향 일관성이 확인되었다.
- 절대 크기: platform onset에서 약 1.2 cm, step onset에서 약 2.9 cm 차이.

---

# 3. 결론

1. **DV1 (xCOM/BOS):** 두 이벤트 모두 step군이 nonstep보다 유의하게 작았다(`***`). Step trial은 xCOM이 BOS rear에 더 가까운, 즉 후방 불안정 위치에 있었다.
2. **DV2 (위치항):** 두 이벤트 모두 유의하였다(`***`, `*`). Platform onset 시점에서 COM 위치 차이가 전략 분화의 주요 원인이었다.
3. **DV3 (속도항):** Step onset에서만 유의하였다(`***`). 섭동 진행 중 step군의 COM이 BOS 대비 전방으로 더 빠르게 이동하여, 후방 stepping이 필요한 동적 불안정 상태에 이르렀다.
4. **종합:** Platform onset 초기 xCOM–BOS 상대 위치(DV1, DV2)는 전략 선택과 연관되어 있었다. 섭동이 진행될수록 속도항(DV3)이 추가로 분화되면서 step 전략의 동적 불안정을 반영하였다. 6개 DV 중 5개가 FDR 유의(5/6)하여, xCOM/BOS 정규화 지표가 step vs nonstep 전략을 구분하는 유효한 지표임을 확인하였다.

---

# 4. Key papers와의 결과 일치도

| 비교 항목 | 선행연구 결과 | 본 연구 결과 | 판정 |
|---|---|---|---|
| Hof 기반 xCOM/BOS의 전략 구분력 (Van Wouwe, 2021) | onset/early 시점 안정성 지표로 유효 | DV1이 onset/step_onset 모두 유의(`***`) | Consistent |
| COM 위치항 정규화의 차이 (Joshi, 2018) | `(COM−BOS)/foot_length` 위치 지표 사용 보고 | DV2가 두 이벤트 모두 유의(`***`, `*`) | Consistent |
| 속도 정규화 항의 분리 해석 (Joshi, 2018) | `VCOM/BOS`를 별도 지표로 해석 | DV3는 step_onset에서만 유의(`***`), onset은 n.s. | Partially consistent |
| 초기 자세와 전략 variability의 연관 (Van Wouwe, 2021) | 초기 xCOM/BOS 상태가 trial-by-trial 전략 변동을 설명 | Platform onset DV1에서 step < nonstep(`***`) — 초기 위치 차이와 전략 연관 확인 | Consistent |
| Foot-length 정규화의 유효성 (Salot, 2016; Patel, 2015) | foot-length normalized xCOM/BOS가 집단 간 비교에 유효 | Primary(norm)과 Supplementary(abs_cm) 모두 동일 방향 유의 | Consistent |

### 차이점

- Van Wouwe는 onset–300 ms window 중심이었고, 본 분석은 `platform_onset`, `step_onset` 이벤트 기반이다. 이벤트 정의가 다르므로 수치 직접 비교는 제한적이다.
- Joshi의 원 연구는 trip-like perturbation(treadmill)이며 lift-off/touchdown 이벤트를 사용하였다. 본 연구는 platform translation + step onset으로 패러다임이 다르지만, 위치항과 속도항을 분리하는 분석 전략은 동일하게 유효하였다.
- 선행연구(Salot, Joshi)는 stroke/aging 그룹 비교(between-group)였으나, 본 연구는 건강 젊은 성인 내 step vs nonstep 비교(within-condition)이다. 같은 집단 내에서도 xCOM/BOS 차이가 전략을 구분하였다는 점에서, 지표의 민감도가 높음을 뒷받침한다.
