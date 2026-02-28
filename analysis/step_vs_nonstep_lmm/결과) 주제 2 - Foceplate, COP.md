# 1. 가설 

동일 perturbation 강도(mixed velocity)에서 step 전략은 nonstep 전략보다 COP 이동 범위, 경로장, 평균 속도가 더 클 것이다. Forward translation perturbation이므로 AP 방향(Y축)에서 주된 차이가 관찰될 것이다. GRF 역시 step에서 더 큰 크기를 보일 것이다.

**분석변수:**
- COP: range (max − min), path length, peak velocity, **mean velocity (= path_length / window_duration)** — X축(ML), Y축(AP) 각각
- GRF: abs peak, range — X축(ML), Y축(AP), Z축(Vertical) 각각
- Ankle Torque: abs peak (→ 별도 문서 `결과) 주제 2 - ankle torque`)

**분석구간:**
- Step trial: `[platform_onset, step_onset]` (실제 step onset)
- Nonstep trial: `[platform_onset, mean step_onset]` (동일 subject × velocity 내 step trial의 step_onset 평균)

**좌표계:** `X = +Right (ML)`, `Y = +Anterior (AP)`, `Z = +Up (Vertical)`
**COP 전처리:** onset-zeroed (perturbation onset 시점을 0으로 보정)
**GRF 전처리:** Bertec forceplate 1000 Hz 수집 → 4th-order Butterworth LPF 10 Hz → empty plate inertia subtraction

---

# 2. 결과 

## 2.1 COP 변수 (Balance/Stability family)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | SE | t | Sig |
|---|---:|---:|---:|---:|---:|---|
| `COP_X_range` (ML, m) | 0.10±0.17 | 0.09±0.01 | 0.01 | 0.02 | 0.59 | n.s. |
| `COP_X_path_length` (ML, m) | 0.14±0.35 | 0.11±0.03 | 0.03 | 0.04 | 0.85 | n.s. |
| `COP_X_peak_velocity` (ML, m/s) | 3.38±17.31 | 1.06±0.43 | 2.31 | 2.04 | 1.13 | n.s. |
| `COP_X_mean_velocity` (ML, m/s) | 0.43±1.61 | 0.25±0.11 | 0.17 | 0.19 | 0.92 | n.s. |
| `COP_Y_range` (AP, m) | 0.10±0.04 | 0.05±0.04 | 0.06 | 0.01 | 9.97 | *** |
| `COP_Y_path_length` (AP, m) | 0.15±0.07 | 0.10±0.07 | 0.07 | 0.01 | 7.71 | *** |
| `COP_Y_peak_velocity` (AP, m/s) | 1.67±0.95 | 1.15±1.04 | 0.72 | 0.13 | 5.45 | *** |
| `COP_Y_mean_velocity` (AP, m/s) | 0.33±0.16 | 0.21±0.14 | 0.15 | 0.02 | 7.79 | *** |

- 모델: `DV ~ step_TF + (1|subject)`, REML, BH-FDR by family (alpha = 0.05)
- trials = 125 (step = 53, nonstep = 72), subjects = 24

### COP 해석

1. **AP 방향(Y축) 4개 변수 모두 유의 (p_fdr < 0.001):** Step 전략에서 COP가 전후 방향으로 더 넓게, 더 긴 경로를 따라, 더 빠르게 이동했다.
   - `COP_Y_range`: step이 nonstep 대비 약 **2.1배** (10.3 cm vs. 4.8 cm)
   - `COP_Y_path_length`: step이 nonstep 대비 약 **1.6배** (15.4 cm vs. 9.6 cm)
   - `COP_Y_peak_velocity`: step이 nonstep 대비 약 **1.5배** (1.67 m/s vs. 1.15 m/s)
   - `COP_Y_mean_velocity`: step이 nonstep 대비 약 **1.6배** (0.33 m/s vs. 0.21 m/s)
2. **ML 방향(X축) 4개 변수 모두 비유의:** Forward translation perturbation에서 좌우 방향 COP 반응은 전략 간 차이를 보이지 않았다.
3. **물리적 해석:** Forward perturbation은 주로 AP 방향 불안정성을 유발하므로, COP 보상 반응도 AP 방향에 집중된다. Step 전략에서 COP의 AP 이동이 더 큰 것은 (a) stepping 직전까지의 balance recovery 시도가 더 크거나, (b) 같은 시간 내에서 더 많은 동적 조정이 필요했음을 시사한다. Nonstep은 fixed-support 전략(ankle/hip)으로 COM을 BOS 내에서 회복하므로, 상대적으로 작은 COP 변위로 충분했다.

---

## 2.2 GRF 변수 (Force/Torque family)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | SE | t | Sig |
|---|---:|---:|---:|---:|---:|---|
| `GRF_X_peak` (ML, N) | 237.21±205.00 | 287.63±225.44 | 1.88 | 10.91 | 0.17 | n.s. |
| `GRF_X_range` (ML, N) | 374.38±327.51 | 452.05±363.33 | 13.85 | 20.61 | 0.67 | n.s. |
| `GRF_Y_peak` (AP, N) | 51.03±29.53 | 25.82±15.12 | 26.33 | 3.56 | 7.40 | *** |
| `GRF_Y_range` (AP, N) | 70.49±38.40 | 34.22±19.42 | 39.04 | 4.43 | 8.81 | *** |
| `GRF_Z_peak` (Vertical, N) | 170.23±141.45 | 168.76±116.30 | 21.31 | 17.55 | 1.21 | n.s. |
| `GRF_Z_range` (Vertical, N) | 267.23±177.03 | 287.01±185.68 | 22.53 | 19.47 | 1.16 | n.s. |

- 모델: 동일 (`DV ~ step_TF + (1|subject)`, REML, BH-FDR by Force/Torque family)

### GRF 해석

1. **AP 방향(Y축) GRF만 유의 (p_fdr < 0.001):**
   - `GRF_Y_peak`: step(51.0 N) > nonstep(25.8 N), 약 **2.0배**
   - `GRF_Y_range`: step(70.5 N) > nonstep(34.2 N), 약 **2.1배**
2. **ML 방향(X축)과 Vertical(Z축) 모두 비유의:** 좌우 및 수직 지면반력에서는 전략 간 차이가 없었다.
3. **Ankle Torque (`AnkleTorqueMid_Y_peak`):** Estimate = −0.0681 Nm/kg, t = −2.005, FDR 비유의 (별도 문서 참조).
4. **물리적 해석:** Step 전략에서 AP 방향 GRF가 약 2배인 것은, stepping 준비 및 실행 과정에서 발판을 통해 전달되는 전후 방향 힘이 크게 증가함을 의미한다. 이는 COP_Y의 더 큰 변위와 일관된 결과이다. COP = f(GRF, moment) 관계에서, AP 방향 GRF 증가가 AP 방향 COP 변위 증가의 직접적 원인이다. Vertical GRF(체중지지)와 ML GRF는 전략 간 동등하였으므로, 전략 차이는 AP 방향 수평력에 국한된다.

---

## 2.3 COP + GRF 종합 요약

| 방향 | COP 결과 | GRF 결과 | 해석 |
|---|---|---|---|
| **AP (Y축)** | 4/4 유의 (step >> nonstep) | 2/2 유의 (step ≈ 2× nonstep) | Forward perturbation의 주 방향. Step 전략에서 더 큰 AP 보상 반응. |
| **ML (X축)** | 0/4 유의 | 0/2 유의 | 좌우 방향 반응은 전략 무관. |
| **Vertical (Z축)** | N/A | 0/2 유의 | 체중지지 하중은 전략 무관. |

- **Force/Torque family 7개 변수 중 2개 유의 (GRF_Y_peak, GRF_Y_range)**
- **Balance/Stability family 중 COP 관련 8개 변수 중 4개 유의 (COP_Y_range, COP_Y_path_length, COP_Y_peak_velocity, COP_Y_mean_velocity)**
- 전체 36개 DV 중 14개 FDR 유의. 유의 변수 중 COP/GRF 관련은 모두 Y축(AP)에 집중.

---

# 3. 결론 

1. **COP:** Step 전략은 nonstep 대비 AP 방향 COP range 약 2.1배, path length 1.6배, peak velocity 1.5배, **mean velocity 1.6배** 유의하게 컸다. ML 방향 COP 변수는 모두 비유의.
2. **GRF:** Step 전략의 AP 방향 GRF peak와 range가 nonstep의 약 2배로 유의. ML 및 Vertical GRF는 비유의.
3. **방향 특이성:** 유의한 차이가 AP 방향(Y축)에만 집중. Forward translation perturbation의 주 불안정 방향과 일치하며, step/nonstep 전략 판별에서 **AP 방향 COP와 GRF가 핵심 지표**임을 시사.
4. **Ankle Torque:** FDR 비유의. Ankle torque 단독으로는 전략 차이를 설명하기 어렵다 (별도 문서 참조).
5. **종합 해석:** Nonstep 전략은 고정 지지면 내에서 ankle/hip torque로 COM을 회복하므로 상대적으로 작은 COP 변위와 GRF로 충분. Step 전략에서는 stepping 직전까지의 balance recovery 시도가 실패하는 과정에서 더 큰 AP 방향 힘과 COP 조정이 발생한 것으로 해석. 이는 step이 "더 큰 perturbation 반응의 결과"라기보다, **동일 강도에서도 초기 balance recovery 과정의 효율성 차이**가 전략 선택을 좌우할 수 있음을 보여줌.

---

# 4. Key papers와의 결과 일치도

## 4.1 Zhao et al., 2020 (support-surface perturbation + stepping preparation)

- **실험:** Force platform (Kistler 9281B), 1500 Hz sampling. Support-surface perturbation during stance, stepping preparation task.
- **COP 결과:** 최대 COP displacement 보고: "maximum CoP displacement … −5.50 ± 0.32 cm … −2.10 ± 0.36 cm"
- **본 연구와 비교:**
  - 본 연구의 COP_Y_range(AP): step = 10.3 cm, nonstep = 4.8 cm. Zhao의 최대 COP displacement 범위와 유사한 order of magnitude.
  - 다만 Zhao는 stepping preparation 조건이고, 본 연구는 자발적 step/nonstep 비교이므로 직접 대응은 제한적.

## 4.2 Dettmer et al., 2016 (support surface translation, NeuroCom)

- Support surface translation 조건에서 COP 변수를 분석.
- 실험 패러다임이 유사하나 perturbation 장비 및 강도 설정이 다를 수 있으므로 수치 직접 비교에는 주의 필요.

## 4.3 Lemay et al., 2014 (COP trajectory length, SCI vs. able-bodied)

- COP total trajectory length가 maximum excursion보다 균형 능력 차이를 더 잘 감별한다고 보고.
- **일치점:** 본 연구에서도 COP_Y_path_length가 유의(t = 7.708)하였으며, range(t = 9.971)와 함께 step/nonstep 판별에 기여. Path length가 단순 range 이상의 정보를 포함한다는 주장과 일관.
- **차이점:** Lemay는 perturbation이 아닌 자발적 standing limits of stability 과제. 적용 맥락이 다름.

## 4.4 Peterson et al., 2018 (APA → step outcomes, Parkinson's disease)

- Anticipatory Postural Adjustments(APA)가 step 결과와 연관됨을 보고.
- **일치점:** Step 전략에서 COP/GRF 반응이 더 크다는 본 연구 결과와 방향이 일치. APA의 크기가 subsequent step의 quality와 관련되므로, pre-step COP 반응 크기 차이가 전략과 연결된다는 해석을 지지.
- **차이점:** Peterson은 PD 환자 대상, waist-pull perturbation. 본 연구는 건강한 젊은 성인, platform translation. Peterson의 APA→step 관계는 step trial 내부의 질적 차이인 반면, 본 연구는 step vs. nonstep 간 비교.

## 4.5 Analysis of Dynamic Balance Against Perturbation in Young and Elderly Subjects

- Platform tilting(기울임) 패러다임으로, 본 연구의 platform translation과 perturbation 유형이 다름.
- 변수 아이디어 참고용. 본 연구 결과와 직접 비교에는 부적합. 서론에서 perturbation 유형 다양성을 설명할 때 인용 가능.
