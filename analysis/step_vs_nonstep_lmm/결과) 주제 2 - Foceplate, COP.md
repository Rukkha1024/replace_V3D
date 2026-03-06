# 1. 가설

동일 perturbation 강도(mixed velocity)에서 step 전략은 nonstep 전략보다 COP 이동 범위, 경로장, 평균 속도가 더 클 것이다. Forward translation perturbation이므로 AP 방향(Y축)에서 주된 차이가 관찰될 것이다. GRF 역시 step에서 더 큰 크기를 보일 것이다.

**분석변수:**
- COP: range (max − min), path length, peak velocity, **mean velocity (= path_length / window_duration)** — X축(ML), Y축(AP) 각각
- GRF: abs peak, range — X축(ML), Y축(AP), Z축(Vertical) 각각
- Ankle Torque: abs peak (`AnkleTorqueMid_Y_peak`)

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
| `COP_X_range` (ML, m) | 0.10±0.17 | 0.09±0.01 | 0.01 | 0.02 | 0.61 | n.s. |
| `COP_X_path_length` (ML, m) | 0.14±0.35 | 0.11±0.03 | 0.03 | 0.04 | 0.84 | n.s. |
| `COP_X_peak_velocity` (ML, m/s) | 3.43±17.47 | 1.08±0.45 | 2.34 | 2.04 | 1.15 | n.s. |
| `COP_X_mean_velocity` (ML, m/s) | 0.44±1.63 | 0.25±0.11 | 0.18 | 0.19 | 0.94 | n.s. |
| `COP_Y_range` (AP, m) | 0.10±0.04 | 0.05±0.04 | 0.06 | 0.01 | 10.09 | *** |
| `COP_Y_path_length` (AP, m) | 0.16±0.07 | 0.10±0.07 | 0.07 | 0.01 | 7.86 | *** |
| `COP_Y_peak_velocity` (AP, m/s) | 1.70±0.94 | 1.16±1.04 | 0.73 | 0.13 | 5.44 | *** |
| `COP_Y_mean_velocity` (AP, m/s) | 0.33±0.16 | 0.21±0.14 | 0.15 | 0.02 | 7.87 | *** |

- 모델: `DV ~ step_TF + (1|subject)`, REML, BH-FDR by family (alpha = 0.05)
- 현재 재실행 표본: trials = `125` (step = `52`, nonstep = `73`), subjects = `24`

### COP 해석

1. **AP 방향(Y축) 4개 변수 모두 유의 (p_fdr < 0.001):** Step 전략에서 COP가 전후 방향으로 더 넓게, 더 긴 경로를 따라, 더 빠르게 이동했다.
   - `COP_Y_range`: step이 nonstep 대비 약 **2.1배** (10.5 cm vs. 4.9 cm)
   - `COP_Y_path_length`: step이 nonstep 대비 약 **1.6배** (15.5 cm vs. 9.8 cm)
   - `COP_Y_peak_velocity`: step이 nonstep 대비 약 **1.5배** (1.70 m/s vs. 1.16 m/s)
   - `COP_Y_mean_velocity`: step이 nonstep 대비 약 **1.6배** (0.33 m/s vs. 0.21 m/s)
2. **ML 방향(X축) 4개 변수 모두 비유의:** Forward translation perturbation에서 좌우 방향 COP 반응은 전략 간 차이를 보이지 않았다.
3. **물리적 해석:** Forward perturbation은 주로 AP 방향 불안정성을 유발하므로, COP 보상 반응도 AP 방향에 집중된다. Step 전략에서 COP의 AP 이동이 더 큰 것은 (a) stepping 직전까지의 balance recovery 시도가 더 크거나, (b) 같은 시간 내에서 더 많은 동적 조정이 필요했음을 시사한다.

---

## 2.2 GRF 변수 (Force/Torque family)

| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | SE | t | Sig |
|---|---:|---:|---:|---:|---:|---|
| `GRF_X_peak` (ML, N) | 239.87±206.07 | 295.27±233.19 | 1.78 | 11.32 | 0.16 | n.s. |
| `GRF_X_range` (ML, N) | 377.84±329.73 | 463.69±374.24 | 13.61 | 20.73 | 0.66 | n.s. |
| `GRF_Y_peak` (AP, N) | 51.80±29.28 | 26.32±15.61 | 26.54 | 3.58 | 7.41 | *** |
| `GRF_Y_range` (AP, N) | 71.47±38.09 | 35.13±20.80 | 39.30 | 4.48 | 8.76 | *** |
| `GRF_Z_peak` (Vertical, N) | 171.53±142.51 | 174.73±126.25 | 23.36 | 17.39 | 1.34 | n.s. |
| `GRF_Z_range` (Vertical, N) | 269.39±178.06 | 296.21±200.45 | 25.39 | 19.20 | 1.32 | n.s. |

- 모델: 동일 (`DV ~ step_TF + (1|subject)`, REML, BH-FDR by Force/Torque family)

### GRF / Ankle Torque 해석

1. **AP 방향(Y축) GRF만 유의 (p_fdr < 0.001):**
   - `GRF_Y_peak`: step(51.8 N) > nonstep(26.3 N), 약 **2.0배**
   - `GRF_Y_range`: step(71.5 N) > nonstep(35.1 N), 약 **2.0배**
2. **ML 방향(X축)과 Vertical(Z축) GRF는 모두 비유의:** 좌우 및 수직 지면반력에서는 전략 간 차이가 없었다.
3. **Ankle Torque (`AnkleTorqueMid_Y_peak`):** step = `1.66±0.36`, nonstep = `1.87±0.30`, Estimate = `−0.14`, t = `−6.60`, `***`.
   - 현재 rerun에서는 ankle torque가 더 이상 FDR 비유의가 아니었다.
   - 방향은 `step < nonstep`이므로, 고정지지(nonstep) 전략에서 stance-side ankle torque 의존이 더 크다고 해석하는 편이 현재 결과와 더 잘 맞는다.
4. **물리적 해석:** Step 전략에서는 AP 방향 GRF와 COP 반응이 크게 증가했지만, stance-side ankle torque peak는 오히려 nonstep에서 더 컸다. 즉, step은 더 큰 AP 방향 힘/압력중심 이동과 연결되고, nonstep은 상대적으로 ankle torque 기반의 fixed-support 제어를 더 많이 사용했을 가능성이 있다.

---

## 2.3 COP + GRF + Ankle Torque 종합 요약

| 방향/계열 | 결과 | 해석 |
|---|---|---|
| **COP AP (Y축)** | 4/4 유의 (`step >> nonstep`) | Forward perturbation의 주 방향. Step 전략에서 더 큰 AP 보상 반응. |
| **GRF AP (Y축)** | 2/2 유의 (`step ≈ 2× nonstep`) | Step 전략에서 더 큰 전후 방향 추진/제동 반응. |
| **ML (X축)** | COP 0/4, GRF 0/2 유의 | 좌우 방향 반응은 전략 무관. |
| **Vertical (Z축)** | GRF 0/2 유의 | 체중지지 하중은 전략 무관. |
| **Ankle Torque** | 1/1 유의 (`step < nonstep`) | Nonstep이 stance-side ankle torque 전략을 더 강하게 사용했을 가능성. |

- **Force/Torque family 7개 변수 중 3개 유의** (`GRF_Y_peak`, `GRF_Y_range`, `AnkleTorqueMid_Y_peak`)
- **COP 관련 8개 변수 중 4개 유의** (`COP_Y_range`, `COP_Y_path_length`, `COP_Y_peak_velocity`, `COP_Y_mean_velocity`)
- 전체 current rerun LMM DV는 `17/38`이 FDR 유의였고, forceplate/COP 관련 유의 변수는 AP 방향 + ankle torque에 집중되었다.

---

# 3. 결론 

1. **COP:** Step 전략은 nonstep 대비 AP 방향 COP range 약 2.1배, path length 1.6배, peak velocity 1.5배, **mean velocity 1.6배** 유의하게 컸다. ML 방향 COP 변수는 모두 비유의였다.
2. **GRF:** Step 전략의 AP 방향 GRF peak와 range가 nonstep의 약 2배로 유의하였다. ML 및 Vertical GRF는 비유의였다.
3. **방향 특이성:** 유의한 COP/GRF 차이는 AP 방향(Y축)에 집중되었다. 이는 forward translation perturbation의 주 불안정 방향과 일치하며, step/nonstep 전략 판별에서 **AP 방향 COP와 GRF가 핵심 지표**임을 시사한다.
4. **Ankle Torque:** 현재 rerun에서는 ankle torque도 유의하였다(`***`). 다만 방향은 `step < nonstep`이어서, step 전략의 힘 증가를 ankle torque 하나로 설명하기보다 **nonstep의 fixed-support 제어가 ankle torque에 더 의존했다**고 해석하는 편이 현재 결과와 맞다.
5. **종합 해석:** Nonstep 전략은 고정 지지면 내에서 ankle/hip torque로 COM을 회복하므로 상대적으로 작은 COP 변위와 GRF로 충분했고, step 전략에서는 stepping 직전까지의 balance recovery 시도 과정에서 더 큰 AP 방향 힘과 COP 조정이 발생한 것으로 해석된다. 즉, 현재 rerun은 step이 단순히 “더 큰 perturbation 반응”이라기보다, **동일 강도에서도 balance recovery 방식이 다르다**는 점을 더 직접적으로 보여준다.

---

# 4. Key papers와의 결과 일치도

## 4.1 Zhao et al., 2020 (support-surface perturbation + stepping preparation)

- **실험:** Force platform (Kistler 9281B), 1500 Hz sampling. Support-surface perturbation during stance, stepping preparation task.
- **COP 결과:** 최대 COP displacement 보고: "maximum CoP displacement … −5.50 ± 0.32 cm … −2.10 ± 0.36 cm"
- **본 연구와 비교:**
  - 본 연구의 `COP_Y_range`(AP): step = 10.5 cm, nonstep = 4.9 cm. Zhao의 최대 COP displacement 범위와 같은 order of magnitude를 보인다.
  - 다만 Zhao는 stepping preparation 조건이고, 본 연구는 자발적 step/nonstep 비교이므로 직접 대응은 제한적이다.

## 4.2 Dettmer et al., 2016 (support surface translation, NeuroCom)

- Support surface translation 조건에서 COP 변수를 분석.
- 실험 패러다임이 유사하나 perturbation 장비 및 강도 설정이 다를 수 있으므로 수치 직접 비교에는 주의가 필요하다.

## 4.3 Lemay et al., 2014 (COP trajectory length, SCI vs. able-bodied)

- COP total trajectory length가 maximum excursion보다 균형 능력 차이를 더 잘 감별한다고 보고.
- **일치점:** 본 연구에서도 `COP_Y_path_length`가 유의하였으며, range와 함께 step/nonstep 판별에 기여하였다. Path length가 단순 range 이상의 정보를 포함한다는 주장과 일관된다.
- **차이점:** Lemay는 perturbation이 아닌 자발적 standing limits of stability 과제였다. 적용 맥락이 다르다.

## 4.4 Peterson et al., 2018 (APA → step outcomes, Parkinson's disease)

- Anticipatory Postural Adjustments(APA)가 step 결과와 연관됨을 보고.
- **일치점:** Step 전략에서 COP/GRF 반응이 더 크다는 본 연구 결과와 방향이 일치한다.
- **보완점:** 현재 rerun에서는 ankle torque가 `step < nonstep`으로 유의했기 때문에, pre-step 제어가 항상 step에서 더 크다고 일반화하기보다, 어떤 제어 채널을 더 쓰는지가 전략별로 다르다고 해석하는 편이 적절하다.

## 4.5 Analysis of Dynamic Balance Against Perturbation in Young and Elderly Subjects

- Platform tilting(기울임) 패러다임으로, 본 연구의 platform translation과 perturbation 유형이 다르다.
- 변수 아이디어 참고용으로는 의미가 있으나, 본 연구 결과와 직접 비교하기에는 부적합하다.
