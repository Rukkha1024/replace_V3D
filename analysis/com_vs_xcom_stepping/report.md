# COM vs xCOM Stepping Predictor Analysis

## 연구 질문

**"COM이 BOS 밖으로 나가야 stepping이 발생하는가?"**

전통적 가설: xCOM이 BOS를 벗어나면(MoS < 0) 보상 stepping이 발생한다.
→ 본 분석은 이 가설이 perturbation stepping 데이터에서 성립하는지 검증한다.

## 데이터 요약

- **184 trials** (112 step, 72 nonstep)
- all_trials_timeseries.csv 기반, platform sheet에서 step_TF join
- 기준 시점(ref_frame): stepping → step_onset_local, nonstep → 동일 (subject, velocity) stepping 평균
- velocities in timeseries: 20, 25, 30, 35, 45, 50, 60, 70, 110, 130, 135, 200 cm/s

---

## 분석 결과

### 1. COM/xCOM inside BOS at step onset (이진 분류)

**COM (convex hull)**

|             | step | nonstep |
|-------------|-----:|--------:|
| **inside**  |  110 |      72 |
| **outside** |    2 |       0 |

Fisher exact p = 0.5210 (유의하지 않음)

**xCOM (MoS > 0)**

|             | step | nonstep |
|-------------|-----:|--------:|
| **inside**  |  105 |      72 |
| **outside** |    7 |       0 |

Fisher exact p = 0.0437

- COM: **98.2%** (110/112) stepping trial에서 COM이 BOS 내부
- xCOM: **93.8%** (105/112) stepping trial에서 xCOM이 BOS 내부
- nonstep: COM, xCOM 모두 **100%** BOS 내부

→ **이진 기준(inside/outside)은 stepping 예측에 사실상 무의미함.**
대부분의 stepping이 COM/xCOM이 BOS 내부에 있을 때 발생한다.

### 2. 연속 MoS 비교 (Mann-Whitney U)

| Metric          | U     | p          | r (rank-biserial) | median (step) | median (nonstep) |
|-----------------|------:|-----------:|-------------------:|--------------:|-----------------:|
| MoS (xCOM-hull) | 1748  | 9.40e-11   | 0.566              | lower         | higher           |
| COM-hull dist   | 2335  | 2.95e-06   | 0.410              | lower         | higher           |

- 두 지표 모두 **step 그룹이 유의하게 낮은 안정성**을 보임
- MoS의 효과 크기(r=0.57)가 COM(r=0.41)보다 큼 → xCOM이 더 민감한 지표

### 3. ROC/AUC

| Metric          | AUC   |
|-----------------|------:|
| MoS (xCOM-hull) | 0.783 |
| COM-hull dist   | 0.705 |

- MoS AUC = 0.783: 양호한 판별력 (0.7~0.8 구간)
- COM AUC = 0.705: 보통 수준

### 4. MoS < 0 시점 분석 (핵심 발견)

**110개 stepping trial (step_onset_local 존재)**

| 지표                        | 값             |
|-----------------------------|---------------:|
| Step onset 시점 MoS < 0     | 7/110 (**6.4%**) |
| Step onset 이후 MoS < 0 도달 | 38/110 (**34.5%**) |
| MoS 최솟값 도달 중앙값       | step onset + **830 ms** |

**속도별 breakdown:**

| velocity | n_step | MoS<0 @ onset | MoS<0 ever after |
|---------:|-------:|---------------:|-----------------:|
|       20 |     13 |   0/13 ( 0.0%) |  10/13 (76.9%)   |
|       25 |      7 |   0/ 7 ( 0.0%) |   4/ 7 (57.1%)   |
|       30 |     30 |   3/30 (10.0%) |   9/30 (30.0%)   |
|       35 |     16 |   3/16 (18.8%) |   5/16 (31.2%)   |
|       45 |      6 |   0/ 6 ( 0.0%) |   0/ 6 ( 0.0%)   |
|       50 |      2 |   0/ 2 ( 0.0%) |   0/ 2 ( 0.0%)   |
|       60 |     11 |   0/11 ( 0.0%) |   6/11 (54.5%)   |
|       70 |      4 |   1/ 4 (25.0%) |   1/ 4 (25.0%)   |
|      110 |      4 |   0/ 4 ( 0.0%) |   0/ 4 ( 0.0%)   |
|      130 |     12 |   0/12 ( 0.0%) |   3/12 (25.0%)   |
|      135 |      2 |   0/ 2 ( 0.0%) |   0/ 2 ( 0.0%)   |
|      200 |      3 |   0/ 3 ( 0.0%) |   0/ 3 ( 0.0%)   |

---

## 해석

### Proactive (선행적) Stepping

- **93.6%** 의 stepping trial에서 step onset 시점 MoS > 0 (안정 상태)
- stepping은 **MoS < 0 (실제 불안정) 이전에** 이미 발생
- 피험자는 perturbation 감지 즉시 **예방적으로** stepping 반응을 시작함
- MoS < 0에 도달하는 경우(34.5%)도 step onset 이후 ~830ms에 도달 → 이는 stepping 동작 중 체중 이동 단계에서 일시적으로 발생하는 현상

### MoS의 역할 재해석

- MoS 자체가 stepping의 **"원인"이 아님** (xCOM과 BOS의 거리 지표일 뿐)
- **이진 임계값(MoS = 0)은 stepping trigger로 무효**:
  - step onset 시점 MoS < 0인 trial은 6.4%에 불과
  - nonstep trial에서 MoS < 0은 0건 (BOS 안쪽에 머무름)
- **연속 MoS 값은 그룹 간 차이를 보임** (AUC=0.78): "MoS가 낮을수록 stepping 확률이 높다"는 **경향(tendency)**이지, "MoS < 0이면 stepping한다"는 **규칙(rule)**이 아님

### 속도 의존성 패턴

- **저속 (20-25 cm/s)**: 느린 섭동 → step 이후에도 체중 이동 과정에서 MoS < 0에 도달하는 비율 높음 (57-77%)
  - 섭동이 지속되면서 stepping 동작 중에도 불안정이 누적됨
- **중속 (30-35 cm/s)**: step onset 시점 MoS < 0이 10-19%로 가장 높음
  - 이 구간에서 가장 "임계적"인 stepping이 발생 (proactive와 reactive의 경계)
- **고속 (45+ cm/s)**: MoS < 0 비율이 전반적으로 낮음
  - 고속 섭동에서 stepping이 매우 빠르게(반사적으로) 발생하여 MoS가 충분히 감소하기 전에 이미 stepping 완료

### 결론

1. **"COM/xCOM이 BOS를 벗어나야 stepping이 발생한다"는 가설은 기각됨**
2. Stepping은 perturbation에 대한 **proactive/anticipatory response**
3. MoS는 stepping의 원인이 아닌 **연속적 경향 지표**: MoS가 작을수록 stepping 가능성 증가 (AUC=0.78)
4. MoS < 0 (실제 불안정)은 대부분 stepping 동작 중에 **결과적으로** 발생하는 현상

---

## 재현

```bash
conda run -n module python analysis/com_vs_xcom_stepping/analyze_com_vs_xcom_stepping.py
```

**입력**: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`, `data/all_data/` (C3D)
**출력**: fig1~fig6 PNG (이 폴더에 생성)

## Figures

| File | Description |
|------|-------------|
| fig1_contingency_bars.png | COM/xCOM inside BOS vs step/nonstep 분할표 |
| fig2_mos_signed_distribution.png | MoS (xCOM-hull) violin+strip by step_TF |
| fig3_com_signed_distribution.png | COM-hull distance violin+strip by step_TF |
| fig4_scatter_com_vs_xcom.png | COM vs xCOM scatter (quadrant) |
| fig5_roc_curve.png | ROC curve (MoS vs COM as stepping predictor) |
| fig6_example_timeseries.png | 대표 stepping trial 시계열 (COM/xCOM vs BOS) |
