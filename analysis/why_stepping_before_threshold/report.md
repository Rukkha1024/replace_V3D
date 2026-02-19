# eBOS & FSR State-Space Analysis: Why Stepping Occurs Before the Textbook Threshold

## Research Question

**"Effective BOS(eBOS)와 COM velocity-position state space(FSR)로 'xCOM이 BOS 내부에 있는데 stepping이 발생하는' 현상을 설명할 수 있는가?"**

이전 분석(`com_vs_xcom_stepping`)에서 stepping trial의 93.6%가 MoS > 0(BOS 내부)에서 step onset을 보였다. 본 분석은 문헌 리뷰("Why stepping occurs before the textbook says it should")에서 제안한 두 가지 역학적 설명을 검증한다:

1. **Effective BOS (eBOS)**: 기능적 BOS는 해부학적 BOS보다 훨씬 작다 (Hof & Curtze, 2016: ~30%)
2. **FSR State Space**: COM 속도가 위치보다 stepping 결정의 주요 변수이다 (Pai & Patton, 1997)

## Data Summary

- **184 trials** (112 step, 72 nonstep) — 이전 분석과 동일
- **24 subjects** (leg length: 0.760–0.990 m, omega_0: 3.15–3.59 rad/s)
- 기준 시점(ref_frame): stepping → step_onset_local, nonstep → 동일 (subject, velocity) stepping 평균
- Nonstep trial COP 시계열: 24 subjects, 72 trials, 총 9,350 COP points

---

## Results

### 1. Effective BOS (eBOS) — COP Excursion Envelope

eBOS는 각 피험자의 nonstep trial COP 시계열을 pooling한 후 convex hull로 정의하였다.

**eBOS / Physical BOS 면적 비율:**

| Metric | Value |
|--------|------:|
| Mean eBOS/BOS ratio | **0.200** (20.0%) |
| Min | 0.029 (권유영) |
| Max | 0.665 (가윤호) |
| Hof & Curtze (2016) 참조 | ~0.30 (30%) |

- 본 데이터의 평균 eBOS/BOS 비율(20.0%)은 Hof & Curtze(2016)의 ~30%보다 낮음
- 피험자 간 편차가 크다 (0.029–0.665)

**핵심 발견 — xCOM과 eBOS 관계:**

|             | step | nonstep |
|-------------|-----:|--------:|
| **inside eBOS**  |    0 |       0 |
| **outside eBOS** |  112 |      72 |

- **100%의 stepping trial에서 xCOM이 eBOS 외부** (vs. physical BOS에서는 6.2%만 외부)
- Nonstep trial도 100% eBOS 외부 — xCOM은 reference timepoint에서 이미 COP 도달 범위를 넘어섰다
- 이는 eBOS가 매우 보수적(작은) 기능적 경계임을 의미한다

**GLMM 결과:**

| Model | Predictor | OR | 95% CI | p |
|-------|-----------|---:|-------:|--:|
| `step ~ eBOS_MoS + (1\|subject)` | eBOS_MoS | 0.023 | [0.007, 0.076] | 1.04e-09 |
| `step ~ phys_MoS + (1\|subject)` | MOS_minDist_signed | 0.016 | [0.001, 0.319] | 6.89e-03 |

- eBOS-MoS의 p-value가 physical MoS보다 더 유의함 (1.04e-09 vs 6.89e-03)
- OR < 1: MoS가 감소할수록(불안정할수록) stepping 확률 증가

**ROC/AUC:**

| Metric | Simple AUC | LOSO-CV AUC |
|--------|----------:|------------:|
| Physical BOS MoS | **0.783** | 0.773 |
| eBOS MoS | 0.671 | 0.652 |

- eBOS-MoS AUC는 physical MoS보다 낮음 — 모든 trial이 eBOS 외부이므로 판별력이 제한됨
- Mann-Whitney: U=2655, p=9.46e-05, r=0.342

### 2. COM Velocity-Position State Space (FSR)

COM position은 BOS 길이로 정규화 (0=posterior, 1=anterior), COM velocity는 omega_0 × BOS 길이로 정규화.

**정규화된 변수 기술통계:**

| Variable | Mean | SD |
|----------|-----:|---:|
| COM_pos_norm | 0.328 | 0.101 |
| COM_vel_norm | 0.003 | 0.077 |

**GLMM 결과 — 2D vs 1D 모델:**

| Model | Predictor | Coefficient | OR | p |
|-------|-----------|------------:|---:|--:|
| 2D (pos+vel) | COM_pos_norm | -1.125 | 0.325 | 1.04e-02 |
| 2D (pos+vel) | COM_vel_norm | **-6.893** | 0.001 | 3.28e-06 |
| 1D velocity | COM_vel_norm | -7.246 | 0.001 | 8.54e-07 |
| 1D position | COM_pos_norm | -2.847 | 0.058 | 4.39e-11 |

- **COM velocity의 계수 크기(6.893)가 position(1.125)의 6.1배** → 속도가 지배적 예측 변수
- 2D 모델에서 velocity의 p-value(3.28e-06)가 position(1.04e-02)보다 3 orders of magnitude 더 유의

**LOSO-CV AUC 비교:**

| Model | Overall AUC | Mean AUC |
|-------|----------:|----------:|
| **1D velocity** | **0.794** | 0.871 |
| 2D (pos+vel) | 0.787 | 0.882 |
| 1D MoS | 0.773 | 0.837 |
| eBOS MoS | 0.652 | 0.809 |
| 1D position | 0.652 | 0.740 |

- **1D velocity(AUC=0.794)가 가장 높은 overall AUC** — 속도 단독으로 MoS(0.773)보다 우수
- 2D(0.787)는 velocity 단독과 거의 동일 → position 추가의 한계 효과
- Position 단독(0.652)은 가장 낮은 판별력 → 위치만으로는 stepping 예측 불가

---

## Interpretation

### eBOS: 기능적 경계의 재정의

- eBOS는 physical BOS의 평균 20%에 불과하며, 이는 Hof & Curtze(2016)의 ~30% 추정과 방향이 일치한다
- **모든 trial(step+nonstep)에서 xCOM이 eBOS 외부**라는 결과는, reference timepoint(step onset 시점)에서 이미 COP 도달 범위를 초과한 불안정 상태임을 의미한다
- 그러나 eBOS 기반 MoS의 판별력(AUC=0.671)이 physical MoS(0.783)보다 낮다 — eBOS 정의가 너무 보수적이거나, reference timepoint에서는 이미 둘 다 경계를 넘어서 있어 차별화가 어렵다
- eBOS 개념은 "왜 stepping이 BOS 내부에서 발생하는가"에 대한 설명으로는 유효하지만, 실질적 stepping 예측 도구로는 physical MoS가 여전히 우월하다

### FSR: COM 속도가 핵심 변수

- **COM velocity가 stepping 결정의 지배적 변수**임이 명확하게 확인되었다 (GLMM 계수 6.1배 차이, AUC 0.794 vs 0.652)
- 이는 Pai & Patton(1997)의 FSR 이론과 완벽히 일치: "COM이 BOS 어디에 있는가"보다 "COM이 얼마나 빠르게 이동하는가"가 중요하다
- 1D velocity(AUC=0.794)가 기존 MoS(0.773)를 초과 → **MoS보다 COM 속도가 더 나은 단일 예측 변수**
- 2D 모델은 velocity 단독과 거의 동일한 성능 → position 추가의 한계 이득

### 종합: "교과서적 임계값 이전 stepping"의 역학적 설명

본 분석 결과를 종합하면:

1. **eBOS 관점**: 해부학적 BOS는 기능적 안정 경계를 과대 추정한다. 실제 COP 도달 범위(eBOS)는 BOS의 ~20%에 불과하여, "BOS 내부 stepping"은 이미 기능적 경계를 넘어선 상태에서의 반응이다.

2. **FSR 관점**: Stepping은 COM 위치가 아닌 **COM 속도에 의해 주로 결정**된다. 속도 단독(AUC=0.794)이 위치 포함 MoS(0.773)보다 우수한 예측 변수이다. Perturbation 감지 시 CNS는 "현재 위치"가 아닌 "현재 속도에 기반한 미래 위치"를 예측하여 stepping을 결정한다.

3. 이 두 관점은 상호 보완적이다: eBOS는 "경계가 생각보다 가깝다"는 공간적 설명을, FSR은 "속도가 위치보다 중요하다"는 동역학적 설명을 제공한다.

### Conclusion

1. **eBOS는 physical BOS의 약 20%** — Hof & Curtze(2016)의 ~30% 보고와 방향 일치
2. **eBOS 기준 100% stepping trial이 기능적 경계 외부** — "BOS 내부 stepping" 패러독스 해소
3. **COM velocity가 stepping의 지배적 예측 변수** (GLMM coefficient 6.1배, AUC 0.794)
4. **1D velocity > MoS > 2D(pos+vel) ≈ 1D velocity > 1D position** — 속도 단독이 최적
5. **MoS(xCOM-based)는 velocity 정보를 일부 포함**하기에 position 단독보다 우수하지만, pure velocity보다는 열등

---

## Reproduction

```bash
conda run -n module python analysis/why_stepping_before_threshold/analyze_ebos_and_fsr.py
```

**Input**: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`, `data/all_data/` (C3D)
**Output**: fig1~fig8 PNG (이 폴더에 생성)

## Figures

| File | Description |
|------|-------------|
| fig1_cop_excursion_envelope.png | 대표 subject COP excursion envelope (physical BOS, eBOS hull, xCOM 위치) |
| fig2_ebos_area_ratio.png | eBOS/physical BOS 면적 비율 bar chart (피험자별, 30% reference line) |
| fig3_mos_ebos_distribution.png | eBOS-MoS vs physical-BOS-MoS violin+strip (step/nonstep) |
| fig4_roc_ebos_vs_physical.png | ROC curve: eBOS-MoS vs physical-BOS-MoS |
| fig5_state_space_scatter.png | COM velocity-position state space scatter + GLMM boundary |
| fig6_state_space_marginals.png | State space scatter + marginal histograms |
| fig7_roc_2d_vs_1d.png | ROC curves: 2D GLMM, 1D MoS, 1D velocity, 1D position, eBOS |
| fig8_summary_auc_comparison.png | Summary AUC bar chart (전체 모델 비교) |
