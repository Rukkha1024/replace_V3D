# FSR-Only Analysis: Stepping Before the Textbook Threshold

## Research Question

**"COP 기반 경계 비교를 제외한 조건에서, COM velocity-position state 변수와 MoS baseline 중 무엇이 stepping을 더 잘 설명/예측하는가?"**

본 분석은 실험실 장비 제약( COP 좌표계와 COM/xCOM 좌표계 직접 비교 불가 )을 반영하여,
COP 기반 기능적 경계 분석은 전부 제외하고 FSR + MoS baseline 비교만 수행했다.

## Data Summary

- **184 trials** (112 step, 72 nonstep)
- **24 subjects** (leg length: 0.760–0.990 m, omega_0: 3.15–3.59 rad/s)
- 기준 시점(`ref_frame`):
  - stepping trial → `step_onset_local`
  - nonstep trial → 동일 `(subject, velocity)`의 stepping 평균 onset
- 사용 변수: `COM_X`, `vCOM_X`, `BOS_minX`, `BOS_maxX`, `MOS_minDist_signed`

> [GPT Comment]
> - Verdict: Exact
> - Basis: `analyze_fsr_only.py`의 입력/스냅샷 컬럼 정의와 실행 로그(184 trials, 24 subjects)가 일치함.
> - Action: 데이터 요약은 유지하되, `ref_frame` 정의가 해석에 미치는 영향은 해석 파트에서 명시.

---

## Results

### 1. FSR 변수 생성 및 기술통계

정규화 정의:
- `COM_pos_norm = (COM_X - BOS_minX) / (BOS_maxX - BOS_minX)`
- `COM_vel_norm = vCOM_X / (omega_0 * (BOS_maxX - BOS_minX))`

기술통계(유효 184 trials):

| Variable | Mean | SD |
|----------|-----:|---:|
| COM_pos_norm | 0.328 | 0.101 |
| COM_vel_norm | 0.003 | 0.077 |

> [GPT Comment]
> - Verdict: Exact
> - Basis: 변수식은 `analyze_fsr_only.py` 구현과 동일하며, 평균/표준편차는 실행 로그와 일치.
> - Action: 본 섹션 수치가 변경되면 downstream GLMM/AUC 해석을 함께 갱신.

### 2. GLMM (2D vs 1D)

| Model | Predictor | Coefficient | OR | p |
|-------|-----------|------------:|---:|--:|
| 2D (pos+vel) | COM_pos_norm | -1.125 | 0.325 | 1.04e-02 |
| 2D (pos+vel) | COM_vel_norm | **-6.893** | 0.001 | 3.27e-06 |
| 1D velocity | COM_vel_norm | -7.246 | 0.001 | 8.54e-07 |
| 1D position | COM_pos_norm | -2.847 | 0.058 | 4.39e-11 |
| 1D MoS | MOS_minDist_signed | -4.160 | 0.016 | 6.89e-03 |

해석 포인트:
- 2D 모델에서 `COM_vel_norm` 절대계수가 `COM_pos_norm`보다 큼.
- 1D velocity 모델은 유의하며, 1D position 단독보다 예측력이 높다(아래 AUC 참조).

> [GPT Comment]
> - Verdict: Partial
> - Basis: 속도 우세 방향성은 데이터와 일치하나, 계수 크기 비교만으로 "지배적" 결론을 단정하기에는 스케일/모형 가정 영향이 있음.
> - Action: 최종 결론은 계수비보다 LOSO AUC 우선으로 기술.

### 3. LOSO-CV AUC 비교

| Model | Overall AUC | Mean AUC |
|-------|------------:|---------:|
| **1D velocity** | **0.794** | 0.871 |
| 2D (pos+vel) | 0.787 | 0.882 |
| 1D MoS | 0.773 | 0.837 |
| 1D position | 0.652 | 0.740 |

핵심 관찰:
- Overall AUC 기준 최상위는 **1D velocity (0.794)**.
- 2D는 1D velocity와 성능이 매우 가깝고(0.787 vs 0.794), 위치 단독의 한계가 크다.
- MoS baseline은 position 단독보다 우수하지만 velocity 단독에는 뒤처진다.

> [GPT Comment]
> - Verdict: Exact
> - Basis: 표 수치는 실행 로그와 동일하며, 모델 범위도 FSR+MoS 4개로 제한되어 있음.
> - Action: 본 문서에서는 `overall_auc`를 1차 기준으로 고정하고 `mean_auc`는 보조지표로만 사용.

---

## Interpretation

### A. 본 분석에서 말할 수 있는 것

1. COP 기반 경계 없이도, step/nonstep 구분에서 속도 정보의 기여가 크다는 경향이 재현된다.
2. MoS는 여전히 유의한 baseline이지만 velocity 단독 예측력보다 낮다.
3. 위치 단독 모델은 성능이 가장 낮아, stepping 설명에 속도 성분이 필수적임을 시사한다.

> [GPT Comment]
> - Verdict: Exact
> - Basis: GLMM/LOSO 모두 동일 방향을 지지하며, COP 불일치 제약과 독립적으로 성립하는 결과임.
> - Action: 실무 결론은 "속도 포함 모델" 우선으로 유지.

### B. 주의해서 해석할 것 (제한사항)

1. 본 구현은 Pai & Patton(1997) FSR의 **개념적 방향**(position+velocity 중요성)은 따르지만,
   원 논문의 동역학 기반 recoverable-region 경계 계산을 직접 재현한 것은 아니다.
2. nonstep `ref_frame`를 stepping onset 평균으로 정하는 설계는 비교 일관성은 높이지만,
   예측 시나리오로 일반화할 때는 보수적으로 해석해야 한다.
3. GLMM은 Bayesian VB 적합값을 기반으로 요약했으므로, p-value 해석은 보조적 증거로 보는 것이 안전하다.

> [GPT Comment]
> - Verdict: Partial
> - Basis: 방법론 제약(구현 모델 vs 이론 모델 차이, ref_frame 정의, 통계 요약 방식)이 존재함.
> - Action: 외부 발표 시 "방법론 동일" 대신 "개념 정합 + 예측모델 기반 검증"으로 표현.

### Conclusion

1. **COP 기반 경계 분석은 좌표계 제약으로 제외**했으며, 본 리포트는 FSR+MoS 결과만 보고한다.
2. **1D velocity가 최고 Overall AUC(0.794)**로, stepping 예측에서 가장 강한 단일 지표였다.
3. **2D(pos+vel)는 1D velocity와 유사한 성능(0.787)**을 보였고, position 단독은 가장 낮았다(0.652).
4. **MoS baseline(0.773)은 유의미하지만 velocity 단독보다는 낮다.**

> [GPT Comment]
> - Verdict: Exact
> - Basis: 결론 문장이 현재 코드 출력(모델/수치/제약)과 직접 대응됨.
> - Action: 후속 작업에서는 ref_frame 대안(예: 고정 시간창) 민감도 분석을 추가 권장.

---

## Reproduction

```bash
conda run -n module python analysis/why_stepping_before_threshold/analyze_fsr_only.py
```

**Input**: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`  
**Output**: fig1–fig4 PNG (이 폴더에 생성)

## Figures

| File | Description |
|------|-------------|
| fig1_state_space_scatter.png | COM position-velocity 산점도 + GLMM decision boundary |
| fig2_state_space_marginals.png | State space 산점도 + 주변분포 |
| fig3_roc_2d_vs_1d.png | ROC curves (2D, 1D velocity, 1D position, 1D MoS) |
| fig4_summary_auc_comparison.png | AUC 비교 bar chart |
