# FSR-Only Analysis: Window Mean Reanalysis

## Research Question

**"COP 기반 경계를 제외했을 때, window mean 방식의 COM state 변수와 MoS 중 어떤 지표가 step/nonstep를 더 잘 예측하는가?"**

이번 리포트는 단일 시점을 쓰지 않고, 각 trial의 시간 구간 평균값으로 다시 계산했다.

## Data Summary

- Trial 수: **184** (step 112, nonstep 72)
- Subject 수: **24**
- 고정 시간창 규칙:
  - 시작: `platform_onset_local`
  - 종료: step trial은 `step_onset_local`
  - 종료(비스텝): subject 평균 `step_onset_local`
- trial당 사용 프레임 수: 평균 **52.483696**, 최소 **16**, 최대 **141**
- 분석 변수: `COM_pos_norm`, `COM_vel_norm`, `MOS_minDist_signed`

> [GPT Comment]
> - Verdict: Exact
> - Basis: `analyze_fsr_only.py` 실행 로그의 trial/frame 통계와 동일하다.
> - Alternative Applied: 기존 단일 시점 추출을 제거하고 window mean 집계를 적용했다.
> - Actual Result (Quant): N=184, step=112, nonstep=72, frames/trial=52.483696 (16~141).
> - Action: 이후 표/해석은 이 window 정의를 기준으로만 읽는다.

---

## Results

### 1. FSR 변수 생성과 기술통계

정규화 식:
- `COM_pos_norm = (COM_X - BOS_minX) / (BOS_maxX - BOS_minX)`
- `COM_vel_norm = vCOM_X / (omega_0 * (BOS_maxX - BOS_minX))`

위 식을 프레임마다 계산한 뒤, 각 trial window에서 평균했다.

| Variable | Mean | SD |
|----------|-----:|---:|
| COM_pos_norm | 0.410634 | 0.104880 |
| COM_vel_norm | 0.032499 | 0.037230 |
| MOS_minDist_signed | 0.064069 | 0.014032 |

> [GPT Comment]
> - Verdict: Exact
> - Basis: `/tmp/why_step_after_exact.txt`의 POS/VEL/MOS 통계와 표 값이 일치한다.
> - Alternative Applied: trial 대표값을 단일 프레임이 아니라 window 평균값으로 바꿨다.
> - Actual Result (Quant): COM_pos_norm 0.410634±0.104880, COM_vel_norm 0.032499±0.037230.
> - Action: GLMM/AUC 해석은 이 새 분포를 전제로 진행한다.

### 2. GLMM (2D vs 1D)

| Model | Predictor | Coefficient | OR | p |
|-------|-----------|------------:|---:|--:|
| 2D (pos+vel) | COM_pos_norm | -2.767431 | 0.062823 | 5.773160e-15 |
| 2D (pos+vel) | COM_vel_norm | -2.415731 | 0.089302 | 1.500722e-01 |
| 1D velocity | COM_vel_norm | -3.228526 | 0.039616 | 5.397432e-02 |
| 1D position | COM_pos_norm | -3.065387 | 0.046636 | 0.000000e+00 |
| 1D MoS | MOS_minDist_signed | -0.894415 | 0.408847 | 5.549248e-01 |

간단 해석:
- 2D 모델에서는 `COM_pos_norm`이 유의하고(`p < 0.001`), `COM_vel_norm`은 유의하지 않았다(`p=0.150072`).
- 1D 모델에서는 position이 velocity보다 더 안정적으로 유의했다.

> [GPT Comment]
> - Verdict: Exact
> - Basis: 실행 로그 GLMM 계수/OR/p를 그대로 표에 반영했다.
> - Alternative Applied: 기존 해석의 "velocity 우위"를 유지하지 않고, 재분석 결과에 맞춰 문장을 수정했다.
> - Actual Result (Quant): 2D에서 COM_pos_norm p=5.77e-15, COM_vel_norm p=1.50e-01.
> - Action: 계수의 절대값보다 교차검증 AUC와 함께 해석한다.

### 3. LOSO-CV AUC 비교

| Model | Overall AUC | Mean AUC | 95% CI (fold AUC) |
|-------|------------:|---------:|------------------:|
| 1D velocity | 0.647445 | 0.838298 | [0.371667, 1.000000] |
| 1D position | 0.642113 | 0.808218 | [0.415000, 1.000000] |
| 2D (pos+vel) | 0.635541 | 0.849719 | [0.457500, 1.000000] |
| 1D MoS | 0.572049 | 0.673665 | [0.047917, 1.000000] |

핵심 관찰:
- 최고 Overall AUC는 **1D velocity (0.647445)**였다.
- 하지만 1D velocity, 1D position, 2D 사이 차이는 작다(최대 차이 0.011904).
- 1D MoS는 가장 낮았다(0.572049).

> [GPT Comment]
> - Verdict: Exact
> - Basis: AUC는 `/tmp/why_step_after_exact.txt` 값과 6자리까지 동일하다.
> - Alternative Applied: 기존 단일 시점 기준 순위를 폐기하고 window mean 기준 순위로 교체했다.
> - Actual Result (Quant): 1D velocity 0.647445 > 1D position 0.642113 > 2D 0.635541 > 1D MoS 0.572049.
> - Action: 실제 적용에서는 velocity 단독과 position 단독을 함께 후보로 검토한다.

---

## Interpretation

### A. 이번 재분석에서 확실히 말할 수 있는 점

1. 단일 시점 대신 구간 평균으로 바꾸면, 모델 성능 순위가 바뀔 수 있다.
2. 이번 데이터에서는 velocity가 가장 높지만, position과 거의 비슷하다.
3. MoS 단독 모델은 다른 모델보다 성능이 낮다.

> [GPT Comment]
> - Verdict: Exact
> - Basis: LOSO AUC 순위와 간격(0.647445 vs 0.642113 vs 0.635541)이 이를 뒷받침한다.
> - Alternative Applied: 이전 결론의 강한 우열 표현을 약화하고 실제 간격 중심으로 해석했다.
> - Actual Result (Quant): 상위 3개 모델 AUC 간 최대 차이=0.011904.
> - Action: 후속 보고에서는 "큰 차이" 대신 "작은 차이"라고 표현한다.

### B. 해석할 때 주의할 점

1. 이 분석은 FSR 개념(위치+속도)을 활용한 예측 모델이다.
2. Pai & Patton 원문 경계(recoverable region)를 그대로 계산한 것은 아니다.
3. LOSO fold 수가 적어서 CI 범위가 넓게 나온다.

> [GPT Comment]
> - Verdict: Partial
> - Basis: 방법론 방향은 맞지만, 원문의 동역학 경계 재현과는 다르다.
> - Alternative Applied: "방법론 동일" 대신 "개념 정합 + 예측모델"로 문구를 낮췄다.
> - Actual Result (Quant): fold AUC 95% CI가 넓음 (예: 1D velocity [0.371667, 1.000000]).
> - Action: 논문 대조 문구는 과장 없이 제한사항을 함께 적는다.

## Conclusion

1. 이 리포트는 window mean 재분석 결과로 완전히 교체되었다.
2. Overall AUC는 `1D velocity (0.647445)`가 가장 높았지만, `1D position (0.642113)`, `2D (0.635541)`와 큰 차이는 아니다.
3. `1D MoS (0.572049)`는 이번 데이터에서 가장 낮은 예측 성능을 보였다.
4. 따라서 이번 데이터 기준 실무 결론은 "velocity만 무조건 우위"가 아니라, **velocity/position 모두 주요 후보**로 보는 것이 맞다.

> [GPT Comment]
> - Verdict: Exact
> - Basis: 결론 문장이 GLMM/AUC 실제 출력값과 직접 대응한다.
> - Alternative Applied: 기존 velocity 강한 우위 결론을 재분석 수치 기반으로 수정했다.
> - Actual Result (Quant): 최고 AUC 0.647445, 차상위 0.642113, 격차 0.005332.
> - Action: 다음 단계는 다른 window 정의(예: 0~200ms) 민감도 분석으로 일반화 가능성을 확인한다.

---

## Reproduction

```bash
conda run -n module python analysis/why_stepping_before_threshold/analyze_fsr_only.py
```

입력 파일:
- `output/all_trials_timeseries.csv`
- `data/perturb_inform.xlsm`

출력 파일:
- `analysis/why_stepping_before_threshold/fig1_state_space_scatter.png`
- `analysis/why_stepping_before_threshold/fig2_state_space_marginals.png`
- `analysis/why_stepping_before_threshold/fig3_roc_2d_vs_1d.png`
- `analysis/why_stepping_before_threshold/fig4_summary_auc_comparison.png`

## Figures

| File | Description |
|------|-------------|
| fig1_state_space_scatter.png | window mean COM position-velocity 산점도 + GLMM 경계 |
| fig2_state_space_marginals.png | state space 산점도 + 주변분포 |
| fig3_roc_2d_vs_1d.png | 4개 모델 ROC 곡선 비교 |
| fig4_summary_auc_comparison.png | 4개 모델 AUC bar 비교 |
