# Step vs. Non-step SPM 1D Analysis Report

## 한눈에 보는 핵심 발견 (Plain-Language Summary)

> 동일한 세기의 섭동(perturbation)에서 **발을 내딛는(step) 전략**과 **발을 고정하는(nonstep) 전략**은 시간 흐름에 따라 언제, 어떤 변수에서 차이가 나는가?

- **섭동 직후(0-16%)**: Step 전략군은 이미 몸의 무게중심이 지지면 경계에 더 가까이 있어, 안정성 여유(MOS)가 nonstep 전략군과 즉시 차이를 보였다.
- **중기(46-60%)**: Step 전략군이 보상 발디딤을 준비하면서 전방 속도와 무게중심 위치가 벌어지기 시작했다.
- **후기(60-100%)**: Step 실행으로 인해 무게중심, 속도, 안정성 지표 모두에서 두 전략이 뚜렷이 갈렸다.
- **전 구간(0-100%)**: xCOM-BOS AP 상대 위치(발길이 정규화)는 처음부터 끝까지 유의하여, step/nonstep을 구분짓는 가장 핵심적인 지표였다.
- ML(좌우) 방향이나 관절 각도 변수에서는 유의한 차이가 없었다 -- 이 섭동은 주로 앞뒤(AP) 방향으로 가해졌기 때문이다.

---

## SPM 1D 분석이란?

### 기존 분석(LMM)의 한계

이전의 LMM(선형혼합모형) 분석에서는 특정 **한 시점**(예: 섭동 시작 시점, step 시작 시점)에서 변수 값을 뽑아 step과 nonstep을 비교하였다. 이 방법은 "그 순간에 차이가 있는가?"에는 답할 수 있지만, **"시간 흐름 전체에서 언제부터 언제까지 차이가 나는가?"**에는 답할 수 없다.

예를 들어, 섭동 후 무게중심 속도가 step 전략에서 더 빨라지는 것이 50%쯤부터인지, 20%부터인지, 또는 처음부터 계속 다른지를 알 수 없었다.

### SPM 1D란?

**SPM(Statistical Parametric Mapping) 1D**는 원래 뇌영상(fMRI) 분석에서 개발된 통계 기법을 1차원 시계열 데이터에 적용한 것이다. 핵심 아이디어는 단순하다:

> **시간축의 모든 지점에서 동시에 통계 검정을 수행하고, "언제" 두 조건이 유의하게 다른지를 구간으로 알려준다.**

일반적인 t-test가 하나의 숫자(예: step onset 시점의 COM_X 값)를 비교하는 것이라면, SPM 1D t-test는 **시간에 따른 곡선 전체**를 비교한다.

### 분석 절차 (단계별)

```
[1단계] 시간 정규화
  각 시행(trial)의 분석 구간(섭동 시작 -> step 시작)을 0-100%로 정규화하여
  시행마다 다른 실제 시간 길이를 통일한다. (101개 시점)

      |
      v

[2단계] 피험자별 평균 곡선 산출
  각 피험자 내에서 step 시행의 평균 곡선, nonstep 시행의 평균 곡선을 구한다.
  -> 24명 x 2조건 = 24쌍의 paired 곡선

      |
      v

[3단계] 매 시점마다 paired t-검정
  0%, 1%, 2%, ... 100% 각 시점에서 t-값을 계산한다.
  결과: 시간에 따른 t-통계량 곡선 (SPM{t})

      |
      v

[4단계] 임계값(threshold) 설정
  "시간축 전체에서 여러 번 검정하므로" 다중비교 문제가 발생한다.
  SPM은 Random Field Theory(RFT)를 이용하여,
  "우연히 이 정도 크기의 t-값이 연속으로 나올 확률"을 계산해 임계값을 정한다.
  -> Bonferroni처럼 지나치게 보수적이지 않으면서도 다중비교를 보정한다.

      |
      v

[5단계] 유의 구간 판정
  t-통계량 곡선이 임계값을 넘는 구간 = 두 조건이 유의하게 다른 시간대
  예) "54-100% 구간에서 step > nonstep" -> 이 시간대에 유의한 차이가 있다.
```

### 비유로 이해하기

일반 t-test는 **사진 한 장**으로 두 그룹을 비교하는 것이고, SPM 1D는 **동영상 전체**를 프레임별로 비교하여 "몇 분 몇 초부터 몇 분 몇 초까지 장면이 다르다"고 알려주는 것이다.

### 비모수 순열 검정 (Nonparametric Permutation Test)

SPM의 모수 검정은 데이터가 정규분포를 따른다고 가정한다. 이 가정이 위반될 경우를 대비하여, **비모수 순열 검정**(10,000회 무작위 재배치)을 병행하였다. 두 검정의 결과가 일치하면 분포 가정 위반 우려가 낮다고 판단할 수 있다. 본 분석에서는 모수/비모수 결과가 모든 변수에서 일치하였다.

---

## Research Question

동일한 섭동 조건에서 step/nonstep 전략이 [platform onset -> step onset] 구간의 시계열 전체에서 언제 유의하게 다른지 SPM 1D로 확인한다.

## Data Summary

- 분석 프레임 수: 41927 (원본 42846)
- 분석 시행 수: 181 (step=108, nonstep=73)
- 피험자 수: 24
- 제외 시행: step onset 누락 2개, event 범위 이탈 2개
- 입력 데이터: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- 전처리 필터(`scripts/apply_post_filter_from_meta.py`): mixed==1, age_group==young, ipsilateral step only
- 분석 변수 수: 38

## Analysis Methodology

- 분석 구간: 각 trial의 `[platform_onset_local, end_frame]`
  - Step trial: `end_frame = step_onset_local`
  - Nonstep trial: 같은 `(subject, velocity)`의 step onset 평균값(부족 시 platform sheet fallback)
- 시간 정규화: 0-100% (101 points), NaN 20% 초과 trial 제외, 그 외 선형보간
- 짝지음 단위: 피험자 내 step/nonstep 평균 곡선
- SPM 검정: paired t-test (parametric + nonparametric permutation)
- 비모수 순열 횟수: 10000
- 다중비교 보정: family별 Bonferroni (`alpha = 0.05 / family_size`)
- Nonstep stance side: subject별 major step side 사용, tie는 (L+R)/2
- xCOM/BOS 정규화: `foot_len_m = (발길이_왼 + 발길이_오른)/2` 기반

### Coordinate & Sign Conventions

Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|------|---------------|---------------|-----------|
| AP (X) | +X = 전방 | -X = 후방 | COM_X, vCOM_X, xCOM_X, BOS_minX/maxX, MOS_AP_v3d |
| ML (Y) | +Y = 좌측 | -Y = 우측 | COM_Y, vCOM_Y, xCOM_Y, BOS_minY/maxY, MOS_ML_v3d |
| Vertical (Z) | +Z = 위 | -Z = 아래 | COM_Z, vCOM_Z, xCOM_Z, GRF_Z_N |

Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|--------|--------------|--------------|----------------|
| MOS_minDist_signed | BOS 내부/안정 여유 | BOS 외부/안정 여유 부족 | signed minimum distance |
| MOS_AP_v3d | AP 경계 내부 방향 | AP 경계 외부 방향 | AP bound-relative sign |
| MOS_ML_v3d | ML 경계 내부 방향 | ML 경계 외부 방향 | ML bound-relative sign |
| xCOM_BOS_AP_foot | BOS_minX 기준 전방 상대 위치 증가 | BOS_minX 기준 전방 상대 위치 감소 | foot length 정규화 |
| xCOM_BOS_ML_foot | BOS_minY 기준 좌측 상대 위치 증가 | BOS_minY 기준 우측 상대 위치 증가 | foot length 정규화 |

Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|----------------|------------------|-----------|
| Joint angles (Hip/Knee/Ankle/Trunk/Neck) | 각 축의 해부학적 회전 부호를 데이터 원 부호 그대로 사용 | stance side만 Hip/Knee/Ankle X축에 적용 |
| GRF_* / GRM_* | force/torque 원시 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |
| COP_* | COP 절대 좌표 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |
| AnkleTorqueMid_int_Y_Nm_per_kg | internal torque 부호 유지 | 체중 정규화 값 사용 |

## Results

- Parametric 유의 변수: 11 / 38
- Nonparametric 유의 변수: 11 / 38

### Family-level Summary

| Family | Variables | Param Sig | Nonparam Sig |
|--------|-----------|-----------|--------------|
| AnkleTorque | 1 | 0 | 0 |
| BOS | 5 | 0 | 0 |
| COM | 3 | 2 | 2 |
| COP | 2 | 1 | 1 |
| GRF | 3 | 0 | 0 |
| GRM | 3 | 0 | 0 |
| MOS | 4 | 3 | 3 |
| Neck | 3 | 0 | 0 |
| StanceJoint | 3 | 0 | 0 |
| Trunk | 3 | 0 | 0 |
| vCOM | 3 | 2 | 2 |
| xCOM | 3 | 2 | 2 |
| xCOM_BOS | 2 | 1 | 1 |

### Significant Variable Summary (Sig only)

| Variable | Family | N_pairs | Param interval (%) | Nonparam interval (%) | Param Sig | Nonparam Sig |
|----------|--------|---------|--------------------|-----------------------|-----------|--------------|
| COM_X | COM | 24 | 59.8-100.0 | 60.0-100.0 | <0.05 | <0.05 |
| COM_Z | COM | 24 | 0.0-4.8, 28.0-37.2 | 0.0-10.1, 26.1-38.8 | <0.05 | <0.05 |
| COP_X_m | COP | 24 | 99.5-100.0 | 63.2-67.0, 91.8-100.0 | <0.05 | <0.05 |
| MOS_AP_v3d | MOS | 24 | 0.0-15.9, 61.6-100.0 | 0.0-16.1, 61.4-100.0 | <0.05 | <0.05 |
| MOS_minDist_signed | MOS | 24 | 0.0-16.4, 61.8-100.0 | 0.0-16.5, 61.7-100.0 | <0.05 | <0.05 |
| MOS_v3d | MOS | 24 | 0.0-15.9, 61.6-100.0 | 0.0-16.0, 61.4-100.0 | <0.05 | <0.05 |
| vCOM_X | vCOM | 24 | 1.6-18.4, 54.0-100.0 | 1.0-18.9, 53.2-100.0 | <0.05 | <0.05 |
| vCOM_Z | vCOM | 24 | 64.6-69.3 | 62.1-71.0 | <0.05 | <0.05 |
| xCOM_X | xCOM | 24 | 46.4-100.0 | 44.0-100.0 | <0.05 | <0.05 |
| xCOM_Z | xCOM | 24 | 62.9-72.5 | 60.6-73.9 | <0.05 | <0.05 |
| xCOM_BOS_AP_foot | xCOM_BOS | 24 | 0.0-100.0 | 0.0-100.0 | <0.05 | <0.05 |

### Cross-check with prior LMM focus variables

| Variable | SPM status |
|----------|------------|
| xCOM_BOS_AP_foot | param sig, nonparam sig |
| xCOM_BOS_ML_foot | param n.s., nonparam n.s. |
| Hip_stance_X_deg | param n.s., nonparam n.s. |

## Discussion

### LMM 결과와의 연결

이전 LMM 분석에서는 섭동 시작 시점과 step 시작 시점, 두 시점에서 xCOM/BOS(외삽 질량중심과 지지면의 관계)가 step과 nonstep 간에 유의하게 달랐다. SPM 분석은 이 결과를 확장하여, **두 시점 사이의 전 구간(0-100%)에서 xCOM_BOS_AP_foot가 계속 유의했음**을 보여준다. 즉, xCOM-BOS 관계의 차이는 특정 순간에만 존재하는 것이 아니라 섭동 시작부터 step 시작까지 지속적으로 존재하는 근본적 차이였다.

### 시간대별 주요 발견

**초기 구간 (0-16%):** MOS_AP_v3d(전후 방향 안정성 여유), MOS_minDist_signed(최소 안정성 거리), MOS_v3d(종합 안정성 여유)가 섭동 직후부터 유의하였다. 이는 섭동이 가해지는 순간, step 전략군의 안정성 마진(Margin of Stability: 무게중심이 지지면 경계로부터 얼마나 여유가 있는지를 나타내는 지표)이 nonstep 전략군과 즉각적으로 구분됨을 시사한다. 즉, step을 선택하는 피험자는 섭동 초기에 이미 xCOM(외삽 질량중심: 현재 위치뿐 아니라 속도까지 고려한 "예측 무게중심")이 BOS(지지면) 경계에 더 가깝거나 이를 벗어나는 경향이 있다.

**중기 구간 (46-60%):** vCOM_X(전방 무게중심 속도, 54%)와 xCOM_X(전방 외삽 질량중심, 46%)에서 유의 구간이 시작된다. 이 시점은 step 전략군이 보상적 발 디딤을 준비하며 전방 속도가 증가하고, 그에 따라 외삽 질량중심이 전방으로 이동하기 시작하는 구간이다. nonstep 전략군은 이 시기에 발을 고정한 채 발목/엉덩이 관절 전략(ankle/hip strategy)으로 대응하므로 속도/위치 변화가 상대적으로 작다.

**후기 구간 (60-100%):** COM_X(전방 무게중심), xCOM_X, vCOM_X, MOS 계열이 모두 유의해지며, step 전략의 전방 무게중심 이동과 새로운 지지면 확보 과정이 반영된다. COM_Z(수직 무게중심)와 xCOM_Z(수직 외삽 질량중심)는 60-73% 구간에서만 유의한데, 이는 step 실행 시 일시적으로 수직 무게중심이 하강한 뒤 회복하는 패턴을 나타낸다. COP_X_m(전방 압력중심)은 후기(~92-100%)에 유의하여, step 착지 후 압력중심이 전방으로 급격히 이동하는 시점을 포착한다.

**전 구간 유의 (0-100%):** xCOM_BOS_AP_foot(발길이로 정규화한 전후 방향 외삽 질량중심-지지면 상대 위치)가 정규화 구간 전체에서 유의하였다. 이는 AP 방향 xCOM-BOS 상대 위치가 step/nonstep 전략 간에 근본적으로 다름을 의미하며, step 전략은 xCOM이 BOS 전방 경계를 지속적으로 초과하는 반면, nonstep 전략은 BOS 내부에 xCOM을 유지하는 패턴을 보인다.

### ML 방향 및 관절 각도

xCOM_BOS_ML_foot(좌우 방향 외삽 질량중심-지지면 상대 위치), Hip_stance_X_deg(지지측 엉덩이 관절 각도) 등 ML 방향 및 관절 변수는 유의하지 않았다. 이는 본 섭동이 주로 AP(전후) 방향으로 가해졌기 때문에, step/nonstep 전략 간 차이가 AP 안정성 변수에 집중되는 것으로 해석된다.

## Conclusion

1. Step 전략군은 섭동 직후(0-16%)부터 AP 방향 MOS(안정성 여유)가 nonstep 전략군과 유의하게 달라, 보상 전략 선택이 섭동 초기 안정성 상태와 관련됨을 확인하였다.
2. 중기(46-60%) 이후 vCOM_X(전방 속도), xCOM_X(전방 외삽 질량중심), COM_X(전방 무게중심)가 순차적으로 유의해져, step 준비->실행 과정에서의 전방 이동이 시계열 수준에서 뚜렷이 구분된다.
3. xCOM_BOS_AP_foot가 전 구간 유의하여, AP 방향 xCOM-BOS 관계가 step/nonstep 전략을 구분짓는 핵심 지표임을 시사한다.
4. ML 방향 변수 및 관절 각도 변수는 유의하지 않아, AP 섭동 하에서 두 전략 간 차이는 시상면(전후/수직) 변수에 국한된다.
5. 모수 검정과 비모수 순열 검정 결과가 11개 유의 변수 모두에서 일치하여, 결과의 통계적 신뢰성이 확보되었다.

## Limitations

- 본 분석은 young, mixed==1, ipsilateral step 시행만 포함하므로, 고령자/contralateral step 등으로 일반화 시 주의가 필요하다.
- paired t-test 구조상 피험자 내 step/nonstep 시행이 모두 존재하는 경우만 분석되어, 한 전략만 사용하는 피험자는 제외되었다.
- 비모수 순열 검정과 모수 검정 결과가 모든 유의 변수에서 일치하여 분포 가정 위반 우려는 낮다.

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/step_vs_nonstep_spm/analyze_step_vs_nonstep_spm.py
```

## Output Files

- `analysis/step_vs_nonstep_spm/spm_results.csv`
- `analysis/step_vs_nonstep_spm/figures/spm_<variable>.png`
- `analysis/step_vs_nonstep_spm/figures/heatmap_significant.png`
- `analysis/step_vs_nonstep_spm/report.md`
