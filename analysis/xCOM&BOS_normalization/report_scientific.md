# xCOM/BOS 정규화 지표를 이용한 Step 전략 분리 검증: 선행연구 공식의 통합 적용과 혼합효과모형 분석

## Abstract
본 보고서는 `methods_list.md`에 정리된 xCOM/BOS 정규화 계열 공식을 현재 섭동 균형 데이터셋에 적용하여, step 전략과 nonstep 전략의 차이를 선형혼합효과모형(linear mixed-effects model, LMM)으로 검증할 수 있는지 평가하였다. 분석 대상은 총 24명 피험자의 125개 trial(53 step, 72 nonstep)이며, `output/all_trials_timeseries.csv`와 `data/perturb_inform.xlsm`를 통합하여 구성하였다. 선행연구의 공식을 현재 데이터 구조에 맞게 재정의하여 세 개의 종속변수(DV1, DV2, DV3)를 구성하고, 각 변수를 `platform onset`과 `step onset` 두 이벤트 시점에서 추출하였다. 모든 모델은 `DV ~ step_TF + (1|subject)` 형태로 적합하였고, 여섯 개 주효과 검정에 대해 Benjamini-Hochberg FDR 보정을 적용하였다. 그 결과, 여섯 개 중 다섯 개 변수에서 step 여부의 유의한 주효과가 확인되었으며, 특히 Hof 기반 위치 지표(DV1)와 COM 위치 기반 지표(DV2)는 두 이벤트 시점 모두에서 전략 구분력을 보였다. 반면 속도 정규화 지표(DV3)는 `step onset`에서만 유의하여, 속도항 기반 분리는 이벤트 시점 의존성이 상대적으로 큰 것으로 해석된다.

## Introduction
외부 섭동에 대한 자세 반응에서 stepping 전략의 선택은 안정성 회복 기전과 직결되며, 이 전략 차이를 정량적으로 설명할 수 있는 지표의 타당성은 임상 및 실험 생체역학 모두에서 핵심 과제이다. 선행연구는 Hof 계열의 extrapolated center of mass(xCOM)와 base of support(BOS) 관계를 이용해 안정성을 해석해 왔고, 특히 발길이 또는 신체 치수 기반 정규화를 통해 피험자 간 스케일 차이를 줄이는 접근을 제안해 왔다. Van Wouwe et al. (2021)은 onset 초기 구간 안정성 해석에서 xCOM/BOS 지표의 설명력을 제시했으며, Salot et al. (2016)과 Joshi et al. (2018)은 위치항과 속도항을 분리한 정규화 지표의 해석 가치를 보고하였다.

현재 데이터셋은 선행연구와 이벤트 정의 및 원자료 구조가 완전히 동일하지 않기 때문에, 동일 공식을 기계적으로 복제하기보다 공통 수학적 의미를 보존하는 방식의 방법론적 적응이 필요하다. 본 분석의 목적은 선행연구 공식을 현재 파이프라인에 통합하여 step 대 nonstep 전략 차이를 통계적으로 검증하고, 어떤 지표가 이벤트별로 더 안정적인 구분력을 보이는지 확인하는 데 있다.

## Methods
분석에는 총 125개 trial과 24명 피험자 데이터가 사용되었고, trial 단위 전략 라벨은 step 53개, nonstep 72개로 구성되었다. 시간축 데이터는 `all_trials_timeseries.csv`에서, 피험자 신체치수와 trial 메타정보는 `perturb_inform.xlsm`의 `platform` 및 `meta` 시트에서 불러왔다. 발길이는 좌우 평균을 mm에서 m로 변환하여 사용하였고, 신장은 cm에서 m로 환산하였다.

종속변수는 선행연구 공식을 바탕으로 세 가지로 정의하였다. 첫째, Hof 기반 지표인 DV1은 `DV1 = (xCOM_hof - BOS_rear) / foot_length`로 계산하여 xCOM의 후방 BOS 대비 상대 위치를 발길이로 정규화하였다. 둘째, 위치항 분리 지표인 DV2는 `DV2 = (COM_X - BOS_rear) / foot_length`로 계산하여 COM 절대 위치 기여를 별도로 평가하였다. 셋째, 속도항 지표인 DV3는 `DV3 = (vCOM_X - vBOS_rear) / sqrt(g*height)`로 정의하였고, `vBOS_rear`는 시계열 BOS 후방 경계의 100 Hz 차분으로 근사하였다. 이때 AP 축 부호는 전방을 양(+)으로, 후방을 음(-)으로 해석하였다.

이벤트는 사용자 확정 조건에 따라 `platform onset`과 `step onset` 두 가지로 제한하였다. `step onset`이 결측인 step trial은 동일한 `(subject, velocity)` 그룹 평균으로 보완하였고, 잔여 결측은 prefilter 단계의 platform 기반 subject-velocity 평균으로 보정하였다. 각 DV와 이벤트 조합에 대해 하나의 LMM을 독립적으로 적합하였으며 모델식은 `DV ~ step_TF + (1|subject)`이다. 추정은 REML로 수행했고, 핵심 검정량은 고정효과 `step_TFstep` 계수의 유의성으로 정의하였다. 다중검정은 여섯 개 주효과 전체를 하나의 family로 간주하여 BH-FDR 보정을 적용하였다.

## Results
주효과 분석에서 여섯 개 지표 중 다섯 개가 FDR 보정 후 유의하였다. DV1의 경우 `platform onset`에서 step군 평균은 0.5091, nonstep군 평균은 0.5715였고, 추정 계수는 -0.0440으로 유의하였다. 같은 지표의 `step onset`에서도 step군 평균 0.2354, nonstep군 평균 0.3232, 추정 계수 -0.0893으로 유의성이 유지되었다. DV2는 `platform onset`에서 step 0.4976, nonstep 0.5525, 계수 -0.0395로 유의하였고, `step onset`에서는 step 0.2648, nonstep 0.2872, 계수 -0.0225로 유의하였다.

DV3는 이벤트 의존적 패턴을 보였다. `platform onset`에서는 step -0.0245, nonstep -0.0291, 추정 계수 -0.0001로 유의하지 않았으나, `step onset`에서는 step 0.0684, nonstep -0.0039, 추정 계수 0.0715로 뚜렷한 유의성이 확인되었다. 전체적으로 보면 위치 기반 또는 xCOM-위치 결합 지표는 두 이벤트에서 일관된 전략 분리 능력을 보인 반면, 속도 정규화 지표는 step 실행 직전 또는 직후에 가까운 이벤트에서만 구분력이 증가하는 경향을 나타냈다.

## Discussion
본 결과는 선행연구에서 제시된 xCOM/BOS 정규화 접근이 현재 데이터 구조에서도 유효하게 작동할 수 있음을 보여준다. 특히 DV1이 두 이벤트에서 모두 강한 유의성을 보인 점은 Van Wouwe 계열의 안정성 해석 틀과 정합적이며, DV2의 유의성은 Joshi 계열 위치항 지표가 독립적인 해석 가치를 갖는다는 보고와 일치한다. 즉, step 전략 선택은 BOS 대비 COM 또는 xCOM의 상대 위치 이동량과 밀접하게 연관되어 있으며, 이는 발길이 정규화를 통해 피험자 간 체격 차이를 보정한 뒤에도 유지되는 효과로 볼 수 있다.

반면 DV3가 `platform onset`에서 유의하지 않고 `step onset`에서만 유의한 결과는 속도항의 민감도가 시간상태에 크게 의존함을 시사한다. 섭동 직후 초기 구간에서는 속도항 기반 상대 안정성 신호가 충분히 분리되지 않다가, 실제 stepping 반응이 조직되는 시점 근처에서 집단 간 차이가 확대될 가능성이 있다. 이러한 패턴은 위치항과 속도항을 분리 해석해야 한다는 선행연구의 제안을 현재 데이터에서도 재확인한 것으로 해석할 수 있다.

## Limitations
본 분석은 이벤트를 `platform onset`과 `step onset` 두 시점으로 제한했기 때문에, 선행연구에서 흔히 사용하는 lift-off, touchdown, 300 ms 시점 기반 비교를 직접 재현하지 못했다. 또한 step onset 결측 trial에 대해 subject-velocity 평균 보완을 사용했으므로, 개별 trial의 실제 반응 시점 변동성이 일부 평활화되었을 수 있다. 마지막으로 본 보고서는 xCOM/BOS 계열 지표에 초점을 두었기 때문에, 관절각, 지면반력, 토크 변수를 포함한 통합 다변량 모형 검증까지는 포함하지 않았다.

## Conclusion
`methods_list.md` 기반의 xCOM/BOS 정규화 공식을 현재 파이프라인에 통합했을 때, step과 nonstep 전략 차이는 LMM에서 통계적으로 검출 가능했다. 여섯 개 지표 중 다섯 개에서 유의한 주효과가 관찰되었고, 특히 DV1과 DV2는 두 이벤트에서 일관된 분리력을 보였다. 따라서 본 데이터셋에서 전략 구분의 핵심 신호는 BOS 대비 위치 기반 정규화 지표에 더 안정적으로 반영되며, 속도항 기반 지표는 이벤트 시점 선택에 따라 해석력이 크게 달라진다.

## Reproducibility
동일 결과 재현은 아래 명령으로 수행할 수 있다. 첫 번째 명령은 설정과 입력 연결 상태를 점검하는 dry-run이며, 두 번째 명령은 실제 분석과 시각화 산출을 수행한다.

```bash
conda run --no-capture-output -n module python analysis/xCOM&BOS_normalization/analyze_xcom_bos_normalization_lmm.py --dry-run
conda run --no-capture-output -n module python analysis/xCOM&BOS_normalization/analyze_xcom_bos_normalization_lmm.py
```

분석 입력은 `output/all_trials_timeseries.csv`와 `data/perturb_inform.xlsm`이며, 시각화 산출은 `fig1_main_effect_forest.png`와 `fig3_violin_significant.png`이다.
