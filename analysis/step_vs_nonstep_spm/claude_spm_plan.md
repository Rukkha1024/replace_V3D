# SPM 1D 분석: Step vs. Non-step 시계열 비교

## Context
현재 LMM 분석은 시행별 **집계값**(peak, mean, range 등)으로 step/nonstep을 비교한다.
SPM(Statistical Parametric Mapping) 1D 분석은 **시계열 전체 구간**에서 두 그룹이 **언제** 차이가 나는지를 연속적으로 검정한다.
분석 구간: [platform onset → step onset], 시간 정규화 0-100% (101 points).

## 통계 설계
- **Paired SPM t-test** (피험자 내 step/nonstep 평균 곡선 쌍)
  - 각 피험자별 step 시행 평균 곡선, nonstep 시행 평균 곡선을 1쌍으로 구성
  - step만 있거나 nonstep만 있는 피험자는 SPM 주분석에서 제외
  - 변수별 QC 이후 유효 pair 수(`N_pairs`)를 기록하고 결과에 함께 보고
  - `spm1d.stats.ttest_paired(Y_step, Y_nonstep)` (parametric, RFT 보정)
  - `spm1d.stats.nonparam.ttest_paired` (비모수 순열검정, 10000회) → 강건성 확인
- **다중비교 보정**: 변수 패밀리별 Bonferroni (보수적 기준)
- Nonstep의 end_frame: 동일 (subject, velocity) step 시행의 step_onset 평균값 (기존 LMM 로직 재사용)

## 기존 분석 로직 적용 (반드시 준수)

### 1. Stance-side 관절각도 변환
기존 LMM의 `add_stance_joint_x_columns()` 로직을 동일 적용:
- `state == "step_r"` → **왼쪽** 관절각 사용 (왼쪽이 stance leg)
- `state == "step_l"` → **오른쪽** 관절각 사용
- `nonstep` → `major_step_side` 기준 (피험자별 step 방향 다수결)
- `tie` → `(L + R) / 2` 평균

적용 관절: Hip, Knee, Ankle의 X축 성분 → 3개 stance 변수 생성
Trunk, Neck은 L/R 구분 없으므로 그대로 사용

### 2. xCOM/BOS 정규화 변수 추가
기존 LMM(`analyze_com_xcom_screening_lmm.py`)의 발 길이(foot length) 기반 정규화 로직을 시계열로 적용:
- `data/perturb_inform.xlsm`의 `meta` 시트에서 `발길이_왼`, `발길이_오른` 행을 읽어 `foot_len_m`를 계산한 뒤 피험자 기준 병합해야 함.
```python
xCOM_BOS_AP_foot = (xCOM_X - BOS_minX) / foot_len_m   # 전후(AP) 방향 발 길이 정규화
xCOM_BOS_ML_foot = (xCOM_Y - BOS_minY) / foot_len_m   # 좌우(ML) 방향 발 길이 정규화
```

### 3. End frame 계산
`_compute_end_frames()` 로직 재사용:
- Step: `step_onset_local` (실제 step onset)
- Nonstep: 동일 (subject, velocity) step 시행의 `step_onset_local` 평균
- Fallback: `perturb_inform.xlsm` platform sheet에서 재구성

### 4. 필터링 기준
- `mixed == 1` only (카운트는 실행 시점 입력 데이터 기준)
- `step_TF in ["step", "nonstep"]`
- Step onset 누락 시행 제외
- Event frame이 기록 범위 벗어나는 시행 제외
- SPM 주분석에서는 step/nonstep 둘 다 존재하는 피험자만 포함

## 분석 변수 목록

### A. 원본 시계열 변수 (CSV 컬럼 직접 사용)

| Family | Variables | 개수 |
|--------|-----------|------|
| COM | COM_X, COM_Y, COM_Z | 3 |
| vCOM | vCOM_X, vCOM_Y, vCOM_Z | 3 |
| xCOM | xCOM_X, xCOM_Y, xCOM_Z | 3 |
| BOS | BOS_area, BOS_minX, BOS_maxX, BOS_minY, BOS_maxY | 5 |
| MOS | MOS_minDist_signed, MOS_AP_v3d, MOS_ML_v3d, MOS_v3d | 4 |
| GRF | GRF_X_N, GRF_Y_N, GRF_Z_N | 3 |
| GRM | GRM_X_Nm_at_FPorigin, GRM_Y_Nm_at_FPorigin, GRM_Z_Nm_at_FPorigin | 3 |
| COP | COP_X_m, COP_Y_m | 2 |
| Ankle Torque | AnkleTorqueMid_int_Y_Nm_per_kg (체중 정규화) | 1 |
| Trunk | Trunk_X_deg, Trunk_Y_deg, Trunk_Z_deg | 3 |
| Neck | Neck_X_deg, Neck_Y_deg, Neck_Z_deg | 3 |

### B. 파생 변수 (스크립트에서 계산)

| Family | Variables | 개수 |
|--------|-----------|------|
| Stance Joint Angles | Hip_stance_X_deg, Knee_stance_X_deg, Ankle_stance_X_deg | 3 |
| xCOM/BOS | xCOM_BOS_AP_foot, xCOM_BOS_ML_foot | 2 |

**총 ~38개 변수**

## 구현 단계

### 1. 환경 준비
```bash
conda run -n module pip install spm1d
mkdir -p analysis/step_vs_nonstep_spm/figures
```

### 2. 스크립트 작성 (`analysis/step_vs_nonstep_spm/analyze_step_vs_nonstep_spm.py`)

```
[Step 1] 데이터 로드 & 필터링
  - polars로 CSV 로드 → mixed==1 필터
  - `data/perturb_inform.xlsm`의 `meta` 시트(`발길이_왼`, `발길이_오른`)에서 `foot_len_m` 계산 후 병합(Join)
  → 검증: step/nonstep trial 카운트 출력, foot_len_m 결측치 없음

[Step 2] major_step_side 계산
  - build_subject_major_step_side() 재사용
  → 검증: 분석 대상 피험자에 major_step_side 누락 없음

[Step 3] Stance-side 관절각 파생 컬럼 추가
  - Hip/Knee/Ankle L/R → stance X (3개)
  → 검증: NaN 없는 stance 컬럼 생성

[Step 4] xCOM/BOS 파생 컬럼 추가
  - foot_len_m을 이용하여 xCOM_BOS_AP_foot, xCOM_BOS_ML_foot 계산
  → 검증: NaN 없는 정규화 컬럼 생성

[Step 5] End frame 계산
  - _compute_end_frames() 로직 적용
  → 검증: 모든 시행에 end_frame 존재

[Step 6] 시간 정규화
  - 각 시행: platform_onset ~ end_frame 구간 → numpy.interp 101 points
  - NaN 20% 초과 시행 제외, 나머지 선형 보간
  → 검증: (N_trials, 101) 행렬, NaN 없음

[Step 7] 피험자별 평균
  - step/nonstep 각각 피험자 내 평균
  - step 또는 nonstep 한 조건만 있는 피험자는 SPM 주분석에서 제외
  → 검증: 변수별 `Y_step`, `Y_nonstep` shape 일치 `(N_pairs, 101)`, `N_pairs` 기록

[Step 8] SPM paired t-test 루프
  - Parametric: spm1d.stats.ttest_paired
  - Nonparametric: spm1d.stats.nonparam.ttest_paired (10000 iterations)
  - 비모수 순열검정은 고정 random seed 사용
  - Bonferroni alpha = 0.05 / family_size
  → 검증: 모든 변수에 대해 SPM 결과 객체 생성

[Step 9] 시각화
  - 변수별 2-panel: (상) mean±SD 곡선, (하) SPM{t} + threshold + 유의 구간
  - 유의 구간 히트맵 (전체 변수 한눈에)
  → 검증: figures/ 디렉토리에 PNG 생성

[Step 10] 결과 출력
  - spm_results.csv (utf-8-sig): 변수, family, N_pairs, 제외 피험자 수, cluster 정보, p-value, 보정 결과
  - report.md: 주요 발견 요약
  → 검증: CSV에 모든 변수 포함
```

### 3. 출력 구조
```
analysis/step_vs_nonstep_spm/
    analyze_step_vs_nonstep_spm.py
    figures/
        spm_{variable_name}.png       # 변수별 SPM 2-panel
        heatmap_significant.png       # 유의 구간 히트맵 요약
    spm_results.csv                   # 전체 결과 테이블
    report.md                         # 주요 결과 요약
```

## 주요 참조 파일
- `analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py` — stance 로직 (L316-383), end_frame (L457-568)
- `analysis/xCOM&BOS_normalization/analyze_com_xcom_screening_lmm.py` — foot_len 로딩/계산 로직 (L159-197)
- `output/all_trials_timeseries.csv` — 입력 데이터 (98 컬럼, 42846 rows; 현재 스냅샷)
- `data/perturb_inform.xlsm` — end_frame fallback용 platform sheet + foot_len 계산용 meta sheet
- `config.yaml` — 플롯 레이아웃 참조

## 검증 계획
1. `import spm1d` 성공 확인
2. 시간 정규화 후 변수별 `(N_pairs, 101)` 행렬에 NaN 없음
3. step/nonstep 중 한 조건만 있는 피험자가 SPM 주분석에서 제외되었는지 확인
4. 기존 LMM에서 유의했던 변수(xCOM_BOS, Hip_stance_X 등)가 SPM에서도 특정 구간에서 유의한지 교차 확인
5. 전체 스크립트 end-to-end 실행 → figures/ + CSV + report 생성 확인
