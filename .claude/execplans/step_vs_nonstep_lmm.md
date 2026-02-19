# Step vs. Non-step Biomechanical LMM 비교 분석 ExecPlan

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document must be maintained in accordance with `.claude/PLANS.md`.

Skills used: `analysis-report`, `pingouin-excel-stat-analysis` (FDR reference only)

---

## Purpose / Big Picture

본 연구는 동일 강도의 perturbation에서 step과 non-step이라는 서로 다른 균형 회복 전략이 나타나는 이유를 규명하기 위해, biomechanical 변수들을 Linear Mixed Model(LMM)로 비교 분석한다. 분석 단위는 개별 trial(subject-velocity-trial)이며, 피험자를 임의 절편(random intercept)으로 설정하여 반복 측정의 상관 구조를 반영한다.

구현 완료 후, 사용자는 아래 명령을 실행하면 LMM 결과가 stdout에 출력되고, publication-quality figure 3개와 report.md가 스크립트 옆에 저장된다:

    conda run -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py

**Biomechanical 변수만 분석 (COM, COP, MoS, joint angles, ankle torque, GRF). EMG는 별도 ExecPlan으로 처리.**

---

## Progress

- [x] (2026-02-19) Milestone 0: R + lmerTest conda 설치, Rscript subprocess 방식으로 검증 완료
- [x] (2026-02-19) Milestone 1: 184 trials (step=112, nonstep=72), 32 DVs 집계 완료
- [x] (2026-02-19) Milestone 2: LMM 32개 적합, FDR 보정 → 22/32 유의
- [x] (2026-02-19) Milestone 3: fig1 forest, fig2 violin, fig3 heatmap 생성
- [x] (2026-02-19) Milestone 4: report.md 작성 완료

---

## Surprises & Discoveries

- Observation: rpy2+pymer4가 Windows conda 환경에서 작동하지 않음 ("R was not built as a library" 오류)
  Evidence: conda-forge R은 Windows에서 shared library로 빌드되지 않아 rpy2가 R.dll을 로드할 수 없음
  Solution: Rscript subprocess 방식으로 우회. 데이터를 CSV로 교환하고 R 스크립트를 subprocess.run()으로 실행

- Observation: conda run이 Korean 경로에서 cp949 인코딩 오류 발생
  Evidence: `UnicodeEncodeError: 'cp949' codec can't encode character '\ufffd'`
  Solution: `conda run --no-capture-output` 플래그로 해결

- Observation: (v1, 0-800ms) 22/32 변수 FDR 유의 → (v2, adaptive window) 13/32로 감소
  Evidence: Force/Torque family 3/7→0/7. GRF/torque 차이는 step onset 이후에만 존재함이 확인됨

- Observation: COP_X_range는 nonstep에서 유의하게 크다 (estimate=-0.0092)
  Evidence: nonstep이 AP 방향 COP 보상(fixed-support strategy)을 더 적극적으로 사용함을 시사

---

## Decision Log

- Decision: Biomechanical 변수만 분석, EMG 제외
  Rationale: EMG 데이터는 별도 프로젝트(EMG_analysis)에 위치하여 별도 ExecPlan으로 처리
  Date/Author: 2026-02-19

- Decision: pymer4 + rpy2 + R의 lmerTest를 설치하여 Satterthwaite 자유도 근사 사용
  Rationale: 논문 통계방법에서 Satterthwaite를 명시. statsmodels.MixedLM은 Wald z-test만 지원하므로 부적합
  Date/Author: 2026-02-19

- Decision: `output/all_trials_timeseries.csv` 전체 trial을 분석 대상으로 사용
  Rationale: 해당 CSV가 이미 mixed velocity 구간의 데이터만 포함 (사용자 확인)
  Date/Author: 2026-02-19

- Decision: analysis-report skill 적용 — Excel/CSV 출력 없음, figures + stdout + report.md만 생성
  Rationale: 사용자가 analysis-report skill 사용을 명시적으로 요청
  Date/Author: 2026-02-19

- Decision: 분석 시간 구간을 고정 0-800ms → [platform_onset, step_onset] per-trial adaptive window로 변경
  Rationale: 논문 방법론에서 CPA 구간(platform onset ~ step onset)을 분석 시간대로 정의. Nonstep trial은 동일 (subject, velocity) 내 step trial의 평균 step_onset을 대입
  Date/Author: 2026-02-19

---

## Outcomes & Retrospective

모든 마일스톤 완료. 32개 biomechanical 변수에 대해 LMM(Satterthwaite df) 적합 및 BH-FDR 보정을 수행하였고, 22/32 변수가 step vs. nonstep 간 FDR-유의한 차이를 보였다. 핵심 발견: ML 방향 안정성 지표가 step 유발의 주요 변별 요인이며, 무릎·고관절 ROM과 수직 GRF/ankle torque가 step에서 유의하게 크다. rpy2 호환성 문제로 Rscript subprocess 방식을 채택하여 Windows 환경에서의 재현성을 확보하였다.

---

## Context and Orientation

### Repository Structure

이 프로젝트(replace_V3D)는 platform translation perturbation 실험의 biomechanical 데이터를 처리하는 파이프라인이다. 핵심 파일 경로:

- **Main timeseries**: `output/all_trials_timeseries.csv` — 프레임 단위(100Hz) 시계열 데이터, 약 42,636행 × 100열. subject, velocity, trial 식별자와 COM, COP, MoS, joint angles, ankle torque, GRF 포함.
- **Trial classification**: `data/perturb_inform.xlsm` (platform sheet) — 1,038개 trial의 step_TF(step/nonstep), state, platform_onset, step_onset.
- **Existing analysis pattern**: `analysis/com_vs_xcom_stepping/analyze_com_vs_xcom_stepping.py` — 데이터 로딩, trial summary, 통계, figure 생성 패턴 참조.

### Key Columns in `all_trials_timeseries.csv`

식별자: `subject`, `velocity`, `trial`, `MocapFrame`, `time_from_platform_onset_s`
이벤트: `platform_onset_local`, `step_onset_local`

Biomechanical 변수:
- COM: `COM_X`, `COM_Y`, `COM_Z`, `vCOM_X`, `vCOM_Y`, `vCOM_Z`
- xCOM: `xCOM_X`, `xCOM_Y`, `xCOM_Z`
- BOS: `BOS_area`, `BOS_minX`, `BOS_maxX`, `BOS_minY`, `BOS_maxY`
- MOS: `MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`, `MOS_v3d`
- Joint angles: `Hip_*_X_deg`, `Knee_*_X_deg`, `Ankle_*_X_deg`, `Trunk_X_deg`, `Neck_X_deg` (좌우, 3축)
- COP: `COP_X_m_onset0`, `COP_Y_m_onset0` (onset zeroed)
- GRF: `GRF_X_N`, `GRF_Y_N`, `GRF_Z_N`
- Ankle torque: `AnkleTorqueMid_int_Y_Nm_per_kg` 등

### Statistical Method: Linear Mixed Model (LMM)

LMM이란 고정 효과(fixed effects)와 임의 효과(random effects)를 함께 추정하는 회귀 모델이다. 피험자 내 반복 측정이 있을 때, 피험자 간 기저값 차이를 임의 절편으로 모델링하여 의사독립성(pseudo-independence) 가정 위반을 해결한다.

pymer4는 Python에서 R의 lme4::lmer() + lmerTest를 호출하는 래퍼 라이브러리이다.

- 모델 공식: `DV ~ step_TF + (1|subject)`
  - `DV`: 종속변수 (trial-level summary statistic)
  - `step_TF`: 고정효과 — 균형회복 전략 (step vs. nonstep, 범주형)
  - `(1|subject)`: 임의절편 — 피험자별 기저값 차이
- 추정: REML (restricted maximum likelihood)
- 유의성: Satterthwaite 자유도 근사로 t-test (lmerTest 제공)
- 다중비교: Benjamini-Hochberg FDR (변수 가족별)

### Variable Families for FDR Correction

가족(family) 단위로 FDR 보정을 적용한다. 같은 가족 내 변수들의 p-value만 함께 보정한다:

1. **Balance/Stability**: COM range, COM path length, vCOM peak, COP range, COP path length, COP peak velocity, MoS min — 총 ~17개
2. **Joint angles**: Hip/Knee/Ankle/Trunk/Neck ROM, peak — 총 ~10개
3. **Force/Torque**: GRF peak, GRF range, ankle torque peak — 총 ~7개

### Analysis Time Window

Platform onset(0ms) ~ 800ms. `time_from_platform_onset_s` ≥ 0.0 AND ≤ 0.8.

---

## Plan of Work

### Milestone 0: pymer4 환경 설치 및 검증

pymer4는 Python에서 R의 lme4/lmerTest를 호출하는 래퍼이다. rpy2가 Python↔R 통신을 담당한다.

1. R이 시스템에 설치되어 있는지 확인 (`Rscript --version`). 없으면 사용자에게 R 설치를 안내하거나, `conda run -n module conda install -c conda-forge r-base r-lmertest`로 conda 내에 R을 설치한다.
2. `conda run -n module pip install rpy2 pymer4` 실행.
3. R 패키지 lmerTest 설치 (lme4는 자동 의존 설치됨).
4. 검증: 6행짜리 toy data에 LMM을 적합해 coefs에 Satterthwaite df가 소수점으로 나오는지 확인.

**Fallback**: pymer4 설치가 불가능하면 statsmodels.formula.api.mixedlm()을 사용하되, Wald z-test임을 report에 명시.

### Milestone 1: 데이터 로딩 및 trial-level 집계

`analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py` 스크립트를 생성한다. `analysis-report` skill의 boilerplate(path bootstrap, matplotlib Agg, Korean font)를 그대로 사용한다.

**1-1. 데이터 로딩** — polars로 CSV 로드, pandas로 xlsm platform sheet 로드, step_TF join.

**1-2. 시간 필터** — `time_from_platform_onset_s`가 [0.0, 0.8] 범위인 프레임만 남김.

**1-3. Trial-level 집계** — `group_by(["subject", "velocity", "trial"])`로 각 trial의 summary statistics 산출:

COM 변수 (X, Y 각각):
- `COM_{axis}_range`: max − min
- `COM_{axis}_path_length`: Σ|frame[i+1] − frame[i]|
- `vCOM_{axis}_peak`: max(|vCOM|)

COP 변수 (X, Y 각각, onset0 열):
- `COP_{axis}_range`: max − min
- `COP_{axis}_path_length`: Σ|frame[i+1] − frame[i]|
- `COP_{axis}_peak_velocity`: max(|Δ/Δt|), Δt = 0.01s

MoS 변수:
- `MOS_minDist_signed_min`: min(MOS_minDist_signed) — 가장 불안정한 순간
- `MOS_AP_v3d_min`: min(MOS_AP_v3d)
- `MOS_ML_v3d_min`: min(MOS_ML_v3d)

Joint angle 변수 (sagittal X축, 좌우 우세측 R):
- `{Joint}_X_range`: max − min (ROM)
- `{Joint}_X_peak`: max(|값|)
- 대상: Hip_R, Knee_R, Ankle_R, Trunk, Neck

GRF 변수:
- `GRF_{axis}_peak`: max(|값|)
- `GRF_{axis}_range`: max − min

Ankle torque:
- `AnkleTorqueMid_int_Y_peak`: max(|AnkleTorqueMid_int_Y_Nm_per_kg|)

최종 결과: 1 trial = 1 row의 pandas DataFrame. 열에 subject, velocity, trial, step_TF + 모든 DV.

### Milestone 2: LMM 적합 및 BH-FDR 보정

**2-1. LMM loop** — 각 DV에 대해:

    from pymer4.models import Lmer
    model = Lmer("DV ~ step_TF + (1|subject)", data=trial_df)
    model.fit(REML=True)

model.coefs DataFrame에서 step_TF 행의 Estimate, SE, df, T-stat, P-val을 추출. 또한 각 group의 mean, sd를 산출.

수렴 실패 시 해당 DV는 결과에 "convergence_failed"로 기록하고 skip.

**2-2. 결과 수집** — DataFrame columns: family, dv, n_step, n_nonstep, mean_step, sd_step, mean_nonstep, sd_nonstep, estimate, SE, df, t_value, p_value

**2-3. BH-FDR** — `statsmodels.stats.multitest.multipletests(pvals, method='fdr_bh')`를 family별로 적용. `p_fdr` 열 추가. `sig` 열에 `*` (p_fdr < 0.05), `**` (p_fdr < 0.01), `***` (p_fdr < 0.001) 표기.

**2-4. stdout 출력** — 전체 결과 테이블을 정렬 출력. FDR 유의 변수 별도 요약.

### Milestone 3: Figure 생성

`analysis-report` skill 규칙: figures는 스크립트와 같은 폴더에 `fig<N>_<desc>.png`로 저장. `DEFAULT_OUT_DIR = SCRIPT_DIR`.

**fig1_lmm_forest_plot.png** — Forest plot. 각 DV의 LMM estimate ± 95% CI를 수평 막대로 표시. 가로축: estimate(step − nonstep), 세로축: DV 이름. family별 색상 구분. FDR 유의한 항목은 굵은 글씨 + filled marker. 0 수직선 표시.

**fig2_violin_significant.png** — FDR 유의한 변수들의 violin + strip plot. step(빨강) vs nonstep(파랑) 비교. 최대 3×3 = 9개 subplot. 유의 변수가 9개 미만이면 subplot 수 조정. p_fdr 값을 subplot 제목에 표시.

**fig3_descriptive_heatmap.png** — Heatmap. 행: DV (family별 그룹), 열: step / nonstep, 값: z-score normalized mean. FDR 유의 항목에 `*` 마커 overlay. colorbar 포함.

### Milestone 4: report.md 작성 및 최종 검증

`analysis-report` skill의 report template에 따라 `analysis/step_vs_nonstep_lmm/report.md` 작성:

    # Step vs. Non-step Biomechanical LMM Analysis
    ## Research Question
    ## Data Summary
    ## Results
    ### 1. LMM Summary Table
    ### 2. Significant Variables
    ### 3. Effect Sizes
    ## Interpretation
    ### Conclusion
    ## Reproduction
    ## Figures

---

## Concrete Steps

작업 디렉토리: `c:\Users\Alice\OneDrive - 청주대학교\근전도 분석 코드\replace_V3D`

**Step 0: 환경 설치**

    # R 확인
    Rscript --version

    # rpy2 + pymer4 설치
    conda run -n module pip install rpy2 pymer4

    # R 패키지 설치
    conda run -n module python -c "from rpy2.robjects.packages import importr; utils = importr('utils'); utils.install_packages('lmerTest', repos='https://cran.r-project.org')"

    # 검증
    conda run -n module python -c "
    import pandas as pd
    from pymer4.models import Lmer
    df = pd.DataFrame({
        'y': [1,2,3,4,5,6,7,8],
        'x': ['a','b','a','b','a','b','a','b'],
        'subj': ['s1','s1','s2','s2','s3','s3','s4','s4']
    })
    m = Lmer('y ~ x + (1|subj)', data=df)
    m.fit(REML=True)
    print(m.coefs)
    print('OK: Satterthwaite df =', m.coefs['DF'].values)
    "

**Step 1: dry-run 테스트**

    conda run -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py --dry-run
    # Expected: "Loaded N trials (X step, Y nonstep). K dependent variables. Dry run complete."

**Step 2: 전체 분석 실행**

    conda run -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py
    # Expected stdout (excerpt):
    #   ============================================================
    #   Step vs. Non-step Biomechanical LMM Analysis
    #   ============================================================
    #   [M1] Loading and aggregating...
    #     Frames: 42636 → filtered: ~NNNNN (0-800ms)
    #     Trials: N (step=X, nonstep=Y)
    #     DVs: K variables
    #   [M2] Fitting LMMs...
    #     1/K COM_X_range          p=0.XXXX  FDR=0.XXXX
    #     ...
    #     FDR significant: M/K variables
    #   [M3] Generating figures...
    #     fig1_lmm_forest_plot.png
    #     fig2_violin_significant.png
    #     fig3_descriptive_heatmap.png
    #   ============================================================
    #   Analysis complete. Output: analysis/step_vs_nonstep_lmm/

---

## Validation and Acceptance

1. 스크립트 실행이 오류 없이 완료
2. `analysis/step_vs_nonstep_lmm/` 폴더에 fig1, fig2, fig3 PNG 파일 생성
3. report.md가 Research Question, Results, Interpretation, Reproduction, Figures 섹션을 포함
4. LMM 결과 검증:
   - df 값이 소수점 (Satterthwaite 자유도 확인)
   - p_fdr ≥ p_value
   - n_step + n_nonstep = 전체 trial 수
5. **No Excel/CSV files** in the output folder (analysis-report skill 규칙)
6. stdout에 주요 통계 출력

---

## Idempotence and Recovery

- 매 실행 시 output 파일을 덮어쓴다 (안전하게 재실행 가능)
- pymer4 설치 실패 시 statsmodels fallback 로직 내장
- `--dry-run` 플래그로 데이터 로딩만 테스트 가능

---

## Artifacts and Notes

### 생성될 파일

    analysis/step_vs_nonstep_lmm/
    ├── analyze_step_vs_nonstep_lmm.py   # 메인 분석 스크립트 (단일 entry point)
    ├── report.md                         # 분석 보고서
    ├── fig1_lmm_forest_plot.png          # Forest plot
    ├── fig2_violin_significant.png       # Violin plots (FDR 유의 변수)
    └── fig3_descriptive_heatmap.png      # Heatmap

### 참조 기존 코드

- `analysis/com_vs_xcom_stepping/analyze_com_vs_xcom_stepping.py` (828행): 데이터 로딩 패턴 (`pl.read_csv`, `load_platform_sheet`), figure 패턴, 색상 (`_step_nonstep_colors()`의 step="#E74C3C", nonstep="#3498DB")
- `.claude/skills/analysis-report/SKILL.md`: 폴더 구조, boilerplate, report template, 검증 기준

---

## Interfaces and Dependencies

**Python packages (module env, 이미 설치됨):**
- polars (0.20+): CSV 로딩, 집계
- pandas: pymer4 입력, step_TF join
- statsmodels (0.14.5): FDR 보정 (`multipletests`), fallback LMM
- matplotlib + seaborn: figure
- numpy, scipy: 수치 계산

**신규 설치 필요:**
- rpy2: Python↔R 통신
- pymer4: LMM 래퍼 (lme4/lmerTest 호출)
- R + lmerTest R package: Satterthwaite 자유도

**주요 함수 시그니처:**

    def load_and_prepare(csv_path: Path, xlsm_path: Path) -> pd.DataFrame:
        """Load timeseries, join step_TF, filter 0-800ms, aggregate to trial-level. Returns 1 row per trial."""

    def aggregate_trial_features(df: pl.DataFrame) -> pl.DataFrame:
        """group_by [subject, velocity, trial] → summary stats for all DVs."""

    def fit_lmm_single(trial_df: pd.DataFrame, dv: str, use_pymer4: bool) -> dict:
        """Fit one LMM. Returns dict with estimate, SE, df, t, p, group means/sds."""

    def fit_lmm_all(trial_df: pd.DataFrame, dv_list: list[str], family_map: dict) -> pd.DataFrame:
        """Loop over all DVs, collect results, apply FDR per family."""

    def fig1_forest(results: pd.DataFrame, out_dir: Path, dpi: int): ...
    def fig2_violin(trial_df: pd.DataFrame, results: pd.DataFrame, out_dir: Path, dpi: int): ...
    def fig3_heatmap(results: pd.DataFrame, out_dir: Path, dpi: int): ...

---

# English Version

## Purpose

Implement LMM analysis comparing biomechanical variables (COM, COP, MoS, joint angles, ankle torque, GRF) between step and non-step balance recovery strategies. Output: stdout statistics + 3 publication-quality figures + report.md. No Excel/CSV (per analysis-report skill).

## Context

Frame-level timeseries data (100Hz, ~42K rows) in `output/all_trials_timeseries.csv` with `data/perturb_inform.xlsm` providing step/nonstep classification. All trials are from "mixed velocity" zone. Analysis window: 0–800ms from platform onset. Trial-level aggregation produces range, path length, peak metrics per trial.

## Method

LMM via pymer4 (R's lme4 + lmerTest): `DV ~ step_TF + (1|subject)`, REML, Satterthwaite df. BH-FDR correction per variable family (balance/stability, joint angles, force/torque).

## Milestones

**M0**: Install pymer4/rpy2/lmerTest, verify with toy LMM (Satterthwaite df = decimal, not integer).

**M1**: Load data → filter 0–800ms → aggregate to trial-level features (COM/COP range/path_length/peak, MoS min, joint ROM/peak, GRF peak/range, torque peak).

**M2**: Fit LMM per DV → collect results → BH-FDR per family → stdout summary table.

**M3**: Generate fig1 (forest plot), fig2 (violin for significant DVs), fig3 (heatmap).

**M4**: Write report.md (research question → data summary → results → interpretation → reproduction → figures).

## Deliverables

    analysis/step_vs_nonstep_lmm/
    ├── analyze_step_vs_nonstep_lmm.py
    ├── report.md
    ├── fig1_lmm_forest_plot.png
    ├── fig2_violin_significant.png
    └── fig3_descriptive_heatmap.png
