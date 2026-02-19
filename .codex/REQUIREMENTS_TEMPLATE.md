# Project Requirements Brief

This brief is completed for the `analysis/why_stepping_before_threshold` window-mean reanalysis rewrite task.

---

## Task Classification (AGENT)

- [x] **Bug Fix** — Replace incorrect behavior with correct behavior. Do not modify surrounding code.
- [ ] **New Feature** — Add functionality that does not currently exist.
- [ ] **Refactor** — Improve code structure without changing behavior.
- [ ] **Config / Data Change** — Modify settings or data without changing logic.

Selected: **Bug Fix**

---

## One-Line Summary (AGENT)

This analysis system recomputes stepping predictors using a platform-onset to step-onset window mean so that users can report quantitatively valid FSR-only results under COP-COM coordinate mismatch constraints.

## Problem Statement (AGENT)

The current report and script are based on a single snapshot frame (`ref_frame`) and no longer match the user-approved analysis definition. The user requested full replacement with a window-based quantitative reanalysis: `platform_onset_local ~ step_onset_local`, with nonstep end frame defined by subject mean step onset. Without this correction, the reported AUC/GLMM values and conclusions are not tied to the requested methodology.

## Ambiguity & Assumptions (AGENT — Think Before Coding)

### Interpretations Considered

- Interpretation A: Keep snapshot results and append a reanalysis section.
- Interpretation B: Replace all snapshot logic and rewrite report with reanalysis-only results.
- **Selected**: Interpretation B (Reason: explicit user instruction: "재분석 수치로 완전히 리포트 재작성").

### Explicit Assumptions

1. `platform_onset_local` exists per trial and can be used as a valid window start.
2. `step_onset_local` is available for stepping trials and can be averaged per subject for nonstep window end.
3. `MocapFrame` is the frame axis for window filtering.
4. Existing GLMM/LOSO model family remains unchanged; only trial summarization changes.

### Clarifying Questions (USER)

1. 없음 (핵심 의사결정은 이미 사용자 지시로 확정됨).

---

## Success Criteria (USER)

1. `analyze_fsr_only.py`가 window mean 방식으로 동작한다.
2. `report.md`가 snapshot 내용을 제거하고 재분석 실제 정량값으로 완전 교체된다.
3. GPT Comment에 `Alternative Applied`와 `Actual Result (Quant)`가 포함된다.

## Verifiable Goals (AGENT — Goal-Driven Execution)

| Original Instruction | Transformed Verifiable Goal |
|---------------------|-----------------------------|
| "수치재분석을 하라" | `platform_onset_local~step_onset_local` window mean으로 4개 모델 AUC/GLMM 값이 stdout에 출력된다. |
| "리포트 완전 재작성" | `analysis/why_stepping_before_threshold/report.md`에서 snapshot/ref_frame 문구와 기존 수치(0.794/0.787 등)가 제거되고, 재분석 수치로 대체된다. |
| "Expected가 아닌 Actual" | 리포트 표의 모든 수치가 실행 로그에서 직접 추출한 값과 일치한다. |

### Verification Commands

    $ conda run -n module python analysis/why_stepping_before_threshold/analyze_fsr_only.py
    Expected output: window definition + 4-model GLMM + 4-model LOSO AUC + fig1~fig4 생성 로그

    $ rg -n "0.794|0.787|snapshot|ref_frame" analysis/why_stepping_before_threshold/report.md
    Expected output: no matches

---

## Inputs

### Data Sources (USER)

- Format(s): CSV, XLSM
- Location / example path: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`
- Schema or structure: trial-level keys (`subject`, `velocity`, `trial`) + frame axis (`MocapFrame`) + onset columns (`platform_onset_local`, `step_onset_local`) + biomechanical columns (`COM_X`, `vCOM_X`, `BOS_minX`, `BOS_maxX`, `MOS_minDist_signed`)
- Volume: 184 trials, 24 subjects

### External Dependencies (AGENT)

- Required libraries: polars, pandas, numpy, scipy, sklearn, matplotlib, seaborn, statsmodels
- Required services: 없음
- Required hardware/environment: conda env `module`, WSL2/Linux shell

## Outputs

### Primary Deliverables (USER)

- Format(s): Python script update, Markdown report rewrite, PNG figures
- Destination: `analysis/why_stepping_before_threshold/`
- Naming convention: `fig1~fig4` 유지

### Secondary Artifacts (AGENT)

- Logs: `/tmp/why_step_*` verification logs
- Intermediate files: temporary verification outputs under `/tmp`
- Documentation: `.codex/REQUIREMENTS_TEMPLATE.md`, `.codex/execplans/*`

---

## Surgical Change Boundary (AGENT — Surgical Changes)

### Files / Modules to Touch

| File | Reason for Change | Scope of Change |
|------|-------------------|-----------------|
| `.codex/REQUIREMENTS_TEMPLATE.md` | requirements discovery completion | full content update |
| `.codex/execplans/why_stepping_window_reanalysis_execplan_ko.md` | Korean ExecPlan requirement | new file |
| `.codex/execplans/why_stepping_window_reanalysis_execplan_en.md` | English ExecPlan requirement | new file |
| `analysis/why_stepping_before_threshold/analyze_fsr_only.py` | snapshot -> window mean logic replacement | method-level edits |
| `analysis/why_stepping_before_threshold/report.md` | quantitative full rewrite | full content replacement |
| `.codex/issue.md` | issue-only logging update | append issue item |
| `/home/alice/.codex/skills/replace-v3d-troubleshooting/SKILL.md` | workaround logging update | append workaround section |

### Files Explicitly NOT to Touch

- `analysis/step_vs_nonstep_lmm/report.md`
- `src/replace_v3d/**`
- `scripts/**`

### Style Match Rules

- [x] Follow the existing variable naming conventions in the codebase.
- [x] Preserve existing indentation and formatting.
- [x] Only delete code that MY changes orphaned — do not clean up unrelated code.

---

## Constraints

### Technical (AGENT)

- Language: Python 3.10+
- Framework: standalone analysis script
- Must use: `conda run -n module python`, polars then pandas
- Must avoid: COP/eBOS logic 재도입

### Operational (AGENT)

- Runtime budget: single run within a few minutes
- Platform: WSL2 + conda
- Deployment: one-off analysis script/report update

## Scope

### In Scope (AGENT)

1. Window mean trial summarization logic 구현
2. GLMM/LOSO 수치 재실행 및 figure 재생성
3. report.md를 실제 정량 결과로 전면 교체

### Out of Scope (USER)

1. COP/eBOS 분석 재도입
2. 다른 analysis 폴더 수정
3. 새로운 모델(딥러닝 등) 추가

## Minimal Implementation Contract (AGENT — Simplicity First)

### Will Build

1. 기존 4개 모델(2D/1D vel/1D pos/1D MoS)의 입력 요약 방식만 window mean으로 변경
2. 해당 출력 수치 기반 report 재작성 + GPT Comment 강화

### Will NOT Build (by design)

- 추가 파일 포맷(Excel/CSV) 출력: 분석 폴더 규칙 위반이라 제외
- 모델 구조 변경: 사용자 요청은 재분석 방식 교체이지 모델 확장이 아님

### No Speculative Code Checklist

- [x] No new helper/utility classes or functions that are only used once.
- [x] No code added "because it might be needed later."
- [x] No validation added for scenarios that cannot happen given internal guarantees.

---

## Prior Art and References (USER)

- Reference implementation: `analysis/why_stepping_before_threshold/analyze_fsr_only.py` (current snapshot version)
- Paper / method: Pai & Patton(1997), Hof & Curtze(2016) conceptual references
- Similar project: 없음

## User Expertise Level (AGENT)

- Domain expertise: biomechanics high
- Coding proficiency: Python intermediate+
- Will the user modify the code after delivery? yes

## Open Questions (AGENT)

1. 없음
2. 없음

---

## Requirements Confirmed

- [x] User has reviewed and approved all sections above.
- [x] All Open Questions have been resolved.
- [x] All Clarifying Questions (Ambiguity & Assumptions) have been answered by the user.
- [x] Verifiable Goals table is fully filled in.
- [x] Surgical Change Boundary has been reviewed and agreed upon.
- [x] Minimal Implementation Contract has been agreed upon.
- [x] Agent may now proceed to author the ExecPlan per `.codex/PLANS.md`.
