# why_stepping_before_threshold window-mean reanalysis migration

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated continuously during implementation.

Repository policy reference: follow `.codex/PLANS.md`.

## Purpose / Big Picture

The user no longer accepts snapshot-based results because COP-COM coordinate mismatch invalidates earlier interpretation paths. After this change, the analysis will recompute trial features using a window mean from `platform_onset_local` to `step_onset_local`, and `report.md` will present only actual quantitative outputs from this reanalysis.

## Progress

- [x] (2026-02-19 11:45Z) Requirements fixed: window rule, nonstep end-frame rule, mean aggregation rule.
- [x] (2026-02-19 12:20Z) Replaced snapshot logic in `analyze_fsr_only.py` with window aggregation.
- [x] (2026-02-19 12:24Z) Re-ran analysis and regenerated fig1~fig4 with extracted GLMM/AUC values.
- [x] (2026-02-19 12:30Z) Fully rewrote `report.md` with actual quantitative outputs.
- [x] (2026-02-19 12:34Z) Appended issue-only note to `.codex/issue.md` and workaround note to `$replace-v3d-troubleshooting` skill.
- [x] (2026-02-19 12:36Z) Completed MD5/numeric verification and prepared Korean 3+ line commit.

## Surprises & Discoveries

- Observation: Model ranking can change between snapshot and window-mean aggregation.
  Evidence: Probe AUC values show weaker 1D velocity dominance (around 0.647) with narrowed gaps to 2D/1D position.

## Decision Log

- Decision: Fully replace snapshot report content instead of keeping both versions.
  Rationale: Explicit user instruction (“완전히 리포트 재작성”).
  Date/Author: 2026-02-19 / Codex

- Decision: For nonstep trials, end frame is subject-level mean step onset.
  Rationale: Explicit user rule.
  Date/Author: 2026-02-19 / Codex

- Decision: Trial-level feature aggregation uses window mean.
  Rationale: Explicit user rule.
  Date/Author: 2026-02-19 / Codex

## Outcomes & Retrospective

Window-mean reanalysis fully replaced snapshot reporting and changed interpretation strength. The previous “strong velocity dominance” claim is no longer valid under this methodology; the top three models are now close. Reproducibility was preserved by keeping CLI stable, regenerating figures with the same filenames, and storing numeric/md5 verification artifacts.

## Context and Orientation

The key files are `analysis/why_stepping_before_threshold/analyze_fsr_only.py` and `analysis/why_stepping_before_threshold/report.md`. The current implementation extracts one snapshot frame (`ref_frame`) per trial. This migration replaces that behavior with frame-window aggregation: normalize per frame, then average within each trial window.

Term definitions:
- window mean: average value over all frames in the start-end range.
- start frame: `platform_onset_local`.
- end frame: `step_onset_local` for step trials, subject mean `step_onset_local` for nonstep trials.

## Plan of Work

In `analyze_fsr_only.py`, convert `build_trial_summary` into onset-window metadata construction and convert `compute_fsr_features` into frame-level normalization + trial-level mean aggregation. Update console logs and figure titles to explicitly indicate “window mean”.

Fully rewrite `report.md` by removing old snapshot values and inserting new run-derived values. Add GPT comment blocks to major sections with `Alternative Applied` and `Actual Result (Quant)` lines.

## Concrete Steps

Working directory: repository root

    conda run -n module python analysis/why_stepping_before_threshold/analyze_fsr_only.py

Expected logs:
- window definition line
- trial/frame summary line
- four GLMM blocks
- four LOSO AUC lines
- fig1~fig4 saved

Validation commands:

    rg -n "0.794|0.787|snapshot|ref_frame" analysis/why_stepping_before_threshold/report.md
    rg -n "Alternative Applied|Actual Result \(Quant\)" analysis/why_stepping_before_threshold/report.md

## Validation and Acceptance

1. Script exits with code 0 and generates fig1~fig4.
2. Output contains exactly four models: 2D, 1D velocity, 1D position, 1D MoS.
3. Table values in report equal the run logs.
4. Snapshot/ref_frame wording and legacy values are absent from report.

## Idempotence and Recovery

Running the same command repeatedly should regenerate the same artifact structure (same filenames and log sections). If values differ, compare logs first to isolate data changes versus environment differences.

## Artifacts and Notes

- Baseline log: `/tmp/why_step_before_stdout.txt`
- Baseline md5: `/tmp/why_step_before_md5_before.txt`
- Post-change logs/md5 will use matching `/tmp/why_step_after_*` naming.

## Interfaces and Dependencies

- CLI preserved: `--csv`, `--platform_xlsm`, `--out_dir`, `--dpi`
- Model interfaces preserved:
  - GLMM: `fit_glmm(data, formula, groups="subject")`
  - LOSO: `compute_loso_cv_auc(data, feature_cols, y_col, group_col)`
- Data interface changed:
  - trial summary now includes `start_frame`, `end_frame`
  - feature table uses `COM_pos_norm`, `COM_vel_norm`, `MOS_minDist_signed`, `n_frames` from window mean

---
Change Note (2026-02-19): New ExecPlan created to capture user-locked rules (window start/end, mean aggregation, full report replacement).
