# Issues (replace_V3D)

Policy:
- This file records **the problem itself** only (symptom / impact / repro file or log).
- Record **solutions/workarounds** in the global skill: `$replace-v3d-troubleshooting`.

---
## 2026-02-19

- [VIZ] `scripts/plot_bos_com_xy_sample.py` did not initialize a Hangul-capable matplotlib font at module load time, so GIF exports with Korean subject names could emit `Glyph ... missing from current font` warnings and render unreadable tofu glyphs in title/metadata text.
- [ANALYSIS] In `analysis/why_stepping_before_threshold`, direct comparison between COP-based boundaries and COM/xCOM metrics is not valid due to coordinate-system incompatibility in the lab setup, making COP-driven boundary analyses non-interpretable for this dataset.
- [ANALYSIS] In `analysis/why_stepping_before_threshold`, switching from single-timepoint snapshots to window-mean aggregation (`platform_onset_local ~ step_onset_local`, nonstep end by subject mean step onset) materially changes model ranking and coefficient significance, so legacy snapshot conclusions cannot be reused for current methodology.
- [TOOLING] There was no standardized skill for end-to-end review/repair of existing `analysis/*` workflows (script reproducibility, quantitative consistency, and report alignment), causing repeated ad-hoc review patterns across analysis topics.

## 2026-02-18

- [JOINT] In `src/replace_v3d/joint_angles/v3d_joint_angles.py`, shank X-axis was previously derived from ankle medial/lateral pair (`LFoot_3-LANK`, `RANK-RFoot_3`) using the same directional information as foot X-axis, causing `Ankle_L_Z_deg`/`Ankle_R_Z_deg` to collapse near zero (micro-degree scale) in joint-angle outputs.
- [VIZ] In `scripts/plot_bos_com_xy_sample.py`, runs against CSV files without `xCOM_X`/`xCOM_Y` cannot render xCOM overlays; without explicit fallback handling this can break expected single-pass GIF generation for mixed historical outputs.
- [VERIFY] Running a repo script copied to `/tmp` can fail with `ModuleNotFoundError: No module named '_bootstrap'`, which blocks baseline/reference GIF generation outside the repository script path.
- [VIZ] For freeze/live BOS comparison in GIF, if axis limits are not fixed from COM + BOS (bbox/hull/union) full-range, one mode can appear clipped or visually rescaled, reducing interpretability across modes.
- [VIZ] GIF export can emit matplotlib warnings for Korean glyphs (`Glyph xxxx missing from font(s) DejaVu Sans`), and Hangul text (e.g., subject names) may render as tofu/missing characters in titles or side-panel metadata.
- [IO] In `scripts/plot_bos_com_xy_sample.py`, when output folders are created with the raw `subject` value, a subject string containing OS-reserved path characters can fail directory creation/saving.

## 2026-02-17

- [ENV] In non-interactive WSL2 shells, `conda` may not be on PATH, causing `conda run -n module ...` to fail (symptom: `conda: command not found`).
- [ENV] When running `main.py --overwrite --skip_unmatched` or `scripts/run_batch_all_timeseries_csv.py`, the OpenMP runtime may abort with `OMP: Error #179: Function Can't open SHM2 failed` (interrupts batch CSV regeneration).
- [ENV] When running `scripts/plot_grid_timeseries.py`, `/home/alice/.config/matplotlib` and fontconfig cache paths are not writable, causing temporary cache warnings on every execution.
- [VIZ] In `scripts/plot_grid_timeseries.py`, the overwrite policy for regenerating files with the same filename is not explicitly documented in code or logs, making it difficult to determine from execution logs alone whether figures were actually overwritten after regeneration.

## 2026-02-16

- [DATA] `data/all_data/251128_방주원_perturb_200_005.c3d`: marker `T10` missing → joint-angle computations may fail in batch pipelines that expect the full marker set.
- [QC] Some trimmed C3D files are exactly 1 frame shorter than the expected interval `[platform_onset-100, platform_offset+100]` (`delta_frames=-1`). (No violations beyond ±1 frame were observed.)
- [VIZ] `scripts/plot_grid_timeseries.py`: The `linewidth/alpha` of the same overlay plot differs between `--sample` and `all` rendering modes, causing `all` results to appear excessively faint (poor readability). Line representation categories effectively branch on factors beyond color/style. (Repro example: `output/figures/grid_timeseries/by_subject/강비은/y0/grf_cop__subject-강비은__velocity-30__all.png`)

## 2026-02-15

- [ENV] In some execution environments, destructive git commands may be blocked by runtime policy (e.g., `git reset --hard`, `git branch -D`).
- [DATA] If a C3D filename token is not mapped by `resolve_subject_from_token()`, the batch/driver may fail (e.g., token `KUO` → alias not registered in `data/perturb_inform.xlsm`).
