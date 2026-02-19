---
name: analysis-report
description: "Create self-contained analysis workflows under analysis/. Each folder bundles a script (.py), report (report.md), and figures (.png). No Excel/CSV output. Produces a report with research question, methodology, results, interpretation structure."
---

# Analysis Report Workflow

## Trigger

- Starting a new analysis topic (research question, exploratory analysis)
- Requests to work under the `analysis/` folder
- Need to bundle statistics + visualization into a single self-contained folder
- Standalone analyses independent of the main pipeline (`scripts/`)

## Non-Negotiable Rules

### Folder Structure

```
analysis/<topic>/
  analyze_<topic>.py      # single entry point
  report.md               # research report
  fig1_<desc>.png         # figures (numbered + descriptive)
  fig2_<desc>.png
  ...
```

- Folder name: `snake_case`, descriptive of the analysis topic
- Script: `analyze_<topic>.py` (single entry point)
- Report: `report.md` (always present)
- Figures: `fig<N>_<short_desc>.png` (numbered sequentially)

### Output Constraints

- **No Excel/CSV generation** — only reproducible figures (.png) and stdout statistics
- `DEFAULT_OUT_DIR = SCRIPT_DIR` — figures saved alongside the script
- If intermediate data is needed, process in memory; do not export to files

### Reporting Conventions (default)

- User-facing report tables should hide numeric `p` values and `df` by default.
- Report significance with `Sig` only: `*`, `**`, `***`, `n.s.` at `alpha=0.05`.
- If multiple comparison correction is used, compute `Sig` from the corrected p-values (e.g., BH-FDR) and state the correction method in Analysis Methodology.
- If variables use `*_peak`, define it explicitly as absolute peak: `abs_peak = max(|x|)` within the analysis window.

### Code Rules

- **No `_bootstrap` module** — use direct `sys.path` setup (see `templates/script_boilerplate.py`)
- Prefer `polars`; use `pandas` when stats APIs require it (pingouin, scipy) **or** for R subprocess CSV I/O
- `matplotlib.use("Agg")` is mandatory (headless rendering)
- Korean font support code must be included
- Use `argparse` to allow input path overrides
- **`--dry-run` flag is mandatory** — loads data and prints summary without running analysis
- Milestone-based `main()` structure: `[M1] Load → [M2] Stats → [M3] Figures`
- Use `templates/script_boilerplate.py` as the starting point for every new script

### Environment Notes

- **`conda run --no-capture-output`**: Always use this flag. Korean paths cause `cp949` encoding errors without it.
  ```bash
  conda run --no-capture-output -n module python analysis/<topic>/analyze_<topic>.py
  ```
- **R subprocess (when needed)**: `rpy2` is broken on Windows conda (R not built as shared library). Use `Rscript.exe` via `subprocess.run()` with explicit `R_HOME`/`PATH` env vars. See `analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py` lines 62-82 for the pattern.
- **PNG files are gitignored**: Commit figures with `git add -f analysis/<topic>/*.png`

### Color Conventions

| Purpose | Key | Color |
|---------|-----|-------|
| Step group | `step` | `#E74C3C` (red) |
| Non-step group | `nonstep` | `#3498DB` (blue) |
| Balance/Stability family | — | `#2ecc71` (green) |
| Joint Angles family | — | `#e67e22` (orange) |
| Force/Torque family | — | `#9b59b6` (purple) |

## Templates

- **Script boilerplate**: `templates/script_boilerplate.py` — copy and modify for each new analysis
- **Report template**: `templates/report_template.md` — includes Analysis Methodology section

## Companion Skills

- **Prior-study-based analyses**: When replicating or following a published study's methodology, use `prior-study-replication` skill alongside this one. That skill provides an alternative `report.md` template with mandatory sections for prior study methods/results, methodological adaptation rationale, and systematic comparison. Code rules and folder structure still follow this skill.

## Workflow

1. **Define research question** — Confirm core question, hypotheses, and required data with the user
2. **Create folder** — Create `analysis/<topic>/` directory
3. **Write script** — Copy `templates/script_boilerplate.py` → `analyze_<topic>.py`, then customize
   - Data loading → preprocessing → statistics → visualization → stdout output
4. **Dry-run verify** — `conda run --no-capture-output -n module python analysis/<topic>/analyze_<topic>.py --dry-run`
   - Confirm data loads correctly, trial counts match expectations
5. **Full run & verify** — Remove `--dry-run` flag, confirm all figures are generated
6. **Write report** — Use `templates/report_template.md` as starting point for `report.md`
   - Research question → data summary → **analysis methodology** → results → interpretation → reproduction → figures
7. **Commit** — `git add -f analysis/<topic>/*.png` then commit the entire folder

## Validation

- `analyze_<topic>.py --dry-run` succeeds (data loading OK)
- `analyze_<topic>.py` full run completes without errors
- All `fig*.png` files are generated in `analysis/<topic>/`
- `report.md` exists with required sections: Research Question, Data Summary, **Analysis Methodology**, Results, Interpretation, Reproduction, Figures
- Key statistics printed to stdout match report.md content
- No Excel/CSV files generated
- User-facing report tables follow `Sig-only` convention (no numeric `p`/`df` columns unless user explicitly requests)

## Completion Checklist

- [ ] `analysis/<topic>/` folder exists
- [ ] `analyze_<topic>.py --dry-run` succeeds
- [ ] `analyze_<topic>.py` full run completes without errors
- [ ] All figure files generated
- [ ] `report.md` complete (all required sections present, including Analysis Methodology)
- [ ] stdout statistics match report.md content
- [ ] No Excel/CSV files present
- [ ] Figures force-added: `git add -f analysis/<topic>/*.png`
- [ ] Committed
