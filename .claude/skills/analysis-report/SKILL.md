---
name: analysis-report
description: >-
  Create self-contained analysis workflows under analysis/.
  Each folder bundles a script (.py), report (report.md), and figures (.png).
  No Excel/CSV output — only figures and stdout statistics.
  Produces a report with research question → results → interpretation structure.
  Triggered when starting a new analysis topic or requesting work under analysis/.
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

### Code Rules

- **No `_bootstrap` module** — use direct `sys.path` setup (see boilerplate below)
- Prefer `polars`; use `pandas` only when stats APIs require it (pingouin, scipy, etc.)
- `matplotlib.use("Agg")` is mandatory (headless rendering)
- Korean font support code must be included
- Use `argparse` to allow input path overrides

## Script Boilerplate

Every `analyze_<topic>.py` starts with this path bootstrap:

```python
"""<Topic> Analysis.

Answers: "<research question>"

Produces:
  - N publication-quality figures (saved alongside this script)
  - stdout summary statistics

Usage:
    conda run -n module python analysis/<topic>/analyze_<topic>.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# path bootstrap (replaces _bootstrap dependency)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

matplotlib.use("Agg")

# Korean font support
_KO_FONTS = ("Malgun Gothic", "NanumGothic", "AppleGothic")
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _fname in _KO_FONTS:
    if _fname in _available:
        plt.rcParams["font.family"] = _fname
        break
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUT_DIR = SCRIPT_DIR  # figures saved alongside script
```

## Report Template

`report.md` follows this structure:

```markdown
# <Analysis Title>

## Research Question

**"<core question in quotes>"**

Background context in 1-2 sentences.

## Data Summary

- **N trials** (distribution by group)
- Data source and preprocessing summary
- Key variables

---

## Results

### 1. <First analysis>

(tables, statistics, p-values, etc.)

### 2. <Second analysis>

...

---

## Interpretation

### <Interpretation subtitle 1>
...

### Conclusion

1. **Key finding 1**
2. **Key finding 2**
...

---

## Reproduction

\```bash
conda run -n module python analysis/<topic>/analyze_<topic>.py
\```

**Input**: (input file paths)
**Output**: fig1–figN PNG (generated in this folder)

## Figures

| File | Description |
|------|-------------|
| fig1_xxx.png | ... |
| fig2_xxx.png | ... |
```

## Workflow

1. **Define research question** — Confirm core question, hypotheses, and required data with the user
2. **Create folder** — Create `analysis/<topic>/` directory
3. **Write script** — Build `analyze_<topic>.py` from boilerplate
   - Data loading → preprocessing → statistics → visualization → stdout output
4. **Run and verify** — `conda run -n module python analysis/<topic>/analyze_<topic>.py`
   - Confirm figure files are generated and stdout statistics are correct
5. **Write report** — Draft `report.md` based on stdout results
   - Research question → data summary → results (tables/stats) → interpretation → reproduction → figures
6. **Commit** — Commit the entire folder as a single commit

## Validation

- `analyze_<topic>.py` runs without errors
- All `fig*.png` files are generated in `analysis/<topic>/`
- `report.md` exists and contains minimum sections (research question, results, interpretation, reproduction, figures)
- Key statistics are printed to stdout
- No Excel/CSV files are generated

## Completion Checklist

- [ ] `analysis/<topic>/` folder exists
- [ ] `analyze_<topic>.py` runs without errors
- [ ] All figure files generated
- [ ] `report.md` complete (all required sections present)
- [ ] stdout statistics match report.md content
- [ ] No Excel/CSV files present
- [ ] Committed
