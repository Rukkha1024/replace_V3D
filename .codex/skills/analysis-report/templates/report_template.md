# <Analysis Title>

## Research Question

**"<core question in quotes>"**

Background context in 1-2 sentences.

## Data Summary

- **N trials** (step=X, nonstep=Y)
- Data source: `output/all_trials_timeseries.csv` (N frames, 100Hz)
- Trial classification: `data/perturb_inform.xlsm` (platform sheet → step_TF)
- Key variables: <list dependent variables>

## Analysis Methodology

- **Analysis window**: <e.g., [platform_onset, step_onset] per trial>
  - Step trials: <actual step_onset_local>
  - Nonstep trials: <mean step_onset of same (subject, velocity) step trials>
- **Statistical model**: <e.g., LMM: DV ~ step_TF + (1|subject), REML>
- **Multiple comparison correction**: <e.g., Benjamini-Hochberg FDR per variable family>
- **Significance reporting**: `Sig` only (`*`, `**`, `***`, `n.s.`, alpha=0.05), hide numeric `p` and `df` in user-facing tables
- **Peak definition**: `*_peak` uses `abs_peak = max(|x|)` within the analysis window
- **Variable families**:
  - <Family 1>: <list variables>
  - <Family 2>: <list variables>
- **Special handling**: <any notable decisions, e.g., exclusion criteria>

---

## Results

### 1. <First analysis>

(tables, statistics, Sig markers, effect sizes)

| Variable | Estimate | SE | t | Sig |
|----------|----------|----|---|-----|
| ... | ... | ... | ... | ... |

### 2. <Second analysis>

...

---

## Interpretation

### <Interpretation subtitle 1>

...

### Conclusion

1. **Key finding 1**
2. **Key finding 2**
3. **Key finding 3**

---

## Reproduction

```bash
conda run --no-capture-output -n module python analysis/<topic>/analyze_<topic>.py
```

**Input**:
- `output/all_trials_timeseries.csv`
- `data/perturb_inform.xlsm`

**Output**: fig1–figN PNG (generated in this folder)

## Figures

| File | Description |
|------|-------------|
| fig1_xxx.png | ... |
| fig2_xxx.png | ... |
| fig3_xxx.png | ... |
