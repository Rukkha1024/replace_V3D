# TEMP: Raw xCOM/BOS Step vs Nonstep (No Filtering)

## Research Question

**"필터링 없이 거친 표본(raw samples)에서 step vs nonstep 간 xCOM/BOS 차이가 존재하는가?"**

## Data

- Input CSV: `/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/replace_V3D/analysis/initial_posture_strategy_lmm/tmp_raw_nometa_output/all_trials_timeseries_raw_nometa.csv`
- step/nonstep label source: `data/perturb_inform.xlsm` (platform sheet join)
- Trials (step/nonstep usable): **184**
- Subjects (usable): **24**
- step: **112**, nonstep: **72**
- 300ms snapshot offset: **+30 frames** (assumes 100 Hz)

## Methodology

- Snapshot 1: `platform_onset_local` (MocapFrame == platform_onset_local)
- Snapshot 2: `platform_onset_local + 30` (if within analysis window)
- DV: `xCOM_BOS_norm = (xCOM_X - BOS_minX) / (BOS_maxX - BOS_minX)`
- Model: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- Multiple comparison: BH-FDR across tested DVs (here: onset, 300ms)

## Results

| dv                  |   estimate |         SE |      df |   t_value |     p_value |       p_fdr | sig   |   mean_step |   sd_step |   mean_nonstep |   sd_nonstep |   n_step |   n_nonstep | converged   |
|:--------------------|-----------:|-----------:|--------:|----------:|------------:|------------:|:------|------------:|----------:|---------------:|-------------:|---------:|------------:|:------------|
| xCOM_BOS_norm_onset | -0.0558655 | 0.0065469  | 160.934 |  -8.53312 | 1.00409e-14 | 2.00817e-14 | ***   |    0.633107 | 0.0673735 |       0.700486 |    0.0726557 |      112 |          72 | True        |
| xCOM_BOS_norm_300ms | -0.0658679 | 0.00911855 | 159.591 |  -7.22351 | 1.96197e-11 | 1.96197e-11 | ***   |    0.316629 | 0.137832  |       0.366763 |    0.124228  |      112 |          72 | True        |

## Notes / Limitations

- No filtering applied (no meta_prefilter, no post-filter): dataset includes heterogeneous subjects/trials.
- Raw CSV generated with meta_prefilter disabled does not contain step_TF; labels are joined from platform sheet.
- Forceplate inertial subtract QC warnings may appear during batch export, but this analysis only uses xCOM/BOS (marker-based).

## Reproduction

```bash
# 1) generate raw output (no meta prefilter)
conda run --no-capture-output -n module python ../../main.py --no-meta_prefilter --skip_unmatched --overwrite --out_dir tmp_raw_nometa_output --out_csv tmp_raw_nometa_output/all_trials_timeseries_raw_nometa.csv

# 2) run this analysis
conda run -n module python analysis/initial_posture_strategy_lmm/tmp_analyze_raw_step_nonstep_xcom_bos_lmm.py
```
