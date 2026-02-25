# TEMP: Raw xCOM/BOS Decomposition LMM (Step vs Nonstep)

## Research Question

**Does step vs nonstep difference come from xCOM numerator, BOS denominator, or both in raw samples?**

## Data

- Input CSV: `/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/replace_V3D/analysis/initial_posture_strategy_lmm/tmp_raw_nometa_output/all_trials_timeseries_raw_nometa.csv`
- Trials: **184**, Subjects: **24**, step=**112**, nonstep=**72**

## Method

- Snapshot events: onset (`platform_onset_local`), 300ms (`platform_onset_local + 30`)
- DVs: `xCOM_rel_minX`, `BOS_rangeX`, `xCOM_BOS_norm` at both events
- Model: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`), BH-FDR over 6 DVs

## Results

| dv                  |     estimate |          SE |      df |   t_value |     p_value |       p_fdr | sig   |   mean_step |   sd_step |   mean_nonstep |   sd_nonstep |   n_step |   n_nonstep | converged   |
|:--------------------|-------------:|------------:|--------:|----------:|------------:|------------:|:------|------------:|----------:|---------------:|-------------:|---------:|------------:|:------------|
| xCOM_rel_minX_onset | -0.0115016   | 0.00136668  | 160.548 | -8.41575  | 2.03971e-14 | 6.11914e-14 | ***   |   0.12943   | 0.0162372 |       0.14441  |    0.0161779 |      112 |          72 | True        |
| BOS_rangeX_onset    | -4.47041e-05 | 0.000716688 | 159.531 | -0.062376 | 0.950341    | 0.950341    | n.s.  |   0.204471  | 0.0134513 |       0.206317 |    0.0118515 |      112 |          72 | True        |
| xCOM_BOS_norm_onset | -0.0558655   | 0.0065469   | 160.934 | -8.53312  | 1.00409e-14 | 6.02451e-14 | ***   |   0.633107  | 0.0673735 |       0.700486 |    0.0726557 |      112 |          72 | True        |
| xCOM_rel_minX_300ms | -0.0113099   | 0.00205272  | 159.687 | -5.50972  | 1.41204e-07 | 2.11805e-07 | ***   |   0.0656689 | 0.0310897 |       0.074771 |    0.0263169 |      112 |          72 | True        |
| BOS_rangeX_300ms    |  0.00478103  | 0.00169911  | 160.617 |  2.81385  | 0.0055073   | 0.00660876  | **    |   0.205841  | 0.0219477 |       0.203537 |    0.0134972 |      112 |          72 | True        |
| xCOM_BOS_norm_300ms | -0.0658679   | 0.00911855  | 159.591 | -7.22351  | 1.96197e-11 | 3.92395e-11 | ***   |   0.316629  | 0.137832  |       0.366763 |    0.124228  |      112 |          72 | True        |

## Reproduction

```bash
conda run -n module python analysis/initial_posture_strategy_lmm/tmp_analyze_raw_step_nonstep_xcom_bos_decompose_lmm.py
```
