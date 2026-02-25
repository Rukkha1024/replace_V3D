# TEMP: Raw Foot-Length Normalized LMM (Step vs Nonstep)

## Data

- Input CSV: `/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/replace_V3D/analysis/initial_posture_strategy_lmm/tmp_raw_nometa_output/all_trials_timeseries_raw_nometa.csv`
- Trials=184, Subjects=24, step=112, nonstep=72

## DVs

- `DV1_norm = (xCOM_X - BOS_minX) / foot_len_m`
- `DV1_abs_cm = (xCOM_X - BOS_minX) * 100`
- `DV2_norm = (COM_X - BOS_minX) / foot_len_m`

## Results

| dv               |   estimate |         SE |      df |   t_value |     p_value |       p_fdr | sig   |   mean_step |   sd_step |   mean_nonstep |   sd_nonstep |   n_step |   n_nonstep | converged   |
|:-----------------|-----------:|-----------:|--------:|----------:|------------:|------------:|:------|------------:|----------:|---------------:|-------------:|---------:|------------:|:------------|
| DV1_norm_onset   | -0.0460392 | 0.0054591  | 160.701 |  -8.43347 | 1.82794e-14 | 6.11914e-14 | ***   |    0.51421  | 0.0578221 |       0.571488 |    0.0653919 |      112 |          72 | True        |
| DV1_norm_300ms   | -0.0460923 | 0.00810308 | 159.703 |  -5.68824 | 5.97193e-08 | 8.95789e-08 | ***   |    0.260396 | 0.11905   |       0.296316 |    0.103573  |      112 |          72 | True        |
| DV1_abs_cm_onset | -1.15016   | 0.136668   | 160.548 |  -8.41575 | 2.03971e-14 | 6.11914e-14 | ***   |   12.943    | 1.62372   |      14.441    |    1.61779   |      112 |          72 | True        |
| DV1_abs_cm_300ms | -1.13099   | 0.205272   | 159.687 |  -5.50972 | 1.41204e-07 | 1.69444e-07 | ***   |    6.56689  | 3.10897   |       7.4771   |    2.63169   |      112 |          72 | True        |
| DV2_norm_onset   | -0.0421139 | 0.00536904 | 160.575 |  -7.84386 | 5.75623e-13 | 1.15125e-12 | ***   |    0.500489 | 0.056813  |       0.552465 |    0.062682  |      112 |          72 | True        |
| DV2_norm_300ms   | -0.0215583 | 0.00661212 | 160.026 |  -3.26041 | 0.00135933  | 0.00135933  | **    |    0.266929 | 0.0913471 |       0.289468 |    0.0685569 |      112 |          72 | True        |
