# Issue 003: Make inverse dynamics forceplate selection config-driven

**Status**: In Progress
**Created**: 2026-03-12

## Background

The current inverse-dynamics path in `replace_V3D` assumes a single active forceplate and can auto-pick it from the largest vertical force.
That behavior is unsafe for this repository because the user needs an explicit forceplate subset such as `[fp1, fp3]`, and lower-limb torque or joint moment outputs must be skipped when only one plate is selected.
This task replaces the auto-select path with a config-driven selection policy and adds a strict single-plate versus multi-plate branch.

## Acceptance Criteria

- [ ] `config.yaml` defines `forceplate.analysis.use_for_inverse_dynamics`, and invalid values fail fast.
- [ ] `scripts/run_batch_all_timeseries_csv.py` no longer offers CLI `--force_plate` and uses only the config-defined plate subset.
- [ ] Single-plate selection keeps raw GRF/GRM/COP but exports lower-limb torque and joint moment columns as NaN.
- [ ] Multi-plate selection uses only the selected plates for lower-limb ankle torque and joint moment computation.
- [ ] The batch script runs in `conda run -n module` and MD5 comparison artifacts are recorded for the validation run.

## Tasks

- [x] 1. Review `plan.md`, the current inverse-dynamics path, and affected files.
- [x] 2. Implement config-driven forceplate loading and single/multi inverse-dynamics branching.
- [x] 3. Update user-facing documentation and execution plan records.
- [x] 4. Run validation, compare MD5 against a reference output, and review the diff before commit.

## Notes

- The working implementation path touches `scripts/run_batch_all_timeseries_csv.py`, `src/replace_v3d/torque/forceplate.py`, and `src/replace_v3d/joint_dynamics/inverse_dynamics.py`.
- The CLI policy is fixed: `--force_plate` is removed, and `forceplate.analysis.use_for_inverse_dynamics` is the only selection source.
- Validation run completed with `conda run -n module python main.py --overwrite --skip_unmatched --out_dir output/qc_forceplate_selection --out_csv output/qc_forceplate_selection/all_trials_timeseries.csv --md5_reference_dir output/qc_forceplate_reference`.
- The reference MD5 was `a83964c58132129c14078b17ed3b01a6`, and the new output MD5 was `e041d7b298c3dfa3ba7770e091f6a7d6`, which is an expected diff because strict single-plate mode now suppresses lower-limb torque and joint moment outputs.
