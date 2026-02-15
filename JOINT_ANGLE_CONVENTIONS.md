## Joint Angle Output Conventions (raw vs anat vs ana0)

This repo computes Visual3D-like 3D joint angles (intrinsic XYZ) from marker-based
segment coordinate systems.

Important: the **raw joint angle computation is not modified** by the conventions
below. Conventions here are *post-processing* applied to the exported time series.

---

## Raw convention (`*_JOINT_ANGLES_preStep.csv`)

Raw output is the direct result of:

- segment frames
- relative rotation (proximal -> distal)
- intrinsic XYZ Euler decomposition

Raw output is kept stable for reproducibility and MD5 validation.

Angle columns end with `*_deg` and include:

- Hip / Knee / Ankle: left and right (e.g., `Hip_L_Y_deg`, `Hip_R_Y_deg`)
- Trunk / Neck: no side (e.g., `Trunk_Y_deg`)

---

## Anatomical presentation convention (`*_anat.csv`)

`*_anat.csv` is a post-processed copy of raw joint angles with a single goal:

Make Y/Z sign meanings consistent between LEFT and RIGHT when comparing sides.

For **Hip/Knee/Ankle** (left side only):

- `*_L_Y_deg = - *_L_Y_deg`
- `*_L_Z_deg = - *_L_Z_deg`

No baseline subtraction is performed.

### Practical interpretation after `_anat`

After the flip (using RIGHT as the reference meaning):

- **Y positive:** adduction (both L/R)
- **Z positive:** internal rotation (both L/R)

X is unchanged.

---

## Baseline-normalized convention (`*_ana0.csv`)

`*_ana0.csv` starts from the same sign-flipped angles as `_anat`, then subtracts a
quiet-standing baseline to remove static offsets.

- Baseline window: **frames 1..11** (inclusive)
- For every `*_deg` column:
  - `angle = angle - mean(angle[1..11])`

This is useful for comparing **Δangles** and for removing small static offsets due to
segment coordinate system misalignment.

---

## Where This Is Implemented

- Postprocess logic: `src/replace_v3d/joint_angles/postprocess.py`
- Single-trial export (raw + `_anat` + `_ana0`): `scripts/run_joint_angles_pipeline.py`
- Batch unified CSV optional suffix columns:
  - `--angles_anat` adds `*_deg_anat`
  - `--angles_ana0` adds `*_deg_ana0`
  - Script: `scripts/run_batch_all_timeseries_csv.py`

