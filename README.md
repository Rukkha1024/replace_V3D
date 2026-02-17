# replace_V3D – COM / xCOM / BoS / MoS (OptiTrack Conventional 39 markers)

This repo contains a **pure-Python** pipeline to compute:

- **COM** (segment-based, De Leva mass fractions + simple marker-defined segment geometry)
- **XCoM / xCOM** (Hof: `xCOM = COM + vCOM / ω0`, `ω0 = sqrt(g/leg_length)`)
- **BoS** polygon (foot landmark markers → convex hull on the ground plane)
- **MoS**
  - `MOS_minDist_signed`: signed minimum distance from xCOM to BoS polygon boundary (+inside, -outside)
  - `MOS_*_v3d`: Visual3D tutorial (Closest_Bound) style — distance to the *closest* BoS bound in AP/ML
    (`MOS_AP_v3d`, `MOS_ML_v3d`, `MOS_v3d`)
  - `MOS_*_dir`: legacy alias of `MOS_*_v3d` (same values; not written to current outputs)

## Important constraints (matching your V3D tutorial logic)

- **MoS pipeline**: no forceplate usage (matches your V3D tutorial logic).
- **Ankle torque pipeline**: uses the C3D `FORCE_PLATFORM` metadata + analog channels.
- BoS uses **foot landmark markers only** (no anthropometric expansion).
- For **step trials**, analysis is reported **up to just before step onset**  
  (`analysis_end = step_onset_local - 1`).  
  No “step-onset split” and no toe-off detection.

## Quick start (single trial)

```bash
conda run -n module python main.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --out_dir output \
  --steps all
```

Outputs:
- `<trial>_MOS_preStep.xlsx` (timeseries + summary + event mapping + COM validation)
- `<trial>_JOINT_ANGLES_preStep.csv` (sign-unified + platform-onset zeroed; Δ from platform onset)
- `<trial>_ankle_torque.xlsx` (if forceplate/analog is present)

Single-trial options:
- `--steps mos`: MOS workbook only
- `--steps angles`: joint angles only
- `--steps torque`: ankle torque only

## Quick start (batch unified time series CSV: MOS + joint angles + ankle torque)

Default 실행은 batch입니다:

```bash
conda run -n module python main.py --overwrite
```

Options:
- `--skip_unmatched`: skip files with subject-token/event mapping failures.
- `--pre_frames`: local-event conversion assumption (default `100`).
- `--encoding`: CSV text encoding (default `utf-8-sig` for Korean text compatibility in Excel).

Output:
- `output/all_trials_timeseries.csv` (long-format; one row per (`subject`,`velocity`,`trial`) x `MocapFrame`)
- Key columns include `subject`, `velocity`, `trial`, `MocapFrame`, `COM_*`, `vCOM_*`, `xCOM_*`, `BOS_*`, `MOS_*`
  (notably: `MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`, `MOS_v3d`).

Notes:
- Torque requires `FORCE_PLATFORM` metadata + analog channels in the C3D.
- If forceplate extraction fails, the script **aborts** (to prevent silently mixed schemas).
- Batch export range is **full trimmed window** by default (not preStep-cut at step onset).
- Duplicate time-axis columns are avoided: the CSV keeps `MocapFrame` (and `time_from_platform_onset_s`) without redundant per-pipeline frame/time columns.
- Legacy MOS alias columns are not saved: use `MOS_AP_v3d` / `MOS_ML_v3d` (not `MOS_AP_dir` / `MOS_ML_dir`).
- By default, some metadata columns are excluded (e.g., `c3d_file`, `subject_token`, `rate_hz`, `Time_s`) to keep one unified schema.
- Joint angle columns (e.g., `Hip_*_deg`, `Knee_*_deg`, `Ankle_*_deg`, `Trunk_*_deg`, `Neck_*_deg`) are sign-unified and platform-onset zeroed (Δ from onset).
- Force/torque columns are also exported as Δ from platform onset: `GRF_*`, `GRM_*`, `AnkleTorque*`.
- COP is exported as both absolute (`COP_*_m`) and onset-zeroed displacement (`COP_*_m_onset0`).

## Grid plots (subject × velocity × variable category)

After generating `output/all_trials_timeseries.csv`, you can render grid plots for quick trial/variable sanity checks.

- Category definitions (variables + subplot layout) live in `config.yaml > plot_grid_timeseries.categories`.
- Default output dir is `output/figures/grid_timeseries/` (also configurable via `config.yaml` or `--out_dir`).
- Default x-axis is **piecewise-normalized** for readability (plot-only; does not change CSV outputs):
  - `[onset-frames, onset]` is linearly warped to a fixed length (= `frames`)
  - `[onset, offset]` is linearly warped to a fixed length (= `frames`)
  - `[offset, offset+frames]` is linearly warped to a fixed length (= `frames`)
  - Disable with `--no-x_piecewise` to plot raw seconds from platform onset.
  - Window size can be changed via `--segment_frames` (default: `100`).
- Default y-axis is **onset-zeroed** for readability (plot-only):
  - Each trial line subtracts its value at platform onset (`t=0`) so all variables start at 0.
  - Note: some columns are already onset-zeroed in the saved CSV (e.g., `*_deg`, `GRF_*`, `GRM_*`, `AnkleTorque*`).
  - Disable with `--no-y_zero_onset`.
- Event vlines: `platform_onset` (red), `platform_offset` (green), `step_onset` (blue dashed).

Sample preview (recommended):

```bash
conda run -n module python scripts/plot_grid_timeseries.py --sample
```

All subject×velocity groups:

```bash
conda run -n module python scripts/plot_grid_timeseries.py --group_by subject_velocity
```

Legacy subject-wise overlay (all velocities together):

```bash
conda run -n module python scripts/plot_grid_timeseries.py --group_by subject
```

Filters (optional):

```bash
conda run -n module python scripts/plot_grid_timeseries.py \
  --only_subjects 김우연,가윤호 \
  --only_velocities 60,70
```

## Notes

- C3D must be **trimmed** to `[platform_onset-100, platform_offset+100]` in the original 100 Hz mocap frames (as per your data rule).
- Marker naming: supports both raw labels (e.g. `251112_KUO_LASI`) and stripped labels (`LASI`).
- Library code lives under `src/replace_v3d/`, but entrypoints remain in `scripts/` (no install step).

## Joint angles (Visual3D-like 3D)

This repo also provides a **pure-Python** 3D joint angle pipeline (Visual3D-style):

- ankle / knee / hip (Left & Right)
- trunk / neck

Angle definition:
- Segment axes: **X=+Right, Y=+Anterior, Z=+Up/Proximal**
- Joint angles: **intrinsic Cardan XYZ** (reference X, floating Y, non-reference Z)
- Uses medial markers: `LShin_3/RShin_3` (knee), `LFoot_3/RFoot_3` (ankle)

```bash
conda run -n module python main.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --out_dir output \
  --steps angles
```

Outputs:
- `<trial>_JOINT_ANGLES_preStep.csv` (sign-unified + platform-onset zeroed; Δ from platform onset)
