# replace_V3D – COM / xCOM / BoS / MoS (OptiTrack Conventional 39 markers)

This repo contains a **pure-Python** pipeline to compute:

- **COM** (segment-based, De Leva mass fractions + simple marker-defined segment geometry)
- **XCoM / xCOM** (Hof: `xCOM = COM + vCOM / ω0`, `ω0 = sqrt(g/leg_length)`)
- **BoS** polygon (foot landmark markers → convex hull on the ground plane)
- **MoS**
  - `MOS_minDist_signed`: signed minimum distance from xCOM to BoS polygon boundary (+inside, -outside)
  - `MOS_*_v3d`: Visual3D tutorial (Closest_Bound) style — distance to the *closest* BoS bound in AP/ML
    (`MOS_AP_v3d`, `MOS_ML_v3d`, `MOS_v3d`)
  - `MOS_*_dir`: backward-compatible alias of `MOS_*_v3d` (same values)
  - `MOS_*_velDir`: legacy velocity-direction switching (debug; can jump when vCOM crosses 0)

## Important constraints (matching your V3D tutorial logic)

- **MoS pipeline**: no forceplate usage (matches your V3D tutorial logic).
- **Ankle torque pipeline**: uses the C3D `FORCE_PLATFORM` metadata + analog channels.
- BoS uses **foot landmark markers only** (no anthropometric expansion).
- For **step trials**, analysis is reported **up to just before step onset**  
  (`analysis_end = step_onset_local - 1`).  
  No “step-onset split” and no toe-off detection.

## Quick start (single trial)

```bash
conda run -n module python scripts/run_mos_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --leg_length_cm 86 \
  --out_dir output
```

Outputs:
- `<trial>_MOS_preStep.xlsx` (timeseries + summary + event mapping + COM validation)

## Quick start (batch MoS time series CSV)

```bash
conda run -n module python scripts/run_batch_mos_timeseries_csv.py \
  --c3d_dir data/all_data \
  --event_xlsm data/perturb_inform.xlsm \
  --out_csv output/all_trials_mos_timeseries.csv \
  --overwrite
```

Options:
- `--skip_unmatched`: skip files with subject-token/event mapping failures.
- `--pre_frames`: local-event conversion assumption (default `100`).
- `--encoding`: CSV text encoding (default `utf-8-sig` for Korean text compatibility in Excel).

Output:
- `output/all_trials_mos_timeseries.csv` (long-format; one row per `subject-velocity-trial` x `MocapFrame`)
- Key columns include `subject-velocity-trial`, `MocapFrame`, `COM_*`, `vCOM_*`, `xCOM_*`, `BOS_*`, `MOS_*`
  (notably: `MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`, `MOS_v3d`).

## Quick start (batch unified time series CSV: MOS + joint angles + ankle torque)

This export is also **long-format** (row = (`subject`,`velocity`,`trial`) x `MocapFrame`), but includes:

- COM / vCOM / xCOM / BOS / MOS
- Visual3D-like 3D joint angles (ankle/knee/hip/trunk/neck)
- Forceplate-based ankle torque time series (GRF/GRM/COP + joint moments)

```bash
conda run -n module python scripts/run_batch_all_timeseries_csv.py \
  --c3d_dir data/all_data \
  --event_xlsm data/perturb_inform.xlsm \
  --out_csv output/all_trials_timeseries.csv \
  --overwrite
```

Notes:
- Torque requires `FORCE_PLATFORM` metadata + analog channels in the C3D.
- If forceplate extraction fails, the script **aborts** (to prevent silently mixed schemas).
- Duplicate time-axis columns are avoided: the CSV keeps `MocapFrame` (and `time_from_platform_onset_s`) without redundant per-pipeline frame/time columns.
- By default, some metadata columns are excluded (e.g., `subject-velocity-trial`, `c3d_file`, `subject_token`, `rate_hz`, `Time_s`).

## Quick start (ankle torque)

```bash
conda run -n module python scripts/run_ankle_torque_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --out_dir output
```

Outputs:
- `<trial>_ankle_torque.xlsx` (GRF/GRM → ankle torque at L/R and mid)

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
conda run -n module python scripts/run_joint_angles_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --out_dir output
```

Outputs:
- `<trial>_JOINT_ANGLES_preStep.xlsx`
- `<trial>_JOINT_ANGLES_preStep.csv` (for MD5 validation)
