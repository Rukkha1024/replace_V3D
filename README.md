# replace_V3D – COM / xCOM / BoS / MoS (OptiTrack Conventional 39 markers)

This repo contains a **pure-Python** pipeline to compute:

- **COM** (segment-based, De Leva mass fractions + simple marker-defined segment geometry)
- **XCoM / xCOM** (Hof: `xCOM = COM + vCOM / ω0`, `ω0 = sqrt(g/leg_length)`)
- **BoS** polygon (foot landmark markers → convex hull on the ground plane)
- **MoS** (signed minimum distance from xCOM to BoS polygon boundary)

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
