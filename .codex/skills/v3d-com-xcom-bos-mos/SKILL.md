---
name: v3d-com-xcom-bos-mos
description: Compute COM, XCoM (Hof), foot-landmark BoS polygon, and MoS using Visual3D-equivalent logic from C3D marker data (no force plate required).
---

## When to use

Use this skill when you need to compute any of:

- Whole-body **COM** (and COM velocity) from C3D marker trajectories
- **XCoM** / eXtrapolated Center of Mass (Hof inverted-pendulum)
- Foot-landmark-based **BoS** polygon (single or double support) **without force plates**
- Signed **MoS/MOS** as the shortest distance from XCoM to the BoS boundary

This skill is designed to work with the marker-set guidance from:

- `motive-marker-v3d-guideline`

## Hard requirements (ask the user if missing)

Before running code, confirm you have:

1. C3D file(s)
2. Subject anthropometrics (minimum)
   - leg length
   - foot length
   - foot width
   - ankle width
   - knee width
3. Event info (choose one)
   - `perturb_inform.xlsm` (recommended)
   - OR explicit platform onset/offset + step onset frames

If **anthropometrics are not provided**, you must request them. This project requires anthropometrics to generate BoS landmarks and XCoM.

## Outputs

Typical outputs per trial:

- `*_COM.csv` (COM position)
- `*_XCOM.csv` (XCoM position)
- `*_BOS_vertices.csv` (BoS hull vertices per frame)
- `*_MOS.csv` (signed MoS time series)
- Optional: `*_COM_validation.txt` (correlation vs V3D COM export)

## Visual3D-equivalent method (conceptual)

Mirror the canonical V3D tutorial logic:

1. **COM** and **COM velocity** from a kinematic model (or a validated proxy)
2. **XCoM** computed from COM and COM velocity using Hof’s inverted-pendulum scaling
3. **BoS** derived from foot landmarks (measured or virtual) to form a polygon (single/double support)
4. **MoS** as the shortest distance from XCoM to the BoS boundary (signed)

See references:

- `references/v3d-logic-summary.md`
- `references/anthropometrics.md`

## Scripted pipeline (recommended)

### Quick start (single trial)

```bash
python scripts/run_mos_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --events /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --out_dir ./outputs \
  --anthro ./assets/example_anthro_kimwooyeon.yaml \
  --v3d_com_ref /path/to/251112_KUO_perturb_60_001_COM.xlsx
```

### Batch mode (folder)

```bash
python scripts/run_mos_pipeline.py \
  --c3d_dir /path/to/c3d_folder \
  --events /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --out_dir ./outputs \
  --anthro ./assets/example_anthro_kimwooyeon.yaml
```

## Implementation notes

- The pipeline assumes C3D marker trajectories are already labeled/filtered.
- No force plate is required; if force channels exist, they are ignored.
- Coordinate axes can vary; the script allows configuring the vertical axis and ground plane.

