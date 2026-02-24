# replace_V3D

A pure-Python biomechanical analysis pipeline for computing center of mass (COM), extrapolated center of mass (xCOM), base of support (BoS), margin of stability (MoS), three-dimensional joint angles, ground reaction forces (GRF), center of pressure (COP), and ankle torque from motion capture and force plate data. This pipeline was developed as a replacement for Visual3D, processing OptiTrack Conventional 39-marker set data acquired during a posterior support-surface translation perturbation task.

## 1. Participants and Experimental Protocol

Twenty-four healthy young adults participated in the study. Each participant stood on a moveable platform that delivered unexpected posterior translations at velocities ranging from 60 to 135 cm/s, presented in a mixed order. Postural responses were classified into two categories based on the recovery strategy employed: **step** trials (53 trials), in which participants took a compensatory step, and **non-step** trials (72 trials), in which balance was recovered without stepping. A total of 125 trials were included after applying the following inclusion criteria:

- Mixed velocity condition only (`mixed = 1`)
- Young adult participants (`age_group = young`)
- Step trials restricted to ipsilateral stepping (i.e., right-dominant participants stepping right, left-dominant stepping left)
- All non-step trials included regardless of dominance

Trial metadata including perturbation velocity, response classification, and participant demographics were recorded in `perturb_inform.xlsm`.

## 2. Data Acquisition

Whole-body kinematics were captured using an OptiTrack motion capture system with the **Conventional 39-marker set** at a sampling rate of 100 Hz. Analog signals (forces and moments) were recorded simultaneously at 1000 Hz via embedded force plates. All data were stored in C3D format.

Each C3D file was pre-trimmed to a window of `[platform_onset - 100, platform_offset + 100]` frames in 100 Hz mocap time prior to processing. Event timings (platform onset, platform offset, step onset) were manually annotated and stored in an Excel workbook (`perturb_inform.xlsm`).

## 3. Data Processing Pipeline

### 3.1 C3D Reading and Marker Extraction

Marker trajectories were extracted from C3D files using a Python-based reader (`src/replace_v3d/io/c3d_reader.py`). Both raw OptiTrack labels (e.g., `251112_KUO_LASI`) and stripped labels (e.g., `LASI`) were supported through automatic label normalization.

### 3.2 Joint Center Estimation

Joint centers were estimated from marker positions as follows:

| Joint | Method |
|-------|--------|
| Hip | Harrington et al. (2007) regression from pelvis width and depth |
| Knee | Midpoint of lateral (LKNE/RKNE) and medial (LShin_3/RShin_3) markers |
| Ankle | Midpoint of lateral (LANK/RANK) and medial (LFoot_3/RFoot_3) markers |
| Elbow | Midpoint of lateral (LELB/RELB) and medial (LUArm_3/RUArm_3) markers |
| Wrist | Midpoint of LWRA/RWRA and LWRB/RWRB |

Hip joint centers were estimated using the Harrington (2007) regression equations, in which pelvis width (PW, the distance between RASI and LASI) and pelvis depth (PD, the distance between the pelvis origin and the midpoint of the posterior superior iliac spines) served as predictors.

### 3.3 Whole-Body Center of Mass

The whole-body COM was computed as the weighted sum of 14 segment COMs using the mass fractions and COM placement ratios of De Leva (1996):

`COM = sum(m_i * COM_i)`

where `m_i` denotes the mass fraction and `COM_i` denotes the position of each segment's center of mass, estimated as an interpolated point between the proximal and distal joint centers at a segment-specific fraction.

| Segment | Mass Fraction | COM Fraction (prox to dist) |
|---------|--------------|----------------------------|
| Head | 0.0694 | (via C7 + head center) |
| Trunk | 0.4346 | 0.797 (pelvis origin to thorax ref) |
| Upper arm (x2) | 0.0271 | 0.436 |
| Forearm (x2) | 0.0162 | 0.430 |
| Hand (x2) | 0.0061 | 0.506 |
| Thigh (x2) | 0.1416 | 0.433 |
| Shank (x2) | 0.0433 | 0.433 |
| Foot (x2) | 0.0137 | 0.500 |

The thorax reference point was defined as a weighted combination of C7 and the sternum marker: `thorax_ref = 0.56 * C7 + 0.44 * STRN`. Trunk COM was placed at 79.7% of the distance from the pelvis origin to this thorax reference. Head COM was located at 100% of the distance from C7 to the head center (average of LFHD, RFHD, LBHD, RBHD).

The resulting COM trajectory was low-pass filtered using a 4th-order zero-phase Butterworth filter with a cutoff frequency of 6 Hz.

### 3.4 COM Velocity and Extrapolated Center of Mass

COM velocity (vCOM) was computed as the central-difference derivative of the filtered COM position:

`vCOM = d(COM) / dt`

The extrapolated center of mass (xCOM) was computed following Hof et al. (2005):

`xCOM = COM + vCOM / sqrt(g / l)`

where `g = 9.81 m/s^2` and `l` denotes participant leg length obtained from the metadata file.

### 3.5 Base of Support

The BoS was defined at each frame as the convex hull of eight foot landmark markers projected onto the ground plane (XY):

`LHEE, LTOE, LANK, LFoot_3, RHEE, RTOE, RANK, RFoot_3`

No anthropometric foot expansion was applied; the BoS boundary was determined solely by the marker positions. Axis-aligned bounds (minX, maxX, minY, maxY) and polygon area were extracted from the convex hull at each frame.

### 3.6 Margin of Stability

Three MoS definitions were computed:

| Variable | Definition |
|----------|-----------|
| `MOS_minDist_signed` | Signed minimum distance from xCOM to the BoS polygon boundary; positive indicates xCOM inside the BoS, negative indicates outside |
| `MOS_AP_v3d` | Distance to the closest anterior-posterior BoS bound: `min(xCOM_X - minX, maxX - xCOM_X)` |
| `MOS_ML_v3d` | Distance to the closest mediolateral BoS bound: `min(xCOM_Y - minY, maxY - xCOM_Y)` |

The `MOS_AP_v3d` and `MOS_ML_v3d` definitions follow the closest-bound approach described in the Visual3D documentation. An overall closest-boundary variable (`MOS_v3d = min(MOS_AP_v3d, MOS_ML_v3d)`) was also computed.

### 3.7 Joint Angles

Three-dimensional joint angles were computed for five joints (hip, knee, ankle, trunk, neck), bilaterally where applicable, using an intrinsic XYZ (Cardan) decomposition. Segment coordinate systems were constructed from anatomical marker clusters using a right-handed frame convention.

All joint angle time series were expressed as change from platform onset (onset-zeroed).

### 3.8 Kinetic Variables

Ground reaction forces (GRF) and moments (GRM) were extracted from the C3D force platform metadata and analog channels. The active force plate was identified based on the maximum vertical force criterion. Force and moment data were transformed to the Stage01 biomechanical coordinate system (Z-up, subject-centered horizontal axes).

**Center of pressure** was computed from the transformed force and moment data:

`COP_X = -M_Y / F_Z` , `COP_Y = M_X / F_Z`

**Ankle torque** was computed by transferring the net ground reaction moment from the force plate origin to the ankle joint center:

`M_ankle = M_origin + (r_origin - r_ankle) x F`

where internal torque was defined as the negation of the external moment. Mid-ankle (average of left and right ankle centers), left ankle, and right ankle reference points were used. Body-mass-normalized sagittal ankle torque (Nm/kg) was also computed.

An inertial correction procedure was applied using pre-computed quiet-standing templates to subtract platform-acceleration artifacts from the measured forces and moments.

All kinetic variables (GRF, COP displacement, GRM, ankle torque) were expressed as change from platform onset (onset-zeroed). COP was additionally exported in absolute coordinates.

### 3.9 Signal Processing Summary

| Parameter | Value |
|-----------|-------|
| Mocap sampling rate | 100 Hz |
| Analog sampling rate | 1000 Hz |
| COM low-pass filter | 6 Hz, 4th-order Butterworth, zero-phase |
| Derivative method | Central difference |
| Joint angle decomposition | Intrinsic XYZ Cardan |
| Onset zeroing | Subtract value at platform onset frame |

## 4. Coordinate System and Sign Conventions

All variables were expressed in the following laboratory coordinate system:

| Axis | Direction | Positive (+) | Negative (-) |
|------|-----------|-------------|-------------|
| X | Anteroposterior (AP) | Anterior (forward) | Posterior (backward) |
| Y | Mediolateral (ML) | Right (lateral) | Left (medial) |
| Z | Vertical | Upward | Downward |

Joint angle sign conventions:

| Joint | X-axis (+/-)  | Y-axis (+/-) | Z-axis (+/-) |
|-------|--------------|-------------|-------------|
| Hip | Flexion / Extension | Adduction / Abduction | Internal rot. / External rot. |
| Knee | Flexion / Extension | Adduction / Abduction | Internal rot. / External rot. |
| Ankle | Dorsiflexion / Plantarflexion | Adduction / Abduction | Internal rot. / External rot. |
| Trunk | Flexion / Extension | Adduction / Abduction | Internal rot. / External rot. |
| Neck | Flexion / Extension | Adduction / Abduction | Internal rot. / External rot. |

## 5. Time Axis and Normalization

Two time representations were maintained:

- **MocapFrame** (100 Hz): sequential frame index within the trimmed C3D file
- **original_DeviceFrame** (1000 Hz): absolute device frame number, preserved as provenance

Exported CSV files retain the raw time axis (`MocapFrame`, `time_from_platform_onset_s`). Piecewise time normalization was applied only for visualization (grid plots), in which the pre-onset and post-onset segments were each linearly warped to a fixed number of frames. This normalization did not affect the exported data.

## 6. Statistical Analysis

### 6.1 Step vs. Non-Step Comparison (Pre-Step Window)

To examine whether pre-step biomechanical responses differed between stepping and non-stepping strategies, linear mixed models (LMMs) were fitted for 34 dependent variables across three categories:

**Balance and stability (17 DVs):** COM range and path length (AP, ML), vCOM peak (AP, ML), COP range, path length, and peak velocity (AP, ML), MoS minimum values (`MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`), and xCOM-to-BoS distance at platform onset and step onset.

**Joint angles (10 DVs):** Range of motion and peak values for hip, knee, ankle (stance-equivalent side), trunk, and neck in the sagittal plane.

**Force and torque (7 DVs):** GRF peak and range (AP, ML, vertical) and sagittal ankle torque peak.

The analysis window was defined as `[platform_onset, step_onset]`. For step trials, the actual step onset frame was used. For non-step trials, the mean step onset of the corresponding subject-velocity group was substituted.

Each model took the form: `DV ~ step_TF + (1 | subject)`, estimated with REML. Multiple comparisons were corrected using the Benjamini-Hochberg false discovery rate (BH-FDR) procedure at alpha = 0.05.

### 6.2 Initial Posture Strategy Analysis (Onset Frame)

To assess whether body configuration at the moment of perturbation onset predicted the subsequent balance recovery strategy, LMMs were fitted for 19 dependent variables extracted at the single platform-onset frame:

**Balance (8 DVs):** COM position (AP, ML), vCOM (AP, ML), MoS (`MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`), and normalized xCOM-to-BoS distance.

**Joint angles at onset (5 DVs):** Absolute (non-onset-zeroed) sagittal-plane angles for hip, knee, ankle (stance-equivalent side), trunk, and neck.

**Force variables at onset (6 DVs):** Absolute COP position (AP, ML), GRF (AP, ML, vertical), and body-mass-normalized sagittal ankle torque.

Each model took the same form: `DV ~ step_TF + (1 | subject)`, with BH-FDR correction at alpha = 0.05.

## 7. Pipeline Execution

All scripts were executed within the `module` conda environment:

```bash
conda run -n module python <script>
```

### 7.1 Batch Export

The full pipeline (marker extraction, joint center estimation, COM/xCOM/BoS/MoS computation, joint angles, GRF/COP/torque) was executed as a single batch process:

```bash
conda run -n module python main.py --overwrite
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--c3d_dir` | `data/all_data` | Directory containing trimmed C3D files |
| `--event_xlsm` | `data/perturb_inform.xlsm` | Event metadata Excel workbook |
| `--out_csv` | `output/all_trials_timeseries.csv` | Output CSV path |
| `--overwrite` | (flag) | Overwrite existing output CSV |
| `--skip_unmatched` | (flag) | Skip trials with unresolved subject/event mapping |
| `--pre_frames` | `100` | Frame buffer for local-absolute conversion |
| `--encoding` | `utf-8-sig` | CSV encoding (BOM for Korean Excel compatibility) |
| `--on_error` | `continue` | Error handling: `continue` or `abort` |
| `--md5_reference_dir` | (none) | Reference directory for MD5 checksum validation |

### 7.2 Grid Visualization

Time-series grid plots (subject x velocity x variable category) were generated from the exported CSV:

```bash
# Sample preview (3 subjects, 2 velocities)
conda run -n module python scripts/plot_grid_timeseries.py --sample

# All subject-velocity groups
conda run -n module python scripts/plot_grid_timeseries.py --group_by subject_velocity

# Subject-wise overlay
conda run -n module python scripts/plot_grid_timeseries.py --group_by subject

# Filtered subsets
conda run -n module python scripts/plot_grid_timeseries.py \
  --only_subjects subject1,subject2 \
  --only_velocities 60,70
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--group_by` | `subject_velocity` | Grouping mode: `subject_velocity`, `subject`, or `total_mean` |
| `--sample` | (flag) | Generate preview only (3 subjects, 2 velocities) |
| `--dpi` | `300` | Figure resolution |
| `--segment_frames` | `100` | Window size around event (frames) |
| `--x_piecewise` | enabled | Piecewise time normalization for display |
| `--y_zero_onset` | enabled | Subtract value at platform onset per trial |
| `--separate_step_nonstep` | disabled | Separate figures for step vs. non-step |

Event reference lines were overlaid: platform onset (red), platform offset (green), and step onset (blue dashed).

### 7.3 Statistical Analysis

```bash
# Step vs. non-step LMM (pre-step window, 34 DVs)
conda run -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py

# Initial posture strategy LMM (onset frame, 19 DVs)
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

## 8. Output Description

### 8.1 Time-Series CSV

The primary output (`output/all_trials_timeseries.csv`) was structured in long format with one row per trial per mocap frame. Column families are summarized below:

| Family | Columns | Unit | Onset-Zeroed |
|--------|---------|------|-------------|
| Identifiers | `subject`, `velocity`, `trial` | — | — |
| Time | `MocapFrame`, `time_from_platform_onset_s` | frame, s | — |
| Events | `platform_onset_local`, `platform_offset_local`, `step_onset_local` | frame | — |
| COM | `COM_X`, `COM_Y`, `COM_Z` | m | No |
| vCOM | `vCOM_X`, `vCOM_Y`, `vCOM_Z` | m/s | No |
| xCOM | `xCOM_X`, `xCOM_Y`, `xCOM_Z` | m | No |
| BoS | `BOS_area`, `BOS_minX`, `BOS_maxX`, `BOS_minY`, `BOS_maxY` | m, m^2 | No |
| MoS | `MOS_minDist_signed`, `MOS_AP_v3d`, `MOS_ML_v3d`, `MOS_v3d` | m | No |
| Joint angles | `Hip_L_X_deg`, ..., `Neck_Z_deg` | deg | Yes |
| GRF | `GRF_X_N`, `GRF_Y_N`, `GRF_Z_N` | N | Yes |
| COP | `COP_X_m`, `COP_Y_m`, `COP_X_m_onset0`, `COP_Y_m_onset0` | m | Both |
| GRM | `GRM_X_Nm`, `GRM_Y_Nm`, `GRM_Z_Nm` | Nm | Yes |
| Ankle torque | `AnkleTorqueMid_*_Nm`, `AnkleTorqueL_*_Nm`, `AnkleTorqueR_*_Nm` | Nm | Yes |

### 8.2 Visualizations

Grid plots were saved to `output/figures/grid_timeseries/`, organized by six variable categories: MoS/BoS, COM family, lower-limb joint angles, upper-body joint angles, ankle torque, and GRF/COP.

### 8.3 Analysis Outputs

LMM results, forest plots, violin plots, and descriptive heatmaps were saved to their respective analysis subdirectories under `analysis/`.

## 9. Project Structure

```
replace_V3D/
├── main.py                          # Pipeline entry point
├── config.yaml                      # Signal processing & plot configuration
├── data/
│   ├── all_data/                    # Trimmed C3D files
│   └── perturb_inform.xlsm         # Event timing & participant metadata
├── scripts/
│   ├── run_batch_all_timeseries_csv.py    # Core batch processor
│   ├── plot_grid_timeseries.py            # Grid visualization
│   ├── apply_post_filter_from_meta.py     # Metadata-based trial filtering
│   └── torque_build_fp_inertial_templates.py  # Inertial template generation
├── src/replace_v3d/
│   ├── cli/                         # Trial matching, CSV export utilities
│   ├── com/                         # COM, joint centers (De Leva, Harrington)
│   ├── geometry/                    # 2D convex hull, polygon operations
│   ├── io/                          # C3D reader, event/metadata loading
│   ├── joint_angles/                # 3D joint angle computation
│   ├── mos/                         # MoS & BoS computation
│   ├── signal/                      # Butterworth filter, baseline zeroing
│   └── torque/                      # Force plate, COP, ankle torque, axis transform
├── analysis/
│   ├── step_vs_nonstep_lmm/        # Pre-step window LMM (34 DVs)
│   └── initial_posture_strategy_lmm/ # Onset-frame LMM (19 DVs)
└── output/
    ├── all_trials_timeseries.csv    # Primary output (long-format)
    └── figures/grid_timeseries/     # Grid visualization plots
```

## 10. References

- De Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223-1230.
- Harrington, M. E., Zavatsky, A. B., Lawson, S. E. M., Yuan, Z., & Theologis, T. N. (2007). Prediction of the hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance imaging. *Journal of Biomechanics*, 40(3), 595-602.
- Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). The condition for dynamic stability. *Journal of Biomechanics*, 38(1), 1-8.
- Van Wouwe, T., Afschrift, M., De Groote, F., & Vanwanseele, B. (2021). Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance. *Journal of Neurophysiology*, 125(5), 1983-1998.
