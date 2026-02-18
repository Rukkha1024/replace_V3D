# Restore Ankle Z Angle by Replacing Shank Axis Lock with Visual3D Method 2 Plane-Fit

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `.codex/PLANS.md` from the repository root.

## Purpose / Big Picture

EN: After this change, ankle internal/external rotation (`Ankle_L_Z_deg`, `Ankle_R_Z_deg`) is no longer structurally collapsed to near-zero micro-degree values. Users can verify this by regenerating joint-angle outputs and checking that ankle Z traces have meaningful range in CSV and in `plot_grid_timeseries` sample figures.

KO: 이 변경 후 발목 내/외회전(`Ankle_L_Z_deg`, `Ankle_R_Z_deg`)이 구조적으로 0 근처(마이크로 degree)로 붕괴되지 않는다. 사용자는 관절각 결과를 재생성한 뒤 CSV 범위와 `plot_grid_timeseries` 샘플 그림에서 ankle Z 파형의 유의미한 변화를 확인해 성공을 검증할 수 있다.

## Progress

- [x] (2026-02-18T12:29Z) 기존 동작 기준(reference) 산출물 생성: 단일 trial JOINT_ANGLES CSV 및 sample grid figure.
- [x] (2026-02-18T12:31Z) `src/replace_v3d/joint_angles/v3d_joint_angles.py`에 LS plane normal helper 추가 및 shank frame 정의를 Method 2 방식으로 교체.
- [x] (2026-02-18T12:35Z) 수정 후 파이프라인 재실행 및 ankle Z 범위 정량 검증 수행 (`Ankle_L_Z_deg ptp: 0.000001→18.470310`, `Ankle_R_Z_deg ptp: 0.000003→7.053579`).
- [x] (2026-02-18T12:35Z) sample grid figure 재생성 및 시각 검증 수행 (`output/figures/grid_timeseries/joint_angles_lower__subject-강비은__velocity-30__sample.png`).
- [x] (2026-02-18T12:35Z) reference/new MD5 비교 파일 작성 (`output/qc/ankle_z_fix/md5/md5_ankle_z_fix_ref_vs_new.tsv`).
- [x] (2026-02-18T12:36Z) 확장 검증 완료: `가윤호/강비은/권유영`의 trial 집합에서 ankle Z ptp 최소값이 모두 1 deg 이상.
- [x] (2026-02-18T12:39Z) `.codex/issue.md` 문제 기록, `$replace-v3d-troubleshooting` 해결책 기록, 한국어 3줄 커밋 메시지로 커밋.

## Surprises & Discoveries

- Observation: 기존 워크트리에 이번 작업과 무관한 수정(`.claude/PLANS.md`, `.codex/PLANS.md`, `gpt_plan.md`)이 이미 존재한다.
  Evidence: `git status --short`.

- Observation: `conda run -n module python - <<'PY'` 형태는 현재 셸에서 stdin 스크립트 실행/출력 캡처가 안정적이지 않았다.
  Evidence: heredoc 실행 시 출력/파일 생성이 발생하지 않았고, `-c` 실행은 정상 동작.

- Observation: Batch 재생성(`main.py --overwrite --skip_unmatched`)에서 기존에도 알려진 데이터 품질/마커 이슈가 재현되었다.
  Evidence: `251128_방주원_perturb_200_005.c3d`는 `Required marker not found: T10`으로 skip, 일부 파일은 forceplate inertial subtract QC 경고 출력.

## Decision Log

- Decision: 버그 유형을 "Bug Fix(교체)"로 분류하고 기존 shank xhint 잠금 로직을 유지하지 않는다.
  Rationale: AGENTS.md의 "bug fixes must replace old logic" 규칙과 문제 원인(축 잠금) 제거 목적에 부합.
  Date/Author: 2026-02-18 / Codex

- Decision: foot frame 정의는 변경하지 않는다.
  Rationale: 범위를 ankle Z 붕괴 원인(shank 정의)으로 제한해 영향 범위를 최소화.
  Date/Author: 2026-02-18 / Codex

- Decision: 검증은 단일 trial + sample figure + 배치 CSV 재생성 기반의 확장 체크를 함께 수행한다.
  Rationale: 수치와 시각을 동시에 확인하고, 기존 `output/all_trials_timeseries.csv` 소비 경로와의 정합성을 검증하기 위함.
  Date/Author: 2026-02-18 / Codex

## Outcomes & Retrospective

EN: Core code replacement and verification are complete. The ankle Z collapse was removed in both single-trial and batch-driven outputs, and MD5 reports confirm expected output changes. Remaining operational step is documentation finalization and commit.

KO: 핵심 코드 교체와 검증이 완료되었다. 단일 trial과 batch 기반 출력 모두에서 ankle Z 붕괴가 해소되었고, MD5 리포트로 출력 변경이 확인되었다. 남은 작업은 문서 기록 마무리와 커밋이다.

## Context and Orientation

EN: The bug originates in `src/replace_v3d/joint_angles/v3d_joint_angles.py::build_segment_frames()`. Previously, shank X was built from ankle medial/lateral markers (`LFoot_3-LANK`, `RANK-RFoot_3`), which mirrors foot X input and suppresses ankle relative Z rotation. This plan replaces shank frame construction with a Visual3D Method 2 style approach using four border targets at knee and ankle to fit a per-frame plane. The helper and frame builder in this module are the only logic changes. All entrypoints (`main.py`, `scripts/plot_grid_timeseries.py`) stay unchanged.

KO: 버그 원인은 `src/replace_v3d/joint_angles/v3d_joint_angles.py::build_segment_frames()`에 있다. 기존에는 shank X를 발목 내/외측 마커(`LFoot_3-LANK`, `RANK-RFoot_3`)로 구성해 foot X와 동일 정보를 공유했고, 그 결과 ankle 상대회전 Z가 억제되었다. 본 계획은 무릎/발목의 4개 border target으로 프레임별 평면을 적합하는 Visual3D Method 2 스타일로 shank frame을 교체한다. 로직 변경은 해당 모듈의 helper/frame 생성부에 한정되며, 엔트리포인트(`main.py`, `scripts/plot_grid_timeseries.py`)는 변경하지 않는다.

## Plan of Work

EN: Add `_fit_plane_normal_ls(P)` to compute per-frame least-squares plane normals via SVD. In shank construction, compute `y0_shank_L/R` from `[knee_lat, knee_med, ankle_lat, ankle_med]` points, keep `z0_shank` as `kneeJC - ankleJC`, then use `_frame_from_yz(..., right_hint=...)` to enforce +Right orientation. Keep foot frame logic unchanged. Run syntax checks, regenerate outputs, validate ankle Z ranges, render sample figure, and compare MD5 against pre-change references.

KO: 프레임별 최소자승 평면 법선을 SVD로 계산하는 `_fit_plane_normal_ls(P)`를 추가한다. shank 구성에서는 `[knee_lat, knee_med, ankle_lat, ankle_med]` 4점으로 `y0_shank_L/R`를 만들고, `z0_shank`는 `kneeJC - ankleJC`를 유지한 채 `_frame_from_yz(..., right_hint=...)`로 +Right 방향을 강제한다. foot frame은 그대로 유지한다. 이후 문법 검사, 출력 재생성, ankle Z 범위 검증, 샘플 그림 생성, 사전 reference 대비 MD5 비교를 수행한다.

## Concrete Steps

All commands are run from repository root:
`/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/replace_V3D`

1. Generate pre-change references.

    conda run -n module python main.py --c3d data/all_data/251128_강비은_perturb_30_004.c3d --event_xlsm data/perturb_inform.xlsm --out_dir output/qc/ankle_z_fix/ref --steps angles

    conda run -n module python scripts/plot_grid_timeseries.py --sample --only_subjects 강비은 --only_velocities 30 --out_dir output/qc/ankle_z_fix/ref_fig

2. Apply code change in `src/replace_v3d/joint_angles/v3d_joint_angles.py`.

3. Run syntax check.

    conda run -n module python -m py_compile src/replace_v3d/joint_angles/v3d_joint_angles.py

4. Regenerate post-change outputs (single-trial + batch CSV).

    conda run -n module python main.py --c3d data/all_data/251128_강비은_perturb_30_004.c3d --event_xlsm data/perturb_inform.xlsm --out_dir output/qc/ankle_z_fix/new --steps angles

    conda run -n module python main.py --overwrite --skip_unmatched

5. Render post-change sample figure.

    conda run -n module python scripts/plot_grid_timeseries.py --sample --only_subjects 강비은 --only_velocities 30 --out_dir output/qc/ankle_z_fix/new_fig

6. Compare ankle Z ranges and MD5.

    conda run -n module python scripts/qc_compare_ankle_z_fix.py

7. Update issue/skill records and commit.

## Validation and Acceptance

EN:
- `py_compile` must pass.
- For trial `강비은-30-004`, `Ankle_L_Z_deg` and `Ankle_R_Z_deg` in post-change CSV must have larger peak-to-peak range than pre-change CSV and not remain micro-degree flat.
- In post-change figure `output/qc/ankle_z_fix/new_fig/joint_angles_lower__subject-강비은__velocity-30__sample.png`, ankle Z traces are visibly non-flat.
- Batch CSV regeneration completes with `--skip_unmatched`, and sample rendering succeeds from updated `output/all_trials_timeseries.csv`.
- MD5 report file (`output/qc/ankle_z_fix/md5/md5_ankle_z_fix_ref_vs_new.tsv`) exists and records reference/new hashes for key artifacts.

KO:
- `py_compile`가 통과해야 한다.
- `강비은-30-004` trial의 post-change CSV에서 `Ankle_L_Z_deg`, `Ankle_R_Z_deg` peak-to-peak가 pre-change 대비 증가하고 마이크로 degree 평탄 상태가 아니어야 한다.
- post-change 그림 `output/qc/ankle_z_fix/new_fig/joint_angles_lower__subject-강비은__velocity-30__sample.png`에서 ankle Z 파형이 육안상 평탄하지 않아야 한다.
- `--skip_unmatched`로 batch CSV 재생성이 완료되고, 갱신된 `output/all_trials_timeseries.csv`로 sample 렌더가 성공해야 한다.
- MD5 리포트(`output/qc/ankle_z_fix/md5/md5_ankle_z_fix_ref_vs_new.tsv`)가 생성되어 핵심 산출물의 reference/new 해시를 기록해야 한다.

## Idempotence and Recovery

EN: All commands are rerunnable. Output directories under `output/qc/ankle_z_fix/` are intentionally overwritten for deterministic comparison. If batch fails on problematic files, rerun with `--skip_unmatched` (already selected) and keep the failure context in `.codex/issue.md` if newly observed.

KO: 모든 명령은 재실행 가능하다. `output/qc/ankle_z_fix/` 하위는 비교를 위해 의도적으로 덮어쓴다. batch가 일부 파일에서 실패하면 `--skip_unmatched`(본 계획 기본값)로 재실행하고, 새로 관찰된 실패 현상은 `.codex/issue.md`에 문제로 기록한다.

## Artifacts and Notes

EN: Keep proof artifacts under `output/qc/ankle_z_fix/`:
- `ref/`, `new/` single-trial CSV outputs
- `ref_fig/`, `new_fig/` sample figures
- `md5/` reports

KO: 검증 근거 산출물은 `output/qc/ankle_z_fix/`에 유지한다.
- 단일 trial CSV: `ref/`, `new/`
- sample figure: `ref_fig/`, `new_fig/`
- MD5 리포트: `md5/`

## Interfaces and Dependencies

EN:
- Modified module: `src/replace_v3d/joint_angles/v3d_joint_angles.py`
- Added internal helper:
  `def _fit_plane_normal_ls(P: np.ndarray) -> np.ndarray`
- Replaced shank frame construction in `build_segment_frames()`:
  from `_frame_from_xz(xhint_shank_*, z0_shank_*)`
  to `_frame_from_yz(y0_shank_*, z0_shank_*, right_hint=...)`
- Runtime commands use existing entrypoints only:
  `main.py`, `scripts/plot_grid_timeseries.py`, `scripts/qc_compare_ankle_z_fix.py`

KO:
- 수정 모듈: `src/replace_v3d/joint_angles/v3d_joint_angles.py`
- 추가 내부 helper:
  `def _fit_plane_normal_ls(P: np.ndarray) -> np.ndarray`
- `build_segment_frames()`의 shank 프레임 구성 교체:
  `_frame_from_xz(xhint_shank_*, z0_shank_*)` → `_frame_from_yz(y0_shank_*, z0_shank_*, right_hint=...)`
- 실행은 기존 엔트리포인트 사용:
  `main.py`, `scripts/plot_grid_timeseries.py`, `scripts/qc_compare_ankle_z_fix.py`

---
Revision note (2026-02-18 / Codex): Updated progress/discoveries/outcomes with completed validation (single-trial, batch, figure, MD5, extended subject checks).
