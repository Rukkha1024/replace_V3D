# Add BOS Hull/Union Overlay to GIF Without CLI Changes

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `.codex/PLANS.md` from the repository root.

## Purpose / Big Picture

EN: After this change, users can run the existing `scripts/plot_bos_com_xy_sample.py` command and see two additional BOS overlays in GIF output: (1) convex hull of all BOS markers and (2) union-like left/right foot hull outlines in one color. The existing legend, axis labels, title, and CLI surface remain unchanged. Users can verify success by generating GIFs for multiple trials and confirming the new lines are visible while baseline labels remain identical.

KO: 이 변경 후 사용자는 기존 `scripts/plot_bos_com_xy_sample.py` 명령을 그대로 실행하면서 GIF에 BOS 오버레이 2종을 추가로 확인할 수 있다: (1) BOS 전체 마커 convex hull, (2) 좌/우 발 hull을 단일 색상으로 표시한 union 형태 outline. 기존 legend, axis label, title, CLI 인터페이스는 유지된다. 여러 trial로 GIF를 생성해 새 라인이 보이고 기존 라벨 포맷이 동일함을 확인하면 성공이다.

## Progress

- [x] (2026-02-18 03:55Z) Drafted requirements and fixed scope: implement `gpt_plan.md` intent while excluding all CLI-related changes.
- [x] (2026-02-18 03:55Z) Implemented overlay computation and GIF artist updates in `scripts/plot_bos_com_xy_sample.py`.
- [x] (2026-02-18 03:59Z) Ran syntax checks and multi-sample runtime validation in conda environment `module` (`가윤호-60-1`, `강비은-30-4`, `권유영-35-1`).
- [x] (2026-02-18 03:59Z) Ran MD5 comparison between reference GIF outputs and new outputs for selected trials (`output/qc/md5_bos_com_xy_ref_vs_new.tsv`).
- [x] (2026-02-18 04:00Z) Updated `.codex/issue.md` (issue only), updated `$replace-v3d-troubleshooting` recipe, and finalized commit-ready state.

## Surprises & Discoveries

- Observation: Repository currently has unrelated modified files (`AGENTS.md`, `.codex/PLANS.md`, `.codex/REQUIREMENTS_TEMPLATE.md`, `gpt_plan.md`, and one untracked archive file).
  Evidence: `git status --short` before implementation.

- Observation: There was no existing `.codex/execplans/` directory.
  Evidence: `ls -la .codex/execplans` failed with “No such file or directory”.

- Observation: Running baseline script copied to `/tmp` failed with `_bootstrap` import error until `PYTHONPATH=scripts` was set.
  Evidence: `ModuleNotFoundError: No module named '_bootstrap'` during initial reference GIF command.

- Observation: MD5 values differed for all three checked GIF files after overlay integration.
  Evidence: `output/qc/md5_bos_com_xy_ref_vs_new.tsv` shows `same=False` for all rows.

## Decision Log

- Decision: Keep CLI unchanged and wire C3D resolution internally using default directory `data/all_data`.
  Rationale: User explicitly chose “CLI 관련 변경 전부 제외,” while still requesting full functional overlay behavior.
  Date/Author: 2026-02-18 / Codex

- Decision: Preserve existing GIF legend/title/axis code and add hull/union as non-legend line artists only.
  Rationale: User requirement states those UI texts must not change.
  Date/Author: 2026-02-18 / Codex

- Decision: Use existing repo utilities (`iter_c3d_files`, `parse_subject_velocity_trial_from_filename`, `resolve_subject_from_token`, `convex_hull_2d`, `read_c3d_points`) instead of custom parsing.
  Rationale: Reduces divergence from core pipeline conventions and improves maintainability.
  Date/Author: 2026-02-18 / Codex

## Outcomes & Retrospective

EN: Core implementation and validation are complete. Overlay-enabled GIFs were generated successfully for three trials, C3D auto-resolution worked in all three runs, and MD5 comparisons against reference GIFs confirm expected output changes. Remaining work is repository bookkeeping (commit and final report alignment).

KO: 핵심 구현과 검증이 완료되었다. 3개 trial에서 overlay GIF 생성이 성공했고, 모든 실행에서 C3D 자동 매칭이 동작했으며, reference 대비 MD5 비교로 예상된 출력 변경이 확인되었다. 남은 작업은 저장소 정리 단계(커밋 및 최종 보고 정합성)다.

## Context and Orientation

EN: The target entry script is `scripts/plot_bos_com_xy_sample.py`. It reads `output/all_trials_timeseries.csv`, selects one `(subject, velocity, trial)` group, and renders static PNG and/or GIF. The GIF currently draws BOS rectangle (`BOS_minX/maxX/minY/maxY`) and COM trajectory status. This plan adds two polygon overlays computed from C3D markers:
- `hull`: convex hull over 8 BOS markers (`LHEE`, `LTOE`, `LANK`, `LFoot_3`, `RHEE`, `RTOE`, `RANK`, `RFoot_3`)
- `union`: concatenated left/right hull polylines with `NaN` separator in one color

KO: 대상 엔트리 스크립트는 `scripts/plot_bos_com_xy_sample.py`다. 이 스크립트는 `output/all_trials_timeseries.csv`를 읽고 `(subject, velocity, trial)` 단위로 trial을 선택해 static PNG/GIF를 생성한다. 기존 GIF는 BOS rectangle(`BOS_minX/maxX/minY/maxY`)과 COM 궤적/상태를 표시한다. 본 계획은 C3D 마커 기반 polygon 오버레이 2개를 추가한다.
- `hull`: 8개 BOS 마커(`LHEE`, `LTOE`, `LANK`, `LFoot_3`, `RHEE`, `RTOE`, `RANK`, `RFoot_3`)의 convex hull
- `union`: 좌/우 발 hull을 `NaN` separator로 이어 단일 색상으로 표시

## Plan of Work

EN: Add internal helper dataclass/functions near geometry/rendering utilities in `scripts/plot_bos_com_xy_sample.py`. Compute frame-wise hull/union arrays from matching C3D and rotate them with the same display transform. Keep failure-safe fallback: if no matching C3D or marker read fails, log warning and render legacy GIF without overlays. Extend `render_gif(...)` with optional `bos_polylines` argument and update overlay artists using the same `bos_idx` that already handles BOS freezing for step trials.

KO: `scripts/plot_bos_com_xy_sample.py` 내부에 dataclass/헬퍼 함수를 추가한다. 매칭되는 C3D에서 프레임별 hull/union 좌표를 계산하고 기존 display 회전 변환과 동일하게 적용한다. C3D 미탐색 또는 마커 로드 실패 시 경고만 출력하고 기존 GIF 경로로 폴백한다. `render_gif(...)`에 선택 인자로 `bos_polylines`를 추가하고, step trial의 BOS freeze에 이미 사용되는 `bos_idx` 기준으로 overlay를 업데이트한다.

## Concrete Steps

All commands are run from repository root `/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/replace_V3D`.

1. Baseline reference GIF generation (before new code) for selected trials.

    conda run -n module python scripts/plot_bos_com_xy_sample.py --save_gif --no-save_png --subject <S> --velocity <V> --trial <T> --out_dir output/figures/bos_com_xy_sample_ref

2. Apply code edits in `scripts/plot_bos_com_xy_sample.py`.

3. Syntax check.

    conda run -n module python -m py_compile scripts/plot_bos_com_xy_sample.py

4. Generate new GIF outputs for multiple trials.

    conda run -n module python scripts/plot_bos_com_xy_sample.py --save_gif --no-save_png --subject <S> --velocity <V> --trial <T> --out_dir output/figures/bos_com_xy_sample_new

5. Compare MD5 hashes (reference vs new).

    conda run -n module python - <<'PY'
    from pathlib import Path
    import hashlib

    def md5(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    ref_dir = Path("output/figures/bos_com_xy_sample_ref")
    new_dir = Path("output/figures/bos_com_xy_sample_new")
    for ref in sorted(ref_dir.glob("*.gif")):
        cand = new_dir / ref.name
        print(ref.name, md5(ref), md5(cand) if cand.exists() else "MISSING")
    PY

6. Commit and documentation updates.

    git add scripts/plot_bos_com_xy_sample.py .codex/execplans/bos_overlay_execplan.md .codex/issue.md
    git commit -m "BOS GIF 오버레이 로직을 CLI 변경 없이 확장" -m "hull/union 선을 step freeze 인덱스와 동기화해 프레임별로 렌더링" -m "기존 legend/axis/title 포맷은 유지하고 C3D 미탐색 시 기존 동작으로 안전 폴백"

## Validation and Acceptance

EN:
- `py_compile` passes with no syntax error.
- At least three trial GIF runs complete successfully.
- GIF visually contains rectangle + hull (dashed) + union (single color), while existing legend/title/axis labels are unchanged.
- Fallback path works when C3D is unavailable (warning printed, GIF still produced).
- MD5 comparison is executed and recorded (differences are expected when overlays are added).

KO:
- `py_compile`이 오류 없이 통과한다.
- 최소 3개 trial GIF 실행이 성공한다.
- GIF에서 rectangle + hull(점선) + union(단일 색상)이 시각적으로 확인되고, 기존 legend/title/axis label 포맷은 유지된다.
- C3D 미존재 시 경고 출력 후 기존 GIF가 생성되는 폴백 경로가 동작한다.
- MD5 비교를 수행하고 결과를 기록한다(overlay 추가로 해시 변경은 정상).

## Idempotence and Recovery

EN: Running render commands repeatedly is safe because output files are overwritten intentionally by the script. If a sample trial fails due to missing C3D mapping, choose another available trial and keep a fallback-case run as evidence. No destructive git operations are required.

KO: 렌더 명령 반복 실행은 스크립트가 동일 파일을 덮어쓰므로 안전하다. 특정 trial에서 C3D 매핑 실패가 나면 사용 가능한 다른 trial로 검증을 이어가고, 폴백 케이스 1건을 증거로 남긴다. 파괴적 git 명령은 사용하지 않는다.

## Artifacts and Notes

EN: Save validation evidence as command outputs in the terminal and concise notes in `.codex/issue.md` (issue only) plus troubleshooting solution recipe update in global skill `$replace-v3d-troubleshooting`.

KO: 검증 근거는 터미널 출력으로 남기고, `.codex/issue.md`에는 문제만 기록한다. 해결/우회 절차는 글로벌 스킬 `$replace-v3d-troubleshooting`에 업데이트한다.

Validation snippets:
    [BOS overlay] using C3D: .../251128_가윤호_perturb_60_001.c3d
    [BOS overlay] using C3D: .../251128_강비은_perturb_30_004.c3d
    [BOS overlay] using C3D: .../251117_권유영_perturb_35_001.c3d
    md5 compare: all 3 rows => same=False

## Interfaces and Dependencies

EN:
- Script: `scripts/plot_bos_com_xy_sample.py`
- Added internal type:
    `BOSPolylines(source_c3d: Path, hull_x: list[np.ndarray], hull_y: list[np.ndarray], union_x: list[np.ndarray], union_y: list[np.ndarray])`
- Extended internal function signature:
    `render_gif(..., bos_polylines: BOSPolylines | None = None) -> int`
- Dependencies used:
    `replace_v3d.cli.batch_utils.iter_c3d_files`
    `replace_v3d.io.events_excel.parse_subject_velocity_trial_from_filename`
    `replace_v3d.io.events_excel.resolve_subject_from_token`
    `replace_v3d.io.c3d_reader.read_c3d_points`
    `replace_v3d.geometry.geometry2d.convex_hull_2d`

KO:
- 스크립트: `scripts/plot_bos_com_xy_sample.py`
- 추가 내부 타입:
    `BOSPolylines(source_c3d: Path, hull_x: list[np.ndarray], hull_y: list[np.ndarray], union_x: list[np.ndarray], union_y: list[np.ndarray])`
- 확장 내부 함수 시그니처:
    `render_gif(..., bos_polylines: BOSPolylines | None = None) -> int`
- 사용 의존성:
    `replace_v3d.cli.batch_utils.iter_c3d_files`
    `replace_v3d.io.events_excel.parse_subject_velocity_trial_from_filename`
    `replace_v3d.io.events_excel.resolve_subject_from_token`
    `replace_v3d.io.c3d_reader.read_c3d_points`
    `replace_v3d.geometry.geometry2d.convex_hull_2d`

---
Revision note (2026-02-18 / Codex): Updated progress/discoveries/outcome after implementation, runtime validation (3 trials), MD5 comparison, and final documentation updates.
