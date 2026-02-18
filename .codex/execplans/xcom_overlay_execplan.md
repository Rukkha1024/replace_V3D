# Add xCOM Overlay to BOS+COM GIF With COM-Aligned Behavior / BOS+COM GIF에 COM 정합형 xCOM 오버레이 추가

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `.codex/PLANS.md` from the repository root.

## Purpose / Big Picture

EN: Users can render one-trial GIFs where COM and xCOM are shown on the same XY frame with synchronized timing, while preserving existing COM behavior. The goal is distinction, not analytical comparison: xCOM must be visually separable (dotted trajectory + triangle current point + step-onset X ghost in live mode), without introducing delta metrics or comparison panels.

KO: 사용자는 COM 기존 동작을 유지한 채 COM/xCOM을 동일 XY 프레임과 동일 시간축에서 함께 볼 수 있다. 목표는 비교 분석이 아니라 시각적 구분이다. 따라서 xCOM은 점선 궤적 + 삼각형 현재점 + live 모드 step-onset X ghost로 분리하고, Δ수치/비교 패널은 추가하지 않는다.

## Progress

- [x] (2026-02-18 15:35Z) 기준선(reference) GIF 1파일 생성 (`가윤호-60-1`, `step_vis=phase_bos`).
- [x] (2026-02-18 15:48Z) `scripts/plot_bos_com_xy_sample.py`에 xCOM optional data-path/artist/legend/ghost 로직 구현.
- [x] (2026-02-18 15:49Z) `py_compile` 문법 검증 통과.
- [x] (2026-02-18 15:51Z) 신규 xCOM GIF 1파일 생성 및 시각 규칙 확인.
- [x] (2026-02-18 15:52Z) xCOM 컬럼 누락 CSV로 경고+COM-only 폴백 검증.
- [x] (2026-02-18 15:52Z) reference/new MD5 비교 파일 생성 (`output/qc/xcom_overlay/md5_ref_vs_new.tsv`).
- [x] (2026-02-18 15:58Z) Git commit (Korean 3-line message).
- [x] (2026-02-18 16:06Z) `render_static_png()`에도 xCOM 구분 디자인(점선/삼각형/step X ghost)을 반영하고 정적 PNG 1장 생성 검증.
- [x] (2026-02-18 16:09Z) 요청에 따라 `output/qc/xcom_overlay` QC 산출 폴더 삭제.

## Surprises & Discoveries

- Observation: `conda run -n module python - <<'PY' ...` 패턴에서 본 환경은 출력이 비어 보일 수 있어 검증 로그가 누락될 수 있다.
  Evidence: heredoc 기반 md5 출력이 비어 있었고, `-c` one-liner로 전환 시 정상 출력됨.

- Observation: matplotlib 기본 폰트(DejaVu Sans)에서 한글 glyph 경고가 지속된다.
  Evidence: GIF 저장 시 `Glyph xxxx missing from font(s) DejaVu Sans` 경고 반복.

## Decision Log

- Decision: xCOM step 동작은 “현재점 marker 전환”이 아닌 “step-onset X ghost 추가”로 COM과 동일하게 맞춘다.
  Rationale: 사용자가 COM의 기존 semantics(현재점 유지 + onset 지점 ghost) 정합을 요구함.
  Date/Author: 2026-02-18 / Codex

- Decision: xCOM 컬럼 누락 시 실패하지 않고 경고 후 COM-only 렌더링으로 폴백한다.
  Rationale: 사용자 확정 요구사항(정상 렌더 지속 + 경고 1회)을 반영.
  Date/Author: 2026-02-18 / Codex

- Decision: static PNG는 변경하지 않고 GIF 경로만 확장한다.
  Rationale: 범위를 최소화하여 회귀 리스크를 줄이고 요구사항(출력 범위 GIF)을 준수.
  Date/Author: 2026-02-18 / Codex

## Outcomes & Retrospective

EN: Implementation and runtime verification are complete for both GIF and static-render design parity. COM rendering behavior is preserved, xCOM is clearly distinguishable, fallback behavior is validated when xCOM columns are removed, and temporary QC artifacts were removed per user request.

KO: GIF와 static 렌더 디자인 정합 범위에 대한 구현과 실행 검증이 완료되었다. COM 동작은 유지되며, xCOM은 명확히 구분되고, xCOM 컬럼 제거 시 폴백도 검증되었다. 사용자 요청에 따라 임시 QC 산출 폴더도 정리했다.

## Context and Orientation

EN: Target entry file is `scripts/plot_bos_com_xy_sample.py`. The script reads `output/all_trials_timeseries.csv`, selects one `(subject, velocity, trial)`, and renders GIF with BOS rectangle, COM trajectory/current point, optional BOS hull/union, and step visualization templates (`step_vis`). xCOM source columns are `xCOM_X`, `xCOM_Y` in the same CSV.

KO: 대상 엔트리 파일은 `scripts/plot_bos_com_xy_sample.py`다. 스크립트는 `output/all_trials_timeseries.csv`를 읽어 `(subject, velocity, trial)` 단위 GIF를 생성한다. 기존 요소는 BOS rectangle, COM 궤적/현재점, optional BOS hull/union, step 시각화 템플릿(`step_vis`)이며, xCOM 입력 컬럼은 동일 CSV의 `xCOM_X`, `xCOM_Y`다.

## Plan of Work

EN: Extend internal dataclasses (`TrialSeries`, `DisplaySeries`) with optional xCOM fields, compute xCOM validity/inside masks in `build_trial_series`, rotate xCOM coordinates in `build_display_series`, and include xCOM in fixed axis limits for GIF. In `render_gif`, add xCOM artists (dotted trajectory, triangle current point, inside/outside color split), phase-split support for `phase_trail/phase_bos`, and live-mode step-onset X ghost for xCOM. Keep CLI unchanged and preserve COM-only behavior when xCOM is unavailable.

KO: 내부 dataclass(`TrialSeries`, `DisplaySeries`)에 xCOM optional 필드를 확장하고, `build_trial_series`에서 xCOM valid/inside 마스크를 계산한다. `build_display_series`에서 xCOM 회전을 적용하고 GIF 고정 축 범위에 xCOM을 포함한다. `render_gif`에서는 xCOM 점선 궤적/삼각형 현재점/inside-outside 색상 분기와 `phase_trail/phase_bos` 분할을 추가하며, live 모드에서 step-onset X ghost를 xCOM에도 동일 적용한다. CLI는 변경하지 않고 xCOM 미존재 시 COM-only 동작을 유지한다.

## Concrete Steps

Run from repo root:

    conda run -n module python scripts/plot_bos_com_xy_sample.py --subject 가윤호 --velocity 60 --trial 1 --step_vis phase_bos --gif_name_suffix bos_com_xy_anim_ref --out_dir output/qc/xcom_overlay/ref
    conda run -n module python -m py_compile scripts/plot_bos_com_xy_sample.py
    conda run -n module python scripts/plot_bos_com_xy_sample.py --subject 가윤호 --velocity 60 --trial 1 --step_vis phase_bos --gif_name_suffix bos_com_xcom_xy_anim --out_dir output/qc/xcom_overlay/new
    conda run -n module python -c "import polars as pl, pandas as pd; from pathlib import Path; src=Path('output/all_trials_timeseries.csv'); out=Path('output/qc/xcom_overlay/all_trials_timeseries_no_xcom.csv'); out.parent.mkdir(parents=True, exist_ok=True); df=pl.read_csv(src, infer_schema_length=10000, encoding='utf8-lossy'); keep=[c for c in df.columns if c not in ('xCOM_X','xCOM_Y','xCOM_Z')]; df.select(keep).to_pandas().to_csv(out, index=False, encoding='utf-8-sig')"
    conda run -n module python scripts/plot_bos_com_xy_sample.py --csv output/qc/xcom_overlay/all_trials_timeseries_no_xcom.csv --subject 가윤호 --velocity 60 --trial 1 --step_vis phase_bos --gif_name_suffix bos_com_no_xcom_fallback --out_dir output/qc/xcom_overlay/fallback

## Validation and Acceptance

EN:
- Single trial run produces exactly one GIF for `step_vis=phase_bos` (live mode only).
- New GIF contains COM (solid line + circle point) and xCOM (dotted line + triangle point).
- Live mode after step onset shows COM X ghost and xCOM X ghost.
- When xCOM columns are missing, warning is printed and COM-only GIF is still generated.
- `output/qc/xcom_overlay/md5_ref_vs_new.tsv` exists and records reference/new hash mismatch as expected.

KO:
- `step_vis=phase_bos` 실행에서 단일 trial 기준 GIF 1파일만 생성된다.
- 신규 GIF에 COM(실선+원형 현재점)과 xCOM(점선+삼각형 현재점)이 함께 보인다.
- live 모드 step 이후 COM X ghost와 xCOM X ghost가 함께 표시된다.
- xCOM 컬럼 누락 시 경고가 출력되고 COM-only GIF가 정상 생성된다.
- `output/qc/xcom_overlay/md5_ref_vs_new.tsv`가 생성되며 reference/new 해시 차이가 기록된다.

## Idempotence and Recovery

EN: Re-running commands is safe because output GIF paths are deterministic and overwritten by the script. If fallback verification CSV exists, it can be reused for future regression checks.

KO: 명령 재실행은 안전하다. 출력 GIF 경로가 고정되어 있고 스크립트가 덮어쓰기를 지원한다. 폴백 검증용 CSV는 재활용 가능하다.

## Artifacts and Notes

- Reference GIF: `output/qc/xcom_overlay/ref/가윤호/가윤호__velocity-60__trial-1__bos_com_xy_anim_ref__step_vis-phase_bos__live.gif`
- New GIF: `output/qc/xcom_overlay/new/가윤호/가윤호__velocity-60__trial-1__bos_com_xcom_xy_anim__step_vis-phase_bos__live.gif`
- Fallback GIF: `output/qc/xcom_overlay/fallback/가윤호/가윤호__velocity-60__trial-1__bos_com_no_xcom_fallback__step_vis-phase_bos__live.gif`
- MD5 report: `output/qc/xcom_overlay/md5_ref_vs_new.tsv`

## Interfaces and Dependencies

EN:
- Updated internal types:
  - `TrialSeries(..., xcom_x: np.ndarray | None, xcom_y: np.ndarray | None, xcom_valid_mask: np.ndarray | None, xcom_inside_mask: np.ndarray | None, ...)`
  - `DisplaySeries(..., xcom_x: np.ndarray | None, xcom_y: np.ndarray | None, ...)`
- Updated internal helpers:
  - `warn_missing_xcom_columns_once(columns: list[str])`
  - `build_gif_legend_handles(*, has_xcom: bool, show_step_ghost: bool)`
  - `apply_gif_right_panel(ax_side, *, has_xcom: bool, show_step_ghost: bool)`
- No CLI changes.

KO:
- 내부 타입 확장:
  - `TrialSeries(..., xcom_x/xcom_y/xcom_valid_mask/xcom_inside_mask optional 추가)`
  - `DisplaySeries(..., xcom_x/xcom_y optional 추가)`
- 내부 헬퍼 확장:
  - `warn_missing_xcom_columns_once(columns: list[str])`
  - `build_gif_legend_handles(*, has_xcom: bool, show_step_ghost: bool)`
  - `apply_gif_right_panel(ax_side, *, has_xcom: bool, show_step_ghost: bool)`
- CLI 변경 없음.

---
Revision note (2026-02-18 / Codex): Added full implementation and verification log for xCOM overlay with COM-aligned step-onset ghost behavior and fallback handling.
