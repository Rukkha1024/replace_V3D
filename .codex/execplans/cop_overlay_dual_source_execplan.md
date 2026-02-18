# Add Corrected COP Overlay to BOS+COM/xCOM GIF (Dual Source Sample) / 보정 COP를 BOS+COM/xCOM GIF에 이중 소스로 오버레이

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `.codex/PLANS.md` from the repository root.

## Purpose / Big Picture

EN: Users can now render corrected COP together with COM/xCOM in the existing GIF workflow, using either absolute COP or onset-zero COP from the pipeline CSV. The two modes are intentionally manual-two-pass, and output filenames include source tags so the two GIFs can be compared without ambiguity.

KO: 사용자는 기존 GIF 워크플로우에서 파이프라인 CSV의 보정 COP를 COM/xCOM과 함께 렌더링할 수 있다. COP 소스는 absolute/onset0 두 가지를 수동 2회 실행으로 생성하며, 파일명에 source 태그를 넣어 비교 시 혼동을 없앤다.

## Progress

- [x] (2026-02-18 16:18Z) `scripts/plot_bos_com_xy_sample.py`에 `--cop_source` CLI 및 source 선택 로직 추가.
- [x] (2026-02-18 16:26Z) `TrialSeries`/`DisplaySeries`에 COP optional 필드, onset0 정렬 shift 메타데이터, COP fallback 상태값 추가.
- [x] (2026-02-18 16:32Z) GIF 렌더러에 COP parity(궤적/현재점/inside-outside/live step ghost) 및 우측 패널/legend 확장 반영.
- [x] (2026-02-18 16:36Z) 파일명 규칙 `__copsrc-absolute|onset0` 반영 및 verbose 로그 확장.
- [x] (2026-02-18 16:38Z) `py_compile` 검증 통과.
- [x] (2026-02-18 16:42Z) absolute/onset0/fallback GIF 생성 검증 + md5 report 생성.
- [x] (2026-02-18 16:45Z) nonstep 회귀 검증은 CSV에 존재하는 footlift trial로 대체 실행.

## Surprises & Discoveries

- Observation: 요청된 nonstep 샘플 `subject=이훈, velocity=20, trial=3`는 `output/all_trials_timeseries.csv`에 존재하지 않았다.
  Evidence: `ValueError: Selected trial has no rows: subject=이훈, velocity=20, trial=3`

- Observation: 한글 subject명 렌더링 시 matplotlib 기본 폰트에서 glyph 경고가 계속 발생한다.
  Evidence: `UserWarning: Glyph xxxx missing from font(s) DejaVu Sans.`

## Decision Log

- Decision: onset0 모드에서 COP만 이동하지 않고 COM/xCOM/BOS를 동일 shift로 이동해 동일 좌표계로 정합한다.
  Rationale: onset0 COP는 기준점이 0으로 이동된 값이므로 나머지 XY 참조도 함께 이동해야 inside/outside 의미가 유지된다.
  Date/Author: 2026-02-18 / Codex

- Decision: COP 컬럼 누락 시 렌더를 실패시키지 않고 경고 후 COM/xCOM-only로 폴백한다.
  Rationale: 기존 xCOM 폴백 정책과 일관되며 배치/샘플 생성 안정성이 높다.
  Date/Author: 2026-02-18 / Codex

- Decision: nonstep 검증 입력은 실제 CSV 존재 trial로 치환하여 검증 목적을 유지한다.
  Rationale: 고정된 검증 커맨드가 데이터셋 현재 상태와 불일치하여 실행 불가능했기 때문이다.
  Date/Author: 2026-02-18 / Codex

## Outcomes & Retrospective

EN: The feature is implemented in one file (`scripts/plot_bos_com_xy_sample.py`) without touching upstream torque/COP computation. Dual-source output naming works, onset0 alignment is applied, and fallback behavior for missing COP columns is verified.

KO: 기능은 단일 파일(`scripts/plot_bos_com_xy_sample.py`)에서 구현되었고, 상위 torque/COP 계산 로직은 변경하지 않았다. dual-source 파일명 규칙이 동작하며 onset0 정렬이 반영되었고, COP 누락 컬럼 폴백도 검증되었다.

## Context and Orientation

EN: The entry script reads long CSV data (`output/all_trials_timeseries.csv`) and renders one-trial GIFs with BOS rectangle, COM, xCOM, and optional BOS hull/union overlays. This plan adds corrected COP overlays from CSV columns only. Raw forceplate signals in `.c3d` are not used for COP rendering in this script.

KO: 엔트리 스크립트는 long CSV(`output/all_trials_timeseries.csv`)를 읽어 trial 단위 GIF를 렌더링하며, BOS rectangle/COM/xCOM/optional BOS hull-union을 표시한다. 본 계획은 CSV 컬럼 기반 보정 COP 오버레이만 추가한다. `.c3d` 원시 forceplate 신호를 이 스크립트에서 COP 렌더링에 직접 사용하지 않는다.

## Plan of Work

EN: Add `--cop_source` (`absolute`/`onset0`) to CLI and propagate it via `RenderConfig`. Build COP arrays in `build_trial_series`, including missing-column fallback and onset0 alignment prerequisites. Extend `DisplaySeries` with rotated COP arrays and display-shift metadata. Update fixed-axis calculation, legend, right panel, animated artists, phase split, and ghost markers. Add source tags in output filenames and verbose logs.

KO: CLI에 `--cop_source` (`absolute`/`onset0`)를 추가하고 `RenderConfig`로 전달한다. `build_trial_series`에서 COP 배열을 구성하며, 누락 컬럼 폴백과 onset0 정렬 전제조건을 처리한다. `DisplaySeries`에 회전된 COP 배열과 display-shift 메타데이터를 확장한다. 고정 축 계산, legend, 우측 패널, 애니메이션 아티스트, phase 분할, ghost 마커를 확장한다. 출력 파일명과 verbose 로그에 source 태그를 반영한다.

## Concrete Steps

Run from repo root:

    conda run -n module python -m py_compile scripts/plot_bos_com_xy_sample.py
    conda run -n module python scripts/plot_bos_com_xy_sample.py --subject 가윤호 --velocity 60 --trial 1 --step_vis phase_bos --cop_source absolute --gif_name_suffix bos_com_xcom_cop --out_dir output/qc/cop_overlay/absolute
    conda run -n module python scripts/plot_bos_com_xy_sample.py --subject 가윤호 --velocity 60 --trial 1 --step_vis phase_bos --cop_source onset0 --gif_name_suffix bos_com_xcom_cop --out_dir output/qc/cop_overlay/onset0
    conda run -n module python scripts/plot_bos_com_xy_sample.py --subject 가윤호 --velocity 60 --trial 2 --step_vis phase_bos --cop_source absolute --gif_name_suffix bos_com_xcom_cop --out_dir output/qc/cop_overlay/nonstep
    conda run -n module python -c "import polars as pl, pandas as pd; from pathlib import Path; src=Path('output/all_trials_timeseries.csv'); out=Path('output/qc/cop_overlay/all_trials_timeseries_no_cop.csv'); out.parent.mkdir(parents=True, exist_ok=True); df=pl.read_csv(src, infer_schema_length=10000, encoding='utf8-lossy'); drop={'COP_X_m','COP_Y_m','COP_Z_m','COP_X_m_onset0','COP_Y_m_onset0','COP_Z_m_onset0'}; keep=[c for c in df.columns if c not in drop]; df.select(keep).to_pandas().to_csv(out, index=False, encoding='utf-8-sig')"
    conda run -n module python scripts/plot_bos_com_xy_sample.py --csv output/qc/cop_overlay/all_trials_timeseries_no_cop.csv --subject 가윤호 --velocity 60 --trial 1 --step_vis phase_bos --cop_source absolute --gif_name_suffix bos_com_xcom_cop_nocop --out_dir output/qc/cop_overlay/fallback
    find output/qc/cop_overlay -name "*.gif" -type f -print0 | sort -z | xargs -0 md5sum > output/qc/cop_overlay/md5_report.tsv

## Validation and Acceptance

EN:
- `absolute` and `onset0` runs both succeed for the same step trial and produce source-tagged GIFs.
- Fallback CSV run prints COP-disabled warning and still renders GIF.
- MD5 report file exists at `output/qc/cop_overlay/md5_report.tsv` with four generated GIF entries.

KO:
- 동일 step trial에서 `absolute`, `onset0` 실행이 모두 성공하고 source 태그가 붙은 GIF가 생성된다.
- fallback CSV 실행 시 COP 비활성 경고가 출력되며 GIF가 계속 생성된다.
- `output/qc/cop_overlay/md5_report.tsv` 파일이 생성되고 GIF 4개 해시가 기록된다.

## Idempotence and Recovery

EN: Re-running any command is safe because output files are deterministic and overwritten by script behavior. If a selected trial does not exist in CSV, choose an existing trial key from the CSV rather than workbook-only rows.

KO: 명령 재실행은 안전하다. 출력 파일 경로가 결정적이며 스크립트가 덮어쓴다. 선택 trial이 CSV에 없으면 workbook 기준이 아니라 CSV에 실제 존재하는 trial 키로 바꿔 실행한다.

## Artifacts and Notes

- Absolute GIF: `output/qc/cop_overlay/absolute/가윤호/가윤호__velocity-60__trial-1__bos_com_xcom_cop__copsrc-absolute__step_vis-phase_bos__live.gif`
- Onset0 GIF: `output/qc/cop_overlay/onset0/가윤호/가윤호__velocity-60__trial-1__bos_com_xcom_cop__copsrc-onset0__step_vis-phase_bos__live.gif`
- Fallback GIF: `output/qc/cop_overlay/fallback/가윤호/가윤호__velocity-60__trial-1__bos_com_xcom_cop_nocop__copsrc-absolute__step_vis-phase_bos__live.gif`
- Nonstep-equivalent GIF (footlift): `output/qc/cop_overlay/nonstep/가윤호/가윤호__velocity-60__trial-2__bos_com_xcom_cop__copsrc-absolute__step_vis-phase_bos__live.gif`
- MD5 report: `output/qc/cop_overlay/md5_report.tsv`

## Interfaces and Dependencies

EN:
- New CLI flag: `--cop_source {absolute,onset0}` (default `absolute`).
- Updated internal fields in `TrialSeries` and `DisplaySeries` for COP arrays/masks and onset0 shift metadata.
- Updated GIF legend/right panel/rendering path to include COP artists when available.

KO:
- 신규 CLI 플래그: `--cop_source {absolute,onset0}` (기본 `absolute`).
- `TrialSeries`/`DisplaySeries` 내부에 COP 배열/마스크 및 onset0 shift 메타데이터 필드 추가.
- GIF legend/우측 패널/렌더 경로를 COP 가용 시 확장.

Revision note (2026-02-18 / Codex): Initial implementation log added with completed verification commands and acceptance evidence.
