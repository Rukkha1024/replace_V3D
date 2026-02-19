# why_stepping_before_threshold window mean 재분석 전환

이 ExecPlan은 living document다. `Progress`, `Surprises & Discoveries`, `Decision Log`, `Outcomes & Retrospective` 섹션은 작업 진행 중 계속 업데이트해야 한다.

저장소 규칙 문서: `.codex/PLANS.md`를 따른다.

## Purpose / Big Picture

사용자는 COP-COM 좌표계 불일치 제약 때문에 snapshot 기반 결과를 더 이상 신뢰하지 않는다. 이 변경 후 사용자는 `platform_onset_local ~ step_onset_local` 구간 평균(window mean) 기준으로 다시 계산된 GLMM/LOSO 정량값을 확인할 수 있고, `report.md`에서 실제 수치 기반 결론을 바로 재현할 수 있다.

## Progress

- [x] (2026-02-19 11:45Z) 요구사항 확정: window 정의, nonstep end frame, mean 집계 방식 고정.
- [x] (2026-02-19 12:20Z) `analyze_fsr_only.py` snapshot 로직 제거 및 window 집계 로직 구현.
- [x] (2026-02-19 12:24Z) 재실행으로 fig1~fig4 생성 및 GLMM/AUC 수치 추출.
- [x] (2026-02-19 12:30Z) `report.md` 완전 재작성 (Actual quant 결과 반영).
- [x] (2026-02-19 12:34Z) `.codex/issue.md` 이슈 기록 + `$replace-v3d-troubleshooting` 워크어라운드 기록.
- [x] (2026-02-19 12:36Z) MD5/수치 검증 완료 및 한국어 3줄 커밋 준비.

## Surprises & Discoveries

- Observation: 기존 snapshot 결과와 window mean 결과의 모델 우열이 달라질 수 있다.
  Evidence: 사전 프로브 AUC에서 1D velocity 우위가 약화(0.647)되고 2D/1D position과 간격이 줄어듦.

## Decision Log

- Decision: snapshot 결과를 유지하지 않고 report를 완전 교체한다.
  Rationale: 사용자 명시 지시("재분석 수치로 완전히 리포트 재작성").
  Date/Author: 2026-02-19 / Codex

- Decision: nonstep 종료시점은 subject 평균 step onset을 사용한다.
  Rationale: 사용자 지정 규칙으로 확정됨.
  Date/Author: 2026-02-19 / Codex

- Decision: trial 요약은 구간평균(mean)으로 통일한다.
  Rationale: 사용자 선택 확정.
  Date/Author: 2026-02-19 / Codex

## Outcomes & Retrospective

window mean 재분석으로 보고서 수치가 전면 갱신되었고, 모델 간 우열 해석도 바뀌었다. 특히 기존 snapshot 결론(velocity 우세)을 그대로 유지할 수 없고, 이번 기준에서는 velocity/position/2D 차이가 작다는 점을 명시했다. 재현 커맨드와 검증 로그(수치/MD5)를 남겨 같은 데이터에서 결과를 다시 확인할 수 있게 했다.

## Context and Orientation

핵심 파일은 `analysis/why_stepping_before_threshold/analyze_fsr_only.py`와 `analysis/why_stepping_before_threshold/report.md`다. 기존 코드는 trial마다 단일 `ref_frame`에서 snapshot을 뽑아 FSR 변수를 만든다. 이번 작업은 각 trial에서 window 프레임 전체를 사용해 frame-level 정규화를 수행한 뒤 평균값으로 trial feature를 만든다.

용어 정의:
- window mean: 시작~종료 프레임 범위의 값 평균.
- start frame: `platform_onset_local`.
- end frame: step trial은 자기 `step_onset_local`, nonstep은 subject 평균 `step_onset_local`.

## Plan of Work

`analyze_fsr_only.py`에서 `build_trial_summary`를 onset window 메타데이터 생성 함수로 바꾸고, `compute_fsr_features`를 frame-level -> trial mean 집계 함수로 바꾼다. 이후 main 출력 메시지와 figure 제목에 window mean 문구를 반영한다.

`report.md`는 기존 snapshot 수치/문장을 지우고 새 실행 결과 수치로 표를 다시 작성한다. 각 주요 섹션 하단에 GPT Comment 블록을 넣고 `Alternative Applied`와 `Actual Result (Quant)` 라인을 포함한다.

## Concrete Steps

작업 디렉터리: 저장소 루트

    conda run -n module python analysis/why_stepping_before_threshold/analyze_fsr_only.py

기대 로그:
- window 정의 출력
- trial/frame 통계 출력
- GLMM 4모델 출력
- LOSO AUC 4모델 출력
- fig1~fig4 저장 로그

검증 커맨드:

    rg -n "0.794|0.787|snapshot|ref_frame" analysis/why_stepping_before_threshold/report.md
    rg -n "Alternative Applied|Actual Result \(Quant\)" analysis/why_stepping_before_threshold/report.md

## Validation and Acceptance

1. 스크립트가 0 exit이고 fig1~fig4를 생성해야 한다.
2. 로그에 모델 4개(2D, 1D velocity, 1D position, 1D MoS)만 있어야 한다.
3. report 표 수치는 stdout 수치와 일치해야 한다.
4. report에서 snapshot/ref_frame 잔재가 없어야 한다.

## Idempotence and Recovery

동일 명령 재실행 시 같은 입력에서 동일 구조의 출력(fig 파일명/로그 항목)을 만든다. 값이 바뀌면 데이터 변경 또는 라이브러리 버전 차이 가능성이 있으므로 실행 로그를 비교해 원인을 분리한다.

## Artifacts and Notes

- 기준 로그: `/tmp/why_step_before_stdout.txt`
- 기준 MD5: `/tmp/why_step_before_md5_before.txt`
- 재실행 로그/MD5는 구현 후 같은 패턴으로 저장

## Interfaces and Dependencies

- 유지 CLI: `--csv`, `--platform_xlsm`, `--out_dir`, `--dpi`
- 유지 모델 인터페이스:
  - GLMM: `fit_glmm(data, formula, groups="subject")`
  - LOSO: `compute_loso_cv_auc(data, feature_cols, y_col, group_col)`
- 변경 데이터 인터페이스:
  - trial summary: `start_frame`, `end_frame` 포함
  - feature table: `COM_pos_norm`, `COM_vel_norm`, `MOS_minDist_signed`, `n_frames`

---
Change Note (2026-02-19): 사용자 확정 규칙(window start/end/mean 집계, full rewrite)을 반영해 신규 ExecPlan을 작성함.
