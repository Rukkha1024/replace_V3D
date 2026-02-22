# replace_V3D 배치 전용 CLI 단순화

이 ExecPlan은 living document다. `Progress`, `Surprises & Discoveries`, `Decision Log`, `Outcomes & Retrospective` 섹션은 작업 진행 중 계속 업데이트해야 한다.

저장소 규칙 문서: `.codex/PLANS.md`를 따른다.

## Purpose / Big Picture

이 변경 후 사용자는 실행 경로를 고민할 필요 없이 `main.py` 배치 명령만 사용하면 된다. 레거시 단일 시행 플래그와 보조 엔트리포인트를 제거해 온보딩 혼선을 없애고, 기존 batch 결과가 그대로 유지되는지 MD5로 검증한다.

## Progress

- [x] (2026-02-22 10:20Z) 변경 전 기준 산출물 생성(`batch_only_before.csv`, MD5, help 텍스트).
- [x] (2026-02-22 10:23Z) `main.py`에서 single-trial CLI 및 실행 분기 제거.
- [x] (2026-02-22 10:24Z) 단일/보조 스크립트 4개 삭제.
- [x] (2026-02-22 10:27Z) `README/.codex/Archive` 단일모드 참조 제거.
- [x] (2026-02-22 10:33Z) 배치 동작 검증 및 before/after MD5 비교(`before=after=3bd4cfc31c17b5759899e0d75837c864`).
- [x] (2026-02-22 10:38Z) `.codex/issue.md` 문제 기록 및 `$replace-v3d-troubleshooting` 해결 기록.
- [ ] 한국어 3줄 커밋 및 불필요 산출물 정리.

## Surprises & Discoveries

- Observation: 내부 문서에 레거시 single/batch 혼합 MOS 헬퍼 스크립트가 남아 있어 정책 정합성 스캔에서 지속적으로 매치되었다.
  Evidence: 레거시 패턴 스캔 결과에서 내부 스킬 스크립트 매치 확인.

## Decision Log

- Decision: 제거 대상 단일 시행 플래그는 호환 래퍼 없이 제거하고 argparse 기본 에러를 그대로 사용한다.
  Rationale: 사용자 확정 정책이며 숨김 호환 경로를 남기지 않기 위함.
  Date/Author: 2026-02-22 / Codex

- Decision: MOS 전용 보조 배치 엔트리포인트도 제거한다.
  Rationale: 단일 공식 배치 경로(`main.py -> run_batch_all_timeseries_csv.py`)만 유지하는 요구사항에 부합.
  Date/Author: 2026-02-22 / Codex

- Decision: 내부 문서 정리는 실행 가능한 명령 기준으로 전면 적용한다.
  Rationale: 사용자 요구(".codex 포함 전면 정리")를 반영해 dead command를 남기지 않기 위함.
  Date/Author: 2026-02-22 / Codex

## Outcomes & Retrospective

배치 전용 구조 전환과 문서 정합성 정리는 완료되었고, 검증에서 batch 출력 해시가 변경 전과 동일함을 확인했다. 또한 argparse 축약 매칭으로 제거 플래그가 우회되는 문제를 발견해 parser 설정을 명시적으로 고정했다. 남은 작업은 최종 커밋과 변경 내역 공유다.

## Context and Orientation

핵심 엔트리포인트는 `main.py`다. 기존에는 배치 기본 경로 외에 레거시 단일 분기가 있었고, 이 분기가 단일 시행 보조 스크립트들을 호출했다. 이번 작업에서는 해당 분기와 스크립트를 제거하고, batch 경로(`scripts/run_batch_all_timeseries_csv.py` + `scripts/apply_post_filter_from_meta.py`)만 남긴다.

문서 측면에서는 사용자 문서(`README.md`)와 내부 문서(`.codex`, `Archive`)에서 삭제된 실행 경로를 모두 정리해 실제 CLI와 문서가 일치하도록 만든다.

## Plan of Work

`main.py`의 parser와 main 분기를 batch-only로 단순화한다. 그 다음 single-trial 및 MOS 보조 엔트리포인트 4개를 삭제한다. 이후 `README`와 내부 문서에서 제거된 플래그/스크립트 참조를 현재 정책에 맞게 교체한다. 마지막으로 before/after 실행과 MD5 비교로 배치 동작 보존을 검증하고, 문제/해결 기록 및 커밋을 완료한다.

## Concrete Steps

작업 디렉터리: 저장소 루트

1. 변경 전 기준 확보.

    conda run -n module python main.py --overwrite --skip_unmatched --max_files 5 --out_csv output/qc/batch_only_before.csv
    md5sum output/qc/batch_only_before.csv > output/qc/batch_only_before.md5
    conda run -n module python main.py --help > output/qc/main_help_before.txt

2. `main.py` batch-only 전환 및 레거시 스크립트 삭제.

3. 문서 정합성 스캔.

    rg -n -S -- "<legacy-entrypoint-or-flag-patterns>" README.md .codex Archive

4. 변경 후 검증.

    conda run -n module python main.py --help > output/qc/main_help_after.txt
    conda run -n module python main.py <removed-single-trial-flag> data/all_data/does_not_matter.c3d
    conda run -n module python main.py --overwrite --skip_unmatched --max_files 5 --out_csv output/qc/batch_only_after.csv
    md5sum output/qc/batch_only_after.csv > output/qc/batch_only_after.md5
    diff -u output/qc/batch_only_before.md5 output/qc/batch_only_after.md5

5. 기록/커밋.

## Validation and Acceptance

1. `main.py --help` 결과에서 제거 대상 단일 시행 플래그가 보이지 않아야 한다.
2. 제거된 단일 시행 플래그로 `main.py` 실행 시 argparse `unrecognized arguments` 에러가 발생해야 한다.
3. `batch_only_before.csv`와 `batch_only_after.csv` MD5가 동일해야 한다.
4. `README/.codex/Archive` 정합성 스캔에서 매치가 없어야 한다.
5. `.codex/issue.md`와 `$replace-v3d-troubleshooting`에 각각 문제/해결 기록이 남아야 한다.

## Idempotence and Recovery

모든 검증 명령은 재실행 가능하다. `output/qc` 산출물은 동일 파일명으로 덮어써도 된다. MD5가 불일치하면 문서 변경이 아니라 실행 경로 변경 가능성을 우선 점검하고, 입력 파일 순서/필터 적용 여부를 재확인한다.

## Artifacts and Notes

- `output/qc/batch_only_before.csv`
- `output/qc/batch_only_before.md5`
- `output/qc/main_help_before.txt`
- `output/qc/batch_only_after.csv`
- `output/qc/batch_only_after.md5`
- `output/qc/main_help_after.txt`

## Interfaces and Dependencies

- 변경 대상 CLI: `main.py`
  - 제거 플래그: 단일 시행 전용 플래그 2개
  - 유지 플래그: batch 관련 옵션 일체
- 유지 엔진:
  - `scripts/run_batch_all_timeseries_csv.py`
  - `scripts/apply_post_filter_from_meta.py`
- 삭제 엔트리포인트:
  - 레거시 joint-angle 단일 실행 스크립트
  - 레거시 ankle-torque 단일 실행 스크립트
  - 레거시 MOS 단일 실행 스크립트
  - 레거시 MOS 보조 배치 스크립트

---
Change Note (2026-02-22): 배치 전용 정책 확정에 따라 신규 ExecPlan(한/영 동기화)을 추가했다.
