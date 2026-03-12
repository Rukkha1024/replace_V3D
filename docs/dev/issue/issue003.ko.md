# 이슈 003: inverse dynamics forceplate 선택을 config 기반으로 고정하기

**상태**: 진행 중
**생성일**: 2026-03-12

## 배경

현재 `replace_V3D`의 inverse dynamics 경로는 단일 active forceplate를 전제로 하며, 최대 수직력 기준으로 plate를 자동 선택할 수 있다.
하지만 이 저장소에서는 `[fp1, fp3]`처럼 사용자가 명시한 plate 부분집합만 써야 하고, plate가 1개만 선택된 경우에는 하지 토크와 관절 모멘트를 계산하지 않아야 한다.
이번 작업은 auto-select 경로를 config 기반 선택으로 교체하고, strict single-plate 모드와 multi-plate 모드를 분리하는 것이다.

## 완료 기준

- [ ] `config.yaml`에 `forceplate.analysis.use_for_inverse_dynamics`가 정의되고, 잘못된 값은 즉시 에러로 처리된다.
- [ ] `scripts/run_batch_all_timeseries_csv.py`에서 CLI `--force_plate`가 제거되고, config에 정의된 plate 집합만 사용한다.
- [ ] single-plate 선택에서는 raw GRF/GRM/COP는 유지하되 하지 토크와 관절 모멘트는 NaN으로 내보낸다.
- [ ] multi-plate 선택에서는 선택된 plate들만 이용해 하지 발목 토크와 관절 모멘트를 계산한다.
- [ ] `conda run -n module` 기준 배치 실행과 MD5 비교 결과가 기록된다.

## 작업 목록

- [x] 1. `plan.md`, 현재 inverse-dynamics 경로, 영향 파일을 검토한다.
- [x] 2. config 기반 forceplate 로딩과 single/multi inverse-dynamics 분기를 구현한다.
- [x] 3. 사용자 문서와 실행 계획 문서를 업데이트한다.
- [x] 4. 검증 실행, MD5 비교, diff 점검 후 커밋한다.

## 참고 사항

- 현재 구현 경로는 `scripts/run_batch_all_timeseries_csv.py`, `src/replace_v3d/torque/forceplate.py`, `src/replace_v3d/joint_dynamics/inverse_dynamics.py`를 중심으로 진행 중이다.
- CLI 정책은 고정했다. `--force_plate`는 제거하고 `forceplate.analysis.use_for_inverse_dynamics`를 단일 선택 기준으로 사용한다.
- 검증은 `conda run -n module python main.py --overwrite --skip_unmatched --out_dir output/qc_forceplate_selection --out_csv output/qc_forceplate_selection/all_trials_timeseries.csv --md5_reference_dir output/qc_forceplate_reference`로 실행했다.
- reference MD5는 `a83964c58132129c14078b17ed3b01a6`, 새 출력 MD5는 `e041d7b298c3dfa3ba7770e091f6a7d6`였고, strict single-plate 정책으로 하지 torque / joint moment가 NaN 처리되었기 때문에 diff는 예상된 결과다.
