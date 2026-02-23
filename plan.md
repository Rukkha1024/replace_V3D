Plan: MOS/BOS zeroing 제거

 Context

 plot_grid_timeseries.py의 --y_zero_onset 플래그가 default=True로 설정되어 있어,
 --no-y_zero_onset을 명시하지 않으면 MOS/BOS를 포함한 모든 컬럼이
 platform_onset(t=0) 시점 값을 빼서 0 기준으로 그려진다.

 MOS의 0은 "XCoM이 BOS 경계 위"를 의미하는 물리적 기준점이므로
 zeroing 시 절대 안정성 정보 및 부호 의미가 파괴된다.
 BOS_area도 절대면적(m²)이므로 zeroing이 무의미하다.

 문제 위치

 - scripts/plot_grid_timeseries.py L129-131: --y_zero_onset default=True
 - L629, L707, L782: y_zero_onset=True이면 col_name 구분 없이 모든 컬럼에 zeroing 적용

 수정 계획

 방법: 컬럼명 prefix 기반 zeroing 제외 목록 추가

 scripts/plot_grid_timeseries.py에 zeroing 제외 prefix 상수 추가:

 # MOS/BOS는 절대 기준값이 있으므로 zeroing 제외
 _NO_ZERO_PREFIXES = ("MOS_", "BOS_")

 def _should_zero(col_name: str, y_zero_onset: bool) -> bool:
     if not y_zero_onset:
         return False
     return not any(col_name.startswith(p) for p in _NO_ZERO_PREFIXES)

 zeroing 적용 3개소(L629, L707, L782)를 아래로 교체:
 # 기존
 if y_zero_onset:
     y_vals = subtract_baseline_at_x(...)

 # 수정
 if _should_zero(col_name, y_zero_onset):
     y_vals = subtract_baseline_at_x(...)

 작업 대상 파일

 - scripts/plot_grid_timeseries.py (L629, L707, L782)

 project plan.md 업데이트

 수정 후 plan.md에 문제점/수정 내용 기록

 검증

 1. --y_zero_onset(default) 상태로 MOS/BOS 그래프 재생성
 2. MOS 값이 양수/음수 절대값으로 표시되는지 확인
 3. 다른 컬럼(joint angle 등)은 여전히 zeroing 적용되는지 확인

---

구현 결정(확정)

- 기존 result column을 replace-only로 적용한다(신규/병행 컬럼 추가 없음).
- legacy MOS alias(`*_dir`)는 코드/API/CSV에서 완전 제거하고 canonical(`*_v3d`)만 유지한다.
- `--y_zero_onset` 기본 동작은 유지하되 `MOS_`, `BOS_` prefix series는 zeroing에서 제외한다.

변경 대상(확정)

- src/replace_v3d/mos/core.py
  - `MOSResult`에서 legacy alias 필드 제거
  - alias 생성/반환 로직 제거
- scripts/run_batch_all_timeseries_csv.py
  - payload에서 legacy alias 컬럼 제거
  - export finalizer 의존 제거
- scripts/plot_grid_timeseries.py
  - `_NO_ZERO_PREFIXES=("MOS_","BOS_")`, `_should_zero(...)` 추가
  - zeroing 3개 지점을 `_should_zero` 기반으로 교체
- README.md
  - canonical-only 정책 및 y-zero 예외 동작 반영

검증 산출물 경로

- output/qc/mos_bos_replace/before_all_trials.md5
- output/qc/mos_bos_replace/after_all_trials.md5
- output/qc/mos_bos_replace/before_fig.md5
- output/qc/mos_bos_replace/after_fig.md5
