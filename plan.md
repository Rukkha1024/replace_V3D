EXCEPLAN
주제: inverse dynamics용 forceplate 선택을 config 기반으로 고정하고,
      선택된 plate 개수에 따라 lower-limb torque / joint moment 계산 정책을 분기한다.

0) 핵심 결정
- 새 config key:
  forceplate.analysis.use_for_inverse_dynamics: [fp1, fp3]

- 이 값은 "고정된 [fp1, fp2] 전용"이 아니라,
  "임의의 forceplate 부분집합"을 받는다.
  예:
  - [fp1]
  - [fp3]
  - [fp1, fp3]
  - [fp2, fp3]
  - [fp1, fp2, fp3]

- 절대 가정하지 말 것:
  - plate 번호가 연속이라는 가정 금지
  - [fp1, fp3]이면 중간 fp2를 자동 포함하는 동작 금지
  - choose_active_force_platform()로 가장 큰 Fz plate 1개를 자동 고르는 동작을
    inverse dynamics 경로에서는 더 이상 사용하지 않음

- 정책:
  A. len(use_for_inverse_dynamics) == 1
     => strict single-plate mode
     => lower-limb torque / joint moment 계산 금지
     => 로그에 반드시 이유 명시
     => raw GRF/GRM/COP는 해당 plate 기준으로 유지 가능

  B. len(use_for_inverse_dynamics) >= 2
     => multi-plate V3D-style mode
     => 선택된 plate들만 사용
     => 좌/우 lower-limb torque / joint moment 계산 허용
     => force assignment는 "선택된 plate 집합 내부"에서만 수행

1) 수정 목표
- 분석에 사용할 forceplate를 config에서 명시적으로 지정한다.
- single-plate 선택이면 lower-limb torque family를 계산하지 않는다.
- multi-plate 선택이면 현재 one-plate 경로 대신 multi-plate 경로로 계산한다.
- [fp1, fp3] 같은 비연속 조합도 정확히 지원한다.

2) scope 정의
이번 patch에서 "lower-limb torque family"는 다음을 포함한다.
- side-specific ankle torque 계열:
  - AnkleTorqueL_*
  - AnkleTorqueR_*
  - AnkleTorqueMid_*  (이름상 해석 혼동을 막기 위해 single-plate에서는 같이 비활성화)
- lower-limb joint moment 계열:
  - Ankle_L/R_ref_*_Nm
  - Knee_L/R_ref_*_Nm
  - Hip_L/R_ref_*_Nm

이번 patch의 1차 목표는 "잘못된 값을 안 내는 것"이다.
즉, single-plate에서는 계산을 억지로 하지 않고 NaN/skip 처리한다.

3) config 설계
3-1. config.yaml에 추가
forceplate:
  analysis:
    use_for_inverse_dynamics: [fp1, fp3]

3-2. validation 규칙
- 필수 key로 취급
- 빈 리스트 금지
- 중복 금지
  예: [fp1, fp1] -> 에러
- 형식 검증
  - fp1/fp2/fp3/... 패턴만 허용
- C3D에 없는 plate 지정 시 에러
- 주의:
  forceplate.coordination.*.enabled 는 "corner override 사용 여부"이지,
  inverse dynamics plate 선택 여부가 아니다.
  즉, analysis selection과 coordination.enabled를 강하게 묶지 말 것.

3-3. 예시
- [fp3]
  -> single-plate strict mode
- [fp1, fp3]
  -> multi-plate mode
  -> fp2는 무시
- [fp1, fp2, fp3]
  -> multi-plate mode
  -> 3개 전체를 후보로 사용

4) 수정 파일별 지시

4-1. config.yaml
- forceplate.analysis.use_for_inverse_dynamics 예시를 추가한다.
- README/주석(있다면)에도 "임의 부분집합 가능"을 명시한다.

4-2. scripts/run_batch_all_timeseries_csv.py
해야 할 일:
- 새 helper 추가:
  _load_inverse_dynamics_forceplate_selection(config_path) -> list[int]
  동작:
  - forceplate.analysis.use_for_inverse_dynamics 읽기
  - ["fp1", "fp3"] -> [1, 3]로 변환
  - 순서 보존 + 중복 제거 금지(중복이면 에러)
  - 잘못된 key면 즉시 에러

- 기존 _load_forceplate_corner_overrides()와 독립적으로 동작하게 만들 것
  즉:
  - corner override 로직은 그대로 유지
  - analysis용 forceplate 선택은 별도 로직으로 분리

- 기존 inverse-dynamics 경로에서 choose_active_force_platform() 의존 제거
  현재 one-plate 자동선택 경로를 아래 2갈래로 분기:
  1) single-plate strict mode
  2) multi-plate mode

- single-plate strict mode 구현:
  - selected_plate_ids = [k] 하나만 선택된 경우
  - 해당 plate의 raw GRF/GRM/COP는 계속 계산 가능
  - 하지만 아래 계산은 하지 않음:
    - compute_ankle_torque_from_net_wrench(...)
    - compute_joint_moment_columns(...)의 lower-limb 결과 사용
  - 대신 torque_payload / joint_moment_payload 중 lower-limb columns는
    전부 NaN payload 생성
  - 로그 추가:
    [INFO] inverse-dynamics forceplates: [fp3]
    [INFO] mode=single_plate_strict
    [INFO] lower-limb torque/joint moment skipped because only one forceplate was selected

- multi-plate mode 구현:
  - selected_plate_ids 길이가 2 이상이면 진입
  - 선택된 plate들 각각에 대해 raw wrench를 따로 계산
  - non-selected plate는 완전히 무시
  - 더 이상 "trial 전체에서 가장 큰 |Fz| plate 1개"를 쓰지 않음
  - per-plate payload 목록을 만든 뒤 새 multi-plate inverse dynamics 함수로 넘김

- 출력 컬럼 정책:
  - 기존 단일 raw 컬럼(GRF_*, GRM_*, COP_*)은
    single-plate에서는 그대로 유지
  - multi-plate에서는 의미가 애매하므로
    1차 patch에서는 legacy 단일 raw 컬럼을 NaN 처리하거나,
    별도 메타컬럼 inverse_dynamics_forceplates="fp1,fp3"를 추가해서
    "이 trial은 multi-plate 계산"임을 명확히 남김
  - 이번 patch의 우선순위는 torque/joint moment 정책의 정확성이다.
    raw column 확장은 후순위여도 됨.

4-3. src/replace_v3d/torque/forceplate.py
해야 할 일:
- helper 추가:
  get_force_platform_by_index(platforms, index_1based) -> ForcePlatform
  또는
  select_force_platforms(platforms, indices_1based) -> list[ForcePlatform]

- 선택 정책:
  - 지정된 index가 실제 C3D에 없으면 에러
  - 반환 순서는 config 순서를 따른다

- choose_active_force_platform()는 삭제하지 말고 남겨두되,
  inverse dynamics 경로에서는 사용하지 않게 변경
  (다른 legacy 용도 보존용)

- 이미 있는 extract_platform_wrenches_lab(...)는 최대한 재사용

4-4. src/replace_v3d/joint_dynamics/inverse_dynamics.py
해야 할 일:
- 현재 함수는 single wrench 입력 전제이므로,
  multi-plate 입력을 받는 새 경로를 추가한다.

권장 구조:
- 새 dataclass 추가
  ForceplateWrenchSeries:
    - plate_index_1based: int
    - fp_origin_lab: np.ndarray
    - grf_lab: np.ndarray
    - grm_lab_at_fp_origin: np.ndarray
    - cop_x_m: np.ndarray
    - cop_y_m: np.ndarray
    - valid_contact_mask: np.ndarray

- 새 함수 추가
  compute_joint_moment_columns_multi(
      ...,
      forceplates: list[ForceplateWrenchSeries],
      ...
  ) -> dict[str, np.ndarray]

- frame별 기본 로직:
  1) 선택된 각 plate에 대해 contact/valid COP 판단
  2) 선택된 plate들 중 실제 contact가 있는 plate만 후보로 사용
  3) 선택된 plate가 1개만 contact이면 그 plate wrench만 사용
  4) 선택된 plate가 2개 이상 contact이면
     선택된 plate 내부에서만 좌/우 foot assignment 수행
  5) 한 plate 위에 두 발이 동시에 얹혀서 분리가 불가능한 frame은
     해당 frame의 lower-limb outputs를 NaN 처리
     (억지 분배 금지)

- 매우 중요:
  [fp1, fp3]이면 "fp1, fp3만" 후보로 쓰고,
  fp2는 contact가 있더라도 완전히 무시한다.

- lower-limb outputs만 우선 보장:
  - Ankle_L/R_ref_*_Nm
  - Knee_L/R_ref_*_Nm
  - Hip_L/R_ref_*_Nm

- trunk/neck moment는 이번 patch에서 기존 경로 유지 가능하면 유지,
  분리 구현이 번거로우면 1차 patch에서는 기존 함수 보존 + lower-limb만 새 경로로 분리
  (핵심은 lower-limb 잘못 계산 방지)

4-5. torque 관련 모듈(현재 ankle torque 계산 경로)
해야 할 일:
- 현재 ankle torque는 one-plate net wrench 전제이므로 정책 분리 필요

single-plate strict mode:
- AnkleTorque* 전부 NaN 처리
- 이유: 이름상 "ankle torque"로 해석되기 쉬워 오해 소지가 큼
- raw force/moment는 따로 남겨도 됨

multi-plate mode:
- 선택된 plate들의 assigned wrench 기반으로만 ankle torque 계산
- 현재 one-net-wrench 기반 계산은 lower-limb torque용으로 사용 금지

5) 로그 규칙
반드시 남길 것:
- 선택된 plate 목록 그대로 출력
  예:
  [INFO] inverse-dynamics forceplates from config: [fp1, fp3]

- 분기 모드 출력
  예:
  [INFO] mode=single_plate_strict
  또는
  [INFO] mode=multi_plate_v3d

- single-plate skip 로그
  예:
  [INFO] lower-limb torque/joint moment skipped: only one forceplate selected

- 잘못된 설정은 조용히 넘어가지 말고 즉시 에러
  예:
  - 빈 리스트
  - 중복
  - fp99처럼 존재하지 않는 plate
  - 형식 오류

6) 구현 순서
Step 1.
- config parser 추가
- [fp1], [fp3], [fp1, fp3] 파싱/검증 완료

Step 2.
- run_batch에서 inverse dynamics 선택 로직을 choose_active_force_platform() 기반에서
  config 기반 선택으로 교체

Step 3.
- single-plate strict mode 먼저 완성
  목표:
  - raw GRF/GRM/COP 유지
  - lower-limb torque/joint moment NaN
  - 로그 정상 출력
  - 파이프라인 crash 없음

Step 4.
- multi-plate payload 구조 추가
  - selected plate별 wrench 시계열 준비
  - non-selected plate 무시 확인

Step 5.
- inverse_dynamics.py에 multi-plate 계산 함수 추가
  - 선택된 plate들만 사용
  - frame-level assignment/invalid 처리 구현

Step 6.
- ankle torque 경로를 single/multi 정책에 맞게 정리

Step 7.
- CSV export에서 NaN columns / metadata columns 정리

7) 검증 계획 (수정 후 실행)
제공된 예시 파일:
- 0904_김종철_perturb_135_001.c3d

검증 Case A.
config:
  use_for_inverse_dynamics: [fp3]
기대 결과:
- mode=single_plate_strict 로그 출력
- lower-limb torque/joint moment columns가 NaN 또는 skip
- raw GRF/GRM/COP는 fp3 기준으로 정상 생성
- 파이프라인 crash 없음

검증 Case B.
config:
  use_for_inverse_dynamics: [fp1, fp3]
기대 결과:
- parser가 비연속 조합을 정상 허용
- fp2 자동 포함 금지
- mode=multi_plate_v3d 로그 출력
- non-selected plate(fp2)는 계산에 사용되지 않음
- 파이프라인 crash 없음

검증 Case C.
config:
  use_for_inverse_dynamics: [fp1, fp1]
기대 결과:
- 즉시 에러
- "duplicate forceplate selection" 메시지 출력

검증 Case D.
config:
  use_for_inverse_dynamics: []
기대 결과:
- 즉시 에러
- "at least one forceplate must be selected" 메시지 출력

검증 Case E.
config:
  use_for_inverse_dynamics: [fp99]
기대 결과:
- 즉시 에러
- "selected forceplate not present in C3D" 메시지 출력

8) 완료 기준 (Definition of Done)
- [fp1], [fp3], [fp1, fp3] 모두 config에서 정상 파싱
- inverse dynamics 경로에서 choose_active_force_platform() 자동선택 제거
- single-plate 선택 시 lower-limb torque family는 계산되지 않음
- multi-plate 선택 시 선택된 plate들만으로 계산
- [fp1, fp3]에서 fp2 자동 포함 없음
- 로그에 mode와 skip 이유가 명확히 남음
- sample c3d로 최소 Case A/B 실행 검증 완료

9) 이번 patch의 의도
- "계산할 수 없는 lower-limb torque를 억지로 내지 않는다"
- "어떤 forceplate를 분석에 쓸지 사용자가 명시적으로 통제한다"
- "선택 집합은 [fp1, fp2] 고정이 아니라 임의 부분집합이다"

10) 한 줄 요약
- config의 use_for_inverse_dynamics는 임의 부분집합을 받는다.
- 1개 선택이면 lower-limb torque/joint moment는 계산 금지.
- 2개 이상 선택이면 선택된 plate들만 사용해서 multi-plate V3D-style 로직으로 계산한다.