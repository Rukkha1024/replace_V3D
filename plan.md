EXCEPLAN
주제: replace_V3D의 multi-plate inverse dynamics를 Visual3D 쪽에 더 가깝게 수정한다.
핵심: (1) force assignment를 COP-ankle heuristic에서 V3D-style block assignment로 교체
      (2) fixed 0.33 inertia를 per-segment inertia profile로 교체
      (3) 이미 들어간 use_for_inverse_dynamics 부분집합 선택은 그대로 유지

================================================================================
0) 이번 수정의 목표 / 비목표
================================================================================

목표
- [fp1], [fp3], [fp1, fp3] 같은 임의 부분집합 선택 기능은 그대로 유지
- len(selected_forceplates) == 1 이면 지금처럼 strict single-plate mode 유지
- len(selected_forceplates) >= 2 인 multi-plate mode에서
  V3D-style force assignment를 적용
- lower-limb joint moment(Hip/Knee/Ankle)가
  "가까운 ankle에 억지 배정"이 아니라
  "contiguous COP block 기반 assignment"로 계산되게 변경
- inertia를 segment별로 다르게 적용하여
  최소한 "모든 segment가 0.33 동일" 상태는 끝낸다

비목표
- angular velocity는 이번 patch 범위 아님
- exact Visual3D model-builder geometry clone(Hanavan full replica)는
  현재 repo 데이터 구조상 1차 patch에서 과도함
- 이번 1차 patch는 "V3D-style에 근접"이 목표이고,
  geometry-complete clone은 다음 단계 확장 포인트로 둔다

고등학생 비유
- 지금 방식: "발목에 더 가까운 쪽 발로 힘을 보낸다"
- 바꿀 방식: "발자국(COP)이 어느 발의 몸무게 중심(COM) 흐름을 더 오래 따라가는지 보고,
  그 블록 전체를 그 발에 준다"

================================================================================
1) 유지할 것
================================================================================

1-1. config 선택 구조는 유지
- forceplate.analysis.use_for_inverse_dynamics: [fp1, fp3]
- 비연속 선택 허용
- fp2 자동 포함 금지
- 선택된 plate만 inverse dynamics에 사용

1-2. single-plate strict mode는 유지
- selected plate 개수 == 1
- lower-limb torque / joint moment는 계속 NaN 또는 skip
- raw GRF / GRM / COP export는 유지
- 로그:
  [INFO] mode=single_plate_strict
  [INFO] lower-limb torque/joint moment skipped because only one forceplate was selected

1-3. ankle torque family 정책은 이번 patch에서 보수적으로 유지
- single-plate: NaN 유지
- multi-plate: 지금처럼 NaN 유지 가능
- 이번 patch의 우선순위는 "joint moment correctness"다

================================================================================
2) 수정 파일
================================================================================

수정 대상
- config.yaml
- scripts/run_batch_all_timeseries_csv.py
- src/replace_v3d/joint_dynamics/anthropometrics.py
- src/replace_v3d/joint_dynamics/inverse_dynamics.py

가능하면 추가
- tests/test_force_assignment_v3d_style.py
- tests/test_inverse_dynamics_selection.py

================================================================================
3) config 확장
================================================================================

현재 키는 유지하고, force assignment QA용 옵션을 analysis 아래에 추가한다.

권장안:
forceplate:
  analysis:
    use_for_inverse_dynamics: [fp1, fp3]
    force_assignment:
      cop_distance_threshold_m: 0.2
      remove_incomplete_assignments: true
      require_segment_projection_on_plate: true
      log_assignment_summary: true
    inertia:
      model: "per_segment_rog_v1"

설명
- use_for_inverse_dynamics:
  이미 있는 키. 그대로 사용
- cop_distance_threshold_m:
  Visual3D default를 따라 0.2
- remove_incomplete_assignments:
  trial 처음/끝에 걸쳐 불완전한 block 제거
- require_segment_projection_on_plate:
  assigned foot의 proximal/distal end가 plate 위에 있어야 통과
- inertia.model:
  1차 patch에서는 per_segment_rog_v1

주의
- 이 옵션들은 len(selected_forceplates) >= 2 인 multi-plate mode에만 의미 있음
- single-plate strict mode에서는 lower-limb kinetics를 여전히 계산하지 않음

================================================================================
4) anthropometrics.py 수정 계획
================================================================================

문제
- 현재 SegmentMassComParams는
  mass_fraction, com_fraction_from_prox만 있음
- inertia 정보가 없어서 inverse_dynamics.py에서 모든 segment에 0.33을 강제로 씀

수정
4-1. dataclass 확장
기존:
  SegmentMassComParams
변경:
  SegmentMassComInertiaParams

필드 추가:
- mass_fraction: float
- com_fraction_from_prox: float
- rog_fraction_xyz: tuple[float, float, float]
- optional:
  - display_name: str | None = None

예시
- thigh: (kx, ky, kz)
- shank: (kx, ky, kz)
- foot: (kx, ky, kz)
- trunk: (kx, ky, kz)
- head: (kx, ky, kz)

4-2. get_body_segment_params() 수정
- mass fraction, COM fraction은 현재 COMModelParams와 연결 유지
- inertia용 rog_fraction_xyz를 segment별로 채운다

중요
- 이번 patch에서는 "모든 segment 0.33 동일" 제거가 목표
- 1차 값은 published anthropometric table 기반의
  segment-specific constants를 넣되,
  code 구조를 나중에 geometry/Hanavan path로 바꾸기 쉽게 설계

4-3. 구현 기준
- 함수명은 그대로 get_body_segment_params() 유지 가능
- 다만 반환 타입이 inertia 포함 구조가 되도록 변경

================================================================================
5) inverse_dynamics.py - force assignment를 V3D-style로 교체
================================================================================

현재 문제
- _assign_force_to_side_by_cop(cop_lab, ankle_L, ankle_R)
  => XY 거리만 보고 left/right를 frame별로 고름
- 이것은 Visual3D의 block assignment와 다름

이번 patch 핵심
- _assign_force_to_side_by_cop는 legacy helper로 남기거나 deprecated 처리
- multi-plate path에서는 더 이상 사용하지 않음

--------------------------------------------------------------------------------
5-A) 새 helper 추가
--------------------------------------------------------------------------------

추가 함수 1
- _find_contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]
설명
- True가 연속된 [start, end_exclusive) 블록 반환
- plate contact block을 나누는 기본 함수

추가 함수 2
- _segment_distance_to_cop_block(
      segment_com: np.ndarray,
      cop_lab: np.ndarray,
      start: int,
      end: int,
  ) -> float
설명
- block 내부에서 segment COM과 COP의 거리 signal을 만들고 mean distance 반환
- distance는 3D Euclidean로 계산
  (COP는 plate z, segment COM은 실제 z)

추가 함수 3
- _project_points_to_plate_xy_mask(
      points_lab: np.ndarray,
      corners_lab: np.ndarray | None,
  ) -> np.ndarray
설명
- foot proximal/distal end가 plate 투영 polygon 안에 있는지 frame별 bool 반환
- corners_lab 없으면 warning 후 projection QA는 skip

추가 함수 4
- _assign_contact_blocks_v3d_style(
      forceplates: list[ForceplateWrenchSeries],
      foot_L_com: np.ndarray,
      foot_R_com: np.ndarray,
      foot_L_prox: np.ndarray,
      foot_L_dist: np.ndarray,
      foot_R_prox: np.ndarray,
      foot_R_dist: np.ndarray,
      threshold_m: float,
      remove_incomplete_assignments: bool,
      require_segment_projection_on_plate: bool,
  ) -> AssignmentResult
설명
- 이번 patch의 핵심 assignment 엔진

권장 반환 구조
@dataclass(frozen=True)
class AssignedBlock:
    plate_index_1based: int
    start: int
    end: int
    assigned_side: int | None   # 0=L, 1=R, None=invalid/unassigned
    mean_distance_L_m: float
    mean_distance_R_m: float
    invalid_reason: str | None

@dataclass(frozen=True)
class AssignmentResult:
    left_mask_per_plate: list[np.ndarray]
    right_mask_per_plate: list[np.ndarray]
    invalid_mask: np.ndarray
    assigned_blocks: list[AssignedBlock]

--------------------------------------------------------------------------------
5-B) block assignment 규칙
--------------------------------------------------------------------------------

각 selected forceplate마다:
1. contact_mask = valid_contact_mask & finite(GRF/GRM/COP)
2. contiguous contact blocks 추출
3. 각 block마다:
   - block mean distance to left foot COM
   - block mean distance to right foot COM
   계산
4. threshold 적용
   - 둘 다 threshold 초과 -> unassigned block
   - 하나만 threshold 이내 -> 그 side로 block 전체 assign
   - 둘 다 threshold 이내면 mean distance 더 작은 쪽 우선
5. 하지만 아래 경우는 invalid block 처리:
   - 같은 plate 위에 두 발이 동시에 올라와 있다고 판단
   - assigned foot의 proximal/distal projection QA 실패
   - incomplete assignment(block이 trial 시작/끝에 붙어 있는 경우) and 옵션 true

"같은 plate 위에 두 발이 동시에 올라옴" 판정
- 최소 1차 기준:
  - 같은 block에서 left/right foot의 proximal/distal projection이 모두 plate 위에 많이 존재
  또는
  - left/right 모두 threshold 이내이고 mean distance 차이가 매우 작음
- 이 block은 side를 고르지 말고 invalid_reason="two_feet_same_plate"

중요한 허용 케이스
- 한 발이 두 plate를 동시에 straddle
  => allowed
  => 동일 side에 여러 plate block이 동시에 assign될 수 있음
  => 이후 inverse dynamics에서 그 side wrench들을 합산

중요한 금지 케이스
- 한 plate에서 양발 분리 불가능
  => 그 frame/block lower-limb outputs는 invalid(NaN)

--------------------------------------------------------------------------------
5-C) multi-plate 구현 위치
--------------------------------------------------------------------------------

현재 _compute_joint_moment_columns_multi_impl(...) 안에서
for fp_series in forceplates:
    assign = _assign_force_to_side_by_cop(...)
를 쓰는 부분을 제거한다.

대신:
1. 먼저 foot_L / foot_R segment state와
   foot proximal/distal, com series를 준비
2. _assign_contact_blocks_v3d_style(...) 호출
3. 반환된 left_mask_per_plate / right_mask_per_plate를 사용해
   각 plate wrench를 left/right side로 masking
4. 같은 side에 여러 plate가 동시에 assign되면
   그대로 external_wrenches list에 여러 개 넣어서 합산되게 유지
5. invalid_mask frame은 양쪽 lower-limb outputs를 NaN 처리

주의
- trunk/head moment 계산은 기존 chain을 최대한 유지
- 하지만 invalid_mask가 lower-limb 외력 오염을 만들면
  trunk/head까지 propagation할지 정책 결정 필요
- 1차 patch에서는 최소한 Hip/Knee/Ankle만 확실하게 NaN 처리
- trunk/head는 기존 유지 가능하되, log에 lower-limb invalid 이유 명시

================================================================================
6) inverse_dynamics.py - inertia 모델 교체
================================================================================

현재 문제
- _segment_inertia_lab(frame, mass_kg, length_m)
  내부에서 kx=ky=kz=0.33 고정

수정
6-1. 함수 시그니처 변경
기존:
  _segment_inertia_lab(frame, mass_kg, length_m)
변경:
  _segment_inertia_lab(
      *,
      frame: np.ndarray,
      mass_kg: float,
      length_m: np.ndarray,
      rog_fraction_xyz: tuple[float, float, float],
  ) -> np.ndarray

6-2. _make_segment_state(...) 변경
- mass_kg만 받지 말고
  segment inertia profile도 같이 받도록 수정
예:
  _make_segment_state(
      ...,
      mass_kg=m_foot,
      rog_fraction_xyz=seg.lower.foot.rog_fraction_xyz,
      ...
  )

6-3. 각 segment에 서로 다른 inertia 적용
- foot_L / foot_R -> seg.lower.foot.rog_fraction_xyz
- shank_L / shank_R -> seg.lower.shank.rog_fraction_xyz
- thigh_L / thigh_R -> seg.lower.thigh.rog_fraction_xyz
- trunk -> seg.upper.trunk.rog_fraction_xyz
- head -> seg.upper.head.rog_fraction_xyz

6-4. 확장 포인트 남기기
- 함수 내부에 TODO 주석:
  "future: geometry-based inertia from proximal/distal radii and segment geometry"
- 즉, 1차 patch는 per-segment ROG
- 2차 patch는 geometry/Hanavan path 추가 가능하도록 구조 정리

================================================================================
7) run_batch_all_timeseries_csv.py 수정 계획
================================================================================

7-1. forceplate selection 관련 기존 로직 유지
- _load_inverse_dynamics_forceplate_selection 유지
- select_force_platforms(...) 유지
- [fp1, fp3] 같은 조합 계속 허용

7-2. single-plate strict path는 그대로 유지
- 현재 len(selected_platforms) == 1 분기 유지
- lower-limb torque/joint moment skip 로그 유지
- 여기서는 이번 patch에서 추가 수정 최소화

7-3. multi-plate path에 새 옵션 전달
- config에서 읽은 값들을
  compute_joint_moment_columns_multi(...)로 전달
예:
  force_assignment_threshold_m
  remove_incomplete_assignments
  require_segment_projection_on_plate

방법
- compute_joint_moment_columns_multi(...) 시그니처 확장
또는
- 새 dataclass ForceAssignmentConfig 전달

권장
@dataclass(frozen=True)
class ForceAssignmentConfig:
    cop_distance_threshold_m: float = 0.2
    remove_incomplete_assignments: bool = True
    require_segment_projection_on_plate: bool = True

7-4. 로그 강화
multi-plate mode에서 trial마다 아래 로그 추가:
- [INFO] inverse-dynamics forceplates from config: [fp1, fp3]
- [INFO] mode=multi_plate_v3d
- [INFO] force assignment summary:
    FP1 block 120-168 -> L, mean_dist=0.083m
    FP3 block 170-220 -> L, mean_dist=0.091m   (straddle allowed)
    FP3 block 300-340 -> invalid(two_feet_same_plate)

이 로그는 나중에 V3D와 수작업 비교할 때 매우 중요함

================================================================================
8) 구현 순서
================================================================================

Step 1.
- anthropometrics.py dataclass 확장
- per-segment rog_fraction_xyz 테이블 추가
- get_body_segment_params() 반환 구조 업데이트

Step 2.
- inverse_dynamics.py의 _segment_inertia_lab / _make_segment_state 수정
- 기존 0.33 고정 제거

Step 3.
- _find_contiguous_true_runs 추가
- _segment_distance_to_cop_block 추가
- _project_points_to_plate_xy_mask 추가
- AssignmentResult / AssignedBlock dataclass 추가

Step 4.
- _assign_contact_blocks_v3d_style(...) 구현
- left/right/invalid block assignment 로직 완성

Step 5.
- _compute_joint_moment_columns_multi_impl(...)에서
  _assign_force_to_side_by_cop 의존 제거
- 새 block assignment 결과로 per-plate left/right masks 생성
- side별 external wrench list 구성

Step 6.
- run_batch_all_timeseries_csv.py에서
  force assignment config 읽기 + 전달
- multi-plate summary log 추가

Step 7.
- invalid block/frame의 lower-limb outputs NaN 처리 확인
- Hip/Knee/Ankle_L/R_ref_*_Nm 우선 보장

================================================================================
9) 테스트 / 검증 계획
================================================================================

9-1. 단위 테스트(가능하면 추가)
파일:
- tests/test_force_assignment_v3d_style.py

케이스 A
- left foot COM이 COP block에 더 가까움
- 기대: block 전체 left assign

케이스 B
- right foot COM이 더 가까움
- 기대: block 전체 right assign

케이스 C
- same foot straddles fp1 + fp3
- 기대: 두 block 모두 same side assign, valid

케이스 D
- two feet on same plate
- 기대: invalid block, lower-limb outputs NaN

케이스 E
- block가 frame 1에서 시작하거나 EOF에서 끝남
- remove_incomplete_assignments=true이면 invalid

9-2. 제공된 sample c3d 실검증
파일:
- 0904_김종철_perturb_135_001.c3d

검증 1
config:
  use_for_inverse_dynamics: [fp3]
기대:
- mode=single_plate_strict
- lower-limb joint moment NaN/skip
- raw GRF/GRM/COP 정상
- crash 없음

검증 2
config:
  use_for_inverse_dynamics: [fp1, fp3]
기대:
- mode=multi_plate_v3d
- fp2 자동 포함 없음
- assignment summary 로그 출력
- sample에서 실제 active plate가 fp3 중심이면
  fp3 block만 주로 assign될 수 있음
- 한 발 contact 구간은 해당 side moment 생성 가능
- 두 발 같은 plate로 판단되는 구간은 invalid 처리

검증 3
config:
  use_for_inverse_dynamics: [fp1, fp1]
기대:
- duplicate selection error

검증 4
config:
  use_for_inverse_dynamics: [fp99]
기대:
- selected forceplate not present in C3D error

================================================================================
10) 완료 기준 (Definition of Done)
================================================================================

완료로 볼 조건
- [fp1], [fp3], [fp1, fp3] selection이 그대로 정상 동작
- single-plate strict mode는 기존 정책 유지
- multi-plate mode에서 _assign_force_to_side_by_cop 를 더 이상 쓰지 않음
- force assignment가 "contiguous COP block + segment COM mean distance + threshold"로 동작
- 한 foot가 두 plate를 straddle하면 valid
- 한 plate 위에 두 발이면 invalid
- Hip/Knee/Ankle lower-limb moments는 invalid frame에서 NaN
- inertia가 더 이상 모든 segment 0.33 동일값이 아님
- sample c3d로 pipeline crash 없이 실행됨
- 로그에 per-block assignment summary가 남음

================================================================================
11) 이번 patch의 한 줄 요약
================================================================================

- plate 선택 기능은 그대로 둔다.
- single-plate strict는 그대로 둔다.
- multi-plate만 "가까운 ankle" 방식에서 "contiguous COP block + foot COM" 방식으로 바꾼다.
- inertia는 global 0.33 하나로 때우지 말고 segment별로 다르게 넣는다.