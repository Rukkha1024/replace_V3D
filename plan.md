EXCEPLAN
주제:
1) 레거시 로직 제거
2) plotting을 새 컬럼 체계로 변경
3) 예외 로깅 준수
+
전제:
- use_for_inverse_dynamics 는 고정 [fp1, fp2]가 아니라 임의 subset 이다.
- 즉 [fp1], [fp3], [fp1, fp3], [fp2, fp3], [fp1, fp2, fp3] 모두 유효해야 한다.

==================================================
A. 최종 목표
==================================================

A-1. single_plate_strict
- forceplate가 1개 선택되면 lower-limb torque / joint moment는 "계산 자체를 하지 않는다".
- 겉으로 NaN만 덮는 게 아니라, 내부 legacy 계산도 아예 호출하지 않는다.
- raw forceplate 값은 per-plate 새 컬럼 체계로만 남긴다.

A-2. multi_plate_v3d
- 선택된 plate들만 사용해 lower-limb joint moment를 계산한다.
- legacy 단일 컬럼(GRF_*, COP_*, AnkleTorque*)은 더 이상 주 출력으로 쓰지 않는다.
- raw forceplate는 per-plate 새 컬럼 체계로 저장한다.

A-3. plotting
- 더 이상 legacy ankle torque / legacy GRF-COP 컬럼을 보지 않는다.
- 새 컬럼 체계만 본다.
- [fp1, fp3]이면 fp1과 fp3만 그림이 나오고 fp2는 자동으로 안 나온다.

A-4. exception logging
- inverse dynamics 관련 예외를 조용히 NaN으로 삼키지 않는다.
- trial context(파일명, subject, velocity, trial, mode, selected forceplates)를 남기고
  traceback까지 로깅한 뒤 fail-fast 한다.

==================================================
B. 이번 patch에서 확정할 정책
==================================================

B-1. 완전 제거할 레거시 출력
- GRF_X_N / GRF_Y_N / GRF_Z_N
- GRM_X_Nm_at_FPorigin / GRM_Y_Nm_at_FPorigin / GRM_Z_Nm_at_FPorigin
- COP_X_m / COP_Y_m
- AnkleTorqueMid_*
- AnkleTorqueL_*
- AnkleTorqueR_*
- raw_wrench_payload["legacy_single_forceplate"]

B-2. 유지할 메타 컬럼
- inverse_dynamics_mode
- inverse_dynamics_forceplates

B-3. 새 raw forceplate 컬럼 체계
선택된 각 plate마다 다음 컬럼을 만든다.

예: fp3 선택 시
- FP3_GRF_X_N
- FP3_GRF_Y_N
- FP3_GRF_Z_N
- FP3_GRM_X_Nm_at_FPorigin
- FP3_GRM_Y_Nm_at_FPorigin
- FP3_GRM_Z_Nm_at_FPorigin
- FP3_COP_X_m
- FP3_COP_Y_m
- FP3_ContactValid

예: [fp1, fp3] 선택 시
- FP1_*
- FP3_*
만 만든다.
FP2_*는 만들지 않는다.

B-4. joint moment 컬럼 체계
이건 기존 inverse dynamics output을 그대로 활용한다.
- Hip_L_ref_X_Nm ~ Hip_R_ref_Z_Nm
- Knee_L_ref_X_Nm ~ Knee_R_ref_Z_Nm
- Ankle_L_ref_X_Nm ~ Ankle_R_ref_Z_Nm
- Trunk_ref_X_Nm ~ Neck_ref_Z_Nm

B-5. single_plate_strict 에서의 joint moment 정책
- 이번 patch에서는 strictness를 위해
  Hip/Knee/Ankle/Trunk/Neck 전체 *_ref_*_Nm 를 NaN으로 둔다.
- 이유:
  single-plate에서 lower limb만 막고 trunk/neck만 살리면
  내부적으로 다시 legacy single-wrench logic을 남기게 되기 때문이다.
- 즉 "계산 불가능한 값을 억지로 안 낸다"를 우선한다.

==================================================
C. 파일별 수정 지시
==================================================

------------------------------------------
C-1. scripts/run_batch_all_timeseries_csv.py
------------------------------------------

목표:
- legacy single-forceplate 계산을 완전히 제거
- raw forceplate를 per-plate 컬럼으로 export
- ankle torque 계열 export 중단
- logging 도입

수정 지시:

1) import 정리
- 제거:
  from replace_v3d.joint_dynamics import compute_joint_moment_columns
  from replace_v3d.torque.ankle_torque import compute_ankle_torque_from_net_wrench
- 유지:
  compute_joint_moment_columns_multi
- 필요 시 추가:
  import logging
  logger = logging.getLogger(__name__)

2) helper 추가
- _expected_forceplate_raw_cols(selected_forceplate_ids: list[int]) -> list[str]
- _build_forceplate_raw_payload(
    per_plate_payloads: list[ForceplateWrenchSeries],
    selected_forceplate_ids: list[int],
    end_frame: int,
    onset0: int,
  ) -> dict[str, np.ndarray]

동작:
- 선택된 plate 각각에 대해 FP{n}_GRF_*, FP{n}_GRM_*, FP{n}_COP_*, FP{n}_ContactValid 생성
- GRF/GRM은 onset0 기준 baseline subtraction 적용
- COP는 absolute coordinate 그대로 둔다
- ContactValid는 0/1 또는 bool 중 하나로 통일한다
  (권장: 0/1 int 보다 bool)

3) helper 추가
- _nan_joint_moment_payload_all(end_frame: int) -> dict[str, np.ndarray]
  -> _expected_joint_moment_cols() 전체를 NaN으로 채움

4) helper 제거 또는 퇴역
- _expected_ankle_torque_cols()
- _nan_ankle_torque_payload()

원칙:
- batch export 메인 경로에서 ankle torque family 자체를 더 이상 쓰지 않는다.
- 해당 helper가 다른 스크립트에서 안 쓰이면 삭제, 쓰이면 "legacy only"로 주석 처리.

5) single_plate_strict 분기 수정
현재:
- compute_ankle_torque_from_net_wrench(...) 호출
- legacy GRF/COP 채움
- legacy_single_forceplate 저장

수정 후:
- compute_ankle_torque_from_net_wrench(...) 호출 삭제
- joint_moment 계산 호출도 없음
- payload 에는 아래만 넣는다:
  - inverse_dynamics_mode = single_plate_strict
  - inverse_dynamics_forceplates
  - FP{n}_GRF_*
  - FP{n}_GRM_*
  - FP{n}_COP_*
  - FP{n}_ContactValid
  - joint_moment_payload = _nan_joint_moment_payload_all(...)
- raw_wrench_payload 는
  {
    "mode": "single_plate_strict",
    "forceplates": per_plate_payloads,
  }
  만 남긴다.
- legacy_single_forceplate 제거

6) multi_plate_v3d 분기 수정
현재:
- legacy GRF/COP를 NaN으로 채움
- ankle torque family도 NaN payload + 별도 계산 흔적이 남아 있음

수정 후:
- legacy GRF/COP payload 자체를 만들지 않음
- ankle torque family payload 자체를 만들지 않음
- payload 에는 아래만 넣는다:
  - inverse_dynamics_mode = multi_plate_v3d
  - inverse_dynamics_forceplates
  - FP{n}_GRF_*
  - FP{n}_GRM_*
  - FP{n}_COP_*
  - FP{n}_ContactValid
  - joint_moment_payload = compute_joint_moment_columns_multi(...)
- 즉 raw forceplate는 per-plate 새 체계만, lower-limb torque는 joint moment 컬럼만 쓴다

7) DataFrame export 점검
- _make_timeseries_dataframe(...)는 payload dict에 들어온 컬럼을 그대로 합친다.
- 여기서 AnkleTorque* / GRF_* / COP_* legacy 컬럼이 안 들어가도 문제 없도록 확인한다.
- CSV schema는 "이번 run의 config subset" 기준으로 안정적으로 유지되게 한다.

8) 로깅 전환
- print("[INFO] ...") -> logger.info(...)
- single_plate_strict / multi_plate_v3d mode 로그도 logger 사용
- per-trial main loop try/except 에서
  logger.exception(
    "[ID][FAIL] file=%s subject=%s velocity=%s trial=%s mode=%s forceplates=%s",
    ...
  )
  형태로 context 남기고 raise
- skip_unmatched 같은 비핵심 분기만 logger.warning / logger.info 사용

------------------------------------------
C-2. src/replace_v3d/joint_dynamics/inverse_dynamics.py
------------------------------------------

목표:
- 예외를 삼키지 않게 한다
- algorithmic invalid와 programming error를 분리한다

수정 지시:

1) logging 추가
- import logging
- logger = logging.getLogger(__name__)

2) compute_joint_moment_columns(...) 수정
현재:
- try:
    return _compute_joint_moment_columns_impl(...)
  except Exception:
    return nan_all

수정 후:
- body_mass 없음 / 비정상 / 입력 forceplate 없음 같은 "예상 가능한 invalid 입력"만 nan_all 반환
- 그 외 Exception은:
  logger.exception("[ID][single][FAIL] ...context...")
  raise
- bare except 금지

3) compute_joint_moment_columns_multi(...) 수정
현재:
- try:
    return _compute_joint_moment_columns_multi_impl(...)
  except Exception:
    return nan_all

수정 후:
- body_mass 없음 / forceplates 비어 있음 같은 expected invalid만 nan_all 반환
- shape mismatch / marker 누락 / assignment bug / index bug 등은
  logger.exception("[ID][multi][FAIL] ...context...")
  raise

4) context 강화
가능하면 logger.exception 안에 아래 정보 포함
- n_frames
- n_forceplates
- selected plate ids
- body_mass_kg
- labels count
- available joint center keys

5) 원칙
- frame-level invalid biomechanics:
  -> 현재처럼 mask 기반 NaN 처리 유지
- code bug / unexpected exception:
  -> 조용히 NaN으로 덮지 말고 실패시켜야 함

------------------------------------------
C-3. scripts/plot_grid_timeseries.py
------------------------------------------

목표:
- plot이 새 컬럼 체계만 보게 변경
- 임의 subset [fp1, fp3]도 자동 대응
- single_plate_strict 에서는 빈 그림 대신 의미 있는 표시

수정 지시:

1) config template expansion 기능 추가
새 config key 제안:
- expand_from_inverse_dynamics_forceplates: true
- title_template: "FP{fp} GRF / COP"
- series.col 에서 "FP{fp}_GRF_X_N" 같은 placeholder 허용

동작:
- config의 forceplate.analysis.use_for_inverse_dynamics 읽기
- 예: [fp1, fp3]이면 category template 하나를 FP1용, FP3용으로 2개 확장
- fp2는 자동 생성하지 않음

2) mode filter 기능 추가
category 또는 subplot 레벨에 아래 키 허용:
- mode_filter: [single_plate_strict, multi_plate_v3d]

동작:
- 현재 그룹(trial/subject/mean)에 해당 mode가 없으면
  subplot 전체를 "N/A for current mode"로 annotate 하거나
  category 자체를 skip

3) all-NaN subplot 처리
- series는 있는데 값이 전부 NaN이면
  빈 축만 남기지 말고
  "Unavailable in single_plate_strict" 같은 문구 표시
- 이유:
  single_plate_strict 에서는 joint moment가 전부 NaN이므로
  사용자가 버그로 오해하지 않게 해야 함

4) missing column validation 강화
- plot 시작 시 expanded categories의 모든 col 이름을 점검
- CSV에 없는 col이 config에 들어 있으면
  logger.error + ValueError 로 즉시 실패
- "조용히 빈 그래프" 금지

------------------------------------------
C-4. config.yaml
------------------------------------------

목표:
- plot category를 새 컬럼 체계로 전환
- legacy ankle torque / legacy GRF-COP category 제거
- subset template 기반으로 forceplate raw panel 자동 생성

수정 지시:

1) 유지
- forceplate.analysis.use_for_inverse_dynamics
예:
  [fp3]
  [fp1, fp3]

2) 삭제할 plotting category
- ankle_torque_internal
  (AnkleTorqueMid_int_*, AnkleTorqueL_int_*, AnkleTorqueR_int_* 참조 제거)
- grf_cop
  (GRF_X_N, GRF_Y_N, GRF_Z_N, COP_X_m, COP_Y_m 참조 제거)

3) 추가할 plotting category 예시
3-1. forceplate raw template
- tag: forceplate_raw_template
- expand_from_inverse_dynamics_forceplates: true
- title_template: "FP{fp} GRF / COP"
- mode_filter: [single_plate_strict, multi_plate_v3d]
- subplots:
  - FP{fp}_GRF_X_N
  - FP{fp}_GRF_Y_N
  - FP{fp}_GRF_Z_N
  - FP{fp}_COP_X_m
  - FP{fp}_COP_Y_m
  - FP{fp}_ContactValid

3-2. ankle joint moment category
- tag: joint_moment_ankle_ref
- title: "Ankle Joint Moment (ref)"
- mode_filter: [multi_plate_v3d]
- subplots:
  - Ankle_L_ref_X_Nm / Ankle_R_ref_X_Nm
  - Ankle_L_ref_Y_Nm / Ankle_R_ref_Y_Nm
  - Ankle_L_ref_Z_Nm / Ankle_R_ref_Z_Nm

3-3. knee joint moment category
- Knee_L_ref_* / Knee_R_ref_*

3-4. hip joint moment category
- Hip_L_ref_* / Hip_R_ref_*

4) 예시 설명
- use_for_inverse_dynamics: [fp3]
  -> plotting은 FP3 raw panel만 자동 생성
  -> ankle/knee/hip moment panel은 mode_filter 또는 all-NaN 처리로 숨김/표시
- use_for_inverse_dynamics: [fp1, fp3]
  -> plotting은 FP1 raw panel + FP3 raw panel 생성
  -> fp2 panel은 없음

==================================================
D. 새 컬럼 체계 명세
==================================================

D-1. raw forceplate columns
plate n에 대해:
- FP{n}_GRF_X_N
- FP{n}_GRF_Y_N
- FP{n}_GRF_Z_N
- FP{n}_GRM_X_Nm_at_FPorigin
- FP{n}_GRM_Y_Nm_at_FPorigin
- FP{n}_GRM_Z_Nm_at_FPorigin
- FP{n}_COP_X_m
- FP{n}_COP_Y_m
- FP{n}_ContactValid

D-2. joint moment columns
- Hip_L_ref_X_Nm ... Hip_R_ref_Z_Nm
- Knee_L_ref_X_Nm ... Knee_R_ref_Z_Nm
- Ankle_L_ref_X_Nm ... Ankle_R_ref_Z_Nm
- Trunk_ref_X_Nm ... Neck_ref_Z_Nm

D-3. metadata columns
- inverse_dynamics_mode
- inverse_dynamics_forceplates

D-4. 제거 대상
- GRF_*
- GRM_*
- COP_*
- AnkleTorque*

==================================================
E. 구현 순서
==================================================

Step 1.
- run_batch_all_timeseries_csv.py 에서 per-plate raw payload helper 작성
- FP{n}_* 컬럼 생성 경로부터 먼저 완성

Step 2.
- single_plate_strict branch에서
  compute_ankle_torque_from_net_wrench 호출 삭제
  legacy_single_forceplate 제거
  joint moment all-NaN 정책 적용

Step 3.
- multi_plate_v3d branch에서
  legacy GRF/COP / AnkleTorque 제거
  per-plate raw + joint moment만 export

Step 4.
- inverse_dynamics.py 에서 bare except 제거
  logger.exception + raise 로 교체

Step 5.
- plot_grid_timeseries.py 에
  expand_from_inverse_dynamics_forceplates
  mode_filter
  all-NaN annotate
  missing-column validation 추가

Step 6.
- config.yaml 의 legacy plotting category 삭제
  새 category template 추가

Step 7.
- dead code 정리
  - compute_joint_moment_columns import 제거 여부 확인
  - ankle torque helper 삭제/퇴역
  - 사용하지 않는 baseline 보정 루프 제거

==================================================
F. 검증 계획
==================================================

검증 파일:
- 0904_김종철_perturb_135_001.c3d

Case 1. single plate
config:
  use_for_inverse_dynamics: [fp3]

기대 결과:
- CSV에 FP3_GRF_*, FP3_GRM_*, FP3_COP_*, FP3_ContactValid 존재
- GRF_*, COP_*, AnkleTorque* legacy 컬럼 없음
- *_ref_*_Nm 전체 NaN
- 로그:
  mode=single_plate_strict
  lower-limb torque/joint moment skipped...
- plot:
  FP3 raw figure만 정상
  ankle/knee/hip moment figure는 숨김 또는 N/A 표시

Case 2. non-contiguous multi-plate
config:
  use_for_inverse_dynamics: [fp1, fp3]

기대 결과:
- CSV에 FP1_* 와 FP3_*만 존재
- FP2_* 없음
- mode=multi_plate_v3d
- joint moment 컬럼 계산됨
- plot:
  FP1 raw / FP3 raw panel만 생성
  fp2 panel 없음

Case 3. 예외 로깅
방법:
- 테스트용으로 compute_joint_moment_columns_multi 내부에서 임시 RuntimeError("test") 발생
기대 결과:
- logger.exception 으로 traceback + trial context 남음
- 조용히 nan_all 반환하지 않음
- run 이 실패하거나 상위에서 명시적으로 처리됨

Case 4. plotting config 검증
- config에 없는 컬럼명 오타를 일부러 넣어본다
기대 결과:
- plot_grid_timeseries.py 가 시작 초기에 ValueError
- 어떤 category/subplot/series가 문제인지 로그로 알려줌

==================================================
G. 완료 기준 (Definition of Done)
==================================================

- single_plate_strict 에서 legacy single-wrench 계산 호출이 0회
- raw_wrench_payload 에 legacy_single_forceplate 키가 없음
- batch export 결과에 legacy GRF_*, GRM_*, COP_*, AnkleTorque* 컬럼이 없음
- 새 per-plate FP{n}_* 컬럼이 config subset 기준으로만 생성됨
- [fp1, fp3]에서 FP2_*가 생성되지 않음
- plotting config에 legacy ankle torque / legacy grf_cop category가 없음
- plot_grid_timeseries 가 selected subset 기준으로 forceplate panel을 자동 생성
- inverse_dynamics.py 에 bare except 없음
- 예외 발생 시 traceback + trial context 로그가 남음
- 제공된 sample c3d 로 Case 1 / Case 2 검증 통과

==================================================
H. 한 줄 요약
==================================================

- 레거시 단일 컬럼/단일 plate 흔적을 완전히 끊고,
- forceplate raw는 FP{n}_* 새 컬럼 체계로 통일하고,
- lower-limb torque는 joint moment(*_ref_*_Nm)만 진짜 출력으로 쓰며,
- plotting은 config template로 [fp1, fp3] 같은 임의 subset까지 자동 대응하고,
- 예외는 더 이상 조용히 NaN으로 삼키지 않고 반드시 로그+traceback을 남긴다.