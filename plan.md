EXCEPLAN
주제: single-plate strict mode에서 Trunk/Neck moment를 살리고,
      logging을 logger로 통일하고,
      문서/주석을 실제 동작과 맞춘다.
      단, forceplate selection semantics([fp1], [fp1, fp3], [fp1, fp2, fp3])는 절대 바꾸지 않는다.

1) 목표
- 현재 유지할 것
  - forceplate.analysis.use_for_inverse_dynamics 의 "임의 부분집합" 의미
  - single_plate_strict / multi_plate_v3d 분기 구조
  - multi-plate force assignment 물리 로직
  - ankle torque NaN 정책

- 이번에 수정할 것
  A. single-plate strict mode에서도 Trunk_ref_*_Nm, Neck_ref_*_Nm 은 계산/보존
  B. print(...) 기반 INFO 출력 제거, logger.info(...)로 통일
  C. docstring / module comment / log wording 을 실제 동작과 맞춤

2) 비목표
- forceplate selection parser 수정 금지
- [fp1, fp3] 지원 방식 수정 금지
- compute_joint_moment_columns_multi()의 물리 계산 로직 변경 금지
- assignment threshold / invalidation policy 변경 금지
- ankle torque physics 변경 금지

3) 수정 파일
- scripts/run_batch_all_timeseries_csv.py
- src/replace_v3d/joint_dynamics/inverse_dynamics.py

4) 구현 원칙
- single-plate에서 "lower-limb만 막고 trunk/neck는 살리는" 방향으로 간다.
- 가장 단순하고 안전한 방식:
  "single-wrench legacy joint moment를 한 번 계산한 뒤,
   lower-limb columns만 NaN 마스킹하고,
   Trunk/Neck columns는 그대로 둔다."
- logging은 batch / inverse_dynamics 모두 logger.info(...) 사용
- 문서/주석은 "무엇을 계산하고, 무엇을 일부러 NaN 처리하는지"를 정확히 적는다.

5) 상세 작업 - run_batch_all_timeseries_csv.py
5-1. single-plate branch 수정
대상:
- if len(selected_platforms) == 1:  ...  branch

수정 지시:
- 현재 single-plate branch는 raw GRF/GRM/COP payload를 만들고,
  ankle torque columns를 NaN 처리하는 구조를 유지한다.
- 여기에 "single-plate용 joint moment payload 생성"을 추가한다.
- 구현 위치는 "joint_moment_payload가 실제로 조립되는 곳"에 둔다.
  같은 계산을 helper 내부/외부에서 중복 호출하지 말 것.

권장 구현:
- legacy_single_forceplate raw wrench
  - fp_origin_lab
  - grf_lab
  - grm_lab_at_fp_origin
  - cop_x_m
  - cop_y_m
  를 그대로 재사용한다.
- compute_joint_moment_columns(...) 를 사용해 single-wrench full moment payload를 얻는다.
- 얻은 payload에서 아래 lower-limb columns만 NaN으로 덮어쓴다:
  - Hip_L_ref_*
  - Hip_R_ref_*
  - Knee_L_ref_*
  - Knee_R_ref_*
  - Ankle_L_ref_*
  - Ankle_R_ref_*
- Trunk_ref_* / Neck_ref_* 는 그대로 유지한다.

주의:
- single-plate strict mode의 의미는 유지한다.
  즉, "개별 lower-limb joint moment는 금지"는 그대로다.
- trunk/neck를 살리는 것은 "lower-limb 금지 정책 완화"가 아니라
  "axial chain output만 보존"이다.

5-2. single-plate log wording 수정
현재 로그 문구는 너무 넓다.
아래처럼 더 정확하게 바꾼다.

기존 의미:
- lower-limb torque/joint moment skipped because only one forceplate was selected

수정 후 의미:
- mode=single_plate_strict
- lower-limb torque/joint moments skipped because only one forceplate was selected
- trunk/neck joint moments are preserved from the legacy single-wrench path

주의:
- 위 문장은 logger.info(...) 로 출력
- print(...) 사용 금지

5-3. batch module top docstring 수정
현재 상단 설명에서
"internal joint moments (*_ref_*_Nm) are computed only in multi-plate inverse dynamics mode"
같은 식으로 읽히는 문장을 고친다.

수정 방향:
- lower-limb internal joint moments only in multi-plate mode
- single-plate strict mode keeps Trunk_ref/Neck_ref and skips lower-limb joint moments
- 표현을 이 뜻으로 명확히 바꾼다

6) 상세 작업 - inverse_dynamics.py
6-1. lower-limb 전용 NaN helper 추가
현재 _nan_all_joint_moment_cols(T)는 trunk/neck까지 전부 NaN 만든다.
이번 수정에서는 lower-limb만 NaN 만드는 helper를 분리한다.

추가할 helper 예시:
- _nan_lower_limb_joint_moment_cols(T)

포함할 prefix:
- Hip_L_ref
- Hip_R_ref
- Knee_L_ref
- Knee_R_ref
- Ankle_L_ref
- Ankle_R_ref

유지할 것:
- 기존 _nan_all_joint_moment_cols(T)는 삭제하지 말고 남겨둔다.
  이유:
  - full fallback 용도로 여전히 유용할 수 있음
  - body mass invalid / forceplate 없음 / 예외 fallback 같은 곳에서 계속 쓸 수 있음

6-2. selective masking helper 추가
아래 중 하나 선택:
- _mask_lower_limb_joint_moment_payload(payload)
또는
- batch 쪽에서 payload.update(_nan_lower_limb_joint_moment_cols(T)) 방식 사용

권장:
- inverse_dynamics.py에 helper를 두고 재사용 가능한 형태로 만든다.
- 역할이 명확하게 보이도록 함수명을 쓴다.

6-3. logging 통일
inverse_dynamics.py 상단에
- import logging
- logger = logging.getLogger(__name__)
추가

아래 print(...)를 전부 logger.info(...) 로 교체:
- mode log
- force assignment summary heading
- assigned block summary 각 줄
- no contact blocks detected
- multi-plate ankle torque columns are set to NaN by design
- 그 외 INFO 성격의 print 전부

주의:
- summary block을 여러 줄로 찍더라도 print 금지
- logger.info("force assignment summary:")
- logger.info("  %s", _format_assigned_block_summary(block))
형태로 통일

6-4. docstring / comment 정합성 수정
대상 1:
- compute_joint_moment_columns_multi(...)

현재 설명:
- lower-limb joint moments from selected multi-plate set

수정 방향:
- "selected multi-plate forceplate set에서 joint moments를 계산한다"
- "lower-limb outputs + Trunk_ref/Neck_ref outputs를 반환한다"
- "lower-limb validity는 force assignment mask를 따른다"
라는 의미가 드러나게 바꾼다.

대상 2:
- _nan_all_joint_moment_cols(T)

수정 방향:
- "all joint moment columns including Trunk_ref and Neck_ref"라는 뜻이 드러나게
  docstring 또는 바로 위 comment 추가

대상 3:
- single-plate 관련 comment
수정 방향:
- "single_plate_strict = lower-limb blocked, trunk/neck kept"
라는 뜻이 분명히 보이게 정리

7) 구현 순서
Step 1.
- inverse_dynamics.py에 logging import/logger 추가
- print(...) -> logger.info(...) 교체
- grep으로 남은 print 정리

Step 2.
- _nan_lower_limb_joint_moment_cols(T) 추가
- 필요하면 selective masking helper 추가

Step 3.
- single-plate strict mode의 joint moment assembly 지점 수정
- compute_joint_moment_columns(...)로 full single-wrench payload 계산
- lower-limb만 NaN mask
- trunk/neck 유지

Step 4.
- batch single-plate 로그 문구 수정
- module top docstring 수정

Step 5.
- compute_joint_moment_columns_multi() docstring, helper comments 수정

8) 검증 계획
입력 샘플:
- 0904_김종철_perturb_135_001.c3d

Case A. single-plate strict
config:
- use_for_inverse_dynamics: [fp3]

기대 결과:
- inverse_dynamics_mode == single_plate_strict
- Trunk_ref_X/Y/Z_Nm 존재
- Neck_ref_X/Y/Z_Nm 존재
- Trunk/Neck columns가 "전부 NaN"이면 실패
- Hip_L/R_ref_*, Knee_L/R_ref_*, Ankle_L/R_ref_* 는 전부 NaN
- ankle torque columns는 기존 정책대로 NaN 유지
- stdout에 print 기반 INFO가 보이지 않고 logger output만 보임

Case B. arbitrary subset 유지 확인
config:
- use_for_inverse_dynamics: [fp1, fp3]

기대 결과:
- multi_plate_v3d로 들어감
- selection semantics unchanged
- [fp1, fp3] 그대로 사용
- trunk/neck + lower-limb moment columns 모두 기존 multi-plate 정책대로 계산
- assignment summary 출력이 logger로만 남음

Case C. doc/comment sanity check
기대 결과:
- batch top docstring이 "single-plate에서도 trunk/neck는 남는다"는 뜻과 맞음
- compute_joint_moment_columns_multi() docstring이 실제 반환 컬럼과 맞음
- _nan_all_joint_moment_cols 설명이 trunk/neck 포함 사실과 맞음

9) grep / 리뷰 체크리스트
- grep -n "print(" scripts/run_batch_all_timeseries_csv.py src/replace_v3d/joint_dynamics/inverse_dynamics.py
  -> 이번 수정 대상 INFO 출력에서 print가 남아 있으면 실패

- grep -n "compute_joint_moment_columns_multi" src/replace_v3d/joint_dynamics/inverse_dynamics.py
  -> docstring이 lower-limb only처럼 오해되면 실패

- single-plate output CSV 확인
  -> Trunk_ref/Neck_ref finite
  -> lower-limb moment NaN

10) 완료 기준 (Definition of Done)
- [fp1, fp3] 같은 arbitrary subset 동작은 그대로 유지
- single_plate_strict에서 Trunk_ref/Neck_ref는 보존
- single_plate_strict에서 lower-limb joint moments는 NaN
- targeted print(...)가 logger.info(...)로 전부 교체
- batch top docstring / multi-plate function docstring / helper comment가 실제 동작과 일치
- sample c3d로 [fp3] 검증 통과