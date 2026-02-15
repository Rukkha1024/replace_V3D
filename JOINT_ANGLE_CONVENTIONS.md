## 관절각 출력 컨벤션 (raw vs anat vs ana0)

본 저장소는 마커 기반 세그먼트 좌표계로부터 Visual3D 방식의 3D 관절각(내재적 XYZ)을
계산합니다.

중요: **raw 관절각 계산 자체는 아래 컨벤션에 의해 변경되지 않습니다.**
아래 컨벤션은 내보낸 시계열에 적용되는 *후처리*입니다.

---

## Raw 컨벤션 (`*_JOINT_ANGLES_preStep.csv`)

Raw 출력은 다음 과정의 직접적인 결과입니다:

- 세그먼트 프레임 구성
- 상대 회전 (근위 → 원위)
- 내재적 XYZ 오일러 분해

Raw 출력은 재현성과 MD5 검증을 위해 변경 없이 유지됩니다.

각도 열은 `*_deg`로 끝나며, 다음을 포함합니다:

- Hip / Knee / Ankle: 좌우 구분 (예: `Hip_L_Y_deg`, `Hip_R_Y_deg`)
- Trunk / Neck: 좌우 구분 없음 (예: `Trunk_Y_deg`)

---

## 해부학적 표현 컨벤션 (`*_anat.csv`)

`*_anat.csv`는 raw 관절각의 후처리 사본으로, 단일 목표를 가집니다:

좌우 비교 시 Y/Z 부호의 의미를 LEFT와 RIGHT 간에 일관되게 만드는 것.

**Hip/Knee/Ankle** (왼쪽만 해당):

- `*_L_Y_deg = - *_L_Y_deg`
- `*_L_Z_deg = - *_L_Z_deg`

기저선 차감은 수행하지 않습니다.

### `_anat` 적용 후 실질적 해석

부호 반전 후 (RIGHT를 기준 의미로 사용):

- **Y 양수:** 내전(adduction) (좌우 동일)
- **Z 양수:** 내회전(internal rotation) (좌우 동일)

X는 변경 없음.

---

## 기저선 정규화 컨벤션 (`*_ana0.csv`)

`*_ana0.csv`는 `_anat`과 동일한 부호 반전 각도에서 시작한 뒤, 정적 기립
기저선을 차감하여 정적 오프셋을 제거합니다.

- 기저선 구간: **프레임 1..11** (양 끝 포함)
- 모든 `*_deg` 열에 대해:
  - `angle = angle - mean(angle[1..11])`

이는 **Δ각도** 비교 및 세그먼트 좌표계 정렬 오차로 인한 작은 정적 오프셋
제거에 유용합니다.

---

## 구현 위치

- 후처리 로직: `src/replace_v3d/joint_angles/postprocess.py`
- 단일 시행 내보내기 (raw + `_anat` + `_ana0`): `scripts/run_joint_angles_pipeline.py`
- 배치 통합 CSV 선택적 접미사 열:
  - `--angles_anat` → `*_deg_anat` 추가
  - `--angles_ana0` → `*_deg_ana0` 추가
  - 스크립트: `scripts/run_batch_all_timeseries_csv.py`
