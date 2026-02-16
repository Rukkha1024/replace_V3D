## 관절각 출력 컨벤션 (표준 출력 = ana0)

본 저장소는 마커 기반 세그먼트 좌표계로부터 Visual3D 방식의 3D 관절각(내재적 XYZ)을 계산합니다.

이제 **관절각의 표준 출력은 ana0 하나만** 사용합니다.

- 저장 파일: `*_JOINT_ANGLES_preStep.csv`
- 의미: **ana0 = (좌우 부호 통일) + (quiet standing 기저선 차감)**

---

## ana0 정의 (이 저장소의 표준)

ana0는 아래 2단계를 순서대로 적용한 관절각 시계열입니다.

### 1) 좌우 부호 통일 (LEFT만 적용)

Hip/Knee/Ankle의 **LEFT** Y/Z 성분만 부호를 반전합니다.

- `*_L_Y_deg = - *_L_Y_deg`
- `*_L_Z_deg = - *_L_Z_deg`

목적: 좌/우 비교 시 Y/Z 부호의 의미를 LEFT와 RIGHT 간에 일관되게 맞춤.

부호 반전 후 (RIGHT 기준 의미 통일):

| 축 | 양수 의미 | 비고 |
|----|----------|------|
| X  | (변경 없음) | 좌우 동일 |
| Y  | 내전(adduction) | 좌우 동일 |
| Z  | 내회전(internal rotation) | 좌우 동일 |

### 2) quiet standing 기저선 차감

정적 기립 구간 평균을 빼서 정적 오프셋을 제거합니다.

- 기저선 구간: **프레임 1..11** (양 끝 포함)
- 모든 `*_deg` 열에 대해:
  - `angle = angle - mean(angle[1..11])`

---

## 구현 위치

- 후처리 로직: `src/replace_v3d/joint_angles/postprocess.py`
- 단일 시행 내보내기(ana0만 저장): `scripts/run_joint_angles_pipeline.py`
- 배치 통합 CSV 내 관절각(ana0 값 사용): `scripts/run_batch_all_timeseries_csv.py`
