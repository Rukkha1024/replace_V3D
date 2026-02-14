요청한 대로 `replace_V3D` repo 기준으로 **V3D 방식(ForcePlatform CORNERS 기반 좌표변환 + GRF/GRM → 임의 점 모멘트 이송)**으로 **양발 전체(=net) ankle torque** 파이프라인을 repo에 추가했습니다.
(제 환경에선 GitHub에 직접 push는 불가라서, **git apply 가능한 patch**와 **업데이트 파일 묶음 zip**을 제공드립니다.)

## 변경/추가된 내용 (repo 반영용)

### 1) 신규 파이프라인

* `scripts/run_ankle_torque_pipeline.py`

  * 입력: C3D + `perturb_inform.xlsm`(platform sheet) + subject
  * 출력: `<trial>_ankle_torque.xlsx`
  * force plate는 **자동 선택(전체 구간 |Fz| 최대)** 또는 `--force_plate`로 강제 지정 가능

### 2) ForcePlate 파서 + 좌표계 변환

* `scripts/forceplate.py`

  * C3D의 `FORCE_PLATFORM` 그룹(`USED/TYPE/CORNERS/ORIGIN/CHANNEL`)과 `ANALOG` 그룹을 읽어서,
  * 1000Hz analog를 **프레임당 10샘플 평균**으로 100Hz로 다운샘플(=V3D export 방식과 동일하게 맞춤)
  * CORNERS로 plate→lab 회전행렬 구성 후 **GRF/GRM을 lab 좌표로 회전**

### 3) Ankle torque 계산 모듈

* `scripts/ankle_torque.py`

  * 외력 모멘트: `M_A_ext = M_O + (r_O - r_A) × F`
  * 내력 모멘트: `M_A_int = -M_A_ext`
  * **주의:** forceplate가 1개(net)라서 L/R은 “해당 위치로 모멘트를 이송했을 때의 값”이지, **진짜 좌/우 분해 토크는 아닙니다**. (양발 전체 토크는 `AnkleTorqueMid_*`가 핵심)

### 4) 이벤트/신체정보 반영

* `scripts/events.py`

  * `perturb_inform.xlsm`의 `meta` 시트에서 **몸무게(kg)**를 읽는 함수 추가
    (`row_key="몸무게"`, column=subject)
  * 본 케이스(김우연)는 53 kg로 자동 로드되어 `Nm/kg` 컬럼까지 채워집니다.

### 5) README 및 gitignore 업데이트

* `README.md`: ankle torque 실행 예시 추가
* `.gitignore`: `*_ankle_torque.xlsx` 추가

### 6) AGENTS 규칙 관련 환경 메모

* `.codex/skills/env_issues.md` 추가:

  * 이 sandbox에는 `conda`/`polars`가 없어서 여기선 시스템 python+pandas로 검증했고,
  * 실제 랩 환경에선 `conda run -n module ...`로 돌리도록 기록했습니다.

---

## 적용 방법 (당신의 GitHub repo에서)

### A) git patch로 적용 (추천)

아래 patch를 repo 루트에서 적용:

```bash
cd replace_V3D
git apply ankle_torque_patch.diff
```

* 패치 파일: [Download ankle_torque_patch.diff](sandbox:/mnt/data/ankle_torque_patch.diff)

### B) 파일 zip을 풀어서 덮어쓰기

아래 zip에는 **추가/수정된 파일만** 들어 있습니다.

* [Download replace_V3D_ankle_torque_update.zip](sandbox:/mnt/data/replace_V3D_ankle_torque_update.zip)

---

## 실행 커맨드 (repo에서)

```bash
conda run -n module python scripts/run_ankle_torque_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --out_dir output
```

옵션:

* `--force_plate 3` : force plate 인덱스(1-based) 강제 지정
* `--mass_kg 53` : 몸무게를 수동 지정(기본은 xlsm meta에서 자동 로드)

---

## 산출물 포맷 (xlsx)

시트 3개:

1. `ankle_torque`

   * `GRF_*`, `GRM_*`, `COP_*`, `FP_origin_*`
   * `L_ankleJC_*`, `R_ankleJC_*`, `AnkleMid_*`
   * `AnkleTorqueMid_ext/int_*`, `AnkleTorqueL_*`, `AnkleTorqueR_*`
   * `AnkleTorqueMid_int_Y_Nm_per_kg` (몸무게 있으면 계산)
2. `meta`

   * platform onset/offset/step onset(원본 프레임, 파일 내 프레임), 샘플링레이트, 선택된 FP 등
3. `force_plate{n}_corners`

   * 선택된 plate의 4개 corner 좌표(랩 좌표, m)

---

## 이 환경에서의 검증 결과

* 샘플 파일(`251112_KUO_perturb_60_001.c3d`) 기준:

  * 자동 선택된 force plate: **3번**
  * platform onset file frame(0-based): **100** (→ time_from_platform_onset_s = 0)
  * body mass: **53 kg** (xlsm meta에서 로드)
* 제가 이전에 만든 참조 xlsx와 비교했을 때:

  * 컬럼/행 구조 동일
  * 수치 차이는 엑셀 저장/부동소수점 반올림 수준(최대 약 0.01 이내)
  * 단, xlsx는 메타데이터 때문에 **MD5는 달라질 수 있음**(숫자는 사실상 동일)

검증용으로 이 환경에서 생성한 결과 파일도 같이 첨부합니다:

* [Download 251112_KUO_perturb_60_001_ankle_torque.xlsx](sandbox:/mnt/data/ankle_output/251112_KUO_perturb_60_001_ankle_torque.xlsx)

---

## (참고) 내가 사용한 “skills”

* Spreadsheet 생성/검증(엑셀 출력), Python 코드 실행, repo 패치 생성/적용 흐름

원하면 다음 단계로,

* 여러 trial batch 처리(폴더 단위 반복 실행),
* step vs non-step 조건별로 platform onset 이후 구간만 자동 slice/export,
* 논문 초고 기준 성능변인(performance) 정의에 맞춘 요약변수(피크/적분/latency 등) 자동 산출
  까지 repo에 이어서 붙일 수 있어요.
