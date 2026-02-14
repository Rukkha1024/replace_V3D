너 요구대로 **“time series만”** 뽑아서 **한 번에(배치) 돌리고, 결과를 하나의 CSV로** 만들 수 있게 **배치 스크립트 하나를 추가**해뒀어. 또한 AGENTS에서 말하는 최소 키 단위가 `subject-velocity-trial`인 것도 그대로 지킴. ([GitHub][2])

---

## 변경 사항 요약 (time series 전용)

추가된 스크립트:

* `scripts/batch/run_batch_mos_timeseries_csv.py`

  * `data/all_data` 아래의 모든 `.c3d`를 순회
  * 파일명에서 `(subject_token, velocity, trial)` 파싱
  * `perturb_inform.xlsm`에서 subject 매칭(토큰→subject)
  * `meta/transpose_meta`에서 **다리길이(leg length)** 읽어서 xCOM 계산(= MOS 계산에 필요)
  * **COM / vCOM / xCOM / BOS / MOS time series**를 프레임 단위로 생성
  * **`MocapFrame` 컬럼명으로 저장** (요구사항)
  * 출력은 **long-format 단일 CSV** (행 = trial×frame)

> step trial은 repo 로직대로 `step_onset_local - 1`까지만 time series를 뽑음(= preStep). ([GitHub][1])

---

## 출력 CSV 스키마 (핵심)

최소 필수 컬럼(너가 지정한 minimum unit):

* `subject-velocity-trial`
* `MocapFrame`

그리고 같이 들어가는 주요 컬럼들(예:):

* `subject`, `velocity`, `trial`, `c3d_file`, `rate_hz`
* `Time_s`
* `COM_X/Y/Z`, `vCOM_X/Y/Z`, `xCOM_X/Y/Z`
* `BOS_area`, `BOS_minX/maxX/minY/maxY`
* `MOS_minDist_signed`, `MOS_AP_dir`, `MOS_ML_dir`
* `Is_platform_onset_frame`, `Is_step_onset_frame`

요약치(평균/최소/최대 등) **전혀 계산/출력 안 함**.

---

## 실행 방법 (AGENTS 준수: conda env `module`)

AGENTS의 환경 규칙대로 `module` env로 실행: ([GitHub][2])

```bash
conda run -n module python scripts/batch/run_batch_mos_timeseries_csv.py \
  --c3d_dir data/all_data \
  --event_xlsm data/perturb_inform.xlsm \
  --out_csv output/all_trials_mos_timeseries.csv \
  --overwrite
```

옵션:

* `--skip_unmatched` : subject 매칭/이벤트 매칭 실패한 파일은 스킵하고 계속 진행

---

## 중요한 주의: 파일명 토큰(KUO 등) → subject 매칭 규칙

너가 말한 “파일명 기반으로 subject 매칭”이 되려면, 엑셀에 **토큰↔subject 매핑 정보가 있어야** 해.

스크립트가 자동으로 찾는 우선순위는:

1. `platform` 시트의 `subject` 값이 토큰과 직접 일치
2. `meta` 시트에서 토큰이 subject 컬럼 헤더로 직접 존재
3. `meta` 시트에 **이니셜/alias/code/id 류 row key**가 있고, 그 행의 셀 값이 토큰과 일치
4. `transpose_meta` 시트에 **이니셜/alias/code/id 류 컬럼**이 있고, 그 값이 토큰과 일치 → 해당 행의 `subject` 반환

즉 파일명이 `251112_KUO_perturb_60_001.c3d` 같은 형태인데 `platform.subject`가 “김우연”처럼 한글 이름이면, (3)이나 (4) 방식의 매핑(예: meta에 “이니셜” 행 추가)이 필요해.

---

## 패치/수정 파일

* [Download patch (git apply용)](sandbox:/mnt/data/replace_V3D_timeseries_only_gitapply.patch)
* [Download modified file zip](sandbox:/mnt/data/replace_V3D_timeseries_only_files.zip)

적용 예시:

```bash
git apply replace_V3D_timeseries_only_gitapply.patch
```

커밋 메시지(AGENTS 요구: 한글): 예) `MoS 타임시리즈 배치 CSV 추출 스크립트 추가` ([GitHub][2])

