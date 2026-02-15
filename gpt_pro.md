아래는 **“shared_files(@stages/01_build_dataset.py)에서 쓰는 forceplate 관성(inertial) subtract 결과를 replace_V3D에서 재현/적용해서, C3D의 forceplate(analog) 값을 사실상 대체”**하기 위한 **코드 수정 플랜 + 좌표축 일치 검증 프로세스**입니다.
(요구하신 방향대로 **Stage 01 로직 전체를 이식하지 않고**, *보정값(템플릿/오프셋)*만 **추출→저장→replace_V3D에서 로드/적용**하는 구조로 잡았습니다.) ([GitHub][1])

## ✅ 현재 구현 상태 (2026-02-15)

현재 `replace_V3D`에는 **shared_files Stage01과 동일 의미의 forceplate inertial subtract**가 이미 구현되어 있습니다.

* **기본 모드 고정**: subtract는 항상 적용됩니다(비활성화 CLI 옵션 제거).
* **템플릿 필수**: `assets/fp_inertial_templates.npz`가 없으면 forceplate 토크 계산은 중단됩니다(= raw로 진행하지 않음).
* **적용 지점**: ankle torque 계산에서 wrench/COP/torque 계산 **이전**에 forceplate analog 6채널(Fx,Fy,Fz,Mx,My,Mz)에 subtract가 적용됩니다.

  * 단일 실행: `scripts/run_ankle_torque_pipeline.py`
  * 배치 통합 CSV: `scripts/run_batch_all_timeseries_csv.py`
* **shared_files 결과 일치 검증**: `scripts/verify_fp_inertial_subtract_vs_shared_files.py`

  * `shared_files/output/01_dataset/forceplate_subtract_diagnostics.parquet`과 비교
  * 전체 unit 기준 `failed=0` 확인
* **템플릿 재생성(1회 작업)**: `scripts/torque_build_fp_inertial_templates.py` → `assets/fp_inertial_templates.npz`
* **Stage1 shift(corrections.csv) 적용은 현재 미구현**이며, 본 repo에서는 관련 CSV를 기본으로 관리하지 않습니다.

---

## 0) 전제 정리 (현재 코드가 하는 일 / 우리가 해야 하는 일)

### replace_V3D 쪽(현재)

* ankle torque 파이프라인은 C3D의 `FORCE_PLATFORM` 메타데이터 + `ANALOG` 채널을 읽어 **plate 좌표(Fx,Fy,Fz,Mx,My,Mz)** → `R_pl2lab`로 **lab 좌표**로 회전해 사용합니다. ([GitHub][2])
* 즉, 지금은 **C3D에 들어있는 forceplate analog가 그대로** 토크 계산에 들어갑니다. ([GitHub][3])

### shared_files Stage01 쪽(근거 로직)

* `config.yaml`의 `forceplate.inertial_removal.subtract` 설정(언로드 데이터, 타이밍 xlsx, 파일 패턴, apply_channels 등)을 기반으로 **velocity별 unloaded 템플릿을 만들고**, trial 데이터에서 그것을 빼서 corrected를 만듭니다. ([GitHub][4])
* subtract 후 COP도 `Cx=-My/Fz`, `Cy=Mx/Fz`로 일관되게 다시 계산합니다. ([GitHub][1])
* 또한 shared_files는 (필요 시) axis_transform을 통해 forceplate 축 정의를 맞춥니다. ([GitHub][4])

### 우리가 구현해야 하는 것(요구사항)

1. replace_V3D의 `.c3d`에 있는 forceplate analog가 **관성 subtract가 안 된 값**이므로
2. shared_files Stage01의 **inertial subtract와 동일한 보정값(= unloaded 템플릿, 필요시 stage1 shift)**으로
3. replace_V3D에서 읽는 forceplate를 **“대체(= corrected를 사용)”**하도록 만들기
4. “c3d는 이미 좌표축 변환됨 / shared_files는 코드로 축 보정”이므로
   **replace 후 축 정의가 C3D와 일치하는지 검증 프로세스 포함**

---

## 1) 전략: “보정값만 추출해서 replace_V3D에서 사용” (권장)

요구하신 아이디어 그대로 구조를 잡습니다:

### (A) 보정값 아티팩트(작은 파일) 2종을 만든다

1. **inertial subtract 템플릿**: velocity별 `{Fx,Fy,Fz,Mx,My,Mz}` 시간파형(보통 100Hz 기준)

   * shared_files config의 `unload_data_dir + timing_xlsx + filename_pattern`로 만들어짐 ([GitHub][4])
2. (선택) **Stage1 shift corrections**: trial별 constant shift (window 적용)

   * Stage01에서 `data\corrections.csv`로 출력되는 값(shift_fx_n, shift_n, shift_mx_nm …) ([GitHub][1])
   * *단, 이건 원본 parquet가 필요해서(shared_files repo에 미푸시) 지금 당장 자동 생성은 불가할 수 있음.* (사용자 로컬에 parquet 있을 때만 생성 가능)

### (B) replace_V3D에서 “C3D를 읽은 뒤, analog를 corrected로 교체”한다

* C3D 바이너리 자체를 재작성하지 않고도(복잡/리스크 큼),
  **파이프라인 내부에서 `analog_avg`를 corrected로 바꿔치기**하면 “실질적 replace”가 됩니다.
* ankle torque가 이미 `read_force_platforms(c3d_path) → analog_avg`를 쓰고 있으니, 그 지점에서 교정값을 적용하면 됩니다. ([GitHub][3])

---

## 2) replace_V3D 수정 설계 (파일/모듈 단위)

### 2.1 새로 추가할 모듈: `src/replace_v3d/torque/forceplate_inertial.py` (신규)

핵심 기능 3개로 쪼갭니다.

#### (1) 템플릿 로더

* `assets/fp_inertial_templates.npz` 같은 파일을 로드해서
  `templates[velocity_int] -> dict(channel->np.ndarray, meta…)` 형태로 반환.

#### (2) inertial subtract 적용기

입력:

* `analog_avg` : `(n_frames, n_analog)`  (replace_V3D는 이미 **mocap frame 평균 analog**를 쓰고 있음) ([GitHub][2])
* `fp` : 선택된 `ForcePlatform` (채널 인덱스 포함)
* `events` : platform_onset_local / platform_offset_local
* `velocity` : filename or args로 얻은 velocity

출력:

* `analog_corrected` : `analog_avg`에서 FP 채널 6개에 템플릿 subtract 적용한 값
* `qc` : 검증용 메트릭(아래 3장)

적용 방식(중요 포인트):

* 템플릿은 velocity별로 **unloaded trial에서 추출된 “platform inertial 파형”**이고, 실제 trial의 onset~offset 길이가 조금씩 달라질 수 있어 **길이 워핑(time-warp)**이 필요합니다.
* shared_files도 velocity별 `offset-onset` 길이를 쓰는 구조라(= unload_range_frames 개념) replace_V3D에서도 동일 컨셉으로 가는게 안전합니다. ([GitHub][4])

권장 워핑:

* trial의 `[onset_local, offset_local]` 구간 길이 `L_trial`
* 템플릿 길이 `L_tpl`
* `np.interp`로 템플릿을 `L_trial` 길이로 리샘플링 후 subtract

#### (3) (선택) Stage1 shift 적용기

* `data\corrections.csv`를 (있다면) 로드해서, 해당 subject/velocity/trial에 대해
  `window_start~window_end`에 `Fx += shift_fx_n`, `Fz += shift_n`, `Mx += shift_mx_nm` … 적용. ([GitHub][1])
* 이건 **inertial subtract와 독립**이므로, 파일 없으면 skip하면 됩니다.

---

### 2.2 ankle torque 파이프라인 연결부 수정: `scripts/run_ankle_torque_pipeline.py`

현재 흐름(중요 부분):

* `fp_coll = read_force_platforms(c3d_path)`
* `analog_avg = fp_coll.analog.values`
* `fp = choose_active_force_platform(analog_avg, fp_coll.platforms)` (또는 사용자 지정)
* `F_lab, M_lab = extract_platform_wrenches_lab(analog_avg, fp)` ([GitHub][3])

수정 포인트:

* `analog_avg`를 **corrected로 교체**한 뒤, 그걸로 `extract_platform_wrenches_lab` 수행.

CLI 플래그(권장):

* `--fp_inertial_templates assets/fp_inertial_templates.npz` (기본값, **필수**)
* `--fp_inertial_policy skip|nearest|interpolate`

  * 기본값 `skip`은 “해당 velocity 템플릿이 없으면 적용하지 않음”인데,
    현재는 subtract가 필수이므로 템플릿이 없으면 **에러로 중단**됩니다.
  * 템플릿이 일부 velocity에만 있으면 `nearest` 또는 `interpolate`를 사용합니다.
* QC 관련:

  * `--fp_inertial_qc_fz_threshold`
  * `--fp_inertial_qc_margin_m`
  * `--fp_inertial_qc_strict` (QC 실패 시 중단)

이렇게 하면, forceplate 값 “대체”는 ankle torque 계산에서 완전히 반영됩니다. ([GitHub][5])

---

## 3) “좌표축이 진짜 일치하는지” 검증 프로세스 (필수 요구사항 대응)

사용자 코멘트대로 핵심 리스크는 이겁니다:

* shared_files는 axis_transform을 코드로 수행
* c3d는 이미 변환된 데이터
* **우리가 템플릿/보정값을 가져와서 subtract 할 때**, 축 정의가 다르면 “뺄셈이 채널을 망가뜨림”

따라서 “replace 후” 아래 검증을 자동으로 수행하도록 설계합니다.

---

### 3.1 검증 1: COP가 forceplate polygon 내부에 존재하는지

이건 축이 꼬이면 바로 깨지는 대표 지표입니다.

* replace_V3D는 corners_lab를 읽어서 R_pl2lab을 만들고, COP를 계산합니다. ([GitHub][2])
* corrected 적용 전/후에 대해:

  * `|Fz| > threshold` 프레임만 골라 COP(XY)가 plate corners의 2D polygon 안에 들어가는 비율을 계산
  * **corrected 이후 inside-ratio가 급락**하면 축/부호/모멘트 정의가 깨졌을 가능성이 큼

권장 출력:

* `cop_inside_ratio_raw`
* `cop_inside_ratio_corrected`
* `n_valid_frames`

---

### 3.2 검증 2: “축 스왑/부호 반전” 후보들을 자동 비교해서 best mapping이 identity인지 확인

shared_files config의 axis_transform이 대표적인 축스왑/부호반전 케이스입니다. ([GitHub][4])
따라서 검증은 “정답이 identity여야 한다”(= 이미 c3d가 변환됨)라는 가정에 기반해:

* 후보 매핑 세트:

  1. identity (아무것도 안 함)  ✅ 기대
  2. Fx↔Fy swap
  3. Fz sign flip
  4. (Fx↔Fy + Fz flip)
  5. (Mx↔My swap)
  6. (Mx↔My + Mz flip) … 등 (config에 준하는 케이스)

각 후보에 대해:

* subtract 적용 → COP inside ratio / baseline drift / GRF 수직성 지표 등을 계산
* **best score가 identity가 아니면 “축 정의 불일치” 경고**로 처리

이 로직을 QC 스크립트로 넣으면, 템플릿 생성/적용 단계에서 실수(축 보정 중복 적용 등)를 자동으로 잡을 수 있습니다.

---

### 3.3 검증 3: platform onset 이전 구간에서 “raw == corrected”인지(= subtract window gating 검증)

inertial subtract는 기본적으로 onset~offset 부근에만 적용되어야 정상입니다. ([GitHub][1])

따라서:

* `frame < platform_onset_local - margin` (예: margin=5)
* 이 구간에서 FP 채널 6개에 대해:

  * `max_abs(raw-corrected)`가 매우 작아야 함 (예: < 1e-6 ~ 1e-3, 데이터 스케일에 맞게)
* 여기서 큰 차이가 나면:

  * window 적용 프레임이 잘못 계산되었거나
  * 템플릿이 전체 구간에 깔리거나
  * 축 mapping을 잘못 적용했을 가능성

---

## 4) “보정값 추출” 구현 플랜 (shared_files output 미푸시 이슈 반영)

사용자 말대로 shared_files는 output/parquet이 repo에 없어서, **추출 코드는 “입력 경로를 외부로 받는 형태”**로 만들어야 합니다. ([GitHub][4])

### 4.1 inertial 템플릿 추출 스크립트 (replace_V3D에 두는 것을 권장)

왜 shared_files에 두지 않나?

* shared_files Stage01 파일은 거대하고(polars 기반) 의존이 많습니다. ([GitHub][1])
* 템플릿 생성은 **unload_data_dir + timing_xlsx**만 있으면 되므로, replace_V3D에 “빌더 스크립트”를 넣고 **템플릿 파일만 assets/에 커밋**하는 편이 더 실용적입니다.

#### 템플릿 생성 스크립트: `scripts/torque_build_fp_inertial_templates.py`

입력:

* `--unload_data_dir /path/to/shared_files/Archive/unload_data`
* `--timing_xlsx /path/to/shared_files/Archive/unload_data/FP_platform_on-offset.xlsx`
* `--out_npz assets/fp_inertial_templates.npz`
* `--axis_transform shared_files_default|none` (기본: shared_files Stage01과 동일 매핑)

이 스크립트는 shared_files config의 핵심 규칙을 그대로 따릅니다:

* 파일패턴: `*_perturb_{velocity}_{trial:03d}_forceplate_3.csv`
* required columns: `MocapFrame, MocapTime, DeviceFrame, Fx,Fy,Fz,Mx,My,Mz`
* velocity별 trial mean
* apply_channels: `[Fx,Fy,Fz,Mx,My,Mz]` ([GitHub][4])

---

### 4.2 Stage1 corrections (선택)

Stage1 correction은 shared_files에서 **parquet**이 있어야 생성됩니다(지금 repo에는 없음). ([GitHub][1])

* 현재 `replace_V3D`에서는 Stage1 shift(corrections.csv) 적용을 **구현하지 않았습니다**.
* 필요하면, shared_files에서 생성된 `corrections.csv`를 기반으로 window 구간에 constant shift를 적용하는 로직을 별도 단계로 추가할 수 있습니다.

---

## 5) 실행/운영(AGENTS 준수 관점)

replace_V3D는 “conda env `module`로 실행” 규칙이 문서화되어 있습니다. ([GitHub][5])
따라서 변경 후 검증 루틴까지 포함해 다음처럼 정리하면 됩니다:

### 5.1 템플릿 생성(1회)

```bash
conda run -n module python scripts/torque_build_fp_inertial_templates.py \
  --unload_data_dir /path/to/shared_files/Archive/unload_data \
  --timing_xlsx /path/to/shared_files/Archive/unload_data/FP_platform_on-offset.xlsx \
  --out_npz assets/fp_inertial_templates.npz
```

### 5.2 ankle torque 실행(템플릿 적용 + QC 출력)

```bash
conda run -n module python scripts/run_ankle_torque_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --fp_inertial_templates assets/fp_inertial_templates.npz \
  --out_dir output
```

### 5.3 QC에서 “축 불일치 경고”가 뜨면

* 템플릿 생성 시 axis_transform 적용 여부를 바꿔서 재생성하거나
* 템플릿 적용 시 “template axis remap 옵션”을 켜서 해결(자동 선택)

---

## 6) 결론: 요구사항을 만족하는 “수정 플랜 체크리스트”

* [ ] (1) replace_V3D에서 **inertial subtract 템플릿(npz)** 로딩 가능
* [ ] (2) ankle torque 파이프라인에서 `analog_avg` → `analog_corrected`로 **대체** 후 계산
* [ ] (3) (선택) stage1 data\corrections.csv가 있으면 window shift 적용
* [ ] (4) replace 후 **축 검증 프로세스** 포함:

  * COP inside ratio(전/후 비교)
  * 축/부호 후보 매핑 비교로 identity 여부 확인
  * onset 전 raw==corrected (window gating 검증)
* [ ] (5) AGENTS 규칙에 맞게 conda env `module`로 실행/검증 안내 포함 ([GitHub][6])

---

## 1) 이번 “proceed”에서 실제로 반영한 것

요구사항을 그대로 충족하도록 `replace_V3D` 쪽에 **Stage01(shared_files) 방식의 inertial subtract 템플릿**을 도입했습니다.

### ✅ 핵심 동작(= shared_files Stage01과 동일한 의미)

* `Archive/unload_data`의 `*_forceplate_3.csv`(unloaded trial) + `FP_platform_on-offset.xlsx`에서

  * velocity별로 `unload_range_frames = median(offset-onset)` 산출
  * `DeviceFrame // frame_ratio`로 1000Hz→100Hz downsample
  * onset 기준으로 `relative_frame=0..unload_range_frames`로 정렬 후 trial 평균/보간
  * 각 채널은 `template[0]==0`이 되도록 baseline shift
* 적용 시에는 (Stage01과 동일하게)

  * C3D의 `platform_onset_local`에 맞춰 template을 붙이고
  * **platform_offset_local 이후는 template 마지막 값을 hold(상수 유지)** 하도록 구현했습니다. ([GitHub][1])

### ✅ 축(좌표계) 이슈 처리

Stage01이 unload csv에 적용하는 축 변환(“Lab action → ISB/GRF reaction”)을 **템플릿 생성 시 기본값으로 그대로 적용**하도록 했습니다( Fx←Fy, Fy←Fx, Fz←-Fz, Mx←My, My←Mx, Mz←-Mz ). ([GitHub][1])
(사용자가 말한 대로 C3D는 이미 변환된 축이고, shared_files가 코드로 변환한 정의와 동일하다고 했으므로, “템플릿도 동일 정의로 만들어서” subtract 하도록 한 구조입니다.)

### ✅ “축이 진짜 일치하는지” 검증 프로세스(요구한 QC)

inertial subtract 적용 **전/후**에 대해,

* 활성 구간(onset~offset)에서 `|Fz| >= threshold` 프레임만 골라
* COP를 구하고(plate→lab),
* **COP가 forceplate corners 범위 안에 들어오는 비율**을 계산합니다.
* 적용 후 COP-in-bounds 비율이 크게 떨어지거나(>0.2 악화), 또는 0.5 미만이면 `qc_failed=True`로 표시하고 경고(또는 strict면 에러)로 처리합니다.

이건 “축이 뒤집히거나 스왑된 템플릿을 잘못 subtract 했을 때” 가장 먼저 무너지는 물리적 지표라서, 실제 운영에서 유효합니다.

---

## 2) 변경된 파일 목록

### (A) 신규 모듈

`src/replace_v3d/torque/forceplate_inertial.py`

* `load_forceplate_inertial_templates(npz)`
* `apply_forceplate_inertial_subtract(...)`

  * velocity missing policy: `skip | nearest | interpolate` 지원 (Stage01 설정 의미 그대로) ([GitHub][1])
  * COP-in-bounds QC(전/후), Fz positive fraction 등 메타 포함

### (B) 신규 스크립트(템플릿 생성기)

`scripts/torque_build_fp_inertial_templates.py`

* Stage01의 unloaded template 생성 로직을 “템플릿만 추출/저장” 형태로 포팅
* 기본 axis transform은 shared_files와 동일 mapping 적용 ([GitHub][1])
* 결과: `assets/fp_inertial_templates.npz` (작아서 커밋 가능)

### (C) 기존 파이프라인 수정

`scripts/run_ankle_torque_pipeline.py`

* C3D에서 forceplate 읽은 뒤, wrench/COP 계산 전에 inertial subtract 적용
* 결과 메타 시트에 적용 여부/정책/QC 지표 기록
* strict 모드 옵션 제공

---

## 3) 적용 방법

현재 repo(`master`)에는 이미 반영되어 있습니다.
따라서 실사용 시에는 **템플릿 파일(`assets/fp_inertial_templates.npz`)만 준비**하면 됩니다.

---

## 4) 템플릿(.npz) 생성 방법

로컬에 `Archive/unload_data`와 `FP_platform_on-offset.xlsx`가 있다는 전제입니다(shared_files repo에는 용량 때문에 미푸시라고 했으니, 로컬에 존재한다고 가정). ([GitHub][1])

```bash
conda run -n module python scripts/torque_build_fp_inertial_templates.py \
  --unload_data_dir /ABS/PATH/Archive/unload_data \
  --timing_xlsx /ABS/PATH/Archive/unload_data/FP_platform_on-offset.xlsx \
  --out_npz assets/fp_inertial_templates.npz
```

* 기본 file pattern: `*_perturb_{velocity}_{trial:03d}_forceplate_3.csv`
* 기본 frame_ratio: `10` (1000Hz/100Hz)
* 기본 axis_transform: shared_files_default(=swap_xy + flip_z) ([GitHub][1])

---

## 5) ankle torque 파이프라인 실행 방법(변경 후)

템플릿 파일이 `assets/fp_inertial_templates.npz`에 있으면 **기본으로 적용**됩니다.
(현재는 subtract가 필수이므로 템플릿이 없으면 경고가 아니라 **에러로 중단**됩니다.)

```bash
conda run -n module python scripts/run_ankle_torque_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연"
```

### velocity 템플릿이 없을 때 정책

기본은 `skip`이라서 해당 velocity 템플릿이 없으면 적용 안 합니다.

* nearest 적용:

```bash
--fp_inertial_policy nearest
```

* interpolate 적용(양쪽 velocity 템플릿이 있을 때 선형 보간):

```bash
--fp_inertial_policy interpolate
```

### 축/QC를 “실패하면 바로 중단”시키고 싶으면

```bash
--fp_inertial_qc_strict
```

메타 시트에 아래가 기록됩니다:

* `fp_inertial_qc_cop_in_bounds_frac_before / after`
* `fp_inertial_qc_failed`
* `fp_inertial_template_policy`, `...velocity_int_used` 등

---

## 6) 커밋 메시지(권장, 한국어)

예:

* `forceplate 관성 subtract 템플릿 적용 및 COP QC 추가`
* 또는 더 구체적으로: `Stage01 방식 forceplate inertial subtract 템플릿 로딩/적용`

---

## 7) 내가 이 환경에서 실제로 확인한 것(정직하게)

* `conda run -n module python scripts/verify_fp_inertial_subtract_vs_shared_files.py`를 실행해서,
  `shared_files/output/01_dataset/forceplate_subtract_diagnostics.parquet`과 비교 기준으로
  **units=949, failed=0**(전수 일치) 확인했습니다.
* subtract 적용이 필요한 모든 엔트리포인트에서(ankle torque / batch all_timeseries) 템플릿이 없거나 적용이 안 되면 **즉시 에러로 중단**되도록 되어 있어,
  raw(forceplate 미보정)로 조용히 진행되는 케이스를 차단했습니다.

---

원하면, 다음 단계로는(추가 질문 없이 내가 제안):

* `run_ankle_torque_pipeline.py`가 출력하는 xlsx에 **“QC 플래그가 True인 trial만 따로 모으는 요약 csv”**를 함께 떨구도록 확장(배치 운영 편의)
* 또는 “C3D 자체를 수정해서 저장”까지(=forceplate 채널 overwrite 후 새로운 c3d 쓰기)도 가능하지만, 그건 C3D writer/파라미터 섹션까지 만져야 해서 범위가 훨씬 커집니다.

[1]: https://raw.githubusercontent.com/Rukkha1024/shared_files/main/stages/01_build_dataset.py "raw.githubusercontent.com"
