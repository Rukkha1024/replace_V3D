요구사항대로 **MoS는 step onset 기준으로 구간을 나누지 않았고**, 분석 구간도 **step onset 직전(frame = step_onset-1)까지만** 계산되도록 정리해서 **repo에 바로 넣어 쓸 수 있는 형태**로 묶었습니다.
(BoS는 **foot landmark marker 기반 convex hull** 그대로, forceplate 미사용, toe-off 검출 없음)

## 1) 결과 파일 (pre-step만)

* **분석 범위(로컬 프레임)**

  * platform_onset(original=1573) → **local=101**
  * step_onset(original=1628) → **local=156**
  * 따라서 MoS/BoS/xCOM 등 결과는 **local 1~155**까지만 있음(**156 포함 안 함**)

* **Excel 출력**:
  [Download](sandbox:/mnt/data/251112_KUO_perturb_60_001_MOS_preStep.xlsx)

포함 시트:

* `timeseries_preStep`: Frame(1~155), COM/vCOM/xCOM, BoS area & bounds, **MOS_minDist_signed**, (옵션) directional MOS(AP/ML), 무릎/발목 각도
* `summary`: baseline(1~100) vs onset~pre-step(101~155) 요약
* `events`: 원본/로컬 프레임 매핑, 분석 end frame
* `validation_COM`: V3D COM과 상관

### COM 검증(요약)

V3D COM(xlsx)와의 상관(이 파일 기준):

* 전체 프레임(1~222): **X=0.99944, Y=0.99907, Z=0.99491**
* pre-step(1~155): **X=0.99388, Y=0.99968, Z=0.99632**

(요구한 “1에 가까워야” 조건은 충족하는 수준으로 맞춰짐. 특히 pre-step 구간에서 충분히 높게 나옴)

---

## 2) Git repo에 넣을 코드 패치(zip)

GitHub에 제가 직접 push는 이 환경 제약상 불가해서, **repo 루트에 그대로 merge/copy**할 수 있게 **추가 파일만** 묶어 드립니다.

[Download](sandbox:/mnt/data/replace_V3D_work_patch.zip)

zip 내용:

* `src/replace_v3d/`

  * `c3d_reader.py` : C3D marker 파싱(POINT:LABELS, points float32)
  * `events.py` : `perturb_inform.xlsm`(platform sheet)에서 onset/offset/step_onset 로딩 + local frame 변환
  * `com.py` : medial marker 포함 joint center 기반 COM + vCOM + xCOM
  * `geometry2d.py` : **외부 의존성 없이** convex hull / polygon distance 구현
  * `mos.py` : foot landmark 기반 BoS hull + **signed min distance MoS**
  * `angles.py` : sagittal knee flex / ankle dorsi(간단 버전)
* `scripts/run_mos_pipeline.py` : 단일 trial 실행 스크립트(설치 없이 실행되도록 `sys.path` 처리 포함)
* `README.md` : 사용법 요약

---

## 3) 실행 커맨드(네 repo에서 그대로)

repo 루트에서:

```bash
python scripts/run_mos_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --leg_length_cm 86 \
  --v3d_com_xlsx /path/to/251112_KUO_perturb_60_001_COM.xlsx \
  --out_dir /path/to/output
```

출력:

* `<stem>_MOS_preStep.xlsx`
  (step trial이면 자동으로 **step_onset_local - 1**까지만 산출)

---

## 4) “MoS 기준 변화시키지 말라” 반영 사항 정리

* ✅ **BoS**: 발 landmark marker들(`LHEE/LTOE/LANK/LFoot_3`, `RHEE/RTOE/RANK/RFoot_3`)의 **convex hull**
  → **foot length/width로 BoS 확장 같은 건 안 함**
* ✅ **MoS**: `MOS_minDist_signed = signed(min distance(xCOM_xy, BoS boundary))`
* ✅ **step onset 기준으로 MoS 구간 분할/지원발 전환 없음**
  → step trial도 결과는 그냥 **pre-step까지만** 내고 끝
* ✅ forceplate 미사용, toe-off 검출 불필요

원하면 다음 단계로는:

* (a) MoS를 “최소거리” 외에 V3D에서 쓰는 특정 방향(예: AP/ML, 혹은 진행방향) 정의가 더 있으면 그 정의를 그대로 맞춰서 컬럼을 정리하거나
* (b) 네 논문 초고 변수(performance 변수들: min, time-to-min, AUC 등)를 `summary`에 자동 계산
  까지 바로 repo에 이어서 붙일게.
