지금 네 repo에서 **MOS_AP_dir / MOS_ML_dir 라인이 “거의 90도”로 꺾여 보이는 이유는 로직이 V3D(Visual3D) 튜토리얼 방식이 아니라**, 매 프레임마다 **vCOM 부호로 ‘어느 쪽 bound를 쓸지’ 스위칭**하고 있기 때문이야.

* vCOM이 0 근처에서 조금만 흔들려도(필터링/노이즈/미분 때문에 흔함) **min bound ↔ max bound가 프레임마다 바뀜**
* 그러면 MOS가 “가까운 bound까지의 거리”가 아니라, 갑자기 “반대편 bound까지의 거리”로 튀어서 **계단/수직에 가까운 점프**가 생김
* Visual3D 튜토리얼의 “Original definition”은 이런 스위칭이 아니라 **두 bound(예: minX, maxX)에 대한 거리 중 더 작은 값(Closest_Bound)**을 선택해 **연속적인(=튀지 않는) MoS**를 만들도록 설명돼 있어. ([wiki.has-motion.com][1])

그래서 “V3D 방식대로” 바꾸려면 핵심은 이거야:

## 네 의도(설계 의도) 정리

1. **V3D 튜토리얼(Original definition)처럼**

   * AP, ML 각각에서 **BoS bound 두 개(min/max)**에 대한 거리를 계산하고
   * 매 프레임 **더 작은 값(closest bound)**을 MoS로 사용한다. ([wiki.has-motion.com][1])

2. 네 기존 코드에서 쓰던 `MOS_AP_dir`, `MOS_ML_dir` 이름은 그대로 두되(분석/plot 호환성 때문에),
   **이 값이 이제부터는 V3D closest-bound 값**이 되도록 바꾼다.
   → 너가 지금 쓰는 “MOS_AP_dir plot” 그대로 다시 그려도 정상 곡선이 나와야 함.

3. 기존 “vCOM 부호 스위칭” 방식은 완전히 버리진 않고,
   **디버그용으로 `MOS_AP_velDir`, `MOS_ML_velDir`로 따로 남겨** 비교 가능하게 한다.
   (이 값은 계속 점프가 생길 수 있음)

4. V3D가 “single value MoS”를 말할 때 흔히 쓰는 것처럼,
   `MOS_v3d = min(MOS_AP_v3d, MOS_ML_v3d)`도 같이 제공한다. ([wiki.has-motion.com][1])

---

## 구현 위치(코드 변경)

현재 repo에는 아래 변경이 반영되어 있어(= **V3D Closest_Bound 방식으로 MoS(AP/ML) 계산**).

* 변경 파일:

  * `src/replace_v3d/mos/core.py`  ✅ 핵심 로직 수정
  * `scripts/run_mos_pipeline.py` ✅ 출력 컬럼 + summary에 v3d 지표 추가
  * `scripts/run_batch_mos_timeseries_csv.py` ✅ 배치 CSV 컬럼 추가
  * `scripts/run_batch_all_timeseries_csv.py` ✅ 통합 배치 CSV 컬럼 추가
  * `README.md` ✅ 컬럼 의미 업데이트

---

## 실행 방법 (로컬에서)

```bash
conda run -n module python scripts/run_mos_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --leg_length_cm 86 \
  --out_dir output
```

배치 long-format CSV(모든 trial × frame):

```bash
conda run -n module python scripts/run_batch_mos_timeseries_csv.py \
  --c3d_dir data/all_data \
  --event_xlsm data/perturb_inform.xlsm \
  --out_csv output/all_trials_mos_timeseries.csv \
  --overwrite
```

---

## 적용 후 “뭘 plot 해야 정상인지”

이제부터 추천 plot:

* **정상(=V3D)**

  * `MOS_AP_v3d` (또는 `MOS_AP_dir` — 둘은 동일값)
  * `MOS_ML_v3d` (또는 `MOS_ML_dir`)
  * `MOS_v3d`

* **비교/디버그(=예전 방식, 점프 가능)**

  * `MOS_AP_velDir`
  * `MOS_ML_velDir`

즉, 네가 지금 보던 “MOS_AP_dir 라인이 수직에 가깝다” 문제는
변경 반영 후에는 **MOS_AP_dir 자체가 V3D값(closest-bound)으로 바뀌어서** 바로 해결되어야 해.

---

## 변경이 바꾸는 핵심(한 줄 요약)

* **Before:** `MOS_AP_dir` = (vCOM 부호로 minX/maxX 선택) → 0 근처에서 계속 뒤집혀 점프
* **After:** `MOS_AP_dir` = `min(x-minX, maxX-x)` (Closest_Bound) → 연속적, V3D 튜토리얼과 동일한 선택 방식 ([wiki.has-motion.com][1])
  (예전 방식은 `MOS_AP_velDir`로 분리)

---

원하면, 네가 지금 쓰는 그 엑셀/CSV(패치 적용 전/후)를 기준으로
`MOS_AP_dir` vs `MOS_AP_velDir`를 한 그림에 겹쳐서 “왜 수직처럼 보였는지”를 프레임 레벨로 딱 짚어줄 수도 있어.

[1]: https://wiki.has-motion.com/doku.php?id=visual3d%3Atutorials%3Aknowledge_discovery%3Aassesing_stability_during_gait "Assesing Stability During Gait [HAS-Motion Software Documentation]"
