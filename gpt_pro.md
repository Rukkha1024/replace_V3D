[패치 ZIP 다운로드](sandbox:/mnt/data/replace_V3D_joint_angles_patch.zip)

## ZIP 구성물

* `PROMPT_FOR_AI.md`
  → **repo에서 실제로 수정 작업할 AI에게 그대로 붙여넣을 “지시 프롬프트”**
* `git_apply.patch`
  → repo root에서 `git apply`로 적용 가능한 패치
* `patch_files/`
  → 패치가 실패해도 **파일 복사로 적용**할 수 있게 동일 파일 트리 제공
* `reference_outputs/`

  * `251112_KUO_perturb_60_001_JOINT_ANGLES_preStep.csv`
  * `md5.txt`

    * MD5: `6896e319c723bbc347eb413ba9a6d551` (reference CSV)

## 패치로 추가/수정되는 것 (요약)

* 신규: `scripts/joint_angles/` (Joint angle 변수 카테고리)

  * `scripts/joint_angles/v3d_joint_angles.py`

    * **Visual3D-like 3D joint angles** (intrinsic XYZ: reference X → floating Y → non-ref Z)
    * Segment axes: **X=+Right, Y=+Anterior, Z=+Up/Proximal**
    * **Medial markers 사용**: `LShin_3/RShin_3`, `LFoot_3/RFoot_3`
    * Hip center는 기존 `com.compute_joint_centers`(Harrington) 재사용
* 신규 CLI: `scripts/run_joint_angles_pipeline.py`

  * 출력: `<trial>_JOINT_ANGLES_preStep.xlsx` + `<trial>_JOINT_ANGLES_preStep.csv`
  * step trial이면 `analysis_end = step_onset_local - 1`
* 수정: `README.md`에 Joint angles quick start 섹션 추가

원하시는 흐름대로 하시면 됩니다:

1. ZIP을 AI에게 전달
2. **`PROMPT_FOR_AI.md` 내용을 그대로** “repo 작업 AI”에게 붙여넣기
3. AI가 `git_apply.patch` 적용 후 실행/MD5 검증까지 수행하도록 구성되어 있습니다.
