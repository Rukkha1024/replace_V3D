당신은 사용자의 Git repo `replace_V3D`(WSL2/conda env: `module`)에서 작업하는 에이전트입니다.  
**반드시 `AGENTS.md`의 Work Procedure(Plan→사용자 확인→수정→Korean commit→Run/MD5→Finalize)를 준수**하세요.

---

# 목표
OptiTrack Conventional(Plug-in-Gait + medial markers) C3D에서 **Visual3D 스타일 3D joint angles**를 계산하는 파이프라인을 repo에 추가합니다.

- Joint: **ankle, knee, hip, trunk, neck**
- Side: **L/R 모두(ankle/knee/hip), trunk/neck은 global(무측)**
- 이벤트/구간: `perturb_inform.xlsm`의 `platform` sheet 기준  
  `analysis_end = step_onset_local - 1` (step trial), step onset이 없으면 전체

---

# Plan (사용자 확인 포함)
아래 변경 사항을 추가하겠습니다. **이 프롬프트 자체가 사용자 확인으로 간주됩니다. 바로 진행하세요.**

1) **Joint angle 변수 카테고리**로 코드 분리  
   - `scripts/joint_angles/` 패키지 추가  
   - Visual3D-like 3D joint angle 계산 모듈 추가

2) 신규 CLI 파이프라인 추가  
   - `scripts/run_joint_angles_pipeline.py`  
   - 입력: `--c3d`, `--event_xlsm`, `--subject` (+ filename에서 velocity/trial 파싱)  
   - 출력: `<trial>_JOINT_ANGLES_preStep.xlsx` + `<trial>_JOINT_ANGLES_preStep.csv`

3) README에 Joint angles quick start 섹션 추가

4) 실행 및 검증  
   - 제공된 reference CSV와 **MD5 비교**로 최소 검증

---

# 적용 방법
이 zip 안의 `git_apply.patch` 또는 `patch_files/`를 사용하세요.

## Option A) git apply (권장)
repo root에서:

```bash
git apply path/to/git_apply.patch
```

## Option B) 파일 복사
`patch_files/` 안의 구조를 repo root에 그대로 덮어쓰기/추가합니다.

---

# 구현/계산 규칙 (코드 내부 요약)
- Segment axes: **X=+Right, Y=+Anterior, Z=+Up/Proximal**
- Joint angles: **intrinsic Cardan XYZ**  
  (Visual3D: reference X → floating Y → non-reference Z)
- Medial markers 사용(필수):
  - Knee medial: `LShin_3`, `RShin_3`
  - Ankle medial: `LFoot_3`, `RFoot_3`
- Hip center: 기존 repo의 `com.compute_joint_centers`(Harrington) 재사용

---

# 실행 커맨드 (예시)
```bash
conda run -n module python scripts/run_joint_angles_pipeline.py \
  --c3d /path/to/251112_KUO_perturb_60_001.c3d \
  --event_xlsm /path/to/perturb_inform.xlsm \
  --subject "김우연" \
  --out_dir output
```

---

# MD5 검증
이 zip의 `reference_outputs/251112_KUO_perturb_60_001_JOINT_ANGLES_preStep.csv`를 기준으로 MD5를 비교합니다.

예:
```bash
md5sum output/251112_KUO_perturb_60_001_JOINT_ANGLES_preStep.csv
md5sum reference_outputs/251112_KUO_perturb_60_001_JOINT_ANGLES_preStep.csv
```

---

# Git commit
- Korean commit message 예시:
  - `관절각 파이프라인 추가: ankle/knee/hip/trunk/neck 3D (V3D 방식)`

---

# Finalize (AGENTS.md 준수)
- 실행 결과/MD5 결과를 보고합니다.
- 만약 conda/env/패키지 이슈가 있으면 `.codex/skills` 또는 `AGENTS.md`에 기록합니다.
