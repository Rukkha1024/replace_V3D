# Issues (replace_V3D)

Policy:
- This file records **문제 자체** only (symptom / impact / repro file or log).
- Record **해결방법/워크어라운드** in the global skill: `$replace-v3d-troubleshooting`.

---
## 2026-02-16

- [DATA] `data/all_data/251128_방주원_perturb_200_005.c3d`: marker `T10` missing → joint-angle computations may fail in batch pipelines that expect the full marker set.
- [QC] 일부 trimmed C3D가 기대 구간 `[platform_onset-100, platform_offset+100]` 대비 정확히 1 frame 짧음(`delta_frames=-1`). (±1 frame 외의 위반은 관측되지 않음)

## 2026-02-15

- [ENV] 일부 실행 환경에서 destructive git 명령이 runtime policy로 차단될 수 있음(예: `git reset --hard`, `git branch -D`).
- [DATA] C3D filename token이 `resolve_subject_from_token()`에서 매핑되지 않으면 배치/드라이버가 실패할 수 있음(예: token `KUO` → `data/perturb_inform.xlsm` alias 미등록).

