# Issues (replace_V3D)

Policy:
- This file records **문제 자체** only (symptom / impact / repro file or log).
- Record **해결방법/워크어라운드** in the global skill: `$replace-v3d-troubleshooting`.

---
## 2026-02-17

- [ENV] 비대화형 WSL2 shell에서 `conda`가 PATH에 없어 `conda run -n module ...` 실행이 실패할 수 있음(증상: `conda: command not found`).

## 2026-02-16

- [DATA] `data/all_data/251128_방주원_perturb_200_005.c3d`: marker `T10` missing → joint-angle computations may fail in batch pipelines that expect the full marker set.
- [QC] 일부 trimmed C3D가 기대 구간 `[platform_onset-100, platform_offset+100]` 대비 정확히 1 frame 짧음(`delta_frames=-1`). (±1 frame 외의 위반은 관측되지 않음)
- [VIZ] `scripts/plot_grid_timeseries.py`: `--sample` vs `all` 렌더에서 동일한 overlay plot의 `linewidth/alpha`가 달라 `all` 결과가 과도하게 옅게 보이며(가독성 저하), 라인 표현 category가 사실상 색/스타일 외 요소로도 분기되는 문제가 있었음. (재현 예: `output/figures/grid_timeseries/by_subject/강비은/y0/grf_cop__subject-강비은__velocity-30__all.png`)

## 2026-02-15

- [ENV] 일부 실행 환경에서 destructive git 명령이 runtime policy로 차단될 수 있음(예: `git reset --hard`, `git branch -D`).
- [DATA] C3D filename token이 `resolve_subject_from_token()`에서 매핑되지 않으면 배치/드라이버가 실패할 수 있음(예: token `KUO` → `data/perturb_inform.xlsm` alias 미등록).
