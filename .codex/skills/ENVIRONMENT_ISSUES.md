# Environment Issues Log

## 2026-02-15

- Context: Replacing Claude visualization commit/branch with Codex output in main worktree.
- Issue: Destructive git operations were blocked by runtime policy.
- Blocked commands:
  - `git reset --hard <commit>`
  - `git branch -d <branch>`
  - `git branch -D <branch>`
- Workaround used:
  - Used non-destructive `git revert --no-edit 5e2b4d2...` to remove Claude commit effects.
  - Kept a single active worktree (`replace_V3D`) and removed auxiliary worktree path to prevent file path confusion.

- Context: Smoke test run via `main.py` with `data/test_data/251112_KUO_perturb_60_001.c3d`.
- Issue: Batch/per-file drivers that rely on `resolve_subject_from_token` failed to resolve token `KUO`
  (missing alias mapping in `data/perturb_inform.xlsm`).
- Workaround used:
  - Run per-file scripts with explicit `--subject`/`--velocity`/`--trial` for quick validation, or
  - Rename/copy the `.c3d` so the parsed token matches `platform.subject` (e.g., `..._김우연_...`).
