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
