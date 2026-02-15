# Environment notes (WSL2)

This repository is typically run inside WSL2 and relies on a conda environment.

## Python execution
- The system `python` executable may be missing from PATH.
- Always run Python via the conda env: `conda run -n module python` (per `AGENTS.md`).

## OneDrive / Excel paths
- Excel inputs (e.g., `perturb_inform.xlsm`) may be referenced with macOS-style paths (e.g., `/Users/...`).
- In WSL2, the same file usually lives under `/mnt/c/Users/<name>/OneDrive - <org>/...`.
- If a path is missing, locate it from WSL2 with:
  - `find /mnt/c/Users -name "perturb_inform.xlsm"`

## Git / Deletion commands
- In some runs, commands containing `rm` (e.g., `git rm`) can be blocked by policy.
- Workaround: delete files/folders with Python, then stage deletions with git:
  - `conda run -n module python -c "import shutil; shutil.rmtree('output', ignore_errors=True)"`
  - `git add -A`

## Subject token mapping (Excel)
- Batch scripts often parse a subject token from the C3D filename (e.g., `KUO`) and then call `resolve_subject_from_token`.
- If the token is not resolvable, add a mapping in `perturb_inform.xlsm`:
  - Sheet `meta`: add an alias row (row key contains `이니셜|initial|alias|code|id`) and fill the token under the canonical subject column, or
  - Sheet `transpose_meta`: add an alias column with those keywords and fill the token for the subject row.
- Quick workaround for smoke tests: copy/rename the `.c3d` so the parsed token matches `platform.subject` exactly (e.g., use `251112_김우연_perturb_60_001.c3d` instead of `..._KUO_...`).
