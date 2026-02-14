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

