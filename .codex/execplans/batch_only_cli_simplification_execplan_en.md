# replace_V3D Batch-Only CLI Simplification

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated continuously during implementation.

Repository policy reference: follow `.codex/PLANS.md`.

## Purpose / Big Picture

After this change, users only need one operational command path: batch execution through `main.py`. We remove legacy single-trial flags and auxiliary entrypoint scripts, then prove behavior preservation with before/after MD5 checks for batch outputs.

## Progress

- [x] (2026-02-22 10:20Z) Captured baseline artifacts (`batch_only_before.csv`, MD5, help output).
- [x] (2026-02-22 10:23Z) Removed single-trial CLI flags and execution branch from `main.py`.
- [x] (2026-02-22 10:24Z) Deleted four single/auxiliary scripts.
- [x] (2026-02-22 10:27Z) Removed single-mode references from `README/.codex/Archive`.
- [x] (2026-02-22 10:33Z) Completed post-change batch validation and before/after MD5 comparison (`before=after=3bd4cfc31c17b5759899e0d75837c864`).
- [x] (2026-02-22 10:38Z) Recorded issue-only notes in `.codex/issue.md` and solution notes in `$replace-v3d-troubleshooting`.
- [ ] Create Korean 3+ line commit and clean unnecessary artifacts.

## Surprises & Discoveries

- Observation: an internal legacy MOS helper script kept triggering policy consistency scans because it still referenced removed single-trial patterns.
  Evidence: consistency scan output repeatedly matched one internal skill script.

## Decision Log

- Decision: Remove the two single-trial flags without compatibility wrappers; keep default argparse error behavior.
  Rationale: user-locked policy and explicit avoidance of hidden legacy paths.
  Date/Author: 2026-02-22 / Codex

- Decision: Remove the MOS-only auxiliary batch entrypoint.
  Rationale: requirement to keep one official batch path (`main.py -> run_batch_all_timeseries_csv.py`).
  Date/Author: 2026-02-22 / Codex

- Decision: Apply full cleanup to internal docs based on currently executable commands.
  Rationale: user explicitly requested full cleanup including `.codex`.
  Date/Author: 2026-02-22 / Codex

## Outcomes & Retrospective

The repository is now structurally batch-only at CLI/runtime/documentation levels, and validation confirmed no change in batch output hash for the sampled run. We also discovered and fixed argparse prefix-abbreviation behavior so removed flags fail exactly as intended. Remaining work is final commit and change handoff.

## Context and Orientation

The main entrypoint is `main.py`. Previously, it had both a default batch path and a legacy single-trial path, which called multiple single-trial helper scripts. This work removes that branch and keeps only the batch path (`scripts/run_batch_all_timeseries_csv.py` + `scripts/apply_post_filter_from_meta.py`).

Documentation is also updated so that user-facing and internal docs no longer contain dead commands or removed script references.

## Plan of Work

Simplify `main.py` parser and control flow to batch-only. Delete four legacy entrypoint scripts. Update `README` and internal docs to replace removed command paths with current batch-valid instructions. Validate behavior via before/after runs and MD5 comparison. Finish with issue/solution logging and commit.

## Concrete Steps

Working directory: repository root

1. Capture pre-change baseline.

    conda run -n module python main.py --overwrite --skip_unmatched --max_files 5 --out_csv output/qc/batch_only_before.csv
    md5sum output/qc/batch_only_before.csv > output/qc/batch_only_before.md5
    conda run -n module python main.py --help > output/qc/main_help_before.txt

2. Convert `main.py` to batch-only and delete legacy scripts.

3. Run documentation consistency scan.

    rg -n -S -- "<legacy-entrypoint-or-flag-patterns>" README.md .codex Archive

4. Run post-change validation.

    conda run -n module python main.py --help > output/qc/main_help_after.txt
    conda run -n module python main.py <removed-single-trial-flag> data/all_data/does_not_matter.c3d
    conda run -n module python main.py --overwrite --skip_unmatched --max_files 5 --out_csv output/qc/batch_only_after.csv
    md5sum output/qc/batch_only_after.csv > output/qc/batch_only_after.md5
    diff -u output/qc/batch_only_before.md5 output/qc/batch_only_after.md5

5. Update issue/solution records and commit.

## Validation and Acceptance

1. `main.py --help` no longer exposes removed single-trial flags.
2. Running `main.py` with a removed single-trial flag fails with argparse `unrecognized arguments`.
3. MD5 for `batch_only_before.csv` and `batch_only_after.csv` is identical.
4. Consistency scan on `README/.codex/Archive` returns no matches.
5. `.codex/issue.md` and `$replace-v3d-troubleshooting` are updated with issue/solution records.

## Idempotence and Recovery

All verification commands are rerunnable. Artifacts under `output/qc` can be overwritten safely. If MD5 diverges, first verify execution path and input ordering before suspecting algorithmic changes.

## Artifacts and Notes

- `output/qc/batch_only_before.csv`
- `output/qc/batch_only_before.md5`
- `output/qc/main_help_before.txt`
- `output/qc/batch_only_after.csv`
- `output/qc/batch_only_after.md5`
- `output/qc/main_help_after.txt`

## Interfaces and Dependencies

- CLI changed in `main.py`:
  - Removed flags: two single-trial-only flags
  - Preserved flags: all batch-related options
- Batch engines preserved:
  - `scripts/run_batch_all_timeseries_csv.py`
  - `scripts/apply_post_filter_from_meta.py`
- Deleted entrypoints:
  - legacy joint-angle single-run script
  - legacy ankle-torque single-run script
  - legacy MOS single-run script
  - legacy MOS auxiliary batch script

---
Change Note (2026-02-22): Added new synced KO/EN ExecPlan for the batch-only policy migration.
