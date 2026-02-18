# Project Requirements Brief

This template must be completed before any ExecPlan is authored. The agent should use this template to guide the requirements discovery session with the user. Fields marked (AGENT) should be filled by the agent through conversation. Fields marked (USER) require direct user input.

---

## Task Classification (AGENT)

<!-- Classify the request type first. The allowed change boundary depends on this classification. -->

- [ ] **Bug Fix** — Replace incorrect behavior with correct behavior. Do not modify surrounding code.
- [ ] **New Feature** — Add functionality that does not currently exist.
- [ ] **Refactor** — Improve code structure without changing behavior.
- [ ] **Config / Data Change** — Modify settings or data without changing logic.

Selected: _______________

---

## One-Line Summary (AGENT)

<!-- Format: "This tool/system [does X] so that [user] can [achieve Y]." -->


## Problem Statement (AGENT)

<!-- Why does this project need to exist? What pain point or gap does it address? Write 2-3 sentences based on user input. -->


## Ambiguity & Assumptions (AGENT — Think Before Coding)

<!-- State all ambiguities and assumptions explicitly before writing a single line of code.
     A wrong assumption means the entire implementation is wrong. Get user confirmation before proceeding. -->

### Interpretations Considered

<!-- If the request has two or more valid interpretations, list all of them and select one with justification. -->

- Interpretation A: ...
- Interpretation B: ...
- **Selected**: ... (Reason: ...)

### Explicit Assumptions

<!-- List every implicit assumption. "It goes without saying" is not allowed here. -->

1.
2.

### Clarifying Questions (USER)

<!-- Include only questions that cannot be resolved by inference. If you can make a reasonable assumption, do so and document it above instead. -->

1.
2.

---

## Success Criteria (USER)

<!-- What does "done" look like? List concrete, observable outcomes. Example: "Running `python main.py --input data.csv` produces a figure saved to `output/fig1.png` showing muscle synergy patterns." -->

1.
2.
3.

## Verifiable Goals (AGENT — Goal-Driven Execution)

<!-- Transform "do X" instructions into "X is verifiably true" statements.
     This table must be fully filled in before ExecPlan authoring begins. -->

| Original Instruction | Transformed Verifiable Goal |
|---------------------|-----------------------------|
| "Add validation"    | "Invalid input raises an error with message Y" |
| "Fix the bug"       | "Input A no longer produces output B; it produces C" |
| (fill in actual)    | (fill in actual) |

### Verification Commands

<!-- Exact commands to run after implementation, with expected output. -->

    $ conda run -n module python scripts/XXX.py
    Expected output: ...

---

## Inputs

### Data Sources (USER)

- Format(s): <!-- e.g., CSV, MAT, C3D, JSON -->
- Location / example path: <!-- e.g., `data/raw/emg_subject01.csv` -->
- Schema or structure: <!-- column names, sampling rate, units, etc. -->
- Volume: <!-- e.g., 12 subjects × 3 conditions × 8 channels -->

### External Dependencies (AGENT)

- Required libraries: <!-- e.g., numpy, scipy, matplotlib -->
- Required services: <!-- e.g., Google Sheets API, database -->
- Required hardware/environment: <!-- e.g., GPU, specific OS -->

## Outputs

### Primary Deliverables (USER)

- Format(s): <!-- e.g., PNG figures, PDF report, new CSV, Python package -->
- Destination: <!-- e.g., `output/` directory, Google Drive, terminal stdout -->
- Naming convention: <!-- e.g., `{subject}_{condition}_synergy.png` -->

### Secondary Artifacts (AGENT)

- Logs: <!-- where and what level -->
- Intermediate files: <!-- cached results, temp files -->
- Documentation: <!-- README, docstrings, usage examples -->

---

## Surgical Change Boundary (AGENT — Surgical Changes)

<!-- Only cut where requested. Do not touch surrounding tissue. -->

### Files / Modules to Touch

<!-- List every file and function/class expected to change before starting. -->

| File | Reason for Change | Scope of Change |
|------|-------------------|-----------------|
|      |                   |                 |

### Files Explicitly NOT to Touch

<!-- Files that must remain untouched even if they appear related. -->

-

### Style Match Rules

- [ ] Follow the existing variable naming conventions in the codebase.
- [ ] Preserve existing indentation and formatting.
- [ ] Only delete code that MY changes orphaned — do not clean up unrelated code.

---

## Constraints

### Technical (AGENT)

- Language: <!-- e.g., Python 3.10+ -->
- Framework: <!-- e.g., none, Django, React -->
- Must use: <!-- libraries or tools that are non-negotiable -->
- Must avoid: <!-- e.g., no MATLAB, no paid APIs -->

### Operational (AGENT)

- Runtime budget: <!-- e.g., must complete in under 5 minutes -->
- Platform: <!-- e.g., Windows 11, Ubuntu 24, macOS -->
- Deployment: <!-- one-off script, CLI tool, installable package, web app -->

## Scope

### In Scope (AGENT)

<!-- Numbered list of what this project WILL do, derived from conversation. -->

1.
2.
3.

### Out of Scope (USER)

<!-- Explicitly state what this project will NOT do to prevent scope creep. -->

1.
2.
3.

## Minimal Implementation Contract (AGENT — Simplicity First)

<!-- This contract is a commitment to achieve the success criteria with the minimum necessary changes.
     Anything not in this contract will not be built. -->

### Will Build

<!-- Only what is directly required to meet the success criteria. -->

1.
2.

### Will NOT Build (by design)

<!-- Tempting additions that are explicitly excluded from this scope, with reasons. -->

- ___: Reason ___
- ___: Reason ___

### No Speculative Code Checklist

- [ ] No new helper/utility classes or functions that are only used once.
- [ ] No code added "because it might be needed later."
- [ ] No validation added for scenarios that cannot happen given internal guarantees.

---

## Prior Art and References (USER)

<!-- Existing code, papers, repos, or examples the agent should study before planning. -->

- Reference implementation: <!-- e.g., `scripts/old_analysis.py` in this repo -->
- Paper / method: <!-- e.g., Chvatal et al. (2011), NMF decomposition -->
- Similar project: <!-- e.g., GitHub link, internal tool -->

## User Expertise Level (AGENT)

<!-- How technical is the user? This determines the level of detail in the ExecPlan. -->

- Domain expertise: <!-- e.g., expert in biomechanics, novice in web dev -->
- Coding proficiency: <!-- e.g., intermediate Python, no JS experience -->
- Will the user modify the code after delivery? <!-- yes / no -->

## Open Questions (AGENT)

<!-- Unresolved items that surfaced during discovery. Must be answered before ExecPlan authoring begins. -->

1.
2.

---

## Requirements Confirmed

- [ ] User has reviewed and approved all sections above.
- [ ] All Open Questions have been resolved.
- [ ] All Clarifying Questions (Ambiguity & Assumptions) have been answered by the user.
- [ ] Verifiable Goals table is fully filled in.
- [ ] Surgical Change Boundary has been reviewed and agreed upon.
- [ ] Minimal Implementation Contract has been agreed upon.
- [ ] Agent may now proceed to author the ExecPlan per `.claude/PLANS.md`.
