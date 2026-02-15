---
name: excel-xlwings-workflow
description: >-
  Automate and validate Excel workbook tasks in this repository using
  xlwings. Use when creating/updating `.xlsx` or `.xlsm` files, writing
  formulas, filling ranges, or generating Excel reports. Enforce Windows
  terminal usage (PowerShell/cmd only) and always reopen the output workbook
  to verify formula/function errors and unexpected blank cells. Minimize sheet
  count and document every Excel Table with a `table_guide` sheet. In
  `analysis/` workflows tied to statistical or aggregation exploration, prefer
  `.py` for Excel generation while allowing `.ipynb` for exploration.
---

# Excel Xlwings Workflow

## Non-Negotiable Rules

- Use `xlwings` for every Excel read/write/formula operation.
- Do not use `openpyxl`, `pandas.ExcelWriter`, or shell-only alternatives as the primary Excel engine unless the user explicitly overrides.
- Default output workbook format to `.xlsx` (switch to `.xlsm` only when the user explicitly needs macros).
- Unless there is a specific reason not to, store written data as **Text** in Excel.
  - Applies when the task is “plain text entry” (labels, IDs, codes, keys, categorical fields) rather than numeric computation or Excel formulas.
  - Rationale: prevents Excel auto-coercion (dropping leading zeros, scientific notation, date auto-parsing).
  - Implementation (xlwings): set the target range/column `number_format` to `"@"` **before** writing values, and write values as strings (`str(...)`) when appropriate.
  - Exception: do not force **Text** format for columns that must remain numeric for calculation/charting/statistics; keep those as numeric formats.
- For `analysis/` tasks in statistical-analysis or aggregation contexts:
  - Prefer `.ipynb` for exploratory analysis/aggregation attempts.
  - Prefer `.py` scripts for final Excel workbook generation.
  - Apply this as a context-specific preference, not as a global hard rule for every workflow.
- Keep sheet count minimal. Default to 2 sheets unless the user requests otherwise:
  - one data sheet (for example `tables`)
  - one metadata sheet `table_guide`
- Write user-facing outputs as Excel Tables (`ListObject`) rather than loose ranges whenever feasible.
- `table_guide` is mandatory when tables are generated. Each table must have a human-readable description.
- Run commands in Windows `PowerShell` or `cmd` syntax only.
- Run Python and pip with:
  - `conda run -n module python ...`
  - `conda run -n module pip ...`
- Verify interpreter path before Python execution:
  - `conda run -n module python -c "import sys; print(sys.executable)"`
- When writing plain labels via `xlwings`, do not start cell text with `=` (for example avoid `=== ... ===`).
  - Reason: Excel interprets leading `=` as a formula and may throw COM error `0x800A03EC`/`-2146827284`.
  - Use non-`=` prefixes for section titles (for example `[section] ...`).

## Standard Workflow

1. Confirm workbook path, required outputs, and minimal sheet design.
2. In `analysis/` statistical or aggregation workflows, use `.ipynb` for exploration when needed.
3. Define table inventory first (table name, target sheet, start cell, and description).
4. Implement final Excel-generation logic in a `.py` script with `xlwings`.
5. Create Excel Tables for outputs, and preserve formulas/formatting where needed.
6. Populate `table_guide` with per-table descriptions.
7. Save workbook and close handles (`wb.close()`, app context exit).
8. Reopen the saved workbook and run validation checks.
9. Open the saved workbook in Excel UI once for visual QA.
10. If any issue is found, fix and rerun steps 4 to 9.

## Sheet and Table Rules

- Prefer one consolidated data sheet unless data volume or user requirements justify more.
- Use stable table names, for example `tbl_<domain>_<scope>`.
- Avoid duplicate/empty sheets and scattered unstructured ranges.
- Required `table_guide` columns:
  - `table_name`
  - `sheet_name`
  - `table_range`
  - `description`
  - `key_columns`
  - `notes` (optional)
- `description` should explain what the table represents and how to interpret it in one short sentence.

## Required Post-Save Validation

Run every check after reopening the output workbook:

- Formula/function error check:
  - Fail if a cell value is one of `#DIV/0!`, `#N/A`, `#NAME?`, `#NULL!`, `#NUM!`, `#REF!`, `#VALUE!`.
- Broken formula check:
  - Fail if required formula cells are missing formulas.
  - Fail if formula text contains broken references (for example, `#REF!`).
- Blank-cell check:
  - Define business-critical required ranges first.
  - Fail if required cells are empty (`None` or empty string after trim).
- Table integrity check:
  - Fail if expected tables are missing.
  - Fail if `table_guide` is missing.
  - Fail if any table in workbook lacks a matching `table_guide` row.
  - Fail if `description` is blank in `table_guide`.
- Record a validation summary:
  - workbook path, sheet/range scanned, issue counts, pass/fail.

- Label safety check (xlwings write path):
  - Fail if any intended plain-label string starts with `=`.
  - Fix by rewriting label text to avoid a leading `=` before writing to cells.

## Validation Snippet (xlwings)

```python
import xlwings as xw

ERROR_TOKENS = {"#DIV/0!", "#N/A", "#NAME?", "#NULL!", "#NUM!", "#REF!", "#VALUE!"}

def is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == "")

def validate_book(path, required_ranges):
    issues = {"errors": [], "blanks": [], "tables": []}
    with xw.App(visible=False, add_book=False) as app:
        wb = app.books.open(path)
        try:
            for sheet_name, addr in required_ranges:
                sheet = wb.sheets[sheet_name]
                rng = sheet.range(addr)
                for cell in rng:
                    value = cell.value
                    formula = cell.formula
                    if isinstance(value, str) and value in ERROR_TOKENS:
                        issues["errors"].append((sheet_name, cell.address, value))
                    if isinstance(formula, str) and "#REF!" in formula:
                        issues["errors"].append((sheet_name, cell.address, "Broken reference in formula"))
                    if is_blank(value):
                        issues["blanks"].append((sheet_name, cell.address))

            guide_exists = any(s.name == "table_guide" for s in wb.sheets)
            if not guide_exists:
                issues["tables"].append(("table_guide", "missing"))
        finally:
            wb.close()
    return issues
```

## Formatting Snippet: Force Text Format for Plain-Text Columns (xlwings)

Use this when you are writing IDs/codes/labels (not formulas and not numeric analytics columns).

```python
import xlwings as xw

def write_text_column(sheet, header_cell, values):
    col_rng = sheet.range(header_cell).expand("down")
    col_rng.number_format = "@"
    col_rng.value = [[str(v) if v is not None else ""] for v in values]
```

## Command Templates (PowerShell/cmd)

NOTE: `scripts/excel_job.py` is a placeholder example path (this repo may not include that file). Replace it with your actual Excel-generation script under `scripts/`.

PowerShell:
- `conda run -n module python -c "import sys; print(sys.executable)"`
- `conda run -n module python .\scripts\excel_job.py --input ".\input.xlsx" --output ".\output.xlsx"`
- `Start-Process ".\output.xlsx"`

cmd:
- `conda run -n module python -c "import sys; print(sys.executable)"`
- `conda run -n module python scripts\excel_job.py --input ".\input.xlsx" --output ".\output.xlsx"`
- `start "" ".\output.xlsx"`

## Incident Playbook: xlwings Label Treated as Formula

- Symptom: `pywintypes.com_error` when writing a string label to a cell, often with code `-2146827284` (Excel `0x800A03EC`).
- Typical trigger: label text starts with `=` such as `=== section ===`.
- Root cause: Excel formula parser is invoked for the label string.
- Fix:
  1. Change label text so the first character is not `=`.
  2. Re-run workbook generation and reopen workbook to verify the target sheet is populated.
  3. Keep this as a pre-write checklist item for all xlwings label writes.

## Completion Report Checklist

- Include workbook output path and touched sheets/ranges.
- Include sheet minimization result (final sheet count and rationale if >2).
- Include created table names and whether each has `table_guide` description.
- Include formula/function error check result.
- Include blank-cell check result for required ranges.
- Confirm the saved file was reopened and visually checked in Excel.
