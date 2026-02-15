---
name: md-style-preserving-edit
description: Preserve existing Markdown document writing style while editing. Use when modifying existing .md files to add new analysis results, sections, or notes without rewriting prior content. Prioritize append/insert over replace unless the user explicitly requests a rewrite; keep headings, terminology, table formats, and link style consistent with the existing document.
---

# Md Style Preserving Edit

## Overview

Edit existing Markdown documents by adding content in a way that matches the document's current writing style and structure.
Default behavior is non-destructive: keep existing text intact and only insert new sections where appropriate.

## Workflow

### 1) Scan Style And Structure (Read-Only)

Before editing, scan the target `.md` file for:
- Heading hierarchy: `#`, `##`, `###` usage and numbering conventions.
- Terminology and labels: window names, variable names, abbreviations, domain terms.
- Table style: column order, alignment, numeric formatting (`M±SD`, decimals), separators.
- Citation/link style: wiki links `[[...]]`, `[@paper]`, plain text references.
- Tone: bullet-heavy vs narrative, Korean vs English mix, sentence endings.

Produce a short "style contract" (internal checklist) and follow it for all inserted text.

### 2) Decide Edit Mode (Default: Insert)

Pick one of the following, in this priority order:
- Insert (default): add a new section into an existing location (recommended for analysis updates).
- Append: add a new section at the end (recommended when the document has stable ordering).
- Replace/Rewrite: only if the user explicitly says to rewrite, reorganize, or replace existing text.

If the user did not explicitly request rewrite, do not overwrite the whole file even if automation would be easier.

### 3) Choose Insertion Location (Style-Compatible)

Preferred insertion rules:
- Insert near the most relevant existing section (e.g., under `# 3. results` if adding new results).
- Do not renumber existing headings unless requested.
- Avoid changing the document's outline unless necessary.

If a safe insertion point cannot be determined, ask the user where to add the new content.

### 4) Write New Content To Match Existing Style

Rules:
- Use the same language mix (Korean/English), punctuation, and bullet style found nearby.
- Match numeric formatting already used in the file (decimals, units, `M±SD`, p-value format).
- If adding tables, keep column order and headers consistent with neighboring tables.
- Reuse existing labels (e.g., `분석구간`, `요약`, `해석상 주의사항`) instead of inventing new ones.

### 5) Prevent Accidental Duplication (When Automating Inserts)

When inserts are produced by scripts, add a small marker line to the inserted block, e.g.:
- `<!-- AUTO_APPEND: <id> -->`

Then:
- If the marker already exists, skip inserting again.
- Do not delete or replace existing content unless explicitly requested.
