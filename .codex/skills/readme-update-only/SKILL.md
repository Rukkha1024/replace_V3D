---
name: readme-update-only
description: Update only the current repository README.md. Use when the user asks to revise, refresh, reorganize, or maintain README content, or when the user explicitly types `$readme-update-only` (or `${skill_name}` for this skill). 
---

# README-Only Update

## Rules

- Edit only `README.md`.
- Treat the current `README.md` as the source of truth unless the user provides explicit new content.
- Do not fetch external references or infer new scope.
- Do not modify or create any other files.
- Keep existing structure, terminology, and tone unless the user asks to change them.

## Trigger Behavior

1. If the prompt includes `$readme-update-only`, activate this workflow immediately.
2. If the prompt is only `$readme-update-only`, perform a direct README maintenance update with minimal safe edits (clarity, consistency, and formatting) based only on the current `README.md`.
3. If requested changes are ambiguous, ask focused clarification questions before editing.

## Editing Procedure

1. Read `README.md` and map heading structure, terms, and formatting patterns.
2. Apply only request-relevant edits in the smallest possible scope.
3. Preserve style and section order unless the user requests reorganization.
4. Report only key changes after editing.
