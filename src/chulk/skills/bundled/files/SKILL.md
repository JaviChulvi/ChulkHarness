# Files Skill

Use this skill when the user request requires reading, editing, creating, or organizing files.

Guidelines:

- Inspect existing files before editing.
- Prefer `apply_patch` for edits to existing files; use `write_file` mainly for creating new files.
- Keep changes small and easy to review.
- Restrict file writes to the project directory.
- Preserve unrelated user changes.
- Validate edits with tests or syntax checks when possible.
