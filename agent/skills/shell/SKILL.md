# Shell Skill

Use this skill when the user request requires terminal inspection or command execution.

Guidelines:

- Prefer read-only commands first.
- Keep commands small and explainable.
- Treat model-generated commands as untrusted input.
- Do not run destructive commands.
- Capture stdout, stderr, exit code, and timeout information.
- Log every command.
