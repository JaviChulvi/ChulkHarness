# Security Policy

ChulkHarness runs model-generated actions through local tools, file access,
shell commands, memory, MCP servers, and traces. Treat this repository as a
security-sensitive developer tool even while it is pre-1.0.

## Supported Versions

ChulkHarness is currently pre-1.0. Security fixes are made on the main branch
until release branches or versioned support windows are introduced.

## Reporting A Vulnerability

Please do not include exploitable details, secrets, tokens, trace artifacts, or
private project data in a public issue.

Preferred reporting path:

1. Use GitHub private vulnerability reporting if it is enabled for the
   repository.
2. If private reporting is unavailable, open a minimal public issue asking for a
   private contact path and omit technical details until a private channel is
   established.

Please include the affected version or commit, a concise impact summary, and
reproduction steps that avoid sharing real credentials or sensitive traces.

## Security Scope

High-priority reports include:

- Tool safety bypasses for shell, file, memory, MCP, or external-service tools.
- Path traversal or project-root boundary bypasses.
- Secret leakage in logs, traces, errors, prompts, or observations.
- Permission-profile failures that allow unintended side effects.
- Prompt-injection paths that can trigger unsafe local actions without the
  expected approval boundary.
