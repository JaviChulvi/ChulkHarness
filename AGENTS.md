# Agent Guide

This file is the canonical guide for coding agents working in this repository. `CLAUDE.md` is a symlink to this file.

## Core Guidelines

These are the primary decision rules for every change:

- **Keep the solution small and explicit.** Implement the simplest complete change that preserves the harness's inspectability. Prefer plain dataclasses, registries, and functions over hidden control flow or speculative abstraction.
- **Solve at the owner.** Put behavior in the code path that owns or observes it. Fix the cause instead of adding staleness checks, skip-first-call branches, broad exception handling, or parallel paths around broken behavior.
- **Search and reuse first.** Search the repository before creating a helper, registry, service, workflow, or file. Reuse or extend the established owner and consolidate in-scope duplication there.
- **Simplify before adding.** Delete obsolete paths and replace incorrect behavior before adding new layers. Add a new abstraction only when it creates a clearer ownership boundary or removes demonstrated complexity.
- **Keep scope controlled.** Avoid unrelated cleanup, speculative compatibility shims, impossible-state scaffolding, and bridges into later roadmap work unless the user asks for them.
- **Ship coherent feature PRs.** Prefer each pull request to deliver one complete, independently testable feature or behavior slice, including its implementation, tests, documentation, and migrations when applicable. Do not fragment a feature into tiny PRs that are difficult to evaluate in isolation, and do not bundle unrelated features merely to make a PR larger.
- **Protect public contracts.** Treat SDK imports, events, results, errors, persisted schemas, tool behavior, and configuration semantics as contracts. Preserve compatibility unless the requested change intentionally updates the contract.
- **Add regression coverage.** Every behavior change needs focused tests. Prefer testing through the public boundary and use fake or injected model clients rather than live services.
- **Finish cleanly.** Remove dead imports, code, comments, and duplicate paths introduced or exposed by the change. Validate the changed owner and inspect the final scoped diff.

## Source Of Truth

Do not use this guide as a snapshot of every model, provider, environment variable, command, default, or roadmap status. Verify current behavior at its owner:

- Product direction, delivery order, dependencies, and implementation status: `TODO.md`.
- Supported public behavior and stability policy: `src/chulk/__init__.py`, `src/chulk/api.py`, `src/chulk/testing.py`, and `docs/release-policy.md`.
- Runtime construction shared by CLI and SDK: `src/chulk/runtime.py`.
- SDK facade, configuration, results, events, and error mapping: `src/chulk/_sdk/`, `src/chulk/results.py`, `src/chulk/events.py`, and `src/chulk/errors.py`.
- Agent orchestration and state transitions: `src/chulk/core/`.
- Provider registration, capabilities, lifecycle, and transports: `src/chulk/llm/factory.py`, `src/chulk/llm/capabilities.py`, and `src/chulk/llm/providers/`.
- Runtime configuration and defaults: `src/chulk/config.py`, `src/chulk/_sdk/config.py`, `.env.example`, and `docs/configuration.md`.
- Shared SQLite policy and forward-only schema migrations: `src/chulk/storage/sqlite.py` and `src/chulk/storage/migrations.py`.
- Subsystem persistence behavior: the relevant store under `src/chulk/memory/`, `src/chulk/sessions/`, `src/chulk/scheduling/`, or another owning package.
- Tools, schemas, permissions, and bounded output: `src/chulk/tools/`, `src/chulk/capabilities.py`, and the corresponding documentation.
- CLI commands and terminal behavior: `src/chulk/main.py` and `src/chulk/cli/`.
- Control-plane server and operator interface: `src/chulk/server/`, `src/chulk/tui/`, and `src/chulk/cli/tui.py`.
- Dependencies, supported Python versions, and development tooling: `pyproject.toml` and `environment.yml`.
- Required automation and validation: `.github/workflows/ci.yml` and `scripts/`.
- User-facing behavior and examples: `README.md`, `docs/`, and `examples/`.

When documentation, tests, and implementation disagree, do not guess. Trace the current behavior through the owning code and public contract, determine which side is wrong, update the in-scope owner, and report any unresolved mismatch.

## Repository Map

ChulkHarness is a lightweight Python agent harness. State, prompts, model calls, tools, memory, skills, events, and traces should remain easy to follow from the code.

```text
src/chulk/
  api.py, __init__.py     # Supported public imports
  runtime.py              # Shared runtime assembly
  _sdk/                   # SDK facade and typed SDK contracts
  core/                   # Agent orchestration, actions, state, and transitions
  llm/                    # Provider registry, capabilities, and transports
  tools/                  # Tool registries, schemas, permissions, and implementations
  storage/                # Shared SQLite connection and migration policy
  memory/                 # Memory models, retrieval, extraction, and persistence
  sessions/               # Conversation and turn persistence
  skills/                 # Skill registry and bundled playbooks
  presets/                # Supported agent presets
  scheduling/             # Scheduled-job models, persistence, and tools
  mcp/, telegram/         # External protocol and channel adapters
  server/, tui/           # Control-plane server and operator interface
  cli/                    # Terminal formatting, progress, and commands
  tracing/                # Trace, artifact, and log primitives
tests/                    # Pytest suite outside the importable package
.chulk/                   # Declarative project config plus ignored runtime state
docs/                     # Maintained SDK and operator documentation
examples/                 # Credential-free public usage examples
typing_tests/             # External-consumer typing contract
```

Verify the current tree before relying on this high-level map.

## Architecture Conventions

### Runtime And Orchestration

- Keep runtime assembly in `src/chulk/runtime.py`; the CLI, SDK, adapters, and tests should consume the same builder.
- Ask the LLM layer for validated actions with `complete_action(...)`; orchestration must not parse provider text directly.
- Keep prompt text in `src/chulk/core/prompts.py` and composition in `src/chulk/core/prompt_builder.py`.
- Keep session-wide data in `AgentState` and per-message execution details in `TurnState`.
- Route action changes through the existing action-loop, transition, and turn-effect owners instead of adding a parallel loop.
- Record tool calls and observations with `ToolCallRecord` and `ObservationRecord` before persisting traces.
- Drive interactive progress, timing, and summaries from `TraceEvent` names through `Agent.event_callback`.
- Keep channel-specific input and delivery behavior in its adapter; reuse the shared runtime and public contracts underneath.

### Providers

- Keep provider-specific behavior inside `src/chulk/llm/`.
- Add providers through `src/chulk/llm/factory.py` and `src/chulk/llm/providers/`.
- Declare capabilities explicitly and preserve the shared provider lifecycle.
- Normalize provider-specific structured-output transports into shared action dataclasses before orchestration.
- Test request shaping, capability handling, fallback, retries, and errors with fake provider clients. Unit tests must not require credentials or network access.

### Public API

- Maintain supported import ergonomics through `src/chulk/__init__.py`, `src/chulk/api.py`, `src/chulk/tools/public.py`, `src/chulk/llm/public.py`, `src/chulk/skills/__init__.py`, and `src/chulk/presets/`.
- Do not expose internal implementation types accidentally. Follow the stability categories in `docs/release-policy.md`.
- When a public contract changes, update its exports, typed results/events/errors, `typing_tests/`, focused public-API tests, examples, and relevant documentation together.
- Keep synchronous and asynchronous behavior aligned where both are supported.

### Persistence And Memory

- Use `src/chulk/storage/sqlite.py` for shared connection, transaction, privacy, backup, and migration policy.
- Add database-wide schema changes as ordered forward-only migrations in `src/chulk/storage/migrations.py`. Do not rely on a freshly created database as proof that an upgrade works.
- Keep subsystem queries and mapping logic in the owning store. Memory-specific SQLite operations belong in `src/chulk/memory/sqlite_store.py`; session and scheduling operations belong in their respective stores.
- Test migrations from supported older schemas, rollback behavior, reopen behavior, and future-version rejection when schema contracts change.
- Keep tools, skills, and memory distinct:
  - Tool: a validated callable action.
  - Skill: procedural instructions loaded into context.
  - Memory: stored user, project, preference, and prior-work facts.
- Treat memories tagged `persona`, `preference`, `style`, or `workflow` as profile context.
- Never store secrets in long-term memory. `MEMORY.md` is an import/export format, not the runtime database.
- Include selected memory ids in memory trace events so retrieval remains debuggable.

### Tools And Safety

Model-generated tool arguments and external content are untrusted input. Enforce safety in Python, not only in prompts.

- Keep side-effecting tools behind registries, explicit capabilities, permission checks, and argument-schema validation.
- Return field-level validation observations for invalid calls so the model can recover.
- Restrict file operations to the normalized configured project root, including symlink-aware boundary checks.
- Block obviously destructive shell commands, use timeouts, and capture stdout, stderr, and exit status.
- Bound observations and tool output sent to the model. Preserve full truncated output only in sensitive trace or artifact storage.
- Log side effects without leaking secrets.
- Preserve cancellation, rollback, and cleanup behavior across ordinary errors and interruption paths.
- Do not add hidden destructive behavior, security bypasses, or permissive fallback paths for convenience.

## Documentation Sync

Update maintained documentation in the same change when implemented behavior changes:

- SDK surface, lifecycle, results, or errors: `docs/sdk.md`, `docs/sdk-errors.md`, and public examples.
- Configuration or providers: `docs/configuration.md`, `docs/providers.md`, and `.env.example`.
- Tools, permissions, or safety: `docs/tools.md`, `docs/permissions.md`, and `docs/safety.md`.
- Events or tracing: `docs/events.md` and `docs/tracing.md`.
- Memory, skills, MCP, or Telegram behavior: the matching topic under `docs/`.
- Install, quickstart, or primary workflows: `README.md`, `docs/quickstart.md`, and `examples/README.md`.
- Stability or release behavior: `docs/release-policy.md`.

Documentation describes implemented behavior; `TODO.md` describes planned and completed roadmap work. Keep both aligned without copying volatile implementation inventories into this guide. Run `python scripts/check_docs.py` after documentation or public-contract changes.

## Environment And Secrets

Use the project Conda environment:

```bash
conda env create -f environment.yml
conda activate chulk
```

Update an existing environment with:

```bash
conda env update -f environment.yml --prune
```

- Use `.env.example` and the configuration owners for the current environment surface; do not duplicate the complete variable inventory here.
- Keep local secrets in `.env`, which is ignored by Git.
- Never commit real keys, passwords, tokens, cookies, sensitive traces, artifacts, backups, exported user data, or local SQLite state.
- Never invent missing credentials. Report the exact missing dependency or configuration.
- Secret-free `.chulk/mcp.json` and `.chulk/skills/` content are declarative project configuration and may be committed.
- Treat every other `.chulk/` entry as runtime state unless its ownership and versioning policy explicitly says otherwise.

## Testing And Validation

- Add or update focused tests for every behavior change.
- Prefer fake or injected LLM clients and deterministic fixtures.
- Test the agent loop from the outside where practical: user input in, typed response, state, event, and trace evidence out.
- Run the smallest relevant pytest file or `-k` selection first. Broaden coverage when the changed owner is shared, persistent, security-sensitive, concurrent, or public.
- Keep tests offline by default. Live provider calls require explicit user intent.

CI is authoritative; inspect `.github/workflows/ci.yml` if commands drift. The full local verification set is:

```bash
python -m pytest --cov=chulk --cov-report=term-missing
python -m ruff check .
python -m mypy src/chulk typing_tests
python -m compileall src/chulk
python scripts/check_docs.py
```

Run credential-free examples when SDK assembly or public usage changes:

```bash
python examples/00_sdk_quickstart.py
python examples/repo_review_bot/app.py
```

Build and verify a clean wheel when packaging, exports, package data, dependencies, or entrypoints change:

```bash
python -m build --wheel --outdir dist
python scripts/check_wheel_install.py dist/*.whl --examples-dir examples
```

Validation rules:

- Documentation-only changes: run the relevant structural check and `git diff --check`.
- Runtime, provider, tool, storage, or public-contract changes: run focused tests followed by the full relevant verification set.
- Concurrency, cancellation, permissions, migrations, or destructive operations: add adversarial regression coverage.
- CLI changes: run the affected command with a credential-free path when possible.
- If the environment blocks validation, report the exact command and blocker instead of inventing configuration.

## Git Workflow

- Never push directly to `main`. Never force-push.
- Start implementation work in an isolated Git worktree on a feature branch. Read-only inspection does not require a worktree.
- Base the worktree on the intended, freshly fetched base branch unless the user names another base.
- Inspect `git status`, the relevant files, and the scoped diff before editing or staging.
- Preserve unrelated user changes and untracked files.
- Stage explicit paths, not the whole worktree.
- Before committing, run `git diff --check`, inspect the staged diff, and confirm no unrelated file is staged.
- Push only the feature branch and open or update a pull request when publishing is requested.
- Before the first push, amend local commits when needed. After a branch is pushed, use follow-up commits and do not rewrite remote history.
- Wait for required CI checks after pushing and repair failures before declaring the work complete.

## Style

- Use Python 3.11+ syntax.
- Prefer type hints on public functions and dataclasses.
- Keep comments short and useful.
- Avoid abstractions until they remove real complexity or establish a necessary ownership boundary.
- Keep Markdown practical and suitable for GitHub.
- When marking roadmap items complete, verify the corresponding code, tests, documentation, or command output first.
