# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Shape

ChulkHarness is a lightweight Python agent harness. The project should stay explicit and inspectable: state, prompts, model calls, tools, memory, skills, and traces should be easy to follow from the code.

Current package layout:

```text
src/
  main.py            # CLI entrypoint
  config.py          # Environment and runtime config
  core/              # Agent turn orchestration and prompts
  llm/               # Provider wrapper and LLM clients
  memory/            # Short-term memory and SQLite long-term memory
  tools/             # Tool primitives and implementations, including memory tools
  skills/            # Skill registry code
  tracing/           # Trace/log primitives
  tests/             # Pytest tests
skills/              # Root-level SKILL.md playbooks
```

Use `TODO.md` as the implementation roadmap. Advance it in order unless the user explicitly asks for a different slice.

## Development Principles

- Keep the harness small, readable, and modular.
- Prefer explicit dataclasses, registries, and plain Python functions over hidden control flow.
- Keep provider-specific logic inside `src/llm/`.
- Ask the LLM layer for validated actions with `complete_action(...)`; the agent loop should not parse provider text directly.
- Keep provider-specific structured-output transports normalized into the shared action dataclasses before orchestration.
- Keep prompt text in `src/core/prompts.py`.
- Keep skill playbooks in root-level `skills/`, outside the Python package.
- Keep side-effecting tools behind registries and safety checks.
- Do not mix skills, tools, and memory:
  - Tool: callable action.
  - Skill: procedural instructions loaded into context from root-level `skills/`.
  - Memory: stored user, project, preference, and prior-work facts in SQLite.
- Treat memories tagged `persona`, `preference`, `style`, or `workflow` as profile context that can shape tone, level of detail, and task-solving style.
- Do not store secrets in long-term memory.
- SQLite is the runtime memory engine; `MEMORY.md` is only a human-readable import/export format.
- Memory trace events should include selected memory ids so memory behavior can be debugged across sessions.
- When marking TODO items complete, verify the corresponding code, tests, or command output first.

## Environment

Use the project Conda environment:

```bash
conda env create -f environment.yml
conda activate chulk
```

Update an existing environment with:

```bash
conda env update -f environment.yml --prune
```

Local secrets belong in `.env`, which is ignored by Git. Keep `.env.example` safe to commit.

Important environment variables:

```bash
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
CHULK_LLM_PROVIDER=openai
CHULK_MODEL=
CHULK_PROJECT_ROOT=
CHULK_DEEPSEEK_BASE_URL=https://api.deepseek.com
CHULK_HISTORY_LIMIT=20
CHULK_LLM_TIMEOUT_SECONDS=60
CHULK_LLM_MAX_RETRIES=2
```

Never commit real API keys, secrets, traces with secrets, or local SQLite state.

The default memory database is `src/store.sqlite`; it is local runtime state and must stay ignored.

## Commands

Run tests:

```bash
python -m pytest
```

Compile-check the package:

```bash
python -m compileall src
```

Run CLI metadata commands:

```bash
python -m src.main --version
python -m src.main --show-config
```

Run a one-shot chat call:

```bash
python -m src.main --once "Hello"
```

This requires `OPENAI_API_KEY` unless the code path injects a fake LLM client in tests.

## Testing Expectations

- Add or update tests for every behavior change.
- Prefer fake or injected LLM clients in tests.
- Do not require network access or a real OpenAI API key for unit tests.
- Test the agent loop from the outside where possible: user message in, assistant response/state/log evidence out.
- Keep OpenAI tests focused on request-shaping with fake clients, not live API calls.
- After meaningful changes, run:

```bash
python -m pytest
python -m compileall src
```

## Safety Rules

Shell and file tools are high-risk. Enforce safety in Python, not only in prompts.

- Block obviously destructive shell commands.
- Use command timeouts.
- Capture stdout, stderr, and exit code.
- Limit output size.
- Restrict file reads/writes to the configured project root.
- Normalize paths before checking boundaries.
- Log side effects.
- Treat model-generated tool arguments as untrusted input.

Do not add hidden destructive behavior or bypasses for convenience.

## Working With Git

- Preserve unrelated user changes.
- Stage only files relevant to the requested task.
- Do not rewrite history or run destructive Git commands unless the user explicitly asks.
- Before saying work is commit-ready, check:

```bash
git status --short --branch
python -m pytest
python -m compileall src
```

## Style

- Use Python 3.11+ syntax.
- Prefer type hints on public functions and dataclasses.
- Keep comments short and useful.
- Avoid large abstractions before they remove real complexity.
- Keep README and TODO aligned with implemented behavior.
- Keep Markdown practical and suitable for GitHub.

## Roadmap Notes

Phase 1 through Phase 3 are implemented. Phase 4 is the next large milestone. Do not blur skills with memory:

- Phase 1: config, CLI, LLM client, short-term history, final answers.
- Phase 2: tool dataclasses, registry, calculator, shell tool, tool-call loop.
- Phase 3: SQLite long-term memory, memory tools, and relevant memory prompt injection.
- Phase 4: lazy-loaded skills.
- Phase 5: logging, traces, reliability.
- Phase 6: planning, reflection, semantic memory, multi-step behavior.

If a requested change touches a later phase, implement only the smallest necessary bridge unless the user asks to move that phase forward.
