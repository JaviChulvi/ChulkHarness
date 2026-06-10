# ChulkHarness

ChulkHarness is a lightweight Python agent harness for building LLM-driven workflows with explicit control over state, tools, memory, skills, prompts, and traces.

It is designed for developers who want a clear, inspectable agent runtime without starting from a large framework. The core idea is simple: keep the agent loop visible, keep tool execution auditable, and make every model decision traceable.

## Core Capabilities

- Conversation state for short-running sessions.
- SQLite-backed long-term memory for durable facts, preferences, and project context.
- Dynamic tool registration and execution.
- Built-in command/shell tooling with safety controls.
- Lazy-loaded skills for domain-specific workflows.
- Structured model responses for tool calls and final answers.
- Trace logs that show messages, selected context, tool calls, observations, and errors.

## Design Principles

- Lightweight Python modules over hidden runtime magic.
- Explicit prompts, state, registries, and tool boundaries.
- Local-first development with simple files and SQLite.
- Safe defaults for commands and file operations.
- Provider-swappable LLM client design.
- Practical enough to extend, small enough to inspect.

## Current Scope

This repository has the Phase 1 chat loop, Phase 2 tool-call loop, Phase 3 SQLite-backed long-term memory, and Phase 4 lazy-loaded skills in place. The roadmap lives in [TODO.md](TODO.md).

The LLM layer is provider-swappable. OpenAI uses native Structured Outputs for the agent action envelope, while DeepSeek uses JSON Output mode plus Chulk-side validation. Both paths normalize into the same internal action types before the agent loop sees them.

Long-term memory is stored in the local SQLite database at `src/store.sqlite`, which is ignored by Git. The agent retrieves relevant memories at the start of each turn and separately injects profile memories tagged `persona`, `preference`, `style`, or `workflow` so durable user preferences can shape responses without being confused with skills.

Memory search uses SQLite FTS when available, with a fallback keyword search and local vector reranking. Memories also track tags, source, confidence, importance, archive state, and access metadata. A human-readable `MEMORY.md` can be imported or exported through memory tools, but SQLite remains the runtime memory engine.

Skills live in the root-level `skills/` directory. Chulk loads only skill metadata at startup, chooses relevant skills with deterministic keyword matching, and injects full `SKILL.md` instructions only for selected skills in the current turn. Skill instructions stay separate from memory and tool schemas.

Shell access and file-writing tools include local guardrails, timeouts, output limits, path checks, and audit-friendly tool results, but untrusted command execution should still be sandboxed in real deployments.

## Planned Structure

```text
src/
  main.py
  config.py
  core/
    agent.py
    prompts.py
  llm/
    client.py
  memory/
    store.py
    sqlite_store.py
  tools/
    builtins.py
    calculator.py
    files.py
    memory.py
    registry.py
    shell.py
  skills/
    registry.py
  tracing/
    logger.py
  tests/
skills/
  shell/
    SKILL.md
  memory/
    SKILL.md
  files/
    SKILL.md
```

## Local Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate chulk
```

That installs ChulkHarness in editable mode with development and OpenAI dependencies.

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
```

If the `chulk` command was installed before a package-layout change, refresh the editable install:

```bash
python -m pip install -e ".[dev,openai]"
```

Create your local environment file:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before running chat against OpenAI.

Choose the LLM provider in `.env`:

```bash
# OpenAI
CHULK_LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
CHULK_MODEL=gpt-4.1-mini

# DeepSeek
CHULK_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
CHULK_MODEL=deepseek-v4-flash
```

Run the current CLI:

```bash
chulk
```

Send a single message and exit:

```bash
chulk --once "Hello"
```

Inspect local configuration:

```bash
chulk --show-config
```

Run tests:

```bash
python -m pytest
```

### Alternative: venv

Conda is the recommended setup for this project. If you prefer `venv`, install the same extras manually:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,openai]"
```

## Common Commands

Run the interactive CLI:

```bash
chulk
```

Run a one-shot message:

```bash
chulk --once "Hello"
```

Built-in tools currently registered at startup:

- `calculator`
- `run_cmd`
- `read_file`
- `write_file`
- `list_files`
- `search_files`
- `save_memory`
- `search_memory`
- `list_memories`
- `delete_memory`
- `update_memory`
- `summarize_memories`
- `archive_memory`
- `restore_memory`
- `compact_memories`
- `import_memories`
- `export_memories`

## Environment

`.env` is intentionally ignored by Git. Use `.env.example` as the shared template for local configuration.

Planned environment variables:

```bash
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
CHULK_LLM_PROVIDER=openai
CHULK_MODEL=
CHULK_PROJECT_ROOT=
CHULK_DEEPSEEK_BASE_URL=https://api.deepseek.com
CHULK_HISTORY_LIMIT=20
CHULK_MAX_SKILLS_PER_TURN=3
CHULK_MAX_SKILL_CONTENT_CHARS=4000
CHULK_LLM_TIMEOUT_SECONDS=60
CHULK_LLM_MAX_RETRIES=2
```

## Development Roadmap

The implementation is currently through Phase 4. The next major milestone is Phase 5:

- Phase 1: Minimal chat agent.
- Phase 2: Tool registry and tool-call loop.
- Phase 3: SQLite-backed memory.
- Phase 4: Lazy-loaded skills.
- Phase 5: Logging, tracing, tests, and reliability hardening.
- Phase 6: Planning, reflection, semantic memory, and multi-step behavior.

See [TODO.md](TODO.md) for the full checklist.
