# ChulkHarness

ChulkHarness is a lightweight Python agent harness for building LLM-driven workflows with explicit control over state, tools, memory, skills, prompts, and traces.

It is designed for developers who want a clear, inspectable agent runtime without starting from a large framework. The core idea is simple: keep the agent loop visible, keep tool execution auditable, and make every model decision traceable.

## Core Capabilities

- Conversation state for short-running sessions.
- Long-term memory backed by local storage.
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

This repository is at the initial implementation stage. The roadmap lives in [TODO.md](TODO.md).

Shell access and file-writing tools require strong permission checks. ChulkHarness will include guardrails, timeouts, output limits, and audit logs, but untrusted command execution should still be sandboxed in real deployments.

## Planned Structure

```text
agent/
  main.py
  agent.py
  llm_client.py
  memory.py
  tool_registry.py
  skill_registry.py
  shell_tool.py
  prompts.py
  logger.py
  config.py
  skills/
    shell/
      SKILL.md
    memory/
      SKILL.md
    files/
      SKILL.md
  tests/
```

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip
```

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

Create your local environment file:

```bash
cp .env.example .env
```

Run the current CLI:

```bash
python -m agent.main
```

Run tests:

```bash
pytest
```

## Environment

`.env` is intentionally ignored by Git. Use `.env.example` as the shared template for local configuration.

Planned environment variables:

```bash
OPENAI_API_KEY=
CHULK_MODEL=
CHULK_PROJECT_ROOT=
```

## Development Roadmap

The implementation should grow in phases:

- Phase 1: Minimal chat agent.
- Phase 2: Tool registry and tool-call loop.
- Phase 3: SQLite-backed memory.
- Phase 4: Lazy-loaded skills.
- Phase 5: Logging, tracing, tests, and reliability.
- Phase 6: Planning, reflection, semantic memory, and multi-step behavior.

See [TODO.md](TODO.md) for the full checklist.
