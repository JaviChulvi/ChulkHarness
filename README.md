# ChulkHarness

ChulkHarness is a lightweight Python agent harness for building LLM-driven workflows with explicit control over state, tools, memory, skills, prompts, and traces.

It is designed for developers who want a clear, inspectable agent runtime without starting from a large framework. The core idea is simple: keep the agent loop visible, keep tool execution auditable, and make every model decision traceable.

## Core Capabilities

- Conversation state for short-running sessions.
- SQLite-backed long-term memory for durable facts, preferences, and project context.
- SQLite-backed session persistence and resume.
- Dynamic tool registration and execution.
- Built-in command/shell tooling with safety controls.
- Lazy-loaded skills for domain-specific workflows.
- Public `from chulk import Agent, Tool` API for embedding the runtime in Python code.
- Provider fallback chains that still satisfy the shared `LLMClient` contract.
- Structured model responses for tool calls and final answers.
- Explicit plan approval mode before tool execution.
- Hybrid MCP client support through hosted OpenAI MCP tools or Chulk-managed bridge tools.
- Trace logs that show messages, selected context, tool calls, observations, and errors.

## Design Principles

- Lightweight Python modules over hidden runtime magic.
- Explicit prompts, state, registries, and tool boundaries.
- Local-first development with simple files and SQLite.
- Safe defaults for commands and file operations.
- Provider-swappable LLM client design.
- Practical enough to extend, small enough to inspect.

## Current Scope

This repository has the Phase 1 chat loop, Phase 2 tool-call loop, Phase 3 SQLite-backed long-term memory, Phase 4 lazy-loaded skills, Phase 5 reliability basics, and the first Phase 6 workflows for plan mode plus session resume in place. It also exposes a small programmable API so the same runtime used by the CLI can be embedded with `from chulk import Agent`. The roadmap lives in [TODO.md](TODO.md).

The LLM layer is provider-swappable. OpenAI, DeepSeek, and local OpenAI-compatible providers use native tool-calling transports by default, with Chulk JSON as a fallback where needed. Provider responses normalize into the same internal action types before the agent loop sees them.

MCP client support uses a hybrid path. OpenAI Responses receives hosted MCP server definitions as native `{"type": "mcp"}` tools. DeepSeek and local providers receive Chulk-managed MCP bridge tools because their native API surface is function tools. Bridge calls are normal Chulk `Tool`s with `external_service` permission, so approval prompts and traces stay on the existing path.

The default `SoftwareEngineer` preset keeps its operating guidance in `chulk/presets/AGENT.md` and injects that playbook into the base system prompt. This keeps durable agent behavior separate from per-turn skills, memory, and tool schemas.

Long-term memory is stored in the local SQLite database at `chulk/store.sqlite`, which is ignored by Git. The agent retrieves relevant memories at the start of each turn and separately injects profile memories tagged `persona`, `preference`, `style`, or `workflow` so durable user preferences can shape responses without being confused with skills.

Memory search uses SQLite FTS when available, with a fallback keyword search and local vector reranking. Memories also track tags, source, confidence, importance, archive state, and access metadata. A human-readable `MEMORY.md` can be imported or exported through memory tools, but SQLite remains the runtime memory engine.

Skills live in the root-level `skills/` directory. Chulk loads only skill metadata at startup, chooses relevant skills with deterministic keyword matching, and injects full `SKILL.md` instructions only for selected skills in the current turn. Skill instructions stay separate from memory and tool schemas.

Traces are stored as JSONL files in `traces/`. Each model request logs the full message list sent to the provider by default, with obvious secrets redacted and a configurable prompt character cap.

Agent session state is split from per-turn state. `AgentState` tracks the conversation, while each user message gets a `TurnState` with timing, model request count, tool-call count, tool call records, observations, errors, and final status. Completed turn snapshots are written to traces so a run can be replayed from the logs.

Sessions are persisted in the same local SQLite database as long-term memory, using separate conversation tables. Use `/sessions` to list recent sessions, `/resume <conversation_id>` to resume one by full id or unique prefix, and `/history` to inspect recent persisted messages for the active session. Resumed sessions reload short-term history, append to the same trace file, and preserve pending `/plan` approvals across restarts.

Planning is optional and controlled per request from the CLI. Use `/plan <request>` for a planned turn. During planning, Chulk allows only read-only reconnaissance tools such as `list_files`, `read_file`, `search_files`, and memory search tools, then asks the model to propose a structured plan action before any mutating execution. Chulk pauses that turn until the user runs `/approve` or `/reject`, then injects the approved plan back into the prompt and traces steps as they move from `pending` to `in_progress`, `completed`, or `blocked`.

Large tool outputs are sent back to the model as bounded head/tail previews. When output is truncated, Chulk stores the full text as a local artifact under `traces/<conversation_id>_artifacts/` and includes the artifact path, length, and SHA-256 hash in the observation metadata. If the omitted middle may matter, the model is instructed to inspect the artifact or run a narrower follow-up tool call before answering. This keeps model context bounded without throwing away important details. Artifact files contain raw local output, so treat them as sensitive runtime data and keep `traces/` out of Git.

Tool arguments are validated against each tool schema before execution. Invalid calls produce structured observations with field-level validation errors, so the model can correct the call or explain the limitation instead of failing silently.

Shell access and file-writing tools include local guardrails, timeouts, output limits, path checks, and audit-friendly tool results, but untrusted command execution should still be sandboxed in real deployments.

## Planned Structure

```text
chulk/
  api.py
  main.py
  config.py
  runtime.py
  cli/
    commands.py
    progress.py
    terminal.py
  core/
    actions.py
    agent.py
    events.py
    observations.py
    prompt_builder.py
    prompts.py
    state.py
    trace_format.py
  llm/
    base.py
    capabilities.py
    client.py
    factory.py
    messages.py
    public.py
    providers/
      openai.py
      deepseek.py
  mcp/
    bridge.py
    config.py
  memory/
    constants.py
    extraction.py
    markdown.py
    models.py
    retrieval.py
    store.py
    sqlite_store.py
  tools/
    builtins.py
    calculator.py
    files.py
    memory.py
    public.py
    registry.py
    schema.py
    shell.py
  skills/
    registry.py
  sessions/
    models.py
    recorder.py
    sqlite_store.py
  tracing/
    logger.py
  presets/
    AGENT.md
    software_engineer.py
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

That installs ChulkHarness in editable mode with development, OpenAI, and MCP dependencies.

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
```

If the `chulk` command was installed before a package-layout change, refresh the editable install:

```bash
python -m pip install -e ".[dev,openai,mcp]"
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

# Local OpenAI-compatible server, such as LM Studio
CHULK_LLM_PROVIDER=local
CHULK_MODEL=google/gemma-4-12b-qat
CHULK_LOCAL_BASE_URL=http://localhost:1234/v1
CHULK_LOCAL_API_KEY=local

# Ollama can use the same local provider with a different base URL
CHULK_LLM_PROVIDER=local
CHULK_MODEL=gemma4:12b
CHULK_LOCAL_BASE_URL=http://localhost:11434/v1
CHULK_LOCAL_API_KEY=ollama
```

The CLI coding agent can use provider fallback with the same public provider objects exposed by `chulk.llm`. Configure the primary provider normally, then add fallback providers as a comma-separated list. Each fallback entry can be `provider` or `provider:model`:

```bash
CHULK_LLM_PROVIDER=deepseek
CHULK_MODEL=deepseek-v4-pro
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
CHULK_LLM_FALLBACK_PROVIDERS=openai:gpt-4.1-mini
```

At runtime this builds a `FallbackChain` equivalent to `FallbackChain([DeepSeekProvider(...), OpenAIProvider(...)])`. The CLI always uses `first_success`: try the primary provider first, then each fallback in order until one succeeds. The `local` provider can also appear in fallback chains, for example `CHULK_LLM_FALLBACK_PROVIDERS=local:google/gemma-4-12b-qat,openai:gpt-4.1-mini`.

Choose the tool permission profile with `CHULK_PERMISSION_PROFILE`. Built-in profiles are `read-only`, `workspace-write` (default), `trusted-local`, and `full-access`. The default allows read, write, and memory tools, but asks before shell, network, external-service, or destructive tool calls.

Configure remote Streamable HTTP MCP servers in `.chulk/mcp.json`:

```json
{
  "servers": [
    {
      "label": "docs",
      "transport": "streamable_http",
      "server_url": "https://mcp.example.com",
      "server_description": "Documentation search",
      "allowed_tools": ["search_docs"],
      "authorization_env": "DOCS_MCP_TOKEN",
      "approval": "always"
    }
  ]
}
```

Use `CHULK_MCP_CONFIG=/path/to/mcp.json` to override the default path. Put secret values in environment variables, not in `mcp.json`; Chulk resolves `authorization_env` at startup and redacts auth fields in config displays and traces. Use `/mcp` to inspect configured servers, provider path, auth status, allowed tools, and discovered bridge tools.

## Programmable API

Use the public API when you want Chulk inside another Python program. Capitalized names are the preferred public aliases.

Create the default coding agent:

```python
from chulk import Agent
from chulk.presets import SoftwareEngineer

a = Agent(preset=SoftwareEngineer())

print(a.run("Inspect this repository and summarize the CLI entrypoint."))
```

Pick specific built-in tools and skills:

```python
from chulk import Agent, Tools, Skills

a = Agent(
    tools=[Tools.read_file, Tools.search_files, Tools.apply_patch],
    skills=[Skills.files, Skills.shell],
)

print(a.run("Find where the CLI is wired and suggest a small cleanup."))
```

Expose one of your own Python functions as a tool:

```python
from chulk import Agent, Tool

@Tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id."""
    return f"Order {order_id} ships tomorrow."

a = Agent(
    tools=[lookup_order],
    skills=[],
)

print(a.run("When does order A-100 ship?"))
```

Use provider fallback:

```python
from chulk import Agent, Tools, Skills
from chulk.llm import FallbackChain, OpenAIProvider, DeepSeekProvider, LocalProvider
from chulk.presets import SoftwareEngineer

a = Agent(
    preset=SoftwareEngineer(),
    llm=FallbackChain(
        providers=[
            OpenAIProvider(model="gpt-4.1-mini"),
            LocalProvider(model="google/gemma-4-12b-qat", base_url="http://localhost:1234/v1"),
            DeepSeekProvider(model="deepseek-v4-flash"),
        ],
        strategy="first_success",
    ),
    tools=[Tools.read_file, Tools.search_files, Tools.apply_patch],
    skills=[Skills.files, Skills.shell, Skills.memory],
)

print(a.run("Inspect the project and update the README"))
```

Ask for an approval plan before mutation:

```python
from chulk import Agent
from chulk.presets import SoftwareEngineer

a = Agent(preset=SoftwareEngineer())

print(a.plan("Add a small public API example to the README."))
print(a.approve())
```

The public handle wraps the same explicit `chulk.core.Agent` used by the CLI. It supports `run(...)`, `plan(...)`, `approve()`, `reject()`, `state`, `conversation_id`, `trace_path`, `tool_registry`, and `skill_registry`.

Run the current CLI:

```bash
chulk
```

The interactive CLI always uses a Hulk-green terminal theme. During interactive turns, Chulk prints compact live progress lines such as memory search, skill selection, model requests, tool calls, command previews, elapsed time, and turn completion. Real terminals also show a small ASCII spinner while the model or a tool is working. Arrow up/down navigates prompt history for the active session when terminal `readline` support is available. The input prompt is intentionally short (`>`) so the transcript does not repeat a heavy label on every line. `chulk --once` remains plain output for scripting.

At the end of each turn, Chulk prints a compact summary with total time, model request count, tools used, selected memory count, selected skills, context estimate, and the trace path. Use `/context` to inspect the latest prompt section breakdown, `/quiet on` to hide live progress, `/verbose on` to include trace-event names in progress lines, and `/summary off` to hide the summary block.

Useful interactive commands:

- `/help`
- `/status`
- `/context`
- `/tools`
- `/mcp`
- `/sessions`
- `/resume <conversation_id>`
- `/history`
- `/trace`
- `/plan <request>`
- `/plan`
- `/approve`
- `/reject`
- `/quiet on|off`
- `/verbose on|off`
- `/summary on|off`
- `/clear`
- `/q`

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
pip install -e ".[dev,openai,mcp]"
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
- `apply_patch`
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
CHULK_LLM_FALLBACK_PROVIDERS=
CHULK_PERMISSION_PROFILE=workspace-write
CHULK_PROJECT_ROOT=
CHULK_MCP_CONFIG=
CHULK_DEEPSEEK_BASE_URL=https://api.deepseek.com
CHULK_LOCAL_BASE_URL=http://localhost:1234/v1
CHULK_LOCAL_API_KEY=
CHULK_HISTORY_LIMIT=20
CHULK_MAX_SKILLS_PER_TURN=3
CHULK_MAX_SKILL_CONTENT_CHARS=4000
CHULK_TRACE_MAX_PROMPT_CHARS=50000
CHULK_MAX_OBSERVATION_CHARS=12000
CHULK_MAX_TOOL_STDOUT_CHARS=8000
CHULK_MAX_TOOL_STDERR_CHARS=4000
CHULK_MAX_REFLECTION_ATTEMPTS=0
CHULK_LLM_TIMEOUT_SECONDS=60
CHULK_LLM_MAX_RETRIES=2
```

Prompt context limits are derived from `CHULK_LLM_PROVIDER` and `CHULK_MODEL` in `chulk/llm/capabilities.py`. Chulk uses the model's context window and default response reserve to budget prompt input, then compacts older conversation messages into a task-local summary when raw history would otherwise be omitted. The latest compact summary is persisted with the session, restored on `/resume`, and shown as its own section in `/context`. Each provider request receives an output limit based only on the remaining context for that specific prompt, not a fixed provider cap. Hosted providers require explicit model metadata; the `local` provider uses conservative default metadata for arbitrary local model names, with known local aliases registered explicitly.

Set `CHULK_MAX_REFLECTION_ATTEMPTS=1` to add a bounded pre-final reflection pass. The reviewer returns structured JSON, can approve the proposed final answer, or can add a `reflection_feedback` observation that sends the agent through one more action loop before the answer is shown.

Use `apply_patch` for normal file edits. It applies unified diffs atomically inside the project root and records changed paths plus SHA-256 metadata. `write_file` remains available for creating new UTF-8 files and guarded whole-file replacements; unsafe targets such as `.env`, credential files, SQLite stores, trace artifacts, caches, and dependency/build folders are blocked.

## Development Roadmap

The implementation now includes the core chat/tool/memory/skill runtime, reliability basics, explicit plan approval mode, session persistence, compact context summaries, usage/cost tracking, provider-native tool calling, and hybrid MCP client support. The next larger milestones are reflection, richer skill routing, local review workflows, optional web/search, subagents, and lifecycle hooks:

- Phase 1: Minimal chat agent.
- Phase 2: Tool registry and tool-call loop.
- Phase 3: SQLite-backed memory.
- Phase 4: Lazy-loaded skills.
- Phase 5: Logging, tracing, tests, and reliability hardening.
- Phase 6: Planning mode, reflection, semantic memory, and multi-step behavior.

See [TODO.md](TODO.md) for the full checklist.
