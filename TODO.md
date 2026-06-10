# Chulk TODO

Practical development roadmap for building a lightweight Python agent harness with explicit state, tools, memory, skills, prompts, and traces.

## 1. Project Overview

ChulkHarness is a lightweight Python agent harness for building LLM-driven workflows while keeping the runtime explicit and inspectable.

It should provide clear building blocks for:

- [ ] Turning user messages into model prompts.
- [ ] Preserving short-term conversation state.
- [x] Storing and retrieving long-term memory.
- [ ] Deciding between direct answers and tool calls.
- [ ] Representing, validating, executing, and observing tool calls.
- [x] Lazy-loading procedural skills without flooding the prompt.
- [ ] Logging and tracing every meaningful step.

ChulkHarness favors transparent Python modules over a large framework surface. The aim is to make the agent loop, tool boundaries, memory injection, skill loading, and trace output easy to inspect and extend.

Core principles:

- [ ] Keep the architecture lightweight and inspectable.
- [ ] Prefer explicit data structures over hidden runtime magic.
- [ ] Make every agent decision traceable.
- [ ] Build incrementally: chat first, then tools, then memory, then skills.
- [ ] Treat safety as part of the design, especially for shell and file tools.

## 2. Initial Architecture

Proposed file structure:

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
  tools/
    registry.py
    shell.py
  skills/
    registry.py
  tracing/
    logger.py
  store.sqlite
  tests/
skills/
  shell/
    SKILL.md
  memory/
    SKILL.md
  files/
    SKILL.md
```

Module responsibilities:

- [ ] `src/main.py`
  - [ ] Provide the CLI entrypoint.
  - [ ] Read user input from the terminal.
  - [ ] Create the agent, registries, memory store, logger, and LLM client.
  - [ ] Run the conversation loop until the user exits.

- [ ] `src/core/agent.py`
  - [ ] Implement the core agent loop.
  - [ ] Build prompts from system instructions, messages, memories, skills, and tools.
  - [ ] Ask the model for either a direct answer or a tool call.
  - [ ] Execute tool calls through the tool registry.
  - [ ] Feed observations back into the model.
  - [ ] Track per-turn state and stop conditions.

- [ ] `src/llm/client.py`
  - [ ] Wrap the model provider API.
  - [ ] Support OpenAI first.
  - [ ] Hide provider-specific request and response details from the agent loop.
  - [ ] Provide text completion and structured JSON completion helpers.
  - [ ] Handle retries, timeouts, rate limits, and provider errors.

- [x] `src/memory/`
  - [x] Manage short-term conversation history.
  - [x] Manage long-term SQLite memory.
  - [x] Save, search, list, delete, and summarize memories.
  - [x] Keep memory retrieval separate from skill loading.

- [ ] `src/tools/registry.py`
  - [ ] Define the `Tool` dataclass.
  - [ ] Register callable tools.
  - [ ] List tool schemas for the model.
  - [ ] Validate and execute tool calls by name.
  - [ ] Convert tool outputs into observations.

- [x] `src/skills/registry.py`
  - [x] Define the `Skill` dataclass.
  - [x] Load skill metadata at startup.
  - [x] Select relevant skills for a user request.
  - [x] Lazy-load full `SKILL.md` files only when needed.
  - [x] Return skill instructions for prompt injection.

- [ ] `src/tools/shell.py`
  - [ ] Implement safe-ish command execution.
  - [ ] Run commands with timeout.
  - [ ] Capture stdout, stderr, and exit code.
  - [ ] Apply basic destructive-command blocking.
  - [ ] Log every executed command.

- [ ] `src/core/prompts.py`
  - [ ] Store prompt templates.
  - [ ] Keep base system prompts, tool prompts, memory prompts, skill prompts, and repair prompts separate.
  - [ ] Make prompt composition readable and testable.

- [x] `src/tracing/logger.py`
  - [x] Write structured logs and traces.
  - [x] Create one trace file per session.
  - [ ] Record user messages, selected memories, selected skills, model responses, tool calls, observations, and errors.

- [ ] `src/config.py`
  - [ ] Centralize model name, API settings, paths, limits, timeouts, and safety options.
  - [ ] Read environment variables.
  - [ ] Provide sensible defaults for local development.

- [ ] `src/store.sqlite`
  - [ ] Store long-term memories.
  - [ ] Optionally store conversations, traces, and tool-call history later.
  - [ ] Treat as local development data, not source code.

- [x] `skills/`
  - [x] Store domain-specific procedural instructions.
  - [x] Keep each skill in its own folder.
  - [x] Start with `shell`, `memory`, and `files`.

- [ ] `src/tests/`
  - [ ] Store unit tests and integration tests.
  - [ ] Mock the LLM client for deterministic agent-loop tests.

## 3. Core Agent Loop

Target behavior:

```text
User message
  -> add to short-term history
  -> search long-term memory
  -> select relevant skills
  -> list available tools
  -> build model prompt
  -> call model
  -> parse response
  -> if final answer: show answer
  -> if tool call: validate tool call
  -> run tool
  -> capture observation
  -> add observation to context
  -> call model again
  -> repeat until final answer or limit reached
```

Core loop tasks:

- [x] Create an `Agent` class.
- [x] Add a `run_turn(user_message: str) -> str` method.
- [x] Add a `conversation_id` for each session.
- [x] Read user input from the CLI.
- [x] Store each user message in short-term memory.
- [x] Build the base system prompt.
- [x] Add recent conversation history to the prompt.
- [x] Retrieve relevant long-term memories for the user request.
- [x] Inject only relevant memories into the prompt.
- [x] Select relevant skills for the user request.
- [x] Lazy-load selected full `SKILL.md` files.
- [x] Inject only selected skill instructions into the prompt.
- [x] Show available tool names, descriptions, and argument schemas to the model.
- [x] Ask the model for a structured response.
- [x] Parse the model response.
- [x] Detect `final_answer` responses.
- [x] Detect `tool_call` responses.
- [x] Validate requested tool names.
- [x] Validate requested tool arguments.
- [x] Execute the requested tool through `ToolRegistry`.
- [x] Convert tool output into an observation message.
- [x] Add the observation to the next model prompt.
- [x] Continue until the model returns `final_answer`.
- [x] Add a maximum number of tool-call iterations per user turn.
- [x] Return a helpful error if the max iteration limit is reached.
- [ ] Log every step of the loop.

Stop conditions:

- [x] Stop when the model returns a valid `final_answer`.
- [x] Stop when the tool-call iteration limit is reached.
- [ ] Stop when a tool returns a fatal safety error.
- [x] Stop when repeated JSON parsing failures exceed a repair limit.
- [x] Stop when the user exits the CLI.

## 4. LLM Client

Build a small provider wrapper before adding agent complexity.

- [x] Create `LLMClient` in `src/llm/client.py`.
- [x] Add a `complete(messages: list[dict]) -> str` method.
- [x] Add a `complete_json(messages: list[dict]) -> dict` method.
- [x] Support OpenAI first.
- [x] Read `OPENAI_API_KEY` from the environment.
- [x] Read the default model from `config.py`.
- [x] Keep provider-specific code inside `llm_client.py`.
- [x] Keep the rest of the agent provider-agnostic.
- [x] Add a provider interface that can support other backends later.
- [x] Support normal text responses.
- [x] Support structured JSON responses for tool calls.
- [x] Add `complete_action(...)` so the agent receives validated `AgentAction` objects instead of parsing provider text directly.
- [x] Use OpenAI native Structured Outputs with a strict action schema.
- [x] Use DeepSeek JSON Output mode with Chulk-side schema validation.
- [x] Normalize provider-specific action envelopes into one internal `AgentAction` type.
- [x] Keep malformed-action repair inside the LLM boundary as a fallback, not as normal agent-loop behavior.
- [x] Add request timeout handling.
- [x] Add retry handling for transient failures.
- [x] Add clear errors for missing API keys.
- [ ] Add clear errors for invalid model names.
- [ ] Add rate-limit handling.
- [ ] Log request metadata without logging secrets.
- [ ] Log model name, latency, and token usage if available.
- [x] Add tests with a mocked client.

Future provider support:

- [ ] Add a minimal local/mock provider for tests.
- [ ] Add support for local LLMs later.
- [x] Add DeepSeek as an additional hosted provider.
- [ ] Add support for additional hosted providers later.
- [x] Keep response normalization in one place.

## 5. Tool System

Tools are callable actions. They do things. The model may request a tool call, but Python validates and executes it.

Next big implementation milestone:

- [x] Build the model action parser for `final_answer` and `tool_call`.
- [x] Wire `ToolRegistry` into `Agent`.
- [x] Show available tools in the prompt.
- [x] Add a safe calculator tool first.
- [x] Run requested tools by name.
- [x] Feed tool observations back into the model.
- [x] Enforce a max tool-call iteration limit.
- [x] Add tests for direct answers, calculator calls, invalid JSON, unknown tools, and max-iteration handling.

Tool dataclass:

- [x] Create a `Tool` dataclass.
- [x] Include `name`.
- [x] Include `description`.
- [x] Include `args_schema`.
- [x] Include `callable`.
- [x] Include `requires_confirmation`.
- [x] Include `timeout_seconds`.
- [x] Include optional `metadata`.

Tool registry:

- [x] Create `ToolRegistry`.
- [x] Add `register(tool: Tool)`.
- [x] Add `get(name: str) -> Tool`.
- [x] Add `list_tools() -> list[Tool]`.
- [x] Add `tool_descriptions_for_prompt() -> str`.
- [x] Add `run(name: str, arguments: dict) -> ToolResult`.
- [x] Prevent duplicate tool names.
- [x] Return clear errors for unknown tools.
- [x] Validate tool arguments before execution.
- [x] Convert tool return values into observations.
- [x] Catch exceptions from tools.
- [x] Return safe error messages to the model.
- [x] Log every tool call and result.

Tool result structure:

- [x] Create a `ToolResult` dataclass.
- [x] Include `tool_name`.
- [x] Include `success`.
- [x] Include `observation`.
- [x] Include optional `stdout`.
- [x] Include optional `stderr`.
- [x] Include optional `exit_code`.
- [x] Include optional `error`.
- [x] Include optional `metadata`.

Example tools:

- [x] `calculator`
  - [x] Evaluate simple arithmetic.
  - [x] Avoid unrestricted `eval`.
  - [x] Support addition, subtraction, multiplication, division, powers, and parentheses.
  - [x] Return clear errors for invalid expressions.

- [x] `shell` / `run_cmd`
  - [x] Run shell commands with timeout and safety checks.
  - [x] Capture stdout, stderr, and exit code.
  - [x] Limit output size.
  - [x] Log every command.

- [x] `read_file`
  - [x] Read a text file from the project directory.
  - [x] Block reads outside the allowed root.
  - [x] Limit max file size.
  - [x] Return helpful errors for missing files or binary files.

- [x] `write_file`
  - [x] Write a text file inside the project directory.
  - [x] Block writes outside the allowed root.
  - [x] Require an explicit overwrite flag for overwrites.
  - [x] Log path and byte count.

- [x] `list_files`
  - [x] List files under the project directory.
  - [x] Support optional glob patterns.
  - [x] Hide ignored folders like `.git`, `.venv`, `__pycache__`, and large dependency directories.

- [x] `search_files`
  - [x] Search text files.
  - [x] Prefer `ripgrep` if available.
  - [x] Fall back to Python search if needed.
  - [x] Limit result count and output size.

## 6. Shell/CMD Tool Safety

The shell tool is dangerous. It should be treated as a local development feature unless it is protected by a sandbox with strong filesystem, network, process, and permission boundaries.

Requirements:

- [x] Implement shell execution in `src/tools/shell.py`.
- [x] Use `subprocess.run` or `asyncio.create_subprocess_shell`.
- [x] Run every command with a timeout.
- [x] Capture stdout.
- [x] Capture stderr.
- [x] Return exit code.
- [x] Use a configurable working directory.
- [x] Default the working directory to the project root.
- [x] Prevent commands from running in arbitrary directories unless explicitly allowed.
- [x] Prevent long-running commands from hanging the agent.
- [x] Kill timed-out processes.
- [x] Return a timeout observation to the model.
- [x] Limit stdout and stderr size.
- [x] Truncate large outputs with a clear marker.
- [x] Log every command executed.
- [x] Log working directory, timeout, exit code, stdout length, and stderr length.

Basic command blocking:

- [x] Block obviously destructive commands.
- [x] Block `rm -rf /`.
- [x] Block `rm -rf *`.
- [x] Block `mkfs`.
- [x] Block `dd`.
- [x] Block fork bombs.
- [x] Block shutdown/reboot commands.
- [x] Block commands that overwrite system paths.
- [x] Block writes outside the configured project directory where possible.
- [x] Add tests for blocked commands.

Future permission model:

- [ ] Add command allowlist mode.
- [ ] Add command denylist mode.
- [ ] Add user confirmation before risky commands.
- [ ] Add per-tool permission levels.
- [ ] Add read-only mode.
- [ ] Add a sandboxed execution backend.
- [ ] Add container execution for untrusted commands.
- [ ] Add a permission UI if a web interface is built later.

Safety notes:

- [x] Never hide destructive behavior behind a friendly tool name.
- [x] Never let the model bypass safety checks by changing wording.
- [x] Keep shell logs auditable.
- [x] Treat model-generated commands as untrusted input.

## 7. Memory System

Memory should be split into short-term and long-term memory.

### Short-Term Memory

Short-term memory is the current conversation context.

- [x] Store current conversation messages.
- [x] Track message role: `system`, `user`, `assistant`, `tool`, `observation`.
- [ ] Track message timestamp.
- [x] Keep messages in order.
- [x] Limit context size.
- [ ] Estimate token count if possible.
- [ ] Drop or summarize older messages when the context grows too large.
- [ ] Add conversation summarization later.
- [x] Keep short-term memory separate from long-term memory.

Short-term memory tasks:

- [x] Add `ConversationMemory`.
- [x] Add `add_user_message`.
- [x] Add `add_assistant_message`.
- [ ] Add `add_tool_call`.
- [x] Add `add_observation`.
- [x] Add `get_recent_messages`.
- [x] Add `trim_to_limit`.
- [ ] Add `summarize_older_messages` later.

### Long-Term Memory

Long-term memory stores durable facts, notes, preferences, and summaries.

Use SQLite initially because it is simple, inspectable, and local.

Next big implementation milestone:

- [x] Create `src/memory/sqlite_store.py`.
- [x] Create a `SQLiteMemoryStore` class that owns connection setup and schema initialization.
- [x] Store the database at `config.store_path`.
- [x] Add a durable `MemoryRecord` dataclass separate from short-term conversation messages.
- [x] Implement `save_memory`, `search_memory`, `list_memories`, `delete_memory`, and `summarize_memories`.
- [x] Use simple keyword search first; defer embeddings and FTS until the base store is stable.
- [x] Add memory tools: `save_memory`, `search_memory`, `list_memories`, `delete_memory`.
- [x] Register memory tools in the default `ToolRegistry`.
- [x] Retrieve relevant memories at the start of each agent turn.
- [x] Inject only relevant memories into the model prompt.
- [x] Track selected memory ids in `AgentState`.
- [x] Add tests proving memory can be saved, searched, listed, deleted, and injected into an agent turn.

Implementation order:

- [x] Step 1: SQLite schema and store API.
- [x] Step 2: Unit tests for save/search/list/delete/summarize.
- [x] Step 3: Memory tools using the store API.
- [x] Step 4: Default registry wiring for memory tools.
- [x] Step 5: Agent retrieval and prompt injection.
- [x] Step 6: Agent tests with mocked LLM showing retrieved memory in the prompt.
- [x] Step 7: TODO/README update and validation.

SQLite schema:

- [x] Create a `memories` table.
- [x] Include `id`.
- [x] Include `content`.
- [x] Include `created_at`.
- [x] Include `updated_at`.
- [x] Include `tags`.
- [x] Include `metadata`.
- [x] Include `importance`.
- [x] Create indexed `memory_tags` table for profile/preference tag lookup.
- [x] Include optional `embedding`.
- [x] Include `source`.
- [x] Include `confidence`.
- [x] Include `archived_at`.
- [x] Include access metadata: `access_count` and `last_accessed_at`.

Long-term memory functions:

- [x] `save_memory(content: str, tags: list[str], metadata: dict) -> str`
- [x] `search_memory(query: str, limit: int = 5) -> list[Memory]`
- [x] `list_memories(limit: int = 50) -> list[Memory]`
- [x] `delete_memory(memory_id: str) -> bool`
- [x] `summarize_memories(query: str | None = None) -> str`

Search phases:

- [x] Phase 1: implement simple keyword search.
- [x] Phase 2: add full-text search with SQLite FTS.
- [x] Phase 3: add embedding-based vector search.
- [x] Phase 4: add hybrid keyword plus embedding ranking.

Future memory features:

- [x] Add memory importance scores.
- [x] Add memory update.
- [x] Add memory delete.
- [x] Add memory compaction.
- [x] Add automatic memory extraction from explicit user messages.
- [x] Add duplicate-memory detection.
- [x] Add memory decay or archival.
- [x] Add memory source tracking.
- [x] Add memory confidence scores.
- [x] Add optional `MEMORY.md` import/export.
- [x] Add tests for all memory operations.

Memory injection rules:

- [x] Do not inject every memory into every prompt.
- [x] Inject only memories relevant to the current user request.
- [x] Keep injected memories short.
- [x] Include memory ids in `AgentState`.
- [x] Include memory ids in trace files.
- [x] Make it clear to the model that memories may be incomplete.

Persona and preference memory:

- [x] Treat memories tagged `persona`, `preference`, `style`, or `workflow` as profile memories.
- [x] Pull profile memories separately from task-relevant search results.
- [x] Use profile memories to shape tone, level of detail, and task-solving style.
- [x] Keep profile memories separate from skill instructions.
- [x] Add conflict handling when two preference memories disagree.
- [x] Add memory confidence or source metadata for user-profile facts.

## 8. Skill System

Skills are lazy-loaded procedural instructions. They are playbooks, not actions.

Important distinction:

- [x] Tool = callable action that executes code or performs an operation.
- [x] Skill = instructions, workflow, or behavior guide that tells the agent how to approach a domain.
- [x] Memory = durable facts, preferences, project context, and prior-work summaries that may shape a turn.

Example:

- [x] A `read_file` tool reads a file.
- [x] A `file editing` skill tells the agent how to safely inspect, patch, test, and summarize file edits.

Skill dataclass:

- [x] Create a `Skill` dataclass.
- [x] Include `name`.
- [x] Include `description`.
- [x] Include `path`.
- [x] Include optional `metadata`.
- [x] Include optional `keywords`.
- [x] Include optional `loaded_content`.

Skill registry:

- [x] Create `SkillRegistry`.
- [x] Scan `skills/` at startup.
- [x] Load only skill metadata at startup.
- [x] Do not load every full `SKILL.md` into context.
- [x] Read each skill description from front matter or a short metadata file.
- [x] Select relevant skills based on the user request.
- [x] Load full `SKILL.md` only when needed.
- [x] Inject loaded skill instructions into the prompt.
- [x] Track which skills were loaded for the turn.
- [x] Log selected skills and why they were selected.

Skill selection strategies:

- [x] Phase 1: keyword matching.
- [ ] Phase 2: LLM classifier/router.
- [ ] Phase 3: embedding similarity.
- [ ] Phase 4: hybrid ranking.

Skill prompt behavior:

- [x] Keep skill instructions separate from memory.
- [x] Keep skill instructions separate from tool schemas.
- [x] Include skill name and source path when injected.
- [x] Limit the number of loaded skills per turn.
- [x] Add a max character budget for skill content.
- [x] Prefer the most relevant skill over many weakly relevant skills.

Example skills:

- [x] `shell`
  - [x] How to use shell safely.
  - [x] When to prefer read-only commands.
  - [x] How to inspect output before acting.

- [x] `memory`
  - [x] When to save durable facts.
  - [x] When to search memory.
  - [x] How to avoid storing sensitive or irrelevant content.

- [x] `files`
  - [x] How to inspect files before editing.
  - [x] How to make small patches.
  - [x] How to validate changes.

- [ ] `web_research`
  - [ ] How to search, compare sources, and cite findings.
  - [ ] How to distinguish current facts from stable knowledge.

- [ ] `python_coding`
  - [ ] How to read Python projects.
  - [ ] How to add tests.
  - [ ] How to run formatting and validation.

- [ ] `debugging`
  - [ ] How to reproduce issues.
  - [ ] How to isolate root cause.
  - [ ] How to confirm fixes.

## 9. Prompting Strategy

Prompt templates should live in `src/core/prompts.py`.

Prompt types:

- [ ] Base system prompt.
- [ ] Tool-use prompt.
- [x] Skill-loaded prompt.
- [x] Memory-injected prompt.
- [ ] JSON tool-call prompt.
- [ ] Reflection prompt.
- [ ] Summarization prompt.
- [x] JSON repair prompt.

Prompt rules:

- [ ] Keep prompts readable.
- [ ] Keep prompts versioned in code.
- [x] Keep skills separate from memory.
- [x] Keep tool schemas separate from skill instructions.
- [x] Do not inject irrelevant memories.
- [x] Do not inject every skill.
- [x] Prefer structured outputs for tool calls.
- [x] Tell the model exactly which JSON formats are valid.
- [x] Tell the model when it may answer directly.
- [x] Tell the model when it should call a tool.
- [x] Tell the model to use observations rather than inventing tool results.
- [ ] Keep safety constraints visible in the prompt and enforced in Python.

Prompt composition tasks:

- [ ] Create `build_system_prompt`.
- [ ] Create `format_messages_for_prompt`.
- [x] Create `format_memories_for_prompt`.
- [x] Create `format_skills_for_prompt`.
- [x] Create `format_tools_for_prompt`.
- [ ] Create `format_observations_for_prompt`.
- [ ] Add tests that snapshot prompt output for simple cases.

## 10. Tool Calling Format

Start with a simple explicit JSON protocol.

Tool call format:

```json
{
  "type": "tool_call",
  "tool_name": "run_cmd",
  "arguments": {
    "command": "ls"
  }
}
```

Direct answer format:

```json
{
  "type": "final_answer",
  "content": "..."
}
```

Parsing tasks:

- [x] Create `parse_model_response(raw: str) -> AgentAction`.
- [x] Parse JSON safely.
- [x] Reject non-object JSON.
- [x] Reject missing `type`.
- [x] Validate known action types.
- [x] Validate `final_answer.content`.
- [x] Validate `tool_call.tool_name`.
- [x] Validate `tool_call.arguments`.
- [x] Support strict-provider `arguments_json` transport while normalizing to `arguments: dict`.
- [x] Handle invalid JSON.
- [x] Ask the model to repair malformed JSON.
- [x] Limit JSON repair attempts.
- [ ] Add helpful error observations for invalid tool calls.
- [x] Add tests for valid final answers.
- [x] Add tests for valid tool calls.
- [x] Add tests for malformed JSON.
- [x] Add tests for unknown tool names.
- [x] Add tests for invalid argument shapes.

Tool-call loop limits:

- [x] Set `MAX_TOOL_CALLS_PER_TURN`.
- [x] Stop after the limit is reached.
- [ ] Tell the model the limit.
- [x] Return a final error message if the limit is exceeded.
- [x] Log every iteration.

Future formats:

- [ ] Consider provider-native tool calling after the custom protocol is understood.
- [ ] Consider JSON Schema validation.
- [ ] Consider Pydantic models for action parsing.
- [ ] Consider multiple tool calls in one model response later.

## 11. Agent State

Create explicit state objects so the agent loop is inspectable.

State fields:

- [x] Current conversation id.
- [ ] Current turn id.
- [x] Messages.
- [x] Loaded memories.
- [x] Loaded skills.
- [ ] Available tools.
- [x] Tool calls.
- [x] Observations.
- [x] Errors.
- [x] Final answer.
- [ ] Token estimates if possible.
- [ ] Start time and end time.
- [ ] Model request count.
- [ ] Tool-call iteration count.

Implementation tasks:

- [x] Create `AgentState` dataclass.
- [ ] Create `TurnState` dataclass.
- [ ] Create `ToolCallRecord` dataclass.
- [ ] Create `ObservationRecord` dataclass.
- [ ] Add serialization helpers.
- [ ] Include state snapshots in traces.
- [ ] Keep state mutation centralized in the agent loop.

## 12. Logging and Tracing

The project should be easy to debug. A trace should explain exactly what the agent saw, decided, called, and returned.

Logging tasks:

- [x] Create `src/tracing/logger.py`.
- [x] Create a `TraceLogger`.
- [x] Create a trace file per session.
- [x] Use JSONL for trace events.
- [x] Log every user message.
- [x] Log selected memories.
- [x] Log selected skills.
- [ ] Log available tools.
- [ ] Log model prompts or prompt summaries.
- [x] Log model responses.
- [x] Log parsed actions.
- [x] Log tool calls.
- [x] Log tool arguments.
- [x] Log tool outputs.
- [x] Log tool errors.
- [x] Log final answers.
- [ ] Log timing information.
- [ ] Log token usage if available.

Trace event examples:

- [ ] `session_started`
- [x] `user_message`
- [x] `memory_search_started`
- [x] `memory_search_completed`
- [ ] `skill_selection_completed`
- [ ] `model_request_started`
- [x] `model_response`
- [ ] `model_response_parsed`
- [ ] `tool_call_started`
- [ ] `tool_call_completed`
- [ ] `tool_call_failed`
- [ ] `final_answer`
- [ ] `session_finished`

Debugging goals:

- [ ] Make it possible to replay a session mentally from the trace.
- [ ] Make it obvious why a tool was called.
- [ ] Make it obvious which memories were injected.
- [ ] Make it obvious which skills were loaded.
- [ ] Make failures visible without exposing secrets.

## 13. Safety and Permissions

Safety must be implemented in Python, not only suggested in prompts.

Shell safety:

- [ ] Block dangerous shell commands.
- [ ] Add command timeout.
- [ ] Add max output size.
- [ ] Use a fixed working directory.
- [ ] Log every command.
- [ ] Return clear safety errors.

Filesystem safety:

- [ ] Limit file reads to the project directory.
- [ ] Limit file writes to the project directory.
- [ ] Normalize paths before checking them.
- [ ] Block path traversal outside the project root.
- [ ] Prevent writes to `.env` unless explicitly confirmed later.
- [ ] Prevent writes to secrets or credential files unless explicitly confirmed later.
- [ ] Require confirmation before overwriting files later.

Permission model:

- [ ] Add user confirmation for risky operations later.
- [ ] Add per-tool permission levels.
- [ ] Add read-only mode.
- [ ] Add trusted and untrusted tool categories.
- [ ] Add command allowlists later.
- [ ] Add command denylists.
- [ ] Add a permission prompt in the CLI.
- [ ] Add a permission UI in a future web app.

Operational safety:

- [ ] Avoid hidden destructive behavior.
- [ ] Make all side effects visible in logs.
- [x] Do not store secrets in traces.
- [ ] Redact environment variables from logs.
- [ ] Redact API keys from errors.
- [ ] Add max tool-call iterations.
- [ ] Add max model retries.
- [x] Add max memory injection size.
- [ ] Add max skill injection size.
- [ ] Add clear audit logs.

## 14. Testing

Use tests to keep the harness understandable and safe as it grows.

Test setup:

- [ ] Choose `pytest`.
- [ ] Add test fixtures for temporary project directories.
- [ ] Add a fake LLM client.
- [ ] Add sample skill folders.
- [ ] Add temporary SQLite stores.
- [ ] Avoid real network calls in unit tests.

Tool tests:

- [ ] Test tool registration.
- [ ] Test duplicate tool registration.
- [ ] Test unknown tool lookup.
- [ ] Test running a registered tool.
- [ ] Test invalid tool arguments.
- [ ] Test tool exception handling.

Shell tool tests:

- [ ] Test successful command.
- [ ] Test stdout capture.
- [ ] Test stderr capture.
- [ ] Test non-zero exit code.
- [ ] Test timeout.
- [ ] Test blocked destructive command.
- [ ] Test output truncation.
- [ ] Test working directory behavior.

Memory tests:

- [x] Test memory database initialization.
- [x] Test `save_memory`.
- [x] Test `search_memory`.
- [x] Test `list_memories`.
- [x] Test `delete_memory`.
- [x] Test keyword/FTS search ranking.
- [x] Test empty search results.
- [x] Test source and confidence metadata.
- [x] Test embedding/vector retrieval.
- [x] Test duplicate detection and compaction.
- [x] Test archive and restore behavior.
- [x] Test Markdown import/export.
- [x] Test explicit memory extraction.

Skill tests:

- [x] Test skill metadata loading.
- [x] Test full skill lazy loading.
- [x] Test keyword skill selection.
- [x] Test missing skill file handling.
- [x] Test max skill limit.
- [x] Test prompt injection formatting.

Parser tests:

- [ ] Test final-answer JSON parsing.
- [ ] Test tool-call JSON parsing.
- [x] Test invalid JSON.
- [ ] Test missing fields.
- [ ] Test unknown action type.
- [ ] Test invalid arguments.

Agent loop tests:

- [x] Test final-answer-only turn with mocked LLM.
- [x] Test one tool call followed by final answer.
- [x] Test tool error followed by model recovery.
- [x] Test max tool-call limit.
- [x] Test memory retrieval injection.
- [x] Test skill selection injection.
- [x] Test trace events are written.

## 15. Milestones

### Phase 1: Minimal Chat Agent

Goal: chat with the model from a CLI with short-term history.

- [x] Create project package structure.
- [x] Add `config.py`.
- [x] Add `main.py` CLI loop.
- [x] Add `LLMClient`.
- [x] Support OpenAI text responses.
- [x] Add short-term conversation history.
- [x] Build base system prompt.
- [x] Send user messages to the model.
- [x] Print final answers.
- [x] Add clean exit command.
- [x] Add basic error handling.
- [x] Add tests with mocked LLM client.

Done when:

- [x] I can start the CLI.
- [x] I can send a message.
- [x] The model can respond.
- [x] Recent conversation history affects later responses.

### Phase 2: Tools

Goal: the model can request a tool call and receive the result.

- [x] Create `Tool` dataclass.
- [x] Create `ToolResult` dataclass.
- [x] Create `ToolRegistry`.
- [x] Add JSON action parser.
- [x] Add direct-answer JSON format.
- [x] Add tool-call JSON format.
- [x] Register tools manually at startup.
- [x] Add calculator tool.
- [x] Add tool-call loop.
- [x] Feed tool observations back to the model.
- [x] Add max tool-call iterations.
- [x] Add tool error handling.
- [x] Add tests for direct answers and calculator tool calls.
- [x] Add tests for invalid JSON, unknown tools, and max-iteration handling.
- [x] Add shell tool after the calculator loop is working safely.

Done when:

- [x] The agent can answer directly.
- [x] The agent can call the calculator.
- [x] The agent can run safe shell commands.
- [x] The agent can use tool output in its final answer.

### Phase 3: Memory

Goal: durable local memory backed by SQLite.

- [x] Create `src/memory/sqlite_store.py`.
- [x] Create SQLite store initialization.
- [x] Create memory schema with `id`, `content`, `created_at`, `updated_at`, `tags`, `metadata`, and `importance`.
- [x] Implement `save_memory`.
- [x] Implement `search_memory`.
- [x] Implement `list_memories`.
- [x] Implement `delete_memory`.
- [x] Implement `summarize_memories`.
- [x] Add simple keyword search.
- [x] Add memory tools for save/search/list/delete.
- [x] Register memory tools at startup.
- [x] Retrieve relevant memories during each turn.
- [x] Inject relevant memories into the prompt.
- [x] Track selected memory ids in `AgentState`.
- [x] Add memory tests.
- [x] Add agent tests proving memory retrieval affects the prompt.

Done when:

- [x] I can save a memory.
- [x] I can search memories.
- [x] I can list memories.
- [x] I can delete memories.
- [x] The agent exposes memory tools.
- [x] The agent can use relevant memories in later turns.
- [x] `AgentState.loaded_memory_ids` shows which memories were injected.

### Phase 4: Skills

Goal: lazy-load procedural instructions based on the user request.

- [x] Create `Skill` dataclass.
- [x] Create `SkillRegistry`.
- [x] Create initial skill folder structure.
- [x] Write `shell/SKILL.md`.
- [x] Write `memory/SKILL.md`.
- [x] Write `files/SKILL.md`.
- [x] Load skill metadata at startup.
- [x] Implement keyword-based skill selection.
- [x] Lazy-load full `SKILL.md` content only when selected.
- [x] Inject selected skill instructions into the prompt.
- [x] Log selected skills.
- [x] Add skill tests.

Done when:

- [x] The agent does not load every skill by default.
- [x] A shell-related request loads the shell skill.
- [x] A memory-related request loads the memory skill.
- [x] A file-related request loads the files skill.
- [x] The trace shows which skills were loaded.

### Phase 5: Reliability

Goal: make the harness easier to debug and harder to break.

- [x] Add structured trace logger.
- [x] Create trace file per session.
- [x] Log model responses.
- [x] Log parsed actions.
- [x] Log tool calls and observations.
- [x] Log errors.
- [x] Add retry handling to the LLM client.
- [x] Add timeout handling to the LLM client.
- [ ] Add safe output truncation.
- [x] Add JSON repair flow.
- [ ] Add stronger validation for tool arguments.
- [x] Add test coverage for common failures.
- [x] Add README usage instructions later.

Done when:

- [ ] Failed tool calls are understandable.
- [ ] Invalid model JSON is handled gracefully.
- [ ] A trace file can explain a full turn.
- [ ] Tests cover the main agent loop.

### Phase 6: Advanced Agent Behavior

Goal: experiment with richer agent behavior after the core mechanics are understood.

- [ ] Add planning prompt.
- [ ] Add explicit plan data structure.
- [ ] Add reflection prompt.
- [ ] Add post-tool reflection.
- [ ] Add conversation summarization.
- [x] Add local embedding/vector memory search.
- [x] Add memory importance scoring.
- [ ] Add external semantic embedding provider integration.
- [ ] Add multi-step task execution.
- [ ] Add optional web/search tool.
- [ ] Add skill router using an LLM classifier.
- [ ] Add embedding-based skill selection.
- [ ] Add hybrid skill ranking.
- [ ] Add provider-native tool calling as an optional mode.

Done when:

- [ ] The agent can plan before acting.
- [ ] The agent can perform multiple tool-backed steps.
- [ ] The agent can summarize older context.
- [ ] The agent can retrieve memories semantically.

## 16. Stretch Goals

Optional future directions:

- [ ] Web UI.
- [ ] FastAPI server.
- [ ] REST API for chat.
- [ ] WebSocket streaming.
- [ ] Plugin system.
- [ ] MCP-like tool interface.
- [ ] Vector database.
- [ ] Multi-agent mode.
- [ ] Local LLM support.
- [ ] OpenTelemetry-style traces.
- [ ] Permission UI.
- [ ] Skill marketplace or folder installer.
- [ ] Conversation replay UI.
- [ ] Trace viewer.
- [ ] Tool execution sandbox.
- [ ] Docker development environment.
- [ ] Configurable model/provider profiles.
- [ ] Import/export memories.
- [ ] Agent evaluation scripts.
- [ ] Benchmark prompts for regression testing.

## 17. Definition of Done

The project is successful when:

- [ ] I can chat with the agent from a CLI.
- [ ] The agent can maintain short-term conversation state.
- [ ] The agent can answer directly when no tool is needed.
- [ ] The agent can call tools through a registry.
- [ ] The agent can run safe shell commands.
- [ ] The agent can capture stdout, stderr, and exit code from shell commands.
- [ ] The agent blocks obviously dangerous shell commands.
- [x] The agent can save long-term memories.
- [x] The agent can search and retrieve long-term memories.
- [x] The agent injects only relevant memories into prompts.
- [x] The agent can load relevant skills lazily.
- [x] The agent does not inject every skill into every prompt.
- [ ] The agent can feed tool observations back into the model.
- [ ] The agent stops after a configured number of tool-call iterations.
- [ ] I can inspect logs to understand every major decision.
- [x] Trace files show user messages, selected memories, selected skills, tool calls, observations, errors, and final answers.
- [ ] The architecture is simple enough to inspect and maintain.
- [x] The code is tested.
- [ ] The safety limitations are documented clearly.

## Immediate Next Actions

- [x] Create the `src/` package.
- [x] Create `src/main.py`.
- [x] Create `src/config.py`.
- [x] Create `src/llm/client.py`.
- [x] Build the simplest possible CLI chat loop.
- [x] Add a mocked LLM test before wiring real API calls.
- [x] Add real OpenAI support.
- [x] Add short-term message history.
- [x] Confirm Phase 1 works before adding tools.
- [x] Complete Phase 2 tool-call loop and built-in tools.
- [x] Complete Phase 3 SQLite long-term memory.
- [x] Add SQLite memory schema and store API tests.
- [x] Add memory tools and register them at startup.
- [x] Inject retrieved memories into the agent prompt.
- [x] Start Phase 4: wire `SkillRegistry` into the agent loop.
- [x] Add keyword-based skill selection.
- [x] Lazy-load selected `SKILL.md` content into the prompt.
- [x] Add tests proving only relevant skills are loaded.
