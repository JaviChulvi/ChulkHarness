# Chulk TODO

Practical development roadmap for building a lightweight Python agent harness with explicit state, tools, memory, skills, prompts, and traces.

## 1. Project Overview

ChulkHarness is a lightweight Python agent harness for building LLM-driven workflows while keeping the runtime explicit and inspectable.

It should provide clear building blocks for:

- [ ] Turning user messages into model prompts.
- [ ] Preserving short-term conversation state.
- [ ] Storing and retrieving long-term memory.
- [ ] Deciding between direct answers and tool calls.
- [ ] Representing, validating, executing, and observing tool calls.
- [ ] Lazy-loading procedural skills without flooding the prompt.
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

- [ ] `src/memory/`
  - [ ] Manage short-term conversation history.
  - [ ] Manage long-term SQLite memory.
  - [ ] Save, search, list, delete, and summarize memories.
  - [ ] Keep memory retrieval separate from skill loading.

- [ ] `src/tools/registry.py`
  - [ ] Define the `Tool` dataclass.
  - [ ] Register callable tools.
  - [ ] List tool schemas for the model.
  - [ ] Validate and execute tool calls by name.
  - [ ] Convert tool outputs into observations.

- [ ] `src/skills/registry.py`
  - [ ] Define the `Skill` dataclass.
  - [ ] Load skill metadata at startup.
  - [ ] Select relevant skills for a user request.
  - [ ] Lazy-load full `SKILL.md` files only when needed.
  - [ ] Return skill instructions for prompt injection.

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

- [ ] `src/tracing/logger.py`
  - [ ] Write structured logs and traces.
  - [ ] Create one trace file per session.
  - [ ] Record user messages, selected memories, selected skills, model responses, tool calls, observations, and errors.

- [ ] `src/config.py`
  - [ ] Centralize model name, API settings, paths, limits, timeouts, and safety options.
  - [ ] Read environment variables.
  - [ ] Provide sensible defaults for local development.

- [ ] `src/store.sqlite`
  - [ ] Store long-term memories.
  - [ ] Optionally store conversations, traces, and tool-call history later.
  - [ ] Treat as local development data, not source code.

- [ ] `skills/`
  - [ ] Store domain-specific procedural instructions.
  - [ ] Keep each skill in its own folder.
  - [ ] Start with `shell`, `memory`, and `files`.

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

- [ ] Create an `Agent` class.
- [ ] Add a `run_turn(user_message: str) -> str` method.
- [ ] Add a `conversation_id` for each session.
- [ ] Read user input from the CLI.
- [ ] Store each user message in short-term memory.
- [ ] Build the base system prompt.
- [ ] Add recent conversation history to the prompt.
- [ ] Retrieve relevant long-term memories for the user request.
- [ ] Inject only relevant memories into the prompt.
- [ ] Select relevant skills for the user request.
- [ ] Lazy-load selected full `SKILL.md` files.
- [ ] Inject only selected skill instructions into the prompt.
- [ ] Show available tool names, descriptions, and argument schemas to the model.
- [ ] Ask the model for a structured response.
- [ ] Parse the model response.
- [ ] Detect `final_answer` responses.
- [ ] Detect `tool_call` responses.
- [ ] Validate requested tool names.
- [ ] Validate requested tool arguments.
- [ ] Execute the requested tool through `ToolRegistry`.
- [ ] Convert tool output into an observation message.
- [ ] Add the observation to the next model prompt.
- [ ] Continue until the model returns `final_answer`.
- [ ] Add a maximum number of tool-call iterations per user turn.
- [ ] Return a helpful error if the max iteration limit is reached.
- [ ] Log every step of the loop.

Stop conditions:

- [ ] Stop when the model returns a valid `final_answer`.
- [ ] Stop when the tool-call iteration limit is reached.
- [ ] Stop when a tool returns a fatal safety error.
- [ ] Stop when repeated JSON parsing failures exceed a repair limit.
- [ ] Stop when the user exits the CLI.

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
- [ ] Support structured JSON responses for tool calls.
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
- [ ] Keep response normalization in one place.

## 5. Tool System

Tools are callable actions. They do things. The model may request a tool call, but Python validates and executes it.

Tool dataclass:

- [ ] Create a `Tool` dataclass.
- [ ] Include `name`.
- [ ] Include `description`.
- [ ] Include `args_schema`.
- [ ] Include `callable`.
- [ ] Include `requires_confirmation`.
- [ ] Include `timeout_seconds`.
- [ ] Include optional `metadata`.

Tool registry:

- [ ] Create `ToolRegistry`.
- [ ] Add `register(tool: Tool)`.
- [ ] Add `get(name: str) -> Tool`.
- [ ] Add `list_tools() -> list[Tool]`.
- [ ] Add `tool_descriptions_for_prompt() -> str`.
- [ ] Add `run(name: str, arguments: dict) -> ToolResult`.
- [ ] Prevent duplicate tool names.
- [ ] Return clear errors for unknown tools.
- [ ] Validate tool arguments before execution.
- [ ] Convert tool return values into observations.
- [ ] Catch exceptions from tools.
- [ ] Return safe error messages to the model.
- [ ] Log every tool call and result.

Tool result structure:

- [ ] Create a `ToolResult` dataclass.
- [ ] Include `tool_name`.
- [ ] Include `success`.
- [ ] Include `observation`.
- [ ] Include optional `stdout`.
- [ ] Include optional `stderr`.
- [ ] Include optional `exit_code`.
- [ ] Include optional `error`.
- [ ] Include optional `metadata`.

Example tools:

- [ ] `calculator`
  - [ ] Evaluate simple arithmetic.
  - [ ] Avoid unrestricted `eval`.
  - [ ] Support addition, subtraction, multiplication, division, powers, and parentheses.
  - [ ] Return clear errors for invalid expressions.

- [ ] `shell` / `run_cmd`
  - [ ] Run shell commands with timeout and safety checks.
  - [ ] Capture stdout, stderr, and exit code.
  - [ ] Limit output size.
  - [ ] Log every command.

- [ ] `read_file`
  - [ ] Read a text file from the project directory.
  - [ ] Block reads outside the allowed root.
  - [ ] Limit max file size.
  - [ ] Return helpful errors for missing files or binary files.

- [ ] `write_file`
  - [ ] Write a text file inside the project directory.
  - [ ] Block writes outside the allowed root.
  - [ ] Require confirmation later for overwrites.
  - [ ] Log previous path, new path, and byte count.

- [ ] `list_files`
  - [ ] List files under the project directory.
  - [ ] Support optional glob patterns.
  - [ ] Hide ignored folders like `.git`, `.venv`, `__pycache__`, and large dependency directories.

- [ ] `search_files`
  - [ ] Search text files.
  - [ ] Prefer `ripgrep` if available.
  - [ ] Fall back to Python search if needed.
  - [ ] Limit result count and output size.

## 6. Shell/CMD Tool Safety

The shell tool is dangerous. It should be treated as a local development feature unless it is protected by a sandbox with strong filesystem, network, process, and permission boundaries.

Requirements:

- [ ] Implement shell execution in `src/tools/shell.py`.
- [ ] Use `subprocess.run` or `asyncio.create_subprocess_shell`.
- [ ] Run every command with a timeout.
- [ ] Capture stdout.
- [ ] Capture stderr.
- [ ] Return exit code.
- [ ] Use a configurable working directory.
- [ ] Default the working directory to the project root.
- [ ] Prevent commands from running in arbitrary directories unless explicitly allowed.
- [ ] Prevent long-running commands from hanging the agent.
- [ ] Kill timed-out processes.
- [ ] Return a timeout observation to the model.
- [ ] Limit stdout and stderr size.
- [ ] Truncate large outputs with a clear marker.
- [ ] Log every command executed.
- [ ] Log working directory, timeout, exit code, stdout length, and stderr length.

Basic command blocking:

- [ ] Block obviously destructive commands.
- [ ] Block `rm -rf /`.
- [ ] Block `rm -rf *`.
- [ ] Block `mkfs`.
- [ ] Block `dd`.
- [ ] Block fork bombs.
- [ ] Block shutdown/reboot commands.
- [ ] Block commands that overwrite system paths.
- [ ] Block writes outside the configured project directory where possible.
- [ ] Add tests for blocked commands.

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

- [ ] Never hide destructive behavior behind a friendly tool name.
- [ ] Never let the model bypass safety checks by changing wording.
- [ ] Keep shell logs auditable.
- [ ] Treat model-generated commands as untrusted input.

## 7. Memory System

Memory should be split into short-term and long-term memory.

### Short-Term Memory

Short-term memory is the current conversation context.

- [ ] Store current conversation messages.
- [ ] Track message role: `system`, `user`, `assistant`, `tool`, `observation`.
- [ ] Track message timestamp.
- [ ] Keep messages in order.
- [ ] Limit context size.
- [ ] Estimate token count if possible.
- [ ] Drop or summarize older messages when the context grows too large.
- [ ] Add conversation summarization later.
- [ ] Keep short-term memory separate from long-term memory.

Short-term memory tasks:

- [ ] Add `ConversationMemory`.
- [ ] Add `add_user_message`.
- [ ] Add `add_assistant_message`.
- [ ] Add `add_tool_call`.
- [ ] Add `add_observation`.
- [ ] Add `get_recent_messages`.
- [ ] Add `trim_to_limit`.
- [ ] Add `summarize_older_messages` later.

### Long-Term Memory

Long-term memory stores durable facts, notes, preferences, and summaries.

Use SQLite initially because it is simple, inspectable, and local.

SQLite schema:

- [ ] Create a `memories` table.
- [ ] Include `id`.
- [ ] Include `content`.
- [ ] Include `created_at`.
- [ ] Include `updated_at`.
- [ ] Include `tags`.
- [ ] Include `metadata`.
- [ ] Include `importance`.
- [ ] Include optional `embedding` later.

Long-term memory functions:

- [ ] `save_memory(content: str, tags: list[str], metadata: dict) -> str`
- [ ] `search_memory(query: str, limit: int = 5) -> list[Memory]`
- [ ] `list_memories(limit: int = 50) -> list[Memory]`
- [ ] `delete_memory(memory_id: str) -> bool`
- [ ] `summarize_memories(query: str | None = None) -> str`

Search phases:

- [ ] Phase 1: implement simple keyword search.
- [ ] Phase 2: add full-text search with SQLite FTS.
- [ ] Phase 3: add embedding-based semantic search.
- [ ] Phase 4: add hybrid keyword plus embedding ranking.

Future memory features:

- [ ] Add memory importance scores.
- [ ] Add memory update.
- [ ] Add memory delete.
- [ ] Add memory compaction.
- [ ] Add automatic memory extraction from conversations.
- [ ] Add duplicate-memory detection.
- [ ] Add memory decay or archival.
- [ ] Add memory source tracking.
- [ ] Add memory confidence scores.
- [ ] Add tests for all memory operations.

Memory injection rules:

- [ ] Do not inject every memory into every prompt.
- [ ] Inject only memories relevant to the current user request.
- [ ] Keep injected memories short.
- [ ] Include memory ids in traces.
- [ ] Make it clear to the model that memories may be incomplete.

## 8. Skill System

Skills are lazy-loaded procedural instructions. They are playbooks, not actions.

Important distinction:

- [ ] Tool = callable action that executes code or performs an operation.
- [ ] Skill = instructions, workflow, or behavior guide that tells the agent how to approach a domain.

Example:

- [ ] A `read_file` tool reads a file.
- [ ] A `file editing` skill tells the agent how to safely inspect, patch, test, and summarize file edits.

Skill dataclass:

- [ ] Create a `Skill` dataclass.
- [ ] Include `name`.
- [ ] Include `description`.
- [ ] Include `path`.
- [ ] Include optional `metadata`.
- [ ] Include optional `keywords`.
- [ ] Include optional `loaded_content`.

Skill registry:

- [ ] Create `SkillRegistry`.
- [ ] Scan `skills/` at startup.
- [ ] Load only skill metadata at startup.
- [ ] Do not load every full `SKILL.md` into context.
- [ ] Read each skill description from front matter or a short metadata file.
- [ ] Select relevant skills based on the user request.
- [ ] Load full `SKILL.md` only when needed.
- [ ] Inject loaded skill instructions into the prompt.
- [ ] Track which skills were loaded for the turn.
- [ ] Log selected skills and why they were selected.

Skill selection strategies:

- [ ] Phase 1: keyword matching.
- [ ] Phase 2: LLM classifier/router.
- [ ] Phase 3: embedding similarity.
- [ ] Phase 4: hybrid ranking.

Skill prompt behavior:

- [ ] Keep skill instructions separate from memory.
- [ ] Keep skill instructions separate from tool schemas.
- [ ] Include skill name and source path when injected.
- [ ] Limit the number of loaded skills per turn.
- [ ] Add a max character budget for skill content.
- [ ] Prefer the most relevant skill over many weakly relevant skills.

Example skills:

- [ ] `shell`
  - [ ] How to use shell safely.
  - [ ] When to prefer read-only commands.
  - [ ] How to inspect output before acting.

- [ ] `memory`
  - [ ] When to save durable facts.
  - [ ] When to search memory.
  - [ ] How to avoid storing sensitive or irrelevant content.

- [ ] `files`
  - [ ] How to inspect files before editing.
  - [ ] How to make small patches.
  - [ ] How to validate changes.

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
- [ ] Skill-loaded prompt.
- [ ] Memory-injected prompt.
- [ ] JSON tool-call prompt.
- [ ] Reflection prompt.
- [ ] Summarization prompt.
- [ ] JSON repair prompt.

Prompt rules:

- [ ] Keep prompts readable.
- [ ] Keep prompts versioned in code.
- [ ] Keep skills separate from memory.
- [ ] Keep tool schemas separate from skill instructions.
- [ ] Do not inject irrelevant memories.
- [ ] Do not inject every skill.
- [ ] Prefer structured outputs for tool calls.
- [ ] Tell the model exactly which JSON formats are valid.
- [ ] Tell the model when it may answer directly.
- [ ] Tell the model when it should call a tool.
- [ ] Tell the model to use observations rather than inventing tool results.
- [ ] Keep safety constraints visible in the prompt and enforced in Python.

Prompt composition tasks:

- [ ] Create `build_system_prompt`.
- [ ] Create `format_messages_for_prompt`.
- [ ] Create `format_memories_for_prompt`.
- [ ] Create `format_skills_for_prompt`.
- [ ] Create `format_tools_for_prompt`.
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

- [ ] Create `parse_model_response(raw: str) -> AgentAction`.
- [ ] Parse JSON safely.
- [ ] Reject non-object JSON.
- [ ] Reject missing `type`.
- [ ] Validate known action types.
- [ ] Validate `final_answer.content`.
- [ ] Validate `tool_call.tool_name`.
- [ ] Validate `tool_call.arguments`.
- [ ] Handle invalid JSON.
- [ ] Ask the model to repair malformed JSON.
- [ ] Limit JSON repair attempts.
- [ ] Add helpful error observations for invalid tool calls.
- [ ] Add tests for valid final answers.
- [ ] Add tests for valid tool calls.
- [ ] Add tests for malformed JSON.
- [ ] Add tests for unknown tool names.
- [ ] Add tests for invalid argument shapes.

Tool-call loop limits:

- [ ] Set `MAX_TOOL_CALLS_PER_TURN`.
- [ ] Stop after the limit is reached.
- [ ] Tell the model the limit.
- [ ] Return a final error message if the limit is exceeded.
- [ ] Log every iteration.

Future formats:

- [ ] Consider provider-native tool calling after the custom protocol is understood.
- [ ] Consider JSON Schema validation.
- [ ] Consider Pydantic models for action parsing.
- [ ] Consider multiple tool calls in one model response later.

## 11. Agent State

Create explicit state objects so the agent loop is inspectable.

State fields:

- [ ] Current conversation id.
- [ ] Current turn id.
- [ ] Messages.
- [ ] Loaded memories.
- [ ] Loaded skills.
- [ ] Available tools.
- [ ] Tool calls.
- [ ] Observations.
- [ ] Errors.
- [ ] Final answer.
- [ ] Token estimates if possible.
- [ ] Start time and end time.
- [ ] Model request count.
- [ ] Tool-call iteration count.

Implementation tasks:

- [ ] Create `AgentState` dataclass.
- [ ] Create `TurnState` dataclass.
- [ ] Create `ToolCallRecord` dataclass.
- [ ] Create `ObservationRecord` dataclass.
- [ ] Add serialization helpers.
- [ ] Include state snapshots in traces.
- [ ] Keep state mutation centralized in the agent loop.

## 12. Logging and Tracing

The project should be easy to debug. A trace should explain exactly what the agent saw, decided, called, and returned.

Logging tasks:

- [ ] Create `src/tracing/logger.py`.
- [ ] Create a `TraceLogger`.
- [ ] Create a trace file per session.
- [ ] Use JSONL for trace events.
- [ ] Log every user message.
- [ ] Log selected memories.
- [ ] Log selected skills.
- [ ] Log available tools.
- [ ] Log model prompts or prompt summaries.
- [ ] Log model responses.
- [ ] Log parsed actions.
- [ ] Log tool calls.
- [ ] Log tool arguments.
- [ ] Log tool outputs.
- [ ] Log tool errors.
- [ ] Log final answers.
- [ ] Log timing information.
- [ ] Log token usage if available.

Trace event examples:

- [ ] `session_started`
- [ ] `user_message`
- [ ] `memory_search_started`
- [ ] `memory_search_completed`
- [ ] `skill_selection_completed`
- [ ] `model_request_started`
- [ ] `model_response_received`
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
- [ ] Do not store secrets in traces.
- [ ] Redact environment variables from logs.
- [ ] Redact API keys from errors.
- [ ] Add max tool-call iterations.
- [ ] Add max model retries.
- [ ] Add max memory injection size.
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

- [ ] Test memory database initialization.
- [ ] Test `save_memory`.
- [ ] Test `search_memory`.
- [ ] Test `list_memories`.
- [ ] Test `delete_memory`.
- [ ] Test keyword search ranking.
- [ ] Test empty search results.

Skill tests:

- [ ] Test skill metadata loading.
- [ ] Test full skill lazy loading.
- [ ] Test keyword skill selection.
- [ ] Test missing skill file handling.
- [ ] Test max skill limit.
- [ ] Test prompt injection formatting.

Parser tests:

- [ ] Test final-answer JSON parsing.
- [ ] Test tool-call JSON parsing.
- [ ] Test invalid JSON.
- [ ] Test missing fields.
- [ ] Test unknown action type.
- [ ] Test invalid arguments.

Agent loop tests:

- [ ] Test final-answer-only turn with mocked LLM.
- [ ] Test one tool call followed by final answer.
- [ ] Test tool error followed by model recovery.
- [ ] Test max tool-call limit.
- [ ] Test memory retrieval injection.
- [ ] Test skill selection injection.
- [ ] Test trace events are written.

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

- [ ] Create `Tool` dataclass.
- [ ] Create `ToolResult` dataclass.
- [ ] Create `ToolRegistry`.
- [ ] Register tools manually at startup.
- [ ] Add calculator tool.
- [ ] Add shell tool.
- [ ] Add JSON action parser.
- [ ] Add direct-answer JSON format.
- [ ] Add tool-call JSON format.
- [ ] Add tool-call loop.
- [ ] Feed tool observations back to the model.
- [ ] Add max tool-call iterations.
- [ ] Add tool error handling.
- [ ] Add tests for the registry, parser, and loop.

Done when:

- [ ] The agent can answer directly.
- [ ] The agent can call the calculator.
- [ ] The agent can run safe shell commands.
- [ ] The agent can use tool output in its final answer.

### Phase 3: Memory

Goal: durable local memory backed by SQLite.

- [ ] Create SQLite store.
- [ ] Create memory schema.
- [ ] Implement `save_memory`.
- [ ] Implement `search_memory`.
- [ ] Implement `list_memories`.
- [ ] Implement `delete_memory`.
- [ ] Add simple keyword search.
- [ ] Add manual memory tool or command.
- [ ] Retrieve relevant memories during each turn.
- [ ] Inject relevant memories into the prompt.
- [ ] Log selected memories.
- [ ] Add memory tests.

Done when:

- [ ] I can save a memory.
- [ ] I can search memories.
- [ ] The agent can use relevant memories in later turns.
- [ ] The trace shows which memories were injected.

### Phase 4: Skills

Goal: lazy-load procedural instructions based on the user request.

- [ ] Create `Skill` dataclass.
- [ ] Create `SkillRegistry`.
- [ ] Create initial skill folder structure.
- [ ] Write `shell/SKILL.md`.
- [ ] Write `memory/SKILL.md`.
- [ ] Write `files/SKILL.md`.
- [ ] Load skill metadata at startup.
- [ ] Implement keyword-based skill selection.
- [ ] Lazy-load full `SKILL.md` content only when selected.
- [ ] Inject selected skill instructions into the prompt.
- [ ] Log selected skills.
- [ ] Add skill tests.

Done when:

- [ ] The agent does not load every skill by default.
- [ ] A shell-related request loads the shell skill.
- [ ] A memory-related request loads the memory skill.
- [ ] A file-related request loads the files skill.
- [ ] The trace shows which skills were loaded.

### Phase 5: Reliability

Goal: make the harness easier to debug and harder to break.

- [ ] Add structured trace logger.
- [ ] Create trace file per session.
- [ ] Log model responses.
- [ ] Log parsed actions.
- [ ] Log tool calls and observations.
- [ ] Log errors.
- [ ] Add retry handling to the LLM client.
- [ ] Add timeout handling to the LLM client.
- [ ] Add safe output truncation.
- [ ] Add JSON repair flow.
- [ ] Add stronger validation for tool arguments.
- [ ] Add test coverage for common failures.
- [ ] Add README usage instructions later.

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
- [ ] Add semantic memory with embeddings.
- [ ] Add memory importance scoring.
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
- [ ] The agent can save long-term memories.
- [ ] The agent can search and retrieve long-term memories.
- [ ] The agent injects only relevant memories into prompts.
- [ ] The agent can load relevant skills lazily.
- [ ] The agent does not inject every skill into every prompt.
- [ ] The agent can feed tool observations back into the model.
- [ ] The agent stops after a configured number of tool-call iterations.
- [ ] I can inspect logs to understand every major decision.
- [ ] Trace files show user messages, selected memories, selected skills, tool calls, observations, errors, and final answers.
- [ ] The architecture is simple enough to inspect and maintain.
- [ ] The code is tested.
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
