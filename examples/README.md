# Chulk SDK Examples

These scripts show the public Python SDK surface added for embedding Chulk in
applications and local automation.

Run them from the repository root:

```bash
python examples/01_basic_agent.py
```

Most examples call a live model. By default they use `openai`, so set:

```bash
export OPENAI_API_KEY=...
```

You can also select another configured provider:

```bash
export CHULK_LLM_PROVIDER=local
export CHULK_MODEL=your-local-model
export CHULK_LOCAL_BASE_URL=http://localhost:1234/v1
```

Example runtime state is written under `examples/runtime/`, which is ignored by
Git.

## Scripts

- `01_basic_agent.py` creates an agent and returns a plain string.
- `02_agent_config.py` builds an agent with explicit `AgentConfig` paths.
- `03_builtin_tools.py` enables the built-in calculator tool.
- `04_custom_tool.py` defines typed custom tools with `@Tool`.
- `05_tool_permissions.py` demonstrates approval callbacks.
- `06_streaming_and_events.py` uses `on_delta` and `on_event`.
- `07_structured_run_result.py` prints `RunResult` metadata.
- `08_plan_approval.py` creates and approves a structured plan.
- `09_async_agent.py` runs the SDK from an async event loop.
- `10_mcp_programmatic.py` configures an MCP server in code.
- `11_software_engineer_preset.py` uses the coding-agent preset.
- `12_local_provider.py` targets a local OpenAI-compatible provider.
