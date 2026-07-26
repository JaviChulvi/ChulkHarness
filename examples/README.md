# Chulk SDK Examples

These scripts show the public Python SDK surface for embedding Chulk in
applications and local automation. Start with the credential-free scripts:

Run them from the repository root:

```bash
python examples/00_sdk_quickstart.py
python examples/repo_review_bot/app.py
python examples/hosted_runtime/hosted_app.py
```

They use `chulk.testing.ScriptedLLMClient`, need no credentials, and make stable
CI examples. Set `CHULK_EXAMPLE_MODE=live` only for scripts that support an
explicit live path.

Most numbered examples call a live model and may incur provider charges. Install
all provider SDKs or the matching individual extra first:

```bash
python -m pip install -e ".[providers]"
```

Use `[openai]` for `openai`, `deepseek`, `local`, `openai-compatible`,
`openrouter`, or `bedrock`; `[anthropic]` for `anthropic`; and `[gemini]` for
`gemini`.

ChulkHarness is not yet published to PyPI, so run the install command from a
source checkout. Examples import `chulk`, and the CLI command is also `chulk`.

The live examples default to `openai`, so set:

```bash
export OPENAI_API_KEY=...
```

You can also select another configured provider:

```bash
export CHULK_LLM_PROVIDER=local
export CHULK_MODEL=your-local-model
export CHULK_LOCAL_BASE_URL=http://localhost:1234/v1
```

The exact provider names are `openai`, `deepseek`, `local`,
`openai-compatible`, `openrouter`, `anthropic`, `bedrock`, and `gemini`.
`CHULK_MODEL` is mandatory for `openai-compatible`, `openrouter`, `anthropic`,
`bedrock`, and `gemini`. See the provider guide for each provider's credential
aliases and base-URL requirements.

The automated provider tests use fake SDK clients and do not make live calls.
Run live examples only with your own test account and verify model access,
credentials, endpoint, billing, and quotas before relying on a provider.

Example runtime state is written under `examples/runtime/`, which is ignored by
Git.

Use the [documentation index](../docs/index.md) for task guides. In particular,
see [quickstart](../docs/quickstart.md), [providers](../docs/providers.md),
[tools](../docs/tools.md), [events](../docs/events.md), and
[safety](../docs/safety.md).

## Scripts

- `00_sdk_quickstart.py` is deterministic by default and supports explicit live-provider opt-in.
- `repo_review_bot/app.py` is a complete deterministic, read-only embedded application with a scrubbed trace walkthrough.
- `hosted_runtime/hosted_app.py` proves sync and async tenant-scoped embedding with in-memory services and zero local runtime files.
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
- `13_per_agent_skills.py` creates a temporary project with a `.chulk/skills/` catalog, scopes selectable skills with `Skills.only(...)`, and pins one always-loaded skill with `Skills.pin(...)`.
