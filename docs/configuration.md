# SDK configuration

`AgentConfig` is the explicit host-owned configuration boundary. Resolution is:

1. explicit constructor fields or `AgentConfig.from_env(...)` arguments;
2. process environment variables;
3. project `.env` values loaded by the internal config layer (process values
   override duplicate `.env` entries);
4. documented SDK defaults rooted at the current working directory.

Pass `project_root` in services, test runners, and desktop applications rather
than relying on the process working directory. Relative `runtime_dir`,
`store_path`, `traces_dir`, and `skills_dir` values resolve against that root.

```python
from pathlib import Path
from chulk import AgentConfig

root = Path(__file__).resolve().parent
config = AgentConfig(
    project_root=root,
    runtime_dir=root / ".chulk",
    permission_profile="read-only",
)
```

The default runtime home is `<project_root>/.chulk`: `store.sqlite` holds
memory/session state, `traces/` holds sensitive JSONL diagnostics, `skills/`
holds project playbooks, and `mcp.json` describes external servers. Applications
own cleanup, retention, permissions, backup, and multi-tenant isolation for
these paths. Never place real secrets in configuration files or traces.

Provider-specific options are covered in [providers](providers.md), while
runtime authority is covered in [permissions](permissions.md).

## Shared model settings

`CHULK_LLM_PROVIDER` selects one of `openai`, `deepseek`, `local`,
`openai-compatible`, `openrouter`, `anthropic`, `bedrock`, or `gemini`.
`CHULK_MODEL` overrides the selected model. OpenAI, DeepSeek, and local have
documented defaults; the other five providers require a non-empty
`CHULK_MODEL` so Chulk never guesses an account-specific model identifier.

`CHULK_LLM_FALLBACK_PROVIDERS` is a comma-separated sequence of
`provider:model` entries. A model may be omitted only for OpenAI, DeepSeek, or
local, whose defaults are known. For example:

```bash
export CHULK_LLM_PROVIDER=anthropic
export CHULK_MODEL=your-primary-model
export CHULK_LLM_FALLBACK_PROVIDERS=openrouter:vendor/fallback-model,openai:gpt-4.1-mini
```

Timeout and retry behavior remains shared across providers through
`CHULK_LLM_TIMEOUT_SECONDS` and `CHULK_LLM_MAX_RETRIES`. See the provider guide
for the exact credential and base-URL precedence.
