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

The default runtime home is `<project_root>/.chulk`, but its contents have two
different ownership classes:

- Secret-free `.chulk/mcp.json` and `.chulk/skills/` playbooks are declarative,
  reviewable project configuration and may be committed.
- `.chulk/store.sqlite`, sidecars, backups, traces, artifacts, and all other
  `.chulk/` contents are sensitive runtime state and must remain ignored.

Applications own cleanup, permissions, backup, and multi-tenant isolation for
runtime state. Local memory retention can be opted into with
`AgentConfig(memory_retention_policy=MemoryRetentionPolicy(...))`; it is
archive-first, namespace-scoped, and does not control hosted memory services.
Never place real secrets in configuration files,
skills, or traces. MCP authorization values belong in environment variables
named by `authorization_env`.

Provider-specific options are covered in [providers](providers.md), while
runtime authority is covered in [permissions](permissions.md).

## Shared model settings

`CHULK_LLM_PROVIDER` selects one of `openai`, `deepseek`, `moonshot`, `local`,
`openai-compatible`, `openrouter`, `anthropic`, `bedrock`, or `gemini`.
`CHULK_MODEL` overrides the selected model. OpenAI, DeepSeek, Moonshot, and local have
documented defaults; the other five providers require a non-empty
`CHULK_MODEL` so Chulk never guesses an account-specific model identifier.

`CHULK_LLM_FALLBACK_PROVIDERS` is a comma-separated sequence of
`provider:model` entries. A model may be omitted only for OpenAI, DeepSeek,
Moonshot, or local, whose defaults are known. For example:

```bash
export CHULK_LLM_PROVIDER=anthropic
export CHULK_MODEL=your-primary-model
export CHULK_LLM_FALLBACK_PROVIDERS=openrouter:vendor/fallback-model,openai:gpt-4.1-mini
```

Timeout and retry behavior remains shared across providers through
`CHULK_LLM_TIMEOUT_SECONDS` and `CHULK_LLM_MAX_RETRIES`. The timeout is also the
per-chunk idle deadline for native async final-answer streams; each received
chunk resets it. See the provider guide for the exact credential and base-URL
precedence.

## Long-running conversation context

Chulk keeps recent raw messages plus a bounded, task-local checkpoint for older
conversation context. The checkpoint records the objective, constraints,
decisions, completed and blocked work, next actions, and safe evidence hints.
It is internal runtime state stored with the conversation summary; durable plan
state remains separate and authoritative.

Older raw messages stay in the session store. When exact evidence is needed,
the agent can deliberately use the read-only `session_search` and `session_read`
tools; Chulk does not automatically retrieve or inject historical matches.

`ContextBudget` reserves the model response allowance from the configured
context window and accounts for every serialized prompt section, including
native tool declarations and JSON fallback. Chulk compacts only conversation
history. If the remaining required prompt still cannot fit, it raises
`ConfigurationError` before making a provider request. The trace records
`context_budget_rejected` with the final context report.

## Configuration diagnostics

Run `chulk doctor` to check the selected primary provider and every configured
fallback for their required model, credential, endpoint, and optional SDK
dependency. The command does not make a billable provider request or prove that
an account can access a particular model. It also verifies that declarative
MCP/skill configuration is trackable while credential files, databases,
sidecars, backups, traces, and artifacts are ignored and not already tracked.
`chulk init` installs the matching narrow Git rules and upgrades older blanket
`.chulk/` ignore rules without relocating existing configuration.

Run `chulk --show-config` to inspect resolved non-secret settings. API keys are
reported only as set or not set. Provider base URLs keep their scheme, host,
port, and path, but Chulk removes user information, query parameters, and
fragments before printing them.
