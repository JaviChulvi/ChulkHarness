# ChulkHarness

ChulkHarness is a lightweight Python agent harness with explicit state, model
calls, tools, memory, skills, events, and traces. It is designed for developers
who want an inspectable runtime and a small embedding API.

## Start here

- New SDK user: [credential-free quickstart](docs/quickstart.md)
- Embedding an application: [documentation index](docs/index.md)
- Runnable patterns: [SDK examples](examples/README.md)
- Talking to a server agent from a phone: [Telegram adapter](docs/telegram.md)
- Project direction: [roadmap](TODO.md)
- Vulnerability reporting: [security policy](SECURITY.md)

ChulkHarness supports Python 3.11, 3.12, and 3.13. It is not published to
PyPI yet, so install it from a source checkout or use a
[verified wheel from a successful CI run](docs/quickstart.md#install-a-verified-ci-wheel).
Python imports and the command are `chulk`.

```bash
python -m pip install -e .
python examples/00_sdk_quickstart.py
```

The first example is deterministic and needs no credentials. Hosted providers
are optional. Install every provider dependency with
`python -m pip install -e ".[providers]"`, or choose an individual extra from
the [provider guide](docs/providers.md).

## SDK

```python
from chulk import Agent, AgentConfig
from chulk.testing import ScriptedLLMClient

client = ScriptedLLMClient([
    {"type": "final_answer", "content": "Hello from Chulk."}
])

with Agent(
    config=AgentConfig(project_root="."),
    llm=client,
    tools=[],
    skills=[],
) as agent:
    print(agent.run("Say hello"))
```

SDK agents default to read-only capabilities and store private runtime state
under `.chulk/`. Read [configuration](docs/configuration.md),
[permissions](docs/permissions.md), [safety](docs/safety.md), and the
[release policy](docs/release-policy.md) before production embedding.

## CLI

Install a provider extra, configure its credentials in a local `.env`, then run.
The exact provider names are `openai`, `deepseek`, `local`,
`openai-compatible`, `openrouter`, `anthropic`, `bedrock`, and `gemini`.

```bash
chulk
chulk --once "Summarize this project"
chulk --show-config
```

Common interactive commands include `/help`, `/plan <request>`, `/approve`,
`/reject`, `/sessions`, `/resume <id>`, `/history`, `/memory`, `/skills`, and
`/mcp`. The CLI and SDK share the same runtime builder but use different safety
defaults; inspect `chulk --show-config` before enabling side effects.

For private remote access from a phone, `chulk-telegram` runs an allowlisted
Telegram bot over outbound long polling. It requires no public server port; see
the [Telegram adapter guide](docs/telegram.md). An optional bounded Tavily tool
adds cited web search without exposing arbitrary network access.

## Development

```bash
conda env create -f environment.yml
conda activate chulk
python -m pytest
python -m ruff check src/chulk examples scripts
python -m mypy typing_tests
python -m compileall src/chulk examples scripts
python scripts/check_docs.py
```

Architecture and contribution rules live in [AGENTS.md](AGENTS.md). Keep local
credentials in `.env`. Secret-free `.chulk/mcp.json` and `.chulk/skills/`
playbooks are reviewable project configuration and may be committed; never
commit API keys, other `.chulk/` runtime state, sensitive traces, artifacts,
backups, or local SQLite databases. The implementation roadmap is
[TODO.md](TODO.md).

ChulkHarness is licensed under the [MIT License](LICENSE).
