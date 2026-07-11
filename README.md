# ChulkHarness

ChulkHarness is a lightweight Python agent harness with explicit state, model
calls, tools, memory, skills, events, and traces. It is designed for developers
who want an inspectable runtime and a small embedding API.

## Start here

- New SDK user: [credential-free quickstart](docs/quickstart.md)
- Embedding an application: [documentation index](docs/index.md)
- Runnable patterns: [SDK examples](examples/README.md)
- Project direction: [roadmap](TODO.md)
- Vulnerability reporting: [security policy](SECURITY.md)

ChulkHarness supports Python 3.11, 3.12, and 3.13. The distribution name is
`chulkharness`; Python imports and the command are `chulk`.

```bash
python -m pip install chulkharness
python examples/00_sdk_quickstart.py
```

The first example is deterministic and needs no credentials. Hosted providers
are optional; for OpenAI use `python -m pip install "chulkharness[openai]"`.

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

Install a provider extra, configure its credentials in a local `.env`, then run:

```bash
chulk
chulk --once "Summarize this project"
chulk --show-config
```

Common interactive commands include `/help`, `/plan <request>`, `/approve`,
`/reject`, `/sessions`, `/resume <id>`, `/history`, `/memory`, `/skills`, and
`/mcp`. The CLI and SDK share the same runtime builder but use different safety
defaults; inspect `chulk --show-config` before enabling side effects.

## Development

```bash
conda env create -f environment.yml
conda activate chulk
python -m pytest
python -m ruff check chulk examples scripts
python -m mypy typing_tests
python -m compileall chulk examples scripts
python scripts/check_docs.py
```

Architecture and contribution rules live in [AGENTS.md](AGENTS.md). Keep local
credentials in `.env`; never commit API keys, `.chulk/` state, sensitive traces,
or local SQLite databases. The implementation roadmap is [TODO.md](TODO.md).

ChulkHarness is licensed under the [MIT License](LICENSE).
