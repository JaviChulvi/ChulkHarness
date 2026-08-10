# ChulkHarness

ChulkHarness is a lightweight Python agent harness with explicit state, model
calls, tools, memory, skills, events, and traces. It is designed for developers
who want an inspectable runtime and a small embedding API.

## Start here

- New SDK user: [credential-free quickstart](docs/quickstart.md)
- Embedding an application: [documentation index](docs/index.md)
- Runnable patterns: [SDK examples](examples/README.md)
- Testing an agent: [evaluation framework](docs/evals.md)
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

A minimal DeepSeek agent:

```python
from chulk import Agent, AgentConfig

with Agent(
    config=AgentConfig.deepseek(model="deepseek-v4-flash"),
    tools=[],
    skills=[],
) as agent:
    print(agent.run("Summarize: I was charged twice for order A-100."))
```

The same agent with two tools and two illustrative, pre-registered workflow
skills:

```python
from chulk import Agent, AgentConfig, Tool


@Tool
def order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "Monday"}


@Tool
def refund_eligibility(days_since_delivery: int) -> dict:
    return {"eligible": days_since_delivery <= 30, "window_days": 30}


with Agent(
    config=AgentConfig.deepseek(model="deepseek-v4-flash"),
    tools=[order_status, refund_eligibility],
    skills=["order-resolution", "returns-workflow"],
    system_prompt="Be concise and friendly.",
) as agent:
    print(agent.run("Where is order A-100, and can I return it after 12 days?"))
```

See the [SDK quickstart](examples/00_sdk_quickstart.py) and
[repository-review application](examples/repo_review_bot/README.md) for complete
runnable examples.

Create a credential-free evaluation suite with `chulk eval init`, run it with
`chulk eval run evals/suite.py:suite`, and enforce explicit quality thresholds
in CI. See [agent evaluations](docs/evals.md).

SDK agents default to read-only capabilities and store private runtime state
under `.chulk/`. Read [configuration](docs/configuration.md),
[permissions](docs/permissions.md), [safety](docs/safety.md), and the
[release policy](docs/release-policy.md) before production embedding.

## CLI

Install a provider extra, configure its credentials in a local `.env`, then run.
The exact provider names are `openai`, `deepseek`, `moonshot`, `local`,
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
