# Credential-free SDK quickstart

ChulkHarness supports Python 3.11–3.13. The distribution is `chulkharness` and
the Python package is `chulk`.

```bash
python -m pip install chulkharness
```

Save this as `quickstart.py`:

```python
from chulk import Agent, AgentConfig
from chulk.testing import ScriptedLLMClient

client = ScriptedLLMClient([
    {"type": "final_answer", "content": "Hello from Chulk."}
])
config = AgentConfig(project_root=".", runtime_dir=".chulk")

with Agent(config=config, llm=client, tools=[], skills=[]) as agent:
    result = agent.run_result("Say hello")
    print(result.content)
    print(f"trace_path: {result.trace_path}")
```

Run `python quickstart.py`. From a source checkout, the richer tool-call example
is `python examples/00_sdk_quickstart.py`.

Both paths default to `chulk.testing.ScriptedLLMClient`, so they need no API
key, provider account, or network connection. It performs a normal validated
tool-call loop and prints:

```text
mode: scripted
Order A-100 is packed and ships tomorrow.
runtime_dir: .../examples/runtime/00_sdk_quickstart
trace_path: .../traces/<conversation-id>.jsonl
```

`runtime_dir` is Chulk's private application state home. It contains the SQLite
store, traces, project skills, and MCP configuration as those features are used.
Do not commit runtime state. The trace is raw, sensitive diagnostic output; see
[tracing](tracing.md) before collecting or sharing it.

## Opt in to a live provider

Install the provider extra, configure credentials, and explicitly select live
mode:

```bash
python -m pip install "chulkharness[openai]"
export OPENAI_API_KEY=...
export CHULK_EXAMPLE_MODE=live
python examples/00_sdk_quickstart.py
```

Live output is intentionally nondeterministic and may incur provider charges.
The supported provider names are `openai`, `deepseek`, `local`,
`openai-compatible`, `openrouter`, `anthropic`, `bedrock`, and `gemini`. The
last five require an explicit `CHULK_MODEL`; install all optional provider SDKs
with `python -m pip install "chulkharness[providers]"`. Provider adapter tests
use injected fakes and make no network calls, so validating a real account,
model entitlement, key, and endpoint remains the application's responsibility.
For application code, inject an `LLMClient` when possible and construct the
agent with an explicit config and capability policy. Continue with the
[SDK guide](sdk.md), [providers](providers.md), and [permissions](permissions.md).
