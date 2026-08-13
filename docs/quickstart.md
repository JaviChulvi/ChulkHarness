# Credential-free SDK quickstart

ChulkHarness supports Python 3.11–3.13. It is not published to PyPI yet, so
clone the repository and install it from its root. The Python package is
`chulk`.

```bash
python -m pip install -e .
```

## Install a verified CI wheel

Every successful CI run on `main` uploads the exact wheel that passed the
clean-install smoke test. Its artifact name contains both the package version
and full source commit SHA:

```text
chulkharness-wheel-<version>-<full-commit-sha>
```

Choose the successful workflow run for the exact commit from the repository's
[Actions page](https://github.com/JaviChulvi/ChulkHarness/actions), then download,
verify, and install its artifact with the GitHub CLI:

```bash
repository=JaviChulvi/ChulkHarness
run_id="WORKFLOW_RUN_ID"
version="PACKAGE_VERSION"
commit_sha="FULL_COMMIT_SHA"
artifact_dir=.artifacts/chulkharness

gh run download "$run_id" \
  --repo "$repository" \
  --name "chulkharness-wheel-${version}-${commit_sha}" \
  --dir "$artifact_dir"

(cd "$artifact_dir" && shasum -a 256 -c SHA256SUMS)
python -m pip install "$artifact_dir"/chulkharness-*.whl
```

Pin both the workflow run and full commit SHA rather than selecting the latest
successful artifact implicitly. CI artifacts are retained for 30 days and are
intended for commit testing, not as permanent releases.

Save this as `quickstart.py`:

```python
from chulk import Agent, AgentConfig, Tool, ToolContext
from chulk.testing import ScriptedLLMClient

OrderStore = dict[str, dict[str, str]]
orders: OrderStore = {
    "A-100": {"status": "packed", "estimated_ship_date": "tomorrow"}
}


@Tool
def order_status(order_id: str, context: ToolContext[OrderStore]) -> dict[str, str]:
    """Look up one order in the host application's store."""
    return context.require_deps().get(
        order_id,
        {"error": f"Unknown order: {order_id}"},
    )


client = ScriptedLLMClient([
    {
        "type": "tool_call",
        "tool_name": "order_status",
        "arguments": {"order_id": "A-100"},
    },
    {
        "type": "final_answer",
        "content": "Order A-100 is packed and expected to ship tomorrow.",
    },
])
config = AgentConfig(project_root=".", runtime_dir=".chulk")

with Agent(
    config=config,
    llm=client,
    tools=[order_status],
    skills=[],
    deps=orders,
) as agent:
    result = agent.run_result("When will order A-100 ship?")

print(result.content)
print(f"trace_path: {result.trace_path}")
```

Run `python quickstart.py`. The model can only supply `order_id`; the order
store is application-owned data injected through `ToolContext`. From a source
checkout, run `python examples/00_sdk_quickstart.py` for the same pattern with
an explicit scripted-or-live switch and isolated example state.

Both paths default to `chulk.testing.ScriptedLLMClient`, so they need no API
key, provider account, or network connection. The scripted client makes the
model decisions repeatable; Chulk still validates the request, executes the
tool, and records the normal run lifecycle. The repository example prints:

```text
mode: scripted
Order A-100 is packed and expected to ship tomorrow.
tool_calls: order_status
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
python -m pip install -e ".[openai]"
export OPENAI_API_KEY=...
export CHULK_EXAMPLE_MODE=live
python examples/00_sdk_quickstart.py
```

Live output is intentionally nondeterministic and may incur provider charges.
The supported provider names are `openai`, `deepseek`, `moonshot`, `local`,
`openai-compatible`, `openrouter`, `anthropic`, `bedrock`, and `gemini`. The
last five require an explicit `CHULK_MODEL`; install all optional provider SDKs
with `python -m pip install -e ".[providers]"`. Provider adapter tests
use injected fakes and make no network calls, so validating a real account,
model entitlement, key, and endpoint remains the application's responsibility.
For application code, inject an `LLMClient` when possible and construct the
agent with an explicit config and capability policy. Continue with the
[SDK guide](sdk.md), [providers](providers.md), and [permissions](permissions.md).
