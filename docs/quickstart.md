# Credential-free SDK quickstart

ChulkHarness supports Python 3.11–3.13. It is not published to PyPI yet, so
clone the repository and install it from its root. The Python package is
`chulk`.

```bash
python -m pip install -e .
```

## Install a verified CI wheel

Every successful canonical `package` job uploads the exact wheel that passed
the clean-install smoke test. Its artifact name contains both the package
version and full source commit SHA:

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
intended for commit or pull-request testing, not as permanent releases. Treat
artifacts built from unreviewed pull requests as untrusted.

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
