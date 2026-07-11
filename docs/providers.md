# Model providers

The base installation has no hosted-provider dependency. Choose one of three
embedding patterns:

- Inject an application-owned `LLMClient` through `Agent(llm=client)`.
- Use `chulk.testing.ScriptedLLMClient` for deterministic tests, examples, and
  offline evaluation.
- Install and configure a hosted or local provider.

OpenAI is optional:

```bash
python -m pip install "chulkharness[openai]"
```

Then use `AgentConfig.openai(...)` or environment variables such as
`OPENAI_API_KEY`. DeepSeek and local OpenAI-compatible endpoints use their
corresponding `AgentConfig` builders. Provider credentials remain host-owned;
do not put them in prompts, tool arguments, memory, or MCP files.

Injected clients implement `chulk.llm.LLMClient`. The shared runtime asks for
validated actions through `complete_action(...)`; provider-specific structured
output is normalized before orchestration. The scripted client intentionally
lives in `chulk.testing` and is not a top-level `chulk` export.

Live calls are nondeterministic, can fail transiently, and may cost money. Test
request shaping with injected fakes rather than real credentials. See
[quickstart](quickstart.md) and [SDK errors](sdk-errors.md).
