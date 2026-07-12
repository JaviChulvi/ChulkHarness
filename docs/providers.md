# Model providers

Chulk has eight built-in provider names:

| Provider name | Transport | Install extra | Default model | Base URL |
| --- | --- | --- | --- | --- |
| `openai` | OpenAI Responses | `openai` | `gpt-4.1-mini` | Provider default |
| `deepseek` | OpenAI-compatible Chat Completions | `openai` | `deepseek-v4-flash` | `https://api.deepseek.com` |
| `local` | OpenAI-compatible Chat Completions | `openai` | `google/gemma-4-12b-qat` | `http://localhost:1234/v1` |
| `openai-compatible` | OpenAI-compatible Chat Completions | `openai` | Required | Required |
| `openrouter` | OpenAI-compatible Chat Completions | `openai` | Required | `https://openrouter.ai/api/v1` |
| `anthropic` | Anthropic Messages | `anthropic` | Required | Provider default |
| `bedrock` | Bedrock OpenAI-compatible Chat Completions | `openai` | Required | Required; no default |
| `gemini` | Google GenerateContent | `gemini` | Required | Provider default |

The base installation has no hosted-provider dependency. Install one SDK extra,
or install every provider SDK together:

```bash
python -m pip install "chulkharness[openai]"
python -m pip install "chulkharness[anthropic]"
python -m pip install "chulkharness[gemini]"
python -m pip install "chulkharness[providers]"
```

The `openai` extra is shared by OpenAI, DeepSeek, local, generic
OpenAI-compatible, OpenRouter, and Bedrock because those adapters use the
OpenAI Python SDK transport. The `providers` extra installs all three SDK
families. It does not install the optional `mcp` extra.

## Environment resolution

Select a provider with `CHULK_LLM_PROVIDER`. Set `CHULK_MODEL` for
`openai-compatible`, `openrouter`, `anthropic`, `bedrock`, and `gemini`; these
five intentionally have no guessed model. The same rule applies to fallback
entries: use `provider:model` in `CHULK_LLM_FALLBACK_PROVIDERS`.

Explicit `AgentConfig` fields override process environment values. Process
environment values override duplicate entries in the project `.env`. Within a
provider's credential aliases, Chulk uses this exact first-nonempty order:

| Provider | Credential precedence | Base-URL precedence |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | Provider default |
| `deepseek` | `CHULK_DEEPSEEK_API_KEY`, `DEEPSEEK_API_KEY` | `CHULK_DEEPSEEK_BASE_URL`, then the documented default |
| `local` | `CHULK_LOCAL_API_KEY` (optional) | `CHULK_LOCAL_BASE_URL`, then the documented default |
| `openai-compatible` | `CHULK_OPENAI_COMPATIBLE_API_KEY` | `CHULK_OPENAI_COMPATIBLE_BASE_URL` (required) |
| `openrouter` | `CHULK_OPENROUTER_API_KEY`, `OPENROUTER_API_KEY` | `CHULK_OPENROUTER_BASE_URL`, then the documented default |
| `anthropic` | `CHULK_ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY` | `CHULK_ANTHROPIC_BASE_URL`, then the provider default |
| `bedrock` | `CHULK_BEDROCK_API_KEY`, `BEDROCK_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK` | `CHULK_BEDROCK_BASE_URL`, `CHULK_BASE_URL` (legacy alias); one is required |
| `gemini` | `CHULK_GEMINI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `CHULK_GEMINI_BASE_URL`, then the provider default |

Provider credentials remain host-owned. Do not put them in prompts, tool
arguments, memory, MCP files, source control, or shared traces.

## Configuration examples

OpenAI, DeepSeek, and local can use their documented model defaults:

```bash
export CHULK_LLM_PROVIDER=openai
export OPENAI_API_KEY=...
```

```bash
export CHULK_LLM_PROVIDER=local
export CHULK_MODEL=your-loaded-model
export CHULK_LOCAL_BASE_URL=http://localhost:1234/v1
```

A generic hosted OpenAI-compatible endpoint requires a key, base URL, and
explicit model:

```bash
export CHULK_LLM_PROVIDER=openai-compatible
export CHULK_MODEL=vendor/model-id
export CHULK_OPENAI_COMPATIBLE_API_KEY=...
export CHULK_OPENAI_COMPATIBLE_BASE_URL=https://models.example/v1
```

OpenRouter, Anthropic, and Gemini require explicit models and accept their
standard ecosystem key names:

```bash
export CHULK_LLM_PROVIDER=openrouter
export CHULK_MODEL=vendor/model-id
export OPENROUTER_API_KEY=...
```

```bash
export CHULK_LLM_PROVIDER=anthropic
export CHULK_MODEL=your-claude-model-id
export ANTHROPIC_API_KEY=...
```

```bash
export CHULK_LLM_PROVIDER=gemini
export CHULK_MODEL=your-gemini-model-id
export GEMINI_API_KEY=...
```

Bedrock support targets its OpenAI-compatible bearer-token endpoint, not the
native boto3/Converse transport. Supply the complete endpoint for the selected
AWS region and an explicit model; Chulk deliberately has no Bedrock endpoint or
model default:

```bash
export CHULK_LLM_PROVIDER=bedrock
export CHULK_MODEL=your-bedrock-model-id
export CHULK_BEDROCK_API_KEY=...
export CHULK_BEDROCK_BASE_URL=https://your-bedrock-openai-endpoint/openai/v1
```

`BEDROCK_API_KEY` and `AWS_BEARER_TOKEN_BEDROCK` are supported key aliases.
`CHULK_BASE_URL` remains a legacy base-URL alias, but new configuration should
use `CHULK_BEDROCK_BASE_URL`.

## SDK construction

The matching builders are `AgentConfig.openai(...)`, `.deepseek(...)`,
`.local(...)`, `.openai_compatible(...)`, `.openrouter(...)`,
`.anthropic(...)`, `.bedrock(...)`, and `.gemini(...)`. Applications can also
inject an application-owned `LLMClient` through `Agent(llm=client)`. The shared
runtime asks providers for validated actions through `complete_action(...)` and
normalizes provider-specific tool calls before orchestration.

`chulk.testing.ScriptedLLMClient` is the deterministic choice for unit tests,
examples, and offline evaluation. Provider adapter tests inject fake SDK
clients and never require credentials or network access. Consequently, those
tests verify Chulk's request shaping and response normalization—not whether a
real key, endpoint, account, model entitlement, quota, or billing setup works.
Live validation remains the application owner's responsibility and may incur
provider charges.

See [configuration](configuration.md), [quickstart](quickstart.md), and
[SDK errors](sdk-errors.md).
