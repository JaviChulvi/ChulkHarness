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
python -m pip install -e ".[openai]"
python -m pip install -e ".[anthropic]"
python -m pip install -e ".[gemini]"
python -m pip install -e ".[providers]"
```

Run these commands from a ChulkHarness source checkout; the project is not yet
published to PyPI.

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
export CHULK_LOCAL_CONTEXT_WINDOW_TOKENS=131072
```

`CHULK_LOCAL_CONTEXT_WINDOW_TOKENS` is the effective context loaded by the
local server, not the model architecture maximum. Chulk defaults it to a
conservative `131072` tokens. Set it to the loaded instance's context; for LM
Studio, use `loaded_instances[].config.context_length`, not
`max_context_length`. Runtime budgeting uses the smaller of this setting and a
catalogued architecture limit. For an uncatalogued local artifact, the setting
is the effective limit. Chulk applies the cap before fallback limits are
aggregated. SDK callers can use
`AgentConfig.local(context_window_tokens=...)` or
`LocalProvider(..., context_window_tokens=...)`.

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

## Model metadata catalog

Bundled pricing and token limits live in the immutable Python catalog at
`chulk/llm/model_catalog.py`. Each canonical model is represented by one
`ModelSpec`; pricing, limits, aliases, lifecycle state, and provenance therefore
resolve to the same record instead of being maintained in parallel tables.
Exact model and explicit alias lookups use a prebuilt dictionary. Resolution
returns the existing frozen record and does not allocate model objects per
request.

The bundled records cover currently callable direct-provider IDs that can use
Chulk's text/action transport: OpenAI Responses models with function calling,
Anthropic Claude models, Gemini GenerateContent models, DeepSeek API models,
and a small set of explicitly identified local models. The catalog also keeps
deprecated IDs until their published shutdown date so existing configurations
retain accurate migration metadata. Retired IDs and endpoint-specific audio,
realtime, image, embedding, moderation, video, search, and managed-agent models
are intentionally excluded when they cannot satisfy Chulk's action contract.

OpenRouter, Bedrock, generic compatible, and arbitrary local model namespaces
are deployment-scoped rather than one stable global inventory. Do not copy
direct-provider prices or limits into those namespaces. Discover their current
IDs from the configured endpoint instead:

- OpenRouter publishes `GET /api/v1/models`, including context, output, and
  endpoint pricing metadata.
- Bedrock exposes model discovery and documents which models support its
  OpenAI-compatible APIs; availability and identifiers vary by region.
- LM Studio and other OpenAI-compatible local servers normally expose
  `GET /v1/models`; limits and prices remain properties of the selected local
  artifact and serving configuration.

See the [OpenRouter models API](https://openrouter.ai/docs/api/api-reference/models/get-models),
[Bedrock model catalog](https://docs.aws.amazon.com/bedrock/latest/userguide/models.html),
[Bedrock OpenAI compatibility table](https://docs.aws.amazon.com/bedrock/latest/userguide/models-api-compatibility.html),
and [LM Studio model-list API](https://lmstudio.ai/docs/developer/openai-compat/models).

To add a released model, append its record to the matching provider tuple;
`MODEL_SPECS` combines those tuples into the immutable catalog. Provider helpers
keep repeated URLs and internal reserve policies concise, while the underlying
record has this shape:

```python
ModelSpec(
    provider="example",
    model="example-model",
    aliases=("example-model-2026-07-01",),
    limits=ModelLimits(
        context_window_tokens=128_000,
        default_response_reserve_tokens=8_192,
        max_input_tokens=128_000,   # Optional when not published independently.
        max_output_tokens=16_384,   # Optional when the provider does not specify it.
        source_urls=("https://provider.example/models/example-model",),
        last_checked=date(2026, 7, 12),
    ),
    pricing=TokenPricing(
        input_per_million=Decimal("0.50"),
        cached_input_per_million=Decimal("0.10"),
        cache_write_input_per_million=Decimal("0.625"),  # Optional.
        output_per_million=Decimal("1.50"),
        source_urls=("https://provider.example/pricing",),
        last_checked=date(2026, 7, 13),
    ),
)
```

Use `Decimal` strings for every price. Limits and pricing each own an independent
`source_urls` tuple and `last_checked` date because providers often publish and
update them separately. Both provenance fields are required on every present
section. For an unpriced local model, only `ModelLimits` is configured. The
limits provenance establishes the published hard limits; it does not establish
Chulk's `default_response_reserve_tokens`, which remains an internal prompt
budget policy.

Provider helper defaults represent a documented batch verification. When one
record is checked or changed independently, pass that helper's
`limits_last_checked`, `pricing_last_checked`, or
`lifecycle_last_checked` override instead of changing a date shared by the
rest of the batch.

For providers that apply one published long-context threshold to the full
request, `TokenPricing.long_context` accepts a `LongContextPricing` record.
The estimator selects it with one comparison against normalized input tokens,
so lookup and calculation remain constant-time. OpenAI Responses usage also
normalizes `input_tokens_details.cache_write_tokens` into the distinct
`cache_write_input_tokens` bucket and prices it with
`cache_write_input_per_million`. Leave `pricing=None` when the provider requires
dimensions Chulk cannot account for exactly, such as multiple cache-write
durations, a scheduled price transition, or a non-token fee. Unknown pricing is
preferable to a plausible but incorrect cost.

Cost estimates cover token charges represented by provider usage metadata.
They do not claim to be a complete account bill and exclude separately managed
charges such as explicit Gemini cache storage duration. Chulk's Gemini provider
does not create explicit caches itself.

Add aliases only for explicitly published identifiers that have the same
pricing, limits, and lifecycle. Do not use an open-ended model-name prefix: add
each new snapshot when it is released so an unknown sibling cannot inherit the
wrong prices. A moving `latest` alias records the provider's published target at
its `last_checked` date and must be reverified when that provider changes the
family. Use separate specs when an old identifier is deprecated or otherwise
differs. Non-stable entries also record `lifecycle_source_urls` and
`lifecycle_last_checked` alongside `status`, `replacement_model`, and
`retired_on`. Catalog construction rejects malformed records, duplicate
identities, invalid replacements, and missing provenance.

`max_input_tokens` and `max_output_tokens` capture independently published hard
limits when available. A known input maximum bounds Chulk's prompt budget. The
output maximum remains inspectable metadata; Chulk does not automatically clamp
a provider request to `max_output_tokens`. The existing
`default_response_reserve_tokens` also continues to control prompt budgeting.
Leave an optional limit as `None` when the provider or local serving
configuration does not publish a reliable value.

## SDK construction

The matching builders are `AgentConfig.openai(...)`, `.deepseek(...)`,
`.local(...)`, `.openai_compatible(...)`, `.openrouter(...)`,
`.anthropic(...)`, `.bedrock(...)`, and `.gemini(...)`. Applications can also
inject an application-owned `LLMClient` through `Agent(llm=client)`. The shared
runtime asks providers for validated actions through `complete_action(...)` and
normalizes provider-specific tool calls before orchestration.

## Action request shaping

Each action request uses one tool-schema channel. If every provider in a
fallback chain supports native tool calling, the system prompt contains only a
compact native-transport status and the schemas travel in the provider's tool
field. Otherwise, Chulk puts the full catalog in the JSON-action prompt and
sends no native tool declarations. A provider that rejects an attempted native
request receives a structurally replaced JSON protocol and catalog on the
fallback attempt; the native and JSON protocols are never appended together.

The provider wire mappings are:

| Transport family | System and conversation data | Native declarations |
| --- | --- | --- |
| OpenAI Responses | `instructions` plus `input` | top-level `tools` and `tool_choice`; hosted MCP entries also use `tools` |
| DeepSeek, local, OpenAI-compatible, OpenRouter, Bedrock | Chat Completions `messages`; local endpoints fold system instructions into the latest user message | top-level `tools` and `tool_choice` |
| Anthropic Messages | top-level `system` plus `messages` | top-level `tools` using `input_schema`, with parallel tool use disabled |
| Gemini GenerateContent | `config.system_instruction` plus `contents` | `config.tools` using `parameters_json_schema`, with automatic execution disabled |

Plan proposal and plan-step-update declarations are included only while that
action is legal for the current turn. Ordinary turns receive neither; approved
active steps receive only the step-update declaration; completed plans receive
neither. When the effective declaration set is empty, provider requests omit
the tool and tool-choice fields instead of sending an empty array.
Low-level `complete_action(...)` callers that intentionally drive planning can
pass `PlanningToolAvailability`; its default exposes no planning pseudo-tools.

Provider-native calls are normalized into Chulk action dataclasses before the
agent loop. Tool results currently return on the next request as bounded textual
observations; Chulk does not persist each provider's native tool-call transcript
or call id as continuation state.

`AsyncAgent` uses each SDK's native async transport: OpenAI's async Responses
or Chat Completions clients, Anthropic's async Messages client, and Gemini's
async GenerateContent client. Cancellation propagates through native requests
and does not advance a fallback chain. A custom client that implements only the
synchronous interface remains supported through a thread-backed compatibility
path, which cannot force-stop an already running synchronous call.

`chulk.testing.ScriptedLLMClient` is the deterministic choice for unit tests,
examples, and offline evaluation. Provider adapter tests inject fake SDK
clients and never require credentials or network access. Consequently, those
tests verify Chulk's request shaping and response normalization—not whether a
real key, endpoint, account, model entitlement, quota, or billing setup works.
Live validation remains the application owner's responsibility and may incur
provider charges.

See [configuration](configuration.md), [quickstart](quickstart.md), and
[SDK errors](sdk-errors.md).
