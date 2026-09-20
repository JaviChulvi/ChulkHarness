# Source-grounded assistant starter

This small embedded application answers from an exact-ID source lookup and
prints the evidence it retrieved. Its tiny Northstar Labs handbook is fictional
and illustrative; it is not customer data.

Run it from the repository root:

```bash
.venv/bin/python examples/grounded_assistant/app.py
```

The default `ScriptedLLMClient` path is deterministic and credential-free. It
exercises Chulk's real action validation, tool execution, result, and trace
path, but it is **not live answer-quality proof**. To opt into the live provider
configured through `examples/common.py`, set `CHULK_EXAMPLE_MODE=live` and the
provider's documented model and credentials.

## Host boundary

After authenticating the application request, the host must load only sources
authorized for that user and workspace and pass them as `ScopedSources`. The
model sees one read-only `get_source(source_id)` tool; it cannot list, search,
or retrieve outside that injected set. Replace `FICTIONAL_SOURCES` with that
already-scoped application query, not with provider credentials or an
unrestricted data client.

The host accepts only `[source:ID]` citations whose IDs the tool actually
retrieved during the run. This proves the cited ID was in the scoped evidence
set; citation presence alone does **not** prove the answer faithfully represents
the source. Production hosts should add domain-specific answer evaluation where
the risk warrants it.

The example disables built-in file, shell, memory, network, external-service,
and utility capabilities and uses the read-only permission profile. If a later
workflow needs writes, keep them separate and add explicit approval; see the
[plan approval example](../08_plan_approval.py).

The printed trace path points to sensitive local runtime diagnostics that may
contain prompts, retrieved source text, tool arguments, and answers. Keep the
runtime directory private; do not commit or share raw traces.
