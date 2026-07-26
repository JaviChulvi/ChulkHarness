# Tracing and diagnostics

Each agent writes internal JSONL trace events under `<runtime_dir>/traces/`.
`RunResult.trace_path` identifies the active file. Traces record model requests,
selected context, tool calls, observations, permission decisions, failures, and
the final answer.

Trace files are raw sensitive runtime output. Redaction covers common secret
forms but cannot prove that arbitrary personal data, proprietary text, tool
output, or credentials are absent. Restrict access, set retention, and scrub
before sharing. Truncated tool-output artifacts beside a trace require the same
handling.

Trace directories and artifact directories are created owner-only (`0700`) on
POSIX; JSONL traces, full-output artifacts, and HTML exports are owner-only
(`0600`). Existing modes are repaired when a sensitive file is opened for
writing. Symlink and non-regular trace, artifact, and export targets are
rejected instead of followed. Deferred loggers preserve lazy behavior and do
not create a trace file until their activation event.

## Opaque artifact reads

When a tool result exceeds the model-facing output limit, its full text is
stored under an opaque `art_<id>` reference. Observations and result metadata
contain the id, byte/character counts, and SHA-256 digest, never a filesystem
path. General file tools continue to reject trace and artifact paths.

Hosts can retrieve evidence through the conversation-bound API:

```python
view = agent.read_artifact(
    artifact_id,
    mode="head_tail",  # also: head, tail, slice
    max_bytes=8192,
)
```

Reads validate the private manifest owner, fixed id-derived filename, regular
file type, recorded size, maximum integrity-read size, and SHA-256 digest
before returning content. A response is always bounded to at most 65,536 source
bytes and reports the returned ranges and whether content was omitted.
`slice` additionally accepts a non-negative byte offset.

Applications may explicitly expose `Tools.read_trace_artifact` to the model.
It is not in the default tool set: adding the tool is the host capability
decision, and normal read permission policy still applies. The tool is bound
to the current conversation, so an id from another trace, a forged id, a path,
or a tampered/missing artifact fails closed.

Action-request trace payloads identify `action_transport`, the effective native
tool names, and a bounded provider-neutral declaration snapshot. The context
report separates message tokens from out-of-band native declaration overhead.
For native requests it also records the larger JSON-fallback message estimate,
which is the value used when reserving and trimming context. This makes the
actual request and its safe fallback budget inspectable without duplicating
schemas in the system prompt; provider SDK wrappers may still use different
field names around the same neutral declarations.

## Envelope schema

New events use schema version `1`:

```json
{
  "schema_version": 1,
  "conversation_id": "conversation-id",
  "turn_id": "turn-id",
  "timestamp": "2026-01-01T00:00:00+00:00",
  "type": "turn_started",
  "payload": {}
}
```

`schema_version`, `conversation_id`, `timestamp`, `type`, and `payload` are
always present. `turn_id` is present for events associated with a turn and is
omitted for session-level events. `session_started` and `session_finished`
bound an explicitly closed runtime session. Finished session and turn payloads
include elapsed milliseconds measured with a monotonic clock.

`Trace.from_jsonl(path)` normalizes both version `1` and legacy unversioned
events (reported as schema version `0`). A trace created before an upgrade can
therefore contain both versions. Unknown future versions fail closed with
`TraceFormatError` instead of being interpreted with the wrong contract.
The reader consumes JSONL incrementally. By default it rejects traces larger
than 64 MiB, traces with more than 100,000 non-empty events, and individual
lines larger than 4 MiB. Callers can set lower or higher positive
`max_bytes`/`max_events` values. The CLI exposes the same controls; trusted
operators can deliberately bypass all three parser limits with `--unbounded`.
Limits fail with an error and never return a partial trace.

The internal event payload catalog remains trace-only and may evolve without a
compatibility release. The envelope version and legacy reader support make
diagnostics migratable; they do not make undocumented payload fields a stable
application API. Use the versioned public [event stream](events.md) and
immutable [SDK results](sdk.md) for integrations.

`HostedRuntime` can use a host-provided trace and artifact service instead of
JSONL files. Hosted trace payloads include the execution scope and versioned
tool-policy evidence. Credential values are never passed to the trace service.
Resource ownership follows the hosted service binding; Chulk does not close a
host-owned trace sink.

Hosted public events, durable audit, diagnostic traces, and artifacts have
independent owners. Deleting or expiring a `TraceSink` record never removes
durable run events, approval decisions, effect intent, or reconciliation
history. `AuditSink` rejects raw prompt, credential, secret, token, and
raw-argument fields before persistence. `EventSink` receives the redacted
schema-v3 application contract, while `ArtifactStore` retains large content
under opaque IDs. See [hosted runtime](hosting.md#events-audit-traces-and-artifacts).

## Offline commands

All trace commands operate on local files:

```bash
chulk trace inspect .chulk/traces/<conversation-id>.jsonl
chulk trace replay .chulk/traces/<conversation-id>.jsonl
chulk trace replay .chulk/traces/<conversation-id>.jsonl --json
chulk trace export .chulk/traces/<conversation-id>.jsonl --format html
chulk trace inspect .chulk/traces/<conversation-id>.jsonl --max-events 50000
chulk trace inspect .chulk/traces/<conversation-id>.jsonl --unbounded
```

`inspect` summarizes the envelope versions, source bytes, event counts, parser
limit mode, and an artifact inventory. Inventory entries expose opaque ids,
safe metadata, and integrity status without artifact content or filesystem
paths. Counts and total recorded bytes provide retention visibility; Chulk
does not delete artifacts automatically. Missing, unrecorded, unsafe,
oversized, size-mismatched, and hash-mismatched artifacts are reported.
`replay`
deterministically reconstructs recorded sessions, turns, model-request counts,
tool outcomes, failures, and answers. It never calls a model, invokes a tool,
or accesses the network, and it does not modify the source trace. This is
diagnostic reconstruction, not executable regression replay. `export` writes
an escaped, self-contained HTML report with the same inventory but never
embeds artifact content.

Command output can repeat sensitive content from the trace and must receive the
same handling as the source file.

The [repository review bot](../examples/repo_review_bot/README.md) includes a
normalized sample walkthrough. Its sequence shows turn start, model request,
tool start/completion, final answer, and turn finish. Real identifiers,
timestamps, prompts, and paths were replaced, so it is suitable for docs but
not a byte-for-byte runtime fixture.
