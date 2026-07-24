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

The internal event payload catalog remains trace-only and may evolve without a
compatibility release. The envelope version and legacy reader support make
diagnostics migratable; they do not make undocumented payload fields a stable
application API. Use the versioned public [event stream](events.md) and
immutable [SDK results](sdk.md) for integrations.

## Offline commands

All trace commands operate on local files:

```bash
chulk trace inspect .chulk/traces/<conversation-id>.jsonl
chulk trace replay .chulk/traces/<conversation-id>.jsonl
chulk trace replay .chulk/traces/<conversation-id>.jsonl --json
chulk trace export .chulk/traces/<conversation-id>.jsonl --format html
```

`inspect` summarizes the envelope versions and event counts. `replay`
deterministically reconstructs recorded sessions, turns, model-request counts,
tool outcomes, failures, and answers. It never calls a model, invokes a tool,
or accesses the network, and it does not modify the source trace. This is
diagnostic reconstruction, not executable regression replay. `export` writes
an escaped, self-contained HTML report.

Command output can repeat sensitive content from the trace and must receive the
same handling as the source file.

The [repository review bot](../examples/repo_review_bot/README.md) includes a
normalized sample walkthrough. Its sequence shows turn start, model request,
tool start/completion, final answer, and turn finish. Real identifiers,
timestamps, prompts, and paths were replaced, so it is suitable for docs but
not a byte-for-byte runtime fixture.
