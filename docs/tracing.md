# Tracing and diagnostics

Each agent writes internal JSONL trace events under `<runtime_dir>/traces/`.
`RunResult.trace_path` identifies the active file. Traces help reconstruct model
requests, selected context, tool calls, observations, permission decisions, and
failures.

Trace files are raw sensitive runtime output. Redaction covers common secret
forms but cannot prove that arbitrary personal data, proprietary text, tool
output, or credentials are absent. Restrict access, set retention, and scrub
before sharing. Truncated tool-output artifacts beside a trace require the same
handling.

The internal trace schema is **trace-only**, may change without a compatibility
release, and must not drive application behavior. Use the versioned public
[event stream](events.md) and immutable [SDK results](sdk.md) for integrations.

The [repository review bot](../examples/repo_review_bot/README.md) includes a
normalized sample walkthrough. Its sequence shows turn start, model request,
tool start/completion, final answer, and turn finish. Real identifiers,
timestamps, prompts, and paths were replaced, so it is suitable for docs but
not a byte-for-byte runtime fixture.
