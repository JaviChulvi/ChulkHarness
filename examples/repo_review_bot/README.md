# Deterministic repository review bot

This complete embedded application reviews a fixed repository fixture without
credentials or network access. It uses only public SDK imports, injects a
host-owned fixture path through `ToolContext`, exposes one read-only tool, and
opts out of file, memory, shell, network, and external-service capabilities.

Run it from the repository root:

```bash
python examples/repo_review_bot/app.py
```

Expected finding:

> High — The setup guide tells users to commit a production API token.

The event line demonstrates the ordered public event stream. The final
`trace_path` points to the internal JSONL trace created in a temporary runtime
directory. Internal traces are sensitive diagnostic artifacts and their schema
is not a public compatibility contract.

The scrubbed [sample trace](sample-trace.jsonl) keeps the main walkthrough:
run start, model request, tool start/completion, final model response, and run
completion. Identifiers, timestamps, prompts, and paths are normalized; no
secret or machine-specific path is retained.

Regenerate it from an actual deterministic run:

```bash
python examples/repo_review_bot/app.py \
  --write-sample examples/repo_review_bot/sample-trace.jsonl
```

See [tools](../../docs/tools.md), [permissions](../../docs/permissions.md),
[events](../../docs/events.md), [tracing](../../docs/tracing.md), and
[safety](../../docs/safety.md) for the production boundaries behind the demo.
