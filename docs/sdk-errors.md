# SDK errors

Chulk exposes one stable exception family from both `chulk` and `chulk.api`:

```python
from chulk import ChulkError, ConfigurationError, ProviderError

try:
    answer = agent.run("Summarize the project")
except ConfigurationError as exc:
    print(exc.details.invalid_field)
except ProviderError as exc:
    if exc.details.retryable:
        retry_later()
except ChulkError as exc:
    logger.warning("Chulk failed", extra=exc.to_dict())
```

The documented categories are `ConfigurationError`, `ProviderError`,
`ToolExecutionError`, `PermissionDeniedError`, `SafetyError`, `TraceError`, and
`MemoryError`. All derive from `ChulkError` and expose `message`, `category`,
immutable `details`, and a redacted `to_dict()` representation.

`HostedServiceDisabledError` is a `ConfigurationError` raised when application
code attempts to use a service that a hosted capability profile explicitly
disabled. Its `details.invalid_field` identifies the service without exposing
host data.

Details are present only when known. They can include provider, model, tool,
invalid field, validation issues, retryability, conversation and turn ids, and
the trace path. Translated errors retain the internal exception in `__cause__`
for debugging. Messages and serialized details redact common credential forms;
do not log `__cause__` or raw trace artifacts without your own sensitive-data
policy.

Tool failures that the agent loop can safely report to the model remain
`ToolResult` observations. They are not raised as fatal SDK exceptions. A
`ToolExecutionError` means a tool-related failure escaped the public operation.

`asyncio.CancelledError` is never converted to a Chulk error, so standard task
cancellation continues to work. `TraceFormatError` remains compatible with
existing `ValueError` catches and is also a public `TraceError`; `MCPConfigError`
similarly remains a `ValueError` and is a `ConfigurationError`.
