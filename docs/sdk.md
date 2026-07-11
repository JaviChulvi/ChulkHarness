# SDK result contract

Chulk returns immutable public snapshots from `run_result`, `plan_result`,
`approve_result`, and `reject_result`. Common records are available from `chulk`;
the complete contract is available from `chulk.results`.

```python
from chulk import Agent, RunStatus
from chulk.results import Cost, RunResult, ToolCall, Usage

result: RunResult = agent.run_result("Summarize the project")
if result.status is RunStatus.COMPLETED:
    print(result.content)

for call in result.tool_calls:
    print(call.tool_name, call.success)
```

`RunStatus`, `PlanStatus`, and `PlanStepStatus` are finite string enums. Unknown
future values convert to their explicit `UNKNOWN` member instead of pretending
to be a current lifecycle state.

## Stable records

`RunResult` and `PlanResult` compose typed records rather than unstructured
containers:

- `Usage` and `Cost` preserve provider-neutral accounting.
- `ToolCall` and `Observation` describe tool activity and returned evidence.
- `ContextReport`, `ContextBudget`, and `ContextSection` describe prompt input.
- `Plan`, `PlanStep`, and `PlanStepEvidence` describe approval and execution.

Sequences are tuples and mappings are recursively read-only. Each result is a
detached snapshot: later runtime state changes cannot alter a result already
returned to the caller.

Unknown pricing is represented by `Cost(amount=None, pricing_known=False)`.
That is distinct from a known free operation whose amount is `Decimal("0")`.

## Serialization

Every public record has `to_dict()`. It returns fresh JSON-oriented lists,
dictionaries, strings, numbers, booleans, and null values. Decimal monetary
values and trace paths serialize as strings. The returned containers are mutable
for adapter convenience and never mutate the source snapshot.

```python
payload = result.to_dict()
payload["errors"].append("adapter annotation")  # result.errors is unchanged
```

`extension_metadata` is recursively read-only on the result and serialized as a
fresh plain dictionary. Stable fields remain separate from extension data so an
adapter can preserve unknown metadata without weakening the typed contract.
