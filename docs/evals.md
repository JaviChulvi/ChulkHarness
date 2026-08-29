# Agent evaluations

`chulk.evals` is the public-stable framework for testing agents created with
the Chulk SDK. It executes the same `Agent.run_result(...)` or
`AsyncAgent.run_result(...)` boundary used by applications and records typed
results and public events for grading.

## Start a suite

Create a starter Python suite and JSONL dataset without overwriting files:

```bash
chulk eval init
chulk eval run evals/suite.py:suite
```

The starter is deterministic and needs no credentials. A suite supplies an
agent factory; the factory receives an isolated `EvalContext` for every case
and trial:

```python
from chulk import Agent, AgentConfig
from chulk.evals import (
    EvalDataset,
    EvalSuite,
    EvalTarget,
    ExactAnswerGrader,
    MetricThreshold,
    StatusGrader,
)


def create_agent(context):
    return Agent(
        config=AgentConfig(project_root=context.workspace),
        llm=context.llm,
        tools=[],
        skills=[],
    )


suite = EvalSuite(
    name="support",
    dataset=EvalDataset.from_jsonl("evals/cases.jsonl"),
    targets=(EvalTarget("support-agent", create_agent),),
    graders=(ExactAnswerGrader(), StatusGrader()),
    required_graders=("answer.exact", "run.status"),
    thresholds={"pass_rate": MetricThreshold(min=0.95)},
)
```

Call `EvalRunner().run(suite)` in Python, or use `AsyncEvalRunner` with an
`AsyncAgent` factory. One agent is reused across a case's turns, while every
trial receives a fresh agent and temporary workspace. Sync suites are serial
by default and honor explicit `concurrency`; async suites use bounded
concurrency. CLI overrides include `--trials`, `--concurrency`, `--timeout`,
`--mode`, `--provider`, `--model`, `--max-total-cost`, and `--fail-fast`.
Optional `sampling` values are exposed through `EvalContext` and retained in
the report provenance. A synchronous timeout requests cooperative cancellation,
waits briefly for the active turn to stop, and halts the suite before launching
another trial so unreported provider cost cannot compound. Custom synchronous
agent doubles should expose `cancel()` or make `close()` stop active work.

## Dataset schema

The canonical format is JSONL schema version `1`, with one case per line:

```json
{"schema_version":1,"id":"refund","turns":[{"input":"Refund order 123","scripted_responses":[{"type":"final_answer","content":"Refunded."}]}],"reference":{"answer":"Refunded.","status":"completed","tool_sequence":[]},"tags":["refund","smoke"]}
```

Cases may contain multiple turns, a case-level or final-turn reference, tags,
metadata, a named Python fixture, and a dataset-relative replay fixture. The
loader rejects duplicate ids, unknown fields and versions, malformed values,
and replay paths that leave the dataset directory. Optional YAML contains a
Python suite reference such as `suite: my_project.evals:suite`.

## Modes and graders

- `scripted` injects `ScriptedLLMClient` through `EvalContext.llm` and is the
  default offline regression mode.
- `replay` executes a bounded Chulk replay fixture without providers or real
  tools.
- `live` uses the target factory's provider. It requires `max_total_cost`;
  unknown pricing requires `EvalSafetyPolicy(allow_unknown_cost=True)`.

Built-in graders cover exact/contained/regex/JSON answers, status and errors,
tool calls, arguments, results and failures, ordered event milestones, skills, memories, plans,
latency, tokens, and cost. `CallableGrader` accepts application checks.
`LLMJudgeGrader` uses a separate tool-free `LLMClient`, requires strict JSON,
supports reference and pairwise grading, and records its redacted response,
judge model/provider, prompt version, usage, and cost. Because judging performs
metered model calls, suites containing a model judge also require
`max_total_cost`; unknown judge pricing requires the same explicit
`allow_unknown_cost` opt-in as live target execution.

`RubricJudgeGrader` is the diagnostic, opt-in alternative. Each
`RubricDimension` supplies concrete scoring levels and a positive weight. Mark
a binary dimension `veto=True` when failure must zero the total. The judge
scores the named items, while Chulk computes the weighted score and applies
vetoes in deterministic host code.

```python
from chulk.evals import (
    RubricDimension,
    RubricJudgeGrader,
)

grader = RubricJudgeGrader(
    judge_client,
    dimensions=(
        RubricDimension(
            "Factual correctness",
            {
                "excellent": "Every material claim is supported by the reference.",
                "fail": "A material claim contradicts or exceeds the reference.",
            },
            weight=2,
        ),
        RubricDimension(
            "Hallucination",
            {
                "pass": "Every claim is traceable to supplied evidence.",
                "fail": "Any unsupported factual claim appears.",
            },
            veto=True,
        ),
    ),
)
```

Its pairwise methods always judge twice, with answer order swapped. Only
order-consistent preferences count as a candidate or baseline win. A
disagreement becomes a tie with `needs_review=true` in grade details, exposing
position bias instead of silently averaging it away. Both calls are included
in judge token and cost accounting.

Only names in `required_graders` affect case pass/fail. Only declared
`MetricThreshold` values affect the suite quality gate. Other graders are
informational. Configuration and runtime failures remain operational errors.
Reports aggregate pass rate, pass@k, and pass_all_k (Pass^k: every trial for a
case succeeds), grader scores, latency percentiles, tokens, cost, and
exceptions. `pass_rate` remains the existing case-level quality metric;
`pass_all_k` makes its all-trials-success semantics explicit alongside the
any-success `pass_at_k` metric. Target, provider, model, and tag dimensions
also record case/trial counts, pass@k, pass_all_k, latency, token/cost totals,
and error rates.

## Safety and fixtures

Evaluation agents allow only tools declared with `READ` permission by default.
Network, memory mutation, write, shell, external, and destructive tools must be
named in `EvalSafetyPolicy.allowed_tool_names`. Prefer named fixtures that
return injected dependencies through `context.deps`; this keeps production
side effects out of tests.

Inputs, answers, tool data, judge output, and events pass through Chulk's
redaction owner before persistence or export. Credentials are never stored in
reports or written to JSON, JSONL, JUnit, or HTML artifacts.

## Reports, baselines, and CI

`SQLiteEvalStore` uses the shared forward-migrated database. The optional
Postgres package exports `PostgreSQLEvalStore` and async-callable adapters.
Both stores scope reports and baselines with `ExecutionScope`.

Stored runs are checkpointed after each completed trial and have an explicit
`running`, `interrupted`, or `completed` status. Resume a process interruption
without repeating completed trials by passing the same configured suite and
store:

```python
report = EvalRunner().run(suite, resume_from="RUN_ID")
```

Resume validates the suite, filtered dataset digest, targets, grader versions
and behavior-affecting configuration, thresholds, safety settings, and
execution configuration before running new work. Reports have `to_dict()` and
`EvalReport.from_dict(...)` round trips for portable recovery.

Baselines match only the same suite plus target fingerprint, case id, and
grader identity/version/configuration fingerprint. Comparisons report new and
removed cases or graders, per-grader score deltas, and `baseline_coverage`.
Matched cases also report paired binary outcomes: current wins, baseline wins,
ties, the discordant count, and the exact two-sided McNemar/binomial p-value.
These statistics use each case's serialized `passed` result, so they compare
the same case identities without changing existing report metrics. Declare a
`baseline_coverage` threshold when incomplete baseline coverage should fail a
suite; otherwise coverage changes remain informational.

```bash
chulk eval list
chulk eval show RUN_ID --json
chulk eval baseline set support RUN_ID
chulk eval compare RUN_ID
chulk eval export RUN_ID report.xml --format junit
```

Use `chulk eval run module:suite --resume RUN_ID` to continue a checkpointed
run. `eval list` filters by suite, lifecycle status, mode, target, provider,
model, tag, ISO start timestamps, limit, and offset. `eval compare
--min-coverage` can turn incomplete baseline coverage into exit code `1`.

Exports support JSON, case JSONL, JUnit XML, and standalone HTML. Exit codes
are `0` for passed, `1` for quality-gate failure, `2` for invalid
configuration/datasets, and `3` for operational failures. JUnit exports retain
case durations and represent threshold failures and operational or incomplete
runs as explicit failing test cases.

The repository's Ubuntu CI workflow includes this credential-free quality gate.
Its gating step is intentionally just the ordinary CLI contract:

```bash
chulk eval run examples/evaluation_suite/app.py:suite \
  --output evaluation-report.xml --format junit --json
```

The command's exit code gates the workflow, and producing the JUnit file checks
the same export contract that downstream CI consumers can use.

For a tool-driven coding example, `examples/software_engineer_eval/app.py`
evaluates the public `SoftwareEngineer()` preset against a disposable buggy
Python project. It requires the agent to read the implementation, apply a
minimal patch, run the regression test, and pass answer, status, error, and
tool-call graders without provider credentials.

Start the authenticated, read-only result viewer with:

```bash
chulk server start --eval-dashboard
```

Open `/evals` and authenticate with the local control token. The dashboard
filters and paginates runs, compares target/provider/model results, shows
baseline deltas and grader evidence, and reports cost and latency. Trace links
are available only when the persisted reference still resolves inside the
configured Chulk trace directory; deleted, external, and unrecorded traces are
shown as unavailable. The browser cannot execute suites or change baselines.
