# Chulk Roadmap

Last reconciled: 2026-07-26

This file is the ordered implementation roadmap for ChulkHarness. It contains
active user stories and intentionally deferred product directions. Completed
implementation history belongs in Git and `CHANGELOG.md`, not in a permanent
ledger of checked micro-tasks.

## How To Use This Roadmap

- Advance stories from top to bottom unless a user explicitly selects a
  different slice.
- Keep each requirement in one canonical story.
- Mark an acceptance item complete only after the corresponding code, test,
  documentation, or command output has been verified.
- Split a story when it cannot be reviewed safely as one change.
- Keep principles as prose. Checkboxes are reserved for concrete outcomes.

## Product Constraints

ChulkHarness should remain a lightweight, explicit, and inspectable Python
agent harness:

- State, prompts, model calls, tools, memory, skills, permissions, and traces
  must remain easy to follow from the code.
- Provider-specific behavior belongs in `src/chulk/llm/`.
- Runtime assembly belongs in `src/chulk/runtime.py` and is shared by CLI and SDK.
- The LLM boundary returns validated action dataclasses; orchestration does not
  parse provider text directly.
- Tools, skills, and memory remain separate concepts.
- Side effects are enforced by Python policy and are never trusted merely
  because a prompt requested them.
- SQLite is the runtime persistence engine; `MEMORY.md` is import/export only.
- Public APIs remain small, typed, and compatible unless a migration is
  documented.

## Verified Baseline

The following product layers are implemented and covered by the current test
suite:

- CLI and SDK entrypoints, public `Agent`/`AsyncAgent` facades, lifecycle, and
  stable error/result/event contracts.
- Structured provider actions, provider-native tools, streaming, usage/cost
  accounting, fallbacks, local providers, and hybrid MCP support.
- Tool schemas, permissions, output bounds, retries, timeouts, shell/file
  containment, and request-scoped dependencies.
- SQLite-backed conversations and long-term memory, memory review modes,
  retrieval, compaction, and session resume.
- Lazy bundled/project skills, plan approval and multi-step execution,
  reflection, traces, deterministic evals, documentation, examples, and clean
  wheel checks.

## Now: Maintenance Reliability

### US-M1: Protect Refactors With Behavioral Regression Gates

As a maintainer, I want one explicit behavioral and coverage gate so that
internal refactors cannot silently change sync/async semantics or skip optional
provider integrations.

Acceptance criteria:

- [x] Add normalized sync/async parity scenarios for direct answers, tool
  success/failure, planning and approval, retries/protocol failures, reflection,
  provider errors, and cancellation.
- [x] Compare durable turn state, model/tool counts, observations, and trace
  event ordering while excluding generated ids and timestamps.
- [x] Measure branch coverage over production code while keeping tests outside
  the importable package.
- [x] Enforce an initial 80% global coverage floor and ratchet it upward as
  coverage grows.
- [x] Run one Python CI lane with `.[dev,providers,mcp]`; keep the full suite
  credential-free and network-free.
- [x] Preserve supported Python 3.11, 3.12, and 3.13 coverage plus lint,
  typecheck, docs, examples, compile, and clean-wheel jobs.

### US-M2: Keep One Trustworthy Roadmap

As a contributor, I want the roadmap to show only current, uniquely owned work
so that the next implementation slice can be chosen without auditing hundreds
of contradictory boxes.

Acceptance criteria:

- [x] Replace the legacy architecture checklist, duplicate milestones, and
  completed-work ledger with ordered user stories.
- [x] Convert design principles and completion criteria to prose.
- [x] Classify legacy open work into active stories or deferred product themes.
- [x] Give every active story explicit acceptance criteria.
- [x] Remove the untracked, superseded `docs/sdk-improvement-audit.md` after
  verifying that its still-useful outcomes are represented here or in current
  documentation.

### US-M3: Make SQLite Upgrades And Concurrent Writes Reliable

As an SDK host, I want one versioned and backed-up SQLite subsystem so that
upgrades and concurrent session/memory activity cannot lose data or corrupt
ordering.

Acceptance criteria:

- [x] Centralize connection policy, transactions, migrations, and backups in a
  small shared storage boundary used by memory and session stores.
- [x] Adopt unversioned legacy databases, run ordered forward-only migrations
  atomically, and reject schema versions newer than the running package.
- [x] Configure foreign keys, explicit busy timeout, WAL, synchronous mode, and
  checkpoint policy deliberately on every applicable connection.
- [x] Create and validate a private online backup before upgrading an existing
  database; never raw-copy a live WAL database.
- [x] Repair existing duplicate message ordinals and enforce unique
  `(conversation_id, ordinal)` ordering.
- [x] Allocate message ordinals and other read-modify-write updates inside
  explicit write transactions.
- [x] Cover migration rollback, future versions, simultaneous initialization,
  lock waiting/timeouts, WAL readers, cross-store writers, exact concurrent
  message ordering, and backup consistency.
- [x] Keep database sidecars and backups private, ignored, and unavailable to
  model-facing file tools.

### US-M4: Make Orchestration Policy Single-Sourced

As a maintainer, I want sync and async execution to share one explicit state
transition policy so that new actions and plan behavior can be added once
without duplicating control flow.

Acceptance criteria:

- [x] Introduce a reducer that maps a small immutable control snapshot and an
  input signal to an explicit transition/effect.
- [x] Keep the reducer free of provider, tool, SQLite, callback, trace, and
  `Agent` dependencies.
- [x] Apply reducer-selected state mutation and observations through one
  transition boundary; keep model request accounting and transport traces in
  the focused model transport service.
- [x] Keep native sync and async transport drivers, but restrict their
  differences to blocking versus awaiting model, tool, text, and backoff work.
- [x] Remove duplicated action-policy branches and recursive loop re-entry.
- [x] Keep `Agent` focused on composition, lifecycle, plan approval/rejection,
  and resource ownership.
- [x] Preserve public signatures, serialized `TurnState`, trace schema/order,
  plan semantics, model accounting, cancellation propagation, and dependency
  cleanup.
- [x] Pass reducer branch tests, normalized sync/async parity tests, and all
  existing core, SDK, CLI, session, and trace tests.

### US-M5: Replace The Agent God Object With Focused Runtime Services

As a maintainer, I want the public Agent to assemble small runtime services so
that prompt/model, tool, plan, and turn-effect behavior can evolve without
growing one central class or coupling the loop to its private methods.

Acceptance criteria:

- [x] Extract prompt construction, context compaction, action requests, and
  reflection requests into an explicit model transport service.
- [x] Extract permission-aware sync/async tool execution and retry/backoff into
  an explicit tool executor.
- [x] Extract plan mutations/evidence and reducer-selected turn effects into
  focused plan and turn-effect services.
- [x] Assemble the services in `Agent`, preserve intentionally mutable runtime
  configuration, and keep `Agent` focused on public lifecycle, memory/skill
  selection, plan approval/rejection, and resource ownership.
- [x] Make the action loop depend only on a narrow runtime port and `TurnState`,
  with no concrete `Agent` import or private-`Agent` calls.
- [x] Feed protocol failures, plan preparation/results, tool results, and
  reflection results back through the reducer; consume and validate its outcome
  at the single turn-effect boundary.
- [x] Add architecture guards for reducer purity, runtime-port coupling, and
  service independence from `Agent`.
- [x] Preserve public signatures, serialized state, traces/order, planning,
  accounting, cancellation, request dependency cleanup, and sync/async parity.
- [x] Pass focused service/reducer tests and the complete repository test and
  static-validation suite.

## Now: Hosted Application Runtime

### US-H1: Establish The Hosted Runtime Boundary

As an application host, I want to supply every runtime service and one immutable
authority scope so that Chulk can run without local databases or directories and
can authorize tools before credentials or side effects.

Acceptance criteria:

- [x] Add complete sync and async hosted service containers with explicit
  resource ownership and scope-bound factories; keep omitted services as the
  backward-compatible local mode.
- [x] Add immutable execution scopes with canonical keys, non-escalating child
  scopes, cross-tenant isolation, persisted resume verification, tool-context
  propagation, and public event attribution.
- [x] Add versioned tool and schema identity, implementation and argument
  digests, policy metadata, host authorization/credential/effect/redaction
  hooks, secret-output withholding, and compatibility defaults.
- [x] Add credential-free sync/async in-memory embedding examples and contract
  tests that create no local runtime files.
- [ ] Await native async persistence, trace, audit, and usage services directly
  throughout the async orchestration path.
- [x] Execute independent `parallel_safe` read-only tool batches concurrently
  while preserving deterministic event and result ordering.

### US-H2: Publish Immutable Agent Definitions And Governed Skills

- [x] Add immutable agent-definition revisions and resolve every run to one
  exact published revision.
- [x] Add host-backed skill publication, evaluation, rollback, and revocation
  while reusing the existing skill lifecycle owner.
- [x] Add public builder, validator, capability-diff, and dry-run authoring
  helpers.

### US-H3: Make Hosted Execution Durable And Controllable

- [x] Add durable run and step state, leases, optimistic revisions, heartbeat,
  cancellation, steering, wait/resume, and unknown-effect reconciliation.
- [x] Replace blocking approvals with durable externally resolvable approval
  records.
- [x] Add retry policies and transaction boundaries across provider, tool,
  approval, and host-service failures.
- [x] Extend durable parent/child orchestration with scoped definitions,
  budgets, progress, and bounded fan-out.

### US-H4: Finish Gateway Resilience And Hosted Acceptance

- [x] Put gateway durability behind host-provided stores and add bounded
  backpressure, dead-letter, reconciliation, and definition/run routing.
- [x] Add versioned public event identities and durable host-provided trace and
  artifact correlation.
- [x] Complete the restart, duplicate-trigger, uncertain-effect, approval, and
  gateway contract suite without credentials or infrastructure dependencies.

## Next: Reliability And SDK Depth

These stories are ordered after the maintenance tranche above.

### Selected: Durable Reminders And Recurring Tasks

- [x] Add a channel-neutral SQLite job store with destination isolation.
- [x] Support one-off and fixed-interval jobs with timezone-aware timestamps.
- [x] Claim due jobs under short recoverable leases and advance recurring jobs
  without schedule drift.
- [x] Bind create/list/cancel tools to the authenticated Telegram chat.
- [x] Keep SDK agents unchanged and require adapters to opt in explicitly
  before scheduling tools or background execution are enabled.
- [x] Execute due prompts through the normal agent permissions and deliver
  results to the originating chat.
- [x] Add deterministic `/reminders` and `/cancel` commands.
- [x] Cover migration, isolation, claiming, recurrence, tools, execution, and
  configuration without wall-clock sleeps or network calls.

### Selected: Telegram Media

- [x] Normalize image, document, audio, and voice-note updates without changing
  the core agent message contract.
- [x] Resolve Telegram file references and enforce a configurable byte bound
  before media reaches a provider.
- [x] Enforce an explicit MIME/extension policy covering common iPhone and Mac
  images, audio, videos, documents, contacts, and calendar exports, with clear
  PDF-export guidance for Pages, Numbers, and Keynote packages.
- [x] Keep provider-specific multimodal processing in `src/chulk/llm/providers/`.
- [x] Preserve captions as user instructions and feed extracted text through
  the normal agent, memory, session, and trace path.
- [x] Keep media bytes out of project files, durable memory, and trace payloads.
- [x] Cover parsing, download limits, provider processing, routing, config, and
  sanitized failures without live network calls.

### US-N1: Productize Trace Replay And Artifacts

- [ ] Include all stable request, context, memory, skill, permission, tool,
  output, final-answer, error, usage, and cost evidence in inspect/export views.
- [ ] Read full truncated-output artifacts safely from trace metadata.
- [ ] Add trace-based and golden action-loop regression replay.
- [ ] Keep raw traces explicitly sensitive and unstable where appropriate.

### US-N2: Add A Sandbox And Richer Permission Policy

- [ ] Distinguish fatal safety-policy violations from recoverable tool failures
  and stop the turn immediately when a fatal violation occurs.
- [ ] Add custom permission profiles, workspace-root allowlists, command prefix
  rules, trusted read-only commands, and durable permission audit records.
- [ ] Define a sandbox backend interface before expanding shell-heavy tools.
- [ ] Provide workspace/temp-copy backends first; treat Docker and Bubblewrap as
  optional platform backends.
- [ ] Add external-input prompt-injection warnings and network domain policy
  before browser/web automation.

### US-N3: Improve Provider Operations

- [ ] Expose stable provider/model capability diagnostics.
- [ ] Add health checks and actionable invalid-model, missing-key, unavailable
  endpoint, unsupported-schema, and rate-limit errors.
- [ ] Add model switching/profiles only after capability and lifecycle semantics
  are stable.

### US-N4: Add Memory Isolation And Retention

- [ ] Add user/workspace namespaces and per-thread memory controls.
- [ ] Add TTL and item-count retention policies without weakening provenance or
  review modes.
- [ ] Add a complete memory/session backup/export command on top of the shared
  SQLite backup primitive.
- [ ] Add contradiction, freshness, and consolidation behavior only after
  namespaces and retention are deterministic.

### US-N5: Strengthen Plans And Durable Goals

- [ ] Add public plan-step risk, expected-tool, budget, and richer acceptance
  fields.
- [ ] Add selected-step approval/skip APIs and blocked-step recovery.
- [ ] Add resumable multi-turn goals with explicit status, time/tool/token
  budgets, steering, cancellation, and persistence.

### US-N6: Finish Async And Testing Ergonomics

- [ ] Add public tool-testing helpers for schema assertions and direct typed
  invocation.
- [ ] Add bounded concurrent read-only tools only after event ownership and
  cancellation semantics are explicit.
- [ ] Add provider and MCP async cleanup hooks.
- [ ] Expand deterministic eval scenarios for memory, skills, plans, safety,
  and retries; keep live-model evals optional.

## Later: Product Directions

These are deliberately uncommitted themes. Promote one into a user story with
acceptance criteria before implementation.

- Safe web research, bounded HTTP fetch, browser verification, screenshots, and
  multimodal inputs.
- First-class Git/review/test-runner/package-manager tools.
- LLM/embedding/hybrid skill routing, skill lifecycle commands, metadata lint,
  and reviewed skill proposals.
- Subagents, parent/child traces, bounded parallel exploration, background
  tasks, automations, and worktree isolation.
- Chulk as an MCP server, JSON-RPC/app servers, WebSocket/REST interfaces, trace
  and conversation UIs, and IDE integration.
- Plugin manifests, trust review, extension hooks, installation, and migration
  from other harness formats.
- Optional notebook, document, spreadsheet, image, code-interpreter, and
  scheduled-task tooling.

## Release And Success Criteria

The next public release should happen only when the active `Now` stories are
complete and the repository passes:

```bash
python -m pytest
python -m compileall src/chulk
python -m ruff check .
python -m mypy src/chulk examples typing_tests
python scripts/check_docs.py
```

Chulk is successful when a host can understand and constrain every model call
and side effect; resume durable work safely; inspect state, memory, plans,
permissions, costs, and traces; extend providers/tools/skills without modifying
the orchestration policy in multiple places; and upgrade local runtime state
without data loss.
