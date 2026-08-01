# Public API and release policy

Chulk documents four stability labels:

| Label | Contract |
|---|---|
| **public-stable** | Supported imports and documented behavior; breaking changes require a compatibility release and migration note. |
| **public-provisional** | Importable compatibility surface that may be replaced after a documented deprecation cycle. |
| **internal** | Implementation detail with no compatibility promise. |
| **trace-only** | Sensitive diagnostic schema that may change independently of the public SDK. |

The supported top-level names are governed by `chulk.__all__`. They are
**public-stable** unless explicitly classified as provisional below. Stable
advanced contracts are exported by `chulk.api`, `chulk.capabilities`,
`chulk.authoring`, `chulk.errors`, `chulk.events`, `chulk.hosting`,
`chulk.results`, `chulk.runs`, `chulk.approvals`, `chulk.skills`,
`chulk.tools`, `chulk.testing`, and the optional `chulk.postgres` module.
`ScriptedLLMClient` is stable only from `chulk.testing`.
The `assert_*_contract` functions in `chulk.testing` are public-stable
compatibility gates for third-party hosted services and gateway stores.
The hosted sync and async service protocols, including `AsyncSkillService`,
`AsyncSkillLifecycleStore`, `AsyncLearningProposalService`,
`AsyncLearningReviewer`, and `AsyncPluginService`, their binding containers,
and resource-ownership semantics are public-stable.

The top-level package resolves stable exports lazily. Plain `import chulk`
loads only package metadata; requesting a named export loads its owning public
surface, while `from chulk import *` intentionally resolves the complete
compatibility API declared by `chulk.__all__`.

`chulk.hosting.reference` is a documented example and contract-test fixture,
not a durable production service implementation.

The SQLite and async SQLite run/approval stores are **public-provisional**
reference adapters. The store protocols, immutable records, transition enums,
parent/child policy and record contracts, and durable execution/approval
coordinators are public-stable. New parent/child store methods are additive;
hosts implementing `RunStore` or `AsyncRunStore` must provide them before
advertising compatibility with this release. Other concrete SQLite
implementations remain internal.

Gateway and scheduling protocols, run-target records, and hosted
definition/run submitters are public-stable. The SQLite gateway ledger remains
a public-provisional reference adapter.

The PostgreSQL factories, forward-only migration entry point, atomic
ownership-transfer functions, and sync/async stores exported by
`chulk.postgres` are **public-provisional**. Importing that module requires the
`postgres` extra; default `chulk` and `chulk.api` imports remain free of
relational-database dependencies.

`AgentHandle`, `AsyncAgentHandle`, `ChatAgent`, `AsyncChatAgent`, and their
lowercase compatibility factories are **public-provisional** for the current
deprecation cycle. Prefer `Agent` and `AsyncAgent` in new code.

Modules below `chulk._sdk`, plus `chulk.core`, `chulk.runtime`, provider
transports, registries, SQLite implementations, and CLI implementation modules,
are **internal** even when Python can import them. JSONL trace event types,
payloads, artifacts, and filesystem naming are **trace-only**. Do not describe
an internal or trace-only name as stable in application documentation.

Public event schema changes follow `EVENT_SCHEMA_VERSION`; unknown future enum
values map to explicit `UNKNOWN` states where documented. Additive fields use
extension mappings. See [SDK](sdk.md), [events](events.md), and
[tracing](tracing.md).

Portable definition JSON follows `AGENT_DEFINITION_SCHEMA_VERSION`. Readers
fail closed on unknown fields and future schema versions. Published versions
are immutable; behavior changes require a new artifact version, and schema
changes require a documented compatibility reader or migration. See
[portable authoring](authoring.md#schema-and-migration-policy).

The OS-independent package claim is backed by required CI on Ubuntu for Python
3.11, 3.12, and 3.13 plus Windows on Python 3.12. The Windows lane runs the
complete credential-free suite, compile and lint checks, and a clean-wheel
install/import smoke test. Provider network calls remain outside required CI.
