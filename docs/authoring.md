# Portable agent authoring

Chulk separates portable behavior from deployment configuration:

| Contract | Owns | Must not own |
|---|---|---|
| `AgentDefinition` | Prompt, model-policy, tool, skill, trigger, workflow, budget, approval-policy, locale, and provenance references | Paths, credentials, backend objects, or runtime state |
| `RuntimeProfile` | Local paths, concrete backends, process configuration, and credential references | A canonical agent revision |
| `ExecutionScope` | Tenant, workspace, actor, agent revision, run identity, and grants for one hosted operation | Behavior or credentials |

`RuntimeProfile` is the clearer public name for the existing `AgentProfile`
deployment contract; `AgentProfile` remains available for compatibility.

## Publication flow

A definition and its procedural skill become executable only after validation,
deterministic evaluation, and explicit host review:

```text
structured request + selected trusted tools
                  |
                  v
           AgentCompiler.compile
                  |
                  v
       draft definition + draft skill
          + reports + review preview
                  |
          host review and publish
                  |
                  v
      exact version + digest resolution
                  |
       local or hosted runtime assembly
```

Compilation never writes to a publication store. The host submits and publishes
the skill through `SkillPublicationManager`, then saves and publishes the
definition through an `AgentDefinitionStore`. A publication report is accepted
only when its `artifact_digest` matches the exact artifact under review.

```python
from chulk import (
    AgentCompiler,
    BudgetDefinition,
    CompilerRequest,
    ExecutionScope,
    TriggerDefinition,
)

request = CompilerRequest(
    agent_id="support-agent",
    version="1.0.0",
    goal="Answer from the approved catalog.",
    prompt=published_prompt.reference,
    model_profile=published_model_policy.reference,
    approval_policy=published_approval_policy.reference,
    selected_tools=("catalog_lookup",),
    triggers=(TriggerDefinition(kind="api.request"),),
    expected_outcomes=("Return a catalog-backed answer.",),
    budget=BudgetDefinition(max_model_requests=4, max_tool_calls=4),
)
package = compiler.compile(request, caller_scope=scope)
assert package.publishable
print(package.preview.to_dict())
```

The compiler can see only the tool names selected by the caller. It resolves
those names through a host-owned `ToolCatalog`, verifies the caller's grants,
and rejects generated workflow nodes that invent a tool, alter its identity,
reduce its effect, or weaken its approval policy. The output is declarative:
generated code, package installation, MCP configuration, raw credentials, and
local paths are rejected.

For the same `CompilerRequest` object, catalog state, and scripted workflow
generator, the definition, skill, reports, and review preview are deterministic.
`CompilerRequest.created_at` is part of the structured input so provenance does
not depend on a hidden clock during compilation.

## Definition lifecycle

`AgentDefinition` is a frozen, JSON-serializable record. All dependencies carry
an exact semantic version and SHA-256 digest. Tool references additionally pin
input and output schemas, implementation identity, and policy identity.

Definition states are:

- `draft`: stored for review and forbidden from execution.
- `published`: available for new runs.
- `deprecated`: retained and still resolvable for compatibility.
- `revoked`: retained for history but blocked from new runs.

Saving different behavior under an existing `(agent_id, version)` fails. Create
a new semantic version for any behavior-affecting change. Publication stores
are scoped by tenant and workspace; another scope cannot read or publish the
record.

`AgentDefinitionRuntime` resolves every dependency before construction and
provides:

- `create_local(...)` for ordinary SDK services.
- `create_hosted(...)` and awaited `create_async_hosted(...)` for injected
  services.
- `dry_run(...)` for side-effect-free dependency resolution.
- `capability_diff(...)` for reviewable tool, skill, and effect changes.

Every constructed runtime writes the definition's agent ID, version, and digest
to runtime metadata, conversation metadata, and each run result's
`extension_metadata["agent_definition"]`.

Revocation prevents a new resolution. An already-resolved in-flight run keeps
its immutable snapshot; historical records remain readable through `get(...)`
for replay and audit.

## Governed portable skills

The portable publication layer extends the existing governed-skill owner. It
does not replace local skill proposals, revision history, lock files, or
filesystem rollback.

`PortableSkill` contains bounded procedural text plus exact tool and skill
includes. Its manifest text cannot grant authority. The host validates catalog
references and an acyclic, fully pinned include graph before deterministic
evaluation.

Publication states are `draft`, `validating`, `evaluating`,
`awaiting_review`, `published`, `deprecated`, and `revoked`. Draft and proposed
content cannot execute. Publication creates an immutable version and digest.
Rollback changes only the active pointer; it does not mutate either revision.
Each publication or rollback pointer change adds a `SkillActivationRecord`
containing the operator, reason, previous reference, and timestamp.

Revocation blocks new resolutions, preserves the immutable record, and can
record the affected definition revisions supplied by the host. In-flight runs
that already resolved the skill keep their snapshot.

Sync and async hosts implement `SkillPublicationStore` or
`AsyncSkillPublicationStore`. The in-memory adapters are credential-free
reference implementations for examples and contract tests, not production
durability services.

## Schema and migration policy

`AGENT_DEFINITION_SCHEMA_VERSION` is currently `1`. Canonical JSON uses sorted
object keys, UTF-8 text, compact separators, finite JSON numbers, and normalized
SHA-256 strings. The resulting digest is stable across supported Python
versions and operating systems.

Readers reject unknown fields and unknown future schema versions instead of
silently changing behavior. Additive or breaking serialized changes require:

1. a schema-version change;
2. an explicit compatibility reader or migration;
3. fixed canonical-digest and round-trip tests;
4. a release-policy migration note.

Application stores should retain the original canonical JSON, version, digest,
publication reports, reviewer, and lifecycle timestamps. Do not rewrite an
already published revision during migration.

See the complete credential-free flow in
`examples/portable_agent/portable_agent.py`, and the hosted service boundary in
[Hosted runtime](hosting.md).
