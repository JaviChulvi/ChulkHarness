# Memory modes and review

Memory availability is an application decision, separate from the general tool
permission profile.

| Mode | Retrieval | Explicit save tool | Inferred candidates |
|---|---|---|---|
| `off` | Disabled | Hidden | Discarded |
| `read-only` | Enabled | Hidden | Discarded |
| `manual` | Enabled | Creates a proposal | Creates proposals |
| `automatic` | Enabled | Saves immediately | Saves immediately |

The SDK default is `read-only`, so a user message cannot silently create a
durable memory. CLI/runtime assembly that does not opt into SDK capabilities
retains its existing automatic behavior for compatibility.

```python
from chulk import Agent, AgentConfig, Capabilities

agent = Agent(
    capabilities=Capabilities(memory="manual"),
    llm=client,
)

for proposal in agent.list_memory_proposals():
    show_for_review(proposal.content, proposal.evidence)
    agent.approve_memory_proposal(proposal.id)
    # or: agent.reject_memory_proposal(proposal.id)
```

Manual proposals live in the same `.chulk/store.sqlite` database and survive
process restarts. They are not available to retrieval until approved. Public
`MemoryProposal` snapshots include status, source, confidence, evidence,
conversation/turn identity, review time, and the accepted memory id when one was
created. They also expose the namespace that owns the proposal. Rejection never
creates a memory.

## Namespace isolation

Every durable memory and proposal belongs to one opaque namespace. Existing
single-project callers map to the compatibility namespace `default`; existing
databases are migrated and backfilled automatically. Hosts that share one
database across tenants or workspaces must bind an explicit namespace once
when constructing the agent:

```python
agent = Agent(
    config=AgentConfig(
        project_root=".",
        memory_namespace="tenant:workspace-42",
    ),
    llm=client,
)
```

Namespace keys are normalized to lowercase and accept 1–128 ASCII letters,
digits, dots, underscores, colons, and hyphens. A scoped store applies the
namespace to prompt retrieval, FTS/LIKE/vector search, tag/profile lookup,
access counters, duplicate detection, CRUD, archive/restore, maintenance,
proposals, compaction, and Markdown import/export. There is deliberately no
cross-namespace admin search.

The Telegram adapter does not accept a caller-selected namespace. It derives
`telegram:chat:<chat_id>` for each private chat, including when several users
share one bot process.

Proposal evidence and accepted memories may contain sensitive user data. Keep
the runtime database private and do not store secrets merely because a prompt
asks the agent to remember them.

Provenance identifies whether a candidate came from an explicit save or model
extraction and retains conversation/turn evidence for review. The host should
show that evidence before approval, apply retention and deletion policy, and
bind a distinct namespace per tenant. Import/export Markdown is scoped
interchange only; SQLite remains the runtime source of truth. See
[configuration](configuration.md) and [safety](safety.md).
