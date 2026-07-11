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
from chulk import Agent, Capabilities

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
created. Rejection never creates a memory.

Proposal evidence and accepted memories may contain sensitive user data. Keep
the runtime database private and do not store secrets merely because a prompt
asks the agent to remember them.

Provenance identifies whether a candidate came from an explicit save or model
extraction and retains conversation/turn evidence for review. The host should
show that evidence before approval, apply retention and deletion policy, and
isolate stores per tenant. Import/export Markdown is a human interchange format;
SQLite remains the runtime source of truth. See [configuration](configuration.md)
and [safety](safety.md).
