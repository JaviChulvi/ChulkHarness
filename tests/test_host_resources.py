"""Typed host resource and application-event contracts."""

from __future__ import annotations

import json

import pytest

from chulk import (
    Agent,
    AgentConfig,
    AgentEvent,
    ApplicationEventIntent,
    ApplicationEventPayload,
    ApplicationEventSchema,
    AsyncAgent,
    EventName,
    HostResource,
    ResourceAvailablePayload,
    ResourcePersistence,
    RunCompletedPayload,
    TurnContextSection,
    tool,
)
from chulk.llm import LLMClient
from chulk.tools import ToolResult


class ScriptedClient(LLMClient):
    provider = "test"
    model = "resources"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, _messages, **_kwargs) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _tool_call(name: str) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": name,
            "arguments_json": "{}",
        }
    )


def _final(content: str = "done") -> str:
    return json.dumps({"type": "final_answer", "content": content})


def _resource(resource_id: str, *, source: str = "search") -> HostResource:
    return HostResource(
        id=resource_id,
        kind="document",
        title=f"Evidence {resource_id}",
        source=source,
        uri=f"https://example.test/{resource_id}",
        excerpt="Allowed public excerpt",
        provenance={"retriever": "host", "credential": "sk-secret-value"},
        relevance={"score": 0.91},
        persistence=ResourcePersistence.HOST_MANAGED,
    )


EVENT_SCHEMA = ApplicationEventSchema(
    namespace="acme.tickets",
    name="status.changed",
    version=1,
    payload_schema={
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["ticket_id", "status"],
        "additionalProperties": False,
    },
)


def test_host_resource_and_application_event_validation() -> None:
    resource = _resource("evidence-1")

    assert resource.provenance["credential"] == "[redacted]"
    assert resource.to_dict()["persistence"] == "host_managed"
    assert HostResource.from_dict(resource.to_dict()) == resource

    with pytest.raises(ValueError, match="excerpt exceeds"):
        HostResource(
            id="large",
            kind="document",
            title="Large",
            source="host",
            excerpt="x" * 2_001,
        )
    with pytest.raises(ValueError, match="https, http, or urn"):
        HostResource(
            id="file",
            kind="document",
            title="Private file",
            source="host",
            uri="file:///private/result.txt",
        )
    with pytest.raises(ValueError, match="JSON-safe"):
        ApplicationEventIntent(
            namespace="acme.tickets",
            name="status.changed",
            schema_version=1,
            payload={"invalid": object()},
            idempotency_key="ticket-1:changed",
        )
    with pytest.raises(ValueError, match="dotted lowercase"):
        ApplicationEventIntent(
            namespace="Tickets",
            name="status.changed",
            schema_version=1,
            payload={},
            idempotency_key="ticket-1:invalid-namespace",
        )
    with pytest.raises(ValueError, match="exceeds 16384 bytes"):
        ApplicationEventIntent(
            namespace="acme.tickets",
            name="status.changed",
            schema_version=1,
            payload={"content": "x" * 17_000},
            idempotency_key="ticket-1:oversized",
        )


def test_resources_and_application_events_share_ordered_run_stream(tmp_path) -> None:
    context_resource = _resource("source-1")
    generated_resource = _resource("artifact-1", source="report_tool")
    intent = ApplicationEventIntent(
        namespace="acme.tickets",
        name="status.changed",
        schema_version=1,
        payload={
            "ticket_id": "T-42",
            "status": "ready",
            "token": "sk-secret-value",
        },
        idempotency_key="T-42:ready",
    )
    schema = ApplicationEventSchema(
        namespace="acme.tickets",
        name="status.changed",
        version=1,
        payload_schema={"type": "object"},
    )

    @tool(application_event_schemas=(schema,))
    def publish_report() -> ToolResult:
        return ToolResult(
            tool_name="publish_report",
            success=True,
            observation="Report published",
            resources=(generated_resource,),
            application_events=(intent,),
        )

    agent = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=ScriptedClient([_tool_call("publish_report"), _final()]),
        tools=[publish_report],
        skills=[],
    )
    events = list(
        agent.run_events(
            "Use the evidence",
            context_sections=[
                TurnContextSection(
                    id="private-context",
                    content="private retrieved content",
                    persist_content=False,
                    resource=context_resource,
                )
            ],
        )
    )

    names = [event.name for event in events]
    context_index = names.index(EventName.RESOURCE_AVAILABLE.value)
    request_index = names.index(EventName.MODEL_REQUEST_STARTED.value)
    tool_index = names.index(EventName.TOOL_CALL_COMPLETED.value)
    tool_resource_index = names.index(
        EventName.RESOURCE_AVAILABLE.value,
        context_index + 1,
    )
    application_index = names.index(EventName.APPLICATION_EVENT.value)
    terminal_index = names.index(EventName.RUN_COMPLETED.value)

    assert context_index < request_index
    assert tool_index < tool_resource_index < application_index < terminal_index
    resource_events = [
        event for event in events
        if isinstance(event.payload, ResourceAvailablePayload)
    ]
    assert [event.payload.resource.id for event in resource_events] == [
        "source-1",
        "artifact-1",
    ]
    application_event = events[application_index]
    assert isinstance(application_event.payload, ApplicationEventPayload)
    assert application_event.idempotency_key == "T-42:ready"
    assert application_event.payload.payload["token"] == "[redacted]"
    assert application_event.causation_id == events[application_index - 1].event_id

    terminal = events[-1]
    assert isinstance(terminal.payload, RunCompletedPayload)
    assert [item.id for item in terminal.payload.result.resources] == [
        "source-1",
        "artifact-1",
    ]
    serialized = json.dumps([event.to_dict() for event in events])
    assert "private retrieved content" not in serialized
    assert "sk-secret-value" not in serialized

    restored = AgentEvent.from_dict(application_event.to_dict())
    assert restored.event_id == application_event.event_id
    assert restored.idempotency_key == "T-42:ready"
    assert restored.to_dict()["payload"]["payload"]["ticket_id"] == "T-42"


def test_unknown_or_invalid_application_event_schema_is_not_published(tmp_path) -> None:
    unknown = ApplicationEventIntent(
        namespace="acme.tickets",
        name="unknown",
        schema_version=1,
        payload={"ticket_id": "T-42"},
        idempotency_key="T-42:unknown",
    )

    @tool(application_event_schemas=(EVENT_SCHEMA,))
    def invalid_publisher() -> ToolResult:
        return ToolResult(
            tool_name="invalid_publisher",
            success=True,
            observation="invalid publication",
            application_events=(unknown,),
        )

    agent = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=ScriptedClient([_tool_call("invalid_publisher"), _final()]),
        tools=[invalid_publisher],
        skills=[],
    )
    events = list(agent.run_events("publish"))

    assert EventName.APPLICATION_EVENT.value not in [event.name for event in events]
    result = events[-1].payload.result
    assert result.tool_calls[0].failure_kind == "invalid_output"
    assert "unregistered application event schema" in (
        result.tool_calls[0].metadata["publication_validation_error"]
    )


@pytest.mark.asyncio
async def test_async_tool_publications_use_the_same_contract(tmp_path) -> None:
    resource = _resource("async-artifact", source="async_tool")
    intent = ApplicationEventIntent(
        namespace="acme.tickets",
        name="status.changed",
        schema_version=1,
        payload={"ticket_id": "T-9", "status": "done"},
        idempotency_key="T-9:done",
    )

    @tool(application_event_schemas=(EVENT_SCHEMA,))
    async def async_publish() -> ToolResult:
        return ToolResult(
            tool_name="async_publish",
            success=True,
            observation="published",
            resources=(resource,),
            application_events=(intent,),
        )

    agent = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=ScriptedClient([_tool_call("async_publish"), _final("async done")]),
        tools=[async_publish],
        skills=[],
    )
    events = [event async for event in agent.run_events_async("publish")]

    assert [event.name for event in events].count("resource.available") == 1
    assert [event.name for event in events].count("application.event") == 1
    terminal = events[-1].payload.result
    assert terminal.content == "async done"
    assert terminal.resources == (resource,)
