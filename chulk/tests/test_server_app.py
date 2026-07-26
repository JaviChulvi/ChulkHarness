"""HTTP security and contract tests for the optional control server."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from chulk.config import load_config
from chulk.llm import LLMClient
from chulk.profiles import ProfileRuntimeFactory, SQLiteProfileStore
from chulk.runtime import create_agent
from chulk.server import ConversationDispatcher, create_control_app
from chulk.events import AgentEvent, ModelDeltaPayload
from chulk.server.security import ControlTokenStore, SlidingWindowRateLimiter
from chulk.scheduling import SQLiteScheduleStore
from chulk.tools.permissions import PermissionRequest, ToolPermissionLevel


class ServerLLM(LLMClient):
    provider = "test"
    model = "server"

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return json.dumps(
            {"type": "final_answer", "content": f"answer {messages[-1]['content']}"}
        )


def _app(tmp_path, *, max_body_bytes=1_000_000):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    profile_store = SQLiteProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    runtime_factory = ProfileRuntimeFactory(config, profile_store=profile_store)

    def build(profile_id, conversation_id, metadata):
        resolved = runtime_factory.resolve(profile_id)
        return create_agent(
            resolved.config,
            llm_client=ServerLLM(),
            tool_specs=[],
            skill_specs=[],
            conversation_id=conversation_id,
            conversation_metadata=dict(metadata or {}),
            profile_id=profile_id,
        )

    dispatcher = ConversationDispatcher(runtime_factory, agent_builder=build)
    tokens = ControlTokenStore(config.runtime_dir / "control.token")
    app = create_control_app(
        config,
        dispatcher=dispatcher,
        token_store=tokens,
        max_body_bytes=max_body_bytes,
        sse_heartbeat_seconds=0.05,
    )
    return app, tokens


def _auth(tokens: ControlTokenStore) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens.load_or_create()}"}


def test_control_token_is_owner_only_and_rotation_invalidates_old_value(tmp_path) -> None:
    store = ControlTokenStore(tmp_path / "runtime" / "control.token")
    first = store.load_or_create()
    second = store.rotate()

    assert first != second
    assert not store.matches(first)
    assert store.matches(second)
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600


def test_http_api_requires_auth_origin_and_browser_csrf(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/v1/profiles").status_code == 401
        assert client.get(
            "/v1/profiles",
            headers={**_auth(tokens), "Origin": "https://evil.example"},
        ).status_code == 403
        response = client.post(
            "/v1/profiles/default/conversations",
            headers={
                **_auth(tokens),
                "Origin": "http://localhost:3000",
                "Content-Type": "application/json",
            },
            json={},
        )
        assert response.status_code == 403
        response = client.post(
            "/v1/profiles/default/conversations",
            headers={
                **_auth(tokens),
                "Origin": "http://localhost:3000",
                "X-Chulk-CSRF": tokens.csrf_token(),
            },
            json={},
        )
        assert response.status_code == 201
    audit = (
        tmp_path / ".chulk" / "control-audit.jsonl"
    ).read_text(encoding="utf-8")
    assert tokens.load_or_create() not in audit
    assert "request body" not in audit


def test_webchat_shell_is_public_but_control_data_stays_authenticated(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        page = client.get("/webchat")
        stylesheet = client.get("/webchat/assets/app.css")
        script = client.get("/webchat/assets/app.js")

        assert page.status_code == 200
        assert "Chulk <span>control room</span>" in page.text
        assert tokens.load_or_create() not in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert stylesheet.status_code == 200
        assert script.status_code == 200
        assert "sessionStorage" in script.text
        assert client.get("/v1/session").status_code == 401
        assert client.get("/webchat/assets/missing.js").status_code == 404


def test_operator_dashboard_is_public_shell_with_authenticated_data(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        page = client.get("/dashboard")
        stylesheet = client.get("/dashboard/assets/dashboard.css")
        script = client.get("/dashboard/assets/dashboard.js")

        assert page.status_code == 200
        assert "Chulk operations board" in page.text
        assert 'id="attention"' in page.text
        assert 'id="work"' in page.text
        assert 'id="usage"' in page.text
        assert 'id="evidence"' in page.text
        assert "<dialog" in page.text
        assert tokens.load_or_create() not in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert stylesheet.status_code == 200
        assert "@media (max-width: 580px)" in stylesheet.text
        assert "@media (prefers-reduced-motion: reduce)" in stylesheet.text
        assert script.status_code == 200
        assert 'api("/v1/profiles")' in script.text
        assert "sessionStorage" in script.text
        assert "innerHTML" not in script.text
        assert client.get("/v1/profiles").status_code == 401
        inbox = client.get(
            "/v1/profiles/default/permissions?status=pending",
            headers=_auth(tokens),
        )
        assert inbox.status_code == 200
        assert inbox.json()["permissions"] == []
        assert client.get("/dashboard/assets/missing.js").status_code == 404


def test_authenticated_webchat_session_creates_pairing_and_lists_routes(
    tmp_path,
) -> None:
    app, tokens = _app(tmp_path)
    origin = "http://localhost:3000"
    with TestClient(app) as client:
        session = client.get(
            "/v1/session",
            headers={**_auth(tokens), "Origin": origin},
        )
        assert session.status_code == 200
        csrf = session.json()["csrf_token"]

        created = client.post(
            "/v1/gateway/pairings",
            headers={
                **_auth(tokens),
                "Origin": origin,
                "X-Chulk-CSRF": csrf,
            },
            json={
                "adapter": "discord",
                "account_id": "primary",
                "profile_id": "default",
                "principal_id": "user-7",
                "ttl_seconds": 600,
            },
        )
        routes = client.get(
            "/v1/gateway/routes",
            headers={**_auth(tokens), "Origin": origin},
        )

        assert created.status_code == 201
        assert len(created.json()["pairing"]["code"]) >= 24
        assert created.json()["pairing"]["principal_id"] == "user-7"
        assert routes.status_code == 200
        assert routes.json()["routes"] == []
        assert client.post(
            "/v1/gateway/pairings",
            headers={
                **_auth(tokens),
                "Origin": origin,
                "X-Chulk-CSRF": csrf,
            },
            json={
                "adapter": "discord",
                "account_id": "primary",
                "profile_id": "missing",
            },
        ).status_code == 404
        assert client.post(
            "/v1/gateway/pairings",
            headers={
                **_auth(tokens),
                "Origin": origin,
                "X-Chulk-CSRF": csrf,
            },
            json={
                "adapter": "discord",
                "account_id": "primary",
                "profile_id": "default",
                "ttl_seconds": 10,
            },
        ).status_code == 400
        assert client.post(
            "/v1/gateway/pairings",
            headers={
                **_auth(tokens),
                "Origin": origin,
                "X-Chulk-CSRF": csrf,
            },
            json={
                "adapter": "unknown",
                "account_id": "primary",
                "profile_id": "default",
            },
        ).status_code == 400


def test_http_api_creates_conversation_and_queues_idempotent_message(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        profiles = client.get("/v1/profiles", headers=_auth(tokens))
        assert profiles.status_code == 200
        assert profiles.json()["profiles"][0]["id"] == "default"
        created = client.post(
            "/v1/profiles/default/conversations",
            headers=_auth(tokens),
            json={"metadata": {"client": "test"}},
        )
        conversation_id = created.json()["conversation"]["id"]
        first = client.post(
            f"/v1/profiles/default/conversations/{conversation_id}/messages",
            headers=_auth(tokens),
            json={"message": "hello", "idempotency_key": "message-1"},
        )
        replay = client.post(
            f"/v1/profiles/default/conversations/{conversation_id}/messages",
            headers=_auth(tokens),
            json={"message": "hello", "idempotency_key": "message-1"},
        )

        assert first.status_code == replay.status_code == 202
        assert first.json()["command"]["id"] == replay.json()["command"]["id"]
        schema = client.get("/v1/schema", headers=_auth(tokens)).json()
        serialized = json.dumps(schema)
        assert schema["openapi"] == "3.1.0"
        assert "raw_traces" in serialized
        assert "trace_path" not in serialized
        assert "credential_refs" not in serialized


def test_operator_routes_are_bounded_and_do_not_expose_raw_traces(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/v1/profiles/default/conversations",
            headers=_auth(tokens),
            json={},
        )
        conversation_id = created.json()["conversation"]["id"]

        conversations = client.get(
            "/v1/profiles/default/conversations?limit=10",
            headers=_auth(tokens),
        )
        traces = client.get(
            "/v1/profiles/default/traces",
            headers=_auth(tokens),
        )
        artifacts = client.get(
            f"/v1/profiles/default/conversations/{conversation_id}/artifacts",
            headers=_auth(tokens),
        )
        proposals = client.get(
            "/v1/profiles/default/proposals",
            headers=_auth(tokens),
        )
        usage = client.get(
            "/v1/profiles/default/usage?limit=10",
            headers=_auth(tokens),
        )

        assert conversations.json()["conversations"][0]["id"] == conversation_id
        assert traces.json()["traces"][0]["conversation_id"] == conversation_id
        assert "trace_path" not in json.dumps(traces.json())
        assert artifacts.json()["artifacts"] == []
        assert proposals.json()["proposals"] == []
        assert usage.json()["entries"] == []
        jobs = client.get(
            "/v1/profiles/default/jobs",
            headers=_auth(tokens),
        )
        assert jobs.status_code == 200
        assert jobs.json()["jobs"] == []


def test_automation_control_and_authenticated_webhook_routes(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteScheduleStore(config.store_path)
    job = store.create(
        adapter="webhook",
        destination_id="owner",
        prompt="handle event",
        next_run_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    with TestClient(app) as client:
        inspected = client.get(
            f"/v1/profiles/default/jobs/{job.id}",
            headers=_auth(tokens),
        )
        assert inspected.status_code == 200
        assert inspected.json()["job"]["id"] == job.id

        paused = client.post(
            f"/v1/profiles/default/jobs/{job.id}/actions",
            headers=_auth(tokens),
            json={
                "action": "pause",
                "revision": job.revision,
                "idempotency_key": "api-pause",
            },
        )
        assert paused.status_code == 200
        assert paused.json()["job"]["status"] == "paused"

        resumed = store.resume(
            job.id,
            expected_revision=paused.json()["job"]["revision"],
            idempotency_key="test-resume",
        )
        webhook = client.post(
            f"/v1/profiles/default/jobs/{job.id}/webhooks",
            headers=_auth(tokens),
            json={},
        )
        assert webhook.status_code == 201
        trigger_id = webhook.json()["trigger"]["id"]
        credential = webhook.json()["credential"]

        denied = client.post(
            f"/v1/profiles/default/automation-webhooks/{trigger_id}",
            headers=_auth(tokens),
            json={
                "credential": "wrong",
                "event_id": "evt-api",
                "payload": {},
            },
        )
        assert denied.status_code == 401
        accepted = client.post(
            f"/v1/profiles/default/automation-webhooks/{trigger_id}",
            headers=_auth(tokens),
            json={
                "credential": credential,
                "event_id": "evt-api",
                "payload": {"value": 1},
            },
        )
        assert accepted.status_code == 202
        replay = client.post(
            f"/v1/profiles/default/automation-webhooks/{trigger_id}",
            headers=_auth(tokens),
            json={
                "credential": credential,
                "event_id": "evt-api",
                "payload": {"value": 2},
            },
        )
        assert (
            replay.json()["trigger_event"]["id"]
            == accepted.json()["trigger_event"]["id"]
        )
        assert resumed.status.value == "active"


def test_permission_endpoint_is_owned_and_idempotent(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/v1/profiles/default/conversations",
            headers=_auth(tokens),
            json={},
        )
        conversation_id = created.json()["conversation"]["id"]
        broker = app.state.dispatcher.permissions("default", conversation_id)
        pending = broker.create(
            PermissionRequest(
                tool_name="shell",
                permission_level=ToolPermissionLevel.SHELL,
                arguments={"command": "printf hello"},
                requires_confirmation=True,
                policy_name="workspace-write",
                reason="shell approval",
            )
        )
        url = (
            f"/v1/profiles/default/conversations/{conversation_id}"
            f"/permissions/{pending.id}"
        )
        first = client.post(
            url,
            headers=_auth(tokens),
            json={"decision": "allow", "idempotency_key": "decision-1"},
        )
        replay = client.post(
            url,
            headers=_auth(tokens),
            json={"decision": "allow", "idempotency_key": "decision-1"},
        )
        conflict = client.post(
            url,
            headers=_auth(tokens),
            json={"decision": "deny", "idempotency_key": "decision-2"},
        )

        assert first.status_code == replay.status_code == 200
        assert first.json()["permission"]["status"] == "allowed"
        assert conflict.status_code == 409
        assert client.post(
            url.replace("/default/", "/missing/"),
            headers=_auth(tokens),
            json={"decision": "allow", "idempotency_key": "other"},
        ).status_code == 404


def test_http_api_rejects_malformed_and_oversized_requests(tmp_path) -> None:
    app, tokens = _app(tmp_path, max_body_bytes=64)
    with TestClient(app) as client:
        malformed = client.post(
            "/v1/profiles/default/conversations",
            headers={**_auth(tokens), "Content-Type": "application/json"},
            content=b"{",
        )
        oversized = client.post(
            "/v1/profiles/default/conversations",
            headers=_auth(tokens),
            json={"metadata": {"value": "x" * 100}},
        )
        chunked = client.post(
            "/v1/profiles/default/conversations",
            headers={**_auth(tokens), "Content-Type": "application/json"},
            content=(part for part in (b'{"metadata":{"value":"', b"x" * 100, b'"}}')),
        )

        assert malformed.status_code == 400
        assert malformed.json()["error"]["code"] == "malformed_json"
        assert oversized.status_code == 413
        assert chunked.status_code == 413


def test_rate_limiter_bounds_repeated_requests() -> None:
    import asyncio

    async def scenario() -> tuple[bool, bool]:
        limiter = SlidingWindowRateLimiter(requests=1, window_seconds=60)
        return await limiter.allow("client"), await limiter.allow("client")

    assert asyncio.run(scenario()) == (True, False)


def test_websocket_gateway_uses_authenticated_shared_dispatch(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    protocol = f"chulk.control.{tokens.load_or_create()}"
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/gateway/ws"):
                pass

        with client.websocket_connect(
            "/v1/gateway/ws",
            subprotocols=[protocol],
        ) as socket:
            socket.send_json(
                {
                    "type": "hello",
                    "schema_version": 1,
                    "profile_id": "default",
                    "client_id": "browser-1",
                }
            )
            assert socket.receive_json()["type"] == "ready"
            socket.send_json(
                {
                    "type": "message",
                    "schema_version": 1,
                    "event_id": "event-1",
                    "idempotency_key": "ws-message-1",
                    "message": "hello websocket",
                }
            )
            accepted = socket.receive_json()
            completed = socket.receive_json()

            assert accepted["type"] == "message.accepted"
            assert accepted["inbox_id"]
            assert completed["type"] == "message.completed"
            assert completed["text"] == "answer hello websocket"
            assert completed["conversation_id"]
            socket.send_json(
                {
                    "type": "message",
                    "schema_version": 1,
                    "event_id": "event-2",
                    "idempotency_key": "ws-message-2",
                    "conversation_id": "missing-conversation",
                    "message": "hello missing",
                }
            )
            assert socket.receive_json()["type"] == "message.accepted"
            missing = socket.receive_json()
            assert missing["status"] == "failed"
            assert missing["error"] == "conversation_not_found"


@pytest.mark.asyncio
async def test_sse_stream_orders_events_and_resumes_from_last_event_id(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    dispatcher = app.state.dispatcher
    conversation = await dispatcher.create_conversation("default")
    conversation_id = conversation["id"]
    journal = dispatcher.journal("default", conversation_id)
    first = journal.append(
        AgentEvent(
            name="model.delta",
            profile_id="default",
            conversation_id=conversation_id,
            payload=ModelDeltaPayload("one"),
        )
    )
    journal.append(
        AgentEvent(
            name="model.delta",
            profile_id="default",
            conversation_id=conversation_id,
            payload=ModelDeltaPayload("two"),
        )
    )

    messages = await _asgi_sse_request(
        app,
        (
            f"/v1/profiles/default/conversations/{conversation_id}/events"
        ),
        token=tokens.load_or_create(),
        last_event_id=first.event_id,
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    ).decode()

    assert f"id: {first.event_id}" not in body
    assert '"text":"two"' in body
    assert "event: model.delta" in body
    await dispatcher.close()


def test_event_endpoint_supports_bounded_reconnect_pages(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/v1/profiles/default/conversations",
            headers=_auth(tokens),
            json={},
        )
        conversation_id = created.json()["conversation"]["id"]
        first = app.state.dispatcher.journal(
            "default",
            conversation_id,
        ).append(
            AgentEvent(
                name="model.delta",
                profile_id="default",
                conversation_id=conversation_id,
                payload=ModelDeltaPayload("bounded"),
            )
        )

        page = client.get(
            f"/v1/profiles/default/conversations/{conversation_id}/events"
            "?follow=false",
            headers=_auth(tokens),
        )
        resumed = client.get(
            f"/v1/profiles/default/conversations/{conversation_id}/events"
            f"?follow=false&after={first.event_id}",
            headers=_auth(tokens),
        )

        assert page.status_code == 200
        assert page.json()["events"][0]["id"] == first.event_id
        assert page.json()["next_cursor"] == first.event_id
        assert resumed.json() == {
            "schema_version": 1,
            "events": [],
            "next_cursor": first.event_id,
        }


async def _asgi_sse_request(
    app,
    path: str,
    *,
    token: str,
    last_event_id: str | None = None,
) -> list[dict]:
    body_sent = asyncio.Event()
    first_receive = True
    messages: list[dict] = []
    headers = [(b"authorization", f"Bearer {token}".encode())]
    if last_event_id is not None:
        headers.append((b"last-event-id", last_event_id.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8765),
    }

    async def receive():
        nonlocal first_receive
        if first_receive:
            first_receive = False
            return {"type": "http.request", "body": b"", "more_body": False}
        await body_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            body_sent.set()

    await app(scope, receive, send)
    return messages
