"""HTTP security and contract tests for the optional control server."""

from __future__ import annotations

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
from chulk.server.security import ControlTokenStore, SlidingWindowRateLimiter


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
        assert "raw_traces" in serialized
        assert "trace_path" not in serialized
        assert "credential_refs" not in serialized


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

        assert malformed.status_code == 400
        assert malformed.json()["error"]["code"] == "malformed_json"
        assert oversized.status_code == 413


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
