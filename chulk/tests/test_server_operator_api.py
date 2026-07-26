"""Operator resource routes and dependency-free client contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from urllib.error import HTTPError
from urllib.request import Request

from starlette.testclient import TestClient

from chulk.children import ChildTask, ChildTaskLineage, ChildTaskSpec, ChildTaskStore
from chulk.config import load_config
from chulk.goals import GoalService, GoalStep, GoalStore
from chulk.profiles import SQLiteProfileStore
from chulk.server import ControlApiClient, ControlApiError
from chulk.scheduling import SQLiteScheduleStore
from chulk.tests.test_server_app import _app, _auth
from chulk.usage import BudgetScope, RunBudget


def test_goal_and_child_routes_are_revision_safe_and_profile_owned(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    profile_store = SQLiteProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    profile_store.create_profile("other", project_root=tmp_path)
    goal_service = GoalService(GoalStore(config.store_path))
    goal = goal_service.create(
        title="Ship operator API",
        acceptance_criteria=("Operator can inspect it.",),
        steps=(
            GoalStep(
                id="build",
                title="Build",
                description="Build the operator surface.",
                acceptance_criterion_ids=("criterion-1",),
            ),
        ),
        budget=RunBudget(scope=BudgetScope.GOAL),
    )
    child_store = ChildTaskStore(config.store_path)
    task = child_store.create(
        ChildTask(
            id="child-one",
            profile_id="default",
            spec=ChildTaskSpec(instruction="Inspect the operator surface."),
            lineage=ChildTaskLineage(),
        )
    )

    with TestClient(app) as client:
        goals = client.get(
            "/v1/profiles/default/goals",
            headers=_auth(tokens),
        )
        inspected_goal = client.get(
            f"/v1/profiles/default/goals/{goal.id}",
            headers=_auth(tokens),
        )
        foreign_goal = client.get(
            f"/v1/profiles/other/goals/{goal.id}",
            headers=_auth(tokens),
        )
        approved = client.post(
            f"/v1/profiles/default/goals/{goal.id}/actions",
            headers=_auth(tokens),
            json={
                "action": "approve",
                "revision": goal.revision,
            },
        )
        stale = client.post(
            f"/v1/profiles/default/goals/{goal.id}/actions",
            headers=_auth(tokens),
            json={
                "action": "run",
                "revision": goal.revision,
            },
        )

        tasks = client.get(
            "/v1/profiles/default/tasks",
            headers=_auth(tokens),
        )
        inspected_task = client.get(
            f"/v1/profiles/default/tasks/{task.id}",
            headers=_auth(tokens),
        )
        foreign_task = client.get(
            f"/v1/profiles/other/tasks/{task.id}",
            headers=_auth(tokens),
        )
        cancelled = client.post(
            f"/v1/profiles/default/tasks/{task.id}/actions",
            headers=_auth(tokens),
            json={
                "action": "cancel",
                "revision": task.revision,
                "reason": "No longer needed.",
            },
        )

    assert goals.status_code == 200
    assert goals.json()["goals"][0]["id"] == goal.id
    assert inspected_goal.json()["goal"]["id"] == goal.id
    assert inspected_goal.json()["events"][0]["kind"] == "goal.created"
    assert foreign_goal.status_code == 404
    assert approved.json()["goal"]["status"] == "approved"
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "goal_conflict"

    assert tasks.status_code == 200
    assert tasks.json()["tasks"][0]["id"] == task.id
    assert inspected_task.json()["events"][0]["kind"] == "child.created"
    assert foreign_task.status_code == 404
    assert cancelled.json()["task"]["status"] == "cancelled"


def test_operator_routes_are_discoverable_and_jobs_no_longer_require_adapter(
    tmp_path,
) -> None:
    app, tokens = _app(tmp_path)
    with TestClient(app) as client:
        jobs = client.get(
            "/v1/profiles/default/jobs?include_terminal=true",
            headers=_auth(tokens),
        )
        schema = client.get("/v1/openapi.json", headers=_auth(tokens)).json()

    assert jobs.status_code == 200
    assert jobs.json()["jobs"] == []
    paths = schema["paths"]
    assert "/v1/profiles/{profile_id}/goals/{goal_id}/actions" in paths
    assert "/v1/profiles/{profile_id}/tasks/{task_id}/actions" in paths
    assert "/v1/profiles/{profile_id}/traces/{conversation_id}" in paths


def test_job_listing_uses_the_requested_profile_owner(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    profile_store = SQLiteProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    other = profile_store.create_profile("other", project_root=tmp_path)
    default_job = SQLiteScheduleStore(config.store_path).create(
        adapter="telegram",
        destination_id="default-owner",
        prompt="default work",
        next_run_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    other_job = SQLiteScheduleStore(
        other.profile.store_path,
        profile_id="other",
    ).create(
        adapter="discord",
        destination_id="other-owner",
        prompt="other work",
        next_run_at=datetime.now(timezone.utc) + timedelta(days=1),
    )

    with TestClient(app) as client:
        default_jobs = client.get(
            "/v1/profiles/default/jobs",
            headers=_auth(tokens),
        ).json()["jobs"]
        other_jobs = client.get(
            "/v1/profiles/other/jobs",
            headers=_auth(tokens),
        ).json()["jobs"]

    assert [item["id"] for item in default_jobs] == [default_job.id]
    assert [item["id"] for item in other_jobs] == [other_job.id]
    assert "prompt" not in default_jobs[0]
    assert "prompt" not in other_jobs[0]


def test_operator_pages_use_resource_bound_cursors(tmp_path) -> None:
    app, tokens = _app(tmp_path)
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    service = GoalService(GoalStore(config.store_path))
    for index in range(2):
        service.create(
            title=f"Goal {index}",
            acceptance_criteria=("Done.",),
            steps=(
                GoalStep(
                    id=f"step-{index}",
                    title="Step",
                    description="Complete it.",
                    acceptance_criterion_ids=("criterion-1",),
                ),
            ),
            budget=RunBudget(scope=BudgetScope.GOAL),
        )

    with TestClient(app) as client:
        first = client.get(
            "/v1/profiles/default/goals?limit=1",
            headers=_auth(tokens),
        )
        cursor = first.json()["next_cursor"]
        second = client.get(
            f"/v1/profiles/default/goals?limit=1&cursor={cursor}",
            headers=_auth(tokens),
        )
        mismatched = client.get(
            f"/v1/profiles/default/jobs?cursor={cursor}",
            headers=_auth(tokens),
        )

    assert cursor
    assert first.json()["goals"][0]["id"] != second.json()["goals"][0]["id"]
    assert second.json()["next_cursor"] is None
    assert mismatched.status_code == 400
    assert "does not match" in mismatched.json()["error"]["message"]


class _Response:
    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class _StreamResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)


def test_control_api_client_shapes_authenticated_operator_requests() -> None:
    requests: list[tuple[Request, float]] = []

    def opener(request: Request, *, timeout: float):
        requests.append((request, timeout))
        return _Response({"schema_version": 1, "ok": True})

    client = ControlApiClient(
        "http://127.0.0.1:8765/",
        "secret-token",
        timeout_seconds=4.5,
        opener=opener,
    )
    client.list_jobs(
        "default",
        status="paused",
        include_terminal=True,
        limit=25,
    )
    client.control_goal(
        "default",
        "goal/one",
        action="pause",
        revision=7,
    )

    first, timeout = requests[0]
    assert timeout == 4.5
    assert first.full_url.endswith(
        "/v1/profiles/default/jobs?status=paused&include_terminal=true&limit=25"
    )
    assert first.get_header("Authorization") == "Bearer secret-token"

    second, _ = requests[1]
    assert second.full_url.endswith("/v1/profiles/default/goals/goal%2Fone/actions")
    assert json.loads(second.data or b"{}") == {
        "action": "pause",
        "revision": 7,
    }


def test_control_api_client_shapes_conversation_and_attention_requests() -> None:
    requests: list[Request] = []

    def opener(request: Request, *, timeout: float):
        assert timeout == 30.0
        requests.append(request)
        return _Response({"schema_version": 1, "ok": True})

    client = ControlApiClient(
        "http://127.0.0.1:8765",
        "secret-token",
        opener=opener,
    )
    client.send_message(
        "operations",
        "conversation/one",
        "Check deployment",
        mode="plan",
        idempotency_key="message-key",
    )
    client.decide_permission(
        "operations",
        "conversation/one",
        "permission/one",
        decision="allow",
        idempotency_key="decision-key",
    )

    message, permission = requests
    assert message.full_url.endswith(
        "/v1/profiles/operations/conversations/conversation%2Fone/messages"
    )
    assert json.loads(message.data or b"{}") == {
        "message": "Check deployment",
        "mode": "plan",
        "idempotency_key": "message-key",
    }
    assert permission.full_url.endswith(
        "/v1/profiles/operations/conversations/conversation%2Fone/"
        "permissions/permission%2Fone"
    )
    assert json.loads(permission.data or b"{}") == {
        "decision": "allow",
        "idempotency_key": "decision-key",
    }


def test_control_api_client_resumes_and_decodes_sse_events() -> None:
    requests: list[Request] = []

    def opener(request: Request, *, timeout: float):
        requests.append(request)
        return _StreamResponse(
            [
                b": heartbeat\n",
                b"\n",
                b"id: event-2\n",
                b"event: model.delta\n",
                b'data: {"id":"event-2","event":{"name":"model.delta"}}\n',
                b"\n",
            ]
        )

    client = ControlApiClient(
        "http://127.0.0.1:8765",
        "secret-token",
        opener=opener,
    )

    events = list(
        client.iter_events(
            "default",
            "conversation-1",
            after="event-1",
        )
    )

    assert requests[0].get_header("Last-event-id") == "event-1"
    assert requests[0].get_header("Accept") == "text/event-stream"
    assert events == [
        {
            "id": "event-2",
            "event": "model.delta",
            "data": {"id": "event-2", "event": {"name": "model.delta"}},
        }
    ]


def test_control_api_client_decodes_stable_api_errors() -> None:
    def opener(_request: Request, *, timeout: float):
        assert timeout == 30.0
        raise HTTPError(
            "http://127.0.0.1/v1/profiles/default/goals/missing",
            404,
            "not found",
            {},
            BytesIO(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {
                            "code": "goal_not_found",
                            "message": "goal is missing",
                        },
                    }
                ).encode()
            ),
        )

    client = ControlApiClient(
        "http://127.0.0.1",
        "secret-token",
        opener=opener,
    )
    try:
        client.get_goal("default", "missing")
    except ControlApiError as exc:
        assert exc.status == 404
        assert exc.code == "goal_not_found"
        assert str(exc) == "goal is missing"
    else:
        raise AssertionError("expected ControlApiError")
