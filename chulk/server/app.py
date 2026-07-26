"""Optional authenticated Starlette application for local control clients."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from chulk.config import Config
from chulk.gateway import SQLiteGatewayRouter
from chulk.profiles import ProfileNotFoundError, ProfileRuntimeFactory
from chulk.server.dispatcher import (
    ConversationBackpressureError,
    ConversationCommandNotFoundError,
    ConversationDispatcher,
    ControlDecisionConflictError,
)
from chulk.server.journal import PublicEventCursorExpiredError
from chulk.server.models import (
    API_SCHEMA_VERSION,
    ApiError,
    AutomationActionRequest,
    ConversationCreateRequest,
    ConversationMessageRequest,
    GatewayPairingRequest,
    OperatorActionRequest,
    PermissionDecisionRequest,
    PlanDecisionRequest,
    ProposalDecisionRequest,
)
from chulk.server.operators import (
    OperatorService,
    boolean_query,
    integer_query,
    parse_timestamp,
)
from chulk.server.permissions import (
    PermissionDecisionConflictError,
    PermissionRequestNotFoundError,
)
from chulk.server.security import (
    ControlAuditLog,
    ControlSecurityMiddleware,
    ControlTokenStore,
    RequestBodyTooLargeError,
)
from chulk.sessions import SessionNotFoundError


class ServerDependencyError(RuntimeError):
    """Raised when the optional server dependencies are unavailable."""


class ApiProblem(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.error = ApiError(code, message, status, details)
        super().__init__(message)


def create_control_app(
    config: Config,
    *,
    dispatcher: ConversationDispatcher | None = None,
    token_store: ControlTokenStore | None = None,
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1",
        "http://localhost",
    ),
    max_body_bytes: int = 1_000_000,
    sse_heartbeat_seconds: float = 15.0,
):
    """Build the optional ASGI app without adding imports to the base SDK."""
    try:
        from starlette.applications import Starlette
        from starlette.responses import FileResponse, JSONResponse
        from starlette.routing import Route, WebSocketRoute
    except ImportError as exc:
        raise ServerDependencyError(
            "Control server dependencies are unavailable; install chulkharness[server]"
        ) from exc
    if sse_heartbeat_seconds <= 0:
        raise ValueError("sse_heartbeat_seconds must be greater than zero")
    runtime_factory = ProfileRuntimeFactory(config)
    controller = dispatcher or ConversationDispatcher(runtime_factory)
    operators = OperatorService(runtime_factory)
    credentials = token_store or ControlTokenStore(
        config.runtime_dir / "control.token"
    )
    credentials.load_or_create()
    router = SQLiteGatewayRouter(config.runtime_dir / "control.sqlite")
    webchat_root = Path(__file__).with_name("webchat")

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await controller.close()

    async def problem_handler(_request: Any, exc: Exception):
        assert isinstance(exc, ApiProblem)
        return JSONResponse(exc.error.to_dict(), status_code=exc.error.status)

    async def value_handler(_request: Any, exc: Exception):
        assert isinstance(exc, ValueError)
        error = ApiError("invalid_request", str(exc), 400)
        return JSONResponse(error.to_dict(), status_code=400)

    async def body_limit_handler(_request: Any, exc: Exception):
        assert isinstance(exc, RequestBodyTooLargeError)
        error = ApiError("request_too_large", str(exc), 413)
        return JSONResponse(error.to_dict(), status_code=413)

    async def not_found_handler(_request: Any, exc: Exception):
        error = ApiError("not_found", str(exc), 404)
        return JSONResponse(error.to_dict(), status_code=404)

    async def profiles(_request):
        items = [
            {
                "id": stored.profile.id,
                "permission_profile": stored.profile.permission_profile,
                "model_profile_id": stored.profile.model_profile_id,
                "execution_backend_id": stored.profile.execution_backend_id,
                "implicit": stored.profile.implicit,
            }
            for stored in runtime_factory.profile_store.list()
        ]
        return _json({"profiles": items})

    async def session_info(_request):
        return _json(
            {
                "csrf_token": credentials.csrf_token(),
                "websocket_path": "/v1/gateway/ws",
            }
        )

    async def gateway_routes(_request):
        return _json(
            {
                "routes": [
                    {
                        "id": item.id,
                        "adapter": item.adapter,
                        "account_id": item.account_id,
                        "profile_id": item.profile_id,
                        "principal_id": item.principal_id,
                        "destination_id": item.destination_id,
                        "thread_id": item.thread_id,
                    }
                    for item in router.list_routes()
                    if item.adapter != "websocket"
                ]
            }
        )

    async def create_gateway_pairing(request):
        body = GatewayPairingRequest.from_dict(await _json_body(request))
        try:
            runtime_factory.resolve(body.profile_id)
        except ProfileNotFoundError as exc:
            raise ApiProblem(404, "profile_not_found", str(exc)) from exc
        challenge = router.create_pairing(
            adapter=body.adapter,
            account_id=body.account_id,
            profile_id=body.profile_id,
            principal_id=body.principal_id,
            ttl_seconds=body.ttl_seconds,
        )
        return _json(
            {
                "pairing": {
                    "id": challenge.id,
                    "code": challenge.code,
                    "adapter": challenge.adapter,
                    "account_id": challenge.account_id,
                    "profile_id": challenge.profile_id,
                    "principal_id": challenge.principal_id,
                    "expires_at": challenge.expires_at.isoformat(),
                }
            },
            status_code=201,
        )

    async def webchat(_request):
        return FileResponse(
            webchat_root / "index.html",
            media_type="text/html",
            headers=_webchat_headers(cache_control="no-store"),
        )

    async def webchat_asset(request):
        name = request.path_params["name"]
        assets = {
            "app.css": "text/css",
            "app.js": "text/javascript",
        }
        media_type = assets.get(name)
        if media_type is None:
            raise ApiProblem(404, "asset_not_found", "webchat asset not found")
        return FileResponse(
            webchat_root / name,
            media_type=media_type,
            headers=_webchat_headers(
                cache_control="public, max-age=300",
                content_security=False,
            ),
        )

    async def create_conversation(request):
        body = ConversationCreateRequest.from_dict(await _json_body(request))
        profile_id = request.path_params["profile_id"]
        try:
            value = await controller.create_conversation(
                profile_id,
                conversation_id=body.conversation_id,
                metadata=body.metadata,
            )
        except ProfileNotFoundError as exc:
            raise ApiProblem(404, "profile_not_found", str(exc)) from exc
        except SessionNotFoundError as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc
        return _json({"conversation": value}, status_code=201)

    async def list_conversations(request):
        try:
            value = operators.conversations(
                request.path_params["profile_id"],
                limit=_limit(request.query_params.get("limit")),
            )
        except ProfileNotFoundError as exc:
            raise ApiProblem(404, "profile_not_found", str(exc)) from exc
        return _json(value)

    async def send_message(request):
        body = ConversationMessageRequest.from_dict(await _json_body(request))
        profile_id, conversation_id = _conversation_params(request)
        try:
            command = await controller.submit(
                profile_id,
                conversation_id,
                body.message,
                mode=body.mode,
                source="api",
                idempotency_key=body.idempotency_key,
            )
        except ConversationBackpressureError as exc:
            raise ApiProblem(429, "conversation_backpressure", str(exc)) from exc
        except (ProfileNotFoundError, SessionNotFoundError) as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc
        return _json({"command": command.to_dict()}, status_code=202)

    async def command_status(request):
        profile_id, conversation_id = _conversation_params(request)
        try:
            command = controller.get_command(
                profile_id,
                conversation_id,
                request.path_params["command_id"],
            )
        except ConversationCommandNotFoundError as exc:
            raise ApiProblem(404, "command_not_found", str(exc)) from exc
        return _json({"command": command.to_dict()})

    async def conversation_events(request):
        try:
            from starlette.responses import StreamingResponse
        except ImportError as exc:
            raise ServerDependencyError from exc
        profile_id, conversation_id = _conversation_params(request)
        try:
            await controller.get_conversation(profile_id, conversation_id)
            journal = controller.journal(profile_id, conversation_id)
            cursor = request.headers.get("last-event-id") or request.query_params.get(
                "after"
            )
            initial = journal.list(conversation_id, after_id=cursor, limit=250)
        except PublicEventCursorExpiredError as exc:
            raise ApiProblem(409, "event_cursor_expired", str(exc)) from exc
        except (ProfileNotFoundError, SessionNotFoundError) as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc

        async def stream() -> AsyncIterator[bytes]:
            records = initial
            last_id = cursor
            elapsed = 0.0
            poll_interval = min(0.25, sse_heartbeat_seconds)
            while True:
                for record in records:
                    last_id = record.event_id
                    yield _sse(record.event.name, record.event_id, record.to_dict())
                if await request.is_disconnected():
                    return
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                try:
                    records = (
                        journal.list(
                            conversation_id,
                            after_id=last_id,
                            limit=250,
                        )
                        if last_id
                        else journal.list(conversation_id, limit=250)
                    )
                except PublicEventCursorExpiredError:
                    yield _sse(
                        "stream.reset",
                        uuid4().hex,
                        {
                            "schema_version": API_SCHEMA_VERSION,
                            "error": {
                                "code": "event_cursor_expired",
                                "message": "resume cursor left the retained journal",
                            },
                        },
                    )
                    return
                if not records and elapsed >= sse_heartbeat_seconds:
                    yield b": heartbeat\n\n"
                    elapsed = 0.0
                elif records:
                    elapsed = 0.0

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def cancel_conversation(request):
        profile_id, conversation_id = _conversation_params(request)
        try:
            cancelled = await controller.cancel(profile_id, conversation_id)
        except (ProfileNotFoundError, SessionNotFoundError) as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc
        return _json({"cancel_requested": cancelled}, status_code=202)

    async def approve_plan(request):
        body = PlanDecisionRequest.from_dict(await _json_body(request))
        profile_id, conversation_id = _conversation_params(request)
        try:
            result = await controller.approve_plan(
                profile_id,
                conversation_id,
                request.path_params["turn_id"],
                idempotency_key=body.idempotency_key,
            )
        except ControlDecisionConflictError as exc:
            raise ApiProblem(409, "plan_decision_conflict", str(exc)) from exc
        return _json({"result": result})

    async def reject_plan(request):
        body = PlanDecisionRequest.from_dict(await _json_body(request))
        profile_id, conversation_id = _conversation_params(request)
        try:
            result = await controller.reject_plan(
                profile_id,
                conversation_id,
                request.path_params["turn_id"],
                idempotency_key=body.idempotency_key,
            )
        except ControlDecisionConflictError as exc:
            raise ApiProblem(409, "plan_decision_conflict", str(exc)) from exc
        return _json({"result": result})

    async def list_permissions(request):
        profile_id, conversation_id = _conversation_params(request)
        try:
            status = request.query_params.get("status")
            items = controller.permissions(profile_id, conversation_id).list(
                status=status,
                limit=_limit(request.query_params.get("limit")),
            )
        except (ProfileNotFoundError, SessionNotFoundError) as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc
        return _json({"permissions": [item.to_dict() for item in items]})

    async def decide_permission(request):
        body = PermissionDecisionRequest.from_dict(await _json_body(request))
        profile_id, conversation_id = _conversation_params(request)
        try:
            result = controller.permissions(profile_id, conversation_id).decide(
                request.path_params["permission_request_id"],
                body.decision,
                idempotency_key=body.idempotency_key,
                reason=body.reason,
            )
        except PermissionRequestNotFoundError as exc:
            raise ApiProblem(404, "permission_not_found", str(exc)) from exc
        except (ProfileNotFoundError, SessionNotFoundError) as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc
        except PermissionDecisionConflictError as exc:
            raise ApiProblem(409, "permission_conflict", str(exc)) from exc
        return _json({"permission": result.to_dict()})

    async def schema(_request):
        return _json(_schema())

    async def goals(request):
        return _json(
            operators.goals(
                request.path_params["profile_id"],
                status=request.query_params.get("status"),
                limit=_limit(request.query_params.get("limit")),
                cursor=request.query_params.get("cursor"),
            )
        )

    async def goal(request):
        try:
            value = operators.goal(
                request.path_params["profile_id"],
                request.path_params["goal_id"],
            )
        except LookupError as exc:
            raise ApiProblem(404, "goal_not_found", str(exc)) from exc
        return _json(value)

    async def control_goal(request):
        body = OperatorActionRequest.from_dict(await _json_body(request))
        try:
            value = operators.control_goal(
                request.path_params["profile_id"],
                request.path_params["goal_id"],
                action=body.action,
                revision=body.revision,
                step_id=body.step_id,
                instruction=body.instruction,
                reason=body.reason,
            )
        except LookupError as exc:
            raise ApiProblem(404, "goal_not_found", str(exc)) from exc
        except RuntimeError as exc:
            raise ApiProblem(409, "goal_conflict", str(exc)) from exc
        return _json({"goal": value})

    async def child_tasks(request):
        return _json(
            operators.child_tasks(
                request.path_params["profile_id"],
                status=request.query_params.get("status"),
                goal_id=request.query_params.get("goal_id"),
                parent_task_id=request.query_params.get("parent_task_id"),
                limit=_limit(request.query_params.get("limit")),
                cursor=request.query_params.get("cursor"),
            )
        )

    async def child_task(request):
        try:
            value = operators.child_task(
                request.path_params["profile_id"],
                request.path_params["task_id"],
            )
        except LookupError as exc:
            raise ApiProblem(404, "child_task_not_found", str(exc)) from exc
        return _json(value)

    async def control_child_task(request):
        body = OperatorActionRequest.from_dict(await _json_body(request))
        try:
            value = operators.control_child_task(
                request.path_params["profile_id"],
                request.path_params["task_id"],
                action=body.action,
                revision=body.revision,
                reason=body.reason,
            )
        except LookupError as exc:
            raise ApiProblem(404, "child_task_not_found", str(exc)) from exc
        except RuntimeError as exc:
            raise ApiProblem(409, "child_task_conflict", str(exc)) from exc
        return _json({"task": value})

    async def jobs(request):
        return _json(
            operators.jobs(
                request.path_params["profile_id"],
                adapter=request.query_params.get("adapter"),
                destination_id=request.query_params.get("destination_id"),
                status=request.query_params.get("status"),
                limit=_limit(request.query_params.get("limit")),
                include_terminal=boolean_query(
                    request.query_params.get("include_terminal"),
                    field="include_terminal",
                ),
                cursor=request.query_params.get("cursor"),
            )
        )

    async def automation_job(request):
        try:
            value = operators.automation_job(
                request.path_params["profile_id"],
                request.path_params["job_id"],
            )
        except LookupError as exc:
            raise ApiProblem(404, "automation_not_found", str(exc)) from exc
        return _json(value)

    async def control_automation(request):
        body = AutomationActionRequest.from_dict(await _json_body(request))
        try:
            value = operators.control_automation(
                request.path_params["profile_id"],
                request.path_params["job_id"],
                action=body.action,
                revision=body.revision,
                idempotency_key=body.idempotency_key,
            )
        except LookupError as exc:
            raise ApiProblem(404, "automation_not_found", str(exc)) from exc
        except RuntimeError as exc:
            raise ApiProblem(409, "automation_conflict", str(exc)) from exc
        return _json({"job": value})

    async def create_automation_webhook(request):
        try:
            value = operators.create_automation_webhook(
                request.path_params["profile_id"],
                request.path_params["job_id"],
            )
        except LookupError as exc:
            raise ApiProblem(404, "automation_not_found", str(exc)) from exc
        return _json(value, status_code=201)

    async def ingest_automation_webhook(request):
        body = await _json_body(request)
        if not isinstance(body, Mapping):
            raise ValueError("request body must be an object")
        credential = body.get("credential")
        event_id = body.get("event_id")
        payload = body.get("payload", {})
        if not isinstance(credential, str) or not credential:
            raise ValueError("credential is required")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id is required")
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be an object")
        try:
            value = operators.ingest_automation_webhook(
                request.path_params["profile_id"],
                request.path_params["trigger_id"],
                credential=credential,
                event_id=event_id,
                payload=dict(payload),
            )
        except LookupError as exc:
            raise ApiProblem(404, "automation_trigger_not_found", str(exc)) from exc
        except PermissionError as exc:
            raise ApiProblem(401, "automation_trigger_unauthorized", str(exc)) from exc
        return _json({"trigger_event": value}, status_code=202)

    async def proposals(request):
        status = request.query_params.get("status", "pending")
        if status == "all":
            status = None
        return _json(
            operators.proposals(
                request.path_params["profile_id"],
                status=status,
                limit=_limit(request.query_params.get("limit")),
                cursor=request.query_params.get("cursor"),
            )
        )

    async def decide_proposal(request):
        body = ProposalDecisionRequest.from_dict(await _json_body(request))
        try:
            result = operators.decide_proposal(
                request.path_params["profile_id"],
                request.path_params["proposal_id"],
                action=body.action,
            )
        except KeyError as exc:
            raise ApiProblem(404, "proposal_not_found", str(exc)) from exc
        return _json({"proposal": result})

    async def proposal(request):
        try:
            value = operators.proposal(
                request.path_params["profile_id"],
                request.path_params["proposal_id"],
            )
        except KeyError as exc:
            raise ApiProblem(404, "proposal_not_found", str(exc)) from exc
        return _json({"proposal": value})

    async def usage(request):
        return _json(
            operators.usage(
                request.path_params["profile_id"],
                start=parse_timestamp(
                    request.query_params.get("start"),
                    field="start",
                ),
                end=parse_timestamp(
                    request.query_params.get("end"),
                    field="end",
                ),
                resource_kind=request.query_params.get("resource_kind"),
                channel=request.query_params.get("channel"),
                conversation_id=request.query_params.get("conversation_id"),
                goal_id=request.query_params.get("goal_id"),
                job_id=request.query_params.get("job_id"),
                child_task_id=request.query_params.get("child_task_id"),
                group_by=request.query_params.get("group_by"),
                limit=_limit(request.query_params.get("limit")),
                cursor=request.query_params.get("cursor"),
            )
        )

    async def traces(request):
        return _json(
            operators.traces(
                request.path_params["profile_id"],
                limit=_limit(request.query_params.get("limit")),
                cursor=request.query_params.get("cursor"),
            )
        )

    async def trace(request):
        try:
            value = operators.trace(
                request.path_params["profile_id"],
                request.path_params["conversation_id"],
            )
        except SessionNotFoundError as exc:
            raise ApiProblem(404, "trace_not_found", str(exc)) from exc
        return _json(value)

    async def artifacts(request):
        profile_id, conversation_id = _conversation_params(request)
        try:
            result = operators.artifacts(
                profile_id,
                conversation_id,
                limit=_limit(request.query_params.get("limit")),
                cursor=request.query_params.get("cursor"),
            )
        except SessionNotFoundError as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc
        return _json(result)

    async def read_artifact(request):
        profile_id, conversation_id = _conversation_params(request)
        try:
            result = operators.read_artifact(
                profile_id,
                conversation_id,
                request.path_params["artifact_id"],
                mode=request.query_params.get("mode", "head_tail"),
                offset=integer_query(
                    request.query_params.get("offset"),
                    field="offset",
                    default=0,
                    minimum=0,
                    maximum=64 * 1024 * 1024,
                ),
                max_bytes=integer_query(
                    request.query_params.get("max_bytes"),
                    field="max_bytes",
                    default=8_192,
                    minimum=1,
                    maximum=65_536,
                ),
            )
        except SessionNotFoundError as exc:
            raise ApiProblem(404, "conversation_not_found", str(exc)) from exc
        return _json({"artifact": result})

    async def gateway_websocket(websocket):
        from chulk.server.gateway_ws import serve_gateway_websocket

        await serve_gateway_websocket(websocket, dispatcher=controller)

    routes = [
        Route("/webchat", webchat, methods=["GET"]),
        Route("/webchat/", webchat, methods=["GET"]),
        Route(
            "/webchat/assets/{name:str}",
            webchat_asset,
            methods=["GET"],
        ),
        Route("/v1/session", session_info, methods=["GET"]),
        Route("/v1/profiles", profiles, methods=["GET"]),
        Route("/v1/gateway/routes", gateway_routes, methods=["GET"]),
        Route(
            "/v1/gateway/pairings",
            create_gateway_pairing,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations",
            create_conversation,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations",
            list_conversations,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/messages",
            send_message,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/commands/{command_id:str}",
            command_status,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/events",
            conversation_events,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/cancel",
            cancel_conversation,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/turns/{turn_id:str}/plan/approve",
            approve_plan,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/turns/{turn_id:str}/plan/reject",
            reject_plan,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/permissions",
            list_permissions,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/permissions/{permission_request_id:str}",
            decide_permission,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/goals",
            goals,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/goals/{goal_id:str}",
            goal,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/goals/{goal_id:str}/actions",
            control_goal,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/tasks",
            child_tasks,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/tasks/{task_id:str}",
            child_task,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/tasks/{task_id:str}/actions",
            control_child_task,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/jobs",
            jobs,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/jobs/{job_id:str}",
            automation_job,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/jobs/{job_id:str}/actions",
            control_automation,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/jobs/{job_id:str}/webhooks",
            create_automation_webhook,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/automation-webhooks/{trigger_id:str}",
            ingest_automation_webhook,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/proposals",
            proposals,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/learning/proposals",
            proposals,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/proposals/{proposal_id:str}",
            proposal,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/learning/proposals/{proposal_id:str}",
            proposal,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/proposals/{proposal_id:str}",
            decide_proposal,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/learning/proposals/{proposal_id:str}",
            decide_proposal,
            methods=["POST"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/usage",
            usage,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/traces",
            traces,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/traces/{conversation_id:str}",
            trace,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/artifacts",
            artifacts,
            methods=["GET"],
        ),
        Route(
            "/v1/profiles/{profile_id:str}/conversations/{conversation_id:str}/artifacts/{artifact_id:str}",
            read_artifact,
            methods=["GET"],
        ),
        Route("/v1/schema", schema, methods=["GET"]),
        Route("/v1/openapi.json", schema, methods=["GET"]),
        WebSocketRoute("/v1/gateway/ws", gateway_websocket),
    ]
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={
            ApiProblem: problem_handler,
            RequestBodyTooLargeError: body_limit_handler,
            ProfileNotFoundError: not_found_handler,
            SessionNotFoundError: not_found_handler,
            ValueError: value_handler,
        },
    )
    app.state.dispatcher = controller
    app.state.token_store = credentials
    app.add_middleware(
        ControlSecurityMiddleware,
        token_store=credentials,
        allowed_origins=allowed_origins,
        max_body_bytes=max_body_bytes,
        audit_log=ControlAuditLog(config.runtime_dir / "control-audit.jsonl"),
        public_get_prefixes=("/webchat",),
    )
    return app


async def _json_body(request) -> object:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise ApiProblem(
            415,
            "unsupported_media_type",
            "request content type must be application/json",
        )
    try:
        return await request.json()
    except RequestBodyTooLargeError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ApiProblem(400, "malformed_json", "request body is not valid JSON") from exc


def _conversation_params(request) -> tuple[str, str]:
    return request.path_params["profile_id"], request.path_params["conversation_id"]


def _limit(value: str | None) -> int:
    if value is None:
        return 100
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError("limit must be an integer") from exc
    if limit < 1 or limit > 1_000:
        raise ValueError("limit must be between 1 and 1000")
    return limit


def _json(value: Mapping[str, Any], *, status_code: int = 200):
    from starlette.responses import JSONResponse

    return JSONResponse(
        {"schema_version": API_SCHEMA_VERSION, **dict(value)},
        status_code=status_code,
    )


def _sse(name: str, event_id: str, value: Mapping[str, Any]) -> bytes:
    data = json.dumps(dict(value), separators=(",", ":"), sort_keys=True)
    return f"id: {event_id}\nevent: {name}\ndata: {data}\n\n".encode()


def _webchat_headers(
    *,
    cache_control: str,
    content_security: bool = True,
) -> dict[str, str]:
    headers = {
        "Cache-Control": cache_control,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }
    if content_security:
        headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
    return headers


def _schema() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Chulk local control API",
            "version": str(API_SCHEMA_VERSION),
        },
        "paths": {
            "/v1/profiles": {"get": {"operationId": "listProfiles"}},
            "/v1/session": {"get": {"operationId": "getSession"}},
            "/v1/gateway/routes": {"get": {"operationId": "listGatewayRoutes"}},
            "/v1/gateway/pairings": {
                "post": {"operationId": "createGatewayPairing"}
            },
            "/v1/profiles/{profile_id}/conversations": {
                "get": {"operationId": "listConversations"},
                "post": {"operationId": "createConversation"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/messages": {
                "post": {"operationId": "sendMessage"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/commands/{command_id}": {
                "get": {"operationId": "getCommand"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/events": {
                "get": {"operationId": "streamEvents"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/cancel": {
                "post": {"operationId": "cancelConversation"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/turns/{turn_id}/plan/approve": {
                "post": {"operationId": "approvePlan"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/turns/{turn_id}/plan/reject": {
                "post": {"operationId": "rejectPlan"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/permissions": {
                "get": {"operationId": "listPermissions"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/permissions/{permission_request_id}": {
                "post": {"operationId": "decidePermission"}
            },
            "/v1/profiles/{profile_id}/goals": {
                "get": {"operationId": "listGoals"}
            },
            "/v1/profiles/{profile_id}/goals/{goal_id}": {
                "get": {"operationId": "getGoal"}
            },
            "/v1/profiles/{profile_id}/goals/{goal_id}/actions": {
                "post": {"operationId": "controlGoal"}
            },
            "/v1/profiles/{profile_id}/tasks": {
                "get": {"operationId": "listChildTasks"}
            },
            "/v1/profiles/{profile_id}/tasks/{task_id}": {
                "get": {"operationId": "getChildTask"}
            },
            "/v1/profiles/{profile_id}/tasks/{task_id}/actions": {
                "post": {"operationId": "controlChildTask"}
            },
            "/v1/profiles/{profile_id}/jobs": {
                "get": {"operationId": "listJobs"}
            },
            "/v1/profiles/{profile_id}/jobs/{job_id}": {
                "get": {"operationId": "getAutomationJob"}
            },
            "/v1/profiles/{profile_id}/jobs/{job_id}/actions": {
                "post": {"operationId": "controlAutomationJob"}
            },
            "/v1/profiles/{profile_id}/jobs/{job_id}/webhooks": {
                "post": {"operationId": "createAutomationWebhook"}
            },
            "/v1/profiles/{profile_id}/automation-webhooks/{trigger_id}": {
                "post": {"operationId": "ingestAutomationWebhook"}
            },
            "/v1/profiles/{profile_id}/proposals": {
                "get": {"operationId": "listProposals"}
            },
            "/v1/profiles/{profile_id}/proposals/{proposal_id}": {
                "get": {"operationId": "getProposal"},
                "post": {"operationId": "decideProposal"}
            },
            "/v1/profiles/{profile_id}/learning/proposals": {
                "get": {"operationId": "listLearningProposals"}
            },
            "/v1/profiles/{profile_id}/learning/proposals/{proposal_id}": {
                "get": {"operationId": "getLearningProposal"},
                "post": {"operationId": "decideLearningProposal"}
            },
            "/v1/profiles/{profile_id}/usage": {
                "get": {"operationId": "queryUsage"}
            },
            "/v1/profiles/{profile_id}/traces": {
                "get": {"operationId": "listTraceMetadata"}
            },
            "/v1/profiles/{profile_id}/traces/{conversation_id}": {
                "get": {"operationId": "getTraceSummary"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/artifacts": {
                "get": {"operationId": "listArtifacts"}
            },
            "/v1/profiles/{profile_id}/conversations/{conversation_id}/artifacts/{artifact_id}": {
                "get": {"operationId": "readArtifact"}
            },
        },
        "components": {
            "securitySchemes": {
                "controlToken": {"type": "http", "scheme": "bearer"}
            },
            "schemas": {
                "AgentEvent": {
                    "type": "object",
                    "required": (
                        "name",
                        "schema_version",
                        "conversation_id",
                        "payload",
                    ),
                },
                "ApiError": {
                    "type": "object",
                    "required": ("schema_version", "error"),
                },
            },
        },
        "security": ({"controlToken": ()},),
        "x-chulk-websocket-path": "/v1/gateway/ws",
        "x-chulk-excluded": ("raw_traces", "credentials", "prompt_dumps"),
    }


__all__ = [
    "ApiProblem",
    "ServerDependencyError",
    "create_control_app",
]
