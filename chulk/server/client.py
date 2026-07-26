"""Small authenticated client for the local operator control API."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import uuid4


class ControlApiError(RuntimeError):
    """Stable client error decoded from a control-plane response."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.details = dict(details or {})
        super().__init__(message)


class ControlApiClient:
    """Synchronous, dependency-free client for bounded operator resources."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 30.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.timeout_seconds = timeout_seconds
        self._opener = opener
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("control API base_url must use http or https")
        if not self.token:
            raise ValueError("control API token cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("control API timeout must be greater than zero")

    def list_profiles(self) -> dict[str, Any]:
        return self._get("/v1/profiles")

    def list_conversations(
        self,
        profile_id: str,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/conversations",
            limit=limit,
        )

    def create_conversation(
        self,
        profile_id: str,
        *,
        conversation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/conversations",
            {
                **(
                    {"conversation_id": conversation_id}
                    if conversation_id is not None
                    else {}
                ),
                **({"metadata": dict(metadata)} if metadata is not None else {}),
            },
        )

    def send_message(
        self,
        profile_id: str,
        conversation_id: str,
        message: str,
        *,
        mode: str = "run",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/messages",
            {
                "message": message,
                "mode": mode,
                "idempotency_key": idempotency_key or uuid4().hex,
            },
        )

    def get_command(
        self,
        profile_id: str,
        conversation_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/commands/{_segment(command_id)}"
        )

    def list_events(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        after: str | None = None,
    ) -> dict[str, Any]:
        """Read one retained event page for reconnect and deterministic UIs."""
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/events",
            after=after,
            follow="false",
        )

    def iter_events(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        after: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the authenticated SSE stream, resuming after a retained id."""
        path = (
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/events"
        )
        request = Request(
            f"{self.base_url}{path}",
            method="GET",
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.token}",
                **({"Last-Event-ID": after} if after is not None else {}),
            },
        )
        try:
            with self._opener(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                yield from _iter_sse(response)
        except HTTPError as exc:
            payload = _decode_error(exc.read(), status=exc.code)
            raise ControlApiError(
                exc.code,
                payload["code"],
                payload["message"],
                details=payload.get("details"),
            ) from exc
        except URLError as exc:
            raise ControlApiError(
                0,
                "connection_error",
                f"control API event stream failed: {exc.reason}",
            ) from exc

    def cancel_conversation(
        self,
        profile_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/cancel",
            {},
        )

    def decide_plan(
        self,
        profile_id: str,
        conversation_id: str,
        turn_id: str,
        *,
        action: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if action not in {"approve", "reject"}:
            raise ValueError("plan action must be approve or reject")
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/turns/{_segment(turn_id)}/plan/{action}",
            {"idempotency_key": idempotency_key or uuid4().hex},
        )

    def list_permissions(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        status: str | None = "pending",
        limit: int = 100,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/permissions",
            status=status,
            limit=limit,
        )

    def list_profile_permissions(
        self,
        profile_id: str,
        *,
        status: str | None = "pending",
        limit: int = 100,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/permissions",
            status=status,
            limit=limit,
        )

    def decide_permission(
        self,
        profile_id: str,
        conversation_id: str,
        permission_request_id: str,
        *,
        decision: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if decision not in {"allow", "deny"}:
            raise ValueError("permission decision must be allow or deny")
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/permissions/"
            f"{_segment(permission_request_id)}",
            {
                "decision": decision,
                "idempotency_key": idempotency_key or uuid4().hex,
                **({"reason": reason} if reason is not None else {}),
            },
        )

    def list_goals(
        self,
        profile_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/goals",
            status=status,
            limit=limit,
            cursor=cursor,
        )

    def get_goal(self, profile_id: str, goal_id: str) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/goals/{_segment(goal_id)}"
        )

    def control_goal(
        self,
        profile_id: str,
        goal_id: str,
        *,
        action: str,
        revision: int,
        step_id: str | None = None,
        instruction: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/goals/{_segment(goal_id)}/actions",
            _revision_body(
                action,
                revision,
                step_id=step_id,
                instruction=instruction,
                reason=reason,
            ),
        )

    def list_tasks(
        self,
        profile_id: str,
        *,
        status: str | None = None,
        goal_id: str | None = None,
        parent_task_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/tasks",
            status=status,
            goal_id=goal_id,
            parent_task_id=parent_task_id,
            limit=limit,
            cursor=cursor,
        )

    def get_task(self, profile_id: str, task_id: str) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/tasks/{_segment(task_id)}"
        )

    def control_task(
        self,
        profile_id: str,
        task_id: str,
        *,
        action: str,
        revision: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/tasks/{_segment(task_id)}/actions",
            _revision_body(
                action,
                revision,
                reason=reason,
            ),
        )

    def list_jobs(
        self,
        profile_id: str,
        *,
        status: str | None = None,
        adapter: str | None = None,
        destination_id: str | None = None,
        include_terminal: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/jobs",
            status=status,
            adapter=adapter,
            destination_id=destination_id,
            include_terminal=str(include_terminal).lower(),
            limit=limit,
            cursor=cursor,
        )

    def get_job(self, profile_id: str, job_id: str) -> dict[str, Any]:
        return self._get(f"/v1/profiles/{_segment(profile_id)}/jobs/{_segment(job_id)}")

    def control_job(
        self,
        profile_id: str,
        job_id: str,
        *,
        action: str,
        revision: int,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/jobs/{_segment(job_id)}/actions",
            _revision_body(
                action,
                revision,
                idempotency_key=idempotency_key or uuid4().hex,
            ),
        )

    def list_proposals(
        self,
        profile_id: str,
        *,
        status: str = "pending",
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/learning/proposals",
            status=status,
            limit=limit,
            cursor=cursor,
        )

    def get_proposal(
        self,
        profile_id: str,
        proposal_id: str,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/learning/proposals/"
            f"{_segment(proposal_id)}"
        )

    def decide_proposal(
        self,
        profile_id: str,
        proposal_id: str,
        *,
        action: str,
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/profiles/{_segment(profile_id)}/learning/proposals/"
            f"{_segment(proposal_id)}",
            {
                "action": action,
            },
        )

    def query_usage(
        self,
        profile_id: str,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        resource_kind: str | None = None,
        channel: str | None = None,
        conversation_id: str | None = None,
        goal_id: str | None = None,
        job_id: str | None = None,
        child_task_id: str | None = None,
        group_by: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/usage",
            start=_timestamp(start),
            end=_timestamp(end),
            resource_kind=resource_kind,
            channel=channel,
            conversation_id=conversation_id,
            goal_id=goal_id,
            job_id=job_id,
            child_task_id=child_task_id,
            group_by=group_by,
            limit=limit,
            cursor=cursor,
        )

    def list_traces(
        self,
        profile_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/traces",
            limit=limit,
            cursor=cursor,
        )

    def get_trace(
        self,
        profile_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/traces/{_segment(conversation_id)}"
        )

    def list_artifacts(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/artifacts",
            limit=limit,
            cursor=cursor,
        )

    def read_artifact(
        self,
        profile_id: str,
        conversation_id: str,
        artifact_id: str,
        *,
        mode: str = "head_tail",
        offset: int = 0,
        max_bytes: int = 8_192,
    ) -> dict[str, Any]:
        return self._get(
            f"/v1/profiles/{_segment(profile_id)}/conversations/"
            f"{_segment(conversation_id)}/artifacts/{_segment(artifact_id)}",
            mode=mode,
            offset=offset,
            max_bytes=max_bytes,
        )

    def _get(self, path: str, **query: object) -> dict[str, Any]:
        return self._request("GET", path, query=query)

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body=body)

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, object] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoded_query = urlencode(
            {
                key: str(value)
                for key, value in (query or {}).items()
                if value is not None
            }
        )
        url = f"{self.base_url}{path}"
        if encoded_query:
            url = f"{url}?{encoded_query}"
        data = (
            json.dumps(dict(body), separators=(",", ":")).encode()
            if body is not None
            else None
        )
        request = Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with self._opener(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return _decode_response(response.read(), status=response.status)
        except HTTPError as exc:
            payload = _decode_error(exc.read(), status=exc.code)
            raise ControlApiError(
                exc.code,
                payload["code"],
                payload["message"],
                details=payload.get("details"),
            ) from exc
        except URLError as exc:
            raise ControlApiError(
                0,
                "connection_error",
                f"control API request failed: {exc.reason}",
            ) from exc


def _revision_body(
    action: str,
    revision: int,
    *,
    idempotency_key: str | None = None,
    step_id: str | None = None,
    instruction: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "revision": revision,
        **(
            {"idempotency_key": idempotency_key or uuid4().hex}
            if idempotency_key is not None
            else {}
        ),
        **({"step_id": step_id} if step_id is not None else {}),
        **({"instruction": instruction} if instruction is not None else {}),
        **({"reason": reason} if reason is not None else {}),
    }


def _segment(value: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError("control API resource id cannot be empty")
    return quote(clean, safe="")


def _timestamp(value: datetime | str | None) -> str | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("control API timestamps must be timezone-aware")
        return value.isoformat()
    return value


def _decode_response(raw: bytes, *, status: int) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlApiError(
            status,
            "invalid_response",
            "control API response is not valid JSON",
        ) from exc
    if not isinstance(value, dict):
        raise ControlApiError(
            status,
            "invalid_response",
            "control API response must be an object",
        )
    return value


def _decode_error(raw: bytes, *, status: int) -> dict[str, Any]:
    try:
        value = _decode_response(raw, status=status)
    except ControlApiError:
        return {
            "code": "http_error",
            "message": f"control API returned HTTP {status}",
        }
    error = value.get("error")
    if not isinstance(error, Mapping):
        return {
            "code": "http_error",
            "message": f"control API returned HTTP {status}",
        }
    return {
        "code": str(error.get("code") or "http_error"),
        "message": str(error.get("message") or f"HTTP {status}"),
        **(
            {"details": dict(error["details"])}
            if isinstance(error.get("details"), Mapping)
            else {}
        ),
    }


def _iter_sse(response: Any) -> Iterator[dict[str, Any]]:
    event_name = "message"
    event_id: str | None = None
    data_lines: list[str] = []
    data_bytes = 0
    for raw_line in response:
        if not isinstance(raw_line, bytes):
            raise ControlApiError(
                200,
                "invalid_event_stream",
                "control API event stream yielded a non-byte line",
            )
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                try:
                    value = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as exc:
                    raise ControlApiError(
                        200,
                        "invalid_event_stream",
                        "control API event stream contains invalid JSON",
                    ) from exc
                if not isinstance(value, dict):
                    raise ControlApiError(
                        200,
                        "invalid_event_stream",
                        "control API event payload must be an object",
                    )
                yield {
                    "id": event_id,
                    "event": event_name,
                    "data": value,
                }
            event_name = "message"
            event_id = None
            data_lines = []
            data_bytes = 0
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_bytes += len(raw_line)
            if data_bytes > 1_000_000:
                raise ControlApiError(
                    200,
                    "event_too_large",
                    "control API event exceeds the client safety limit",
                )
            data_lines.append(value)


__all__ = ["ControlApiClient", "ControlApiError"]
