"""Immutable authority context for local and hosted executions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any
from uuid import uuid4


_MAX_SCOPE_VALUE_CHARS = 256


class ExecutionScopeError(ValueError):
    """Raised when an execution scope is malformed or broadens authority."""


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """Host-supplied identity and authority boundary for one execution."""

    tenant_id: str
    workspace_id: str
    actor_id: str | None
    agent_id: str
    agent_version: str
    run_id: str
    conversation_id: str | None = None
    trigger_id: str | None = None
    channel_id: str | None = None
    parent_run_id: str | None = None
    grants: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "workspace_id",
            "agent_id",
            "agent_version",
            "run_id",
        ):
            object.__setattr__(self, name, _required_value(name, getattr(self, name)))
        for name in (
            "actor_id",
            "conversation_id",
            "trigger_id",
            "channel_id",
            "parent_run_id",
        ):
            object.__setattr__(self, name, _optional_value(name, getattr(self, name)))
        clean_grants = frozenset(_required_value("grant", grant) for grant in self.grants)
        object.__setattr__(self, "grants", clean_grants)

    @classmethod
    def local(
        cls,
        *,
        agent_id: str = "local",
        agent_version: str = "development",
        run_id: str | None = None,
        conversation_id: str | None = None,
        profile_id: str = "default",
    ) -> "ExecutionScope":
        """Create the explicit scope used by backward-compatible local mode."""
        return cls(
            tenant_id="local",
            workspace_id=profile_id,
            actor_id=profile_id,
            agent_id=agent_id,
            agent_version=agent_version,
            run_id=run_id or f"run_{uuid4().hex}",
            conversation_id=conversation_id,
            grants=frozenset({"local"}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["grants"] = sorted(self.grants)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionScope":
        """Parse a serialized host scope and validate every field."""
        if not isinstance(payload, dict):
            raise ExecutionScopeError("execution scope must be a dictionary")
        known = {
            "tenant_id",
            "workspace_id",
            "actor_id",
            "agent_id",
            "agent_version",
            "run_id",
            "conversation_id",
            "trigger_id",
            "channel_id",
            "parent_run_id",
            "grants",
        }
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ExecutionScopeError(
                f"execution scope contains unknown fields: {', '.join(unknown)}"
            )
        grants = payload.get("grants", ())
        if isinstance(grants, str) or not isinstance(grants, (list, tuple, set, frozenset)):
            raise ExecutionScopeError("execution scope grants must be a collection")
        try:
            return cls(
                tenant_id=payload["tenant_id"],
                workspace_id=payload["workspace_id"],
                actor_id=payload.get("actor_id"),
                agent_id=payload["agent_id"],
                agent_version=payload["agent_version"],
                run_id=payload["run_id"],
                conversation_id=payload.get("conversation_id"),
                trigger_id=payload.get("trigger_id"),
                channel_id=payload.get("channel_id"),
                parent_run_id=payload.get("parent_run_id"),
                grants=frozenset(grants),
            )
        except KeyError as exc:
            raise ExecutionScopeError(
                f"execution scope is missing required field: {exc.args[0]}"
            ) from exc

    @property
    def canonical_json(self) -> str:
        """Return a collision-safe stable encoding for storage and signatures."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def key(self) -> str:
        """Return an opaque collision-safe storage key for the complete scope."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def with_conversation(self, conversation_id: str) -> "ExecutionScope":
        """Bind a conversation once without changing the authority boundary."""
        clean_id = _required_value("conversation_id", conversation_id)
        if self.conversation_id not in {None, clean_id}:
            raise ExecutionScopeError(
                "execution scope conversation_id does not match the runtime conversation"
            )
        return replace(self, conversation_id=clean_id)

    def child(
        self,
        *,
        run_id: str,
        agent_id: str | None = None,
        agent_version: str | None = None,
        conversation_id: str | None = None,
        trigger_id: str | None = None,
        channel_id: str | None = None,
        grants: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    ) -> "ExecutionScope":
        """Create a descendant scope that can only preserve or narrow grants."""
        child_grants = self.grants if grants is None else frozenset(grants)
        if not child_grants.issubset(self.grants):
            raise ExecutionScopeError("child execution scope cannot broaden grants")
        return ExecutionScope(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            actor_id=self.actor_id,
            agent_id=agent_id or self.agent_id,
            agent_version=agent_version or self.agent_version,
            run_id=run_id,
            conversation_id=conversation_id,
            trigger_id=trigger_id,
            channel_id=channel_id or self.channel_id,
            parent_run_id=self.run_id,
            grants=child_grants,
        )

    def assert_same_authority(self, other: "ExecutionScope") -> None:
        """Reject a persisted or resumed scope from another authority boundary."""
        if not isinstance(other, ExecutionScope):
            raise ExecutionScopeError("expected an ExecutionScope")
        fields = ("tenant_id", "workspace_id", "agent_id", "agent_version")
        mismatches = [name for name in fields if getattr(self, name) != getattr(other, name)]
        if mismatches:
            raise ExecutionScopeError(
                "execution scope authority mismatch: " + ", ".join(mismatches)
            )
        if not other.grants.issubset(self.grants):
            raise ExecutionScopeError("persisted execution scope broadens host grants")

    def assert_resumable(self, persisted: "ExecutionScope") -> None:
        """Verify an exact persisted run boundary before state is resumed."""
        self.assert_same_authority(persisted)
        fields = ("actor_id", "run_id", "conversation_id")
        mismatches = [
            name
            for name in fields
            if getattr(self, name) != getattr(persisted, name)
        ]
        if mismatches:
            raise ExecutionScopeError(
                "execution scope resume mismatch: " + ", ".join(mismatches)
            )


def _required_value(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ExecutionScopeError(f"execution scope {name} must be a string")
    clean = value.strip()
    if not clean:
        raise ExecutionScopeError(f"execution scope {name} cannot be empty")
    if len(clean) > _MAX_SCOPE_VALUE_CHARS or "\x00" in clean:
        raise ExecutionScopeError(f"execution scope {name} is invalid")
    return clean


def _optional_value(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _required_value(name, value)
