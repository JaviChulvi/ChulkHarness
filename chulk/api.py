"""Public programmable API for Chulk."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Callable

from chulk.config import (
    Config,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MODEL,
    LLMFallbackProviderConfig,
    load_config,
)
from chulk.core.context import TurnContextSection
from chulk.core import Agent, TraceEvent
from chulk.llm import LLMClient
from chulk.mcp import MCPServerConfig, build_mcp_server_config
from chulk.runtime import create_agent as create_runtime_agent
from chulk.tools import ToolExecutionContext
from chulk.tools.permissions import PermissionDecision, PermissionDecisionRecord, PermissionRequest


EventCallback = Callable[["AgentEvent"], None]
DeltaCallback = Callable[[str], None]
PermissionCallback = Callable[[PermissionRequest, PermissionDecisionRecord], PermissionDecision | bool]


@dataclass(frozen=True)
class AgentPreset:
    """Reusable collection of prompt, tools, skills, and default behavior."""

    system_prompt: str | None = None
    tools: tuple[object, ...] = field(default_factory=tuple)
    skills: tuple[object, ...] = field(default_factory=tuple)

    @classmethod
    def chat(cls, *, system_prompt: str | None = None) -> "AgentPreset":
        """Return a no-tool, no-skill preset for plain chat embedding."""
        return cls(system_prompt=system_prompt, tools=(), skills=())


@dataclass(frozen=True)
class AgentEvent:
    """One public event emitted during an SDK agent run."""

    type: str
    payload: dict

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": dict(self.payload)}


@dataclass(frozen=True)
class PlanSnapshot:
    """Public snapshot of a plan at the end of an SDK call."""

    summary: str
    status: str
    steps: list[dict]
    created_at: str | None = None
    approved_at: str | None = None
    rejected_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "status": self.status,
            "steps": list(self.steps),
            "created_at": self.created_at,
            "approved_at": self.approved_at,
            "rejected_at": self.rejected_at,
        }


@dataclass(frozen=True)
class RunResult:
    """Structured result for one SDK agent turn."""

    content: str
    status: str
    turn_id: str | None
    conversation_id: str
    trace_path: Path | None
    usage: dict | None = None
    cost: dict | None = None
    context_report: dict | None = None
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    loaded_skill_names: list[str] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    plan: PlanSnapshot | None = None
    extension_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "usage": self.usage,
            "cost": self.cost,
            "context_report": self.context_report,
            "tool_calls": list(self.tool_calls),
            "observations": list(self.observations),
            "loaded_skill_names": list(self.loaded_skill_names),
            "loaded_memory_ids": list(self.loaded_memory_ids),
            "errors": list(self.errors),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "extension_metadata": self.extension_metadata,
        }


@dataclass(frozen=True)
class PlanResult:
    """Structured result for a planning turn awaiting approval or rejection."""

    content: str
    status: str
    plan: PlanSnapshot | None
    turn_id: str | None
    conversation_id: str
    trace_path: Path | None
    context_report: dict | None = None
    loaded_skill_names: list[str] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "context_report": self.context_report,
            "loaded_skill_names": list(self.loaded_skill_names),
            "loaded_memory_ids": list(self.loaded_memory_ids),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class AgentConfig:
    """Programmatic SDK configuration with environment fallback."""

    project_root: str | Path | None = None
    runtime_dir: str | Path | None = None
    provider: str | None = None
    model: str | None = None
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str | None = None
    local_api_key: str | None = None
    local_base_url: str | None = None
    permission_profile: str | None = None
    store_path: str | Path | None = None
    traces_dir: str | Path | None = None
    skills_dir: str | Path | None = None
    mcp_config_path: str | Path | None = None
    mcp_servers: Iterable[MCPServerConfig] | None = None
    llm_fallback_providers: Iterable[LLMFallbackProviderConfig] | None = None
    history_limit: int | None = None
    max_tool_calls_per_turn: int | None = None
    max_skills_per_turn: int | None = None
    max_skill_content_chars: int | None = None
    shell_timeout_seconds: int | None = None
    llm_timeout_seconds: float | None = None
    llm_max_retries: int | None = None
    trace_max_prompt_chars: int | None = None
    max_observation_chars: int | None = None
    max_tool_stdout_chars: int | None = None
    max_tool_stderr_chars: int | None = None
    max_reflection_attempts: int | None = None

    def __post_init__(self) -> None:
        if self.mcp_servers is not None:
            object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        if self.llm_fallback_providers is not None:
            object.__setattr__(self, "llm_fallback_providers", tuple(self.llm_fallback_providers))

    def with_overrides(self, **overrides: Any) -> "AgentConfig":
        """Return a copy with any AgentConfig field overridden."""
        return replace(self, **overrides)

    @staticmethod
    def fallback_provider(provider: str, model: str) -> LLMFallbackProviderConfig:
        """Create one provider fallback entry for AgentConfig."""
        return LLMFallbackProviderConfig(provider=provider, model=model)

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path | None = None,
        runtime_dir: str | Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        permission_profile: str | None = None,
        **overrides: Any,
    ) -> "AgentConfig":
        """Create SDK config from the current environment with optional overrides."""
        values: dict[str, Any] = {
            "project_root": Path.cwd() if project_root is None else project_root,
            "runtime_dir": runtime_dir,
            "provider": provider or os.getenv("CHULK_LLM_PROVIDER") or None,
            "model": model or os.getenv("CHULK_MODEL") or None,
            "permission_profile": permission_profile or os.getenv("CHULK_PERMISSION_PROFILE") or None,
        }
        values.update(overrides)
        return cls(**{key: value for key, value in values.items() if value is not None})

    @classmethod
    def openai(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for OpenAI-backed agents."""
        values = dict(kwargs)
        if api_key is not None:
            values["openai_api_key"] = api_key
        return cls.from_env(provider="openai", model=model or DEFAULT_MODEL, **values)

    @classmethod
    def deepseek(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for DeepSeek-backed agents."""
        values = dict(kwargs)
        if api_key is not None:
            values["deepseek_api_key"] = api_key
        if base_url is not None:
            values["deepseek_base_url"] = base_url
        return cls.from_env(
            provider="deepseek",
            model=model or DEFAULT_DEEPSEEK_MODEL,
            **values,
        )

    @classmethod
    def local(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for local OpenAI-compatible providers."""
        values = dict(kwargs)
        if api_key is not None:
            values["local_api_key"] = api_key
        if base_url is not None:
            values["local_base_url"] = base_url
        return cls.from_env(
            provider="local",
            model=model or DEFAULT_LOCAL_MODEL,
            **values,
        )

    def to_config(self) -> Config:
        """Build the internal runtime config."""
        env = dict(os.environ)
        _set_env(env, "CHULK_PROJECT_ROOT", self.project_root)
        _set_env(env, "CHULK_LLM_PROVIDER", self.provider)
        _set_env(env, "CHULK_MODEL", self.model)
        _set_env(env, "OPENAI_API_KEY", self.openai_api_key)
        _set_env(env, "CHULK_DEEPSEEK_API_KEY", self.deepseek_api_key)
        _set_env(env, "CHULK_DEEPSEEK_BASE_URL", self.deepseek_base_url)
        _set_env(env, "CHULK_LOCAL_API_KEY", self.local_api_key)
        _set_env(env, "CHULK_LOCAL_BASE_URL", self.local_base_url)
        _set_env(env, "CHULK_PERMISSION_PROFILE", self.permission_profile)
        _set_env(env, "CHULK_MCP_CONFIG", self.mcp_config_path)
        _set_env(env, "CHULK_HISTORY_LIMIT", self.history_limit)
        _set_env(env, "CHULK_MAX_TOOL_CALLS_PER_TURN", self.max_tool_calls_per_turn)
        _set_env(env, "CHULK_MAX_SKILLS_PER_TURN", self.max_skills_per_turn)
        _set_env(env, "CHULK_MAX_SKILL_CONTENT_CHARS", self.max_skill_content_chars)
        _set_env(env, "CHULK_SHELL_TIMEOUT_SECONDS", self.shell_timeout_seconds)
        _set_env(env, "CHULK_LLM_TIMEOUT_SECONDS", self.llm_timeout_seconds)
        _set_env(env, "CHULK_LLM_MAX_RETRIES", self.llm_max_retries)
        _set_env(env, "CHULK_TRACE_MAX_PROMPT_CHARS", self.trace_max_prompt_chars)
        _set_env(env, "CHULK_MAX_OBSERVATION_CHARS", self.max_observation_chars)
        _set_env(env, "CHULK_MAX_TOOL_STDOUT_CHARS", self.max_tool_stdout_chars)
        _set_env(env, "CHULK_MAX_TOOL_STDERR_CHARS", self.max_tool_stderr_chars)
        _set_env(env, "CHULK_MAX_REFLECTION_ATTEMPTS", self.max_reflection_attempts)
        config = load_config(env)
        updates: dict[str, object] = {}
        runtime_dir = Path(self.runtime_dir).resolve() if self.runtime_dir is not None else None
        if runtime_dir is not None:
            if self.store_path is None:
                updates["store_path"] = runtime_dir / "store.sqlite"
            if self.traces_dir is None:
                updates["traces_dir"] = runtime_dir / "traces"
        if self.store_path is not None:
            updates["store_path"] = Path(self.store_path).resolve()
        if self.traces_dir is not None:
            updates["traces_dir"] = Path(self.traces_dir).resolve()
        if self.skills_dir is not None:
            updates["skills_dir"] = Path(self.skills_dir).resolve()
        if self.mcp_servers is not None:
            updates["mcp_servers"] = tuple(self.mcp_servers)
        if self.llm_fallback_providers is not None:
            updates["llm_fallback_providers"] = tuple(self.llm_fallback_providers)
        return replace(config, **updates) if updates else config


class MCP:
    """Public MCP server builders."""

    @staticmethod
    def streamable_http(
        *,
        label: str,
        server_url: str,
        server_description: str = "",
        allowed_tools: Iterable[str] = (),
        authorization: str | None = None,
        authorization_env: str | None = None,
        approval: str = "always",
        defer_loading: bool = False,
    ) -> MCPServerConfig:
        return build_mcp_server_config(
            label=label,
            server_url=server_url,
            server_description=server_description,
            allowed_tools=allowed_tools,
            authorization=authorization,
            authorization_env=authorization_env,
            approval=approval,
            defer_loading=defer_loading,
        )


class AgentHandle:
    """Small ergonomic wrapper around the explicit core Agent runtime."""

    def __init__(self, runtime: Agent, *, on_event: EventCallback | None = None) -> None:
        self.runtime = runtime
        self._base_event_callback = runtime.event_callback
        self._on_event = on_event
        self._active_on_event: EventCallback | None = None
        self._active_on_delta: DeltaCallback | None = None
        self.runtime.event_callback = self._dispatch_event

    @property
    def state(self):
        return self.runtime.state

    @property
    def conversation_id(self) -> str:
        return self.runtime.state.conversation_id

    @property
    def trace_path(self) -> Path | None:
        trace_logger = getattr(self.runtime, "trace_logger", None)
        return getattr(trace_logger, "path", None)

    @property
    def tool_registry(self):
        return self.runtime.tool_registry

    @property
    def skill_registry(self):
        return self.runtime.skill_registry

    def run(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        """Run one normal agent turn, optionally receiving streamed answer deltas."""
        return self.run_result(
            message,
            on_delta=on_delta,
            on_event=on_event,
            context_sections=context_sections,
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
            tool_context=tool_context,
        ).content

    def run_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> RunResult:
        """Run one normal agent turn and return structured SDK metadata."""
        content = self._with_callbacks(
            lambda: self.runtime.run_turn(
                message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            ),
            on_delta=on_delta,
            on_event=on_event,
        )
        return self._run_result(content)

    def __call__(self, message: str) -> str:
        return self.run(message)

    def plan(self, message: str) -> str:
        """Run one planned turn that pauses for approval before mutation."""
        return self.plan_result(message).content

    def plan_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> PlanResult:
        """Run one planned turn and return the created plan snapshot."""
        content = self._with_callbacks(
            lambda: self.runtime.run_planned_turn(message),
            on_delta=on_delta,
            on_event=on_event,
        )
        return self._plan_result(content)

    def approve(self) -> str:
        """Approve and continue a pending plan."""
        return self.approve_result().content

    def approve_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        """Approve a pending plan and return the execution result."""
        has_pending_plan = self.runtime.has_pending_plan()
        content = self._with_callbacks(lambda: self.runtime.approve_plan(), on_delta=on_delta, on_event=on_event)
        if not has_pending_plan:
            return self._no_pending_plan_result(content)
        return self._run_result(content)

    def reject(self) -> str:
        """Reject a pending plan."""
        return self.reject_result().content

    def reject_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        """Reject a pending plan and return the rejection result."""
        has_pending_plan = self.runtime.has_pending_plan()
        content = self._with_callbacks(lambda: self.runtime.reject_plan(), on_delta=on_delta, on_event=on_event)
        if not has_pending_plan:
            return self._no_pending_plan_result(content)
        return self._run_result(content)

    def _with_callbacks(
        self,
        call: Callable[[], str],
        *,
        on_delta: DeltaCallback | None,
        on_event: EventCallback | None,
    ) -> str:
        previous_on_delta = self._active_on_delta
        previous_on_event = self._active_on_event
        self._active_on_delta = on_delta
        self._active_on_event = on_event
        try:
            return call()
        finally:
            self._active_on_delta = previous_on_delta
            self._active_on_event = previous_on_event

    def _dispatch_event(self, event_type: str, payload: dict) -> None:
        if self._base_event_callback is not None:
            self._base_event_callback(event_type, payload)
        event = AgentEvent(event_type, payload)
        if self._on_event is not None:
            self._on_event(event)
        if self._active_on_event is not None:
            self._active_on_event(event)
        if event_type == TraceEvent.MODEL_STREAM_DELTA and self._active_on_delta is not None:
            text = payload.get("text")
            if isinstance(text, str) and text:
                self._active_on_delta(text)

    def _last_turn(self):
        return self.runtime.state.turns[-1] if self.runtime.state.turns else None

    def _run_result(self, content: str) -> RunResult:
        turn = self._last_turn()
        usage_totals = turn.model_usage_totals if turn is not None else self.runtime.state.last_usage_report or {}
        return RunResult(
            content=content,
            status=turn.status if turn is not None else "unknown",
            turn_id=turn.turn_id if turn is not None else self.runtime.state.current_turn_id,
            conversation_id=self.conversation_id,
            trace_path=self.trace_path,
            usage=usage_totals.get("usage") if isinstance(usage_totals, dict) else None,
            cost=usage_totals.get("cost") if isinstance(usage_totals, dict) else None,
            context_report=turn.context_reports[-1] if turn is not None and turn.context_reports else self.runtime.state.last_context_report,
            tool_calls=[record.to_dict() for record in turn.tool_calls] if turn is not None else [],
            observations=[record.to_dict() for record in turn.observations] if turn is not None else [],
            loaded_skill_names=list(turn.loaded_skill_names) if turn is not None else list(self.runtime.state.loaded_skill_names),
            loaded_memory_ids=list(turn.loaded_memory_ids) if turn is not None else list(self.runtime.state.loaded_memory_ids),
            errors=list(turn.errors) if turn is not None else list(self.runtime.state.errors),
            plan=_plan_snapshot(turn.active_plan if turn is not None else self.runtime.state.active_plan),
            extension_metadata=turn.extension_metadata if turn is not None else {},
        )

    def _no_pending_plan_result(self, content: str) -> RunResult:
        return RunResult(
            content=content,
            status="no_pending_plan",
            turn_id=None,
            conversation_id=self.conversation_id,
            trace_path=self.trace_path,
        )

    def _plan_result(self, content: str) -> PlanResult:
        turn = self._last_turn()
        return PlanResult(
            content=content,
            status=turn.status if turn is not None else "unknown",
            plan=_plan_snapshot(turn.active_plan if turn is not None else self.runtime.state.active_plan),
            turn_id=turn.turn_id if turn is not None else self.runtime.state.current_turn_id,
            conversation_id=self.conversation_id,
            trace_path=self.trace_path,
            context_report=turn.context_reports[-1] if turn is not None and turn.context_reports else self.runtime.state.last_context_report,
            loaded_skill_names=list(turn.loaded_skill_names) if turn is not None else list(self.runtime.state.loaded_skill_names),
            loaded_memory_ids=list(turn.loaded_memory_ids) if turn is not None else list(self.runtime.state.loaded_memory_ids),
            errors=list(turn.errors) if turn is not None else list(self.runtime.state.errors),
        )


class AsyncAgentHandle:
    """Async wrapper around the synchronous SDK handle."""

    def __init__(self, handle: AgentHandle) -> None:
        self.handle = handle

    @property
    def runtime(self) -> Agent:
        return self.handle.runtime

    @property
    def state(self):
        return self.handle.state

    @property
    def conversation_id(self) -> str:
        return self.handle.conversation_id

    @property
    def trace_path(self) -> Path | None:
        return self.handle.trace_path

    @property
    def tool_registry(self):
        return self.handle.tool_registry

    @property
    def skill_registry(self):
        return self.handle.skill_registry

    async def run(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        return (
            await self.run_result(
                message,
                on_delta=on_delta,
                on_event=on_event,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        ).content

    async def run_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> RunResult:
        previous_on_delta = self.handle._active_on_delta
        previous_on_event = self.handle._active_on_event
        self.handle._active_on_delta = on_delta
        self.handle._active_on_event = on_event
        try:
            content = await self.runtime.run_turn_async(
                message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        finally:
            self.handle._active_on_delta = previous_on_delta
            self.handle._active_on_event = previous_on_event
        return self.handle._run_result(content)

    async def plan(self, message: str) -> str:
        return await asyncio.to_thread(self.handle.plan, message)

    async def plan_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> PlanResult:
        return await asyncio.to_thread(self.handle.plan_result, message, on_delta=on_delta, on_event=on_event)

    async def approve(self) -> str:
        return (await self.approve_result()).content

    async def approve_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        has_pending_plan = self.runtime.has_pending_plan()
        previous_on_delta = self.handle._active_on_delta
        previous_on_event = self.handle._active_on_event
        self.handle._active_on_delta = on_delta
        self.handle._active_on_event = on_event
        try:
            content = await self.runtime.approve_plan_async()
        finally:
            self.handle._active_on_delta = previous_on_delta
            self.handle._active_on_event = previous_on_event
        if not has_pending_plan:
            return self.handle._no_pending_plan_result(content)
        return self.handle._run_result(content)

    async def reject(self) -> str:
        return await asyncio.to_thread(self.handle.reject)

    async def reject_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        return await asyncio.to_thread(self.handle.reject_result, on_delta=on_delta, on_event=on_event)


def agent(
    *,
    config: Config | AgentConfig | None = None,
    preset: AgentPreset | None = None,
    llm: LLMClient | Any | None = None,
    tools: Iterable[object] | None = None,
    skills: object | Iterable[object] | None = None,
    system_prompt: str | None = None,
    conversation_id: str | None = None,
    permission_callback: PermissionCallback | None = None,
    on_event: EventCallback | None = None,
    mcp: Iterable[MCPServerConfig] | None = None,
    redaction_callback: Callable[[str, str, dict], str] | None = None,
    redaction_fail_closed: bool = False,
) -> AgentHandle:
    """Create a configured Chulk agent handle."""
    runtime_config = _coerce_config(config)
    selected_tools = tools if tools is not None else (preset.tools if preset is not None else None)
    selected_skills = skills if skills is not None else (preset.skills if preset is not None else None)
    selected_prompt = system_prompt or (preset.system_prompt if preset is not None else None)
    runtime = create_runtime_agent(
        runtime_config,
        conversation_id=conversation_id,
        llm_client=llm,
        tool_specs=selected_tools,
        skill_specs=selected_skills,
        system_prompt=selected_prompt,
        permission_callback=permission_callback,
        mcp_servers=tuple(mcp) if mcp is not None else None,
        redaction_callback=redaction_callback,
        redaction_fail_closed=redaction_fail_closed,
    )
    return AgentHandle(runtime, on_event=on_event)


def chat_agent(**kwargs: Any) -> AgentHandle:
    """Create a plain chat agent with no tools or skills configured."""
    _ensure_chat_kwargs(kwargs)
    return agent(tools=[], skills=[], **kwargs)


def async_agent(**kwargs: Any) -> AsyncAgentHandle:
    """Create an async SDK handle backed by the synchronous runtime."""
    return AsyncAgentHandle(agent(**kwargs))


def async_chat_agent(**kwargs: Any) -> AsyncAgentHandle:
    """Create an async plain chat agent with no tools or skills configured."""
    _ensure_chat_kwargs(kwargs)
    return AsyncAgentHandle(chat_agent(**kwargs))


def _coerce_config(config: Config | AgentConfig | None) -> Config:
    if config is None:
        return load_config()
    if isinstance(config, AgentConfig):
        return config.to_config()
    return config


def _ensure_chat_kwargs(kwargs: dict[str, Any]) -> None:
    disallowed = [
        name
        for name in ("preset", "tools", "skills")
        if name in kwargs and kwargs[name] is not None
    ]
    if disallowed:
        joined = ", ".join(disallowed)
        raise ValueError(f"ChatAgent does not accept {joined}; use Agent for configured tools or skills")


def _set_env(env: dict[str, str], key: str, value: object) -> None:
    if value is not None:
        env[key] = str(value)


def _plan_snapshot(plan: Any | None) -> PlanSnapshot | None:
    if plan is None:
        return None
    payload = plan.to_dict()
    return PlanSnapshot(
        summary=payload.get("summary") or "",
        status=payload.get("status") or "unknown",
        steps=list(payload.get("steps") or []),
        created_at=payload.get("created_at"),
        approved_at=payload.get("approved_at"),
        rejected_at=payload.get("rejected_at"),
    )


Agent = agent
AsyncAgent = async_agent
ChatAgent = chat_agent
AsyncChatAgent = async_chat_agent


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "AgentHandle",
    "AgentPreset",
    "AsyncAgent",
    "AsyncAgentHandle",
    "AsyncChatAgent",
    "ChatAgent",
    "MCP",
    "PlanResult",
    "PlanSnapshot",
    "RunResult",
    "agent",
    "async_agent",
    "async_chat_agent",
    "chat_agent",
]
