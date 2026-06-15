"""Runtime assembly for configured Chulk agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chulk.config import Config
from chulk.core import Agent, AgentState
from chulk.core.context import ContextBudget
from chulk.core.prompts import BASE_SYSTEM_PROMPT
from chulk.llm import LLMClient, create_llm_client, resolve_model_capabilities
from chulk.memory import ConversationMemory, SQLiteMemoryStore
from chulk.sessions import SQLiteSessionStore, SessionRecorder
from chulk.skills import SkillRegistry
from chulk.tools import Tool, ToolRegistry, create_default_tool_registry
from chulk.tracing import JSONLTraceLogger


class LLMClientFactory(Protocol):
    """Factory used by tests and the CLI to inject an LLM client."""

    def __call__(self, config: Config) -> LLMClient:
        """Return an LLM client for the given runtime config."""


@dataclass(frozen=True)
class RuntimeToolContext:
    """Context required to bind project-scoped tool references."""

    project_root: Path
    shell_timeout_seconds: int
    memory_store: SQLiteMemoryStore | None = None


def create_agent(
    config: Config,
    llm_client_factory: Callable[[Config], LLMClient] | None = None,
    *,
    conversation_id: str | None = None,
    llm_client: LLMClient | None = None,
    tool_specs: Iterable[object] | None = None,
    skill_specs: Iterable[object] | None = None,
    system_prompt: str | None = None,
) -> Agent:
    """Create the configured Chulk agent runtime."""
    if llm_client is not None and llm_client_factory is not None:
        raise ValueError("Pass either llm_client or llm_client_factory, not both")

    if llm_client_factory is None:
        llm_client_factory = _default_llm_client_factory
    model_capabilities = resolve_model_capabilities(config.llm_provider, config.model)
    context_budget = ContextBudget(
        max_prompt_tokens=model_capabilities.context_window_tokens,
        response_reserve_tokens=model_capabilities.default_response_reserve_tokens,
    )
    memory_store = SQLiteMemoryStore(config.store_path)
    session_store = SQLiteSessionStore(config.store_path)
    skill_registry = SkillRegistry(
        config.skills_dir,
        max_skills=config.max_skills_per_turn,
        max_content_chars=config.max_skill_content_chars,
    )
    skill_registry.load_metadata()
    pinned_skill_names = _register_skill_specs(skill_registry, skill_specs)
    state = _create_agent_state(session_store, conversation_id)
    trace_logger = JSONLTraceLogger(config.traces_dir, state.conversation_id)
    conversation_memory = ConversationMemory(max_messages=config.history_limit)
    if conversation_id is not None:
        latest_summary = session_store.load_latest_summary(state.conversation_id)
        recent_messages = session_store.load_recent_messages(
            state.conversation_id,
            config.history_limit,
            after_ordinal=latest_summary.source_message_count if latest_summary is not None else 0,
        )
        conversation_memory.replace(
            recent_messages,
            conversation_summary=latest_summary.content if latest_summary is not None else None,
            summary_message_count=latest_summary.source_message_count if latest_summary is not None else 0,
        )
        state.messages = conversation_memory.recent()
        state.conversation_summary = conversation_memory.conversation_summary
    session_recorder = SessionRecorder(
        session_store,
        state.conversation_id,
        provider=config.llm_provider,
        model=config.model,
        trace_path=trace_logger.path,
    )
    client = llm_client if llm_client is not None else llm_client_factory(config)
    if hasattr(client, "bind_config"):
        client = client.bind_config(config)  # type: ignore[assignment, attr-defined]
    agent = Agent(
        client,
        state=state,
        memory=conversation_memory,
        memory_store=memory_store,
        skill_registry=skill_registry,
        trace_logger=trace_logger,
        tool_registry=_create_tool_registry(config, memory_store, tool_specs),
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        max_skills_per_turn=config.max_skills_per_turn,
        max_skill_content_chars=config.max_skill_content_chars,
        trace_max_prompt_chars=config.trace_max_prompt_chars,
        max_observation_chars=config.max_observation_chars,
        max_tool_stdout_chars=config.max_tool_stdout_chars,
        max_tool_stderr_chars=config.max_tool_stderr_chars,
        context_budget=context_budget,
        event_callback=session_recorder.callback,
        pinned_skill_names=pinned_skill_names,
        system_prompt=system_prompt or BASE_SYSTEM_PROMPT,
    )
    agent.session_store = session_store
    agent.session_recorder = session_recorder
    return agent


def _create_agent_state(session_store: SQLiteSessionStore, conversation_id: str | None) -> AgentState:
    """Create fresh state or rebuild state for an existing conversation."""
    if conversation_id is None:
        return AgentState()

    conversation = session_store.get_conversation(conversation_id)
    state = AgentState(conversation_id=conversation.id)
    state.turns = session_store.load_turns(conversation.id)
    if not state.turns:
        return state

    latest_turn = state.turns[-1]
    state.current_turn_id = latest_turn.turn_id
    state.loaded_memory_ids = list(latest_turn.loaded_memory_ids)
    state.extracted_memory_ids = list(latest_turn.extracted_memory_ids)
    state.loaded_skill_names = list(latest_turn.loaded_skill_names)
    state.available_tool_names = list(latest_turn.available_tool_names)
    state.errors = [error for turn in state.turns for error in turn.errors]
    state.final_answer = latest_turn.final_answer
    if latest_turn.context_reports:
        state.last_context_report = latest_turn.context_reports[-1]
    for turn in reversed(state.turns):
        if turn.status == "waiting_for_approval" and turn.active_plan is not None and not turn.plan_approved:
            state.active_plan = turn.active_plan
            state.pending_plan_turn_id = turn.turn_id
            break
    return state


def _default_llm_client_factory(config: Config) -> LLMClient:
    return create_llm_client(
        provider=config.llm_provider,
        model=config.model,
        openai_api_key=config.openai_api_key,
        deepseek_api_key=config.deepseek_api_key,
        deepseek_base_url=config.deepseek_base_url,
        local_api_key=config.local_api_key,
        local_base_url=config.local_base_url,
        timeout_seconds=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )


def _create_tool_registry(
    config: Config,
    memory_store: SQLiteMemoryStore,
    tool_specs: Iterable[object] | None,
) -> ToolRegistry:
    if tool_specs is None:
        return create_default_tool_registry(
            config.project_root,
            config.shell_timeout_seconds,
            memory_store=memory_store,
        )

    context = RuntimeToolContext(
        project_root=config.project_root,
        shell_timeout_seconds=config.shell_timeout_seconds,
        memory_store=memory_store,
    )
    registry = ToolRegistry()
    for spec in tool_specs:
        tool = _resolve_tool_spec(spec, context)
        registry.register(tool)
    return registry


def _resolve_tool_spec(spec: object, context: RuntimeToolContext) -> Tool:
    if isinstance(spec, Tool):
        return spec
    if hasattr(spec, "to_tool"):
        return spec.to_tool(context)  # type: ignore[no-any-return, attr-defined]
    raise TypeError(f"Unsupported tool spec: {spec!r}")


def _register_skill_specs(registry: SkillRegistry, skill_specs: Iterable[object] | None) -> list[str]:
    pinned_skill_names: list[str] = []
    if skill_specs is None:
        return pinned_skill_names
    for spec in skill_specs:
        if hasattr(spec, "register"):
            pinned_name = spec.register(registry)  # type: ignore[attr-defined]
            if pinned_name:
                pinned_skill_names.append(str(pinned_name))
            continue
        if isinstance(spec, str):
            pinned_skill_names.append(spec)
            continue
        raise TypeError(f"Unsupported skill spec: {spec!r}")
    return pinned_skill_names
