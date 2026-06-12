"""Command-line entrypoint for ChulkHarness."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Callable

from src import __version__
from src.cli import (
    CLICommandContext,
    EXIT_COMMANDS,
    PromptHistory,
    ProgressReporter,
    ProgressSettings,
    TerminalUI,
    handle_cli_command,
)
from src.config import Config, load_config
from src.core import Agent, AgentState
from src.llm import LLMClient, LLMConfigurationError, LLMError, create_llm_client
from src.memory import ConversationMemory, SQLiteMemoryStore
from src.sessions import SQLiteSessionStore, SessionRecorder
from src.skills import SkillRegistry
from src.tools import create_default_tool_registry
from src.tracing import JSONLTraceLogger


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="chulk",
        description="Run the ChulkHarness agent runtime.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the ChulkHarness version and exit.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved local configuration and exit.",
    )
    parser.add_argument(
        "--once",
        metavar="MESSAGE",
        help="Send one message to the agent and exit.",
    )
    return parser


def format_config(config: Config) -> str:
    """Format non-secret configuration values for terminal output."""
    values = {
        "project_root": config.project_root,
        "skills_dir": config.skills_dir,
        "store_path": config.store_path,
        "traces_dir": config.traces_dir,
        "llm_provider": config.llm_provider,
        "model": config.model,
        "openai_api_key": "set" if config.openai_api_key else "not set",
        "deepseek_api_key": "set" if config.deepseek_api_key else "not set",
        "deepseek_base_url": config.deepseek_base_url,
        "history_limit": config.history_limit,
        "max_tool_calls_per_turn": config.max_tool_calls_per_turn,
        "max_skills_per_turn": config.max_skills_per_turn,
        "max_skill_content_chars": config.max_skill_content_chars,
        "shell_timeout_seconds": config.shell_timeout_seconds,
        "llm_timeout_seconds": config.llm_timeout_seconds,
        "llm_max_retries": config.llm_max_retries,
        "trace_max_prompt_chars": config.trace_max_prompt_chars,
        "max_observation_chars": config.max_observation_chars,
        "max_tool_stdout_chars": config.max_tool_stdout_chars,
        "max_tool_stderr_chars": config.max_tool_stderr_chars,
    }
    lines = ["ChulkHarness configuration:"]
    lines.extend(f"  {key}: {value}" for key, value in values.items())
    return "\n".join(lines)


def create_agent(
    config: Config,
    llm_client_factory: Callable[[Config], LLMClient] | None = None,
    *,
    conversation_id: str | None = None,
) -> Agent:
    """Create the agent runtime."""
    if llm_client_factory is None:
        llm_client_factory = _default_llm_client_factory
    memory_store = SQLiteMemoryStore(config.store_path)
    session_store = SQLiteSessionStore(config.store_path)
    skill_registry = SkillRegistry(
        config.skills_dir,
        max_skills=config.max_skills_per_turn,
        max_content_chars=config.max_skill_content_chars,
    )
    skill_registry.load_metadata()
    state = _create_agent_state(session_store, conversation_id)
    trace_logger = JSONLTraceLogger(config.traces_dir, state.conversation_id)
    conversation_memory = ConversationMemory(max_messages=config.history_limit)
    if conversation_id is not None:
        conversation_memory.messages = session_store.load_recent_messages(state.conversation_id, config.history_limit)
        conversation_memory.trim_to_limit()
        state.messages = conversation_memory.recent()
    session_recorder = SessionRecorder(
        session_store,
        state.conversation_id,
        provider=config.llm_provider,
        model=config.model,
        trace_path=trace_logger.path,
    )
    agent = Agent(
        llm_client_factory(config),
        state=state,
        memory=conversation_memory,
        memory_store=memory_store,
        skill_registry=skill_registry,
        trace_logger=trace_logger,
        tool_registry=create_default_tool_registry(
            config.project_root,
            config.shell_timeout_seconds,
            memory_store=memory_store,
        ),
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        max_skills_per_turn=config.max_skills_per_turn,
        max_skill_content_chars=config.max_skill_content_chars,
        trace_max_prompt_chars=config.trace_max_prompt_chars,
        max_observation_chars=config.max_observation_chars,
        max_tool_stdout_chars=config.max_tool_stdout_chars,
        max_tool_stderr_chars=config.max_tool_stderr_chars,
        event_callback=session_recorder.callback,
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
        timeout_seconds=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )


def run_chat_loop(
    agent: Agent,
    *,
    config: Config | None = None,
    terminal: TerminalUI | None = None,
    agent_factory: Callable[[str], Agent] | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> int:
    """Run the interactive chat loop."""
    terminal = terminal or TerminalUI.themed()
    progress_settings = ProgressSettings()
    progress_reporter = ProgressReporter(
        terminal,
        output_func,
        config=config,
        agent=agent,
        settings=progress_settings,
        previous_callback=agent.event_callback,
    )
    agent.event_callback = progress_reporter.callback
    session_store = SQLiteSessionStore(config.store_path) if config is not None else None
    prompt_history = PromptHistory.create(enabled=input_func is input)
    _load_prompt_history(prompt_history, session_store, agent)
    if config is not None:
        output_func(terminal.banner(config, agent))
    else:
        output_func("ChulkHarness CLI")
    output_func(terminal.hint())

    def switch_agent(next_agent: Agent) -> None:
        progress_reporter.close()
        progress_reporter.agent = next_agent
        progress_reporter.previous_callback = next_agent.event_callback
        next_agent.event_callback = progress_reporter.callback
        command_context.agent = next_agent
        _load_prompt_history(prompt_history, session_store, next_agent)

    command_context = CLICommandContext(
        agent=agent,
        config=config,
        terminal=terminal,
        progress_settings=progress_settings,
        output_func=output_func,
        session_store=session_store,
        agent_factory=agent_factory,
        switch_agent=switch_agent,
    )

    while True:
        try:
            user_message = input_func(terminal.prompt())
        except EOFError:
            output_func(terminal.bye(command_context.agent))
            return 0
        except KeyboardInterrupt:
            output_func("\n" + terminal.bye(command_context.agent))
            return 0

        if not user_message.strip():
            continue

        normalized_message = user_message.strip().lower()

        if normalized_message in EXIT_COMMANDS:
            output_func(terminal.bye(command_context.agent))
            return 0

        prompt_history.add(user_message)

        try:
            if handle_cli_command(user_message.strip(), command_context):
                continue
        except LLMError as exc:
            output_func(terminal.error(f"error: {exc}"))
            return 1
        except Exception as exc:
            output_func(terminal.error(f"error: unexpected failure: {exc}"))
            return 1
        finally:
            progress_reporter.close()

        if command_context.agent.has_pending_plan():
            output_func(terminal.warning("A plan is waiting for approval. Use /approve to execute it or /reject to cancel it."))
            continue

        try:
            assistant_response = command_context.agent.run_turn(user_message)
        except LLMError as exc:
            output_func(terminal.error(f"error: {exc}"))
            return 1
        except Exception as exc:
            output_func(terminal.error(f"error: unexpected failure: {exc}"))
            return 1
        finally:
            progress_reporter.close()

        output_func(terminal.assistant_message(assistant_response))


def _load_prompt_history(
    prompt_history: PromptHistory,
    session_store: SQLiteSessionStore | None,
    agent: Agent,
) -> None:
    """Load arrow-key prompt history from the active persisted session."""
    if session_store is None:
        prompt_history.replace(agent.memory.messages)
        return
    messages = session_store.list_messages(agent.state.conversation_id, limit=200)
    prompt_history.replace(messages)


def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    llm_client_factory: Callable[[Config], LLMClient] | None = None,
) -> int:
    """Run the current CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    terminal = TerminalUI.themed()

    if args.version:
        print(f"ChulkHarness {__version__}")
        return 0

    if args.show_config:
        print(format_config(load_config()))
        return 0

    try:
        config = load_config()
        agent = create_agent(config, llm_client_factory)
    except (ValueError, LLMConfigurationError) as exc:
        output_func(terminal.error(f"configuration error: {exc}"))
        return 1

    if args.once is not None:
        try:
            output_func(agent.run_turn(args.once))
        except LLMError as exc:
            output_func(f"error: {exc}")
            return 1
        return 0

    return run_chat_loop(
        agent,
        config=config,
        terminal=terminal,
        agent_factory=lambda conversation_id: create_agent(
            config,
            llm_client_factory,
            conversation_id=conversation_id,
        ),
        input_func=input_func,
        output_func=output_func,
    )


if __name__ == "__main__":
    raise SystemExit(main())
