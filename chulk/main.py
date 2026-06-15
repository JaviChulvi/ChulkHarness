"""Command-line entrypoint for ChulkHarness."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Callable

from chulk import __version__
from chulk.cli import (
    CLICommandContext,
    EXIT_COMMANDS,
    PromptHistory,
    ProgressReporter,
    ProgressSettings,
    TerminalUI,
    handle_cli_command,
)
from chulk.config import Config, load_config
from chulk.core import Agent
from chulk.llm import (
    DeepSeekProvider,
    FallbackChain,
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LocalProvider,
    OpenAIProvider,
)
from chulk.presets import software_engineer
from chulk.runtime import create_agent
from chulk.sessions import SQLiteSessionStore
from chulk.tools.permissions import PermissionDecision, PermissionDecisionRecord, PermissionRequest


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
        "llm_fallback_providers": _format_fallback_providers(config),
        "permission_profile": config.permission_profile,
        "openai_api_key": "set" if config.openai_api_key else "not set",
        "deepseek_api_key": "set" if config.deepseek_api_key else "not set",
        "deepseek_base_url": config.deepseek_base_url,
        "local_api_key": "set" if config.local_api_key else "not set",
        "local_base_url": config.local_base_url,
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
        "max_reflection_attempts": config.max_reflection_attempts,
    }
    lines = ["ChulkHarness configuration:"]
    lines.extend(f"  {key}: {value}" for key, value in values.items())
    return "\n".join(lines)


def create_cli_agent(
    config: Config,
    llm_client_factory: Callable[[Config], LLMClient] | None = None,
    *,
    conversation_id: str | None = None,
) -> Agent:
    """Create the default CLI coding-agent runtime."""
    preset = software_engineer()
    if llm_client_factory is not None:
        return create_agent(
            config,
            llm_client_factory,
            conversation_id=conversation_id,
            tool_specs=preset.tools,
            skill_specs=preset.skills,
            system_prompt=preset.system_prompt,
        )
    return create_agent(
        config,
        conversation_id=conversation_id,
        llm_client=create_cli_llm(config),
        tool_specs=preset.tools,
        skill_specs=preset.skills,
        system_prompt=preset.system_prompt,
    )


def create_cli_llm(config: Config) -> FallbackChain:
    """Create the CLI LLM chain from public provider objects."""
    providers: list[OpenAIProvider | DeepSeekProvider | LocalProvider] = [
        _create_provider_spec(config.llm_provider, config.model)
    ]
    providers.extend(
        _create_provider_spec(provider_config.provider, provider_config.model)
        for provider_config in config.llm_fallback_providers
    )
    return FallbackChain(providers=providers)


def _create_provider_spec(provider: str, model: str) -> OpenAIProvider | DeepSeekProvider | LocalProvider:
    if provider == "openai":
        return OpenAIProvider(model=model)
    if provider == "deepseek":
        return DeepSeekProvider(model=model)
    if provider == "local":
        return LocalProvider(model=model)
    raise LLMConfigurationError(f"Unsupported CLI LLM provider: {provider}")


def _format_fallback_providers(config: Config) -> str:
    if not config.llm_fallback_providers:
        return "none"
    return ", ".join(f"{provider.provider}:{provider.model}" for provider in config.llm_fallback_providers)


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
    permission_callback = _make_cli_permission_callback(
        terminal,
        input_func=input_func,
        output_func=output_func,
        before_prompt=progress_reporter.close,
    )
    agent.permission_callback = permission_callback
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
        next_agent.permission_callback = permission_callback
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
        agent = create_cli_agent(config, llm_client_factory)
        agent.permission_callback = _make_cli_permission_callback(
            terminal,
            input_func=input_func,
            output_func=output_func,
        )
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
        agent_factory=lambda conversation_id: create_cli_agent(
            config,
            llm_client_factory,
            conversation_id=conversation_id,
        ),
        input_func=input_func,
        output_func=output_func,
    )


def _make_cli_permission_callback(
    terminal: TerminalUI,
    *,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
    before_prompt: Callable[[], None] | None = None,
) -> Callable[[PermissionRequest, PermissionDecisionRecord], PermissionDecision]:
    """Create the CLI permission approval callback."""

    def approve(request: PermissionRequest, record: PermissionDecisionRecord) -> PermissionDecision:
        if before_prompt is not None:
            before_prompt()
        output_func(terminal.permission_request(request, record))
        while True:
            try:
                answer = input_func(terminal.permission_prompt())
            except (EOFError, KeyboardInterrupt):
                output_func(terminal.warning("permission denied"))
                return PermissionDecision.DENY

            decision = _parse_permission_answer(answer)
            if decision is not None:
                return decision
            output_func(terminal.warning("Enter y to approve or n to deny."))

    return approve


def _parse_permission_answer(answer: str) -> PermissionDecision | None:
    normalized = answer.strip().lower()
    if normalized in {"y", "yes", "a", "allow", "approve"}:
        return PermissionDecision.ALLOW
    if normalized in {"", "n", "no", "d", "deny", "reject"}:
        return PermissionDecision.DENY
    return None


if __name__ == "__main__":
    raise SystemExit(main())
