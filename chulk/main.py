"""Command-line entrypoint for ChulkHarness."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
import sys
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

from chulk import __version__
from chulk.cli import (
    CLICommandContext,
    EXIT_COMMANDS,
    PromptHistory,
    ProgressReporter,
    ProgressSettings,
    TerminalUI,
    command_completion_candidates,
    handle_cli_command,
)
from chulk.cli.entrypoints import (
    EXIT_CONFIGURATION_ERROR,
    EXIT_OK,
    json_text,
    run_doctor_command,
    run_exec_command,
    run_init_command,
    run_trace_command,
)
from chulk.cli.parser import build_parser
from chulk.config import Config, load_cli_config
from chulk.core import Agent
from chulk.llm import (
    AnthropicProvider,
    BedrockProvider,
    BindableLLM,
    DeepSeekProvider,
    FallbackChain,
    GeminiProvider,
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LocalProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from chulk.presets import software_engineer
from chulk.runtime import create_agent
from chulk.sessions import AmbiguousSessionError, SessionNotFoundError, SQLiteSessionStore
from chulk.tools.permissions import PermissionDecision, PermissionDecisionRecord, PermissionRequest


def format_config(config: Config) -> str:
    """Format non-secret configuration values for terminal output."""
    values = {
        "project_root": config.project_root,
        "runtime_dir": config.runtime_dir,
        "skills_dir": config.skills_dir,
        "skills_dirs": ", ".join(str(path) for path in config.skills_dirs),
        "store_path": config.store_path,
        "traces_dir": config.traces_dir,
        "mcp_config_path": config.mcp_config_path,
        "mcp_servers": f"{len(config.mcp_servers)} configured",
        "llm_provider": config.llm_provider,
        "model": config.model,
        "llm_fallback_providers": _format_fallback_providers(config),
        "permission_profile": config.permission_profile,
        "openai_api_key": "set" if config.openai_api_key else "not set",
        "deepseek_api_key": "set" if config.deepseek_api_key else "not set",
        "deepseek_base_url": _format_base_url(config.deepseek_base_url),
        "local_api_key": "set" if config.local_api_key else "not set",
        "local_base_url": _format_base_url(config.local_base_url),
        "local_context_window_tokens": config.local_context_window_tokens,
        "openai_compatible_api_key": "set" if config.openai_compatible_api_key else "not set",
        "openai_compatible_base_url": _format_base_url(config.openai_compatible_base_url),
        "openrouter_api_key": "set" if config.openrouter_api_key else "not set",
        "openrouter_base_url": _format_base_url(config.openrouter_base_url),
        "anthropic_api_key": "set" if config.anthropic_api_key else "not set",
        "anthropic_base_url": _format_base_url(config.anthropic_base_url),
        "bedrock_api_key": "set" if config.bedrock_api_key else "not set",
        "bedrock_base_url": _format_base_url(config.bedrock_base_url),
        "gemini_api_key": "set" if config.gemini_api_key else "not set",
        "gemini_base_url": _format_base_url(config.gemini_base_url),
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


def _format_base_url(value: str | None) -> str:
    """Render a base URL without exposing credentials or URL parameters."""
    if value is None or not value.strip():
        return "not set"
    clean_value = value.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in clean_value):
        return "set (value hidden)"
    try:
        parsed = urlsplit(clean_value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "set (value hidden)"
        hostname = parsed.hostname
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "set (value hidden)"
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


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
    providers: list[LLMClient | BindableLLM] = [
        _create_provider_spec(config.llm_provider, config.model)
    ]
    providers.extend(
        _create_provider_spec(provider_config.provider, provider_config.model)
        for provider_config in config.llm_fallback_providers
    )
    return FallbackChain(providers=providers)


def _create_provider_spec(
    provider: str,
    model: str,
) -> (
    OpenAIProvider
    | DeepSeekProvider
    | LocalProvider
    | OpenAICompatibleProvider
    | OpenRouterProvider
    | AnthropicProvider
    | BedrockProvider
    | GeminiProvider
):
    if provider == "openai":
        return OpenAIProvider(model=model)
    if provider == "deepseek":
        return DeepSeekProvider(model=model)
    if provider == "local":
        return LocalProvider(model=model)
    if provider == "openai-compatible":
        return OpenAICompatibleProvider(model=model)
    if provider == "openrouter":
        return OpenRouterProvider(model=model)
    if provider == "anthropic":
        return AnthropicProvider(model=model)
    if provider == "bedrock":
        return BedrockProvider(model=model)
    if provider == "gemini":
        return GeminiProvider(model=model)
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
    error_func: Callable[[str], None] | None = None,
) -> int:
    """Run the interactive chat loop."""
    error_func = error_func or _print_stderr
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
    if hasattr(prompt_history, "configure_completion"):
        prompt_history.configure_completion(command_completion_candidates())
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
        response_func=lambda response: (
            None
            if progress_reporter.streamed_answer
            else output_func(terminal.assistant_message(response))
        ),
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

        progress_reporter.reset_stream_state()
        handled = False
        try:
            handled = handle_cli_command(user_message.strip(), command_context)
        except LLMError as exc:
            error_func(terminal.error(f"error: {exc}"))
            return 1
        except Exception as exc:
            error_func(terminal.error(f"error: unexpected failure: {exc}"))
            return 1
        finally:
            progress_reporter.close()
        if handled:
            progress_reporter.flush_summary()
            continue

        if command_context.agent.has_pending_plan():
            output_func(terminal.warning("A plan is waiting for approval. Use /approve to execute it or /reject to cancel it."))
            continue

        try:
            assistant_response = command_context.agent.run_turn(user_message)
        except LLMError as exc:
            error_func(terminal.error(f"error: {exc}"))
            return 1
        except Exception as exc:
            error_func(terminal.error(f"error: unexpected failure: {exc}"))
            return 1
        finally:
            progress_reporter.close()

        if not progress_reporter.streamed_answer:
            output_func(terminal.assistant_message(assistant_response))
        progress_reporter.flush_summary()


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
    error_func: Callable[[str], None] | None = None,
    llm_client_factory: Callable[[Config], LLMClient] | None = None,
) -> int:
    """Run the current CLI."""
    error_func = error_func or _print_stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    terminal = TerminalUI.themed(color=args.color)

    if args.version:
        output_func(f"ChulkHarness {__version__}")
        return EXIT_OK

    if args.command == "init":
        return run_init_command(
            args.project_root,
            mode=args.init_mode,
            json_output=args.json_output,
            output_func=output_func,
            error_func=error_func,
        )
    if args.command == "doctor":
        return run_doctor_command(json_output=args.json_output, output_func=output_func)
    if args.command == "trace":
        return run_trace_command(
            args.trace_command,
            args.path,
            json_output=args.json_output,
            output_path=getattr(args, "output", None),
            force=bool(getattr(args, "force", False)),
            output_func=output_func,
            error_func=error_func,
        )

    if args.show_config:
        try:
            output_func(format_config(load_cli_config()))
        except (OSError, ValueError) as exc:
            error_func(terminal.error(f"configuration error: {exc}"))
            return EXIT_CONFIGURATION_ERROR
        return EXIT_OK

    try:
        config = load_cli_config()
    except (OSError, ValueError, LLMConfigurationError) as exc:
        if args.command == "exec" and getattr(args, "json_output", False):
            output_func(json_text({"ok": False, "status": "configuration_error", "error": str(exc)}))
        else:
            error_func(terminal.error(f"configuration error: {exc}"))
        return EXIT_CONFIGURATION_ERROR

    if args.command == "exec" or args.once is not None:
        if args.resume or args.continue_session:
            error_func(terminal.error("configuration error: --resume and --continue are interactive-only"))
            return EXIT_CONFIGURATION_ERROR
        message = " ".join(args.message) if args.command == "exec" else str(args.once)
        json_output = bool(getattr(args, "json_output", False))
        return run_exec_command(
            message,
            agent_factory=lambda: create_cli_agent(config, llm_client_factory),
            json_output=json_output,
            output_func=output_func,
            error_func=error_func,
        )

    try:
        conversation_id = _resolve_startup_conversation(config, args)
    except (SessionNotFoundError, AmbiguousSessionError) as exc:
        error_func(terminal.error(f"session error: {exc}"))
        return EXIT_CONFIGURATION_ERROR
    try:
        agent = create_cli_agent(config, llm_client_factory, conversation_id=conversation_id)
        agent.permission_callback = _make_cli_permission_callback(
            terminal,
            input_func=input_func,
            output_func=output_func,
        )
    except (SessionNotFoundError, AmbiguousSessionError) as exc:
        error_func(terminal.error(f"session error: {exc}"))
        return EXIT_CONFIGURATION_ERROR
    except (ValueError, LLMConfigurationError) as exc:
        error_func(terminal.error(f"configuration error: {exc}"))
        return EXIT_CONFIGURATION_ERROR

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
        error_func=error_func,
    )


def _resolve_startup_conversation(config: Config, args: Namespace) -> str | None:
    if args.resume:
        return str(args.resume)
    if not args.continue_session:
        return None
    latest = SQLiteSessionStore(config.store_path).latest_conversation()
    if latest is None:
        raise SessionNotFoundError("No persisted session is available to continue")
    return latest.id


def _print_stderr(message: str) -> None:
    print(message, file=sys.stderr)


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
