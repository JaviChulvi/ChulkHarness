"""Command-line entrypoint for ChulkHarness."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from dataclasses import replace
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
from chulk.cli.gateway import run_gateway_command
from chulk.cli.profiles import run_profile_command
from chulk.cli.plugins import run_plugin_command
from chulk.cli.models import run_model_command
from chulk.cli.usage import run_usage_command
from chulk.cli.sessions import run_session_command
from chulk.cli.parser import build_parser
from chulk.config import Config, LLMFallbackProviderConfig, load_cli_config
from chulk.core import Agent
from chulk.errors import ChulkError
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
    MoonshotProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from chulk.presets import software_engineer
from chulk.model_profiles import (
    ModelCapabilityRequirements,
    ModelProfileNotFoundError,
    ModelProfileService,
    ModelProfileStore,
    ModelProfileValidator,
    ResolvedModelRuntime,
)
from chulk.profiles import (
    AgentProfile,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    ProfileOwnershipError,
    ProfileRuntimeFactory,
)
from chulk.gateway import SQLiteGatewayLedger, SQLiteGatewayRouter
from chulk.plugins import LocalPluginRegistry
from chulk.runtime import create_agent
from chulk.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    SQLiteSessionStore,
)
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
)
from chulk.usage import ExactCost, RunBudget, UsageDimensions
from chulk.usage import UsageLedger


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
        "profile_id": config.profile_id,
        "openai_api_key": "set" if config.openai_api_key else "not set",
        "deepseek_api_key": "set" if config.deepseek_api_key else "not set",
        "deepseek_base_url": _format_base_url(config.deepseek_base_url),
        "moonshot_api_key": "set" if config.moonshot_api_key else "not set",
        "moonshot_base_url": _format_base_url(config.moonshot_base_url),
        "local_api_key": "set" if config.local_api_key else "not set",
        "local_base_url": _format_base_url(config.local_base_url),
        "local_context_window_tokens": config.local_context_window_tokens,
        "openai_compatible_api_key": "set"
        if config.openai_compatible_api_key
        else "not set",
        "openai_compatible_base_url": _format_base_url(
            config.openai_compatible_base_url
        ),
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
    profile: AgentProfile | None = None,
    runtime_metadata: dict | None = None,
    run_budget: RunBudget | None = None,
    usage_channel: str = "cli",
) -> Agent:
    """Create the default CLI coding-agent runtime."""
    preset = software_engineer()
    if profile is not None and profile.execution_backend_id != "host":
        raise ValueError(
            f"execution backend {profile.execution_backend_id!r} is not configured for the CLI host"
        )
    system_prompt = (
        profile.system_prompt
        if profile is not None and profile.system_prompt is not None
        else preset.system_prompt
    )
    allowed_skill_names = profile.allowed_skills if profile is not None else None
    memory_namespace = profile.memory_namespace if profile is not None else None
    mcp_servers = config.mcp_servers
    if profile is not None and profile.allowed_mcp_servers is not None:
        allowed_mcp_servers = set(profile.allowed_mcp_servers)
        mcp_servers = tuple(
            server for server in mcp_servers if server.label in allowed_mcp_servers
        )
    if llm_client_factory is not None:
        return create_agent(
            config,
            llm_client_factory,
            conversation_id=conversation_id,
            tool_specs=preset.tools,
            skill_specs=preset.skills,
            system_prompt=system_prompt,
            profile_id=profile.id if profile is not None else config.profile_id,
            memory_namespace=memory_namespace,
            allowed_skill_names=allowed_skill_names,
            mcp_servers=mcp_servers,
            runtime_metadata=runtime_metadata,
            run_budget=run_budget,
            usage_dimensions=UsageDimensions(
                profile_id=profile.id if profile is not None else config.profile_id,
                channel=usage_channel,
            ),
        )
    return create_agent(
        config,
        conversation_id=conversation_id,
        llm_client=create_cli_llm(config),
        tool_specs=preset.tools,
        skill_specs=preset.skills,
        system_prompt=system_prompt,
        profile_id=profile.id if profile is not None else config.profile_id,
        memory_namespace=memory_namespace,
        allowed_skill_names=allowed_skill_names,
        mcp_servers=mcp_servers,
        runtime_metadata=runtime_metadata,
        run_budget=run_budget,
        usage_dimensions=UsageDimensions(
            profile_id=profile.id if profile is not None else config.profile_id,
            channel=usage_channel,
        ),
    )


def resolve_cli_model(
    base_config: Config,
    config: Config,
    profile: AgentProfile,
    service: ModelProfileService,
    *,
    requested_profile_id: str | None = None,
    channel: str = "cli",
    build_client: bool = True,
) -> tuple[Config, ResolvedModelRuntime, FallbackChain | None]:
    """Resolve one model selection and preserve the legacy default path."""
    runtime = service.resolve_for_agent(
        profile,
        requested_profile_id=requested_profile_id,
        channel=channel,
    )
    if runtime.legacy_compatibility:
        return config, runtime, None
    candidates = runtime.candidates
    selected_config = replace(
        config,
        llm_provider=candidates[0].profile.provider,
        model=candidates[0].profile.model,
        llm_fallback_providers=tuple(
            LLMFallbackProviderConfig(
                provider=candidate.profile.provider,
                model=candidate.profile.model,
            )
            for candidate in candidates[1:]
        ),
    )
    return (
        selected_config,
        runtime,
        service.create_chain(base_config, runtime) if build_client else None,
    )


def create_selected_cli_agent(
    base_config: Config,
    config: Config,
    profile: AgentProfile,
    service: ModelProfileService,
    llm_client_factory: Callable[[Config], LLMClient] | None,
    *,
    requested_profile_id: str | None = None,
    conversation_id: str | None = None,
    channel: str = "cli",
) -> tuple[Agent, Config, ResolvedModelRuntime]:
    """Build a CLI agent from the current constrained model selection."""
    selected_config, runtime, chain = resolve_cli_model(
        base_config,
        config,
        profile,
        service,
        requested_profile_id=requested_profile_id,
        channel=channel,
        build_client=llm_client_factory is None,
    )
    selected_factory = llm_client_factory
    if chain is not None:

        def _chain_factory(_config: Config) -> LLMClient:
            return chain

        selected_factory = _chain_factory
    agent = create_cli_agent(
        selected_config,
        selected_factory,
        conversation_id=conversation_id,
        profile=profile,
        runtime_metadata={"model_selection": runtime.selection.to_dict()},
        run_budget=_model_run_budget(service, runtime),
        usage_channel=channel,
    )
    return agent, selected_config, runtime


def _model_run_budget(
    service: ModelProfileService,
    runtime: ResolvedModelRuntime,
) -> RunBudget | None:
    profile = service.store.get(runtime.selection.requested_profile_id)
    if profile.max_cost_per_turn is None:
        return None
    return RunBudget(
        max_cost=ExactCost(
            profile.max_cost_per_turn,
            currency="USD",
            pricing_known=True,
        )
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
    | MoonshotProvider
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
    if provider == "moonshot":
        return MoonshotProvider(model=model)
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
    return ", ".join(
        f"{provider.provider}:{provider.model}"
        for provider in config.llm_fallback_providers
    )


def run_chat_loop(
    agent: Agent,
    *,
    config: Config | None = None,
    terminal: TerminalUI | None = None,
    agent_factory: Callable[[str], Agent] | None = None,
    model_selector: Callable[[str, str], tuple[Agent, Config, str]] | None = None,
    model_profile_id: str | None = None,
    active_model_profile_id: str | None = None,
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
    session_store = (
        SQLiteSessionStore(config.store_path) if config is not None else None
    )
    prompt_history = PromptHistory.create(enabled=input_func is input)
    if hasattr(prompt_history, "configure_completion"):
        prompt_history.configure_completion(command_completion_candidates())
    _load_prompt_history(prompt_history, session_store, agent)
    if config is not None:
        output_func(terminal.banner(config, agent))
    else:
        output_func("ChulkHarness CLI")
    output_func(terminal.hint())

    def switch_runtime(next_agent: Agent, next_config: Config | None = None) -> None:
        nonlocal config, session_store
        progress_reporter.close()
        previous_agent = command_context.agent
        progress_reporter.agent = next_agent
        progress_reporter.previous_callback = next_agent.event_callback
        next_agent.event_callback = progress_reporter.callback
        next_agent.permission_callback = permission_callback
        command_context.agent = next_agent
        if next_config is not None:
            config = next_config
            command_context.config = next_config
            progress_reporter.config = next_config
            session_store = SQLiteSessionStore(next_config.store_path)
            command_context.session_store = session_store
        _load_prompt_history(prompt_history, session_store, next_agent)
        previous_agent.close()

    def switch_agent(next_agent: Agent) -> None:
        switch_runtime(next_agent)

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
        model_profile_id=model_profile_id,
        active_model_profile_id=active_model_profile_id,
        model_selector=model_selector,
        switch_runtime=lambda next_agent, next_config: switch_runtime(
            next_agent,
            next_config,
        ),
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
        except ChulkError as exc:
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
            output_func(
                terminal.warning(
                    "A plan is waiting for approval. Use /approve to execute it or /reject to cancel it."
                )
            )
            continue
        if command_context.agent.has_resumable_plan():
            output_func(
                terminal.warning(
                    "An approved plan is waiting to continue. Use /approve to resume it."
                )
            )
            continue

        try:
            assistant_response = command_context.agent.run_turn(user_message)
        except LLMError as exc:
            error_func(terminal.error(f"error: {exc}"))
            return 1
        except ChulkError as exc:
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
            getattr(args, "path", None),
            execute_fixture_path=getattr(args, "execute_fixture", None),
            json_output=args.json_output,
            output_path=getattr(args, "output", None),
            force=bool(getattr(args, "force", False)),
            max_bytes=getattr(args, "max_bytes", None),
            max_events=getattr(args, "max_events", None),
            unbounded=bool(getattr(args, "unbounded", False)),
            output_func=output_func,
            error_func=error_func,
        )

    try:
        base_config = load_cli_config()
    except (OSError, ValueError, LLMConfigurationError) as exc:
        if args.command == "exec" and getattr(args, "json_output", False):
            output_func(
                json_text(
                    {"ok": False, "status": "configuration_error", "error": str(exc)}
                )
            )
        else:
            error_func(terminal.error(f"configuration error: {exc}"))
        return EXIT_CONFIGURATION_ERROR

    try:
        profile_factory = ProfileRuntimeFactory(base_config)
        if args.command == "profile":
            return run_profile_command(
                args.profile_command,
                store=profile_factory.profile_store,
                profile_id=getattr(args, "profile_id", None),
                project_root=getattr(args, "project_root", None),
                permission_profile=getattr(args, "permission_profile", None),
                model_profile_id=getattr(args, "model_profile", "default"),
                execution_backend_id=getattr(args, "execution_backend", "host"),
                allowed_skills=(
                    tuple(args.allowed_skills)
                    if getattr(args, "allowed_skills", None) is not None
                    else None
                ),
                allowed_mcp_servers=(
                    tuple(args.allowed_mcp_servers)
                    if getattr(args, "allowed_mcp_servers", None) is not None
                    else None
                ),
                credential_environment_names=tuple(getattr(args, "credential_env", ())),
                system_prompt=getattr(args, "system_prompt", None),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        resolved_profile = profile_factory.resolve_cli(getattr(args, "profile", None))
        config = resolved_profile.config
        profile = resolved_profile.profile
        if args.command == "server":
            from chulk.server.lifecycle import run_server_command

            return run_server_command(
                args.server_command,
                config=base_config,
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 8765),
                allow_remote=bool(getattr(args, "allow_remote", False)),
                enable_eval_dashboard=bool(getattr(args, "eval_dashboard", False)),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "tui":
            from chulk.cli.tui import run_tui_command

            return run_tui_command(
                config=base_config,
                profile_id=profile.id,
                url=args.url,
                token_file=getattr(args, "token_file", None),
                refresh_seconds=args.refresh_seconds,
                no_color=bool(getattr(args, "no_color", False)),
                error_func=error_func,
            )
        if args.command == "gateway":
            from chulk.telegram.main import run_telegram_gateway

            control_path = base_config.runtime_dir / "control.sqlite"
            adapter_name = getattr(args, "adapter", "telegram")
            account_id = getattr(args, "account", "primary")

            def start_gateway() -> int:
                if adapter_name == "discord":
                    from chulk.discord.main import run_discord_gateway

                    return run_discord_gateway(
                        config,
                        control_db_path=control_path,
                        profile_runtime_factory=profile_factory,
                        account_id=account_id,
                    )
                return run_telegram_gateway(
                    config,
                    control_db_path=control_path,
                    profile_runtime_factory=profile_factory,
                )

            return run_gateway_command(
                args.gateway_command,
                ledger=SQLiteGatewayLedger(control_path),
                router=SQLiteGatewayRouter(control_path),
                profile_store=profile_factory.profile_store,
                start_func=start_gateway,
                adapter=adapter_name,
                account_id=account_id,
                route_command=getattr(args, "route_command", None),
                route_id=getattr(args, "route_id", None),
                profile_id=getattr(args, "profile", None),
                principal_id=getattr(args, "principal", None),
                destination_id=getattr(args, "destination", None),
                thread_id=getattr(args, "thread", None),
                pairing_ttl_seconds=getattr(args, "ttl_seconds", 600),
                include_disabled=bool(
                    getattr(args, "include_disabled", False)
                ),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "plugins":
            return run_plugin_command(
                args.plugin_command,
                registry=LocalPluginRegistry(
                    config.runtime_dir,
                    profile_id=profile.id,
                ),
                path=getattr(args, "path", None),
                approved_by=getattr(args, "approved_by", None),
                plugin_name=getattr(args, "plugin_name", None),
                repository_url=getattr(args, "repository_url", None),
                commit_sha=getattr(args, "commit", None),
                allowed_git_hosts=tuple(
                    getattr(args, "allowed_git_hosts", ())
                ),
                approve_authority_changes=bool(
                    getattr(args, "approve_authority_changes", False)
                ),
                reason=getattr(args, "reason", None),
                revoked_by=getattr(args, "revoked_by", None),
                catalog_path=getattr(args, "catalog_path", None),
                query=getattr(args, "query", None),
                category=getattr(args, "category", None),
                version=getattr(args, "version", None),
                limit=int(getattr(args, "limit", 50)),
                acknowledge_host_authority=bool(
                    getattr(
                        args,
                        "acknowledge_host_authority",
                        False,
                    )
                ),
                granted_capabilities=(
                    tuple(args.granted_capabilities)
                    if getattr(args, "granted_capabilities", None)
                    is not None
                    else None
                ),
                json_output=bool(
                    getattr(args, "json_output", False)
                ),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "usage":
            return run_usage_command(
                args.usage_command,
                ledger=UsageLedger(
                    config.store_path,
                    profile_id=profile.id,
                ),
                start=getattr(args, "start", None),
                end=getattr(args, "end", None),
                group_by=getattr(args, "by", None),
                resource_kind=getattr(args, "resource_kind", None),
                channel=getattr(args, "channel", None),
                limit=getattr(args, "limit", 100),
                cursor=getattr(args, "cursor", None),
                output_path=getattr(args, "output", None),
                export_format=getattr(args, "format", "json"),
                max_entries=getattr(args, "max_entries", 10_000),
                force=bool(getattr(args, "force", False)),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "goal":
            from chulk.cli.goals import run_goal_command
            from chulk.goals import GoalService, GoalStore

            return run_goal_command(
                args.goal_command,
                service=GoalService(
                    GoalStore(
                        config.store_path,
                        profile_id=profile.id,
                    )
                ),
                session_store=SQLiteSessionStore(config.store_path),
                goal_id=getattr(args, "goal_id", None),
                conversation_id=getattr(args, "conversation_id", None),
                turn_id=getattr(args, "turn_id", None),
                step_id=getattr(args, "step_id", None),
                expected_revision=getattr(args, "revision", None),
                actor=getattr(args, "actor", "cli"),
                instruction=(
                    " ".join(args.instruction)
                    if getattr(args, "instruction", None)
                    else None
                ),
                reason=getattr(args, "reason", None),
                status=getattr(args, "status", None),
                limit=getattr(args, "limit", 100),
                max_model_calls=getattr(args, "max_model_calls", None),
                max_tool_calls=getattr(args, "max_tool_calls", None),
                max_tokens=getattr(args, "max_tokens", None),
                max_cost=getattr(args, "max_cost", None),
                deadline=getattr(args, "deadline", None),
                output_path=getattr(args, "output", None),
                force=bool(getattr(args, "force", False)),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "child":
            from chulk.children import ChildTaskStore, DelegationService
            from chulk.cli.children import run_child_command

            return run_child_command(
                args.child_command,
                service=DelegationService(
                    ChildTaskStore(
                        config.store_path,
                        profile_id=profile.id,
                    )
                ),
                task_id=getattr(args, "task_id", None),
                expected_revision=getattr(args, "revision", None),
                actor=getattr(args, "actor", "cli"),
                reason=getattr(args, "reason", None),
                status=getattr(args, "status", None),
                goal_id=getattr(args, "goal", None),
                parent_task_id=getattr(args, "parent", None),
                limit=getattr(args, "limit", 100),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "automation":
            from chulk.cli.automations import run_automation_command
            from chulk.scheduling import SQLiteScheduleStore

            return run_automation_command(
                args.automation_command,
                store=SQLiteScheduleStore(
                    config.store_path,
                    profile_id=profile.id,
                ),
                job_id=getattr(args, "job_id", None),
                expected_revision=getattr(args, "revision", None),
                idempotency_key=getattr(args, "idempotency_key", None),
                status=getattr(args, "status", None),
                limit=getattr(args, "limit", 100),
                actor=getattr(args, "actor", "cli"),
                prompt=getattr(args, "prompt", None),
                run_at=getattr(args, "run_at", None),
                timezone_name=getattr(args, "timezone", "UTC"),
                interval_seconds=getattr(args, "interval_seconds", None),
                cron=getattr(args, "cron", None),
                rrule=getattr(args, "rrule", None),
                trigger_kind=getattr(args, "kind", None),
                source_resource_id=getattr(args, "source", None),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "session":
            return run_session_command(
                args.session_command,
                store=SQLiteSessionStore(config.store_path),
                profile_id=profile.id,
                query=(
                    " ".join(args.query)
                    if getattr(args, "query", None) is not None
                    else None
                ),
                conversation_id=getattr(args, "conversation_id", None),
                ordinal=getattr(args, "ordinal", None),
                before=getattr(args, "before", 3),
                after=getattr(args, "after", 3),
                limit=getattr(args, "limit", 20),
                cursor=getattr(args, "cursor", None),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        if args.command == "eval":
            from chulk.cli.evals import run_eval_command
            from chulk.evals import SQLiteEvalStore

            eval_command = args.eval_command
            if eval_command == "baseline":
                eval_command = f"baseline-{args.eval_baseline_command}"
            return run_eval_command(
                eval_command,
                store=SQLiteEvalStore(config.store_path),
                suite_ref=getattr(args, "suite", None),
                report_id=getattr(args, "report_id", None),
                baseline_id=getattr(args, "baseline_id", None),
                suite_name=getattr(args, "suite_name", None),
                resume_from=getattr(args, "resume_from", None),
                tags=tuple(getattr(args, "tag", ())),
                trials=getattr(args, "trials", None),
                concurrency=getattr(args, "concurrency", None),
                timeout_seconds=getattr(args, "timeout_seconds", None),
                mode=getattr(args, "mode", None),
                provider=getattr(args, "provider", None),
                model=getattr(args, "model", None),
                max_total_cost=getattr(args, "max_total_cost", None),
                allow_unknown_cost=bool(getattr(args, "allow_unknown_cost", False)),
                fail_fast=bool(getattr(args, "fail_fast", False)),
                status=getattr(args, "status", None),
                target_name=getattr(args, "target_name", None),
                started_after=getattr(args, "started_after", None),
                started_before=getattr(args, "started_before", None),
                limit=getattr(args, "limit", 100),
                offset=getattr(args, "offset", 0),
                min_baseline_coverage=getattr(
                    args,
                    "min_baseline_coverage",
                    None,
                ),
                output_path=getattr(args, "output", None),
                export_format=getattr(args, "format", None),
                init_path=getattr(args, "path", None),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        model_service = ModelProfileService(
            ModelProfileStore(
                base_config.runtime_dir / "control.sqlite",
                base_config=base_config,
            ),
            ModelProfileValidator(base_config),
        )
        if args.command == "model":
            return run_model_command(
                args.model_command,
                service=model_service,
                agent_profile=profile,
                profile_id=getattr(args, "model_profile_id", None),
                provider=getattr(args, "provider", None),
                model=getattr(args, "model", None),
                credential_ref=getattr(args, "credential_ref", None),
                endpoint_ref=getattr(args, "endpoint_ref", None),
                fallback_profile_ids=tuple(getattr(args, "fallback", ())),
                required_capabilities=ModelCapabilityRequirements(
                    structured_output=bool(
                        getattr(args, "require_structured_output", False)
                    ),
                    json_mode=bool(getattr(args, "require_json_mode", False)),
                    streaming=bool(getattr(args, "require_streaming", False)),
                    native_tool_calling=bool(
                        getattr(args, "require_native_tools", False)
                    ),
                    hosted_mcp_tools=bool(getattr(args, "require_hosted_mcp", False)),
                ),
                context_window_tokens=getattr(args, "context_window_tokens", None),
                response_reserve_tokens=getattr(
                    args,
                    "response_reserve_tokens",
                    None,
                ),
                max_output_tokens=getattr(args, "max_output_tokens", None),
                max_cost_per_turn=getattr(args, "max_cost_per_turn", None),
                channel=getattr(args, "channel", None),
                probe=bool(getattr(args, "probe", False)),
                json_output=bool(getattr(args, "json_output", False)),
                output_func=output_func,
                error_func=error_func,
            )
        requested_model_profile_id = getattr(args, "model_profile", None)
        config, _model_runtime, _model_chain = resolve_cli_model(
            base_config,
            config,
            profile,
            model_service,
            requested_profile_id=requested_model_profile_id,
            build_client=False,
        )
    except (
        OSError,
        ProfileAlreadyExistsError,
        ProfileNotFoundError,
        ProfileOwnershipError,
        ModelProfileNotFoundError,
        ValueError,
    ) as exc:
        if args.command == "exec" and getattr(args, "json_output", False):
            output_func(
                json_text(
                    {"ok": False, "status": "configuration_error", "error": str(exc)}
                )
            )
        else:
            error_func(terminal.error(f"configuration error: {exc}"))
        return EXIT_CONFIGURATION_ERROR

    if args.show_config:
        output_func(format_config(config))
        return EXIT_OK

    if args.command == "exec" or args.once is not None:
        if args.resume or args.continue_session:
            error_func(
                terminal.error(
                    "configuration error: --resume and --continue are interactive-only"
                )
            )
            return EXIT_CONFIGURATION_ERROR
        message = " ".join(args.message) if args.command == "exec" else str(args.once)
        json_output = bool(getattr(args, "json_output", False))
        return run_exec_command(
            message,
            agent_factory=lambda: create_selected_cli_agent(
                base_config,
                resolved_profile.config,
                profile,
                model_service,
                llm_client_factory,
                requested_profile_id=requested_model_profile_id,
            )[0],
            json_output=json_output,
            output_func=output_func,
            error_func=error_func,
        )

    try:
        conversation_id = _resolve_startup_conversation(config, args)
    except (SessionNotFoundError, AmbiguousSessionError) as exc:
        error_func(terminal.error(f"session error: {exc}"))
        return EXIT_CONFIGURATION_ERROR
    selection_state = {"override": requested_model_profile_id}
    try:
        agent, config, _model_runtime = create_selected_cli_agent(
            base_config,
            resolved_profile.config,
            profile,
            model_service,
            llm_client_factory,
            conversation_id=conversation_id,
            requested_profile_id=selection_state["override"],
        )
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

    def select_interactive_model(
        model_profile_id: str,
        active_conversation_id: str,
    ) -> tuple[Agent, Config, str]:
        try:
            SQLiteSessionStore(config.store_path).get_conversation(
                active_conversation_id
            )
        except SessionNotFoundError:
            resumable_conversation_id = None
        else:
            resumable_conversation_id = active_conversation_id
        next_agent, next_config, next_runtime = create_selected_cli_agent(
            base_config,
            resolved_profile.config,
            profile,
            model_service,
            llm_client_factory,
            requested_profile_id=model_profile_id,
            conversation_id=resumable_conversation_id,
        )
        try:
            model_service.use_for_agent(
                profile,
                model_profile_id,
                channel="cli",
            )
        except Exception:
            next_agent.close()
            raise
        selection_state["override"] = None
        return (
            next_agent,
            next_config,
            next_runtime.selection.selected_profile_id,
        )

    return run_chat_loop(
        agent,
        config=config,
        terminal=terminal,
        agent_factory=lambda conversation_id: create_selected_cli_agent(
            base_config,
            resolved_profile.config,
            profile,
            model_service,
            llm_client_factory,
            requested_profile_id=selection_state["override"],
            conversation_id=conversation_id,
        )[0],
        model_selector=select_interactive_model,
        model_profile_id=_model_runtime.selection.requested_profile_id,
        active_model_profile_id=_model_runtime.selection.selected_profile_id,
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

    def approve(
        request: PermissionRequest, record: PermissionDecisionRecord
    ) -> PermissionDecision:
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
