"""Non-agent CLI diagnostics, initialization, and trace commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from chulk.config import Config, load_config, resolve_cli_environment
from chulk.llm.capabilities import resolve_runtime_model_capabilities
from chulk.storage.private_files import write_private_text
from chulk.tracing import Trace, TraceFormatError


_PROVIDER_SDK_REQUIREMENTS = {
    "openai": ("openai", "openai", "openai"),
    "deepseek": ("openai", "openai", "openai"),
    "local": ("openai", "openai", "openai"),
    "openai-compatible": ("openai", "openai", "openai"),
    "openrouter": ("openai", "openai", "openai"),
    "anthropic": ("anthropic", "anthropic", "anthropic"),
    "bedrock": ("openai", "openai", "openai"),
    "gemini": ("google.genai", "google-genai", "gemini"),
}

_GITIGNORE_HEADING = "# Chulk runtime state"
_GITIGNORE_RULES = (
    ".env",
    ".env.*",
    "!.env.example",
    "!.chulk/",
    ".chulk/*",
    "!.chulk/mcp.json",
    "!.chulk/skills/",
    "!.chulk/skills/**",
    "traces/",
    "chulk/store.sqlite",
    "*.sqlite",
    "*.sqlite3",
    "*.sqlite-*",
    "*.sqlite3-*",
    "*.sqlite.bak",
    "*.sqlite3.bak",
    "*.sqlite.backup*",
    "*.sqlite3.backup*",
)


@dataclass(frozen=True)
class DiagnosticCheck:
    """One human- and machine-readable doctor result."""

    name: str
    status: str
    detail: str
    remedy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    """Collected local runtime diagnostics."""

    project_root: Path
    checks: tuple[DiagnosticCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project_root": str(self.project_root),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class InitChange:
    """One filesystem result from ``chulk init``."""

    path: Path
    action: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "action": self.action}


def run_doctor(*, environ: dict[str, str] | None = None) -> DoctorReport:
    """Run offline configuration, credential, runtime, and ignore checks."""
    env = resolve_cli_environment(environ)
    project_root = Path(env["CHULK_PROJECT_ROOT"]).expanduser().resolve()
    checks: list[DiagnosticCheck] = []
    try:
        config = load_config(env)
    except (OSError, ValueError) as exc:
        checks.append(
            DiagnosticCheck(
                "configuration",
                "fail",
                str(exc),
                "Correct the reported environment or .env value, then rerun chulk doctor.",
            )
        )
        return DoctorReport(project_root, tuple(checks))

    project_root = config.project_root
    checks.append(DiagnosticCheck("configuration", "pass", "configuration parsed successfully"))
    checks.append(_provider_check(config))
    checks.append(_model_check(config))
    checks.append(_runtime_check(config))
    checks.extend(_mcp_checks(config))
    checks.append(_gitignore_check(config))
    return DoctorReport(project_root, tuple(checks))


def format_doctor_report(report: DoctorReport) -> str:
    lines = ["Chulk doctor", f"  project  {report.project_root}"]
    markers = {"pass": "ok", "warn": "warn", "fail": "fail"}
    for check in report.checks:
        lines.append(f"  [{markers.get(check.status, check.status)}] {check.name}: {check.detail}")
        if check.remedy:
            lines.append(f"         fix: {check.remedy}")
    lines.append("  result   " + ("ready" if report.ok else "action required"))
    return "\n".join(lines)


def initialize_project(project_root: Path | str, *, mode: str = "coding-agent") -> tuple[InitChange, ...]:
    """Create safe project-local Chulk scaffolding without overwriting files."""
    root = Path(project_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    clean_mode = mode.strip().lower()
    if clean_mode not in {"sdk", "coding-agent", "read-only"}:
        raise ValueError("init mode must be sdk, coding-agent, or read-only")
    permission_profile = "workspace-write" if clean_mode == "coding-agent" else "read-only"
    changes: list[InitChange] = []

    runtime_dir = root / ".chulk"
    skills_dir = runtime_dir / "skills"
    mcp_path = runtime_dir / "mcp.json"
    env_example_path = root / ".env.example"
    gitignore_path = root / ".gitignore"
    for target in (runtime_dir, skills_dir, mcp_path, env_example_path, gitignore_path):
        _validate_init_target(root, target)

    for directory in (runtime_dir, skills_dir):
        existed = directory.exists()
        directory.mkdir(parents=True, exist_ok=True)
        changes.append(InitChange(directory, "exists" if existed else "created"))

    if mcp_path.exists():
        changes.append(InitChange(mcp_path, "exists"))
    else:
        mcp_path.write_text(json.dumps({"servers": []}, indent=2) + "\n", encoding="utf-8")
        changes.append(InitChange(mcp_path, "created"))

    if env_example_path.exists():
        changes.append(InitChange(env_example_path, "exists"))
    else:
        env_example_path.write_text(_env_example(permission_profile), encoding="utf-8")
        changes.append(InitChange(env_example_path, "created"))

    action = _ensure_gitignore(gitignore_path)
    changes.append(InitChange(gitignore_path, action))
    return tuple(changes)


def format_init_changes(project_root: Path | str, changes: tuple[InitChange, ...]) -> str:
    root = Path(project_root).expanduser().resolve()
    lines = ["Chulk initialized", f"  project  {root}"]
    for change in changes:
        try:
            shown_path = change.path.relative_to(root)
        except ValueError:
            shown_path = change.path
        lines.append(f"  {change.action:<7} {shown_path}")
    return "\n".join(lines)


def inspect_trace(
    path: Path | str,
    *,
    max_bytes: int | None = None,
    max_events: int | None = None,
    unbounded: bool = False,
) -> dict[str, Any]:
    return Trace.from_jsonl(
        path,
        max_bytes=max_bytes,
        max_events=max_events,
        unbounded=unbounded,
    ).summary()


def replay_trace(
    path: Path | str,
    *,
    max_bytes: int | None = None,
    max_events: int | None = None,
    unbounded: bool = False,
) -> dict[str, Any]:
    """Reconstruct recorded turns without running tools, models, or network calls."""
    return Trace.from_jsonl(
        path,
        max_bytes=max_bytes,
        max_events=max_events,
        unbounded=unbounded,
    ).replay()


def format_trace_summary(summary: dict[str, Any]) -> str:
    event_types = summary.get("event_types", {})
    type_text = ", ".join(f"{name} x{count}" for name, count in event_types.items())
    lines = [
        "Chulk trace",
        f"  path          {summary.get('path')}",
        f"  conversation  {summary.get('conversation_id')}",
        f"  schemas       {_format_schema_versions(summary.get('schema_versions'))}",
        f"  events        {summary.get('event_count')}",
        f"  bytes         {summary.get('source_byte_count')}",
        f"  limits        {'bounded' if summary.get('limits_applied') else 'trusted unbounded'}",
        f"  sessions      {summary.get('session_count')}",
        f"  turns         {summary.get('turn_count')}",
        f"  failures      {summary.get('failure_count')}",
        f"  started       {summary.get('started_at')}",
        f"  ended         {summary.get('ended_at')}",
        f"  event types   {type_text or 'none'}",
        f"  artifacts     {summary.get('artifact_count', 0)} "
        f"({summary.get('artifact_total_bytes', 0)} bytes)",
        f"  integrity     {_format_integrity(summary.get('artifact_integrity'))}",
    ]
    if summary.get("final_answer"):
        lines.extend(["  final answer", *[f"    {line}" for line in str(summary["final_answer"]).splitlines()]])
    lines.append("  warning       traces may contain sensitive runtime data")
    return "\n".join(lines)


def format_trace_replay(replay: dict[str, Any]) -> str:
    """Format a deterministic, explicitly non-executing trace reconstruction."""
    lines = [
        "Chulk trace replay",
        f"  path          {replay.get('path')}",
        f"  conversation  {replay.get('conversation_id')}",
        f"  schemas       {_format_schema_versions(replay.get('schema_versions'))}",
        f"  events        {replay.get('event_count')}",
        f"  sessions      {replay.get('session_count')}",
        f"  turns         {replay.get('turn_count')}",
        "  mode          read-only; no model, tool, or network execution",
    ]
    turns = replay.get("turns")
    if isinstance(turns, list):
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue
            lines.append(
                f"  turn {index}        {turn.get('turn_id')} [{turn.get('status', 'unknown')}]"
            )
            if turn.get("user_message") is not None:
                lines.extend(
                    [
                        "    user",
                        *[f"      {line}" for line in str(turn["user_message"]).splitlines()],
                    ]
                )
            lines.append(f"    model calls  {turn.get('model_request_count', 0)}")
            tool_calls = turn.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                lines.append("    tools")
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    lines.append(
                        f"      {tool_call.get('tool_name') or 'unknown'} [{tool_call.get('status')}]"
                    )
            if turn.get("final_answer") is not None:
                lines.extend(
                    [
                        "    answer",
                        *[f"      {line}" for line in str(turn["final_answer"]).splitlines()],
                    ]
                )
    lines.append("  warning       replay output may contain sensitive runtime data")
    return "\n".join(lines)


def _format_schema_versions(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "unknown"
    return ", ".join(str(item) for item in value)


def _format_integrity(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key} x{count}" for key, count in sorted(value.items()))


def export_trace_html(
    path: Path | str,
    *,
    output_path: Path | str | None = None,
    force: bool = False,
    max_bytes: int | None = None,
    max_events: int | None = None,
    unbounded: bool = False,
) -> Path:
    trace = Trace.from_jsonl(
        path,
        max_bytes=max_bytes,
        max_events=max_events,
        unbounded=unbounded,
    )
    destination = (
        Path(output_path).expanduser().absolute()
        if output_path is not None
        else trace.path.with_suffix(".html")
    )
    if destination.resolve() == trace.path.resolve():
        raise ValueError(f"Trace export output cannot overwrite the source trace: {trace.path}")
    if destination.exists() and not force:
        raise FileExistsError(f"Output already exists: {destination}. Pass --force to replace it.")
    write_private_text(
        destination,
        trace.to_html(),
        overwrite=force,
        private_parent=False,
    )
    return destination


def _provider_check(config: Config) -> DiagnosticCheck:
    providers = _configured_provider_models(config)
    missing_settings: list[str] = []
    missing_packages: list[str] = []
    for label, provider, _model in providers:
        missing_settings.extend(
            f"{label} {provider} ({setting})"
            for setting in _missing_provider_settings(config, provider)
        )
        if missing_package := _missing_provider_package(provider):
            missing_packages.append(f"{label} {provider} ({missing_package})")
    if missing_settings or missing_packages:
        details = []
        remedies = []
        if missing_settings:
            details.append(
                "required provider settings are missing for: "
                + ", ".join(missing_settings)
            )
            remedies.append("Set the listed variables in the environment or project .env.")
        if missing_packages:
            details.append(
                "required provider packages are missing for: "
                + ", ".join(missing_packages)
            )
            remedies.append("Install the listed provider extras.")
        return DiagnosticCheck(
            "provider",
            "fail",
            "; ".join(details),
            " ".join(remedies),
        )
    if len(providers) == 1:
        provider = providers[0][1]
        detail = (
            f"{provider} provider configured"
            if provider == "local"
            else f"{provider} provider requirements are set"
        )
        return DiagnosticCheck("provider", "pass", detail)
    return DiagnosticCheck(
        "provider",
        "pass",
        f"primary and {len(providers) - 1} fallback provider(s) are configured",
    )


def _missing_provider_package(provider: str) -> str | None:
    requirement = _PROVIDER_SDK_REQUIREMENTS.get(provider)
    if requirement is None:
        return None
    module_name, package_name, extra_name = requirement
    try:
        available = find_spec(module_name) is not None
    except (ImportError, ValueError):
        available = False
    if available:
        return None
    return f"{package_name}; install from the source checkout with: python -m pip install -e '.[{extra_name}]'"


def _missing_provider_settings(config: Config, provider: str) -> tuple[str, ...]:
    """Return unresolved environment requirements for one configured provider."""
    if provider == "openai":
        return () if _has_provider_value(config.openai_api_key) else ("OPENAI_API_KEY",)
    if provider == "deepseek":
        return (
            ()
            if _has_provider_value(config.deepseek_api_key)
            else ("CHULK_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY",)
        )
    if provider == "local":
        return ()
    if provider == "openai-compatible":
        missing = []
        if not _has_provider_value(config.openai_compatible_api_key):
            missing.append("CHULK_OPENAI_COMPATIBLE_API_KEY")
        if not _has_provider_value(config.openai_compatible_base_url):
            missing.append("CHULK_OPENAI_COMPATIBLE_BASE_URL")
        return tuple(missing)
    if provider == "openrouter":
        return (
            ()
            if _has_provider_value(config.openrouter_api_key)
            else ("CHULK_OPENROUTER_API_KEY or OPENROUTER_API_KEY",)
        )
    if provider == "anthropic":
        return (
            ()
            if _has_provider_value(config.anthropic_api_key)
            else ("CHULK_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY",)
        )
    if provider == "bedrock":
        missing = []
        if not _has_provider_value(config.bedrock_api_key):
            missing.append(
                "CHULK_BEDROCK_API_KEY, BEDROCK_API_KEY, or AWS_BEARER_TOKEN_BEDROCK"
            )
        if not _has_provider_value(config.bedrock_base_url):
            missing.append("CHULK_BEDROCK_BASE_URL or CHULK_BASE_URL")
        return tuple(missing)
    if provider == "gemini":
        return (
            ()
            if _has_provider_value(config.gemini_api_key)
            else ("CHULK_GEMINI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY",)
        )
    return ()


def _has_provider_value(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _model_check(config: Config) -> DiagnosticCheck:
    providers = _configured_provider_models(config)
    capabilities = []
    invalid: list[str] = []
    for label, provider, model in providers:
        try:
            capabilities.append(
                resolve_runtime_model_capabilities(
                    provider,
                    model,
                    local_context_window_tokens=config.local_context_window_tokens,
                )
            )
        except ValueError as exc:
            invalid.append(f"{label} {provider}/{model}: {exc}")
    if invalid:
        return DiagnosticCheck(
            "model",
            "fail",
            "invalid model configuration: " + "; ".join(invalid),
            "Set primary and fallback models to registered models.",
        )
    if len(providers) > 1:
        return DiagnosticCheck(
            "model",
            "pass",
            f"capability metadata is available for {len(providers)} configured models",
        )
    capabilities_for_primary = capabilities[0]
    return DiagnosticCheck(
        "model",
        "pass",
        f"{config.llm_provider}/{config.model}, {capabilities_for_primary.context_window_tokens} token context",
    )


def _runtime_check(config: Config) -> DiagnosticCheck:
    errors = [
        error
        for label, path in (
            ("runtime directory", config.runtime_dir),
            ("SQLite store directory", config.store_path.parent),
            ("trace directory", config.traces_dir),
        )
        if (error := _directory_write_error(label, path)) is not None
    ]
    if config.store_path.exists():
        if not config.store_path.is_file():
            errors.append(f"SQLite store path is not a file: {config.store_path}")
        elif not os.access(config.store_path, os.W_OK):
            errors.append(f"SQLite store path is not writable: {config.store_path}")
    if not errors:
        return DiagnosticCheck(
            "runtime",
            "pass",
            "runtime, SQLite store, and trace destinations are writable",
        )
    return DiagnosticCheck(
        "runtime",
        "fail",
        "; ".join(errors),
        "Choose writable runtime, SQLite store, and trace destinations.",
    )


def _mcp_checks(config: Config) -> list[DiagnosticCheck]:
    if not config.mcp_servers:
        return [DiagnosticCheck("mcp", "pass", "no MCP servers configured")]
    missing = [
        server.authorization_env
        for server in config.mcp_servers
        if server.authorization_env and not server.authorization
    ]
    if missing:
        names = ", ".join(str(name) for name in missing)
        return [
            DiagnosticCheck(
                "mcp",
                "fail",
                f"missing authorization environment variables: {names}",
                "Set the missing variables without putting secret values in mcp.json.",
            )
        ]
    return [DiagnosticCheck("mcp", "pass", f"{len(config.mcp_servers)} server(s) configured")]


def _gitignore_check(config: Config) -> DiagnosticCheck:
    git_root = _git_root(config.project_root)
    if git_root is None:
        return DiagnosticCheck("gitignore", "warn", "project is not inside a Git worktree")
    try:
        project_prefix = config.project_root.relative_to(git_root)
    except ValueError:
        project_prefix = Path()
    tracked = _tracked_runtime_paths(git_root, config)
    if tracked:
        return DiagnosticCheck(
            "gitignore",
            "fail",
            "runtime paths are already tracked by Git: " + ", ".join(tracked),
            "Remove runtime state from the index with git rm --cached, then keep the ignore rules.",
        )
    runtime_candidates = tuple(
        str(project_prefix / candidate)
        for candidate in (
            ".env",
            ".chulk/store.sqlite",
            ".chulk/store.sqlite-wal",
            ".chulk/store.sqlite.backup-20260724",
            ".chulk/traces/example.jsonl",
            ".chulk/traces/example_artifacts/art_example.txt",
            "traces/example.jsonl",
            "traces/example_artifacts/art_example.txt",
            "chulk/store.sqlite",
            "chulk/store.sqlite-wal",
            "chulk/store.sqlite.backup-20260724",
            "state.sqlite",
        )
    )
    missing = [
        candidate
        for candidate in runtime_candidates
        if not _git_ignores(git_root, candidate)
    ]
    declarative_candidates = tuple(
        str(project_prefix / candidate)
        for candidate in (
            ".chulk/mcp.json",
            ".chulk/skills/example/SKILL.md",
        )
    )
    blocked_config = [
        candidate
        for candidate in declarative_candidates
        if _git_ignores(git_root, candidate)
    ]
    if not missing and not blocked_config:
        return DiagnosticCheck(
            "gitignore",
            "pass",
            "declarative MCP/skill config is trackable; credentials and runtime state are ignored",
        )
    details: list[str] = []
    if missing:
        details.append("runtime paths are not fully ignored: " + ", ".join(missing))
    if blocked_config:
        details.append(
            "declarative project config is incorrectly ignored: "
            + ", ".join(blocked_config)
        )
    return DiagnosticCheck(
        "gitignore",
        "fail",
        "; ".join(details),
        "Run chulk init to install the narrow declarative-config and runtime-state rules.",
    )


def _git_root(project_root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(result.stdout.strip()).resolve() if result.returncode == 0 and result.stdout.strip() else None


def _git_ignores(git_root: Path, relative_path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", "--", relative_path],
            cwd=git_root,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _configured_provider_models(config: Config) -> tuple[tuple[str, str, str], ...]:
    return (
        ("primary", config.llm_provider, config.model),
        *(
            (f"fallback #{index}", fallback.provider, fallback.model)
            for index, fallback in enumerate(config.llm_fallback_providers, start=1)
        ),
    )


def _directory_write_error(label: str, path: Path) -> str | None:
    if path.exists():
        if not path.is_dir():
            return f"{label} is not a directory: {path}"
        if not os.access(path, os.W_OK):
            return f"{label} is not writable: {path}"
        return None
    if path.is_symlink():
        return f"{label} is a broken symlink: {path}"

    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        return f"parent for {label} is not a directory: {ancestor}"
    if not os.access(ancestor, os.W_OK):
        return f"parent for {label} is not writable: {ancestor}"
    return None


def _tracked_runtime_paths(git_root: Path, config: Config) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=git_root,
            check=False,
            capture_output=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()

    runtime_prefix = _relative_to_git_root(config.runtime_dir, git_root)
    trace_prefix = _relative_to_git_root(config.traces_dir, git_root)
    store_path = _relative_to_git_root(config.store_path, git_root)
    mcp_path = _relative_to_git_root(config.mcp_config_path, git_root)
    skills_prefix = _relative_to_git_root(config.skills_dir, git_root)
    project_prefix = _relative_to_git_root(config.project_root, git_root)
    tracked: list[str] = []
    for raw_path in result.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not raw_path:
            continue
        path = Path(raw_path)
        project_path = None
        if project_prefix is not None:
            try:
                project_path = path.relative_to(project_prefix)
            except ValueError:
                pass
        if (
            _is_sensitive_project_path(project_path)
            or path == store_path
            or _is_at_or_below(path, trace_prefix)
            or (
                _is_at_or_below(path, runtime_prefix)
                and path != mcp_path
                and not _is_at_or_below(path, skills_prefix)
            )
        ):
            tracked.append(raw_path)
    return tuple(sorted(tracked))


def _is_at_or_below(path: Path, prefix: Path | None) -> bool:
    return prefix is not None and (path == prefix or prefix in path.parents)


def _is_sensitive_project_path(path: Path | None) -> bool:
    if path is None:
        return False
    name = path.name.lower()
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    return (
        ".sqlite" in name
        or ".sqlite3" in name
        or name.endswith((".bak", ".backup"))
        or ".backup-" in name
    )


def _relative_to_git_root(path: Path, git_root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(git_root)
    except ValueError:
        return None


def _env_example(permission_profile: str) -> str:
    return "\n".join(
        [
            "OPENAI_API_KEY=",
            "DEEPSEEK_API_KEY=",
            "CHULK_LLM_PROVIDER=openai",
            "CHULK_MODEL=",
            f"CHULK_PERMISSION_PROFILE={permission_profile}",
            "CHULK_RUNTIME_DIR=.chulk",
            "",
        ]
    )


def _validate_init_target(project_root: Path, target: Path) -> None:
    if target.is_symlink():
        raise ValueError(f"Refusing to initialize symlinked path: {target}")
    try:
        target.resolve(strict=False).relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to initialize path outside project root: {target}") from exc


def _ensure_gitignore(path: Path) -> str:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    managed_lines = {_GITIGNORE_HEADING, *_GITIGNORE_RULES}
    unmanaged_lines = [
        line
        for line in existing.splitlines()
        if line.strip() not in managed_lines
    ]
    while unmanaged_lines and not unmanaged_lines[-1].strip():
        unmanaged_lines.pop()
    managed_block = "\n".join((_GITIGNORE_HEADING, *_GITIGNORE_RULES))
    content = (
        "\n".join(unmanaged_lines) + "\n\n" + managed_block + "\n"
        if unmanaged_lines
        else managed_block + "\n"
    )
    if content == existing:
        return "exists"
    path.write_text(content, encoding="utf-8")
    return "created" if not existing else "updated"


__all__ = [
    "DiagnosticCheck",
    "DoctorReport",
    "InitChange",
    "TraceFormatError",
    "export_trace_html",
    "format_doctor_report",
    "format_init_changes",
    "format_trace_replay",
    "format_trace_summary",
    "initialize_project",
    "inspect_trace",
    "replay_trace",
    "run_doctor",
]
