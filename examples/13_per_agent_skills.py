"""Scope a codebase skill catalog per agent."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chulk import Agent, AgentConfig, Skills  # noqa: E402
from chulk.config import Config  # noqa: E402


def write_skill(project_root: Path, name: str, content: str) -> None:
    skill_dir = project_root / ".chulk" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def prepare_demo_project(project_root: Path) -> None:
    write_skill(
        project_root,
        "code-review",
        "# Code Review Skill\n\nUse this skill when reviewing Python code or tests.\n",
    )
    write_skill(
        project_root,
        "sql",
        "# SQL Skill\n\nUse this skill when analyzing database queries or schema changes.\n",
    )
    write_skill(
        project_root,
        "team-style",
        "# Team Style Skill\n\nAlways keep recommendations concise and implementation-oriented.\n",
    )
    write_skill(
        project_root,
        "unused",
        "# Unused Skill\n\nUse this skill when drafting marketing copy.\n",
    )


def require_provider_credentials(config: Config) -> None:
    if config.llm_provider == "openai" and not configured(config.openai_api_key):
        raise SystemExit("Set OPENAI_API_KEY or choose another CHULK_LLM_PROVIDER.")
    if config.llm_provider == "deepseek" and not configured(config.deepseek_api_key):
        raise SystemExit("Set CHULK_DEEPSEEK_API_KEY, DEEPSEEK_API_KEY, or choose another provider.")
    if config.llm_provider == "openai-compatible":
        if not configured(config.openai_compatible_api_key):
            raise SystemExit("Set CHULK_OPENAI_COMPATIBLE_API_KEY.")
        if not configured(config.openai_compatible_base_url):
            raise SystemExit("Set CHULK_OPENAI_COMPATIBLE_BASE_URL.")
    if config.llm_provider == "openrouter" and not configured(config.openrouter_api_key):
        raise SystemExit("Set CHULK_OPENROUTER_API_KEY or OPENROUTER_API_KEY.")
    if config.llm_provider == "anthropic" and not configured(config.anthropic_api_key):
        raise SystemExit("Set CHULK_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.")
    if config.llm_provider == "bedrock":
        if not configured(config.bedrock_api_key):
            raise SystemExit(
                "Set CHULK_BEDROCK_API_KEY, BEDROCK_API_KEY, or "
                "AWS_BEARER_TOKEN_BEDROCK."
            )
        if not configured(config.bedrock_base_url):
            raise SystemExit("Set CHULK_BEDROCK_BASE_URL or legacy CHULK_BASE_URL.")
    if config.llm_provider == "gemini" and not configured(config.gemini_api_key):
        raise SystemExit(
            "Set CHULK_GEMINI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY."
        )


def configured(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def main() -> None:
    with TemporaryDirectory(prefix="chulk-skills-example-") as temp_dir:
        project_root = Path(temp_dir)
        prepare_demo_project(project_root)
        config = AgentConfig.from_env(project_root=project_root, runtime_dir=".chulk")
        runtime_config = config.to_config()
        require_provider_credentials(runtime_config)
        assistant = Agent(
            config=config,
            tools=[],
            skills=[
                Skills.only("code-review", "sql"),
                Skills.pin("team-style"),
            ],
        )
        response = assistant.run(
            "Review this Python snippet and suggest the smallest safe fix: "
            "def total(items): return sum(item.price for item in item)"
        )
        print("\n=== Assistant ===")
        print(response)
        print("\n=== Loaded Skills ===")
        print(", ".join(assistant.state.loaded_skill_names) or "none")
        print("\n=== Registered Skills ===")
        print(", ".join(skill.name for skill in assistant.skill_registry.list_skills()))
        print(f"\ntrace_path: {assistant.trace_path}")


if __name__ == "__main__":
    main()
