"""Filesystem-first source files for portable Chulk agent definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from chulk.authoring.compiler import AgentCompiler, CompiledAgentPackage, CompilerRequest
from chulk.authoring.models import BudgetDefinition, TriggerDefinition
from chulk.hosting import ExecutionScope


@dataclass(frozen=True, slots=True)
class AgentDirectory:
    """A reviewable directory that names host-published dependencies."""

    path: Path
    agent_id: str
    version: str
    goal: str
    prompt_name: str
    prompt_version: str
    model_profile_name: str
    model_profile_version: str
    approval_policy_name: str
    approval_policy_version: str
    tools: tuple[str, ...]
    instructions: str
    constraints: tuple[str, ...] = ()
    expected_outcomes: tuple[str, ...] = ()
    triggers: tuple[TriggerDefinition, ...] = ()
    budget: BudgetDefinition = BudgetDefinition()

    @classmethod
    def load(cls, path: Path | str) -> "AgentDirectory":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "agent.toml"
        instructions_path = root / "instructions.md"
        try:
            manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
            instructions = instructions_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"agent directory requires {exc.filename}") from exc
        if not instructions:
            raise ValueError("instructions.md cannot be empty")
        return cls(
            path=root,
            agent_id=_text(manifest, "agent", "id"),
            version=_text(manifest, "agent", "version"),
            goal=_text(manifest, "agent", "goal"),
            prompt_name=_text(manifest, "prompt", "name"),
            prompt_version=_text(manifest, "prompt", "version"),
            model_profile_name=_text(manifest, "model_profile", "name"),
            model_profile_version=_text(manifest, "model_profile", "version"),
            approval_policy_name=_text(manifest, "approval_policy", "name"),
            approval_policy_version=_text(manifest, "approval_policy", "version"),
            tools=_strings(manifest.get("agent", {}).get("tools"), "agent.tools", minimum=1),
            instructions=instructions,
            constraints=_strings(
                manifest.get("agent", {}).get("constraints", []), "agent.constraints"
            ),
            expected_outcomes=_strings(
                manifest.get("agent", {}).get("expected_outcomes", []),
                "agent.expected_outcomes",
            ),
            triggers=_triggers(manifest.get("agent", {}).get("triggers", [])),
            budget=_budget(manifest.get("agent", {}).get("budget", {})),
        )

    def to_request(self, compiler: AgentCompiler) -> CompilerRequest:
        """Resolve named host artifacts and return the compiler input."""
        if compiler.model_profiles is None or compiler.approval_policies is None:
            raise ValueError("directory authoring requires model and approval catalogs")
        prompt = compiler.prompts.reference(self.prompt_name, self.prompt_version)
        if compiler.prompts.resolve(prompt) != self.instructions:
            raise ValueError("instructions.md does not match the published prompt")
        return CompilerRequest(
            agent_id=self.agent_id,
            version=self.version,
            goal=self.goal,
            prompt=prompt,
            model_profile=compiler.model_profiles.reference(
                self.model_profile_name, self.model_profile_version
            ),
            approval_policy=compiler.approval_policies.reference(
                self.approval_policy_name, self.approval_policy_version
            ),
            selected_tools=self.tools,
            constraints=self.constraints,
            expected_outcomes=self.expected_outcomes,
            triggers=self.triggers,
            budget=self.budget,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the portable source fields without exposing instruction content."""
        return {
            "path": str(self.path),
            "agent_id": self.agent_id,
            "version": self.version,
            "goal": self.goal,
            "prompt": {"name": self.prompt_name, "version": self.prompt_version},
            "model_profile": {
                "name": self.model_profile_name,
                "version": self.model_profile_version,
            },
            "approval_policy": {
                "name": self.approval_policy_name,
                "version": self.approval_policy_version,
            },
            "tools": list(self.tools),
            "constraints": list(self.constraints),
            "expected_outcomes": list(self.expected_outcomes),
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "budget": self.budget.to_dict(),
        }

    def compile(
        self,
        compiler: AgentCompiler,
        *,
        caller_scope: ExecutionScope,
    ) -> CompiledAgentPackage:
        """Compile through Chulk's ordinary safe owner."""
        return compiler.compile(self.to_request(compiler), caller_scope=caller_scope)


def initialize_agent_directory(path: Path | str, *, agent_id: str) -> tuple[Path, ...]:
    """Create a minimal agent directory without replacing user files."""
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {
        root / "agent.toml": _manifest(agent_id),
        root / "instructions.md": "You are a concise, safe assistant.\n",
    }
    created: list[Path] = []
    for target, content in files.items():
        if not target.exists():
            target.write_text(content, encoding="utf-8")
            created.append(target)
    return tuple(created)


def _text(manifest: dict[str, Any], section: str, key: str) -> str:
    table = manifest.get(section)
    value = table.get(key) if isinstance(table, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"agent.toml requires [{section}].{key}")
    return value.strip()


def _strings(value: object, field: str, *, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"agent.toml {field} must be a list of strings")
    result = tuple(dict.fromkeys(item.strip() for item in value))
    if len(result) < minimum:
        raise ValueError(f"agent.toml {field} must not be empty")
    return result


def _triggers(value: object) -> tuple[TriggerDefinition, ...]:
    if not isinstance(value, list):
        raise ValueError("agent.toml triggers must be a list")
    triggers: list[TriggerDefinition] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("agent.toml triggers must contain tables")
        kind = item.get("kind")
        config = item.get("config", {})
        if not isinstance(kind, str) or not isinstance(config, dict):
            raise ValueError("agent.toml triggers require string kind and table config")
        triggers.append(TriggerDefinition(kind=kind, config=config))
    return tuple(triggers)


def _budget(value: object) -> BudgetDefinition:
    if not isinstance(value, dict):
        raise ValueError("agent.toml budget must be a table")
    return BudgetDefinition(**value)


def _manifest(agent_id: str) -> str:
    return f'''[agent]
id = "{agent_id}"
version = "1.0.0"
goal = "Describe the agent's bounded goal."
tools = ["replace-with-a-published-tool"]
expected_outcomes = ["Describe the expected result."]

[prompt]
name = "{agent_id}-prompt"
version = "1.0.0"

[model_profile]
name = "default-model"
version = "1.0.0"

[approval_policy]
name = "default-approval"
version = "1.0.0"
'''


__all__ = ["AgentDirectory", "initialize_agent_directory"]
