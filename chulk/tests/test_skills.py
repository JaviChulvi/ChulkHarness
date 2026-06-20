"""Tests for lazy-loaded skills."""

from pathlib import Path

from chulk.skills import Skill, SkillRegistry, bundled_skills_dir


def write_skill(skills_dir: Path, name: str, content: str) -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_skill_registry_loads_metadata_without_full_content(tmp_path):
    skills_dir = tmp_path / "skills"
    write_skill(
        skills_dir,
        "shell",
        "# Shell Skill\n\nUse this skill when the user request requires terminal inspection or command execution.\n",
    )

    registry = SkillRegistry(skills_dir)
    registry.load_metadata()

    skill = registry.get_skill("shell")
    assert skill is not None
    assert skill.description.startswith("Use this skill when")
    assert "command" in skill.keywords
    assert skill.loaded_content is None


def test_skill_registry_supports_simple_front_matter(tmp_path):
    skills_dir = tmp_path / "skills"
    write_skill(
        skills_dir,
        "python",
        "\n".join(
            [
                "---",
                "name: python-coding",
                "description: Python implementation workflow.",
                "keywords: [python, pytest, refactor]",
                "owner: core",
                "---",
                "# Python Coding",
            ]
        ),
    )

    registry = SkillRegistry(skills_dir)
    registry.load_metadata()

    skill = registry.get_skill("python-coding")
    assert skill is not None
    assert skill.description == "Python implementation workflow."
    assert {"python", "pytest", "refactor"} <= set(skill.keywords)
    assert skill.metadata == {"owner": "core"}


def test_skill_registry_selects_and_lazy_loads_relevant_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "shell", "# Shell Skill\n\nUse this skill when commands are needed.\n")
    write_skill(skills_dir, "memory", "# Memory Skill\n\nUse this skill when memory should be saved.\n")

    registry = SkillRegistry(skills_dir)
    registry.load_metadata()
    selections = registry.select_skills("run a shell command")

    assert [selection.skill.name for selection in selections] == ["shell"]
    assert registry.get_skill("shell").loaded_content is None

    loaded = registry.load_selected_skills("run a shell command")

    assert [selection.skill.name for selection in loaded] == ["shell"]
    assert registry.get_skill("shell").loaded_content.startswith("# Shell Skill")
    assert registry.get_skill("memory").loaded_content is None


def test_skill_registry_respects_selection_limit(tmp_path):
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "shell", "# Shell Skill\n\nUse this skill when commands are needed.\n")
    write_skill(skills_dir, "files", "# Files Skill\n\nUse this skill when file edits are needed.\n")

    registry = SkillRegistry(skills_dir, max_skills=1)
    registry.load_metadata()
    selections = registry.select_skills("run a command and edit a file")

    assert len(selections) == 1


def test_skill_registry_loads_multiple_directories_with_project_override(tmp_path):
    project_skills_dir = tmp_path / ".chulk" / "skills"
    project_files_path = write_skill(
        project_skills_dir,
        "files",
        "# Files Skill\n\nProject-specific file workflow.\n",
    )

    registry = SkillRegistry(
        project_skills_dir,
        skills_dirs=(bundled_skills_dir(), project_skills_dir),
    )
    registry.load_metadata()

    assert {"files", "shell", "memory"} <= {skill.name for skill in registry.list_skills()}
    assert registry.get_skill("files").path == project_files_path
    assert registry.load_content("files").startswith("# Files Skill\n\nProject-specific")


def test_skill_registry_missing_skill_file_raises(tmp_path):
    missing_path = tmp_path / "missing" / "SKILL.md"
    registry = SkillRegistry(tmp_path / "skills")
    registry.register(Skill(name="missing", description="Missing skill", path=missing_path))

    try:
        registry.load_content("missing")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected missing skill file to fail")
