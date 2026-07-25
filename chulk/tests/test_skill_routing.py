"""Tests for deterministic, explainable skill routing."""

from pathlib import Path

from chulk.skills import SkillRegistry, explicit_skill_names


def write_skill(
    root: Path,
    name: str,
    *,
    description: str,
    manifest_fields: str = "",
) -> None:
    skill_root = root / name
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        f"""\
---
schema_version: 1
name: {name}
version: 1.0.0
description: {description}
{manifest_fields}---
# {name}

Instructions for {name}.
""",
        encoding="utf-8",
    )


def test_capability_filter_runs_before_keyword_selection(tmp_path):
    write_skill(
        tmp_path,
        "deploy",
        description="Deploy applications.",
        manifest_fields="""\
platforms: [linux]
required_tools: [run_cmd]
required_capabilities: [shell, network]
""",
    )
    registry = SkillRegistry(tmp_path)
    registry.load_metadata()
    registry.configure_environment(
        platform="darwin",
        available_tools={"run_cmd"},
        capabilities={"shell", "network"},
    )

    result = registry.route_skills("deploy the application")

    assert result.selections == ()
    assert result.decisions[0].to_dict()["reason"] == "platform_unavailable:darwin"
    assert result.decisions[0].status == "rejected"


def test_required_tools_and_capabilities_control_visibility(tmp_path):
    write_skill(
        tmp_path,
        "deploy",
        description="Deploy applications.",
        manifest_fields="""\
required_tools: [run_cmd, upload]
required_capabilities: [shell, network]
forbidden_capabilities: [restricted]
""",
    )
    registry = SkillRegistry(tmp_path)
    registry.load_metadata()

    registry.configure_environment(
        available_tools={"run_cmd"},
        capabilities={"shell", "network"},
    )
    assert registry.skill_visibility("deploy") == (False, "missing_tools:upload")

    registry.configure_environment(
        available_tools={"run_cmd", "upload"},
        capabilities={"shell"},
    )
    assert registry.skill_visibility("deploy") == (
        False,
        "missing_capabilities:network",
    )

    registry.configure_environment(
        available_tools={"run_cmd", "upload"},
        capabilities={"shell", "network", "restricted"},
    )
    assert registry.skill_visibility("deploy") == (
        False,
        "forbidden_capabilities:restricted",
    )

    registry.configure_environment(
        available_tools={"run_cmd", "upload"},
        capabilities={"shell", "network"},
    )
    assert registry.skill_visibility("deploy") == (True, "compatible")


def test_explicit_multiple_skills_and_bundle_includes_take_precedence(tmp_path):
    write_skill(
        tmp_path,
        "review",
        description="Review code.",
        manifest_fields="includes: [tests]\n",
    )
    write_skill(tmp_path, "tests", description="Run tests.")
    write_skill(tmp_path, "release", description="Prepare a release.")
    registry = SkillRegistry(tmp_path)
    registry.load_metadata()

    result = registry.route_skills(
        "/review /release prepare the change",
        limit=3,
    )

    assert explicit_skill_names("/review+/release prepare") == (
        "review",
        "release",
    )
    assert [selection.skill.name for selection in result.selections] == [
        "review",
        "tests",
        "release",
    ]
    assert [selection.stage for selection in result.selections] == [
        "explicit",
        "include",
        "explicit",
    ]
    assert result.explicit_skill_names == ("review", "release")


def test_pins_precede_keyword_matches_and_budget_is_explained(tmp_path):
    write_skill(tmp_path, "always", description="Always-on workflow.")
    write_skill(tmp_path, "review", description="Review Python code.")
    write_skill(tmp_path, "python", description="Implement Python code.")
    registry = SkillRegistry(tmp_path, max_skills=2)
    registry.load_metadata()

    result = registry.route_skills(
        "review Python code",
        pinned_names=("always",),
    )

    assert [selection.skill.name for selection in result.selections] == [
        "always",
        "review",
    ]
    omitted = {
        decision.skill_name: decision
        for decision in result.decisions
        if decision.status == "omitted"
    }
    assert omitted["python"].stage == "budget"
    assert omitted["python"].reason == "skill_count_limit"


def test_explicit_unknown_and_invisible_skills_return_deterministic_errors(tmp_path):
    write_skill(
        tmp_path,
        "network",
        description="Network workflow.",
        manifest_fields="required_capabilities: [network]\n",
    )
    registry = SkillRegistry(tmp_path)
    registry.load_metadata()
    registry.configure_environment(capabilities=set())

    result = registry.route_skills("/missing /network do work")

    assert [(error.skill_name, error.reason) for error in result.errors] == [
        ("missing", "unknown_skill"),
        ("network", "missing_capabilities:network"),
    ]


def test_keyword_fallback_is_deterministic(tmp_path):
    write_skill(tmp_path, "alpha", description="Review Python code.")
    write_skill(tmp_path, "beta", description="Review Python code.")
    registry = SkillRegistry(tmp_path)
    registry.load_metadata()

    first = registry.route_skills("review Python code")
    second = registry.route_skills("review Python code")

    assert [selection.skill.name for selection in first.selections] == [
        selection.skill.name for selection in second.selections
    ]
    assert [decision.to_dict() for decision in first.decisions] == [
        decision.to_dict() for decision in second.decisions
    ]


def test_optional_reranker_can_reorder_and_falls_back_on_invalid_output(tmp_path):
    write_skill(tmp_path, "alpha", description="Review Python code.")
    write_skill(tmp_path, "beta", description="Review Python code.")
    registry = SkillRegistry(
        tmp_path,
        reranker=lambda _request, selections: reversed(
            [selection.skill.name for selection in selections]
        ),
    )
    registry.load_metadata()

    reranked = registry.route_skills("review Python code")

    assert [selection.skill.name for selection in reranked.selections] == [
        "beta",
        "alpha",
    ]
    assert all(selection.stage == "rerank" for selection in reranked.selections)

    registry.reranker = lambda _request, _selections: ["missing"]
    fallback = registry.route_skills("review Python code")

    assert [selection.skill.name for selection in fallback.selections] == [
        "alpha",
        "beta",
    ]
