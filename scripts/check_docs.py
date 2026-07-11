#!/usr/bin/env python3
"""Validate the maintained documentation, examples, and scrubbed trace."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TOPICS = (
    "quickstart.md",
    "sdk.md",
    "configuration.md",
    "providers.md",
    "tools.md",
    "permissions.md",
    "skills.md",
    "memory.md",
    "events.md",
    "mcp.md",
    "tracing.md",
    "safety.md",
    "sdk-errors.md",
    "release-policy.md",
)
LINK_SOURCES = tuple(DOCS / name for name in ("index.md", *TOPICS)) + (
    ROOT / "README.md",
    ROOT / "examples" / "README.md",
    ROOT / "examples" / "repo_review_bot" / "README.md",
)
SAMPLE_TRACE = ROOT / "examples" / "repo_review_bot" / "sample-trace.jsonl"
EXPECTED_TRACE_TYPES = (
    "turn_started",
    "model_request_started",
    "tool_call_started",
    "tool_call_completed",
    "model_request_started",
    "final_answer",
    "turn_finished",
)


def check_manifest() -> None:
    missing = [name for name in TOPICS if not (DOCS / name).is_file()]
    _require(not missing, f"missing documentation topics: {', '.join(missing)}")
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    unlinked = [name for name in TOPICS if f"]({name})" not in index]
    _require(not unlinked, f"docs/index.md does not directly link: {', '.join(unlinked)}")


def check_links() -> None:
    failures: list[str] = []
    pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    for source in LINK_SOURCES:
        _require(source.is_file(), f"missing link source: {source.relative_to(ROOT)}")
        for raw_target in pattern.findall(source.read_text(encoding="utf-8")):
            target = raw_target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{source.relative_to(ROOT)} -> {raw_target}")
    _require(not failures, "broken relative links:\n  " + "\n  ".join(failures))


def check_public_policy() -> None:
    import chulk
    import chulk.testing as testing

    policy = (DOCS / "release-policy.md").read_text(encoding="utf-8")
    for label in ("public-stable", "public-provisional", "internal", "trace-only"):
        _require(label in policy, f"release policy is missing the {label} label")
    _require("chulk.__all__" in policy, "release policy must govern all top-level public exports")
    _require(bool(chulk.__all__), "chulk.__all__ must declare the top-level public surface")
    _require(
        testing.__all__ == ["ScriptedLLMClient"],
        "chulk.testing must expose only ScriptedLLMClient",
    )
    _require(
        "ScriptedLLMClient" not in chulk.__all__,
        "ScriptedLLMClient must not become a top-level export",
    )


def check_boundaries() -> None:
    combined = "\n".join((DOCS / name).read_text(encoding="utf-8").lower() for name in TOPICS)
    requirements = {
        "async cancellation": ("cancellation", "async"),
        "thread-backed compatibility": ("thread-backed",),
        "serialized facade work": ("serializes",),
        "MCP external trust": ("external services", "trust boundary"),
        "untrusted model input": ("untrusted input",),
        "sensitive traces": ("sensitive diagnostic",),
    }
    for description, terms in requirements.items():
        _require(all(term in combined for term in terms), f"documentation is missing {description}")


def check_public_example_imports() -> None:
    source = ROOT / "examples" / "repo_review_bot" / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    allowed = {"chulk", "chulk.testing"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("chulk"):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names if alias.name.startswith("chulk"))
    _require(imported <= allowed, f"review bot imports non-public Chulk modules: {sorted(imported - allowed)}")


def check_command_references() -> None:
    combined = "\n".join(source.read_text(encoding="utf-8") for source in LINK_SOURCES)
    commands = {
        "python examples/00_sdk_quickstart.py": ROOT / "examples" / "00_sdk_quickstart.py",
        "python examples/repo_review_bot/app.py": ROOT / "examples" / "repo_review_bot" / "app.py",
        "python scripts/check_docs.py": ROOT / "scripts" / "check_docs.py",
    }
    for command, target in commands.items():
        _require(command in combined, f"maintained docs do not reference: {command}")
        _require(target.is_file(), f"documented command target is missing: {target.relative_to(ROOT)}")


def check_sample_trace() -> None:
    from chulk.tracing.reader import Trace

    text = SAMPLE_TRACE.read_text(encoding="utf-8")
    forbidden = (
        r"/Users/",
        r"/home/",
        r"[A-Za-z]:\\",
        r"sk-[A-Za-z0-9]",
        r"(?i)(api[_-]?key|authorization|password)\s*[:=]\s*[^\s\"']+",
    )
    for pattern in forbidden:
        _require(re.search(pattern, text) is None, f"sample trace contains forbidden pattern: {pattern}")

    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = Trace.from_jsonl(SAMPLE_TRACE)
    _require(len(parsed.events) == len(events), "trace reader did not parse every sample event")
    _require(tuple(event.get("type") for event in events) == EXPECTED_TRACE_TYPES, "sample trace sequence changed")
    for index, event in enumerate(events, start=1):
        _require(set(event) == {"created_at", "payload", "type"}, f"trace event {index} has an invalid envelope")
        _require(isinstance(event["payload"], dict), f"trace event {index} payload must be an object")


def main() -> int:
    checks = (
        check_manifest,
        check_links,
        check_public_policy,
        check_boundaries,
        check_public_example_imports,
        check_command_references,
        check_sample_trace,
    )
    try:
        for check in checks:
            check()
    except (AssertionError, OSError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"docs check failed: {exc}", file=sys.stderr)
        return 1
    print(f"docs check passed: {len(TOPICS)} topics, links, exports, boundaries, and trace hygiene")
    return 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    raise SystemExit(main())
