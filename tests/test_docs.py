from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_documentation_checker_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_docs.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "20 topics" in completed.stdout
    assert "trace hygiene" in completed.stdout


def test_index_routes_sdk_cli_and_repository_developers() -> None:
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "Embedding the SDK" in index
    assert "Operating the CLI" in index
    assert "Developing ChulkHarness" in index


def test_release_policy_covers_public_internal_async_and_external_boundaries() -> None:
    policy = (ROOT / "docs" / "release-policy.md").read_text(encoding="utf-8")
    sdk = (ROOT / "docs" / "sdk.md").read_text(encoding="utf-8")
    mcp = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")

    assert "chulk.__all__" in policy
    assert "chulk.testing" in policy
    assert "chulk._sdk" in policy and "internal" in policy
    assert "thread-backed" in sdk and "Cancellation" in sdk
    assert "trust boundary" in mcp and "external services" in mcp


def test_roadmap_stays_compact_ordered_and_deduplicated() -> None:
    roadmap = (ROOT / "TODO.md").read_text(encoding="utf-8")
    lines = roadmap.splitlines()

    assert len(lines) < 400
    assert "## Now: Maintenance Reliability" in roadmap
    assert "## Next: Reliability And SDK Depth" in roadmap
    assert "## Later: Product Directions" in roadmap

    task_pattern = re.compile(r"^\s*- \[[ xX]\]\s+(.*)$")
    tasks = [match.group(1).strip().casefold() for line in lines if (match := task_pattern.match(line))]
    duplicates = [task for task, count in Counter(tasks).items() if count > 1]

    assert not duplicates, f"Roadmap tasks must have one canonical owner: {duplicates}"
