"""Import-boundary tests for the top-level compatibility package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"


def _run_fresh_import(source: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SOURCE_ROOT)
        if not existing
        else os.pathsep.join((str(SOURCE_ROOT), existing))
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_plain_package_import_loads_only_the_version_contract() -> None:
    completed = _run_fresh_import(
        """
        import json
        import sys

        before = set(sys.modules)
        import chulk

        loaded = sorted(
            name
            for name in sys.modules
            if name not in before
            and (name == "chulk" or name.startswith("chulk."))
        )
        print(json.dumps(loaded))
        print(len(chulk.__all__))
        print("Agent" in vars(chulk))
        """
    )

    loaded, export_count, agent_materialized = completed.stdout.splitlines()
    assert json.loads(loaded) == ["chulk", "chulk._version"]
    assert export_count == "626"
    assert agent_materialized == "False"


def test_all_stable_exports_resolve_lazily_with_compatible_aliases() -> None:
    completed = _run_fresh_import(
        """
        from importlib import import_module

        import chulk

        namespace = {}
        exec("from chulk import *", namespace)

        assert all(name in namespace for name in chulk.__all__)
        assert all(name in dir(chulk) for name in chulk.__all__)
        api = import_module("chulk.api")
        root_names = set(chulk.__all__)
        api_names = set(api.__all__)
        assert len(root_names) == len(chulk.__all__)
        assert len(api_names) == len(api.__all__)
        assert root_names - api_names == {
            "Authoring", "PermissionDecision", "PermissionDecisionRecord",
            "PermissionRequest", "Plugins", "Research", "Skills", "Tool",
            "ToolPermissionLevel", "Tools", "__version__", "plugins", "skills",
            "tool", "tools",
        }
        assert api_names - root_names == {
            "ContentIntegrityError", "ContentLimitError", "ContentNotFoundError",
            "ContentOwnershipError",
        }
        for name in root_names & api_names:
            assert getattr(chulk, name) is getattr(api, name), name
            assert namespace[name] is getattr(api, name), name
        assert namespace["Tool"] is namespace["tool"]
        assert namespace["Tools"] is namespace["tools"]
        assert namespace["Skills"] is namespace["skills"]
        assert namespace["Plugins"] is namespace["plugins"]
        try:
            chulk.not_a_public_export
        except AttributeError:
            pass
        else:
            raise AssertionError("unknown package attributes must fail")
        print("ok")
        """
    )

    assert completed.stdout.strip() == "ok"


def test_advanced_api_imports_without_root_import_ordering() -> None:
    completed = _run_fresh_import(
        """
        import chulk.api

        assert chulk.api.Agent
        assert chulk.api.RunStore
        print("ok")
        """
    )

    assert completed.stdout.strip() == "ok"


def test_tool_package_keeps_public_refs_after_implementation_imports() -> None:
    completed = _run_fresh_import(
        """
        import chulk.telegram.bot
        from chulk import tools
        from chulk.tools.public import ToolRef

        ref_names = (
            "calculator",
            "read_file",
            "list_files",
            "search_files",
            "search_memory",
            "list_memories",
            "summarize_memories",
        )
        assert all(isinstance(getattr(tools, name), ToolRef) for name in ref_names)
        print("ok")
        """
    )

    assert completed.stdout.strip() == "ok"
