"""Import-boundary tests for the top-level compatibility package."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / 'src'

def _run_fresh_import(source: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(SOURCE_ROOT) if not existing else os.pathsep.join((str(SOURCE_ROOT), existing))
    return subprocess.run([sys.executable, '-c', textwrap.dedent(source)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)

def test_plain_package_import_loads_only_the_version_contract() -> None:
    completed = _run_fresh_import('\n        import json\n        import sys\n\n        before = set(sys.modules)\n        import chulk\n\n        loaded = sorted(\n            name\n            for name in sys.modules\n            if name not in before\n            and (name == "chulk" or name.startswith("chulk."))\n        )\n        print(json.dumps(loaded))\n        print(len(chulk.__all__))\n        print("Agent" in vars(chulk))\n        ')
    loaded, export_count, agent_materialized = completed.stdout.splitlines()
    assert json.loads(loaded) == ['chulk', 'chulk._version']
    assert export_count == '624'
    assert agent_materialized == 'False'

def test_all_stable_exports_resolve_lazily_with_compatible_aliases() -> None:
    completed = _run_fresh_import('\n        from importlib import import_module\n\n        import chulk\n\n        namespace = {}\n        exec("from chulk import *", namespace)\n\n        assert all(name in namespace for name in chulk.__all__)\n        assert all(name in dir(chulk) for name in chulk.__all__)\n        api = import_module("chulk.api")\n        root_names = set(chulk.__all__)\n        api_names = set(api.__all__)\n        assert len(root_names) == len(chulk.__all__)\n        assert len(api_names) == len(api.__all__)\n        assert root_names - api_names == {\n            "Authoring", "PermissionDecision", "PermissionDecisionRecord",\n            "PermissionRequest", "Plugins", "Research", "Skills", "Tool",\n            "ToolPermissionLevel", "Tools", "__version__", "plugins", "skills",\n            "tool", "tools",\n        }\n        assert api_names - root_names == {\n            "ContentIntegrityError", "ContentLimitError", "ContentNotFoundError",\n            "ContentOwnershipError",\n        }\n        for name in root_names & api_names:\n            assert getattr(chulk, name) is getattr(api, name), name\n            assert namespace[name] is getattr(api, name), name\n        assert namespace["Tool"] is namespace["tool"]\n        assert namespace["Tools"] is namespace["tools"]\n        assert namespace["Skills"] is namespace["skills"]\n        assert namespace["Plugins"] is namespace["plugins"]\n        try:\n            chulk.not_a_public_export\n        except AttributeError:\n            pass\n        else:\n            raise AssertionError("unknown package attributes must fail")\n        print("ok")\n        ')
    assert completed.stdout.strip() == 'ok'

def test_advanced_api_imports_without_root_import_ordering() -> None:
    completed = _run_fresh_import('\n        import chulk.api\n\n        assert chulk.api.Agent\n        assert chulk.api.RunStore\n        print("ok")\n        ')
    assert completed.stdout.strip() == 'ok'

def test_tool_package_keeps_public_refs_after_implementation_imports() -> None:
    completed = _run_fresh_import('\n        import chulk.telegram.bot\n        from chulk import tools\n        from chulk.tools.public import ToolRef\n\n        ref_names = (\n            "calculator",\n            "read_file",\n            "list_files",\n            "search_files",\n            "search_memory",\n            "list_memories",\n            "summarize_memories",\n        )\n        assert all(isinstance(getattr(tools, name), ToolRef) for name in ref_names)\n        print("ok")\n        ')
    assert completed.stdout.strip() == 'ok'
