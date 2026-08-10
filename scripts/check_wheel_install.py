#!/usr/bin/env python3
"""Install a built ChulkHarness wheel in a clean venv and smoke-test it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap


def main() -> int:
    args = parse_args()
    wheel_path = choose_wheel(args.wheels)
    examples_dir = args.examples_dir.resolve() if args.examples_dir is not None else None

    with tempfile.TemporaryDirectory(prefix="chulk-wheel-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        venv_dir = temp_path / "venv"
        app_dir = temp_path / "app"
        app_dir.mkdir()

        run([sys.executable, "-m", "venv", str(venv_dir)])
        python = venv_python(venv_dir)
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", str(wheel_path)])

        smoke_script = temp_path / "smoke_public_api.py"
        smoke_script.write_text(public_api_smoke_source(), encoding="utf-8")
        run(
            [str(python), str(smoke_script)],
            cwd=app_dir,
            clean_pythonpath=True,
            extra_env={"CHULK_SMOKE_SOURCE_ROOT": str(Path.cwd().resolve())},
        )

        if examples_dir is not None:
            import_examples(python, examples_dir, app_dir)

    print("Clean wheel install smoke passed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path, help="Path to one or more built .whl files")
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Optional examples directory to import against the installed wheel",
    )
    return parser.parse_args()


def choose_wheel(wheels: list[Path]) -> Path:
    existing = [path.resolve() for path in wheels if path.exists()]
    if not existing:
        requested = ", ".join(str(path) for path in wheels)
        raise SystemExit(f"Wheel not found: {requested}")
    return sorted(existing, key=lambda path: path.stat().st_mtime)[-1]


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    clean_pythonpath: bool = False,
    extra_env: dict[str, str] | None = None,
) -> None:
    display = " ".join(command)
    print(f"+ {display}")
    env = os.environ.copy()
    if clean_pythonpath:
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra_env or {})
    subprocess.run(command, cwd=cwd, env=env, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def public_api_smoke_source() -> str:
    return textwrap.dedent(
        """
        import importlib.util
        from importlib.metadata import metadata, version
        import os
        from pathlib import Path

        import chulk
        from chulk import Agent, AgentConfig, Skills, Tool, Tools
        from chulk.evals import (
            EvalCase,
            EvalDataset,
            EvalReference,
            EvalReport,
            EvalRunner,
            EvalSuite,
            EvalTarget,
            EvalTurn,
            ExactAnswerGrader,
        )
        from chulk.presets import SoftwareEngineer
        from chulk.results import RunResult, RunStatus
        from chulk.skills import SkillRegistry, bundled_skills_dir


        @Tool
        def ping() -> str:
            \"\"\"Return pong.\"\"\"
            return "pong"


        assert callable(Agent)
        assert chulk.__version__ == version("chulkharness")
        assert AgentConfig is not None
        assert ping.name == "ping"
        assert Tools.calculator.name == "calculator"
        assert Skills.files.name == "files"

        package_root = Path(chulk.__file__).resolve().parent
        source_root = Path(os.environ["CHULK_SMOKE_SOURCE_ROOT"]).resolve()
        assert not package_root.is_relative_to(source_root)
        assert (package_root / "py.typed").is_file()
        eval_dashboard = package_root / "server" / "eval_dashboard"
        assert (eval_dashboard / "index.html").is_file()
        assert (eval_dashboard / "evals.css").is_file()
        assert (eval_dashboard / "evals.js").is_file()

        package_metadata = metadata("chulkharness")
        assert package_metadata["Name"] == "chulkharness"
        assert package_metadata["Requires-Python"] == ">=3.11"
        assert package_metadata["License-Expression"] == "MIT"
        project_urls = package_metadata.get_all("Project-URL") or []
        assert any(url.startswith("Repository, https://github.com/JaviChulvi/ChulkHarness") for url in project_urls)
        assert importlib.util.find_spec("chulk.tests") is None

        project_root = Path.cwd().resolve()
        config = AgentConfig.local(
            project_root=project_root,
            runtime_dir=".chulk",
            permission_profile="read-only",
        ).to_config()
        assert config.runtime_dir == project_root / ".chulk"
        assert config.store_path == project_root / ".chulk" / "store.sqlite"
        assert config.traces_dir == project_root / ".chulk" / "traces"
        assert config.skills_dir == project_root / ".chulk" / "skills"
        assert config.mcp_config_path == project_root / ".chulk" / "mcp.json"

        bundled_dir = bundled_skills_dir()
        registry = SkillRegistry(bundled_dir, skills_dirs=(bundled_dir,))
        registry.load_metadata()
        skill_names = {skill.name for skill in registry.list_skills()}
        assert {"files", "memory", "shell"} <= skill_names

        preset = SoftwareEngineer()
        assert preset.system_prompt
        assert preset.tools
        assert preset.skills is None

        class FakeAgent:
            def run_result(self, message, **kwargs):
                del message, kwargs
                return RunResult(
                    "wheel ready",
                    RunStatus.COMPLETED,
                    None,
                    "wheel-smoke",
                    None,
                )

            def close(self):
                pass

        eval_case = EvalCase(
            "wheel",
            (EvalTurn("run"),),
            EvalReference(answer="wheel ready"),
        )
        eval_suite = EvalSuite(
            "wheel-smoke",
            EvalDataset((eval_case,)),
            (EvalTarget("fake", lambda _context: FakeAgent()),),
            (ExactAnswerGrader(),),
            required_graders=("answer.exact",),
        )
        eval_report = EvalRunner().run(eval_suite)
        assert isinstance(eval_report, EvalReport)
        assert eval_report.passed

        print("metadata, typing, evals, dashboard assets, resources, and runtime defaults are available")
        """
    ).strip()


def import_examples(python: Path, examples_dir: Path, app_dir: Path) -> None:
    if not examples_dir.exists():
        raise SystemExit(f"Examples directory not found: {examples_dir}")

    copied_examples = app_dir / "examples"
    shutil.copytree(
        examples_dir,
        copied_examples,
        ignore=shutil.ignore_patterns("__pycache__", "runtime"),
    )

    import_script = app_dir / "import_examples.py"
    import_script.write_text(example_import_source(), encoding="utf-8")
    run([str(python), str(import_script)], cwd=app_dir, clean_pythonpath=True)


def example_import_source() -> str:
    return textwrap.dedent(
        """
        import importlib.util
        from pathlib import Path
        import sys


        examples_dir = Path.cwd() / "examples"
        sys.path.insert(0, str(examples_dir))

        for path in sorted(examples_dir.glob("*.py")):
            module_name = f"_chulk_example_{path.stem.replace('-', '_')}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            print(f"imported {path.name}")
        """
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
