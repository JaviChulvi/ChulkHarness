from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _run_example(
    path: str,
    *,
    cwd: Path = ROOT,
    runtime_dir: Path | None = None,
    arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for name in (
        "OPENAI_API_KEY",
        "CHULK_LLM_PROVIDER",
        "CHULK_MODEL",
        "CHULK_EXAMPLE_MODE",
    ):
        env.pop(name, None)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing else os.pathsep.join((str(ROOT), existing))
    if runtime_dir is not None:
        env["CHULK_EXAMPLE_RUNTIME_DIR"] = str(runtime_dir)
    return subprocess.run(
        [sys.executable, str(ROOT / path), *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_quickstart_reaches_first_result_without_credentials(tmp_path: Path) -> None:
    completed = _run_example("examples/00_sdk_quickstart.py", runtime_dir=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "mode: scripted" in completed.stdout
    assert "Order A-100 is packed and ships tomorrow." in completed.stdout
    assert "runtime_dir:" in completed.stdout
    assert "trace_path:" in completed.stdout


def test_review_bot_is_deterministic_and_runs_outside_checkout_cwd(tmp_path: Path) -> None:
    completed = _run_example("examples/repo_review_bot/app.py", cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "Repository review" in completed.stdout
    assert "High — The setup guide tells users to commit a production API token." in completed.stdout
    assert "tool.call.started" in completed.stdout
    assert "tool.call.completed" in completed.stdout
    assert completed.stdout.rstrip().splitlines()[-1].startswith("trace_path:")


def test_review_bot_generates_a_normalized_trace_from_its_real_run(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    completed = _run_example(
        "examples/repo_review_bot/app.py",
        cwd=tmp_path,
        arguments=("--write-sample", str(sample)),
    )

    assert completed.returncode == 0, completed.stderr
    text = sample.read_text(encoding="utf-8")
    assert '"type":"turn_started"' in text
    assert '"type":"tool_call_completed"' in text
    assert '"type":"turn_finished"' in text
    assert str(ROOT) not in text
    assert str(tmp_path) not in text


def test_review_bot_uses_only_supported_public_import_boundaries() -> None:
    source = (ROOT / "examples" / "repo_review_bot" / "app.py").read_text(encoding="utf-8")

    assert "from chulk import" in source
    assert "from chulk.testing import ScriptedLLMClient" in source
    assert "from chulk.core" not in source
    assert "from chulk._sdk" not in source
