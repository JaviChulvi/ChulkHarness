"""Tests for host-only plugin CLI operations."""

from __future__ import annotations

import json

from chulk.main import main
from chulk.tests.test_plugin_manifests import write_plugin


def test_cli_inspects_without_importing_plugin_code(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    package = write_plugin(tmp_path / "packages")
    sentinel = tmp_path / "imported"
    (package / "sample_plugin" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )

    exit_code = main(["plugins", "inspect", str(package), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["inspection"]["manifest"]["name"] == "sample-plugin"
    assert not sentinel.exists()


def test_cli_registers_lists_and_audits_exact_local_plugin(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    package = write_plugin(tmp_path / "packages")

    rejected = main(
        [
            "plugins",
            "register",
            str(package),
            "--approved-by",
            "operator",
        ]
    )
    assert rejected == 2
    assert "acknowledgement" in capsys.readouterr().err

    registered = main(
        [
            "plugins",
            "register",
            str(package),
            "--approved-by",
            "operator",
            "--acknowledge-host-authority",
            "--grant-capability",
            "files:read",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert registered == 0
    assert payload["plugin"]["review"]["approved_by"] == "operator"

    listed = main(["plugins", "list", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert listed == 0
    assert [item["name"] for item in payload["plugins"]] == [
        "sample-plugin"
    ]

    audited = main(["plugins", "audit", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert audited == 0
    assert payload["ok"] is True
    assert payload["verified_plugins"] == ["sample-plugin"]


def test_cli_audit_fails_after_package_tamper(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    package = write_plugin(tmp_path / "packages")
    assert (
        main(
            [
                "plugins",
                "register",
                str(package),
                "--approved-by",
                "operator",
                "--acknowledge-host-authority",
            ]
        )
        == 0
    )
    capsys.readouterr()
    (package / "sample_plugin" / "tools.py").write_text(
        "def create_tool():\n    return 'tampered'\n",
        encoding="utf-8",
    )

    exit_code = main(["plugins", "audit", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["findings"][0]["code"] == "lock_mismatch"
