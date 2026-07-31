"""Tests for exact local plugin registration and startup verification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import stat
import sys
import threading
import time

import pytest

import chulk.plugins.registry as registry_module
from chulk.plugins import (
    LocalPluginRegistry,
    PluginCategory,
    PluginLoadError,
    PluginLockError,
    PluginRegistrationError,
    PluginVerificationError,
)
from .test_plugin_manifests import plugin_manifest, write_plugin


def registry(tmp_path):
    return LocalPluginRegistry(
        tmp_path / "runtime",
        profile_id="default",
    )


@pytest.fixture(autouse=True)
def isolate_sample_plugin_modules():
    for name in tuple(sys.modules):
        if name == "sample_plugin" or name.startswith("sample_plugin."):
            sys.modules.pop(name, None)
    yield
    for name in tuple(sys.modules):
        if name == "sample_plugin" or name.startswith("sample_plugin."):
            sys.modules.pop(name, None)


def test_registration_requires_explicit_host_authority_acknowledgement(
    tmp_path,
):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)

    with pytest.raises(
        PluginRegistrationError,
        match="host-authority acknowledgement",
    ):
        plugins.register_local(
            package,
            approved_by="operator",
            acknowledge_host_authority=False,
        )

    assert plugins.list() == ()


def test_registration_writes_private_exact_credential_free_lock(tmp_path):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)

    entry = plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("network", "files:read"),
    )

    assert entry.name == "sample-plugin"
    assert entry.digest.startswith("sha256:")
    assert entry.review.granted_capabilities == (
        "files:read",
        "network",
    )
    assert plugins.verify_startup().verified_plugins == ("sample-plugin",)
    lock_text = plugins.lock.path.read_text(encoding="utf-8")
    assert "SAMPLE_TOKEN" in lock_text
    assert "secret-value" not in lock_text
    if os.name == "posix":
        assert stat.S_IMODE(plugins.lock.path.stat().st_mode) == 0o600


def test_registration_is_idempotent_only_for_the_same_reviewed_identity(
    tmp_path,
):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    first = plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("network",),
    )

    repeated = plugins.register_local(
        package,
        approved_by="another-operator",
        acknowledge_host_authority=True,
        granted_capabilities=("network",),
    )

    assert repeated == first
    with pytest.raises(PluginRegistrationError, match="already registered"):
        plugins.register_local(
            package,
            approved_by="operator",
            acknowledge_host_authority=True,
            granted_capabilities=("files:read",),
        )


def test_registration_rejects_undeclared_or_incompatible_authority(
    tmp_path,
):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)

    with pytest.raises(
        PluginRegistrationError,
        match="undeclared",
    ):
        plugins.register_local(
            package,
            approved_by="operator",
            acknowledge_host_authority=True,
            granted_capabilities=("shell",),
        )

    incompatible = write_plugin(
        tmp_path / "other",
        manifest=plugin_manifest().replace(
            'requires_python: ">=3.11"',
            'requires_python: ">=99"',
        ),
    )
    with pytest.raises(
        PluginRegistrationError,
        match="active Python version",
    ):
        plugins.register_local(
            incompatible,
            approved_by="operator",
            acknowledge_host_authority=True,
        )


def test_startup_verification_fails_closed_after_package_tamper(tmp_path):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )
    (package / "sample_plugin" / "tools.py").write_text(
        "def create_tool():\n    return 'changed'\n",
        encoding="utf-8",
    )

    report = plugins.audit()

    assert report.ok is False
    assert report.findings[0].code == "lock_mismatch"
    with pytest.raises(PluginVerificationError, match="reviewed lock"):
        plugins.verify_startup()


def test_startup_verification_rejects_added_python_bytecode(tmp_path):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    bytecode = (
        package
        / "sample_plugin"
        / "__pycache__"
        / "tools.cpython-312.pyc"
    )
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"unreviewed executable bytes")

    report = plugins.audit()

    assert report.ok is False
    assert report.findings[0].code == "package_invalid"
    with pytest.raises(PluginVerificationError, match="bytecode"):
        plugins.verify_startup()


def test_lock_rejects_unknown_fields_profile_mismatch_and_symlinks(
    tmp_path,
):
    plugins = registry(tmp_path)
    plugins.lock.path.parent.mkdir(parents=True)
    plugins.lock.path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "other",
                "plugins": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PluginLockError, match="profile"):
        plugins.list()

    plugins.lock.path.unlink()
    target = tmp_path / "target.lock"
    target.write_text("{}", encoding="utf-8")
    try:
        plugins.lock.path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(PluginLockError, match="regular file"):
        plugins.list()


def test_lock_rejects_coerced_values_and_noncanonical_source_paths(
    tmp_path,
):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    payload = json.loads(plugins.lock.path.read_text(encoding="utf-8"))
    payload["plugins"]["sample-plugin"]["version"] = 123
    plugins.lock.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PluginLockError, match="string fields"):
        plugins.list()

    payload["plugins"]["sample-plugin"]["version"] = "1.2.3"
    payload["plugins"]["sample-plugin"]["source_path"] = str(
        package / ".." / package.name
    )
    plugins.lock.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PluginLockError, match="canonical"):
        plugins.list()


def test_loader_requires_reviewed_and_runtime_capabilities(tmp_path):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )

    with pytest.raises(PluginLoadError, match="does not grant"):
        plugins.load_entry_point(
            "sample-plugin",
            PluginCategory.TOOL,
            "sample",
        )

    loaded = plugins.load_entry_point(
        "sample-plugin",
        PluginCategory.TOOL,
        "sample",
        available_capabilities=("files:read",),
    )

    assert loaded.plugin_name == "sample-plugin"
    assert callable(loaded.value)
    assert loaded.value().__class__ is object
    assert not (package / "sample_plugin" / "__pycache__").exists()


def test_loader_normalizes_unsupported_category_errors(tmp_path):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
    )

    with pytest.raises(PluginLoadError, match="unsupported plugin category"):
        plugins.load_entry_point(
            "sample-plugin",
            "not-a-category",
            "sample",
        )


def test_loader_refuses_module_namespace_owned_outside_package(tmp_path):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sample_plugin.py").write_text("value = 1\n", encoding="utf-8")
    sys.path.insert(0, str(outside))
    try:
        __import__("sample_plugin")
        with pytest.raises(PluginLoadError, match="already owned"):
            plugins.load_entry_point(
                "sample-plugin",
                "tool",
                "sample",
                available_capabilities=("files:read",),
            )
    finally:
        sys.path.remove(str(outside))
        sys.modules.pop("sample_plugin", None)


def test_loader_wraps_plugin_import_failures(tmp_path):
    package = write_plugin(tmp_path)
    (package / "sample_plugin" / "__init__.py").write_text(
        "raise RuntimeError('plugin exploded')\n",
        encoding="utf-8",
    )
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )

    with pytest.raises(PluginLoadError, match="plugin exploded"):
        plugins.load_entry_point(
            "sample-plugin",
            "tool",
            "sample",
            available_capabilities=("files:read",),
        )

    assert "sample_plugin" not in sys.modules


def test_loader_discards_modules_when_entry_point_is_not_callable(
    tmp_path,
):
    package = write_plugin(
        tmp_path,
        manifest=plugin_manifest().replace(
            "sample_plugin.tools:create_tool",
            "sample_plugin.tools:value",
        ),
    )
    (package / "sample_plugin" / "tools.py").write_text(
        "value = 42\n",
        encoding="utf-8",
    )
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )

    with pytest.raises(PluginLoadError, match="callable factories"):
        plugins.load_entry_point(
            "sample-plugin",
            "tool",
            "sample",
            available_capabilities=("files:read",),
        )

    assert "sample_plugin" not in sys.modules
    assert "sample_plugin.tools" not in sys.modules


def test_loader_serializes_process_global_import_state(
    tmp_path,
    monkeypatch,
):
    package = write_plugin(tmp_path)
    plugins = registry(tmp_path)
    plugins.register_local(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )
    original_import = registry_module.importlib.import_module
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    def controlled_import(module_name):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        return original_import(module_name)

    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        controlled_import,
    )

    def load(*, started=None):
        if started is not None:
            started.set()
        return plugins.load_entry_point(
            "sample-plugin",
            "tool",
            "sample",
            available_capabilities=("files:read",),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(load)
        assert first_entered.wait(timeout=2)
        second = pool.submit(load, started=second_started)
        assert second_started.wait(timeout=2)
        time.sleep(0.05)
        with calls_lock:
            assert calls == 1
        release_first.set()
        assert callable(first.result(timeout=2).value)
        assert callable(second.result(timeout=2).value)


def test_registration_rejects_missing_plugin_dependencies(tmp_path):
    manifest = plugin_manifest().replace(
        "dependencies: {}",
        'dependencies:\n  missing-plugin: ">=1"',
    )
    package = write_plugin(tmp_path, manifest=manifest)
    plugins = registry(tmp_path)

    with pytest.raises(
        PluginRegistrationError,
        match="requires registered plugin",
    ):
        plugins.register_local(
            package,
            approved_by="operator",
            acknowledge_host_authority=True,
        )


def test_audit_detects_dependency_cycles(tmp_path):
    first = write_plugin(
        tmp_path / "first",
        manifest=plugin_manifest(name="first-plugin").replace(
            "dependencies: {}",
            'dependencies:\n  second-plugin: ">=1"',
        ),
    )
    first = first.rename(first.parent / "first-plugin")
    second = write_plugin(
        tmp_path / "second",
        manifest=plugin_manifest(name="second-plugin").replace(
            "dependencies: {}",
            'dependencies:\n  first-plugin: ">=1"',
        ),
    )
    second = second.rename(second.parent / "second-plugin")
    plugins = registry(tmp_path)

    first_inspection = plugins.inspect(first)
    second_inspection = plugins.inspect(second)
    now = "2026-01-01T00:00:00+00:00"
    from chulk.plugins import (
        PluginLockEntry,
        PluginRegistrationStatus,
        PluginReview,
        PluginSourceKind,
    )

    entries = {}
    for inspected in (first_inspection, second_inspection):
        package = inspected.package
        entries[package.manifest.name] = PluginLockEntry(
            name=package.manifest.name,
            version=package.manifest.version,
            digest=package.digest,
            source_kind=PluginSourceKind.LOCAL_DIRECTORY,
            source_path=package.root,
            status=PluginRegistrationStatus.ENABLED,
            manifest=package.manifest,
            review=PluginReview(
                approved_by="operator",
                approved_at=now,
                acknowledged_host_authority=True,
            ),
            installed_at=now,
        )
    plugins.lock.write(entries)

    report = plugins.audit()

    assert report.ok is False
    assert {
        finding.code for finding in report.findings
    } == {"dependency_cycle"}
