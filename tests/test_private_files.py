"""Adversarial coverage for private filesystem publication."""

from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

import chulk.storage.private_files as private_files


pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptors")


def _publish(path: Path, content: str = "safe") -> Path:
    return private_files.write_private_text(
        path, content, overwrite=False, private_parent=False, atomic=True
    )


def test_atomic_write_has_no_replaceable_temporary_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync

    def attack_temporary_name(descriptor: int) -> None:
        assert not list(tmp_path.glob(".fixture.json.*.tmp"))
        real_fsync(descriptor)

    monkeypatch.setattr(private_files.os, "fsync", attack_temporary_name)
    destination = _publish(tmp_path / "fixture.json")
    assert destination.read_text(encoding="utf-8") == "safe"


def test_atomic_write_enforces_private_mode_despite_umask(tmp_path: Path) -> None:
    previous_umask = os.umask(0o777)
    try:
        destination = _publish(tmp_path / "fixture.json")
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(destination.stat().st_mode) == private_files.PRIVATE_FILE_MODE


def test_atomic_write_rejects_embedded_nul_basename(tmp_path: Path) -> None:
    destination = tmp_path / "prefix\0ignored"
    with pytest.raises(private_files.PrivateFileError, match="NUL"):
        _publish(destination)
    assert not (tmp_path / "prefix").exists()


def test_atomic_write_traverses_ancestors_by_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "safe"
    nested = parent / "nested"
    nested.mkdir(parents=True)
    moved = tmp_path / "moved"
    attacker = tmp_path / "attacker"
    (attacker / "nested").mkdir(parents=True)
    real_open = os.open
    swapped = False

    def swap_after_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "safe" and not swapped:
            swapped = True
            parent.rename(moved)
            parent.symlink_to(attacker, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(private_files.os, "open", swap_after_open)
    monkeypatch.setattr(private_files, "_require_atomic_primitives", lambda: None)
    _publish(nested / "fixture.json")
    assert (moved / "nested" / "fixture.json").read_text(encoding="utf-8") == "safe"
    assert not (attacker / "nested" / "fixture.json").exists()


@pytest.mark.parametrize("target", ["regular", "fifo", "symlink"])
def test_atomic_force_is_rejected_without_displacing_target(
    tmp_path: Path, target: str,
) -> None:
    destination = tmp_path / "fixture.json"
    preserved = tmp_path / "preserved"
    preserved.write_text("old", encoding="utf-8")
    if target == "regular":
        destination.write_text("old", encoding="utf-8")
    elif target == "fifo":
        os.mkfifo(destination)
    else:
        destination.symlink_to(preserved)
    identity = destination.lstat()

    with pytest.raises(private_files.PrivateFileError, match="does not replace"):
        private_files.write_private_text(destination, "new", atomic=True, overwrite=True)

    assert destination.lstat().st_ino == identity.st_ino
    assert preserved.read_text(encoding="utf-8") == "old"
    if target == "regular":
        assert destination.read_text(encoding="utf-8") == "old"


def test_directory_fsync_failure_does_not_report_committed_publish_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(private_files.os, "fsync", fail_directory_fsync)
    destination = _publish(tmp_path / "fixture.json")
    assert destination.read_text(encoding="utf-8") == "safe"


@pytest.mark.parametrize(
    "failure", [TypeError("bad dir_fd"), AttributeError("no linkat"), OSError("unsupported")]
)
def test_atomic_capability_failures_are_private_file_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    def fail() -> None:
        raise failure

    monkeypatch.setattr(private_files, "_require_atomic_primitives", fail)
    with pytest.raises(private_files.PrivateFileError, match="unsupported"):
        _publish(tmp_path / "fixture.json")
    assert not (tmp_path / "fixture.json").exists()
