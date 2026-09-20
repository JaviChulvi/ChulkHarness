"""Owner-only filesystem primitives for sensitive runtime text."""

from __future__ import annotations

import ctypes
from contextlib import suppress
import errno
import os
from pathlib import Path
import stat
import sys


PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


PrivateFileError = OSError


def prepare_private_directory(path: Path | str) -> Path:
    """Create or repair a private directory without following a target symlink."""
    directory = Path(path)
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError:
        directory.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
        mode = directory.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"Sensitive runtime directory is not a regular directory: {directory}")
    if os.name == "posix":
        os.chmod(directory, PRIVATE_DIRECTORY_MODE, follow_symlinks=False)
    return directory


def write_private_text(
    path: Path | str, content: str, *, append: bool = False,
    overwrite: bool = True, private_parent: bool = True, atomic: bool = False,
) -> Path:
    """Write owner-only UTF-8 text while rejecting symlink and special targets."""
    destination = Path(path)
    if atomic:
        if append:
            raise ValueError("Atomic private text publication cannot append")
        _write_private_text_atomic(destination, content, overwrite, private_parent)
        return destination
    (prepare_private_directory(destination.parent) if private_parent
     else destination.parent.mkdir(parents=True, exist_ok=True))
    exists = _validate_private_file_target(destination)
    if exists and not overwrite and not append:
        raise FileExistsError(f"Output already exists: {destination}")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    if not overwrite and not append:
        flags |= os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, PRIVATE_FILE_MODE)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Sensitive runtime target is not a regular file: {destination}")
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "a" if append else "w", encoding="utf-8",
                       newline="") as stream:
            descriptor = -1
            stream.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return destination


def _write_private_text_atomic(
    destination: Path, content: str, overwrite: bool, private_parent: bool,
) -> None:
    if overwrite:
        raise PrivateFileError("Atomic private text publication does not replace existing files")
    absolute = Path(os.path.abspath(destination.expanduser()))
    parent_fd = descriptor = -1
    try:
        _require_atomic_primitives()
        parent_fd = _open_verified_parent(absolute.parent, private_parent)
        descriptor = os.open(".", os.O_WRONLY | os.O_TMPFILE, PRIVATE_FILE_MODE,
                             dir_fd=parent_fd)
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        payload = content.encode("utf-8")
        while payload:
            payload = payload[os.write(descriptor, payload):]
        os.fsync(descriptor)
        _link_open_file(descriptor, parent_fd, absolute.name)
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    except FileExistsError as exc:
        raise FileExistsError(f"Output already exists: {destination}") from exc
    except (AttributeError, OSError, TypeError) as exc:
        raise PrivateFileError(f"Atomic private text publication is unsupported: {exc}") from exc
    finally:
        for opened in (descriptor, parent_fd):
            with suppress(OSError):
                os.close(opened)


def _open_verified_parent(path: Path, private: bool) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    for component in path.parts[1:]:
        try:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, PRIVATE_DIRECTORY_MODE, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        os.close(descriptor)
        descriptor = child
    if private:
        os.fchmod(descriptor, PRIVATE_DIRECTORY_MODE)
    return descriptor


def _require_atomic_primitives() -> None:
    if (sys.platform != "linux" or any(not hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_TMPFILE"))
            or os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd):
        raise PrivateFileError("Atomic private text publication requires Linux openat, mkdirat, and O_TMPFILE")


def _link_open_file(source_fd: int, parent_fd: int, name: str) -> None:
    if "\0" in name:
        raise PrivateFileError("Atomic private text destination contains a NUL byte")
    linkat = ctypes.CDLL(None, use_errno=True).linkat
    linkat.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    if linkat(source_fd, b"", parent_fd, os.fsencode(name), 0x1000) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), name)
        raise OSError(error, os.strerror(error), name)


def _validate_private_file_target(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"Sensitive runtime target is not a regular file: {path}")
    if os.name == "posix":
        os.chmod(path, PRIVATE_FILE_MODE, follow_symlinks=False)
    return True


__all__ = ["PRIVATE_DIRECTORY_MODE", "PRIVATE_FILE_MODE", "prepare_private_directory", "write_private_text"]
