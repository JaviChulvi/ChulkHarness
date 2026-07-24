"""Owner-only filesystem primitives for sensitive runtime text."""

from __future__ import annotations

import os
from pathlib import Path
import stat


PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


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
    path: Path | str,
    content: str,
    *,
    append: bool = False,
    overwrite: bool = True,
) -> Path:
    """Write owner-only UTF-8 text while rejecting symlink and special targets."""
    destination = Path(path)
    prepare_private_directory(destination.parent)
    exists = _validate_private_file_target(destination)
    if exists and not overwrite and not append:
        raise FileExistsError(f"Output already exists: {destination}")

    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if append else os.O_TRUNC
    if not overwrite and not append:
        flags |= os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, PRIVATE_FILE_MODE)
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"Sensitive runtime target is not a regular file: {destination}")
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(
            descriptor,
            "a" if append else "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            descriptor = -1
            stream.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return destination


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


__all__ = [
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "prepare_private_directory",
    "write_private_text",
]
