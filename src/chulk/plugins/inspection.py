"""Static local plugin package inspection and compatibility checks."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as distribution_version
import os
from pathlib import Path
import stat
import sys

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from chulk._version import __version__
from chulk.plugins.manifest import (
    PluginManifestError,
    load_plugin_manifest,
    resolve_plugin_resource,
)
from chulk.plugins.models import (
    PLUGIN_MANIFEST_FILENAME,
    PluginInspection,
    PluginPackage,
)


_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "venv",
    }
)
_IGNORED_FILES = frozenset({".DS_Store"})
_BYTECODE_SUFFIXES = (".pyc", ".pyo")
_MAX_PACKAGE_FILES = 10_000
_MAX_PACKAGE_BYTES = 100_000_000
_MAX_FILE_BYTES = 20_000_000


class PluginInspectionError(ValueError):
    """Raised when static package inspection fails closed."""


def inspect_plugin_directory(path: Path | str) -> PluginInspection:
    """Inspect one local package without importing or executing its code."""
    root = Path(path).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise PluginInspectionError(
            f"plugin package is not a regular directory: {root}"
        )
    root = root.resolve(strict=True)
    manifest_path = root / PLUGIN_MANIFEST_FILENAME
    try:
        manifest = load_plugin_manifest(manifest_path)
    except PluginManifestError as exc:
        raise PluginInspectionError(str(exc)) from exc
    if root.name.lower() != manifest.name:
        raise PluginInspectionError(
            "plugin manifest name must match its package directory"
        )
    for migration in manifest.migrations:
        try:
            resolve_plugin_resource(root, migration)
        except PluginManifestError as exc:
            raise PluginInspectionError(str(exc)) from exc
    for instruction in manifest.instructions:
        try:
            resolve_plugin_resource(root, instruction)
        except PluginManifestError as exc:
            raise PluginInspectionError(str(exc)) from exc
    _validate_entry_point_modules(root, manifest.entry_points)
    files, total_bytes = _package_files(root)
    digest = _digest_files(root, files)
    missing: list[str] = []
    incompatible: list[str] = []
    for dependency in manifest.python_dependencies:
        try:
            installed = distribution_version(dependency.name)
        except PackageNotFoundError:
            if not dependency.optional:
                missing.append(dependency.name)
            continue
        if Version(installed) not in SpecifierSet(dependency.version_spec):
            if not dependency.optional:
                incompatible.append(
                    f"{dependency.name} {installed} "
                    f"not in {dependency.version_spec}"
                )
    python_version = Version(
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    return PluginInspection(
        package=PluginPackage(
            root=root,
            manifest_path=manifest_path,
            manifest=manifest,
            digest=digest,
            file_count=len(files),
            total_bytes=total_bytes,
        ),
        chulk_compatible=Version(__version__)
        in SpecifierSet(manifest.requires_chulk),
        python_compatible=python_version
        in SpecifierSet(manifest.requires_python),
        missing_python_dependencies=tuple(sorted(missing)),
        incompatible_python_dependencies=tuple(sorted(incompatible)),
    )


def _package_files(root: Path) -> tuple[tuple[Path, ...], int]:
    files: list[Path] = []
    total_bytes = 0
    for directory, directory_names, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink():
                raise PluginInspectionError(
                    f"plugin package cannot contain symlinks: "
                    f"{child.relative_to(root)}"
                )
            if name == "__pycache__":
                raise PluginInspectionError(
                    "plugin package cannot contain Python bytecode caches: "
                    f"{child.relative_to(root)}"
                )
            if name in _IGNORED_DIRECTORIES:
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(filenames):
            if name in _IGNORED_FILES:
                continue
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise PluginInspectionError(
                    f"plugin package cannot contain symlinks: "
                    f"{path.relative_to(root)}"
                )
            if name.endswith(_BYTECODE_SUFFIXES):
                raise PluginInspectionError(
                    "plugin package cannot contain Python bytecode: "
                    f"{path.relative_to(root)}"
                )
            if not stat.S_ISREG(mode):
                raise PluginInspectionError(
                    f"plugin package contains a non-regular file: "
                    f"{path.relative_to(root)}"
                )
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise PluginInspectionError(
                    f"plugin package file escapes its root: {path}"
                ) from exc
            size = resolved.stat().st_size
            if size > _MAX_FILE_BYTES:
                raise PluginInspectionError(
                    f"plugin package file exceeds {_MAX_FILE_BYTES} bytes: "
                    f"{path.relative_to(root)}"
                )
            total_bytes += size
            if total_bytes > _MAX_PACKAGE_BYTES:
                raise PluginInspectionError(
                    f"plugin package exceeds {_MAX_PACKAGE_BYTES} bytes"
                )
            files.append(path)
            if len(files) > _MAX_PACKAGE_FILES:
                raise PluginInspectionError(
                    f"plugin package exceeds {_MAX_PACKAGE_FILES} files"
                )
    return tuple(sorted(files)), total_bytes


def _digest_files(root: Path, files: tuple[Path, ...]) -> str:
    digest = sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _validate_entry_point_modules(root: Path, entries: tuple[object, ...]) -> None:
    for entry in entries:
        target = str(getattr(entry, "target"))
        module_name = target.partition(":")[0]
        relative = Path(*module_name.split("."))
        module_file = root / relative.with_suffix(".py")
        package_file = root / relative / "__init__.py"
        if not module_file.is_file() and not package_file.is_file():
            raise PluginInspectionError(
                f"entry point module is not present in the plugin package: "
                f"{module_name}"
            )
        candidate = module_file if module_file.is_file() else package_file
        if candidate.is_symlink():
            raise PluginInspectionError(
                f"entry point module cannot be a symlink: {module_name}"
            )


__all__ = [
    "PluginInspectionError",
    "inspect_plugin_directory",
]
