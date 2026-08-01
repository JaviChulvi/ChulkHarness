"""Quarantined plugin artifacts and exact managed installations."""

from __future__ import annotations

from base64 import urlsafe_b64decode
import csv
from dataclasses import dataclass
from email.parser import BytesParser
from hashlib import sha256
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

from chulk.plugins.inspection import (
    PluginInspectionError,
    inspect_plugin_directory,
)
from chulk.plugins.manifest import load_plugin_manifest
from chulk.plugins.models import (
    PluginInspection,
    PluginLockEntry,
    PluginSourceKind,
)


_MAX_WHEEL_BYTES = 100_000_000
_MAX_WHEEL_MEMBERS = 10_000
_MAX_WHEEL_MEMBER_BYTES = 20_000_000
_MAX_WHEEL_EXPANDED_BYTES = 100_000_000
_MAX_COMPRESSION_RATIO = 200
_COPY_BUFFER_BYTES = 1024 * 1024
_COMPILED_SUFFIXES = (
    ".dll",
    ".dylib",
    ".pyd",
    ".so",
)


class PluginArtifactError(ValueError):
    """A plugin source cannot be quarantined or installed safely."""


@dataclass(frozen=True, slots=True)
class PreparedPluginPackage:
    """Exact static package identity inside host-owned quarantine."""

    inspection: PluginInspection
    source_kind: PluginSourceKind
    source_reference: str
    artifact_digest: str
    quarantine_root: Path


class PluginPackageStore:
    """Owner-private quarantine and immutable managed package store."""

    def __init__(self, runtime_dir: Path | str) -> None:
        self.root = (
            Path(runtime_dir).expanduser().resolve() / "plugins"
        )
        self.quarantine_dir = self.root / "quarantine"
        self.installed_dir = self.root / "installed"

    def prepare(self, source: Path | str) -> PreparedPluginPackage:
        """Copy or extract an operator-provided package into quarantine."""
        path = Path(source).expanduser()
        if path.is_dir():
            return self._prepare_directory(path)
        if path.is_file() and path.suffix.lower() == ".whl":
            return self._prepare_wheel(path)
        raise PluginArtifactError(
            "plugin source must be a local directory or exact .whl file"
        )

    def promote(
        self,
        prepared: PreparedPluginPackage,
    ) -> PluginInspection:
        """Copy a verified quarantine package to an immutable install path."""
        package = prepared.inspection.package
        suffix = package.digest.removeprefix("sha256:")[:20]
        parent = (
            self.installed_dir
            / package.manifest.name
            / f"{package.manifest.version}-{suffix}"
        )
        destination = parent / package.manifest.name
        if destination.exists():
            inspected = inspect_plugin_directory(destination)
            if inspected.package.digest != package.digest:
                raise PluginArtifactError(
                    "managed plugin install path contains another digest"
                )
            return inspected
        parent.mkdir(parents=True, exist_ok=True)
        _owner_private(parent, directory=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".install-", dir=parent)
        )
        copied = temporary / package.manifest.name
        try:
            shutil.copytree(package.root, copied, symlinks=True)
            inspected = inspect_plugin_directory(copied)
            if inspected.package.digest != package.digest:
                raise PluginArtifactError(
                    "plugin changed while copying from quarantine"
                )
            try:
                os.replace(copied, destination)
            except FileExistsError:
                inspected = inspect_plugin_directory(destination)
                if inspected.package.digest != package.digest:
                    raise PluginArtifactError(
                        "managed plugin install raced with another digest"
                    )
            _make_tree_owner_private(destination)
            return inspect_plugin_directory(destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def verify_artifact(self, entry: PluginLockEntry) -> None:
        """Verify retained supply-chain evidence for a managed lock."""
        if not entry.artifact_digest:
            return
        if entry.source_kind is PluginSourceKind.PREBUILT_WHEEL:
            artifact = (
                self.quarantine_dir
                / entry.artifact_digest.removeprefix("sha256:")
                / "artifact.whl"
            )
            if (
                artifact.is_symlink()
                or not artifact.is_file()
                or _file_digest(artifact) != entry.artifact_digest
            ):
                raise PluginArtifactError(
                    "retained plugin wheel does not match its locked digest"
                )

    def _prepare_directory(self, source: Path) -> PreparedPluginPackage:
        try:
            inspection = inspect_plugin_directory(source)
        except PluginInspectionError as exc:
            raise PluginArtifactError(str(exc)) from exc
        artifact_digest = inspection.package.digest
        target_parent = (
            self.quarantine_dir
            / artifact_digest.removeprefix("sha256:")
        )
        target = target_parent / inspection.package.manifest.name
        if not target.exists():
            target_parent.mkdir(parents=True, exist_ok=True)
            _owner_private(target_parent, directory=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=".directory-", dir=target_parent)
            )
            copied = temporary / inspection.package.manifest.name
            try:
                shutil.copytree(
                    inspection.package.root,
                    copied,
                    symlinks=True,
                )
                copied_inspection = inspect_plugin_directory(copied)
                if copied_inspection.package.digest != artifact_digest:
                    raise PluginArtifactError(
                        "local plugin changed while entering quarantine"
                    )
                try:
                    os.replace(copied, target)
                except FileExistsError:
                    pass
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        quarantined = inspect_plugin_directory(target)
        if quarantined.package.digest != artifact_digest:
            raise PluginArtifactError(
                "quarantined plugin digest does not match its source"
            )
        _make_tree_owner_private(target)
        return PreparedPluginPackage(
            inspection=quarantined,
            source_kind=PluginSourceKind.LOCAL_DIRECTORY,
            source_reference=str(inspection.package.root),
            artifact_digest=artifact_digest,
            quarantine_root=target_parent,
        )

    def _prepare_wheel(self, source: Path) -> PreparedPluginPackage:
        if source.is_symlink() or not source.is_file():
            raise PluginArtifactError(
                "plugin wheel must be a regular file"
            )
        source = source.resolve(strict=True)
        if source.stat().st_size > _MAX_WHEEL_BYTES:
            raise PluginArtifactError(
                f"plugin wheel exceeds {_MAX_WHEEL_BYTES} bytes"
            )
        artifact_digest = _file_digest(source)
        target_parent = (
            self.quarantine_dir
            / artifact_digest.removeprefix("sha256:")
        )
        existing_packages = tuple(
            path
            for path in target_parent.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ) if target_parent.is_dir() else ()
        if len(existing_packages) == 1:
            retained = target_parent / "artifact.whl"
            if (
                retained.is_symlink()
                or not retained.is_file()
                or _file_digest(retained) != artifact_digest
            ):
                raise PluginArtifactError(
                    "quarantined plugin wheel digest mismatch"
                )
            inspection = inspect_plugin_directory(existing_packages[0])
            _validate_wheel_identity(source, inspection)
            return PreparedPluginPackage(
                inspection=inspection,
                source_kind=PluginSourceKind.PREBUILT_WHEEL,
                source_reference=str(source),
                artifact_digest=artifact_digest,
                quarantine_root=target_parent,
            )
        target_parent.mkdir(parents=True, exist_ok=True)
        _owner_private(target_parent, directory=True)
        retained_wheel = target_parent / "artifact.whl"
        if not retained_wheel.exists():
            temporary_wheel = target_parent / ".artifact.whl.tmp"
            try:
                shutil.copyfile(source, temporary_wheel)
                if _file_digest(temporary_wheel) != artifact_digest:
                    raise PluginArtifactError(
                        "plugin wheel changed while entering quarantine"
                    )
                os.replace(temporary_wheel, retained_wheel)
                _owner_private(retained_wheel, directory=False)
            finally:
                temporary_wheel.unlink(missing_ok=True)
        elif _file_digest(retained_wheel) != artifact_digest:
            raise PluginArtifactError(
                "quarantined plugin wheel digest mismatch"
            )
        temporary = Path(
            tempfile.mkdtemp(prefix=".wheel-", dir=target_parent)
        )
        payload = temporary / "payload"
        payload.mkdir()
        try:
            with zipfile.ZipFile(source) as archive:
                _validate_wheel_archive(archive)
                _safe_extract_wheel(archive, payload)
            manifest = load_plugin_manifest(payload)
            package = temporary / manifest.name
            payload.rename(package)
            inspection = inspect_plugin_directory(package)
            _validate_wheel_identity(source, inspection)
            destination = target_parent / manifest.name
            try:
                os.replace(package, destination)
            except FileExistsError:
                pass
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            if isinstance(exc, PluginArtifactError):
                raise
            raise PluginArtifactError(
                f"invalid prebuilt plugin wheel: {exc}"
            ) from exc
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        inspection = inspect_plugin_directory(destination)
        _make_tree_owner_private(destination)
        return PreparedPluginPackage(
            inspection=inspection,
            source_kind=PluginSourceKind.PREBUILT_WHEEL,
            source_reference=str(source),
            artifact_digest=artifact_digest,
            quarantine_root=target_parent,
        )


def _validate_wheel_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if not infos or len(infos) > _MAX_WHEEL_MEMBERS:
        raise PluginArtifactError(
            f"plugin wheel must contain 1-{_MAX_WHEEL_MEMBERS} members"
        )
    expanded = 0
    names: set[str] = set()
    for info in infos:
        name = info.filename
        if (
            "\x00" in name
            or "\\" in name
            or name.startswith("/")
            or name in names
        ):
            raise PluginArtifactError(
                f"unsafe or duplicate wheel member: {name!r}"
            )
        names.add(name)
        pure = PurePosixPath(name)
        if not pure.parts or ".." in pure.parts:
            raise PluginArtifactError(
                f"wheel member escapes package root: {name!r}"
            )
        if any(
            not part
            or ":" in part
            or any(ord(character) < 32 for character in part)
            or part.rstrip(" .") != part
            for part in pure.parts
        ):
            raise PluginArtifactError(
                f"wheel member is not portable or safe: {name!r}"
            )
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise PluginArtifactError(
                f"plugin wheel cannot contain symlinks: {name}"
            )
        if info.file_size > _MAX_WHEEL_MEMBER_BYTES:
            raise PluginArtifactError(
                f"wheel member exceeds {_MAX_WHEEL_MEMBER_BYTES} bytes: {name}"
            )
        expanded += info.file_size
        if expanded > _MAX_WHEEL_EXPANDED_BYTES:
            raise PluginArtifactError(
                f"expanded wheel exceeds {_MAX_WHEEL_EXPANDED_BYTES} bytes"
            )
        if (
            info.file_size > 1_000_000
            and (
                info.compress_size == 0
                or info.file_size / info.compress_size
                > _MAX_COMPRESSION_RATIO
            )
        ):
            raise PluginArtifactError(
                f"wheel member compression ratio is unsafe: {name}"
            )
        if name.lower().endswith(_COMPILED_SUFFIXES):
            raise PluginArtifactError(
                "in-process plugin wheels must be pure Python"
            )
        if ".data/scripts/" in name.lower():
            raise PluginArtifactError(
                "plugin wheels cannot install executable scripts"
            )
    _verify_wheel_record(archive)


def _safe_extract_wheel(
    archive: zipfile.ZipFile,
    destination: Path,
) -> None:
    root = destination.resolve(strict=True)
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        target = root.joinpath(*pure.parts)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = target.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(root)
        except ValueError as exc:
            raise PluginArtifactError(
                f"wheel member escapes extraction root: {info.filename}"
            ) from exc
        remaining = info.file_size
        with archive.open(info) as source, target.open("xb") as output:
            while remaining:
                chunk = source.read(min(_COPY_BUFFER_BYTES, remaining))
                if not chunk:
                    raise PluginArtifactError(
                        f"truncated wheel member: {info.filename}"
                    )
                output.write(chunk)
                remaining -= len(chunk)
            if source.read(1):
                raise PluginArtifactError(
                    f"wheel member exceeds declared size: {info.filename}"
                )


def _verify_wheel_record(archive: zipfile.ZipFile) -> None:
    records = [
        name
        for name in archive.namelist()
        if name.endswith(".dist-info/RECORD")
    ]
    if len(records) != 1:
        raise PluginArtifactError(
            "plugin wheel must contain exactly one RECORD"
        )
    record_name = records[0]
    try:
        rows = tuple(
            csv.reader(
                io.StringIO(
                    archive.read(record_name).decode("utf-8")
                )
            )
        )
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise PluginArtifactError("plugin wheel RECORD is invalid") from exc
    recorded: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or row[0] in recorded:
            raise PluginArtifactError("plugin wheel RECORD is invalid")
        recorded[row[0]] = (row[1], row[2])
    for info in archive.infolist():
        if info.is_dir() or info.filename == record_name:
            continue
        value = recorded.get(info.filename)
        if value is None:
            raise PluginArtifactError(
                f"wheel member is missing from RECORD: {info.filename}"
            )
        hash_value, size_value = value
        if not hash_value.startswith("sha256="):
            raise PluginArtifactError(
                f"wheel RECORD requires sha256: {info.filename}"
            )
        try:
            expected = urlsafe_b64decode(
                hash_value.removeprefix("sha256=") + "=="
            )
            expected_size = int(size_value)
        except (ValueError, TypeError) as exc:
            raise PluginArtifactError(
                f"wheel RECORD entry is invalid: {info.filename}"
            ) from exc
        content = archive.read(info.filename)
        if len(content) != expected_size or sha256(content).digest() != expected:
            raise PluginArtifactError(
                f"wheel RECORD mismatch: {info.filename}"
            )


def _validate_wheel_identity(
    wheel_path: Path,
    inspection: PluginInspection,
) -> None:
    try:
        distribution, version, _build, tags = parse_wheel_filename(
            wheel_path.name
        )
    except ValueError as exc:
        raise PluginArtifactError("plugin wheel filename is invalid") from exc
    manifest = inspection.package.manifest
    if any(
        tag.abi != "none" or tag.platform != "any"
        for tag in tags
    ):
        raise PluginArtifactError(
            "in-process plugin wheels must use a pure-Python any-platform tag"
        )
    if (
        canonicalize_name(distribution)
        != canonicalize_name(manifest.name)
        or str(version) != manifest.version
    ):
        raise PluginArtifactError(
            "wheel name/version does not match the plugin manifest"
        )
    metadata_paths = tuple(
        path
        for path in inspection.package.root.rglob("METADATA")
        if path.parent.name.endswith(".dist-info")
    )
    if len(metadata_paths) != 1:
        raise PluginArtifactError(
            "plugin wheel must contain exactly one METADATA file"
        )
    metadata = BytesParser().parsebytes(metadata_paths[0].read_bytes())
    if (
        canonicalize_name(metadata.get("Name", ""))
        != canonicalize_name(manifest.name)
        or metadata.get("Version", "") != manifest.version
    ):
        raise PluginArtifactError(
            "wheel METADATA does not match the plugin manifest"
        )
    wheel_paths = tuple(
        path
        for path in inspection.package.root.rglob("WHEEL")
        if path.parent.name.endswith(".dist-info")
    )
    if len(wheel_paths) != 1:
        raise PluginArtifactError(
            "plugin wheel must contain exactly one WHEEL metadata file"
        )
    wheel_metadata = BytesParser().parsebytes(wheel_paths[0].read_bytes())
    if wheel_metadata.get("Root-Is-Purelib", "").lower() != "true":
        raise PluginArtifactError(
            "in-process plugin wheel must declare Root-Is-Purelib: true"
        )
    declared = {
        canonicalize_name(item.name)
        for item in manifest.python_dependencies
    }
    wheel_dependencies: set[str] = set()
    for raw in metadata.get_all("Requires-Dist", ()):
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as exc:
            raise PluginArtifactError(
                f"wheel dependency metadata is invalid: {raw}"
            ) from exc
        if requirement.marker is not None:
            marker_text = str(requirement.marker)
            if "extra" in marker_text or not requirement.marker.evaluate():
                continue
        wheel_dependencies.add(canonicalize_name(requirement.name))
    undeclared = sorted(wheel_dependencies - declared)
    if undeclared:
        raise PluginArtifactError(
            "wheel dependencies are not separately declared and reviewed: "
            + ", ".join(undeclared)
        )


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_BUFFER_BYTES):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _owner_private(path: Path, *, directory: bool) -> None:
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600)


def _make_tree_owner_private(root: Path) -> None:
    if os.name != "posix":
        return
    for directory, _names, filenames in os.walk(root):
        os.chmod(directory, 0o700)
        for name in filenames:
            os.chmod(Path(directory) / name, 0o600)


__all__ = [
    "PluginArtifactError",
    "PluginPackageStore",
    "PreparedPluginPackage",
]
