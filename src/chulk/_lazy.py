"""Small helpers for compatibility packages with lazy public exports."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from importlib import import_module


def resolve_export(
    name: str,
    *,
    public_names: Iterable[str],
    owner_modules: Iterable[str],
    namespace: MutableMapping[str, object],
) -> object:
    """Resolve one declared export from its focused implementation module."""
    if name not in public_names:
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
    for module_name in owner_modules:
        module = import_module(module_name)
        if name in vars(module):
            value = vars(module)[name]
            namespace[name] = value
            return value
    raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")


def public_dir(
    public_names: Iterable[str],
    namespace: MutableMapping[str, object],
) -> list[str]:
    """Return declared lazy exports alongside materialized module globals."""
    return sorted({*namespace, *public_names})
