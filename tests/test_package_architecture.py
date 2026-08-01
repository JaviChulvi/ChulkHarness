"""Repository-wide dependency guards for production modules."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src" / "chulk"


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE)
    if relative.name == "__init__.py":
        suffix = relative.parent.parts
    else:
        suffix = relative.with_suffix("").parts
    return ".".join(("chulk", *suffix))


def _nearest_module(name: str, modules: set[str]) -> str | None:
    candidate = name
    while candidate.startswith("chulk"):
        if candidate in modules:
            return candidate
        if "." not in candidate:
            return None
        candidate = candidate.rsplit(".", 1)[0]
    return None


def _dependency_graph() -> dict[str, set[str]]:
    paths = tuple(SOURCE.rglob("*.py"))
    module_paths = {_module_name(path): path for path in paths}
    modules = set(module_paths)
    graph = {module: set() for module in modules}

    for module, path in module_paths.items():
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = resolve_name(
                        "." * node.level + (node.module or ""),
                        package,
                    )
                else:
                    base = node.module or ""
                candidates.extend(
                    f"{base}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
                candidates.append(base)

            for candidate in candidates:
                dependency = _nearest_module(candidate, modules)
                if dependency is not None and dependency != module:
                    graph[module].add(dependency)
    return graph


def _strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    stacked: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        stacked.add(module)

        for dependency in sorted(graph[module]):
            if dependency not in indices:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in stacked:
                lowlinks[module] = min(lowlinks[module], indices[dependency])

        if lowlinks[module] != indices[module]:
            return
        component: list[str] = []
        while True:
            dependency = stack.pop()
            stacked.remove(dependency)
            component.append(dependency)
            if dependency == module:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for module in sorted(graph):
        if module not in indices:
            visit(module)
    return sorted(components)


def test_production_module_graph_is_acyclic() -> None:
    """Keep local and type-only imports from recreating package cycles."""
    cycles = _strongly_connected_components(_dependency_graph())

    assert cycles == []
