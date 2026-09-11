from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

PACKAGE_NAME = "landuse_sentence_relevance"
_KNOWN_LAYERS = frozenset(
    {
        "analysis",
        "bootstrap",
        "config",
        "domain",
        "models",
        "observability",
        "sources",
        "storage",
        "web",
        "workflow",
    }
)
_ALL_LAYERS = _KNOWN_LAYERS | {"root"}
_ALLOWED_DEPENDENCIES: Mapping[str, frozenset[str]] = {
    "analysis": frozenset({"analysis", "domain"}),
    "bootstrap": _ALL_LAYERS,
    "config": frozenset({"config"}),
    "domain": frozenset({"domain"}),
    "models": frozenset({"models"}),
    "observability": frozenset({"observability"}),
    "sources": frozenset({"sources", "domain", "observability"}),
    "storage": frozenset({"storage", "domain", "config", "observability"}),
    "web": frozenset({"web", "domain", "workflow", "observability", "bootstrap", "config"}),
    "workflow": frozenset({"workflow", "domain", "storage"}),
}


def architecture_violations(project_root: Path = Path(".")) -> tuple[str, ...]:
    """Return deterministic dependency-boundary and cycle violations."""

    package_root = project_root / "src" / PACKAGE_NAME
    module_paths = _module_paths(package_root)
    if not module_paths:
        return (f"missing package directory: {package_root}",)

    known_modules = frozenset(module_paths)
    graph: dict[str, tuple[str, ...]] = {}
    violations: set[str] = set()
    for module, path in sorted(module_paths.items()):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            violations.add(f"{module} has invalid Python syntax: {error.msg}")
            graph[module] = ()
            continue

        targets = _internal_imports(tree, module, path.name == "__init__.py", known_modules)
        graph[module] = tuple(sorted(target for target in targets if target in known_modules))
        source_layer = _layer(module)
        allowed = _ALLOWED_DEPENDENCIES.get(source_layer, frozenset())
        if source_layer not in _ALLOWED_DEPENDENCIES and source_layer != "root":
            violations.add(f"{_display(module)} belongs to an unknown layer {source_layer}")
        for target in targets:
            target_layer = _layer(target)
            if target_layer != source_layer and target_layer not in allowed:
                violations.add(f"{_display(module)} imports forbidden layer {target_layer}")

    violations.update(_cycle_violations(graph))
    return tuple(sorted(violations))


def _module_paths(package_root: Path) -> dict[str, Path]:
    if not package_root.is_dir():
        return {}
    return {
        _module_name(path, package_root): path
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _module_name(path: Path, package_root: Path) -> str:
    relative = path.relative_to(package_root)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE_NAME, *parts))


def _internal_imports(
    tree: ast.AST,
    source_module: str,
    source_is_package: bool,
    known_modules: frozenset[str],
) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names if _is_internal(alias.name))
        elif isinstance(node, ast.ImportFrom):
            targets.update(_from_import_targets(node, source_module, source_is_package, known_modules))
    return targets


def _from_import_targets(
    node: ast.ImportFrom,
    source_module: str,
    source_is_package: bool,
    known_modules: frozenset[str],
) -> set[str]:
    if node.level:
        base = _relative_base(source_module, source_is_package, node.level)
        candidate = ".".join((*base, *(node.module.split(".") if node.module else ())))
    else:
        candidate = node.module or ""
    if not _is_internal(candidate):
        return set()

    targets = {_nearest_known_module(candidate, known_modules)}
    for alias in node.names:
        alias_target = f"{candidate}.{alias.name}"
        if alias_target in known_modules:
            targets.add(alias_target)
    return targets


def _relative_base(source_module: str, source_is_package: bool, level: int) -> tuple[str, ...]:
    source_parts = source_module.split(".")
    container = source_parts if source_is_package else source_parts[:-1]
    keep = len(container) - level + 1
    return tuple(container[: max(keep, 0)])


def _nearest_known_module(candidate: str, known_modules: frozenset[str]) -> str:
    parts = candidate.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        if prefix in known_modules:
            return prefix
    return candidate


def _is_internal(module: str) -> bool:
    return module == PACKAGE_NAME or module.startswith(f"{PACKAGE_NAME}.")


def _layer(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "root"


def _display(module: str) -> str:
    return module.removeprefix(f"{PACKAGE_NAME}.")


def _cycle_violations(graph: Mapping[str, Iterable[str]]) -> set[str]:
    active: dict[str, int] = {}
    path: list[str] = []
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(module: str) -> None:
        if module in visited:
            return
        active[module] = len(path)
        path.append(module)
        for target in graph.get(module, ()):
            if target in active:
                cycle = _canonical_cycle(path[active[target] :])
                cycles.add("circular import: " + " -> ".join((*cycle, cycle[0])))
            else:
                visit(target)
        path.pop()
        del active[module]
        visited.add(module)

    for module in sorted(graph):
        visit(module)
    return cycles


def _canonical_cycle(nodes: list[str]) -> tuple[str, ...]:
    rotations = [tuple(nodes[index:] + nodes[:index]) for index in range(len(nodes))]
    return min(rotations)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check package dependency boundaries and circular imports.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root to inspect")
    args = parser.parse_args()
    violations = architecture_violations(args.root)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print("Architecture check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
