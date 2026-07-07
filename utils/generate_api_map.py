#!/usr/bin/env python
"""
Generate a structured API index of public Python symbols in the repository.

Parses all .py source files using the ``ast`` module (no imports required) and
writes a concise Markdown reference to ``docs/API_MAP.md``.  Useful for quickly
locating classes, functions, and methods without grepping the full codebase.

Usage:
    python utils/generate_api_map.py [root_dir]

If root_dir is omitted, uses the parent of the directory containing this
script (i.e. the physiotwin4d project root).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


# Directories that are never first-party source code
SKIP_DIRS: set[str] = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
    ".venv",
}

OUTPUT_FILE = "docs/API_MAP.md"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def first_docstring_line(node: ast.AST) -> str:
    """Return the first non-empty line of a class or function docstring."""
    body = getattr(node, "body", [])
    if not body:
        return ""
    first = body[0]
    if not isinstance(first, ast.Expr):
        return ""
    val = first.value
    if not isinstance(val, ast.Constant) or not isinstance(val.value, str):
        return ""
    for line in val.value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def format_arg(arg: ast.arg, default: ast.expr | None) -> str:
    """Format one argument, appending ``=default`` when a default is present."""
    if default is None:
        return arg.arg
    try:
        default_str = ast.unparse(default)
    except Exception:
        default_str = "..."
    return f"{arg.arg}={default_str}"


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a readable signature string from an AST function node."""
    args = node.args

    # Positional args: defaults are right-aligned
    n_no_default = len(args.args) - len(args.defaults)
    padded: list[ast.expr | None] = [None] * n_no_default + list(args.defaults)

    parts: list[str] = [format_arg(a, d) for a, d in zip(args.args, padded)]

    # *args or bare * separator
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    # Keyword-only args
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(format_arg(arg, default))

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)})"


def is_public(name: str) -> bool:
    """Return True for public names; also allow ``__init__``."""
    return name == "__init__" or not name.startswith("_")


def module_all_names(tree: ast.Module) -> set[str] | None:
    """Return the names listed in ``__all__`` if defined at module level, else None."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    names: set[str] = set()
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.add(elt.value)
                    return names
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class MethodEntry:
    """A single public method within a class."""

    __slots__ = ("lineno", "signature", "summary")

    def __init__(self, signature: str, lineno: int, summary: str) -> None:
        self.signature = signature
        self.lineno = lineno
        self.summary = summary


class ClassEntry:
    """A public class with its public methods."""

    __slots__ = ("lineno", "methods", "name", "summary")

    def __init__(self, name: str, lineno: int, summary: str) -> None:
        self.name = name
        self.lineno = lineno
        self.summary = summary
        self.methods: list[MethodEntry] = []


class FunctionEntry:
    """A public module-level function."""

    __slots__ = ("lineno", "signature", "summary")

    def __init__(self, signature: str, lineno: int, summary: str) -> None:
        self.signature = signature
        self.lineno = lineno
        self.summary = summary


class ModuleEntry:
    """All public symbols extracted from one source file."""

    __slots__ = ("classes", "functions", "rel_path")

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.classes: list[ClassEntry] = []
        self.functions: list[FunctionEntry] = []


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_module(path: Path, root: Path) -> ModuleEntry | None:
    """Parse *path* and return a :class:`ModuleEntry`, or ``None`` on error / empty."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    rel_path = path.relative_to(root).as_posix()
    entry = ModuleEntry(rel_path)
    all_names = module_all_names(tree)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if all_names is not None and node.name not in all_names:
                continue
            if not is_public(node.name):
                continue
            cls_entry = ClassEntry(
                name=node.name,
                lineno=node.lineno,
                summary=first_docstring_line(node),
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not is_public(item.name):
                        continue
                    cls_entry.methods.append(
                        MethodEntry(
                            signature=format_signature(item),
                            lineno=item.lineno,
                            summary=first_docstring_line(item),
                        )
                    )
            entry.classes.append(cls_entry)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if all_names is not None and node.name not in all_names:
                continue
            if not is_public(node.name):
                continue
            entry.functions.append(
                FunctionEntry(
                    signature=format_signature(node),
                    lineno=node.lineno,
                    summary=first_docstring_line(node),
                )
            )

    if not entry.classes and not entry.functions:
        return None
    return entry


def find_python_files(root: Path) -> list[Path]:
    """Return sorted .py files under *root*, skipping non-source directories."""
    results: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS or p.startswith(".") for p in parts):
            continue
        results.append(path)
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(modules: list[ModuleEntry]) -> str:
    """Render *modules* to a Markdown string."""
    lines: list[str] = [
        "# API Map",
        "",
        "_Generated by `utils/generate_api_map.py`. Do not edit manually._",
        "_Re-run `py utils/generate_api_map.py` whenever public APIs change._",
        "",
    ]

    for mod in modules:
        lines.append(f"## {mod.rel_path}")
        lines.append("")

        # Interleave classes and functions in source order
        items: list[tuple[int, ClassEntry | FunctionEntry]] = []
        for cls in mod.classes:
            items.append((cls.lineno, cls))
        for fn in mod.functions:
            items.append((fn.lineno, fn))
        items.sort(key=lambda t: t[0])

        for _, item in items:
            if isinstance(item, ClassEntry):
                tail = f": {item.summary}" if item.summary else ""
                lines.append(f"- **class {item.name}** (line {item.lineno}){tail}")
                for method in item.methods:
                    mtail = f": {method.summary}" if method.summary else ""
                    lines.append(
                        f"  - `{method.signature}` (line {method.lineno}){mtail}"
                    )
            else:
                tail = f": {item.summary}" if item.summary else ""
                lines.append(f"- `{item.signature}` (line {item.lineno}){tail}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs/API_MAP.md from Python source files."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Project root directory (default: parent of utils/).",
    )
    args = parser.parse_args()

    if args.root_dir is not None:
        root = Path(args.root_dir).resolve()
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            return 1
    else:
        # Default: parent of the directory containing this script
        root = Path(__file__).resolve().parent.parent

    py_files = find_python_files(root)
    print(f"Scanning {len(py_files)} Python files under {root}")

    modules: list[ModuleEntry] = []
    for path in py_files:
        mod = parse_module(path, root)
        if mod is not None:
            modules.append(mod)

    print(f"Found public APIs in {len(modules)} modules")

    output_path = root / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(modules), encoding="utf-8")
    print(f"Written: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
