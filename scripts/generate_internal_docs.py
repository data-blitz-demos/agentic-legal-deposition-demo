from __future__ import annotations

"""Generate internal artifact + function reference documentation."""

import argparse
import ast
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "internal" / "code_function_reference.md"

PYTHON_TARGETS = [
    ROOT / "backend" / "app",
    ROOT / "mcp_servers",
    ROOT / "scripts",
]
JS_TARGETS = [
    ROOT / "frontend" / "app.js",
]


@dataclass(frozen=True)
class FunctionDoc:
    """One function documentation row in the generated markdown output."""

    name: str
    signature: str
    description: str
    line: int


@dataclass(frozen=True)
class ArtifactDoc:
    """One artifact summary plus extracted function documentation rows."""

    path: Path
    summary: str
    functions: list[FunctionDoc]


def _normalize_text(value: str) -> str:
    """Collapse whitespace/newlines to one readable sentence-style string."""

    return re.sub(r"\s+", " ", value.strip())


def _safe_sentence_from_name(name: str) -> str:
    """Return fallback description text from a function name."""

    cleaned = name.replace("_", " ")
    cleaned = re.sub(r"(?<!^)([A-Z])", r" \1", cleaned).strip().lower()
    cleaned = cleaned.replace(".", " in ")
    return f"Internal helper that handles {cleaned}."


def _iter_python_files() -> list[Path]:
    """Return sorted Python files from configured targets."""

    files: list[Path] = []
    for target in PYTHON_TARGETS:
        if target.is_file():
            if target.suffix == ".py":
                files.append(target)
            continue
        if target.exists():
            files.extend(sorted(target.rglob("*.py")))
    return sorted({path.resolve() for path in files})


def _module_summary(tree: ast.Module) -> str:
    """Extract module-level summary from module docstring when available."""

    raw = ast.get_docstring(tree) or ""
    text = _normalize_text(raw)
    if text:
        return text
    return "No module docstring present; see function-level details below."


def _py_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, qualified_name: str) -> str:
    """Build a readable Python function signature string from AST fields."""

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    result = f"{prefix} {qualified_name}({args})"
    if node.returns is not None:
        result = f"{result} -> {ast.unparse(node.returns)}"
    return result


class _FunctionCollector(ast.NodeVisitor):
    """Collect top-level functions and class methods with scope-aware names."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.items: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []

    def _push(self, value: str) -> None:
        self.scope.append(value)

    def _pop(self) -> None:
        if self.scope:
            self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Ignore nested functions inside functions to keep reference focused on public/module-level flow.
        if any(item.startswith("<fn:") for item in self.scope):
            return

        qualified = ".".join([*self.scope, node.name]) if self.scope else node.name
        self.items.append((node, qualified))

        self._push(f"<fn:{node.name}>")
        self.generic_visit(node)
        self._pop()


def _collect_python_functions(tree: ast.Module) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]:
    """Return top-level functions and class methods in stable source order."""

    collector = _FunctionCollector()
    collector.visit(tree)
    return sorted(collector.items, key=lambda item: item[0].lineno)


def _document_python_file(path: Path) -> ArtifactDoc:
    """Extract module and function documentation from one Python source file."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    summary = _module_summary(tree)

    functions: list[FunctionDoc] = []
    for node, qualified_name in _collect_python_functions(tree):
        raw_doc = ast.get_docstring(node) or ""
        desc = _normalize_text(raw_doc) if raw_doc else _safe_sentence_from_name(qualified_name)
        functions.append(
            FunctionDoc(
                name=qualified_name,
                signature=_py_signature(node, qualified_name),
                description=desc,
                line=node.lineno,
            )
        )

    return ArtifactDoc(path=path, summary=summary, functions=functions)


def _extract_js_module_summary(lines: list[str]) -> str:
    """Return top-of-file JS block comment as artifact summary."""

    text = "\n".join(lines[:60])
    match = re.search(r"/\*\*(.*?)\*/", text, flags=re.S)
    if not match:
        return "Frontend controller and UI logic module."
    body = re.sub(r"^\s*\*\s?", "", match.group(1), flags=re.M)
    normalized = _normalize_text(body)
    return normalized or "Frontend controller and UI logic module."


def _extract_js_function_comment(lines: list[str], start_index: int) -> str:
    """Extract in-function leading comment block when present."""

    idx = start_index + 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and lines[idx].strip().startswith("/**"):
        buffer: list[str] = []
        while idx < len(lines):
            buffer.append(lines[idx])
            if "*/" in lines[idx]:
                break
            idx += 1
        text = "\n".join(buffer)
        text = re.sub(r"/\*\*|\*/", "", text)
        text = re.sub(r"^\s*\*\s?", "", text, flags=re.M)
        normalized = _normalize_text(text)
        if normalized:
            return normalized

    if idx < len(lines) and lines[idx].strip().startswith("//"):
        notes: list[str] = []
        while idx < len(lines) and lines[idx].strip().startswith("//"):
            notes.append(lines[idx].strip().lstrip("/").strip())
            idx += 1
        normalized = _normalize_text(" ".join(notes))
        if normalized:
            return normalized

    return ""


def _document_js_file(path: Path) -> ArtifactDoc:
    """Extract function signatures and comments from one JavaScript file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    summary = _extract_js_module_summary(lines)
    pattern = re.compile(r"^\s*(async\s+)?function\s+([A-Za-z0-9_]+)\s*\((.*?)\)\s*\{")

    functions: list[FunctionDoc] = []
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        async_prefix = "async " if match.group(1) else ""
        name = match.group(2)
        args = match.group(3).strip()
        desc = _extract_js_function_comment(lines, idx) or _safe_sentence_from_name(name)
        functions.append(
            FunctionDoc(
                name=name,
                signature=f"{async_prefix}function {name}({args})",
                description=desc,
                line=idx + 1,
            )
        )

    return ArtifactDoc(path=path, summary=summary, functions=functions)


def _collect_artifacts() -> list[ArtifactDoc]:
    """Build complete artifact/function documentation data structure."""

    artifacts: list[ArtifactDoc] = []
    for path in _iter_python_files():
        artifacts.append(_document_python_file(path))
    for path in JS_TARGETS:
        if path.exists():
            artifacts.append(_document_js_file(path))
    return sorted(artifacts, key=lambda item: str(item.path.relative_to(ROOT)))


def _render_markdown(artifacts: list[ArtifactDoc]) -> str:
    """Render artifact and function documentation into markdown text."""

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    total_functions = sum(len(item.functions) for item in artifacts)
    lines: list[str] = [
        "# Internal Code Artifact and Function Reference",
        "",
        f"Generated at `{generated_at}`.",
        "",
        "This file is generated by `python scripts/generate_internal_docs.py`.",
        "",
        "## Coverage",
        "",
        f"- Artifacts documented: {len(artifacts)}",
        f"- Functions documented: {total_functions}",
        "",
    ]

    for artifact in artifacts:
        rel = artifact.path.relative_to(ROOT)
        lines.extend(
            [
                f"## `{rel}`",
                "",
                f"**Artifact purpose:** {artifact.summary}",
                "",
            ]
        )
        if not artifact.functions:
            lines.extend(
                [
                    "No functions detected in this artifact.",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                "| Function | Signature | Description |",
                "| --- | --- | --- |",
            ]
        )
        for func in artifact.functions:
            signature = func.signature.replace("|", "\\|")
            description = func.description.replace("|", "\\|")
            lines.append(
                f"| `{func.name}` (`L{func.line}`) | `{signature}` | {description} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Generate and write the internal documentation markdown artifact."""

    parser = argparse.ArgumentParser(description="Generate internal artifact/function docs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    artifacts = _collect_artifacts()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(artifacts), encoding="utf-8")
    print(f"Wrote internal docs to {output_path}")


if __name__ == "__main__":
    main()
