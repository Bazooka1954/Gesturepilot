"""AST-based boundary enforcement for the tracking layer.

MediaPipe is a heavy dependency and none of its types are part of GesturePilot's public
contract. This test scans the tracking package and fails if MediaPipe is imported or
referenced anywhere other than the tracker implementation.

Prose may mention MediaPipe (docstrings legitimately explain the design); code may
not reach for it. An AST walk distinguishes the two.
"""

from __future__ import annotations

import ast
from pathlib import Path

TRACKING_DIR = Path(__file__).resolve().parents[3] / "src" / "gesturepilot" / "tracking"

#: The single module permitted to touch MediaPipe.
ALLOWED = {"tracker.py"}


def _find_mediapipe_references(source: str, filename: str) -> list[tuple[int, str]]:
    """Return ``(lineno, reason)`` for every MediaPipe reference in executable code."""

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.hits: list[tuple[int, str]] = []

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name.split(".")[0] == "mediapipe":
                    self.hits.append((node.lineno, f"import {alias.name}"))
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module and node.module.split(".")[0] == "mediapipe":
                self.hits.append((node.lineno, f"from {node.module} import ..."))
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id == "mediapipe":
                self.hits.append((node.lineno, "use of 'mediapipe'"))
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "mediapipe":
                self.hits.append((node.lineno, "use of 'mediapipe.<attr>'"))
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(ast.parse(source, filename=filename))
    return visitor.hits


def test_mediapipe_confined_to_tracker_implementation() -> None:
    offenders: list[str] = []
    for file_path in sorted(TRACKING_DIR.glob("*.py")):
        if file_path.name in ALLOWED:
            continue
        hits = _find_mediapipe_references(file_path.read_text(encoding="utf-8"), file_path.name)
        offenders += [f"  {file_path.name}:{line}: {reason}" for line, reason in hits]

    assert offenders == [], (
        "MediaPipe must be confined to tracking/tracker.py. Offences:\n" + "\n".join(offenders)
    )


def test_public_api_exposes_no_mediapipe_symbols() -> None:
    """Nothing exported from the package should be a MediaPipe object."""
    import gesturepilot.tracking as tracking

    for name in tracking.__all__:
        exported = getattr(tracking, name)
        module = getattr(type(exported), "__module__", "")
        assert not module.startswith("mediapipe"), f"{name} leaked from {module}"
