"""Structural boundary enforcement for the landmark-processing layer.

Three architectural promises are worth more than any individual measurement, because they
are the ones that stop being true slowly: this package must stay free of the vision
backends, it must stay a pure function of its inputs, and it must not reach for a clock or
the filesystem behind the caller's back.

Those are checked here by walking the source with :mod:`ast` rather than by trusting
review. Prose may name MediaPipe or OpenCV freely — docstrings legitimately explain why
those libraries are excluded — so only executable code is examined.

One deliberate dependency remains: ``gesturepilot.tracking.types`` for ``TrackedHand``,
``Landmark``, ``Handedness``, and ``LANDMARK_COUNT``. Reusing the tracking layer's own
vocabulary is what keeps the two layers agreeing on what a hand is; duplicating those
types to avoid the import would be worse. See ``docs/processing.md``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[3] / "src" / "gesturepilot" / "processing"
TRACKER_FILE = (
    Path(__file__).resolve().parents[3] / "src" / "gesturepilot" / "tracking" / "tracker.py"
)

#: Third-party roots this package must never import. The backend libraries, plus NumPy —
#: all derived measurements here are exact float arithmetic that a test can pin to a
#: literal, which a vectorised library would quietly replace with its own tolerances.
FORBIDDEN_IMPORTS = frozenset({"mediapipe", "cv2", "numpy", "PIL", "torch", "tensorflow"})

#: The one GesturePilot package this layer is allowed to depend on, and only for the
#: domain types the tracking layer publishes.
ALLOWED_INTERNAL_IMPORTS = frozenset({"gesturepilot.tracking.types"})

#: Callables that would make a result depend on something other than its arguments.
FORBIDDEN_CALLS = frozenset(
    {"time", "perf_counter", "monotonic", "time_ns", "monotonic_ns", "open"}
)


def _source_files() -> list[Path]:
    files = sorted(PACKAGE_DIR.glob("*.py"))
    assert files, f"no Python files found in {PACKAGE_DIR}"
    return files


def _imported_roots(tree: ast.AST) -> set[str]:
    """Return the top-level name of every module imported anywhere in ``tree``."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_modules(tree: ast.AST) -> set[str]:
    """Return every absolute module name imported anywhere in ``tree``."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def _call_names(tree: ast.AST) -> set[str]:
    """Return the bare name of every function called anywhere in ``tree``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def _parsed(file_path: Path) -> ast.AST:
    return ast.parse(file_path.read_text(encoding="utf-8"), filename=file_path.name)


def _offending_files(check: Callable[[Path], list[str]]) -> list[str]:
    """Return a report line for every source file that trips ``check``."""
    offenders: list[str] = []
    for file_path in _source_files():
        offenders.extend(f"  {file_path.name}: {reason}" for reason in check(file_path))
    return offenders


def _backend_imports(file_path: Path) -> list[str]:
    return [
        f"imports {root}"
        for root in sorted(_imported_roots(_parsed(file_path)) & FORBIDDEN_IMPORTS)
    ]


def _forbidden_calls(file_path: Path) -> list[str]:
    return [f"calls {name}()" for name in sorted(_call_names(_parsed(file_path)) & FORBIDDEN_CALLS)]


def test_the_detector_actually_detects() -> None:
    """Guard against a vacuous check.

    The AST walk above is the whole enforcement mechanism, so a typo in it would silently
    pass everything. The tracking tracker really does import MediaPipe, so running the same
    detector over it proves the detector works.
    """
    assert _backend_imports(TRACKER_FILE), (
        f"the detector found no backend import in {TRACKER_FILE.name}, which does import "
        "MediaPipe — the boundary checks above would pass vacuously"
    )


def test_no_backend_imports() -> None:
    """MediaPipe, OpenCV, and NumPy must not appear in executable code."""
    offenders = _offending_files(_backend_imports)
    assert offenders == [], (
        "Landmark processing must stay free of the vision backends. Offences:\n"
        + "\n".join(offenders)
    )


def test_no_wall_clock_or_file_access() -> None:
    """Determinism and privacy: the caller supplies time, and nothing is read or written.

    A hidden ``time.monotonic()`` would make every measurement irreproducible in a test,
    and an ``open()`` would put a filesystem or network path into a package that only ever
    handles numbers.
    """
    offenders = _offending_files(_forbidden_calls)
    assert offenders == [], (
        "Landmark processing must not read a clock or a file. Offences:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("file_path", _source_files(), ids=lambda path: path.name)
def test_only_the_tracking_domain_types_are_imported_from_gesturepilot(file_path: Path) -> None:
    """Internal imports are limited to the tracking layer's published types.

    This is the layer boundary made explicit: the only thing this package shares with the
    tracking layer is its vocabulary. Reaching into ``tracking.tracker`` would pull the
    MediaPipe backend along with it.
    """
    internal = {
        module
        for module in _imported_modules(_parsed(file_path))
        if module.split(".")[0] == "gesturepilot"
        and not module.startswith("gesturepilot.processing")
    }
    assert internal <= ALLOWED_INTERNAL_IMPORTS, (
        f"{file_path.name} may only import {sorted(ALLOWED_INTERNAL_IMPORTS)}, "
        f"found {sorted(internal)}"
    )


@pytest.mark.parametrize("file_path", _source_files(), ids=lambda path: path.name)
def test_every_module_declares_its_exports(file_path: Path) -> None:
    """Each module ends with an ``__all__``, so its public surface is deliberate.

    Without this, a helper added to ``geometry`` would silently become part of the
    package's contract the moment someone imported it from the package root.
    """
    tree = _parsed(file_path)
    exported = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ]
    assert exported, f"{file_path.name} has no __all__"


def test_public_api_exposes_no_backend_objects() -> None:
    """Nothing re-exported from the package may come from a backend or NumPy."""
    import gesturepilot.processing as processing

    for name in processing.__all__:
        exported = getattr(processing, name)
        module = getattr(type(exported), "__module__", "")
        root = module.split(".")[0]
        assert root not in FORBIDDEN_IMPORTS, f"{name} leaked from {module}"


def test_public_api_mentions_no_gesture_words() -> None:
    """The layer must expose geometry, not classifications.

    ``HandFeatures`` is descriptive by contract. If a future name like ``is_pointing`` or
    ``is_pinching`` appeared in the public API, this stage would have grown a judgement it
    is not supposed to make — and the classifier phase would have less left to own.
    """
    import gesturepilot.processing as processing

    banned = ("gesture", "pinch", "fist", "pointing", "swipe", "wave", "victory")
    offenders = [
        name for name in processing.__all__ if any(word in name.lower() for word in banned)
    ]
    assert offenders == [], f"Gesture vocabulary has leaked into the processing API: {offenders}"


def test_the_package_docstring_points_at_the_architecture_notes() -> None:
    """``docs/processing.md`` is where the design decisions and their limits are argued."""
    package = PACKAGE_DIR / "__init__.py"
    assert "docs/processing.md" in package.read_text(encoding="utf-8")
