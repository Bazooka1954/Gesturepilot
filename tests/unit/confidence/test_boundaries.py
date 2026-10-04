"""Structural boundary enforcement for the temporal confidence filter.

This is the first layer with memory, which makes it the first place where a shortcut would
be *easy*: state can quietly accumulate across hands, a clock can creep in through
``time.monotonic()``, and an accepted gesture is one refactor away from becoming an action.
The things this package must never do are therefore worth the same mechanical enforcement
the classifier gets for the things it must never import:

* it must depend on nothing but the classifier's published domain types — no camera, no
  OpenCV, no MediaPipe, no NumPy, and no reaching back into processing or tracking
* it must not read a clock, start a thread, touch a file, or open a socket, because a
  stability verdict that varied between two runs on the same input could not be reasoned
  about or tested
* it must name no individual gesture, so adding a sixth gesture later is a classifier change
  and not a filter change
* it must hold exactly the state it documents, so a run cannot be built from something the
  documentation does not mention

Those are checked by walking the source with :mod:`ast` rather than by trusting review.
Prose may name MediaPipe or OpenCV freely — docstrings legitimately explain why those
libraries are excluded — so only executable code is examined.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[3] / "src" / "gesturepilot" / "confidence"
CAMERA_FILE = PACKAGE_DIR.parent / "camera" / "camera.py"

#: Third-party roots this package must never import. The vision backends, the automation
#: surface, NumPy — confidence arithmetic is plain float addition and division that a test
#: pins to literal numbers — and the concurrency and clock modules, since a filter that
#: owns a thread or reads a clock is no longer a function of its inputs.
FORBIDDEN_IMPORTS = frozenset(
    {
        "mediapipe",
        "cv2",
        "numpy",
        "PIL",
        "torch",
        "tensorflow",
        "pywinauto",
        "pyautogui",
        "keyboard",
        "mouse",
        "win32api",
        "threading",
        "asyncio",
        "concurrent",
        "multiprocessing",
        "time",
        "datetime",
        "random",
        "subprocess",
        "socket",
        "requests",
    }
)

#: The GesturePilot package this layer may depend on: the classifier's published domain
#: types — ``Gesture``, ``GestureClassification``, and the confidence range constants —
#: because classification output is its only input. Submodules are included.
ALLOWED_INTERNAL_PREFIX = "gesturepilot.classifier"

#: Callables that would make a verdict depend on something other than its argument. The
#: timestamp the filter accepts is passed in by the caller; nothing here may go and get one.
FORBIDDEN_CALLS = frozenset(
    {
        "time",
        "perf_counter",
        "monotonic",
        "time_ns",
        "monotonic_ns",
        "sleep",
        "open",
        "input",
        "eval",
        "exec",
        "compile",
        "__import__",
    }
)

#: The attributes the filter holds, beyond its configuration. Each is part of the run the
#: documentation describes; anything else would be state a caller could not see.
EXPECTED_FILTER_STATE = frozenset(
    {
        "_config",
        "_candidate",
        "_candidate_observations",
        "_confidence_sum",
        "_last_timestamp",
    }
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
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _gesture_attribute_names(tree: ast.AST) -> set[str]:
    """Return every ``Gesture.<MEMBER>`` label named in executable code."""
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Gesture"
    }


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

    The AST walk is the whole enforcement mechanism, so a typo in it would silently pass
    everything. The confidence package really does import none of these, but the camera
    layer really does import OpenCV, so running the same detector over it proves the
    detector works.
    """
    found = _imported_roots(_parsed(CAMERA_FILE)) & FORBIDDEN_IMPORTS
    assert "cv2" in found, (
        f"the detector found no OpenCV import in {CAMERA_FILE.name}, which does import it — "
        "the boundary checks above would pass vacuously"
    )


def test_no_backend_automation_or_concurrency_imports() -> None:
    """MediaPipe, OpenCV, NumPy, automation libraries, threads, and clocks must not appear."""
    offenders = _offending_files(_backend_imports)
    assert offenders == [], (
        "Temporal confidence filtering must stay free of the vision backends, the "
        "automation surface, and any concurrency or clock module. Offences:\n"
        + "\n".join(offenders)
    )


def test_no_wall_clock_file_or_exec() -> None:
    """Determinism, privacy, and no action: a verdict must follow from its input alone.

    A hidden ``time.monotonic()`` would make the same classifications stabilise differently
    on two runs, an ``open()`` would put a filesystem path into a package that only handles
    numbers, and ``eval`` would put a code path there too.
    """
    offenders = _offending_files(_forbidden_calls)
    assert offenders == [], (
        "Confidence filtering must not read a clock, a file, or execute code. Offences:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("file_path", _source_files(), ids=lambda path: path.name)
def test_only_the_classifier_layer_is_imported_from_gesturepilot(file_path: Path) -> None:
    """Internal imports are limited to ``gesturepilot.classifier``.

    The layer boundary made explicit. Reaching into ``camera`` or ``tracking`` would pull in
    the MediaPipe backend by association and would give temporal filtering a second,
    competing definition of what an observation is.
    """
    internal = {
        module
        for module in _imported_modules(_parsed(file_path))
        if module.split(".")[0] == "gesturepilot"
        and not module.startswith("gesturepilot.confidence")
    }
    assert all(
        module == ALLOWED_INTERNAL_PREFIX or module.startswith(f"{ALLOWED_INTERNAL_PREFIX}.")
        for module in internal
    ), (
        f"{file_path.name} may only import {ALLOWED_INTERNAL_PREFIX} and its submodules, "
        f"found {sorted(internal)}"
    )


@pytest.mark.parametrize("file_path", _source_files(), ids=lambda path: path.name)
def test_no_gesture_is_named_in_executable_code(file_path: Path) -> None:
    """The filter is label-agnostic; adding a sixth gesture is a classifier change.

    A ``Gesture.POINT`` special case in here would mean the temporal policy quietly knows
    which gestures exist, so adding one would need a change in two layers. Prose may name
    them freely — only executable code is examined.
    """
    found = sorted(_gesture_attribute_names(_parsed(file_path)))
    assert found == [], (
        f"{file_path.name} names {found}; the filter must decide from the label's identity "
        "and confidence, never from which gesture it happens to be"
    )


@pytest.mark.parametrize("file_path", _source_files(), ids=lambda path: path.name)
def test_every_module_declares_its_exports(file_path: Path) -> None:
    """Each module ends with an ``__all__``, so its public surface is deliberate.

    Without this, a helper added to ``types`` would silently become part of the package's
    contract the moment someone imported it from the package root.
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
    import gesturepilot.confidence as confidence

    for name in confidence.__all__:
        exported = getattr(confidence, name)
        module = getattr(type(exported), "__module__", "")
        root = module.split(".")[0]
        assert root not in FORBIDDEN_IMPORTS, f"{name} leaked from {module}"


def test_public_api_is_small_and_typed() -> None:
    """Every exported name is a type.

    The filter sits directly upstream of a safety stage, so its surface is kept to what a
    caller genuinely needs: the filter, its configuration, its report, the two policy enums,
    and the error hierarchy.
    """
    import gesturepilot.confidence as confidence

    assert len(confidence.__all__) <= 10
    for name in confidence.__all__:
        assert inspect.isclass(getattr(confidence, name)), f"{name} is not a type"


def test_the_filter_holds_only_its_documented_state() -> None:
    """No instance attribute beyond the ones the design names.

    The module docstring claims the entire state is the candidate, its run length, and its
    confidence sum — plus the config and the caller's last timestamp. A sixth attribute
    would be state that survives between observations without appearing in the design, which
    is exactly the kind of thing that makes a temporal filter impossible to reason about.
    """
    from gesturepilot.confidence import ConfidenceFilter, ConfidenceFilterConfig

    filter_ = ConfidenceFilter(ConfidenceFilterConfig())
    assert set(vars(filter_)) == EXPECTED_FILTER_STATE


def test_the_public_surface_holds_no_frame_or_action_vocabulary() -> None:
    """No camera ``Frame``, no landmark type, and nothing that sounds like an action.

    If one of those appeared, this layer would be holding something that belongs to another
    stage, and a caller would be able to confuse a frame with a classification or a stable
    gesture with a confirmed one.
    """
    import gesturepilot.confidence as confidence

    banned = (
        "frame",
        "camera",
        "cv2",
        "cv2",
        "hand",
        "landmark",
        "action",
        "dispatch",
        "event",
        "safety",
        "trigger",
    )
    offenders = [name for name in confidence.__all__ if any(w in name.lower() for w in banned)]
    assert offenders == [], f"Another stage's vocabulary has leaked into the filter: {offenders}"


def test_the_package_docstring_points_at_the_architecture_notes() -> None:
    """``docs/confidence-filter.md`` is where the design decisions and limits are argued."""
    package = PACKAGE_DIR / "__init__.py"
    assert "docs/confidence-filter.md" in package.read_text(encoding="utf-8")


def test_the_documented_timestamp_policy_is_still_documented() -> None:
    """The filter module's timestamp rules are a contract, not decoration.

    A caller cannot infer whether an equal, missing, or backwards timestamp is legal, and
    getting that wrong produces a filter that quietly accumulates evidence across an
    interruption. Each rule named in the docstring must still be spelled out.
    """
    text = (PACKAGE_DIR / "filter.py").read_text(encoding="utf-8")
    for phrase in ("Omitted", "Equal", "Backwards", "Too far ahead"):
        assert phrase in text, f"the {phrase!r} timestamp rule is no longer documented"
