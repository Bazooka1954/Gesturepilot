"""Structural boundary enforcement for the gesture-classification layer.

Recognition is the one stage where a shortcut would be most tempting and most damaging.
It is the first place a hand's geometry could plausibly be turned into an action, so the
things this package must never do are worth the same mechanical enforcement as the
things it must never import:

* it must depend on nothing but the processing layer's published domain types — no
  camera, no OpenCV, no MediaPipe, no NumPy, and no reaching into the tracking layer for
  a shortcut
* it must not read a clock, a file, or the network, because a classifier that varies
  between two calls on the same hand cannot be reasoned about or tested
* it must not mutate the features it is handed, since processing output is shared with
  the smoothing stage and any other consumer
* it must not filter on confidence, smooth across frames, or debounce, because those
  decisions belong to later pipeline stages and hiding them here would make a
  classification impossible to reason about from one input

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

PACKAGE_DIR = Path(__file__).resolve().parents[3] / "src" / "gesturepilot" / "classifier"
PROCESSOR_FILE = (
    Path(__file__).resolve().parents[3] / "src" / "gesturepilot" / "processing" / "processor.py"
)

#: Third-party roots this package must never import. The vision backends, the automation
#: surface, and NumPy — recognition is exact float arithmetic on ratios that a test pins
#: to literal numbers, which a vectorised library would quietly replace with its own
#: tolerances.
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
    }
)

#: The GesturePilot package this layer may depend on. Only its published domain types —
#: ``HandFeatures``, ``FINGER_CHAINS``, ``distance_2d``, ``Finger`` — because it consumes
#: processing output and invents no geometry of its own. Submodules are included:
#: ``gesturepilot.processing.topology`` is where the named landmark chains live.
ALLOWED_INTERNAL_PREFIX = "gesturepilot.processing"

#: Callables that would make a verdict depend on something other than its argument.
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
    }
)

#: Attributes that would give recognition state between frames. Anything that remembers a
#: previous call belongs to the confidence filter or the safety state machine.
FORBIDDEN_ATTRIBUTES = frozenset({"timestamp", "frame_sequence", "handedness"})


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


def _referenced_attributes(tree: ast.AST) -> set[str]:
    """Return every attribute name read anywhere in ``tree``."""
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }


def _call_names(tree: ast.AST) -> set[str]:
    """Return the bare name of every function called anywhere in ``tree``."""
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
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
    everything. The processing package really does import nothing from the backends, but
    the camera layer really does import OpenCV, so running the same detector over it
    proves the detector works.
    """
    camera = PROCESSOR_FILE.parent.parent / "camera" / "camera.py"
    assert sorted(_imported_roots(_parsed(camera)) & FORBIDDEN_IMPORTS) == ["cv2"], (
        f"the detector found no OpenCV import in {camera.name}, which does import it — "
        "the boundary checks above would pass vacuously"
    )


def test_no_backend_or_automation_imports() -> None:
    """MediaPipe, OpenCV, NumPy, and automation libraries must not appear in code."""
    offenders = _offending_files(_backend_imports)
    assert offenders == [], (
        "Gesture classification must stay free of the vision backends and the automation "
        "surface. Offences:\n" + "\n".join(offenders)
    )


def test_no_wall_clock_file_or_exec() -> None:
    """Determinism, privacy, and no action: a verdict must follow from its input alone.

    A hidden ``time.monotonic()`` would make the same hand classify differently on two
    calls, an ``open()`` would put a filesystem path into a package that only handles
    numbers, and ``eval`` would put a code path there too.
    """
    offenders = _offending_files(_forbidden_calls)
    assert offenders == [], (
        "Gesture classification must not read a clock, a file, or execute code. Offences:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("file_path", _source_files(), ids=lambda path: path.name)
def test_only_the_processing_layer_is_imported_from_gesturepilot(file_path: Path) -> None:
    """Internal imports are limited to ``gesturepilot.processing``.

    The layer boundary made explicit. Reaching into ``tracking`` would pull in the
    MediaPipe backend by association and would give recognition a second, competing
    definition of what a hand is.
    """
    internal = {
        module
        for module in _imported_modules(_parsed(file_path))
        if module.split(".")[0] == "gesturepilot"
        and not module.startswith("gesturepilot.classifier")
    }
    assert all(
        module == ALLOWED_INTERNAL_PREFIX or module.startswith(f"{ALLOWED_INTERNAL_PREFIX}.")
        for module in internal
    ), (
        f"{file_path.name} may only import {ALLOWED_INTERNAL_PREFIX} and its submodules, "
        f"found {sorted(internal)}"
    )


@pytest.mark.parametrize("file_path", _source_files(), ids=lambda path: path.name)
def test_temporal_and_handedness_fields_are_never_read(file_path: Path) -> None:
    """No rule may consult a timestamp, a frame number, or a handedness label.

    This is the mechanical form of the two promises that matter most for correctness: the
    classifier has no memory, and handedness is source metadata the tracking layer may
    have inverted. Reading any of the three attributes would make a verdict depend on
    something other than the hand's geometry.
    """
    found = sorted(_referenced_attributes(_parsed(file_path)) & FORBIDDEN_ATTRIBUTES)
    assert found == [], (
        f"{file_path.name} reads {found}, which must not influence any rule: a static "
        "classifier has no memory, and handedness may be inverted"
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


def test_no_landmark_index_is_written_out_by_hand() -> None:
    """Landmarks are reached through ``FINGER_CHAINS``, never as integer literals.

    A bare ``8`` where the index fingertip belongs is exactly the kind of line that
    survives a topology change and quietly measures the wrong joint. Subscripting the
    normalised landmark array with a non-``LandmarkIndex`` literal is the only way this
    can happen, so that is what is checked.
    """
    offenders: list[str] = []
    for file_path in _source_files():
        tree = _parsed(file_path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
                continue
            value = node.slice.value
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if getattr(node.value, "id", "") in ("normalized", "normalized_landmarks"):
                offenders.append(f"{file_path.name} indexes landmarks with literal {value}")
    assert offenders == [], "\n".join(offenders)


def test_public_api_exposes_no_backend_objects() -> None:
    """Nothing re-exported from the package may come from a backend or NumPy."""
    import gesturepilot.classifier as classifier

    for name in classifier.__all__:
        exported = getattr(classifier, name)
        module = getattr(type(exported), "__module__", "")
        root = module.split(".")[0]
        assert root not in FORBIDDEN_IMPORTS, f"{name} leaked from {module}"


def test_public_api_is_small_and_typed() -> None:
    """Every exported name is a type or a small immutable constant.

    The classifier is the first stage whose output an action would depend on, so its
    surface is kept to what a caller genuinely needs: a label, a config, a result, the
    evidence behind it, the score constants, and the two errors.
    """
    import gesturepilot.classifier as classifier

    assert len(classifier.__all__) <= 10
    for name in classifier.__all__:
        exported = getattr(classifier, name)
        assert inspect.isclass(exported) or isinstance(exported, (float, int, tuple)), (
            f"{name} is neither a type nor an immutable constant"
        )


def test_the_classifier_holds_no_mutable_state() -> None:
    """``GestureClassifier`` stores its config and nothing else.

    A second instance attribute would be state that survives between calls, which is the
    one thing a static classifier must not have.
    """
    from gesturepilot.classifier import ClassifierConfig, GestureClassifier

    classifier = GestureClassifier(ClassifierConfig())
    assert vars(classifier) == {"_config": classifier.config}


def test_the_public_surface_holds_no_frame_objects() -> None:
    """No camera ``Frame``, no ``TrackingResult``, and no ``Handedness`` in the API.

    If one of those appeared, this layer would be holding something that belongs to an
    earlier stage, and a caller would be able to confuse a frame with a classification.
    """
    import gesturepilot.classifier as classifier

    banned = ("frame", "camera", "cv2", "handedness", "trackedhand", "trackingresult")
    offenders = [name for name in classifier.__all__ if any(w in name.lower() for w in banned)]
    assert offenders == [], f"Earlier-stage vocabulary has leaked into the classifier: {offenders}"


def test_the_package_docstring_points_at_the_architecture_notes() -> None:
    """``docs/classifier.md`` is where the design decisions and their limits are argued."""
    package = PACKAGE_DIR / "__init__.py"
    assert "docs/classifier.md" in package.read_text(encoding="utf-8")


def test_the_documented_rule_table_matches_the_rules() -> None:
    """The module docstring's table is a contract, not decoration.

    A gesture table that has drifted from the code is worse than no table, because it is
    the first thing a reviewer reads. Each gesture named in the table must be one the
    classifier can actually return.
    """
    from gesturepilot.classifier import RECOGNIZABLE_GESTURES

    table = (PACKAGE_DIR / "classifier.py").read_text(encoding="utf-8")
    for gesture in RECOGNIZABLE_GESTURES:
        assert gesture.name in table, f"{gesture.name} is missing from the documented table"
