"""Tests that the camera layer stays isolated from OpenCV.

Requirement: the gesture engine must never need to know that OpenCV is doing the
capturing. That is enforced here structurally rather than by convention.
"""

from __future__ import annotations

import ast
import inspect
import typing
from pathlib import Path

import numpy as np
import pytest

import gesturepilot.camera as camera_pkg
from gesturepilot.camera import Camera, CameraConfig, CameraProperties, Frame, WebcamCamera

from .conftest import FakeCapture, FakeCaptureFactory, FakeClock


class MockCamera:
    """A backend-free camera, written without OpenCV.

    Existing only to prove the abstraction is genuinely satisfiable by something that
    is not a webcam. If this compiles and passes the protocol check, a future test
    double or file-backed camera needs no OpenCV at all.
    """

    def __init__(self) -> None:
        self._config = CameraConfig()
        self._open = False
        self._props = CameraProperties(1280, 720, 30.0)

    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def properties(self) -> CameraProperties:
        return self._props

    @property
    def measured_fps(self) -> float | None:
        return None

    @property
    def frames_read(self) -> int:
        return 0

    def open(self) -> None:
        self._open = True

    def read(self) -> Frame:
        return Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), timestamp=0.0, sequence=0)

    def close(self) -> None:
        self._open = False

    def __enter__(self) -> MockCamera:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_webcam_camera_satisfies_the_protocol() -> None:
    assert isinstance(WebcamCamera(CameraConfig()), Camera)


def test_mock_camera_satisfies_the_protocol_without_opencv() -> None:
    """A non-OpenCV backend must be able to satisfy the interface."""
    assert isinstance(MockCamera(), Camera)


def test_mock_camera_is_usable_standalone() -> None:
    with MockCamera() as mock:
        frame = mock.read()

    assert frame.width == 4
    assert mock.is_open is False


def test_protocol_exposes_the_expected_surface() -> None:
    expected = {"open", "read", "close"}
    protocol_members = {
        name for name in dir(Camera) if not name.startswith("_") and callable(getattr(Camera, name))
    }

    assert expected.issubset(protocol_members)


# --------------------------------------------------------------------------
# No OpenCV leakage
# --------------------------------------------------------------------------


def _camera_module_dir() -> Path:
    return Path(camera_pkg.__file__).parent


def _imported_roots(path: Path) -> set[str]:
    """Top-level module names imported by a file.

    Uses the AST rather than text matching: these modules are allowed to *mention*
    OpenCV in prose, so only real imports count as leakage.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_types_module_does_not_import_opencv() -> None:
    """Value objects must stay backend-agnostic."""
    assert "cv2" not in _imported_roots(_camera_module_dir() / "types.py")


def test_errors_module_does_not_import_opencv() -> None:
    assert "cv2" not in _imported_roots(_camera_module_dir() / "errors.py")


def test_only_camera_module_imports_opencv() -> None:
    """Confines the OpenCV dependency to a single file."""
    module_dir = _camera_module_dir()
    importers = {path.name for path in module_dir.glob("*.py") if "cv2" in _imported_roots(path)}

    assert importers == {"camera.py"}


def test_frame_exposes_numpy_not_opencv() -> None:
    """Callers receive plain NumPy arrays; no OpenCV types leak into frames."""
    frame_hints = typing.get_type_hints(Frame)

    assert frame_hints["image"] is np.ndarray


def test_camera_protocol_hints_do_not_mention_opencv() -> None:
    """No OpenCV type appears anywhere in the public interface's annotations."""
    hints: list[str] = []
    for name, member in inspect.getmembers(Camera):
        if name.startswith("_"):
            continue
        target = member.fget if isinstance(member, property) else member
        if callable(target):
            hints.append(str(typing.get_type_hints(target)))

    joined = " ".join(hints)

    assert "cv2" not in joined
    assert "VideoCapture" not in joined


def test_public_exports_are_backend_agnostic() -> None:
    """Everything re-exported from the package must be free of OpenCV types."""
    for name in camera_pkg.__all__:
        exported = getattr(camera_pkg, name)
        annotations = str(getattr(exported, "__annotations__", {}))
        assert "cv2" not in annotations, f"{name} leaks an OpenCV annotation"


# --------------------------------------------------------------------------
# Error surface is uniform
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_name",
    ["CameraOpenError", "FrameReadError", "CameraStateError"],
)
def test_every_camera_error_derives_from_camera_error(error_name: str) -> None:
    from gesturepilot.camera.errors import CameraError

    assert issubclass(getattr(camera_pkg, error_name), CameraError)


def test_error_hierarchy_is_exported() -> None:
    for name in ("CameraError", "CameraOpenError", "CameraStateError", "FrameReadError"):
        assert name in camera_pkg.__all__


def test_capture_factory_injection_avoids_opencv_entirely() -> None:
    """A camera built from a fake capture never constructs a real handle."""
    factory = FakeCaptureFactory(FakeCapture())
    camera = WebcamCamera(CameraConfig(), capture_factory=factory, clock=FakeClock())

    camera.open()

    assert factory.requested_indices == [0]
    camera.close()
