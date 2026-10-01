"""Global guard for unit tests.

The unit suite must never touch a physical webcam: it has to pass on a machine with no
camera, on one where the camera is busy, and in CI. Every unit test therefore injects
a fake capture handle, and this autouse fixture makes that a hard guarantee by
breaking any attempt to construct a real ``cv2.VideoCapture``.
"""

from __future__ import annotations

import cv2
import pytest


@pytest.fixture(autouse=True)
def forbid_real_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make constructing a real ``cv2.VideoCapture`` an immediate error.

    Without this, a test that forgot to inject ``capture_factory`` would quietly open
    the developer's webcam instead of failing.
    """

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Unit tests must not construct a real cv2.VideoCapture. "
            "Inject capture_factory=FakeCaptureFactory(...) instead."
        )

    monkeypatch.setattr(cv2, "VideoCapture", _blocked)


@pytest.fixture(autouse=True)
def forbid_real_landmarker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make loading the real hand-landmarker model an immediate error.

    Unit tests must run on a machine with no model asset and no GPU. Inject
    ``landmarker_factory=FakeLandmarkerFactory(...)`` so tests stay deterministic and
    need neither ``models/hand_landmarker.task`` nor the real MediaPipe engine.
    """

    from mediapipe.tasks.python.vision import HandLandmarker

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Unit tests must not load the real MediaPipe HandLandmarker. "
            "Inject landmarker_factory=FakeLandmarkerFactory(...) instead."
        )

    monkeypatch.setattr(HandLandmarker, "create_from_options", _blocked)
