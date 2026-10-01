"""Shared fakes for hand-tracking unit tests.

Tests drive the full conversion path — camera ``Frame`` in, ``TrackingResult`` out —
without loading a model or touching a webcam. Only the final inference step is faked,
which is the point: everything we own is still exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gesturepilot.camera.types import Frame


@dataclass
class FakeLandmark:
    """Stand-in for a MediaPipe ``NormalizedLandmark``."""

    x: float
    y: float
    z: float


@dataclass
class FakeCategory:
    """Stand-in for a MediaPipe ``Category``."""

    category_name: str
    score: float = 0.99
    index: int = 0


@dataclass
class FakeHand:
    """One fake hand's worth of raw backend output."""

    landmarks: list[FakeLandmark] = field(default_factory=list)
    handedness: list[FakeCategory] | None = None


def make_landmarks(count: int = 21, offset: float = 0.0) -> list[FakeLandmark]:
    """Build ``count`` plausible, deterministic landmarks."""
    return [
        FakeLandmark(x=0.1 + offset + i * 0.01, y=0.2 + i * 0.01, z=-0.05 + i * 0.001)
        for i in range(count)
    ]


class FakeLandmarkerResult:
    """Stand-in for ``HandLandmarkerResult``.

    Mirrors the real shape exactly: only ``hand_landmarks``, ``handedness``, and
    ``hand_world_landmarks``. Notably absent is any per-frame detection confidence.
    """

    def __init__(
        self,
        hand_landmarks: list[list[FakeLandmark]] | None = None,
        handedness: list[list[FakeCategory]] | None = None,
        hand_world_landmarks: list[list[FakeLandmark]] | None = None,
    ) -> None:
        self.hand_landmarks = hand_landmarks if hand_landmarks is not None else []
        self.handedness = handedness if handedness is not None else []
        self.hand_world_landmarks = hand_world_landmarks or []


class FakeLandmarker:
    """Fake ``HandLandmarker`` that records how it was called.

    Attributes:
        results: Consumed one per detection call; the last entry repeats once
            exhausted. Keeps multi-frame tests short.
        calls: Every ``(method, timestamp_ms_or_None)`` seen, for order assertions.
        closed: Whether ``close()`` was called.
    """

    def __init__(self, *results: Any) -> None:
        self._results = list(results) or [FakeLandmarkerResult()]
        self._index = 0
        self.calls: list[tuple[str, int | None]] = []
        self.closed = False
        self.raise_on_detect: Exception | None = None

    def _next(self) -> Any:
        result = self._results[min(self._index, len(self._results) - 1)]
        self._index += 1
        if self.raise_on_detect is not None:
            raise self.raise_on_detect
        return result

    def detect(self, image: Any) -> Any:
        self.calls.append(("detect", None))
        return self._next()

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any:
        self.calls.append(("detect_for_video", timestamp_ms))
        return self._next()

    def close(self) -> None:
        self.closed = True


class FakeLandmarkerFactory:
    """Callable that returns a :class:`FakeLandmarker` instead of a real one.

    Records the options it was handed so tests can assert the config reached the
    backend.
    """

    def __init__(self, landmarker: FakeLandmarker | None = None) -> None:
        self.landmarker = landmarker or FakeLandmarker()
        self.options: list[Any] = []

    def __call__(self, options: Any) -> Any:
        self.options.append(options)
        return self.landmarker


def make_frame(
    width: int = 64, height: int = 48, timestamp: float = 1.0, sequence: int = 0
) -> Frame:
    """Build a small BGR test frame."""
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    return Frame(image=image, timestamp=timestamp, sequence=sequence)
