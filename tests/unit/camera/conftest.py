"""Test doubles for the camera layer.

These stand in for a physical webcam and for wall-clock time, so the unit suite is
deterministic and hardware-free.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np
import pytest

#: A typical colour frame: 640x480, three channels (BGR), uint8.
DEFAULT_SHAPE = (480, 640, 3)


def make_frame(width: int = 640, height: int = 480, channels: int = 3) -> np.ndarray:
    """Build a deterministic test frame."""
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.full(shape, 128, dtype=np.uint8)


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture``.

    Mimics the small surface the camera layer actually uses: ``isOpened``, ``set``,
    ``get``, ``read``, and ``release``.
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        opens: bool = True,
        frames: Sequence[tuple[bool, np.ndarray | None]] | None = None,
        props: dict[int, float] | None = None,
    ) -> None:
        self.device_index = device_index
        self._opens = opens
        self._frames = list(frames) if frames is not None else None
        self._props = dict(props) if props is not None else {}
        self.release_count = 0
        self.set_calls: list[tuple[int, float]] = []
        self.read_count = 0

    # -- cv2.VideoCapture surface ----------------------------------------

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV spelling
        return self._opens and self.release_count == 0

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        return self._props.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self._frames is None:
            return True, make_frame()
        if self._frames:
            return self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.release_count += 1


class FakeCaptureFactory:
    """Callable stand-in for the capture factory injected into ``WebcamCamera``."""

    def __init__(self, capture: Any = None, *, raises: Exception | None = None) -> None:
        self._capture = capture if capture is not None else FakeCapture()
        self._raises = raises
        self.requested_indices: list[int] = []

    def __call__(self, device_index: int) -> Any:
        self.requested_indices.append(device_index)
        if self._raises is not None:
            raise self._raises
        return self._capture

    @property
    def capture(self) -> Any:
        return self._capture


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now += seconds
        return self


@pytest.fixture
def clock() -> FakeClock:
    """A frozen-at-zero monotonic clock."""
    return FakeClock()


@pytest.fixture
def capture_factory() -> FakeCaptureFactory:
    """A factory returning a capture that opens and yields valid frames."""
    return FakeCaptureFactory(FakeCapture())


@pytest.fixture
def capture(capture_factory: FakeCaptureFactory) -> FakeCapture:
    """The capture behind the default ``capture_factory`` fixture."""
    return capture_factory.capture


@pytest.fixture
def make_factory() -> Callable[..., FakeCaptureFactory]:
    """Factory-of-factories for building configured capture doubles in a test."""
    return FakeCaptureFactory


__all__ = [
    "DEFAULT_SHAPE",
    "FakeCapture",
    "FakeCaptureFactory",
    "FakeClock",
    "cv2",
    "make_frame",
]
