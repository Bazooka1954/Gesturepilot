"""Typed value objects for the camera layer.

This module is deliberately free of any OpenCV (``cv2``) import. Everything the rest
of GesturePilot needs to describe a camera, a frame, and measured throughput is
expressed with the standard library plus NumPy, so higher layers never depend on the
capture backend.

Privacy note: frames produced here are handed to the local application and nothing
else. Nothing in this module writes frames to disk, logs their contents, or transmits
them anywhere.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Requested camera settings.

    These values are *requests* to the driver, not guarantees. A webcam may ignore a
    resolution or frame rate it cannot deliver. Use :attr:`CameraProperties` to read
    back what the device actually negotiated.

    Attributes:
        device_index: Zero-based capture device index. ``0`` is the usual default.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Requested capture frame rate, or ``None`` to leave the device default
            alone. A value here is a request and must not be reported as the real
            capture rate; see :class:`FpsMeter`.
    """

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float | None = 30.0

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError(f"device_index must be >= 0, got {self.device_index}")
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be > 0 or None, got {self.fps}")

    @property
    def resolution(self) -> tuple[int, int]:
        """Requested ``(width, height)``."""
        return (self.width, self.height)


@dataclass(frozen=True)
class CameraProperties:
    """Requested settings alongside what the device reported back.

    Keeping both sides side by side makes it obvious at a glance when a camera
    refused the requested resolution or frame rate.
    """

    requested_width: int
    requested_height: int
    requested_fps: float | None
    actual_width: int | None = None
    actual_height: int | None = None
    actual_fps: float | None = None

    @property
    def actual_resolution(self) -> tuple[int, int] | None:
        """Device-reported ``(width, height)``, or ``None`` if unreported."""
        if self.actual_width is None or self.actual_height is None:
            return None
        return (self.actual_width, self.actual_height)

    @property
    def matched_request(self) -> bool:
        """Whether the device honoured the requested resolution.

        ``True`` when the resolution was reported and matches the request exactly.
        Unknown (unreported) resolutions return ``False``.
        """
        return self.actual_resolution == (self.requested_width, self.requested_height)


@dataclass(frozen=True)
class Frame:
    """A single captured frame and its metadata.

    Attributes:
        image: The captured pixels as a NumPy array, in OpenCV's native BGR channel
            order and dtype. Treated as read-only by convention: callers must not
            mutate it, and the camera never writes to it after returning.
        timestamp: Monotonic timestamp in seconds taken immediately after capture.
            Monotonic rather than wall-clock so intervals and FPS math stay correct
            if the system clock is adjusted.
        sequence: Zero-based, monotonically increasing capture counter. Lets a future
            producer/consumer pipeline detect gaps or reordering.
    """

    image: np.ndarray
    timestamp: float
    sequence: int

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.image.shape[0])

    @property
    def channels(self) -> int:
        """Number of channels per pixel (3 for a colour BGR frame)."""
        return int(self.image.shape[2]) if self.image.ndim == 3 else 1


@dataclass
class FpsMeter:
    """Sliding-window measurement of actual capture frame rate.

    Distinguishes *measured* throughput from the *requested* rate in
    :class:`CameraConfig`. A camera asked for 30 FPS may deliver 12, and this is how
    you find out.

    The clock is injectable so tests can drive time deterministically without
    sleeping.

    Attributes:
        window: Width in seconds of the trailing measurement window.
    """

    window: float = 1.0
    _clock: Callable[[], float] = time.monotonic
    _samples: deque[float] = field(init=False, repr=False)
    _last_tick: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be > 0, got {self.window}")
        self._samples = deque()

    def tick(self) -> float | None:
        """Record a capture and return the seconds elapsed since the previous one.

        Returns:
            Seconds since the preceding :meth:`tick`, or ``None`` for the first tick
            (there is no interval to report yet).
        """
        now = self._clock()
        previous = self._last_tick
        self._last_tick = now
        self._samples.append(now)
        self._prune(now)
        return None if previous is None else now - previous

    @property
    def fps(self) -> float | None:
        """Measured capture FPS over the trailing window.

        Returns:
            Frames per second, or ``None`` until at least two frames have been
            recorded inside the window.
        """
        if len(self._samples) < 2:
            return None
        span = self._samples[-1] - self._samples[0]
        if span <= 0:
            return None
        # n timestamps describe n-1 intervals, so divide by (n-1).
        return (len(self._samples) - 1) / span

    @property
    def sample_count(self) -> int:
        """Number of ticks currently inside the measurement window."""
        return len(self._samples)

    def reset(self) -> None:
        """Discard all recorded samples and the previous tick."""
        self._samples.clear()
        self._last_tick = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._samples and self._samples[0] < cutoff:
            self._samples.popleft()
