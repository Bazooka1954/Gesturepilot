"""Camera abstraction and the OpenCV-backed webcam implementation.

Two things live here:

* :class:`Camera` -- the abstract interface the rest of GesturePilot codes against.
* :class:`WebcamCamera` -- the concrete implementation that drives ``cv2.VideoCapture``.

This is the only module in the camera layer that imports ``cv2``. Callers receive
:class:`~gesturepilot.camera.types.Frame` objects holding plain NumPy arrays, so the
gesture engine never needs to know that OpenCV is doing the capturing. Swapping in a
mock or file-backed camera is a matter of satisfying :class:`Camera`.

Both ``cv2.VideoCapture`` construction and the monotonic clock are injectable, which is
what lets the whole layer be unit tested without a physical webcam and without
sleeping.

Privacy: frames are returned to the calling application and nothing else. This module
never writes, saves, logs, or transmits frame data.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import cv2

from gesturepilot.camera.errors import CameraOpenError, CameraStateError, FrameReadError
from gesturepilot.camera.types import CameraConfig, CameraProperties, FpsMeter, Frame

#: Builds a raw capture handle for a device index. Injectable for testing.
CaptureFactory = Callable[[int], "cv2.VideoCapture"]


def _default_capture_factory(device_index: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(device_index)


def _sanitise_number(value: float) -> float | None:
    """Normalise a driver-reported float, rejecting unusable sentinels.

    Drivers commonly report ``0.0``, negatives, or NaN for properties they do not
    support. Those become ``None`` so callers cannot mistake them for real values.
    """
    if value != value:  # NaN
        return None
    if value <= 0:
        return None
    return value


@runtime_checkable
class Camera(Protocol):
    """Abstract camera interface.

    Lifecycle is ``CLOSED -> OPEN -> CLOSED``. Implementations must tolerate
    :meth:`close` being called repeatedly and must release the underlying device on
    close rather than relying on process termination.
    """

    @property
    def config(self) -> CameraConfig:
        """The settings this camera was configured with."""

    @property
    def is_open(self) -> bool:
        """Whether the device is currently open."""

    @property
    def properties(self) -> CameraProperties:
        """Requested settings plus whatever the device reported back."""

    @property
    def measured_fps(self) -> float | None:
        """Measured capture FPS, or ``None`` if not enough frames have been read."""

    @property
    def frames_read(self) -> int:
        """Total number of frames successfully returned since :meth:`open`."""

    def open(self) -> None:
        """Open the device.

        Raises:
            CameraStateError: If the camera is already open.
            CameraOpenError: If the device could not be initialised.
        """

    def read(self) -> Frame:
        """Capture and return a single frame.

        Raises:
            CameraStateError: If the camera is not open.
            FrameReadError: If capture fails or yields an unusable image.
        """

    def close(self) -> None:
        """Release the device. Safe to call more than once."""


class WebcamCamera:
    """A physical capture device driven through ``cv2.VideoCapture``.

    Args:
        config: Requested device index, resolution, and frame rate.
        capture_factory: Builds the raw capture handle. Override in tests to inject a
            fake device.
        clock: Monotonic time source. Override in tests to control FPS measurement.
        fps_window: Trailing window, in seconds, for FPS measurement.
    """

    def __init__(
        self,
        config: CameraConfig | None = None,
        *,
        capture_factory: CaptureFactory | None = None,
        clock: Callable[[], float] | None = None,
        fps_window: float = 1.0,
    ) -> None:
        self._config = config or CameraConfig()
        self._capture_factory = capture_factory or _default_capture_factory
        self._clock = clock or time.monotonic
        self._fps_meter = FpsMeter(window=fps_window, _clock=self._clock)
        self._capture: cv2.VideoCapture | None = None
        self._properties = CameraProperties(
            requested_width=self._config.width,
            requested_height=self._config.height,
            requested_fps=self._config.fps,
        )
        self._frames_read = 0

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def properties(self) -> CameraProperties:
        return self._properties

    @property
    def measured_fps(self) -> float | None:
        """Measured capture FPS. Never derived from the requested rate."""
        return self._fps_meter.fps

    @property
    def frames_read(self) -> int:
        return self._frames_read

    # -- lifecycle --------------------------------------------------------

    def open(self) -> None:
        if self._capture is not None:
            raise CameraStateError(
                f"Camera {self._config.device_index} is already open. "
                "Call close() before opening it again."
            )

        capture = self._capture_factory(self._config.device_index)

        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise CameraOpenError(
                f"Could not open camera device {self._config.device_index}. "
                "The device may be missing, disconnected, already in use by another "
                "application, or the index may be out of range."
            )

        self._apply_settings(capture)
        self._capture = capture
        self._properties = self._read_properties(capture)
        self._fps_meter.reset()
        self._frames_read = 0

    def read(self) -> Frame:
        capture = self._require_open()
        self._fps_meter.tick()

        ok, image = capture.read()

        if not ok or image is None:
            raise FrameReadError(
                f"Failed to read a frame from camera {self._config.device_index}. "
                "The device may have been disconnected or claimed by another process."
            )

        if image.size == 0 or image.ndim not in (2, 3):
            raise FrameReadError(
                f"Camera {self._config.device_index} returned an unusable frame "
                f"(shape={getattr(image, 'shape', None)}, size={getattr(image, 'size', 0)})."
            )

        frame = Frame(
            image=image,
            timestamp=self._clock(),
            sequence=self._frames_read,
        )
        self._frames_read += 1
        return frame

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        self._fps_meter.reset()
        self._frames_read = 0
        self._properties = CameraProperties(
            requested_width=self._config.width,
            requested_height=self._config.height,
            requested_fps=self._config.fps,
        )

    # -- context manager --------------------------------------------------

    def __enter__(self) -> WebcamCamera:
        # Tolerate an already-open camera so the documented
        # `with open_camera(config) as camera:` idiom works: open_camera() already
        # opened the device, and re-entering it must not raise.
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_open(self) -> cv2.VideoCapture:
        if self._capture is None:
            raise CameraStateError(
                "Camera is not open. Call open() (or use the camera as a context "
                "manager) before reading frames."
            )
        return self._capture

    def _apply_settings(self, capture: cv2.VideoCapture) -> None:
        """Request the configured settings.

        These are advisory: many webcams silently substitute a supported mode. The
        outcome is read back by :meth:`_read_properties`.
        """
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        if self._config.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(self._config.fps))

    def _read_properties(self, capture: cv2.VideoCapture) -> CameraProperties:
        """Read back what the device actually negotiated."""
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.CAP_PROP_FPS)

        return CameraProperties(
            requested_width=self._config.width,
            requested_height=self._config.height,
            requested_fps=self._config.fps,
            actual_width=int(width) if width > 0 else None,
            actual_height=int(height) if height > 0 else None,
            actual_fps=_sanitise_number(fps),
        )


def open_camera(
    config: CameraConfig | None = None,
    *,
    capture_factory: CaptureFactory | None = None,
    clock: Callable[[], float] | None = None,
) -> WebcamCamera:
    """Construct a webcam camera and open it.

    Convenience for the common case. The returned camera is already open and should
    be closed by the caller.

    Raises:
        CameraOpenError: If the device could not be opened.
    """
    camera = WebcamCamera(config, capture_factory=capture_factory, clock=clock)
    camera.open()
    return camera
