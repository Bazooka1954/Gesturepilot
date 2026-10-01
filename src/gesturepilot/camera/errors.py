"""Camera-specific exception hierarchy.

Callers interact with camera failures through these types rather than catching raw
OpenCV errors. This keeps the ``cv2`` dependency confined to
:mod:`gesturepilot.camera.camera` and gives troubleshooting-oriented messages.

The hierarchy separates three distinct failure modes:

``CameraOpenError``
    The device could not be initialised (missing, busy, or unsupported index).
``FrameReadError``
    The device is open but a frame could not be retrieved or was unusable.
``CameraStateError``
    An operation was attempted that the camera's current state does not permit.

All inherit from ``CameraError``, so a caller that does not care about the
distinction can catch just that one.
"""


class CameraError(Exception):
    """Base class for every camera-related failure."""


class CameraOpenError(CameraError):
    """Raised when a camera device cannot be opened.

    Typical causes are an invalid device index, no camera attached, or the device
    already being held by another application.
    """


class FrameReadError(CameraError):
    """Raised when a frame cannot be read from an open camera.

    Indicates the capture call failed, or returned an unusable image. A camera that
    repeatedly raises this is usually disconnected or contended by another process.
    """


class CameraStateError(CameraError):
    """Raised when an operation is invalid for the camera's current state.

    For example, reading a frame from a camera that is not open, or opening a camera
    that is already open.
    """


__all__ = [
    "CameraError",
    "CameraOpenError",
    "CameraStateError",
    "FrameReadError",
]
