"""Camera layer: device abstraction, lifecycle, and frame capture.

This is the only GesturePilot layer that knows a capture backend exists. Everything
above it consumes :class:`~gesturepilot.camera.types.Frame` objects and calls
:class:`~gesturepilot.camera.camera.Camera`, so the gesture engine stays independent of
OpenCV.

Typical use::

    from gesturepilot.camera import CameraConfig, open_camera

    with open_camera(CameraConfig(device_index=0, width=1280, height=720)) as camera:
        frame = camera.read()
        print(frame.width, frame.height, camera.measured_fps)
"""

from gesturepilot.camera.camera import Camera, WebcamCamera, open_camera
from gesturepilot.camera.errors import (
    CameraError,
    CameraOpenError,
    CameraStateError,
    FrameReadError,
)
from gesturepilot.camera.types import CameraConfig, CameraProperties, FpsMeter, Frame

__all__ = [
    "Camera",
    "CameraConfig",
    "CameraError",
    "CameraOpenError",
    "CameraProperties",
    "CameraStateError",
    "Frame",
    "FrameReadError",
    "FpsMeter",
    "WebcamCamera",
    "open_camera",
]
