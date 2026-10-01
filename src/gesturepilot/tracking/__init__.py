"""Hand-tracking layer: frame in, structured tracking result out.

This layer answers one question — *where are the hands in this frame?* It does not
interpret them. Gesture classification, confidence filtering, and smoothing all belong
to later phases.

Typical use::

    from gesturepilot.camera import CameraConfig, WebcamCamera
    from gesturepilot.tracking import TrackerConfig, MediaPipeHandTracker

    with WebcamCamera(CameraConfig()) as camera, MediaPipeHandTracker(TrackerConfig()) as tracker:
        while True:
            frame = camera.read()
            result = tracker.process(frame)
            if result.has_hands:
                hand = result.primary_hand

No MediaPipe type is exposed anywhere in this API.
"""

from gesturepilot.tracking.assets import (
    DEFAULT_MODEL_FILENAME,
    MODEL_PATH_ENV_VAR,
    resolve_model_path,
)
from gesturepilot.tracking.errors import (
    ModelAssetError,
    TrackerInitializationError,
    TrackerStateError,
    TrackingError,
    TrackingProcessingError,
)
from gesturepilot.tracking.tracker import HandTracker, MediaPipeHandTracker
from gesturepilot.tracking.types import (
    LANDMARK_COUNT,
    Handedness,
    Landmark,
    RunningMode,
    TrackedHand,
    TrackerConfig,
    TrackingResult,
)

__all__ = [
    "DEFAULT_MODEL_FILENAME",
    "LANDMARK_COUNT",
    "MODEL_PATH_ENV_VAR",
    "HandTracker",
    "Handedness",
    "Landmark",
    "MediaPipeHandTracker",
    "ModelAssetError",
    "RunningMode",
    "TrackerConfig",
    "TrackerInitializationError",
    "TrackerStateError",
    "TrackingError",
    "TrackingProcessingError",
    "TrackingResult",
    "TrackedHand",
    "resolve_model_path",
]
