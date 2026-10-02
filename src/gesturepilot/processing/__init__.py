"""Landmark processing: tracked hand in, measured geometry out.

This layer implements the **Landmark Processing** stage of the pipeline. It takes the
21-landmark :class:`~gesturepilot.tracking.types.TrackedHand` that the tracking stage
produced and returns a :class:`~gesturepilot.processing.types.HandFeatures`: the same
hand expressed in coordinates that do not depend on where it is in the frame or how large
it appears, plus the generic geometric measurements a classifier will need.

It recognises **no gestures**. A fist, a pinch, a point, a swipe — none of those words
appear in this package, and no threshold here encodes one. It also performs no confidence
filtering: rejecting unreliable classifications is its own pipeline stage, and burying
that decision in here would hide it. Filtering in the sense of *temporal smoothing* is
different, and lives here because it must happen before anything is measured.

Boundaries: this package imports neither MediaPipe nor OpenCV, and never touches the
camera. It reads only the domain types the tracking layer publishes — enforced by AST
tests in ``tests/unit/processing/test_boundaries.py``.

Coordinate conventions
----------------------

Tracker image space, then normalised hand space. In the first, ``x`` and ``y`` are
normalised across the frame and ``z`` is depth relative to the wrist. In the second,
every landmark is expressed relative to the wrist and divided by the hand scale, so one
unit means "one palm length". See ``docs/processing.md`` for the full derivation and for
what this does *not* buy you — in particular, normalised 2-D landmarks give approximate,
not exact, camera-distance invariance.

Typical use::

    from gesturepilot.processing import LandmarkProcessor, ProcessorConfig

    processor = LandmarkProcessor(ProcessorConfig(smoothing=True))

    features = processor.process(hand, timestamp=result.timestamp)
    print(features.handedness, features.hand_scale, features.finger(Finger.INDEX))

    processor.reset()   # when tracking restarts or the hands in view change

No MediaPipe, OpenCV, or NumPy type is exposed anywhere in this API.
"""

from gesturepilot.processing.errors import (
    DegenerateHandError,
    FilterStateError,
    InvalidLandmarksError,
    ProcessingError,
)
from gesturepilot.processing.filters import LandmarkSmoother, OneEuroFilter
from gesturepilot.processing.geometry import (
    add,
    angle_at,
    angle_between,
    centroid,
    distance,
    distance_2d,
    dot,
    hand_scale,
    magnitude,
    optional_angle_at,
    optional_angle_between,
    orientation_from_landmarks,
    palm_center,
    relative_to,
    subtract,
)
from gesturepilot.processing.normalization import (
    normalize_hand,
    normalize_landmarks,
    normalize_point,
)
from gesturepilot.processing.processor import LandmarkProcessor
from gesturepilot.processing.topology import (
    FINGER_CHAINS,
    FINGERS,
    HAND_SCALE_LANDMARKS,
    PALM_LANDMARKS,
    Finger,
    FingerChain,
    LandmarkIndex,
)
from gesturepilot.processing.types import (
    MAX_ANGLE_RADIANS,
    MAX_ROLL_RADIANS,
    FilterConfig,
    FingerGeometry,
    HandFeatures,
    NormalizedHand,
    NormalizedLandmark,
    PalmOrientation,
    ProcessedLandmark,
    ProcessorConfig,
)

__all__ = [
    "FINGER_CHAINS",
    "FINGERS",
    "HAND_SCALE_LANDMARKS",
    "MAX_ANGLE_RADIANS",
    "MAX_ROLL_RADIANS",
    "PALM_LANDMARKS",
    "DegenerateHandError",
    "FilterConfig",
    "FilterStateError",
    "Finger",
    "FingerChain",
    "FingerGeometry",
    "HandFeatures",
    "InvalidLandmarksError",
    "LandmarkIndex",
    "LandmarkProcessor",
    "LandmarkSmoother",
    "NormalizedHand",
    "NormalizedLandmark",
    "OneEuroFilter",
    "PalmOrientation",
    "ProcessedLandmark",
    "ProcessingError",
    "ProcessorConfig",
    "add",
    "angle_at",
    "angle_between",
    "centroid",
    "distance",
    "distance_2d",
    "dot",
    "hand_scale",
    "magnitude",
    "normalize_hand",
    "normalize_landmarks",
    "normalize_point",
    "optional_angle_at",
    "optional_angle_between",
    "orientation_from_landmarks",
    "palm_center",
    "relative_to",
    "subtract",
]
