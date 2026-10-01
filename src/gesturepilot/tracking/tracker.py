"""Hand-tracker abstraction and the MediaPipe-backed implementation.

This is the **only** module in GesturePilot that imports MediaPipe. Everything above
it consumes :class:`~gesturepilot.tracking.types.TrackingResult`, which is built from
plain domain objects. No MediaPipe type crosses this boundary — enforced by tests in
``tests/unit/tracking/test_boundaries.py``.

Design decisions
----------------

**VIDEO running mode by default.** Webcam frames arrive as a sequential time series,
so the tracker is given that temporal context: the model carries tracking state
between frames, which yields more stable, temporally coherent landmarks than
re-detecting each frame in isolation. ``RunningMode.IMAGE`` is available for
deterministic, stateless inspection. ``LIVE_STREAM`` is deliberately not supported —
it requires asynchronous callbacks, which this phase avoids.

**Timestamps come from the camera, never from wall-clock time.** ``VIDEO`` mode needs a
strictly increasing integer millisecond timestamp. The camera layer already provides a
monotonic per-frame timestamp, so it is converted rather than re-measured. When two
frames land inside the same millisecond, the timestamp is nudged forward by 1 ms to
preserve strict monotonicity, since MediaPipe rejects a repeated or decreasing stamp.

**BGR to RGB happens here.** The camera emits OpenCV BGR arrays and must stay unaware
of MediaPipe. ``cv.cvtColor`` allocates a new array, so the caller's frame is never
modified.

**Injection for testability.** Both the landmarker factory and the frame source are
injectable, so unit tests exercise the whole conversion path without loading the model
or touching a webcam.

Privacy: frames are converted and handed to the local inference engine. Nothing is
saved, logged, or transmitted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from gesturepilot.camera.types import Frame
from gesturepilot.tracking.assets import resolve_model_path
from gesturepilot.tracking.errors import (
    TrackerInitializationError,
    TrackerStateError,
    TrackingError,
    TrackingProcessingError,
)
from gesturepilot.tracking.types import (
    Handedness,
    Landmark,
    RunningMode,
    TrackedHand,
    TrackerConfig,
    TrackingResult,
)

#: Builds a landmarker from fully-built options. Override in tests.
LandmarkerFactory = Callable[[HandLandmarkerOptions], Any]

#: Maps MediaPipe's handedness labels onto our domain enum. Anything unrecognised
#: becomes ``Handedness.UNKNOWN`` rather than being guessed at.
_HANDEDNESS_BY_LABEL: dict[str, Handedness] = {
    "Left": Handedness.LEFT,
    "Right": Handedness.RIGHT,
}

_RUNNING_MODE_MAP: dict[RunningMode, VisionTaskRunningMode] = {
    RunningMode.IMAGE: VisionTaskRunningMode.IMAGE,
    RunningMode.VIDEO: VisionTaskRunningMode.VIDEO,
}


@runtime_checkable
class HandTracker(Protocol):
    """Abstract hand-tracking interface.

    Lifecycle is ``CREATED -> INITIALIZED -> CLOSED``. Implementations must release
    the underlying model on close rather than relying on process termination.
    """

    @property
    def config(self) -> TrackerConfig:
        """The configuration this tracker was built with."""

    @property
    def is_initialized(self) -> bool:
        """Whether the tracker is ready to process frames."""

    def initialize(self) -> None:
        """Load the model asset and prepare for processing.

        Raises:
            ModelAssetError: If the model asset is missing.
            TrackerInitializationError: If the model could not be loaded.
            TrackerStateError: If already initialised.
        """

    def process(self, frame: Frame) -> TrackingResult:
        """Track hands in one frame.

        Args:
            frame: A frame from the camera layer. Not modified.

        Returns:
            A :class:`TrackingResult`. Zero hands is a valid outcome, not an error.

        Raises:
            TrackerStateError: If not initialised, or already closed.
            TrackingProcessingError: If the frame is unusable or inference failed.
        """

    def close(self) -> None:
        """Release the model. Safe to call more than once."""


class MediaPipeHandTracker:
    """Hand tracking backed by the MediaPipe Tasks ``HandLandmarker``.

    Args:
        config: Tracker settings, including optional explicit model path.
        landmarker_factory: Builds the landmarker from options. Defaults to
            ``HandLandmarker.create_from_options``. Override in tests to avoid
            loading the real model.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        landmarker_factory: LandmarkerFactory | None = None,
    ) -> None:
        self._config = config or TrackerConfig()
        self._landmarker_factory = landmarker_factory or _default_landmarker_factory
        self._landmarker: Any | None = None
        self._last_timestamp_ms: int | None = None

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> TrackerConfig:
        return self._config

    @property
    def is_initialized(self) -> bool:
        return self._landmarker is not None

    # -- lifecycle --------------------------------------------------------

    def initialize(self) -> None:
        if self._landmarker is not None:
            raise TrackerStateError(
                "Tracker is already initialized. Call close() before initializing again."
            )

        model_path = self._resolve_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RUNNING_MODE_MAP[self._config.running_mode],
            num_hands=self._config.max_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_hand_presence_confidence=self._config.min_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )

        try:
            self._landmarker = self._landmarker_factory(options)
        except Exception as exc:
            raise TrackerInitializationError(
                f"Failed to initialise the MediaPipe hand landmarker using model "
                f"'{model_path}'. The asset may be corrupt or incompatible with this "
                f"MediaPipe version ({mp.__version__})."
            ) from exc

        if self._landmarker is None:
            raise TrackerInitializationError(
                f"Landmarker factory returned None for model '{model_path}'."
            )

        self._last_timestamp_ms = None

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            close = getattr(landmarker, "close", None)
            if callable(close):
                close()
        self._last_timestamp_ms = None

    def __enter__(self) -> MediaPipeHandTracker:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- processing -------------------------------------------------------

    def process(self, frame: Frame) -> TrackingResult:
        landmarker = self._require_initialized()
        image = self._validate_frame(frame)

        # cvtColor allocates a new array; the caller's frame is left untouched.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            if self._config.running_mode is RunningMode.VIDEO:
                raw = landmarker.detect_for_video(
                    mp_image, self._next_timestamp_ms(frame.timestamp)
                )
            else:
                raw = landmarker.detect(mp_image)
        except TrackingError:
            # Already one of ours (e.g. a backend reusing the hierarchy); don't re-wrap.
            raise
        except Exception as exc:
            raise TrackingProcessingError(
                f"Hand tracking failed while processing frame {frame.sequence}. "
                f"The model backend raised: {exc}"
            ) from exc

        return self._to_result(raw, frame)

    # -- internals --------------------------------------------------------

    def _require_initialized(self) -> Any:
        if self._landmarker is None:
            raise TrackerStateError(
                "Tracker is not initialized. Call initialize() (or use the tracker as "
                "a context manager) before processing frames."
            )
        return self._landmarker

    def _resolve_model(self) -> Path:
        """Locate the model asset. ``ModelAssetError`` propagates unchanged."""
        return resolve_model_path(self._config.model_path)

    def _validate_frame(self, frame: Frame) -> np.ndarray:
        if not isinstance(frame, Frame):
            raise TrackingProcessingError(f"Expected a camera Frame, got {type(frame).__name__}.")

        image = frame.image
        if not isinstance(image, np.ndarray):
            raise TrackingProcessingError(f"Frame image must be a NumPy array, got {type(image)}.")
        if image.size == 0:
            raise TrackingProcessingError(f"Frame {frame.sequence} is empty.")
        if image.ndim != 3:
            raise TrackingProcessingError(
                f"Frame {frame.sequence} must be a colour (3D) array, got shape {image.shape}."
            )
        if image.dtype != np.uint8:
            raise TrackingProcessingError(
                f"Frame {frame.sequence} must be uint8, got {image.dtype}."
            )
        if not math.isfinite(frame.timestamp):
            raise TrackingProcessingError(
                f"Frame {frame.sequence} timestamp must be finite, got {frame.timestamp}."
            )
        return image

    def _next_timestamp_ms(self, timestamp_seconds: float) -> int:
        """Convert a monotonic second-based frame stamp to MediaPipe milliseconds.

        MediaPipe VIDEO mode requires strictly increasing integer timestamps. Two
        consecutive frames can land in the same millisecond at high frame rates, so the
        value is nudged forward to keep it strictly increasing.
        """
        milliseconds = int(timestamp_seconds * 1000)
        if self._last_timestamp_ms is not None and milliseconds <= self._last_timestamp_ms:
            milliseconds = self._last_timestamp_ms + 1
        self._last_timestamp_ms = milliseconds
        return milliseconds

    def _to_result(self, raw: Any, frame: Frame) -> TrackingResult:
        raw_landmarks: Any = getattr(raw, "hand_landmarks", None) or []
        raw_handedness: Any = getattr(raw, "handedness", None) or []

        hands: list[TrackedHand] = []
        for index, hand_landmarks in enumerate(raw_landmarks):
            hands.append(self._to_hand(hand_landmarks, raw_handedness, index))

        # Order is preserved exactly as the tracker produced it; no normalisation.
        return TrackingResult(
            hands=tuple(hands),
            timestamp=frame.timestamp,
            frame_sequence=frame.sequence,
        )

    def _to_hand(
        self,
        hand_landmarks: Any,
        raw_handedness: Any,
        index: int,
    ) -> TrackedHand:
        handedness = Handedness.UNKNOWN
        handedness_score: float | None = None

        if index < len(raw_handedness) and raw_handedness[index]:
            category = raw_handedness[index][0]
            label = getattr(category, "category_name", None)
            if isinstance(label, str):
                handedness = _HANDEDNESS_BY_LABEL.get(label, Handedness.UNKNOWN)
            score = getattr(category, "score", None)
            if score is not None:
                handedness_score = float(score)

        try:
            landmarks = tuple(
                Landmark(x=float(point.x), y=float(point.y), z=float(point.z))
                for point in hand_landmarks
            )
            return TrackedHand(
                landmarks=landmarks,
                handedness=handedness,
                handedness_score=handedness_score,
                # MediaPipe's Tasks API emits no per-frame detection confidence.
                detection_score=None,
            )
        except (ValueError, AttributeError, TypeError) as exc:
            raise TrackingProcessingError(
                f"Tracker backend returned an unusable landmark set for hand {index}: {exc}"
            ) from exc


def _default_landmarker_factory(options: HandLandmarkerOptions) -> Any:
    return HandLandmarker.create_from_options(options)
