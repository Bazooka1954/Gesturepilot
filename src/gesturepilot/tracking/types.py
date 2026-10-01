"""Hand-tracking domain types.

These are GesturePilot's own types. They carry no MediaPipe objects and know nothing
about MediaPipe — including its category-label vocabulary, which the tracking module
owns. That keeps every layer above this one backend-agnostic.

Coordinates stay **normalised**. Nothing here converts to pixels: the mapping from
normalised space to screen space belongs to the landmark-processing phase, which
knows the frame dimensions.

Privacy: this module handles numbers only. It has no file, network, or logging access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

#: Number of landmarks the MediaPipe hand-landmarker model emits per hand.
LANDMARK_COUNT = 21


class Handedness(Enum):
    """Which hand a detection belongs to.

    ``UNKNOWN`` is a first-class state, not an error. MediaPipe can return an
    unrecognised label, and downstream stages must handle it without guessing.
    """

    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


class RunningMode(Enum):
    """How the tracker consumes frames.

    ``IMAGE``
        Each frame is processed independently with no temporal state. Simplest and
        fully deterministic, but the model re-detects from scratch every frame and
        results can jitter between adjacent frames.

    ``VIDEO``
        Frames are processed as a time series. The tracker carries state between
        frames, which gives more stable, temporally coherent landmarks. Requires
        strictly increasing millisecond timestamps.

    ``LIVE_STREAM`` is intentionally absent: it introduces asynchronous callbacks,
    which this phase deliberately avoids.
    """

    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class Landmark:
    """A single hand landmark in normalised image coordinates.

    Attributes:
        x: Horizontal position, nominally ``0.0``–``1.0`` across the frame.
        y: Vertical position, nominally ``0.0``–``1.0`` down the frame.
        z: Depth relative to the wrist, roughly centred on ``0.0`` and negative
            toward the camera. The scale is relative, not metric.

    x and y can sit marginally outside ``[0.0, 1.0]`` for landmarks at a frame edge,
    so they are not range-checked. Every value is checked for finiteness, because a
    NaN would silently corrupt every downstream calculation.
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not math.isfinite(value):
                raise ValueError(f"Landmark.{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class TrackedHand:
    """One detected hand and everything known about it this frame.

    Attributes:
        landmarks: Exactly ``LANDMARK_COUNT`` landmarks, in the model's own order.
            Ordered and immutable; do not reorder per-hand indexing later.
        handedness: Which hand, or ``UNKNOWN``.
        handedness_score: Model confidence in the handedness label, or ``None``.
        detection_score: Per-hand detection confidence, or ``None``.

            The MediaPipe hand-landmarker does **not** emit a per-frame detection or
            tracking confidence; its detection/presence/tracking confidences are
            thresholds applied at initialisation and live in :class:`TrackerConfig`.
            This field exists so that can be filled in later without an API break.
    """

    landmarks: tuple[Landmark, ...]
    handedness: Handedness = Handedness.UNKNOWN
    handedness_score: float | None = None
    detection_score: float | None = None

    def __post_init__(self) -> None:
        if len(self.landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"TrackedHand requires exactly {LANDMARK_COUNT} landmarks, "
                f"got {len(self.landmarks)}"
            )

    @property
    def landmark_count(self) -> int:
        """Number of landmarks on this hand."""
        return len(self.landmarks)

    def landmark(self, index: int) -> Landmark:
        """Return the landmark at ``index``.

        Raises:
            IndexError: If ``index`` is outside the landmark array.
        """
        return self.landmarks[index]


@dataclass(frozen=True)
class TrackingResult:
    """The outcome of tracking one frame.

    **Having no hands is a valid result, not an error.** ``hands`` is simply empty.

    Attributes:
        hands: Detected hands, in the order the tracker produced them. Empty when no
            hand is present. Order is preserved, not normalised — associating hands
            with identities is a later phase's job.
        timestamp: The originating frame's monotonic timestamp, passed through
            unchanged. Not re-derived from wall-clock time.
        frame_sequence: The originating frame's sequence number, passed through
            unchanged. Lets downstream stages detect dropped or reordered frames.
    """

    hands: tuple[TrackedHand, ...]
    timestamp: float
    frame_sequence: int

    @property
    def hand_count(self) -> int:
        """Number of hands detected."""
        return len(self.hands)

    @property
    def has_hands(self) -> bool:
        """Whether at least one hand was detected."""
        return bool(self.hands)

    @property
    def primary_hand(self) -> TrackedHand | None:
        """The first detected hand, or ``None`` when no hand is present.

        "Primary" means first in tracker order, nothing more. Choosing a dominant
        hand is a later phase's responsibility.
        """
        return self.hands[0] if self.hands else None


@dataclass(frozen=True)
class TrackerConfig:
    """Hand-tracker settings, validated on construction.

    Attributes:
        max_hands: Maximum hands to detect. More than one is supported.
        min_detection_confidence: Threshold for declaring a hand present.
        min_presence_confidence: Threshold for a hand to remain in view.
        min_tracking_confidence: Threshold for tracking an existing hand.
        model_path: Explicit path to the hand-landmarker asset. When ``None``, the
            asset is discovered via :func:`gesturepilot.tracking.assets.resolve_model_path`.
        running_mode: Sequential (IMAGE) or time-series (VIDEO) processing.
    """

    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: str | None = None
    running_mode: RunningMode = RunningMode.VIDEO

    def __post_init__(self) -> None:
        if not 1 <= self.max_hands <= 10:
            raise ValueError(f"max_hands must be between 1 and 10, got {self.max_hands}")
        for name in (
            "min_detection_confidence",
            "min_presence_confidence",
            "min_tracking_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within 0.0-1.0, got {value}")

    @property
    def model_path_str(self) -> str | None:
        """``model_path`` as a string, for APIs that expect one."""
        return str(self.model_path) if self.model_path is not None else None
