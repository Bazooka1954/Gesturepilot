"""Landmark-processing domain types.

These are GesturePilot's own types, frozen and validated on construction. They consume
:class:`~gesturepilot.tracking.types.TrackedHand` and produce
:class:`HandFeatures`; nothing here imports MediaPipe, OpenCV, or NumPy.

Coordinate conventions
----------------------

Three coordinate spaces appear below, each with one job.

**1. Tracker image space** (``ProcessedLandmark``) — the space the tracker emits:
``x`` and ``y`` are normalised across the frame, ``z`` is depth relative to the wrist.
Identical in meaning to :class:`~gesturepilot.tracking.types.Landmark`. This is where
temporal smoothing happens, because a filter should run on the raw measurement, not on
a derived quantity.

**2. Normalised hand space** (``NormalizedLandmark``) — dimensionless. Each landmark is
expressed relative to the wrist and divided by the hand scale, so one unit means "one
palm length". Where the hand sits in frame and how large it appears both cancel out.
Every derived measurement — angles, segment lengths, tip distances — is computed here.

**3. Scalar measurements** — plain floats: lengths in hand-scale units, angles in
radians, and the hand scale itself in tracker image units.

World landmarks
---------------

The tracker does **not** expose world landmarks: :class:`~gesturepilot.tracking.types.TrackedHand`
carries image-space landmarks only. There is therefore no metric 3-D reconstruction
here, and no claim of true camera-distance invariance — see ``docs/processing.md``.

An angle that cannot be formed because two landmarks coincide is reported as ``None``,
never ``NaN``. A folded finger is a normal pose, and a downstream stage must be able to
tell "this joint has no defined direction" apart from "this joint is straight".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gesturepilot.processing.topology import FINGERS, Finger
from gesturepilot.tracking.types import LANDMARK_COUNT, Handedness

#: Every joint angle this package reports lies in ``[0, pi]``.
MAX_ANGLE_RADIANS = math.pi

#: The bound on :attr:`PalmOrientation.roll`. Roll measures an *axis*, so it is folded
#: into a half-turn rather than a full turn; see that class for why.
MAX_ROLL_RADIANS = math.pi / 2.0


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class ProcessedLandmark:
    """One landmark in tracker image space, after processing.

    Same convention as :class:`~gesturepilot.tracking.types.Landmark`: ``x`` and ``y``
    are normalised image positions, ``z`` is depth relative to the wrist.

    This is a *new* object, not the tracker's. With smoothing enabled the values are
    filtered; with it disabled they are a faithful copy. Either way the source
    ``TrackedHand`` is left untouched.

    ``x`` and ``y`` are not range-checked: a landmark at a frame edge legitimately sits
    slightly outside ``[0.0, 1.0]``. All three are checked for finiteness, because one
    ``NaN`` would silently poison every measurement derived from it.
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            _require_finite(f"ProcessedLandmark.{name}", value)

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """``(x, y, z)`` for the pure-math helpers in :mod:`geometry`."""
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class NormalizedLandmark:
    """One landmark in normalised hand space.

    Wrist-relative, divided by the hand scale, dimensionless. ``(0.0, 0.0, 0.0)`` is
    the wrist itself, and a value of ``1.0`` means "one palm length away".

    The wrist is exactly zero by construction. The division of ``z`` by an image-plane
    scale is an *approximation*: the backend's depth axis is only roughly commensurate
    with its image axes. It is done anyway, so that all three components share one unit
    and no consumer has to special-case depth.
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            _require_finite(f"NormalizedLandmark.{name}", value)

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """``(x, y, z)`` for the pure-math helpers in :mod:`geometry`."""
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class NormalizedHand:
    """A whole hand expressed in normalised hand space.

    Attributes:
        origin: The wrist, in tracker image space. Kept so a consumer can convert a
            normalised point back to the frame if it needs to; the processor itself
            never does, because nothing downstream of it knows the frame size.
        scale: The divisor that was applied — the wrist-to-middle-knuckle distance in
            tracker image units. Also the raw measurement of how large the hand
            appeared this frame, which is useful in its own right.
        landmarks: All landmarks in normalised hand space, in the original topology
            order.
        palm_center: The palm centroid, also in normalised hand space.
    """

    origin: ProcessedLandmark
    scale: float
    landmarks: tuple[NormalizedLandmark, ...]
    palm_center: NormalizedLandmark

    def __post_init__(self) -> None:
        _require_finite("NormalizedHand.scale", self.scale)
        if self.scale <= 0.0:
            raise ValueError(f"NormalizedHand.scale must be > 0, got {self.scale}")
        if len(self.landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"NormalizedHand requires exactly {LANDMARK_COUNT} landmarks, "
                f"got {len(self.landmarks)}"
            )

    def landmark(self, index: int) -> NormalizedLandmark:
        """Return the normalised landmark at ``index``.

        Raises:
            IndexError: If ``index`` is outside the landmark array.
        """
        return self.landmarks[index]


@dataclass(frozen=True)
class FingerGeometry:
    """Geometric measurements for one finger.

    All lengths are in normalised hand units, so they are independent of where the hand
    is in the frame and how large it appears. Angles are measured **in the image plane**
    only: the backend's depth axis is too noisy for a meaningful 3-D joint angle, and a
    reliable planar angle beats an unreliable spatial one.

    Attributes:
        finger: Which digit these measurements belong to.
        proximal_angle: Angle at the base joint (knuckle, or the thumb's carpometacarpal
            joint), between the segment arriving from the wrist and the segment leaving
            toward the middle joint.
        middle_angle: Angle at the middle joint.
        distal_angle: Angle at the joint nearest the fingertip.
        segment_lengths: The four bone lengths in order —
            ``(root->base, base->middle, middle->distal, distal->tip)``.
        tip_to_palm: Distance from the fingertip to the palm centre, in normalised hand
            units.
        tip_to_index_mcp: Distance from the fingertip to the index-finger knuckle.

            This is the lateral reference used **instead of** a handedness-based
            thumb/index side test. The index knuckle is where it is because of the
            geometry, so the measurement means the same thing for a left hand and a
            right hand without anyone having to decide whether the camera is mirrored.
    """

    finger: Finger
    proximal_angle: float | None
    middle_angle: float | None
    distal_angle: float | None
    segment_lengths: tuple[float, float, float, float]
    tip_to_palm: float
    tip_to_index_mcp: float

    def __post_init__(self) -> None:
        for name in ("proximal_angle", "middle_angle", "distal_angle"):
            value = getattr(self, name)
            if value is None:
                continue
            _require_finite(f"FingerGeometry.{name}", value)
            if not 0.0 <= value <= MAX_ANGLE_RADIANS:
                raise ValueError(f"FingerGeometry.{name} must be within [0.0, pi], got {value}")
        if len(self.segment_lengths) != 4:
            raise ValueError(
                f"FingerGeometry.segment_lengths needs exactly 4 entries, "
                f"got {len(self.segment_lengths)}"
            )
        for index, length in enumerate(self.segment_lengths):
            _require_finite(f"FingerGeometry.segment_lengths[{index}]", length)
            if length < 0.0:
                raise ValueError(
                    f"FingerGeometry.segment_lengths[{index}] must be >= 0, got {length}"
                )
        for name in ("tip_to_palm", "tip_to_index_mcp"):
            value = getattr(self, name)
            _require_finite(f"FingerGeometry.{name}", value)
            if value < 0.0:
                raise ValueError(f"FingerGeometry.{name} must be >= 0, got {value}")

    @property
    def is_fully_measured(self) -> bool:
        """Whether every joint angle was well defined.

        ``False`` when at least one joint collapsed, so a consumer can ask "is this
        measurement complete?" instead of testing three fields for ``None``.
        """
        return None not in (self.proximal_angle, self.middle_angle, self.distal_angle)


@dataclass(frozen=True)
class PalmOrientation:
    """Image-plane orientation of the palm.

    Derived from two geometry-chosen reference directions, so it is well defined for a
    left hand and a right hand alike. Both components are ``None`` when the direction
    they are measured from collapsed.

    Attributes:
        roll: Direction of the across-palm axis, index knuckle minus pinky knuckle,
            folded into ``[-pi/2, pi/2]`` because an axis has no inherent sign — which end
            is the index side is precisely the question handedness would answer. ``0.0``
            means the palm's knuckle line runs across the frame; the sign says which way
            it leans. Mirroring a hand negates ``roll`` exactly.
        tilt: Direction of the along-palm axis, wrist to middle knuckle, as an angle in
            ``[-pi, pi]``. This direction *is* oriented, so the full range is kept.
            ``0.0`` points up the frame (the image ``-y`` direction), positive toward
            ``+x``.
    """

    roll: float | None
    tilt: float | None

    def __post_init__(self) -> None:
        if self.roll is not None:
            _require_finite("PalmOrientation.roll", self.roll)
            if not -MAX_ROLL_RADIANS <= self.roll <= MAX_ROLL_RADIANS:
                raise ValueError(
                    f"PalmOrientation.roll must be within [-pi/2, pi/2], got {self.roll}"
                )
        if self.tilt is not None:
            _require_finite("PalmOrientation.tilt", self.tilt)
            if not -MAX_ANGLE_RADIANS <= self.tilt <= MAX_ANGLE_RADIANS:
                raise ValueError(f"PalmOrientation.tilt must be within [-pi, pi], got {self.tilt}")

    @property
    def is_fully_measured(self) -> bool:
        """Whether both orientation components were well defined."""
        return self.roll is not None and self.tilt is not None


@dataclass(frozen=True)
class FilterConfig:
    """Parameters for the One Euro filter used by the smoothing stage.

    Attributes:
        min_cutoff: Cutoff frequency in Hz used when the hand is still. Lower is
            smoother but laggier. Must be ``> 0``.
        beta: How aggressively the cutoff is raised in response to speed. ``0.0``
            gives a fixed cutoff — heavily smoothed, heavy lag. Larger values track fast
            motion more closely. Must be ``>= 0``.
        derivative_cutoff: Cutoff frequency in Hz for the low-pass applied to the
            estimated derivative that drives the adaptive cutoff. Must be ``> 0``.

    These are the algorithm's three published knobs; the defaults are the reference
    implementation's starting point, not a tuned GesturePilot profile.
    """

    min_cutoff: float = 1.0
    beta: float = 0.0
    derivative_cutoff: float = 1.0

    def __post_init__(self) -> None:
        _require_finite("FilterConfig.min_cutoff", self.min_cutoff)
        _require_finite("FilterConfig.beta", self.beta)
        _require_finite("FilterConfig.derivative_cutoff", self.derivative_cutoff)
        if self.min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff must be > 0, got {self.min_cutoff}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.derivative_cutoff <= 0.0:
            raise ValueError(f"derivative_cutoff must be > 0, got {self.derivative_cutoff}")


@dataclass(frozen=True)
class ProcessorConfig:
    """Settings for :class:`~gesturepilot.processing.processor.LandmarkProcessor`.

    Attributes:
        smoothing: Whether to run the One Euro filter over the raw landmarks before
            measuring anything. ``False`` by default, which makes
            :meth:`~gesturepilot.processing.processor.LandmarkProcessor.process` a pure
            function of its arguments and requires no timestamp. ``True`` makes it
            stateful per hand and requires the caller to supply one.
        filter: Filter parameters, used only when ``smoothing`` is ``True``.
    """

    smoothing: bool = False
    filter: FilterConfig = FilterConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.filter, FilterConfig):
            raise TypeError(f"filter must be a FilterConfig, got {type(self.filter).__name__}")


@dataclass(frozen=True)
class HandFeatures:
    """Everything the processor derived about one hand, this frame.

    The single output type of the landmark-processing stage. It is deliberately
    descriptive: it reports geometry and source metadata and makes **no** gesture
    judgement. Nothing here says a hand is open, closed, pointing, or pinching. That
    belongs to the classifier, which is a later phase.

    Attributes:
        landmarks: All landmarks in tracker image space after processing — filtered if
            smoothing is enabled, a faithful copy of the input otherwise. In the
            original topology order. New objects; the source ``TrackedHand`` is never
            modified.
        normalization: The wrist-relative, scale-normalised view these features were
            measured in. Carries the origin and the hand scale as well as the
            normalised landmarks.
        fingers: One :class:`FingerGeometry` per finger, in :data:`FINGERS` order.
        orientation: Image-plane palm orientation.

        handedness: The tracker's label for this hand, passed through unchanged. This
            is source metadata, not a geometric input: nothing below is computed from
            it. See ``docs/processing.md`` for why handedness is not used to flip or
            mirror geometry.
        handedness_score: The tracker's confidence in that label, or ``None``.
        detection_score: The tracker's per-hand detection confidence, or ``None``.
            The MediaPipe Tasks backend does not emit one, so it is normally ``None``.
        timestamp: Monotonic seconds for the originating frame, or ``None`` when the
            caller did not supply one. Pass-through only; never re-derived.
        frame_sequence: The originating frame's sequence number, or ``None``.
    """

    landmarks: tuple[ProcessedLandmark, ...]
    normalization: NormalizedHand
    fingers: tuple[FingerGeometry, ...]
    orientation: PalmOrientation
    handedness: Handedness = Handedness.UNKNOWN
    handedness_score: float | None = None
    detection_score: float | None = None
    timestamp: float | None = None
    frame_sequence: int | None = None

    def __post_init__(self) -> None:
        if len(self.landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"HandFeatures requires exactly {LANDMARK_COUNT} landmarks, "
                f"got {len(self.landmarks)}"
            )
        if len(self.fingers) != len(FINGERS):
            raise ValueError(
                f"HandFeatures requires exactly {len(FINGERS)} finger measurements, "
                f"got {len(self.fingers)}"
            )
        if {finger.finger for finger in self.fingers} != set(FINGERS):
            raise ValueError("HandFeatures.fingers must cover every Finger exactly once")
        for name in ("handedness_score", "detection_score"):
            score = getattr(self, name)
            if score is None:
                continue
            _require_finite(f"HandFeatures.{name}", score)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"HandFeatures.{name} must be within [0.0, 1.0], got {score}")
        if self.timestamp is not None:
            _require_finite("HandFeatures.timestamp", self.timestamp)
        if self.frame_sequence is not None and self.frame_sequence < 0:
            raise ValueError(f"HandFeatures.frame_sequence must be >= 0, got {self.frame_sequence}")

    # -- convenience accessors -------------------------------------------

    @property
    def normalized_landmarks(self) -> tuple[NormalizedLandmark, ...]:
        """The normalised landmarks, in topology order."""
        return self.normalization.landmarks

    @property
    def palm_center(self) -> NormalizedLandmark:
        """The palm centre in normalised hand space."""
        return self.normalization.palm_center

    @property
    def hand_scale(self) -> float:
        """The wrist-to-middle-knuckle distance in tracker image units.

        Zero can never occur — a collapsed hand raises
        :class:`~gesturepilot.processing.errors.DegenerateHandError` instead. This is
        also the raw "how big did the hand look" signal, kept alongside the
        position-independent features.
        """
        return self.normalization.scale

    @property
    def wrist(self) -> ProcessedLandmark:
        """The wrist in tracker image space."""
        return self.landmarks[0]

    @property
    def wrist_normalized(self) -> NormalizedLandmark:
        """The wrist in normalised hand space. Always exactly the origin."""
        return self.normalization.landmarks[0]

    def landmark(self, index: int) -> ProcessedLandmark:
        """Return the processed landmark at ``index``.

        Raises:
            IndexError: If ``index`` is outside the landmark array.
        """
        return self.landmarks[index]

    def normalized(self, index: int) -> NormalizedLandmark:
        """Return the normalised landmark at ``index``.

        Raises:
            IndexError: If ``index`` is outside the landmark array.
        """
        return self.normalization.landmarks[index]

    def finger(self, finger: Finger) -> FingerGeometry:
        """Return the measurements for ``finger``.

        Raises:
            KeyError: If ``finger`` is not a member of :class:`Finger`.
        """
        for measured in self.fingers:
            if measured.finger is finger:
                return measured
        raise KeyError(f"No geometry recorded for {finger}")


__all__ = [
    "MAX_ANGLE_RADIANS",
    "MAX_ROLL_RADIANS",
    "FilterConfig",
    "FingerGeometry",
    "HandFeatures",
    "NormalizedHand",
    "NormalizedLandmark",
    "PalmOrientation",
    "ProcessedLandmark",
    "ProcessorConfig",
]
