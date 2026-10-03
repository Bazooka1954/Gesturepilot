"""Posable synthetic hands for gesture-classification unit tests.

Every pose here is built by forward kinematics from a palm skeleton plus one number per
finger, so the geometry that reaches the classifier is known rather than guessed, and a
failure points at a specific joint bend instead of a mysterious literal. Nothing is
random, nothing is snapshotted, and no test needs a webcam or a model asset.

Poses are produced by running 21 hand-placed coordinates through the *real*
:class:`~gesturepilot.processing.processor.LandmarkProcessor`. That matters: the
classifier is only ever going to see genuine processing output, so tests that hand-built
a :class:`~gesturepilot.processing.types.HandFeatures` would be testing against a shape
the pipeline never produces. Building it for real also means the extension ratios these
tests depend on are computed by processing's own normalisation, and a change there shows
up here as a test failure rather than as a silent behavioural drift.

Curl convention
---------------
Each finger takes a single ``curl`` in degrees: the total bend across its two
interphalangeal joints, split evenly between them. ``0`` is straight, and larger values
fold the finger back towards the palm. The bend is applied at the two joints *after* the
knuckle rather than at the knuckle itself, because rotating a whole finger about its
knuckle leaves the extension ratio unchanged — only folding it changes how straight it
is. That makes ``curl`` a clean, monotonic knob for the classifier's threshold band, and
it is why these poses vary the curl without also varying apparent finger length.

Finger bone lengths, the palm skeleton, and the wrist anchor are fixed, so the hand
scale is ``hypot(0.005, 0.170)`` for every pose here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from gesturepilot.processing.processor import LandmarkProcessor
from gesturepilot.processing.topology import FINGERS, Finger, LandmarkIndex
from gesturepilot.processing.types import FingerGeometry, HandFeatures
from gesturepilot.tracking.types import Handedness, Landmark, TrackedHand

#: The wrist, in normalised image coordinates with ``y`` increasing downward.
WRIST: tuple[float, float] = (0.500, 0.800)

#: The wrist-to-middle-knuckle span, i.e. the hand scale of every pose in this module.
HAND_SCALE: float = math.hypot(0.505 - 0.500, 0.630 - 0.800)

#: Knuckle positions for the four non-thumb fingers.
KNUCKLES: Mapping[Finger, tuple[float, float]] = {
    Finger.INDEX: (0.450, 0.640),
    Finger.MIDDLE: (0.505, 0.630),
    Finger.RING: (0.560, 0.640),
    Finger.PINKY: (0.605, 0.662),
}

#: The three bones of each non-thumb finger, ``(knuckle->pip, pip->dip, dip->tip)``.
BONES: Mapping[Finger, tuple[float, float, float]] = {
    Finger.INDEX: (0.120, 0.066, 0.054),
    Finger.MIDDLE: (0.130, 0.072, 0.055),
    Finger.RING: (0.122, 0.068, 0.052),
    Finger.PINKY: (0.098, 0.056, 0.045),
}

#: Base direction of each non-thumb finger in degrees, where ``-90`` points up the frame.
SPREADS: Mapping[Finger, float] = {
    Finger.INDEX: -96.0,
    Finger.MIDDLE: -90.0,
    Finger.RING: -84.0,
    Finger.PINKY: -78.0,
}

#: ``(base, middle, distal, tip)`` landmark indices per non-thumb finger.
_JOINTS: Mapping[Finger, tuple[LandmarkIndex, ...]] = {
    Finger.INDEX: (
        LandmarkIndex.INDEX_MCP,
        LandmarkIndex.INDEX_PIP,
        LandmarkIndex.INDEX_DIP,
        LandmarkIndex.INDEX_TIP,
    ),
    Finger.MIDDLE: (
        LandmarkIndex.MIDDLE_MCP,
        LandmarkIndex.MIDDLE_PIP,
        LandmarkIndex.MIDDLE_DIP,
        LandmarkIndex.MIDDLE_TIP,
    ),
    Finger.RING: (
        LandmarkIndex.RING_MCP,
        LandmarkIndex.RING_PIP,
        LandmarkIndex.RING_DIP,
        LandmarkIndex.RING_TIP,
    ),
    Finger.PINKY: (
        LandmarkIndex.PINKY_MCP,
        LandmarkIndex.PINKY_PIP,
        LandmarkIndex.PINKY_DIP,
        LandmarkIndex.PINKY_TIP,
    ),
}

#: The thumb's bones when spread out, as ``(cmc, mcp, ip, tip)``. Collinear enough for
#: the extension ratio to read as fully extended.
THUMB_SPREAD: tuple[tuple[float, float], ...] = (
    (0.415, 0.735),
    (0.360, 0.690),
    (0.320, 0.650),
    (0.290, 0.610),
)

#: The thumb's bones when folded across the palm, ending on the middle knuckle.
THUMB_CURLED: tuple[tuple[float, float], ...] = (
    (0.415, 0.735),
    (0.400, 0.680),
    (0.430, 0.655),
    (0.460, 0.690),
)

#: A curl, in degrees, that folds a finger well past the classifier's ``curled_threshold``
#: while staying a legal pose the processor accepts. Around ``150`` degrees measures as an
#: extension ratio of ``0.53``, comfortably inside the folded half of the band.
FIST_CURL: float = 150.0

#: A curl part-way through the ambiguous band, where extension and folding score alike.
#: Around ``90`` degrees measures as ``0.81``, right between the two thresholds.
HALF_CURL: float = 90.0

#: A curl just short of the straight end of the band: as extended as a real finger gets.
#: Around ``60`` degrees measures as ``0.91``, a hair under ``extended_threshold``, so it
#: still classifies as extended without scoring a saturated ``1.0``.
SLIGHT_CURL: float = 60.0

#: A curl right at the folded end of the band, where a few more degrees would not change
#: the verdict at all.
FOLDING_CURL: float = 120.0


def _step(position: tuple[float, float], length: float, degrees: float) -> tuple[float, float]:
    """Move ``length`` from ``position`` along ``degrees``, where ``-90`` is up."""
    radians = math.radians(degrees)
    return (position[0] + length * math.cos(radians), position[1] + length * math.sin(radians))


def _finger(
    finger: Finger,
    curl: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Place a finger's three moving joints, returning ``(pip, dip, tip)``.

    The first bone is laid down along the finger's base direction; each later bone turns
    by half the curl, so the total fold across the two interphalangeal joints is ``curl``.
    """
    half = curl / 2.0
    direction = SPREADS[finger]
    position = KNUCKLES[finger]
    joints = []
    for index, length in enumerate(BONES[finger]):
        if index:
            direction += half
        position = _step(position, length, direction)
        joints.append(position)
    return joints[0], joints[1], joints[2]


def _pinch_thumb(index_tip: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    """Lay the thumb out so its tip lands against ``index_tip``.

    The tip is placed first and the interphalangeal joint halfway to the knuckle, so the
    gap between the two fingertips is a known small number instead of whatever a fixed
    thumb pose happened to produce. Both lengths stay positive, which the processor
    requires.
    """
    thumb_mcp = THUMB_SPREAD[1]
    approach = _step(thumb_mcp, math.dist(thumb_mcp, index_tip) * 0.5, -50.0)
    return (THUMB_SPREAD[0], thumb_mcp, approach, (index_tip[0] - 0.010, index_tip[1] + 0.008))


def hand_coordinates(
    *,
    index: float = 0.0,
    middle: float = 0.0,
    ring: float = 0.0,
    pinky: float = 0.0,
    thumb: str = "spread",
    index_tip_offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[tuple[float, float, float], ...]:
    """Build 21 landmark coordinates for a posed hand.

    Args:
        index: Total interphalangeal bend of the index finger, in degrees.
        middle: The same for the middle finger.
        ring: The same for the ring finger.
        pinky: The same for the pinky finger.
        thumb: ``"spread"`` for an extended thumb, ``"curled"`` for one folded across the
            palm, or ``"pinch"`` to bring the thumb tip to the index tip.
        index_tip_offset: Shift applied to the index fingertip before a ``"pinch"`` thumb
            is laid out, so a test can place the fingertips near each other without
            exactly on top of each other.

    Returns:
        ``(x, y, z)`` for all 21 landmarks in topology order, all at ``z = 0.0``.

    Raises:
        ValueError: If ``thumb`` is not one of the three supported placements.
    """
    if thumb not in ("spread", "curled", "pinch"):
        raise ValueError(f"thumb must be 'spread', 'curled', or 'pinch', got {thumb!r}")
    curls = {
        Finger.INDEX: index,
        Finger.MIDDLE: middle,
        Finger.RING: ring,
        Finger.PINKY: pinky,
    }
    placed: dict[LandmarkIndex, tuple[float, float]] = {LandmarkIndex.WRIST: WRIST}
    for finger in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
        placed[_JOINTS[finger][0]] = KNUCKLES[finger]
        for landmark, point in zip(
            _JOINTS[finger][1:], _finger(finger, curls[finger]), strict=True
        ):
            placed[landmark] = point
    tip_x, tip_y = placed[LandmarkIndex.INDEX_TIP]
    placed[LandmarkIndex.INDEX_TIP] = (tip_x + index_tip_offset[0], tip_y + index_tip_offset[1])
    if thumb == "spread":
        thumb_points = THUMB_SPREAD
    elif thumb == "curled":
        thumb_points = THUMB_CURLED
    else:
        thumb_points = _pinch_thumb(placed[LandmarkIndex.INDEX_TIP])
    thumb_joints = (
        LandmarkIndex.THUMB_CMC,
        LandmarkIndex.THUMB_MCP,
        LandmarkIndex.THUMB_IP,
        LandmarkIndex.THUMB_TIP,
    )
    for landmark, point in zip(thumb_joints, thumb_points, strict=True):
        placed[landmark] = point
    return tuple((placed[landmark][0], placed[landmark][1], 0.0) for landmark in LandmarkIndex)


def hand_features(
    coordinates: Sequence[tuple[float, float, float]],
    handedness: Handedness = Handedness.LEFT,
) -> HandFeatures:
    """Run explicit coordinates through the real processor and return its features."""
    hand = TrackedHand(
        landmarks=tuple(Landmark(x=x, y=y, z=z) for x, y, z in coordinates),
        handedness=handedness,
    )
    return LandmarkProcessor().process(hand)


def posed(
    *,
    handedness: Handedness = Handedness.LEFT,
    **pose: float | str | tuple[float, float],
) -> HandFeatures:
    """Build a hand from :func:`hand_coordinates` and process it."""
    return hand_features(hand_coordinates(**pose), handedness=handedness)


def open_palm(handedness: Handedness = Handedness.LEFT) -> HandFeatures:
    """Every digit extended and spread."""
    return posed(handedness=handedness)


def fist(handedness: Handedness = Handedness.LEFT) -> HandFeatures:
    """Every digit folded into the palm, thumb tucked across the knuckles."""
    return posed(
        index=FIST_CURL,
        middle=FIST_CURL,
        ring=FIST_CURL,
        pinky=FIST_CURL,
        thumb="curled",
        handedness=handedness,
    )


def point(handedness: Handedness = Handedness.LEFT) -> HandFeatures:
    """Index extended, the other three folded."""
    return posed(
        middle=FIST_CURL,
        ring=FIST_CURL,
        pinky=FIST_CURL,
        thumb="curled",
        handedness=handedness,
    )


def two_fingers(handedness: Handedness = Handedness.LEFT) -> HandFeatures:
    """Index and middle extended, ring and pinky folded."""
    return posed(
        ring=FIST_CURL,
        pinky=FIST_CURL,
        thumb="curled",
        handedness=handedness,
    )


def pinch(handedness: Handedness = Handedness.LEFT) -> HandFeatures:
    """Thumb tip brought to the index tip, the other three fingers folded."""
    return posed(
        index=HALF_CURL,
        middle=FIST_CURL,
        ring=FIST_CURL,
        pinky=FIST_CURL,
        thumb="pinch",
        index_tip_offset=(-0.030, 0.020),
        handedness=handedness,
    )


def curled_thumb_fist(handedness: Handedness = Handedness.LEFT) -> HandFeatures:
    """A fist whose thumb tip happens to land against the index tip.

    The awkward overlap named in ``docs/classifier.md``: both ``FIST`` and ``PINCH`` have
    their conditions met, and which one wins is decided by the margin test rather than by
    the rules preferring one.
    """
    return posed(
        index=FIST_CURL,
        middle=FIST_CURL,
        ring=FIST_CURL,
        pinky=FIST_CURL,
        thumb="pinch",
        handedness=handedness,
    )


def mirrored(
    coordinates: Sequence[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], ...]:
    """Reflect a hand horizontally, producing its mirror image.

    A construction, not a behaviour: nothing in the classifier package mirrors anything.
    """
    return tuple((1.0 - x, y, z) for x, y, z in coordinates)


def translated(
    coordinates: Sequence[tuple[float, float, float]],
    dx: float,
    dy: float,
) -> tuple[tuple[float, float, float], ...]:
    """Shift a hand within the frame."""
    return tuple((x + dx, y + dy, z) for x, y, z in coordinates)


def scaled(
    coordinates: Sequence[tuple[float, float, float]],
    factor: float,
) -> tuple[tuple[float, float, float], ...]:
    """Scale a hand about its own wrist, as if it moved closer to the camera."""
    wrist_x, wrist_y, wrist_z = coordinates[0]
    return tuple(
        (
            wrist_x + (x - wrist_x) * factor,
            wrist_y + (y - wrist_y) * factor,
            wrist_z + (z - wrist_z) * factor,
        )
        for x, y, z in coordinates
    )


def without_finger(features: HandFeatures, finger: Finger) -> HandFeatures:
    """Return ``features`` with ``finger``'s three bones collapsed to zero length.

    Zero-length bones are the one malformed shape the classifier has to survive without
    dividing by zero, and the frozen domain type will not build them, so they are forced
    in with ``object.__setattr__``. The input is a fresh fixture value, so nothing shared
    is disturbed.
    """
    measurements = tuple(
        _collapsed_bones(features.finger(other)) if other is finger else features.finger(other)
        for other in FINGERS
    )
    object.__setattr__(features, "fingers", measurements)
    return features


def _collapsed_bones(geometry: FingerGeometry) -> FingerGeometry:
    object.__setattr__(geometry, "segment_lengths", (0.0, 0.0, 0.0, 0.0))
    return geometry


__all__ = [
    "BONES",
    "FIST_CURL",
    "FOLDING_CURL",
    "HAND_SCALE",
    "HALF_CURL",
    "KNUCKLES",
    "SLIGHT_CURL",
    "THUMB_CURLED",
    "THUMB_SPREAD",
    "WRIST",
    "curled_thumb_fist",
    "fist",
    "hand_coordinates",
    "hand_features",
    "mirrored",
    "open_palm",
    "pinch",
    "point",
    "posed",
    "scaled",
    "translated",
    "two_fingers",
    "without_finger",
]
