"""Controlled synthetic hands for landmark-processing unit tests.

Every landmark here is placed by hand, so the geometry that comes out is known exactly
and can be asserted against literal numbers. Nothing is random, nothing is snapshotted,
and no test needs a webcam, a model asset, or a sleeping call.

The reference hand is a right hand, palm toward the camera, fingers extended, drawn in
normalised image coordinates with ``y`` increasing downward as in a captured frame. It is
deliberately *not* anatomically perfect — real hands are never that regular — but its
key quantities are exact and stable:

* the wrist sits at ``(0.500, 0.800)``
* the middle knuckle at ``(0.505, 0.630)``, giving a wrist-to-knuckle span of
  ``hypot(0.005, 0.170)``
* the palm centroid is the exact mean of the wrist and the four finger knuckles

Transformations are provided as explicit helpers rather than being applied inside
``make_hand``, so a test that means to check mirroring cannot quietly get a translated
hand instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from gesturepilot.processing.types import ProcessedLandmark
from gesturepilot.tracking.types import Handedness, Landmark, TrackedHand

#: ``(x, y, z)`` for all 21 landmarks, in the standard topology order.
HAND_COORDINATES: tuple[tuple[float, float, float], ...] = (
    (0.500, 0.800, 0.000),  # 0  WRIST
    (0.430, 0.760, -0.010),  # 1  THUMB_CMC
    (0.380, 0.700, -0.020),  # 2  THUMB_MCP
    (0.345, 0.655, -0.025),  # 3  THUMB_IP
    (0.320, 0.620, -0.030),  # 4  THUMB_TIP
    (0.455, 0.640, -0.010),  # 5  INDEX_MCP
    (0.450, 0.520, -0.015),  # 6  INDEX_PIP
    (0.448, 0.455, -0.015),  # 7  INDEX_DIP
    (0.447, 0.400, -0.015),  # 8  INDEX_TIP
    (0.505, 0.630, -0.010),  # 9  MIDDLE_MCP
    (0.507, 0.495, -0.015),  # 10 MIDDLE_PIP
    (0.508, 0.425, -0.015),  # 11 MIDDLE_DIP
    (0.508, 0.365, -0.015),  # 12 MIDDLE_TIP
    (0.552, 0.638, -0.010),  # 13 RING_MCP
    (0.560, 0.510, -0.015),  # 14 RING_PIP
    (0.564, 0.445, -0.015),  # 15 RING_DIP
    (0.566, 0.390, -0.015),  # 16 RING_TIP
    (0.596, 0.652, -0.008),  # 17 PINKY_MCP
    (0.610, 0.545, -0.012),  # 18 PINKY_PIP
    (0.617, 0.495, -0.012),  # 19 PINKY_DIP
    (0.622, 0.450, -0.012),  # 20 PINKY_TIP
)

#: The wrist-to-middle-knuckle span, which is the hand scale of :data:`HAND_COORDINATES`.
HAND_SCALE: float = math.hypot(0.505 - 0.500, 0.630 - 0.800)

#: ``(x, y)`` palm centroid of :data:`HAND_COORDINATES`.
PALM_CENTRE: tuple[float, float] = (0.5216, 0.672)

#: Mean ``z`` of the wrist and the four finger knuckles.
PALM_CENTRE_Z: float = -0.0076


def make_landmarks(
    coordinates: Sequence[tuple[float, float, float]] = HAND_COORDINATES,
) -> tuple[Landmark, ...]:
    """Build tracker landmarks from explicit coordinates."""
    return tuple(Landmark(x=x, y=y, z=z) for x, y, z in coordinates)


def make_hand(
    *,
    coordinates: Sequence[tuple[float, float, float]] = HAND_COORDINATES,
    handedness: Handedness = Handedness.LEFT,
    handedness_score: float | None = 0.97,
    detection_score: float | None = None,
) -> TrackedHand:
    """Build a :class:`TrackedHand` from explicit coordinates."""
    return TrackedHand(
        landmarks=make_landmarks(coordinates),
        handedness=handedness,
        handedness_score=handedness_score,
        detection_score=detection_score,
    )


def make_processed(
    coordinates: Sequence[tuple[float, float, float]] = HAND_COORDINATES,
) -> tuple[ProcessedLandmark, ...]:
    """Build processing-owned landmarks from explicit coordinates."""
    return tuple(ProcessedLandmark(x=x, y=y, z=z) for x, y, z in coordinates)


def mirrored(
    coordinates: Sequence[tuple[float, float, float]] = HAND_COORDINATES,
) -> tuple[tuple[float, float, float], ...]:
    """Reflect horizontally about the frame's vertical centre line.

    Used to build the *mirror image* of a hand. This is a construction, not a behaviour:
    nothing in the processing package performs it.
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
    """Scale a hand about its own wrist, leaving the wrist in place.

    Scaling about the wrist rather than the origin keeps the result resembling a hand
    moving closer to or further from the camera, which is exactly the nuisance the
    normalisation is supposed to cancel.
    """
    wrist_x, wrist_y, wrist_z = coordinates[0]
    return tuple(
        (
            wrist_x + (x - wrist_x) * factor,
            wrist_y + (y - wrist_y) * factor,
            wrist_z + (z - wrist_z) * factor,
        )
        for x, y, z in coordinates
    )


def collapsed() -> tuple[tuple[float, float, float], ...]:
    """Twenty-one landmarks stacked on a single point: geometrically unusable."""
    return tuple((0.5, 0.5, 0.0) for _ in range(21))


def with_collapsed_joint(index: int) -> tuple[tuple[float, float, float], ...]:
    """A hand whose landmark ``index`` is moved onto its predecessor.

    Models a tightly curled finger: the landmarks stay distinct enough for the hand to
    have a scale, but one joint's two segments become collinear-and-overlapping, so its
    interior angle has no defined direction.
    """
    coordinates = list(HAND_COORDINATES)
    coordinates[index] = coordinates[index - 1]
    return tuple(coordinates)


def finger_curl() -> tuple[tuple[float, float, float], ...]:
    """A hand whose index, middle, ring, and pinky tips sit on their middle joints.

    A legal, realistic pose that keeps a measurable hand scale while collapsing several
    interior joint angles. Used to prove that a folded hand yields ``None`` angles rather
    than a crash or a NaN.
    """
    coordinates = list(HAND_COORDINATES)
    coordinates[8] = coordinates[7]  # INDEX_TIP onto INDEX_DIP
    coordinates[12] = coordinates[11]  # MIDDLE_TIP onto MIDDLE_DIP
    coordinates[16] = coordinates[15]  # RING_TIP onto RING_DIP
    coordinates[20] = coordinates[19]  # PINKY_TIP onto PINKY_DIP
    return tuple(coordinates)
