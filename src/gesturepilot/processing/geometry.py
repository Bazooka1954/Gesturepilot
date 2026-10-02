"""Small, pure geometry helpers for landmark processing.

Every function here is a plain typed function over tuples of floats. There is no
class hierarchy, no generic vector framework, and no state. That is deliberate: the
amount of linear algebra this project needs is small and well known, and a hand-rolled
maths library would be more surface area to maintain than it saves.

**Zero-length vectors.** The functions that merely *measure* — :func:`magnitude`,
:func:`dot`, :func:`distance` — are total: they return a finite number for any input,
including all zeros. The functions that would have to *divide* by such a vector raise
:class:`~gesturepilot.processing.errors.DegenerateHandError` rather than return
``inf`` or ``NaN``. Where a degenerate measurement is a legitimate result rather than a
failure, an ``optional_`` variant returns ``None`` instead of raising: a folded finger
has no defined joint angle, but the hand is perfectly real.

**2-D versus 3-D.** ``x`` and ``y`` come from the image and are metric relative to each
other. ``z`` is the backend's depth estimate, only roughly commensurate with them, and
far noisier. Angles are therefore taken in the image plane by dropping ``z``, and the
3-D helpers are used only where a depth-aware magnitude is genuinely wanted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from gesturepilot.processing.errors import DegenerateHandError, InvalidLandmarksError
from gesturepilot.processing.topology import (
    HAND_SCALE_LANDMARKS,
    PALM_LANDMARKS,
    LandmarkIndex,
)
from gesturepilot.processing.types import ProcessedLandmark

#: A point or vector of any dimension. Used where the helpers work generically.
Vector = tuple[float, ...]

#: An image-plane point: ``(x, y)`` in normalised frame coordinates.
Point2 = tuple[float, float]

#: A point including the backend's relative depth: ``(x, y, z)``.
Point3 = tuple[float, float, float]


def _check_same_dimension(a: Vector, b: Vector) -> None:
    if len(a) != len(b):
        raise ValueError(f"Expected vectors of equal length, got {len(a)} and {len(b)}")


def subtract(a: Vector, b: Vector) -> Vector:
    """Return ``a - b``, component by component.

    Raises:
        ValueError: If the two vectors have different dimensions.
    """
    _check_same_dimension(a, b)
    return tuple(x - y for x, y in zip(a, b, strict=True))


def add(a: Vector, b: Vector) -> Vector:
    """Return ``a + b``, component by component.

    Raises:
        ValueError: If the two vectors have different dimensions.
    """
    _check_same_dimension(a, b)
    return tuple(x + y for x, y in zip(a, b, strict=True))


def magnitude(vector: Vector) -> float:
    """Return the Euclidean length of ``vector``.

    Total: a zero vector has magnitude ``0.0``, which is a real answer, not a failure.
    Uses :func:`math.fsum` so a long vector of small squares does not lose precision.
    """
    return math.sqrt(math.fsum(component * component for component in vector))


def dot(a: Vector, b: Vector) -> float:
    """Return the dot product of ``a`` and ``b``.

    Raises:
        ValueError: If the two vectors have different dimensions.
    """
    _check_same_dimension(a, b)
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def distance(a: Vector, b: Vector) -> float:
    """Return the straight-line distance between two points."""
    return magnitude(subtract(a, b))


def distance_2d(a: Vector, b: Vector) -> float:
    """Return the distance between two points in the image plane, ignoring depth."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def relative_to(point: Vector, origin: Vector, scale: float = 1.0) -> Vector:
    """Express ``point`` relative to ``origin``, optionally divided by ``scale``.

    Args:
        point: The point to express.
        origin: The reference point to subtract.
        scale: Divisor applied to the result. ``1.0`` leaves it in the input units.

    Returns:
        ``(point - origin) / scale`` as a new vector.

    Raises:
        ValueError: If ``point`` and ``origin`` have different dimensions.
        DegenerateHandError: If ``scale`` is zero or negative.
    """
    if scale <= 0.0 or not math.isfinite(scale):
        raise DegenerateHandError(
            f"Cannot normalise against a scale of {scale!r}; it must be a finite positive value."
        )
    return tuple((x - o) / scale for x, o in zip(point, origin, strict=True))


def angle_between(a: Vector, b: Vector) -> float:
    """Return the angle in radians between two vectors, in ``[0.0, pi]``.

    The ``acos`` argument is clamped to ``[-1.0, 1.0]``. Rounding alone can push it a
    hair outside that range, and ``math.acos`` raises rather than returning a sensible
    angle, so the clamp turns a numerical edge case into the correct boundary value.

    Raises:
        ValueError: If the two vectors have different dimensions.
        DegenerateHandError: If either vector has zero length. A zero vector has no
            direction, so the angle between them is genuinely undefined. Use
            :func:`optional_angle_between` where ``None`` is an acceptable answer.
    """
    length_a = magnitude(a)
    length_b = magnitude(b)
    if length_a == 0.0 or length_b == 0.0:
        raise DegenerateHandError(
            "Cannot compute an angle against a zero-length vector; it has no direction."
        )
    cosine = dot(a, b) / (length_a * length_b)
    return math.acos(max(-1.0, min(1.0, cosine)))


def optional_angle_between(a: Vector, b: Vector) -> float | None:
    """Return the angle between two vectors, or ``None`` if either is degenerate.

    The safe counterpart to :func:`angle_between`. Use it when a collapsed segment is a
    plausible input rather than a bug — a tightly curled finger, for instance.
    """
    if magnitude(a) == 0.0 or magnitude(b) == 0.0:
        return None
    return angle_between(a, b)


def angle_at(a: Vector, b: Vector, c: Vector) -> float:
    """Return the angle in radians at vertex ``b``, between the rays ``b -> a`` and ``b -> c``.

    Raises:
        DegenerateHandError: If ``b`` coincides with ``a`` or ``c``.
    """
    return angle_between(subtract(a, b), subtract(c, b))


def optional_angle_at(a: Vector, b: Vector, c: Vector) -> float | None:
    """Return the angle at vertex ``b``, or ``None`` if either segment is degenerate."""
    return optional_angle_between(subtract(a, b), subtract(c, b))


def centroid(points: Sequence[Vector]) -> Vector:
    """Return the component-wise mean of ``points``.

    Raises:
        ValueError: If ``points`` is empty or its members differ in dimension.
    """
    if not points:
        raise ValueError("centroid() needs at least one point")
    dimension = len(points[0])
    if any(len(point) != dimension for point in points):
        raise ValueError("centroid() needs all points to have the same dimension")
    count = len(points)
    return tuple(math.fsum(point[i] for point in points) / count for i in range(dimension))


def landmark_point(landmark: ProcessedLandmark) -> Point3:
    """Return a landmark as a plain ``(x, y, z)`` tuple."""
    return landmark.as_tuple


def _require_topology(landmarks: Sequence[ProcessedLandmark]) -> None:
    if len(landmarks) <= max(HAND_SCALE_LANDMARKS) or len(landmarks) <= max(PALM_LANDMARKS):
        raise InvalidLandmarksError(
            f"A full hand needs at least {max(HAND_SCALE_LANDMARKS) + 1} landmarks, "
            f"got {len(landmarks)}"
        )


def palm_center(landmarks: Sequence[ProcessedLandmark]) -> Point3:
    """Return the palm centre as the centroid of the wrist and the four finger knuckles.

    Averaging the wrist with the knuckles gives a point that sits in the middle of the
    palm rather than at the base of it, and that moves only with the palm — fingertips
    swinging through it barely registers. That makes it a far steadier reference than
    any single landmark.

    The result is 3-D: ``x`` and ``y`` are the palm centre in the image, ``z`` is the
    mean of the knuckles' depth relative to the wrist.

    Raises:
        InvalidLandmarksError: If fewer landmarks are supplied than the topology needs.
    """
    _require_topology(landmarks)
    points = [landmark_point(landmarks[index]) for index in PALM_LANDMARKS]
    x, y, z = centroid(points)
    return (x, y, z)


def hand_scale(landmarks: Sequence[ProcessedLandmark]) -> float:
    """Return the wrist-to-middle-knuckle distance in the image plane.

    This is GesturePilot's unit of hand size. The wrist-to-middle-knuckle span is
    preferred over a knuckle-to-knuckle width because it is the longest bone in the palm
    and, crucially, because it barely moves as the fingers open and close — a reference
    built from fingertips would shift with the very motion a classifier is measuring.

    Raises:
        InvalidLandmarksError: If fewer landmarks are supplied than the topology needs.
        DegenerateHandError: If the two landmarks coincide, which would make every
            normalised measurement a division by zero.
    """
    _require_topology(landmarks)
    start_index, end_index = HAND_SCALE_LANDMARKS
    span = distance_2d(
        landmark_point(landmarks[start_index]),
        landmark_point(landmarks[end_index]),
    )
    if span == 0.0:
        raise DegenerateHandError(
            f"Hand scale collapsed: landmarks {start_index.name} and {end_index.name} "
            "coincide, so the hand has no measurable size."
        )
    return span


def orientation_from_landmarks(
    landmarks: Sequence[ProcessedLandmark],
) -> tuple[float | None, float | None]:
    """Return ``(roll, tilt)`` of the palm in the image plane.

    Computed from two geometry-chosen reference directions — index knuckle minus pinky
    knuckle for *roll*, wrist to middle knuckle for *tilt* — so it is well defined for
    both hands without consulting handedness. Either component is ``None`` if its
    reference direction collapses.

    **Roll is an axis, not a ray.** The across-palm axis has no inherent sign: which end
    is the index side is exactly the question handedness would answer, and that
    consultation is what this package refuses to make. So the raw direction is folded into
    ``[-pi/2, pi/2]``, where ``0.0`` means "across the frame" and the sign says which way
    the palm leans. Mirroring a hand then negates ``roll`` exactly, rather than shifting
    it by ``pi`` — the measurement behaves identically for both hands, which is the whole
    point of not consulting the label.

    **Tilt is a ray.** The wrist-to-knuckle direction is oriented — the knuckle is
    downstream of the wrist — so ``tilt`` keeps the full ``[-pi, pi]`` range. It is
    measured from "up the frame" (image ``-y``) toward ``+x``, so a hand pointing at the
    ceiling reads ``0.0`` rather than an arbitrary ``atan2`` branch cut.

    Angles are unaffected by the translation and scaling applied during normalisation, so
    this is computed in normalised hand space purely for consistency; the numbers are
    identical either way.

    Raises:
        InvalidLandmarksError: If fewer landmarks are supplied than the topology needs.
    """
    _require_topology(landmarks)
    wrist = landmark_point(landmarks[LandmarkIndex.WRIST])
    middle_mcp = landmark_point(landmarks[LandmarkIndex.MIDDLE_MCP])
    index_mcp = landmark_point(landmarks[LandmarkIndex.INDEX_MCP])
    pinky_mcp = landmark_point(landmarks[LandmarkIndex.PINKY_MCP])

    across = (index_mcp[0] - pinky_mcp[0], index_mcp[1] - pinky_mcp[1])
    along = (middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])

    roll = (
        math.remainder(math.atan2(across[1], across[0]), math.pi) if across != (0.0, 0.0) else None
    )
    tilt = math.atan2(along[0], -along[1]) if along != (0.0, 0.0) else None
    return roll, tilt


__all__ = [
    "Point2",
    "Point3",
    "Vector",
    "add",
    "angle_at",
    "angle_between",
    "centroid",
    "distance",
    "distance_2d",
    "dot",
    "hand_scale",
    "landmark_point",
    "magnitude",
    "optional_angle_at",
    "optional_angle_between",
    "orientation_from_landmarks",
    "palm_center",
    "relative_to",
    "subtract",
]
