"""Turning one tracked hand into position- and size-independent coordinates.

The tracker hands over landmarks in normalised *image* space: where each point sits in
the frame, at whatever size the hand happened to appear. Both of those vary constantly
during normal use — the user leans, the hand drifts, they move closer to the camera —
and neither says anything about the pose being recognised. A classifier comparing raw
coordinates would spend all of its effort compensating for camera placement.

Normalisation removes both nuisances with two operations:

**Translate.** Subtract the wrist, so the hand's position in the frame drops out. The
wrist is the right origin because it is the one landmark that is always present, never
hidden by the hand itself, and sits at the base of everything being measured.

**Scale.** Divide by the hand scale — the wrist-to-middle-knuckle distance, computed in
:func:`gesturepilot.processing.geometry.hand_scale`. After this, one unit means "one
palm length", so the same numbers describe a hand filling the frame and a hand held at
arm's length.

What this does **not** deliver
-----------------------------

Camera-distance invariance is *approximate*, not exact. Normalised image coordinates
conflate apparent size with distance, so dividing by the hand scale removes the apparent
size and with it most of the distance dependence. It cannot recover foreshortening: a
hand turned edge-on to the camera projects smaller, and no single scalar undoes that.
The backend also emits no world landmarks, so there is no metric 3-D reconstruction
available to do better. A gesture that depends on fine depth discrimination should not
rely on what comes out of here.

Degeneracy
----------

If the wrist and the middle knuckle coincide, the scale is zero and every normalised
coordinate would be infinite. That is refused with
:class:`~gesturepilot.processing.errors.DegenerateHandError` rather than papered over.
The inputs are never modified: normalisation allocates new values and the caller's
landmarks are left untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

from gesturepilot.processing.errors import InvalidLandmarksError
from gesturepilot.processing.geometry import hand_scale, landmark_point, palm_center, relative_to
from gesturepilot.processing.topology import LandmarkIndex
from gesturepilot.processing.types import NormalizedHand, NormalizedLandmark, ProcessedLandmark


def normalize_point(
    point: ProcessedLandmark,
    origin: ProcessedLandmark,
    scale: float,
) -> NormalizedLandmark:
    """Return ``point`` in normalised hand space relative to ``origin``.

    Raises:
        DegenerateHandError: If ``scale`` is not a finite positive value.
    """
    dx, dy, dz = relative_to(landmark_point(point), landmark_point(origin), scale)
    return NormalizedLandmark(x=dx, y=dy, z=dz)


def normalize_landmarks(
    landmarks: Sequence[ProcessedLandmark],
    origin: ProcessedLandmark,
    scale: float,
) -> tuple[NormalizedLandmark, ...]:
    """Return every landmark in normalised hand space, preserving order.

    Raises:
        DegenerateHandError: If ``scale`` is not a finite positive value.
    """
    return tuple(normalize_point(landmark, origin, scale) for landmark in landmarks)


def normalize_hand(landmarks: Sequence[ProcessedLandmark]) -> NormalizedHand:
    """Normalise one hand against its own wrist and hand scale.

    Args:
        landmarks: Exactly one hand's landmarks, in topology order. Not modified.

    Returns:
        A :class:`~gesturepilot.processing.types.NormalizedHand` holding the wrist as
        the origin, the scale that was applied, every landmark in normalised space, and
        the normalised palm centre.

    Raises:
        InvalidLandmarksError: If no landmarks were supplied.
        DegenerateHandError: If the hand has no measurable size — see
            :func:`gesturepilot.processing.geometry.hand_scale`.
    """
    if not landmarks:
        raise InvalidLandmarksError("Cannot normalise an empty landmark sequence")

    wrist = landmarks[LandmarkIndex.WRIST]
    scale = hand_scale(landmarks)
    origin = landmark_point(wrist)

    # The palm centre is measured in tracker image space and then normalised by the
    # same origin and scale as the landmarks, so it is consistent with them by
    # construction rather than by a second, slightly different derivation.
    centre_x, centre_y, centre_z = relative_to(palm_center(landmarks), origin, scale)

    return NormalizedHand(
        origin=wrist,
        scale=scale,
        landmarks=normalize_landmarks(landmarks, wrist, scale),
        palm_center=NormalizedLandmark(x=centre_x, y=centre_y, z=centre_z),
    )


__all__ = [
    "normalize_hand",
    "normalize_landmarks",
    "normalize_point",
]
