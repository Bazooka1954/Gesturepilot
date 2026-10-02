"""Landmark-processing exception hierarchy.

Callers handle processing failures through these types rather than raw ``ValueError`` or
``TypeError``. Each one marks a failure the caller can actually do something about.

``ProcessingError``
    Base class for every landmark-processing failure. Catch this to handle them all.
``InvalidLandmarksError``
    The input hand is not usable: wrong landmark count, a non-finite coordinate, or
    not a :class:`~gesturepilot.tracking.types.TrackedHand` at all.
``DegenerateHandError``
    The hand is well-formed but geometrically unusable — a collapsed scale
    reference, a zero-length segment, or two landmarks sitting on top of each other.
    Nothing sensible can be normalised or measured from it.
``FilterStateError``
    Temporal smoothing was asked to do something its state cannot support: a missing
    or non-monotonic timestamp, a non-finite sample, or a parameter outside the
    filter's valid range.

The split matters. A caller can distinguish "the tracker gave me rubbish"
(``InvalidLandmarksError``) from "the hand is real but folded into a single point"
(``DegenerateHandError``) and choose different recovery for each — the first is a
detection problem, the second is a legitimate pose the pipeline must tolerate.
"""

from __future__ import annotations


class ProcessingError(Exception):
    """Base class for every landmark-processing failure."""


class InvalidLandmarksError(ProcessingError):
    """The supplied hand could not be read as a usable set of landmarks.

    Covers a malformed input object, a landmark count that does not match the
    topology, and non-finite coordinates. Always raised *before* any arithmetic runs,
    so a bad value can never propagate into a feature silently.
    """


class DegenerateHandError(ProcessingError):
    """The hand is well-formed but geometrically degenerate.

    Raised when a reference distance collapses to zero — the wrist and the middle-finger
    knuckle on top of each other, or two joints of one finger coincident. Dividing by
    such a distance would produce ``inf`` or ``NaN``, so the operation is refused
    instead. Joint angles that cannot be formed this way are reported as ``None`` in
    :class:`~gesturepilot.processing.types.FingerGeometry`, because a folded finger is a
    normal pose rather than a broken hand; a collapsed *hand scale* is not, so that
    still raises.
    """


class FilterStateError(ProcessingError):
    """Temporal filtering was asked to do something its state cannot support.

    For example a non-increasing timestamp, a non-finite sample, or calling the filter
    without the timestamp the smoothing stage requires.
    """


__all__ = [
    "DegenerateHandError",
    "FilterStateError",
    "InvalidLandmarksError",
    "ProcessingError",
]
