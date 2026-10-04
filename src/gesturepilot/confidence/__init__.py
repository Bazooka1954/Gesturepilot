"""Temporal confidence filtering: classifications in, stability verdicts out.

This layer implements the **Confidence Filter** stage of the pipeline. It takes the
:class:`~gesturepilot.classifier.types.GestureClassification` results the classifier
produced and answers the question the classifier deliberately cannot: *has this gesture now
been seen often enough, and strongly enough, to be called stable?*

It is the first stage with memory. It adds none of its own: no landmarks, no frames, no
camera, no OS, no events, no actions.

Boundaries: this package imports only from ``gesturepilot.classifier``'s published domain
types. It reaches neither the camera, nor MediaPipe, nor OpenCV, nor NumPy, nor any
automation library; it reads no clock, starts no thread, and touches no file or socket —
enforced by AST tests in ``tests/unit/confidence/test_boundaries.py``. A caller supplies
timestamps if it wants time-based interruption detection; the filter never asks for one.

**Accepted is not authorised.** A gesture this layer calls stable has been seen
consistently. Whether that is enough to close a window is the safety state machine's
decision, in a later phase, and this package deliberately provides no way to act.

``docs/confidence-filter.md`` records the state model, the threshold semantics, and the
known limitations.

Typical use::

    from gesturepilot.confidence import ConfidenceFilter, ConfidenceFilterConfig

    stability = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
    for hand in hands_this_frame:
        report = stability.observe(classifier.classify(hand), timestamp=frame.timestamp)
        if report.is_stable:
            print(f"{report.accepted} held for {report.candidate_observations} observations")

No MediaPipe, OpenCV, or NumPy type is exposed anywhere in this API.
"""

from gesturepilot.confidence.errors import (
    ConfidenceError,
    FilterStateError,
    InvalidClassificationError,
)
from gesturepilot.confidence.filter import ConfidenceFilter
from gesturepilot.confidence.types import (
    CandidateChangePolicy,
    ConfidenceFilterConfig,
    FilterState,
    StabilityReport,
)

__all__ = [
    "CandidateChangePolicy",
    "ConfidenceError",
    "ConfidenceFilter",
    "ConfidenceFilterConfig",
    "FilterState",
    "FilterStateError",
    "InvalidClassificationError",
    "StabilityReport",
]
