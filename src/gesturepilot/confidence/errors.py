"""Confidence-filter exception hierarchy.

Callers handle temporal filtering failures through these types rather than raw
``ValueError`` or ``TypeError``. The split mirrors :mod:`gesturepilot.processing.errors`:
rubbish in, versus a well-formed request the filter's own state cannot support.

``ConfidenceError``
    Base class for every confidence-filter failure. Catch this to handle them all.
``InvalidClassificationError``
    The classification handed in is not something the filter can judge: not a
    :class:`~gesturepilot.classifier.types.GestureClassification` at all, or one
    carrying a confidence that is not a finite number in ``[0.0, 1.0]``.
``FilterStateError``
    Temporal filtering was asked to do something its state cannot support: a missing or
    non-monotonic timestamp, or a timestamp out of range.

Note what is *not* here. A classification the filter rejects — an ``UNKNOWN``, or a
gesture below the confidence threshold — is not an error. That is the filter answering,
which is its entire job. Only input the filter could not evaluate raises, so a caller
never has to wrap an ordinary ambiguous pose in a ``try``.
"""

from __future__ import annotations


class ConfidenceError(Exception):
    """Base class for every confidence-filter failure."""


class InvalidClassificationError(ConfidenceError):
    """The classification handed to the filter could not be judged.

    Covers a non-:class:`~gesturepilot.classifier.types.GestureClassification` object,
    and one whose confidence is not finite or not within ``[0.0, 1.0]``.

    :class:`~gesturepilot.classifier.types.GestureClassification` rejects those values on
    construction, so reaching this error means the object was built by bypassing
    validation — ``object.__setattr__``, an unpickled instance, or a future version of
    the classifier. Checking anyway is the same last line of defence the processor makes
    before it starts dividing by hand scale: a single ``NaN`` would compare false against
    the threshold and so quietly stop every gesture being accepted, forever, with no error
    anywhere.
    """


class FilterStateError(ConfidenceError):
    """Temporal filtering was asked to do something its state cannot support.

    Raised when a timestamp is not a real number, not finite, negative, or moves backwards
    relative to the last one the filter saw. A backwards timestamp means the caller handed
    over observations out of order; rather than produce a plausible-looking but wrong
    report, the filter raises, because the failure would otherwise surface much later as a
    gesture that inexplicably refuses to stabilise.
    """


__all__ = [
    "ConfidenceError",
    "FilterStateError",
    "InvalidClassificationError",
]
