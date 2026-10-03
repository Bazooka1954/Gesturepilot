"""Gesture-classification exception hierarchy.

Callers handle classification failures through these types rather than a raw
``ValueError`` or ``TypeError``.

``ClassificationError``
    Base class for every classification failure. Catch this to handle them all.
``InvalidFeaturesError``
    The object handed to the classifier is not something it can measure — not a
    :class:`~gesturepilot.processing.types.HandFeatures` at all, or one holding values
    that are ``NaN`` or infinite.

The split is deliberately small. Recognition does not fail in many ways: either a hand
was readable or it was not. A hand that could not be classified is **not** a failure —
it is ``Gesture.UNKNOWN``, a normal result that happens often enough to deserve a
first-class label — so the only exceptions here are for input the rules could not even
be evaluated against. That is a much narrower category than "no gesture matched", and
keeping the two apart is what stops a caller from wrapping an ordinary ambiguous pose
in a ``try``.
"""

from __future__ import annotations


class ClassificationError(Exception):
    """Base class for every gesture-classification failure."""


class InvalidFeaturesError(ClassificationError):
    """The features handed to the classifier could not be measured.

    Raised for a non-:class:`~gesturepilot.processing.types.HandFeatures` object, and
    for one whose landmarks or measurements contain ``NaN`` or infinity. The domain
    types reject those values on construction, so reaching this error means the object
    was built by bypassing validation — ``object.__setattr__``, an unpickled instance,
    or a future version of the processing layer. Checking anyway is the same
    last-line-of-defence the processor makes before it starts dividing by hand scale: a
    single ``NaN`` would otherwise compare false against every threshold and silently
    produce ``UNKNOWN`` for every hand, forever, with no error anywhere.
    """


__all__ = ["ClassificationError", "InvalidFeaturesError"]
