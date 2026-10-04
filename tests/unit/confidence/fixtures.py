"""Synthetic classifications for confidence-filter unit tests.

The filter's only input is a :class:`~gesturepilot.classifier.types.GestureClassification`,
so a synthetic one is the whole world it needs: no camera, no landmarks, no model asset, no
timing. That keeps these tests hermetic and, more usefully, keeps them *about* the filter —
a real classification would vary with every threshold in the classifier, so a failure here
would point at a rule rather than at the temporal logic under test.

The builders produce results that satisfy
:class:`~gesturepilot.classifier.types.GestureClassification`'s own contract, because that
contract is what the filter is entitled to assume: a recognised label must be the highest
scoring one, and the reported margin must be the real gap between the top two scores. A
hand-built result that broke those rules would be testing a fiction.

:class:`UNKNOWN` mirrors the classifier's own formula, ``confidence = 1 - winning_score``, so
an ``UNKNOWN`` with high confidence means weak evidence — the inversion a caller is most
likely to get wrong, and therefore worth exercising here.
"""

from __future__ import annotations

from gesturepilot.classifier.types import (
    RECOGNIZABLE_GESTURES,
    Gesture,
    GestureClassification,
)

#: The margin every recognised fixture reports unless a test asks for another. Chosen to sit
#: comfortably above the classifier's own ``min_margin`` so the fixtures describe confident
#: verdicts rather than marginal ones.
DEFAULT_MARGIN: float = 0.4


def classification(
    gesture: Gesture,
    confidence: float,
    *,
    margin: float = DEFAULT_MARGIN,
) -> GestureClassification:
    """A recognised classification of ``gesture`` at ``confidence``.

    The winner scores ``confidence`` and every rival scores ``confidence - margin``, which
    is what makes both the reported margin and the "highest scorer is the reported label"
    invariant true by construction.

    Args:
        gesture: The label to report. Must be recognised; use :func:`unknown` for
            ``UNKNOWN``.
        confidence: Strength of the label, in ``[0.0, 1.0]``. Must be ``> 0.0``: scores
            cannot go below zero, so a recognised gesture with zero confidence would have no
            strictly lower rival and could not be a valid result at all.
        margin: Reported lead over the runner-up, halved if halving is what keeps the
            runner-up strictly below the winner.
    """
    if not gesture.is_recognized:
        raise ValueError(f"{gesture.name} is not recognised; use unknown() instead")
    if confidence <= 0.0:
        raise ValueError(
            f"a recognised classification needs confidence > 0.0, got {confidence!r}; "
            f"use unknown() for a result with no confidence in any gesture"
        )
    lead = min(margin, confidence / 2.0)
    scores = dict.fromkeys(RECOGNIZABLE_GESTURES, 0.0)
    scores[gesture] = confidence
    for rival in RECOGNIZABLE_GESTURES:
        if rival is not gesture:
            scores[rival] = confidence - lead
    return GestureClassification(
        gesture=gesture,
        confidence=confidence,
        margin=lead,
        scores=scores,
    )


def unknown(confidence: float) -> GestureClassification:
    """An ``UNKNOWN`` classification at ``confidence``.

    ``confidence`` means confidence that *no* gesture applies: the winning rule score is
    ``1 - confidence``, all five scores are level, and the margin is therefore zero.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be within [0.0, 1.0], got {confidence!r}")
    level = 1.0 - confidence
    scores = dict.fromkeys(RECOGNIZABLE_GESTURES, level)
    return GestureClassification(
        gesture=Gesture.UNKNOWN,
        confidence=confidence,
        margin=0.0,
        scores=scores,
    )


def corrupt_confidence(result: GestureClassification, confidence: float) -> GestureClassification:
    """A copy of ``result`` whose confidence bypassed validation.

    :class:`~gesturepilot.classifier.types.GestureClassification` rejects a ``NaN`` or an
    out-of-range confidence on construction, so the only way to hand the filter one is to
    forge it the way a faulty unpickling or a future classifier version would. The filter
    still checks, because a ``NaN`` compares false against every threshold and would
    otherwise stop every gesture being accepted, silently, forever.
    """
    forged = GestureClassification(
        gesture=result.gesture,
        confidence=result.confidence,
        margin=result.margin,
        scores=result.scores,
    )
    object.__setattr__(forged, "confidence", confidence)
    return forged


__all__ = ["DEFAULT_MARGIN", "classification", "corrupt_confidence", "unknown"]
