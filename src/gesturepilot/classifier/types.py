"""Static gesture-classification domain types.

A **static** classifier reads one frame of geometry and answers once. It has no memory:
nothing here looks at an earlier frame, a timestamp, or a frame sequence number. Deciding
what to make of a *sequence* of classifications — smoothing, debouncing, hold-to-confirm
— belongs to the confidence filter and safety state machine further down the pipeline,
and folding any of that in here would make a classification impossible to test from a
single input.

The vocabulary
--------------
Five static finger configurations are recognised, plus an explicit
:attr:`Gesture.UNKNOWN`:

``OPEN_PALM``
    Every digit extended and spread, the way a hand rests held up to be read.
``FIST``
    Every digit folded into the palm.
``POINT``
    Index extended, the other three fingers folded. The thumb is deliberately not
    part of this gesture: a pointing hand holds its thumb in whatever position is
    comfortable, and demanding one position would reject real pointing hands.
``TWO_FINGERS``
    Index and middle extended, ring and pinky folded. Again the thumb is excluded.
``PINCH``
    Thumb tip and index tip brought together, the other three fingers folded.
``UNKNOWN``
    No gesture applies, or the evidence does not separate one from another.

Confidence
----------
``confidence`` means *how strongly the reported label is supported*, in
``[0.0, 1.0]``. For a recognised gesture it blends how completely that gesture's
conditions were met with how far it outscored its nearest rival. For
:attr:`Gesture.UNKNOWN` it means the opposite — confidence that **no** gesture
applies, which is *high* exactly when the evidence was weak or the candidates were
neck and neck. A confident ``UNKNOWN`` is a good outcome, not a weak one.

A caller that wants "how strongly does this look like a point?" should read
:meth:`GestureClassification.score_for` rather than the confidence. Nothing here is
learned, downloaded, or calibrated against a dataset: every threshold is a
:class:`ClassifierConfig` field with a documented meaning and a validated range.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from gesturepilot.processing.topology import FINGERS, Finger

#: The lowest confidence any classification may report.
MIN_CONFIDENCE = 0.0

#: The highest confidence any classification may report.
MAX_CONFIDENCE = 1.0

#: Slack allowed when re-deriving a classification's margin from its scores. The
#: classifier computes both from the same floats in one place, so this only absorbs
#: rounding in a hand-built result, not real disagreement.
_MARGIN_TOLERANCE = 1e-9


class Gesture(Enum):
    """The closed set of labels this classifier can emit.

    Values are lowercase strings so a label is readable in a log line or a config file
    without this package's ``Enum`` ceremony leaking into the caller's data.

    Members:

    ``OPEN_PALM``, ``FIST``, ``POINT``, ``PINCH``, ``TWO_FINGERS``
        The recognised static configurations, described in this module's docstring.
    ``UNKNOWN``
        The explicit "none of the above, or not with enough confidence" answer. It is a
        real state rather than an absent value or an exception, because "I saw a hand
        and it was not one of the five" is an outcome the caller has to handle.
    """

    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"
    PINCH = "pinch"
    TWO_FINGERS = "two_fingers"
    UNKNOWN = "unknown"

    @property
    def is_recognized(self) -> bool:
        """Whether this label is one of the five recognised gestures.

        ``False`` only for :attr:`UNKNOWN`. A classifier result is "recognised" when the
        hand matched a rule *and* beat its rivals by enough; ``is_recognized`` says which
        labels are capable of that.
        """
        return self is not Gesture.UNKNOWN

    def __str__(self) -> str:
        return self.value


#: The gestures a rule can actually produce, in rule order. ``UNKNOWN`` is absent because
#: it is a decision about the other scores, not evidence in its own right.
RECOGNIZABLE_GESTURES: tuple[Gesture, ...] = (
    Gesture.OPEN_PALM,
    Gesture.FIST,
    Gesture.POINT,
    Gesture.PINCH,
    Gesture.TWO_FINGERS,
)


@dataclass(frozen=True)
class ClassifierConfig:
    """Every threshold the classifier uses, with nothing hard-coded inside the rules.

    All fields are validated on construction, so an out-of-range value is reported where
    it was written rather than as a silent misclassification much later. The defaults
    are a starting point for a palm held toward the camera — deliberately conservative,
    biased toward ``UNKNOWN`` over a wrong guess, since a wrong label is worse for a
    control surface than no label.

    Attributes:
        extended_threshold: Extension ratio at or above which a digit counts as fully
            extended. A perfectly straight digit scores ``1.0``; see
            :mod:`gesturepilot.classifier.classifier` for the ratio's definition.
        curled_threshold: Extension ratio at or below which a digit counts as fully
            folded. Must be strictly below ``extended_threshold`` — the gap between the
            two is the window in which a digit is genuinely ambiguous, and a zero-width
            window would silently turn every measurement into a yes/no answer.
        pinch_close_threshold: Thumb-tip-to-index-tip distance, in hand-scale units, at
            or below which the tips count as touching.
        pinch_open_threshold: The same distance at or above which they count as clearly
            apart. Must exceed ``pinch_close_threshold`` for the same reason.
        min_score: The lowest winning rule score that still counts as a recognition.
            Below this the best-matching gesture was not convincingly present at all.
        min_margin: The smallest lead over the runner-up that still counts as a
            recognition. This is what turns a pose that matches two rules equally well
            into ``UNKNOWN`` rather than an arbitrary winner.
        strong_margin: The lead at which the separation term is fully saturated. A
            smaller lead still counts, it just makes the reported confidence lower.
    """

    extended_threshold: float = 0.92
    curled_threshold: float = 0.70
    pinch_close_threshold: float = 0.12
    pinch_open_threshold: float = 0.30
    min_score: float = 0.55
    min_margin: float = 0.12
    strong_margin: float = 0.30

    def __post_init__(self) -> None:
        for name, value in (
            ("extended_threshold", self.extended_threshold),
            ("curled_threshold", self.curled_threshold),
            ("pinch_close_threshold", self.pinch_close_threshold),
            ("pinch_open_threshold", self.pinch_open_threshold),
            ("min_score", self.min_score),
            ("min_margin", self.min_margin),
            ("strong_margin", self.strong_margin),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"ClassifierConfig.{name} must be a real number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"ClassifierConfig.{name} must be finite, got {value!r}")
        if not 0.0 <= self.curled_threshold < self.extended_threshold <= 1.0:
            raise ValueError(
                "ClassifierConfig requires "
                f"0.0 <= curled_threshold < extended_threshold <= 1.0, "
                f"got curled_threshold={self.curled_threshold!r} and "
                f"extended_threshold={self.extended_threshold!r}"
            )
        if not 0.0 <= self.pinch_close_threshold < self.pinch_open_threshold:
            raise ValueError(
                "ClassifierConfig requires "
                f"0.0 <= pinch_close_threshold < pinch_open_threshold, "
                f"got pinch_close_threshold={self.pinch_close_threshold!r} "
                f"and pinch_open_threshold={self.pinch_open_threshold!r}"
            )
        for name, value in (("min_score", self.min_score), ("min_margin", self.min_margin)):
            if not 0.0 <= value <= MAX_CONFIDENCE:
                raise ValueError(
                    f"ClassifierConfig.{name} must be within [0.0, 1.0], got {value!r}"
                )
        if self.strong_margin <= 0.0:
            raise ValueError(
                f"ClassifierConfig.strong_margin must be > 0, got {self.strong_margin!r}"
            )


@dataclass(frozen=True)
class GestureClassification:
    """One frame's verdict: which gesture, how strongly, and against what.

    Attributes:
        gesture: The reported label. :attr:`Gesture.UNKNOWN` when nothing matched
            convincingly.
        confidence: Strength of that label, in ``[0.0, 1.0]``. See this module's
            docstring for how to read it, and in particular for why a confident
            ``UNKNOWN`` is a success.
        margin: How far the winning rule outscored the runner-up. Reported separately
            because it is the half of the confidence that a caller can act on: a high
            score with no margin means two gestures are equally plausible, which is the
            case where an action is most dangerous and no action is safest.
        scores: The raw evidence for every recognised gesture, keyed by gesture. Read it
            to see which alternative was close; it is the same numbers the rules
            computed, kept instead of discarded so a classification can be explained
            after the fact.
    """

    gesture: Gesture
    confidence: float
    margin: float
    scores: Mapping[Gesture, float]

    def __post_init__(self) -> None:
        if not isinstance(self.gesture, Gesture):
            raise TypeError(f"gesture must be a Gesture, got {type(self.gesture).__name__}")
        scores = MappingProxyType(dict(self.scores))
        for name, value in (("confidence", self.confidence), ("margin", self.margin)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"GestureClassification.{name} must be a real number, got {value!r}"
                )
            if not math.isfinite(value):
                raise ValueError(f"GestureClassification.{name} must be finite, got {value!r}")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"GestureClassification.confidence must be within [0.0, 1.0], "
                f"got {self.confidence!r}"
            )
        if not 0.0 <= self.margin <= MAX_CONFIDENCE:
            raise ValueError(
                f"GestureClassification.margin must be within [0.0, 1.0], got {self.margin!r}"
            )
        if set(scores) != set(RECOGNIZABLE_GESTURES):
            raise ValueError(
                f"GestureClassification.scores must cover exactly {list(RECOGNIZABLE_GESTURES)}, "
                f"got {list(scores)}"
            )
        for gesture, score in scores.items():
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise TypeError(f"score for {gesture.name} must be a real number, got {score!r}")
            if not math.isfinite(score):
                raise ValueError(f"score for {gesture.name} must be finite, got {score!r}")
            if not MIN_CONFIDENCE <= score <= MAX_CONFIDENCE:
                raise ValueError(
                    f"score for {gesture.name} must be within [0.0, 1.0], got {score!r}"
                )
        expected_margin = self._score_gap(scores)
        if abs(self.margin - expected_margin) > _MARGIN_TOLERANCE:
            raise ValueError(
                f"GestureClassification.margin is {self.margin!r} but its scores differ by "
                f"{expected_margin!r}"
            )
        if self.gesture.is_recognized:
            best = self.ranked(scores)[0][0]
            if self.gesture is not best:
                raise ValueError(
                    f"GestureClassification reports {self.gesture.name} but "
                    f"{best.name} scores highest"
                )
        object.__setattr__(self, "scores", scores)

    @staticmethod
    def ranked(scores: Mapping[Gesture, float]) -> tuple[tuple[Gesture, float], ...]:
        """The scores ordered best first, with deterministic tie-breaking.

        Ties resolve by :data:`RECOGNIZABLE_GESTURES` order rather than by dictionary
        order, so two identical poses always produce the same label — a coin flip that
        is invisible in the output is still a coin flip.
        """
        order = {gesture: index for index, gesture in enumerate(RECOGNIZABLE_GESTURES)}
        return tuple(sorted(scores.items(), key=lambda item: (-item[1], order[item[0]])))

    @classmethod
    def _score_gap(cls, scores: Mapping[Gesture, float]) -> float:
        """The gap between the best and second-best scores.

        Assumes ``scores`` covers every recognisable gesture, which
        :meth:`__post_init__` has already established before this runs.
        """
        ranked = cls.ranked(scores)
        return max(0.0, ranked[0][1] - ranked[1][1])

    @property
    def is_recognized(self) -> bool:
        """Whether a gesture was recognised, as opposed to ``UNKNOWN``."""
        return self.gesture.is_recognized

    @property
    def runner_up(self) -> Gesture | None:
        """The highest-scoring gesture that was not reported, or ``None``.

        ``None`` only when no rule scored above zero at all, meaning the hand matched
        nothing even partly and there is no candidate to run second.
        """
        ranked = self.ranked(self.scores)
        if self.gesture.is_recognized:
            return ranked[1][0]
        return ranked[1][0] if ranked[1][1] > 0.0 else None

    def score_for(self, gesture: Gesture) -> float:
        """The raw evidence score for ``gesture``.

        Raises:
            KeyError: If ``gesture`` is :attr:`Gesture.UNKNOWN`, which has no score of
                its own.
        """
        if not isinstance(gesture, Gesture):
            raise TypeError(f"gesture must be a Gesture, got {type(gesture).__name__}")
        return self.scores[gesture]

    def __str__(self) -> str:
        return f"{self.gesture.name} at {self.confidence:.2f} (margin {self.margin:.2f})"


@dataclass(frozen=True)
class HandMeasurements:
    """The evidence every rule is computed from, kept so a decision can be explained.

    Attributes:
        extensions: Extension ratio per finger, in ``[0.0, 1.0]``, where ``1.0`` is a
            perfectly straight digit. Read one with :meth:`extension`.
        thumb_index_gap: Image-plane distance between the thumb tip and the index tip in
            hand-scale units, so ``0.0`` is the fingertips touching and ``1.0`` is one
            palm length apart.
    """

    extensions: Mapping[Finger, float]
    thumb_index_gap: float

    def __post_init__(self) -> None:
        extensions = MappingProxyType(dict(self.extensions))
        if set(extensions) != set(FINGERS):
            raise ValueError(
                f"HandMeasurements.extensions must cover every {list(FINGERS)}, "
                f"got {list(extensions)}"
            )
        for finger, ratio in extensions.items():
            if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
                raise TypeError(f"extension for {finger.name} must be a real number, got {ratio!r}")
            if not math.isfinite(ratio):
                raise ValueError(f"extension for {finger.name} must be finite, got {ratio!r}")
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    f"extension for {finger.name} must be within [0.0, 1.0], got {ratio!r}"
                )
        if not isinstance(self.thumb_index_gap, (int, float)) or isinstance(
            self.thumb_index_gap, bool
        ):
            raise TypeError(
                f"HandMeasurements.thumb_index_gap must be a real number, "
                f"got {self.thumb_index_gap!r}"
            )
        if not math.isfinite(self.thumb_index_gap):
            raise ValueError(
                f"HandMeasurements.thumb_index_gap must be finite, got {self.thumb_index_gap!r}"
            )
        if self.thumb_index_gap < 0.0:
            raise ValueError(
                f"HandMeasurements.thumb_index_gap must be >= 0.0, got {self.thumb_index_gap!r}"
            )
        object.__setattr__(self, "extensions", extensions)

    def extension(self, finger: Finger) -> float:
        """The extension ratio for ``finger``.

        Raises:
            KeyError: If ``finger`` is not a member of :class:`Finger`.
        """
        try:
            return self.extensions[finger]
        except KeyError:
            raise KeyError(f"No extension ratio recorded for {finger}") from None

    def __str__(self) -> str:
        ratios = " ".join(f"{finger.name}={ratio:.2f}" for finger, ratio in self.extensions.items())
        return f"{ratios} gap={self.thumb_index_gap:.2f}"


__all__ = [
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "RECOGNIZABLE_GESTURES",
    "ClassifierConfig",
    "Gesture",
    "GestureClassification",
    "HandMeasurements",
]
