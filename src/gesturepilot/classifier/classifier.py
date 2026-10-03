"""The static gesture classifier: one frame of geometry in, one verdict out.

How the rules work
------------------
Two kinds of evidence are derived from
:class:`~gesturepilot.processing.types.HandFeatures`, and both are ratios rather than
raw distances, so where the hand sits in frame and how large it appears cancel out.

**Extension ratio**, one per digit, in ``[0.0, 1.0]``::

    extension = |base -> tip| / (length of that digit's own bones)

The numerator is the straight-line reach from the digit's first joint to its tip; the
denominator is the summed length of the three bones in between, so the ratio is ``1.0``
when the digit is straight and falls as the digit folds back on itself. The wrist-root
segment is excluded on purpose: it belongs to the palm, not the digit, and including it
would make the ratio depend on how long each finger happens to be. Every landmark comes
from :data:`~gesturepilot.processing.topology.FINGER_CHAINS`, so no index is written out
by hand, and the distance itself is the processing layer's own
:func:`~gesturepilot.processing.geometry.distance_2d`.

**Thumb–index gap** is the image-plane distance between the two fingertips in
hand-scale units, taken from
:attr:`~gesturepilot.processing.types.HandFeatures.normalized_landmarks`. Image plane
only, for the same reason the joint angles are: the tracker's ``z`` is too noisy for a
distance that decides whether two fingertips are touching.

A digit's extension score is a **ramp**, not a comparison. It reaches ``1.0`` at
``extended_threshold``, falls to ``0.0`` at ``curled_threshold``, and is linear between.
That is deliberate on two counts. It means a half-curled finger contributes ``0.5``
rather than a yes or no, so the margin between candidate gestures degrades smoothly
instead of falling off a cliff at a threshold. And it means the reported confidence
tracks how far past the threshold the pose actually is, which is what makes confidence
worth reading.

Each gesture is then the **weakest** of its required conditions — a ``min``, not a
product — so a score reads as "how close was the least convincing requirement", and one
collapsed finger cannot be averaged away by four perfect ones:

=============  ====================================================================
Gesture        Conditions that must all hold
=============  ====================================================================
OPEN_PALM      index, middle, ring, pinky extended, and the thumb extended
FIST           index, middle, ring, pinky folded, and the thumb folded
POINT          index extended; middle, ring, pinky folded
TWO_FINGERS    index and middle extended; ring and pinky folded
PINCH          thumb tip within reach of the index tip; middle, ring, pinky folded
=============  ====================================================================

The thumb is part of ``OPEN_PALM`` and ``FIST`` but deliberately absent from ``POINT``,
``TWO_FINGERS``, and ``PINCH``. Those three are about which *fingers* are up, and a real
pointing hand holds its thumb wherever is comfortable — tucked along the palm, or spread
for balance. Requiring one position would reject hands their owner would call a point.
For ``PINCH`` the thumb enters through the gap instead, which is the thing that actually
separates a pinch from a point: in both, the index is up and the other fingers are
down, and the only real difference is where the thumb tip ended up.

Deciding between them
---------------------
The highest-scoring rule wins, but a win alone is not enough. The label is ``UNKNOWN``
unless the score reaches ``min_score`` **and** the lead over the runner-up reaches
``min_margin`` — a pose that satisfies ``POINT`` and ``TWO_FINGERS`` equally is a hand in
the middle of a movement, and naming it either one would be a coin flip presented as a
fact. Handedness plays no part: it may be inverted by the tracking layer (see
``docs/tracking.md``), and the index knuckle is a geometry-chosen reference, so neither
:attr:`~gesturepilot.processing.types.HandFeatures.handedness` nor anything else here is
read.

What this cannot do
-------------------
Static classification is limited in ways that belong in the design notes rather than in a
workaround. See ``docs/classifier.md``: a finger pointing at the camera foreshortens into
a straight-looking one, ``PINCH`` and ``FIST`` genuinely overlap when a thumb crosses the
index tip, a hand seen edge-on defeats any image-plane measurement, and this classifier
has no memory, so it will happily name a different gesture on two consecutive frames of a
hand that is still moving.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from gesturepilot.classifier.errors import InvalidFeaturesError
from gesturepilot.classifier.types import (
    MAX_CONFIDENCE,
    ClassifierConfig,
    Gesture,
    GestureClassification,
    HandMeasurements,
)
from gesturepilot.processing.geometry import distance_2d
from gesturepilot.processing.topology import FINGER_CHAINS, FINGERS, Finger, FingerChain
from gesturepilot.processing.types import HandFeatures

#: Digit chain lookup by finger, so no landmark index is ever written out by hand.
_CHAIN_BY_FINGER: Mapping[Finger, FingerChain] = {chain.finger: chain for chain in FINGER_CHAINS}

#: The three non-index fingers that point, two-finger, and pinch all require folded.
_FOLDED_ALONG_INDEX: tuple[Finger, ...] = (Finger.MIDDLE, Finger.RING, Finger.PINKY)

#: The highest landmark index any finger chain references. Lets a truncated landmark
#: array be rejected without hard-coding 21 anywhere in this package.
_HIGHEST_CHAIN_LANDMARK: int = max(max(chain.polyline) for chain in FINGER_CHAINS)


def _clamp(value: float, low: float = 0.0, high: float = MAX_CONFIDENCE) -> float:
    """``value`` confined to ``[low, high]``."""
    return low if value < low else high if value > high else value


def _ramp(value: float, low: float, high: float) -> float:
    """``0.0`` at or below ``low``, ``1.0`` at or above ``high``, linear between."""
    return _clamp((value - low) / (high - low))


def _require_number(value: object, description: str) -> float:
    """Return ``value`` as a float, or raise :class:`InvalidFeaturesError`.

    ``bool`` is rejected explicitly: it is an ``int`` in Python, so ``True`` would
    otherwise sail through as the number ``1.0`` and quietly become a length or an angle.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidFeaturesError(f"{description} is not a real number, got {value!r}")
    if not math.isfinite(value):
        raise InvalidFeaturesError(f"{description} must be finite, got {value!r}")
    return float(value)


def _extension_ratio(features: HandFeatures, finger: Finger) -> float:
    """How straight ``finger`` is, in ``[0.0, 1.0]``.

    A digit whose three bones have no length at all cannot be straightened, and is
    reported as fully folded. That is hypothetical rather than a real pose — the
    processing layer already refuses a hand with collapsed bones — but dividing by zero
    here would turn a hand into an exception, and "as folded as it can be" is both true
    and harmless.
    """
    chain = _CHAIN_BY_FINGER[finger]
    bone_length = sum(features.finger(finger).segment_lengths[1:])
    if bone_length <= 0.0:
        return 0.0
    reach = distance_2d(
        features.normalized(chain.joints[0]).as_tuple,
        features.normalized(chain.tip).as_tuple,
    )
    return _clamp(reach / bone_length)


def _validate_landmarks(features: HandFeatures) -> None:
    """Reject a landmark array that no rule could be evaluated against."""
    landmarks = features.normalization.landmarks
    if len(landmarks) <= _HIGHEST_CHAIN_LANDMARK:
        raise InvalidFeaturesError(
            f"HandFeatures.normalization.landmarks has {len(landmarks)} entries, too few to "
            f"hold landmark {_HIGHEST_CHAIN_LANDMARK} that the finger chains reference"
        )
    for index, point in enumerate(landmarks):
        for axis in ("x", "y", "z"):
            _require_number(getattr(point, axis, None), f"normalized_landmarks[{index}].{axis}")


def _validate_fingers(features: HandFeatures) -> None:
    """Reject finger measurements that do not line up with the topology."""
    if len(features.fingers) != len(FINGERS):
        raise InvalidFeaturesError(
            f"HandFeatures.fingers has {len(features.fingers)} entries, "
            f"expected one per {list(FINGERS)}"
        )
    for expected, geometry in zip(FINGERS, features.fingers, strict=True):
        if not isinstance(geometry.finger, Finger):
            raise InvalidFeaturesError(
                f"HandFeatures.fingers entry {expected.name} has a non-Finger "
                f"identity {geometry.finger!r}"
            )
        if geometry.finger is not expected:
            raise InvalidFeaturesError(
                f"HandFeatures.fingers is expected in {list(FINGERS)} order, found "
                f"{geometry.finger.name} where {expected.name} was due"
            )
        lengths = geometry.segment_lengths
        if len(lengths) != 4:
            raise InvalidFeaturesError(
                f"HandFeatures.finger({expected.name}).segment_lengths has {len(lengths)} "
                "entries, expected 4"
            )
        for position, length in enumerate(lengths):
            value = _require_number(
                length, f"HandFeatures.finger({expected.name}).segment_lengths[{position}]"
            )
            if value < 0.0:
                raise InvalidFeaturesError(
                    f"HandFeatures.finger({expected.name}).segment_lengths[{position}] must be "
                    f">= 0.0, got {length!r}"
                )


def _measure(features: HandFeatures) -> HandMeasurements:
    """Derive every piece of evidence from ``features``, validating as it goes."""
    _validate_landmarks(features)
    _validate_fingers(features)
    thumb_tip = _CHAIN_BY_FINGER[Finger.THUMB].tip
    index_tip = _CHAIN_BY_FINGER[Finger.INDEX].tip
    return HandMeasurements(
        extensions={finger: _extension_ratio(features, finger) for finger in FINGERS},
        thumb_index_gap=distance_2d(
            features.normalized(thumb_tip).as_tuple,
            features.normalized(index_tip).as_tuple,
        ),
    )


class GestureClassifier:
    """Rule-based recognition of static hand configurations.

    Stateless and deterministic: the same features always produce the same verdict, and
    nothing is remembered between calls. That is what makes every decision reproducible
    from a single input, and it is why there is no ``reset()`` here even though the
    camera and tracking layers have one — this classifier owns no time-varying state.

    Typical use::

        from gesturepilot.classifier import GestureClassifier

        classifier = GestureClassifier()
        result = classifier.classify(features)

        if result.is_recognized:
            print(result.gesture, result.confidence)

    :meth:`classify` holds no state, so calling it from several threads is fine. What is
    not fine is sharing a mutable threshold configuration with other code, which is why
    :class:`ClassifierConfig` is frozen.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        """Build a classifier.

        Args:
            config: Thresholds to use. Defaults to :class:`ClassifierConfig`'s, which is
                biased toward ``UNKNOWN`` over a wrong guess.
        """
        if config is None:
            config = ClassifierConfig()
        elif not isinstance(config, ClassifierConfig):
            raise TypeError(f"config must be a ClassifierConfig, got {type(config).__name__}")
        self._config = config

    @property
    def config(self) -> ClassifierConfig:
        """The thresholds this classifier was built with."""
        return self._config

    def measure(self, features: HandFeatures) -> HandMeasurements:
        """The evidence this classifier would use for ``features``, without deciding.

        Exposed so a decision can be inspected, debugged, or reused by a later stage
        without re-deriving it. Performs the same validation as :meth:`classify`.

        Raises:
            InvalidFeaturesError: If ``features`` is not a readable
                :class:`~gesturepilot.processing.types.HandFeatures`.
        """
        if not isinstance(features, HandFeatures):
            raise InvalidFeaturesError(
                f"features must be a HandFeatures, got {type(features).__name__}"
            )
        return _measure(features)

    def classify(self, features: HandFeatures) -> GestureClassification:
        """Name the gesture ``features`` shows, or ``UNKNOWN`` if nothing is convincing.

        ``features`` is read, never modified.

        Raises:
            InvalidFeaturesError: If ``features`` is not a readable
                :class:`~gesturepilot.processing.types.HandFeatures`. Features this
                classifier can measure but not name are **not** an error — they come
                back as :attr:`~gesturepilot.classifier.types.Gesture.UNKNOWN`.
        """
        measurements = self.measure(features)
        scores = self._score(measurements)
        ranked = GestureClassification.ranked(scores)
        best, best_score = ranked[0]
        margin = best_score - ranked[1][1]
        if best_score < self._config.min_score or margin < self._config.min_margin:
            gesture = Gesture.UNKNOWN
            confidence = 1.0 - best_score
        else:
            gesture = best
            confidence = best_score * (0.5 + 0.5 * min(1.0, margin / self._config.strong_margin))
        return GestureClassification(
            gesture=gesture,
            confidence=_clamp(confidence),
            margin=margin,
            scores=scores,
        )

    def _extension_score(self, measurements: HandMeasurements, finger: Finger) -> float:
        """``1.0`` when ``finger`` is straight, ``0.0`` when folded, linear between."""
        return _ramp(
            measurements.extension(finger),
            self._config.curled_threshold,
            self._config.extended_threshold,
        )

    def _touching_score(self, measurements: HandMeasurements) -> float:
        """``1.0`` when the thumb and index tips touch, ``0.0`` when clearly apart."""
        return 1.0 - _ramp(
            measurements.thumb_index_gap,
            self._config.pinch_close_threshold,
            self._config.pinch_open_threshold,
        )

    def _score(self, measurements: HandMeasurements) -> Mapping[Gesture, float]:
        """Score every recognised gesture as the weakest condition it requires."""
        extended = {finger: self._extension_score(measurements, finger) for finger in FINGERS}
        folded = {finger: 1.0 - extended[finger] for finger in FINGERS}
        return {
            Gesture.OPEN_PALM: min(extended.values()),
            Gesture.FIST: min(folded.values()),
            Gesture.POINT: min(
                extended[Finger.INDEX],
                *(folded[finger] for finger in _FOLDED_ALONG_INDEX),
            ),
            Gesture.TWO_FINGERS: min(
                extended[Finger.INDEX],
                extended[Finger.MIDDLE],
                folded[Finger.RING],
                folded[Finger.PINKY],
            ),
            Gesture.PINCH: self._touching_score(measurements)
            * min(folded[finger] for finger in _FOLDED_ALONG_INDEX),
        }


__all__ = ["GestureClassifier", "HandMeasurements"]
