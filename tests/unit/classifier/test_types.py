"""Validation and invariant tests for the classifier's domain types.

These are the contracts a caller can rely on without reading the rules: a label is from a
closed set, a confidence is a real number in a stated range, a threshold that could not
produce a sensible answer is rejected where it was written, and a classification is
internally consistent — the gesture it names really is the one that scored highest, and
the margin it reports really is the gap between the top two scores.

The last pair of invariants is what makes :class:`GestureClassification` safe to hand
around. A result that claimed ``POINT`` while ``TWO_FINGERS`` scored higher would be
worse than no result at all, and the position where that mistake could be made is when
something constructs a result by hand rather than through the classifier.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from gesturepilot.classifier.types import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    RECOGNIZABLE_GESTURES,
    ClassifierConfig,
    Gesture,
    GestureClassification,
    HandMeasurements,
)
from gesturepilot.processing.topology import FINGERS, Finger


def _scores(**overrides: float) -> dict[Gesture, float]:
    """A score map where one gesture leads and the rest are flat at zero."""
    scores = dict.fromkeys(RECOGNIZABLE_GESTURES, 0.0)
    for gesture, value in overrides.items():
        scores[Gesture[gesture]] = value
    return scores


class TestGesture:
    def test_only_unknown_is_unrecognised(self) -> None:
        unrecognised = [gesture for gesture in Gesture if not gesture.is_recognized]
        assert unrecognised == [Gesture.UNKNOWN]

    def test_recognisable_gestures_are_the_five_configurations(self) -> None:
        assert RECOGNIZABLE_GESTURES == (
            Gesture.OPEN_PALM,
            Gesture.FIST,
            Gesture.POINT,
            Gesture.PINCH,
            Gesture.TWO_FINGERS,
        )

    def test_unknown_has_no_score_of_its_own(self) -> None:
        assert Gesture.UNKNOWN not in RECOGNIZABLE_GESTURES

    @pytest.mark.parametrize(
        ("gesture", "expected"),
        [
            (Gesture.OPEN_PALM, "open_palm"),
            (Gesture.TWO_FINGERS, "two_fingers"),
            (Gesture.UNKNOWN, "unknown"),
        ],
    )
    def test_str_is_the_value(self, gesture: Gesture, expected: str) -> None:
        assert str(gesture) == expected

    def test_values_are_unique(self) -> None:
        values = [gesture.value for gesture in Gesture]
        assert len(set(values)) == len(values)


class TestClassifierConfig:
    def test_defaults_are_valid(self) -> None:
        config = ClassifierConfig()
        assert 0.0 <= config.curled_threshold < config.extended_threshold <= 1.0
        assert config.strong_margin > 0.0

    def test_is_frozen(self) -> None:
        config = ClassifierConfig()
        with pytest.raises(AttributeError):
            config.min_score = 0.1

    @pytest.mark.parametrize(
        "field",
        [
            "extended_threshold",
            "curled_threshold",
            "pinch_close_threshold",
            "pinch_open_threshold",
            "min_score",
            "min_margin",
            "strong_margin",
        ],
    )
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_thresholds_are_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            ClassifierConfig(**{field: value})

    @pytest.mark.parametrize("field", ["min_score", "strong_margin", "extended_threshold"])
    def test_non_numeric_thresholds_are_rejected(self, field: str) -> None:
        with pytest.raises(TypeError, match=field):
            ClassifierConfig(**{field: "high"})

    def test_a_boolean_is_not_a_threshold(self) -> None:
        with pytest.raises(TypeError, match="min_score"):
            ClassifierConfig(min_score=True)

    @pytest.mark.parametrize(
        ("curled", "extended"),
        [(0.9, 0.8), (0.5, 0.5), (-0.1, 0.9), (0.5, 1.1)],
    )
    def test_extension_window_must_be_a_real_window(self, curled: float, extended: float) -> None:
        with pytest.raises(ValueError, match="curled_threshold"):
            ClassifierConfig(curled_threshold=curled, extended_threshold=extended)

    @pytest.mark.parametrize(("close", "apart"), [(0.3, 0.3), (0.4, 0.2), (-0.1, 0.2)])
    def test_pinch_window_must_be_a_real_window(self, close: float, apart: float) -> None:
        with pytest.raises(ValueError, match="pinch_close_threshold"):
            ClassifierConfig(pinch_close_threshold=close, pinch_open_threshold=apart)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_recognition_thresholds_stay_in_the_unit_range(self, value: float) -> None:
        with pytest.raises(ValueError, match="min_score"):
            ClassifierConfig(min_score=value)
        with pytest.raises(ValueError, match="min_margin"):
            ClassifierConfig(min_margin=value)

    def test_strong_margin_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="strong_margin"):
            ClassifierConfig(strong_margin=0.0)

    def test_a_narrow_window_is_allowed_when_ordered(self) -> None:
        config = ClassifierConfig(curled_threshold=0.99, extended_threshold=1.0)
        assert config.extended_threshold == 1.0


class TestGestureClassification:
    def test_scores_become_read_only(self) -> None:
        result = GestureClassification(
            gesture=Gesture.POINT,
            confidence=0.9,
            margin=1.0,
            scores=_scores(POINT=1.0),
        )
        with pytest.raises(TypeError):
            result.scores[Gesture.FIST] = 0.5

    def test_is_recognized_follows_the_label(self) -> None:
        recognised = GestureClassification(
            gesture=Gesture.POINT, confidence=0.9, margin=1.0, scores=_scores(POINT=1.0)
        )
        unknown = GestureClassification(
            gesture=Gesture.UNKNOWN, confidence=0.9, margin=0.5, scores=_scores(POINT=0.5)
        )
        assert recognised.is_recognized
        assert not unknown.is_recognized

    def test_runner_up_names_the_nearest_rival(self) -> None:
        result = GestureClassification(
            gesture=Gesture.TWO_FINGERS,
            confidence=0.8,
            margin=0.3,
            scores=_scores(TWO_FINGERS=1.0, POINT=0.7),
        )
        assert result.runner_up is Gesture.POINT

    def test_runner_up_is_none_when_nothing_scored(self) -> None:
        result = GestureClassification(
            gesture=Gesture.UNKNOWN, confidence=1.0, margin=0.0, scores=_scores()
        )
        assert result.runner_up is None

    def test_score_for_rejects_unknown(self) -> None:
        result = GestureClassification(
            gesture=Gesture.UNKNOWN, confidence=1.0, margin=0.0, scores=_scores()
        )
        with pytest.raises(KeyError):
            result.score_for(Gesture.UNKNOWN)

    def test_score_for_rejects_a_non_gesture(self) -> None:
        result = GestureClassification(
            gesture=Gesture.UNKNOWN, confidence=1.0, margin=0.0, scores=_scores()
        )
        with pytest.raises(TypeError, match="Gesture"):
            result.score_for("point")

    def test_str_summarises_the_verdict(self) -> None:
        result = GestureClassification(
            gesture=Gesture.PINCH, confidence=0.75, margin=1.0, scores=_scores(PINCH=1.0)
        )
        assert str(result) == "PINCH at 0.75 (margin 1.00)"

    def test_a_label_that_did_not_win_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="scores highest"):
            GestureClassification(
                gesture=Gesture.POINT,
                confidence=0.9,
                margin=0.3,
                scores=_scores(POINT=0.4, TWO_FINGERS=0.7),
            )

    def test_a_margin_that_does_not_match_the_scores_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="margin"):
            GestureClassification(
                gesture=Gesture.POINT,
                confidence=0.9,
                margin=0.9,
                scores=_scores(POINT=1.0, TWO_FINGERS=0.7),
            )

    @pytest.mark.parametrize("missing", [Gesture.POINT, Gesture.PINCH])
    def test_scores_must_cover_every_recognisable_gesture(self, missing: Gesture) -> None:
        scores = _scores(POINT=1.0)
        scores.pop(missing)
        with pytest.raises(ValueError, match="scores must cover"):
            GestureClassification(
                gesture=Gesture.UNKNOWN, confidence=0.0, margin=1.0, scores=scores
            )

    def test_an_extra_gesture_in_the_scores_is_rejected(self) -> None:
        scores = _scores(POINT=1.0)
        scores[Gesture.UNKNOWN] = 0.5
        with pytest.raises(ValueError, match="scores must cover"):
            GestureClassification(
                gesture=Gesture.UNKNOWN, confidence=0.0, margin=0.5, scores=scores
            )

    @pytest.mark.parametrize("field", ["confidence", "margin"])
    @pytest.mark.parametrize("value", ["high", True, None])
    def test_confidence_and_margin_must_be_real_numbers(self, field: str, value: object) -> None:
        valid = GestureClassification(
            gesture=Gesture.UNKNOWN, confidence=0.5, margin=0.0, scores=_scores()
        )
        with pytest.raises(TypeError, match=field):
            dataclasses.replace(valid, **{field: value})

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan])
    def test_confidence_is_bounded_and_finite(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            GestureClassification(
                gesture=Gesture.UNKNOWN,
                confidence=confidence,
                margin=0.0,
                scores=_scores(),
            )

    @pytest.mark.parametrize("margin", [-0.1, 1.1, math.inf])
    def test_margin_is_bounded_and_finite(self, margin: float) -> None:
        with pytest.raises(ValueError, match="margin"):
            GestureClassification(
                gesture=Gesture.UNKNOWN, confidence=0.0, margin=margin, scores=_scores()
            )

    def test_a_gesture_is_required(self) -> None:
        with pytest.raises(TypeError, match="gesture"):
            GestureClassification(
                gesture="point",
                confidence=0.0,
                margin=0.0,
                scores=_scores(),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, "high"])
    def test_scores_must_be_finite_numbers_in_range(self, score: object) -> None:
        with pytest.raises((TypeError, ValueError), match="POINT"):
            GestureClassification(
                gesture=Gesture.UNKNOWN,
                confidence=0.0,
                margin=0.0,
                scores=_scores(POINT=score),
            )

    def test_ties_resolve_to_the_declared_rule_order(self) -> None:
        tied = _scores()
        for gesture in RECOGNIZABLE_GESTURES:
            tied[gesture] = 1.0
        ranked = GestureClassification.ranked(tied)
        assert [gesture for gesture, _ in ranked] == list(RECOGNIZABLE_GESTURES)

    def test_ranking_is_highest_first(self) -> None:
        ranked = GestureClassification.ranked(_scores(FIST=0.2, POINT=0.9, PINCH=0.5))
        assert ranked[:3] == ((Gesture.POINT, 0.9), (Gesture.PINCH, 0.5), (Gesture.FIST, 0.2))
        assert all(score == 0.0 for _, score in ranked[3:])

    def test_confidence_bounds_are_the_documented_ones(self) -> None:
        assert (MIN_CONFIDENCE, MAX_CONFIDENCE) == (0.0, 1.0)


class TestHandMeasurements:
    def test_valid_measurements_are_accepted(self) -> None:
        measurements = HandMeasurements(extensions=dict.fromkeys(FINGERS, 0.5), thumb_index_gap=0.2)
        assert measurements.extension(Finger.INDEX) == 0.5
        assert measurements.thumb_index_gap == 0.2

    def test_extensions_become_read_only(self) -> None:
        measurements = HandMeasurements(extensions=dict.fromkeys(FINGERS, 0.5), thumb_index_gap=0.2)
        with pytest.raises(TypeError):
            measurements.extensions[Finger.INDEX] = 1.0

    def test_every_finger_must_be_covered(self) -> None:
        with pytest.raises(ValueError, match="must cover every"):
            HandMeasurements(extensions={Finger.INDEX: 1.0}, thumb_index_gap=0.2)

    @pytest.mark.parametrize("ratio", [-0.1, 1.1, math.nan, "straight"])
    def test_extensions_must_be_finite_numbers_in_range(self, ratio: object) -> None:
        with pytest.raises((TypeError, ValueError), match="INDEX"):
            HandMeasurements(
                extensions={**dict.fromkeys(FINGERS, 0.5), Finger.INDEX: ratio},
                thumb_index_gap=0.2,
            )

    @pytest.mark.parametrize("gap", [-0.1, math.inf, None])
    def test_gap_must_be_a_finite_non_negative_number(self, gap: object) -> None:
        with pytest.raises((TypeError, ValueError), match="thumb_index_gap"):
            HandMeasurements(extensions=dict.fromkeys(FINGERS, 0.5), thumb_index_gap=gap)

    def test_extension_rejects_an_unknown_finger(self) -> None:
        measurements = HandMeasurements(extensions=dict.fromkeys(FINGERS, 0.5), thumb_index_gap=0.2)
        with pytest.raises(KeyError, match="recorded"):
            measurements.extension("index")

    def test_str_lists_every_ratio_and_the_gap(self) -> None:
        measurements = HandMeasurements(
            extensions=dict.fromkeys(FINGERS, 0.5), thumb_index_gap=0.25
        )
        text = str(measurements)
        assert "INDEX=0.50" in text
        assert text.endswith("gap=0.25")
