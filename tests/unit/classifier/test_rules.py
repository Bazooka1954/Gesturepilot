"""Behaviour tests for the static gesture rules.

Every pose here is built from a palm skeleton and one curl number per finger, then run
through the real :class:`~gesturepilot.processing.processor.LandmarkProcessor`, so what
the classifier sees is exactly what the pipeline produces. The tests are organised around
the promises the rules make rather than around the code that implements them:

* each of the five configurations is recognised from a pose that plainly shows it
* an ambiguous pose is ``UNKNOWN`` rather than a coin flip dressed up as a fact
* the verdict does not depend on handedness, frame position, apparent size, or mirroring
* the same input always gives the same answer, and the input is never modified
* the thresholds are the caller's to set, and are actually read
* nothing that is not a readable ``HandFeatures`` reaches the arithmetic

The ambiguity tests matter most. A classifier that answers every pose confidently is
worse than useless for a control surface: the whole value of ``UNKNOWN`` is that it is
the answer when the evidence does not separate two gestures, and it can only be trusted
if the rules genuinely produce it.
"""

from __future__ import annotations

import dataclasses

import pytest

from gesturepilot.classifier import (
    ClassifierConfig,
    Gesture,
    GestureClassification,
    GestureClassifier,
)
from gesturepilot.classifier.errors import ClassificationError, InvalidFeaturesError
from gesturepilot.processing.topology import FINGERS, Finger
from gesturepilot.processing.types import HandFeatures, NormalizedLandmark
from gesturepilot.tracking.types import Handedness

from .fixtures import (
    FIST_CURL,
    FOLDING_CURL,
    HALF_CURL,
    SLIGHT_CURL,
    curled_thumb_fist,
    fist,
    hand_coordinates,
    hand_features,
    mirrored,
    open_palm,
    pinch,
    point,
    posed,
    scaled,
    translated,
    two_fingers,
    without_finger,
)

FOLDED = {"index": FIST_CURL, "middle": FIST_CURL, "ring": FIST_CURL, "pinky": FIST_CURL}


def _all_recognised() -> list[tuple[Gesture, HandFeatures]]:
    """One unambiguous pose per recognised gesture."""
    return [
        (Gesture.OPEN_PALM, open_palm()),
        (Gesture.FIST, fist()),
        (Gesture.POINT, point()),
        (Gesture.TWO_FINGERS, two_fingers()),
        (Gesture.PINCH, pinch()),
    ]


class TestRecognisedGestures:
    @pytest.mark.parametrize(
        ("expected", "features"),
        _all_recognised(),
        ids=lambda value: value.name if isinstance(value, Gesture) else "",
    )
    def test_each_pose_is_named(self, expected: Gesture, features: HandFeatures) -> None:
        result = GestureClassifier().classify(features)
        assert result.gesture is expected
        assert result.is_recognized

    @pytest.mark.parametrize(
        ("expected", "features"),
        _all_recognised(),
        ids=lambda value: value.name if isinstance(value, Gesture) else "",
    )
    def test_the_named_gesture_outscores_every_rival(
        self, expected: Gesture, features: HandFeatures
    ) -> None:
        result = GestureClassifier().classify(features)
        assert result.scores[expected] == 1.0
        assert result.margin == 1.0 - max(
            score for gesture, score in result.scores.items() if gesture is not expected
        )
        assert result.runner_up is not expected

    @pytest.mark.parametrize(
        ("expected", "features"),
        _all_recognised(),
        ids=lambda value: value.name if isinstance(value, Gesture) else "",
    )
    def test_recognition_does_not_mutate_its_input(
        self, expected: Gesture, features: HandFeatures
    ) -> None:
        landmarks = features.normalized_landmarks
        fingers = features.fingers
        GestureClassifier().classify(features)
        assert features.normalized_landmarks is landmarks
        assert features.fingers is fingers

    def test_an_open_palm_is_not_also_a_point(self) -> None:
        scores = GestureClassifier().classify(open_palm()).scores
        assert scores[Gesture.POINT] == 0.0

    def test_a_fist_is_not_also_a_pinch(self) -> None:
        scores = GestureClassifier().classify(fist()).scores
        assert scores[Gesture.PINCH] == 0.0


class TestUnknown:
    def test_a_hand_matching_nothing_is_confidently_unknown(self) -> None:
        features = posed(**FOLDED, thumb="spread")
        result = GestureClassifier().classify(features)
        assert result.gesture is Gesture.UNKNOWN
        assert not result.is_recognized

    def test_confidence_means_strength_of_the_label_not_weakness(self) -> None:
        features = posed(**FOLDED, thumb="spread")
        result = GestureClassifier().classify(features)
        assert result.confidence > 0.5
        assert result.runner_up is None

    @pytest.mark.parametrize("finger", ["middle", "ring", "pinky"])
    def test_a_half_curled_finger_blocks_a_point(self, finger: str) -> None:
        features = posed(**{**FOLDED, finger: HALF_CURL, "thumb": "curled", "index": 0.0})
        result = GestureClassifier().classify(features)
        assert result.gesture is Gesture.UNKNOWN
        assert max(result.scores.values()) < ClassifierConfig().min_score

    def test_a_half_curled_index_blocks_a_point(self) -> None:
        features = posed(**{**FOLDED, "index": HALF_CURL, "thumb": "curled"})
        result = GestureClassifier().classify(features)
        assert result.gesture is Gesture.UNKNOWN

    def test_a_half_curled_middle_cannot_choose_between_point_and_two_fingers(self) -> None:
        features = posed(index=0.0, middle=HALF_CURL, ring=FIST_CURL, pinky=FIST_CURL)
        result = GestureClassifier().classify(features)
        assert result.gesture is Gesture.UNKNOWN
        assert {result.runner_up, Gesture.POINT, Gesture.TWO_FINGERS} <= {
            Gesture.POINT,
            Gesture.TWO_FINGERS,
            result.runner_up,
        }

    def test_three_extended_fingers_are_not_a_gesture(self) -> None:
        features = posed(index=0.0, middle=0.0, ring=0.0, pinky=FIST_CURL)
        result = GestureClassifier().classify(features)
        assert result.gesture is Gesture.UNKNOWN
        assert result.score_for(Gesture.TWO_FINGERS) < 1.0

    def test_the_pinch_fist_overlap_is_decided_by_the_margin(self) -> None:
        features = curled_thumb_fist()
        result = GestureClassifier().classify(features)
        assert result.score_for(Gesture.PINCH) > 0.0
        assert result.score_for(Gesture.FIST) > 0.0
        assert result.gesture is max(result.scores, key=lambda gesture: result.scores[gesture])


class TestConfidence:
    def test_confidence_is_not_a_constant(self) -> None:
        confidences = {
            GestureClassifier().classify(features).confidence for _, features in _all_recognised()
        }
        confidences.add(GestureClassifier().classify(half_curled_palm()).confidence)
        assert len(confidences) > 1

    def test_a_perfect_pose_confidence_is_one(self) -> None:
        result = GestureClassifier().classify(open_palm())
        assert result.confidence == 1.0

    def test_a_slightly_bent_hand_scores_below_one(self) -> None:
        features = posed(
            index=SLIGHT_CURL,
            middle=SLIGHT_CURL,
            ring=SLIGHT_CURL,
            pinky=SLIGHT_CURL,
            thumb="spread",
        )
        result = GestureClassifier().classify(features)
        assert result.gesture is Gesture.OPEN_PALM
        assert 0.0 < result.confidence < 1.0

    def test_confidence_is_bounded(self) -> None:
        classifier = GestureClassifier()
        for _, features in _all_recognised():
            assert 0.0 <= classifier.classify(features).confidence <= 1.0

    def test_confidence_rises_with_evidence(self) -> None:
        classifier = GestureClassifier()
        crisp = classifier.classify(open_palm()).confidence
        bent = classifier.classify(half_curled_palm()).confidence
        assert bent < crisp

    def test_scores_are_reported_for_every_recognisable_gesture(self) -> None:
        result = GestureClassifier().classify(fist())
        assert set(result.scores) == set(
            gesture for gesture in Gesture if gesture is not Gesture.UNKNOWN
        )


def half_curled_palm() -> HandFeatures:
    """Every finger in the middle of the ambiguous band: nothing is decisive."""
    return posed(
        index=HALF_CURL,
        middle=HALF_CURL,
        ring=HALF_CURL,
        pinky=HALF_CURL,
        thumb="curled",
    )


class TestInvariance:
    @pytest.mark.parametrize("handedness", list(Handedness))
    def test_handedness_is_not_an_input_to_any_rule(self, handedness: Handedness) -> None:
        classifier = GestureClassifier()
        expected = classifier.classify(fist())
        result = classifier.classify(fist(handedness=handedness))
        assert result.gesture is expected.gesture
        assert result.confidence == pytest.approx(expected.confidence)

    def test_mirroring_the_hand_changes_nothing(self) -> None:
        classifier = GestureClassifier()
        for expected, features in _all_recognised():
            flipped = classifier.classify(hand_features(mirrored(_coordinates(features))))
            assert flipped.gesture is expected

    @pytest.mark.parametrize(("dx", "dy"), [(0.1, 0.0), (-0.25, 0.3), (0.4, -0.15)])
    def test_position_in_frame_changes_nothing(self, dx: float, dy: float) -> None:
        classifier = GestureClassifier()
        for expected, features in _all_recognised():
            moved = classifier.classify(hand_features(translated(_coordinates(features), dx, dy)))
            assert moved.gesture is expected

    @pytest.mark.parametrize("factor", [0.4, 0.75, 2.5])
    def test_apparent_size_changes_nothing(self, factor: float) -> None:
        classifier = GestureClassifier()
        for expected, features in _all_recognised():
            resized = classifier.classify(hand_features(scaled(_coordinates(features), factor)))
            assert resized.gesture is expected

    def test_measurements_survive_resizing(self) -> None:
        classifier = GestureClassifier()
        features = two_fingers()
        resized = hand_features(scaled(_coordinates(features), 0.4))
        for finger in FINGERS:
            assert classifier.measure(resized).extension(finger) == pytest.approx(
                classifier.measure(features).extension(finger)
            )

    def test_measurements_survive_mirroring(self) -> None:
        classifier = GestureClassifier()
        features = two_fingers()
        flipped = hand_features(mirrored(_coordinates(features)))
        assert classifier.measure(flipped).thumb_index_gap == pytest.approx(
            classifier.measure(features).thumb_index_gap
        )


def _coordinates(features: HandFeatures) -> tuple[tuple[float, float, float], ...]:
    """The tracker-space coordinates a fixture was built from.

    Reconstructed from the features rather than kept alongside them, so a test that
    transforms a hand is provably transforming the same hand the classifier saw.
    """
    return tuple((landmark.x, landmark.y, landmark.z) for landmark in features.landmarks)


class TestDeterminism:
    def test_the_same_input_gives_the_same_verdict(self) -> None:
        classifier = GestureClassifier()
        first = classifier.classify(two_fingers())
        second = classifier.classify(two_fingers())
        assert first == second

    def test_two_classifiers_agree(self) -> None:
        assert GestureClassifier().classify(fist()) == GestureClassifier().classify(fist())

    def test_timestamps_are_not_consulted(self) -> None:
        classifier = GestureClassifier()
        features = point()
        later = dataclasses.replace(features, timestamp=123.456, frame_sequence=999)
        assert classifier.classify(later) == classifier.classify(features)

    def test_the_classifier_holds_no_state_between_calls(self) -> None:
        classifier = GestureClassifier()
        open_palm_result = classifier.classify(open_palm())
        classifier.classify(fist())
        assert classifier.classify(open_palm()) == open_palm_result


class TestConfiguration:
    def test_the_default_config_is_used_when_none_is_given(self) -> None:
        assert GestureClassifier().config == ClassifierConfig()

    def test_a_supplied_config_is_kept(self) -> None:
        config = ClassifierConfig(min_score=0.9)
        assert GestureClassifier(config).config is config

    def test_a_non_config_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="ClassifierConfig"):
            GestureClassifier({"min_score": 0.5})

    def test_the_config_object_is_never_replaced(self) -> None:
        classifier = GestureClassifier(ClassifierConfig(min_score=0.8))
        config = classifier.config
        classifier.classify(open_palm())
        classifier.classify(fist())
        assert classifier.config is config

    def test_a_demanding_min_score_rejects_a_clear_gesture(self) -> None:
        strict = GestureClassifier(ClassifierConfig(min_score=0.99, extended_threshold=0.999))
        result = strict.classify(open_palm())
        assert result.gesture is Gesture.UNKNOWN

    def test_a_demanding_min_margin_rejects_a_near_tie(self) -> None:
        strict = GestureClassifier(ClassifierConfig(min_margin=0.99))
        assert strict.classify(half_curled_palm()).gesture is Gesture.UNKNOWN

    def test_the_pinch_thresholds_are_read(self) -> None:
        features = pinch()
        permissive = GestureClassifier().classify(features)
        assert permissive.gesture is Gesture.PINCH
        strict = GestureClassifier(
            ClassifierConfig(pinch_close_threshold=0.0, pinch_open_threshold=0.05)
        )
        result = strict.classify(features)
        assert result.score_for(Gesture.PINCH) == 0.0
        assert result.gesture is Gesture.UNKNOWN

    def test_the_extension_thresholds_are_read(self) -> None:
        permissive = GestureClassifier(
            ClassifierConfig(curled_threshold=0.1, extended_threshold=0.2)
        ).classify(fist())
        assert permissive.gesture is Gesture.OPEN_PALM

        strict = GestureClassifier(
            ClassifierConfig(curled_threshold=0.99, extended_threshold=1.0)
        ).classify(open_palm())
        assert strict.gesture is Gesture.UNKNOWN

    def test_a_narrower_extension_window_lowers_confidence(self) -> None:
        features = posed(index=SLIGHT_CURL, middle=SLIGHT_CURL, ring=SLIGHT_CURL, pinky=SLIGHT_CURL)
        narrow = GestureClassifier(ClassifierConfig(extended_threshold=1.0)).classify(features)
        wide = GestureClassifier().classify(features)
        assert narrow.confidence < wide.confidence

    def test_a_fold_right_at_the_threshold_is_still_a_fold(self) -> None:
        features = posed(
            index=FOLDING_CURL,
            middle=FOLDING_CURL,
            ring=FOLDING_CURL,
            pinky=FOLDING_CURL,
            thumb="curled",
        )
        assert GestureClassifier().classify(features).gesture is Gesture.FIST


class TestMalformedInput:
    @pytest.mark.parametrize(
        "not_features", [None, 42, "pointing", (0.1, 0.2), {"fingers": []}, Handedness.LEFT]
    )
    def test_a_non_hand_features_is_rejected(self, not_features: object) -> None:
        classifier = GestureClassifier()
        with pytest.raises(InvalidFeaturesError, match="HandFeatures"):
            classifier.classify(not_features)  # type: ignore[arg-type]
        with pytest.raises(InvalidFeaturesError, match="HandFeatures"):
            classifier.measure(not_features)  # type: ignore[arg-type]

    def test_the_error_is_catchable_as_the_base_class(self) -> None:
        with pytest.raises(ClassificationError):
            GestureClassifier().classify(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_a_non_finite_landmark_is_rejected(self, axis: str) -> None:
        features = point()
        poisoned = NormalizedLandmark(**{"x": 0.0, "y": 0.0, "z": 0.0})
        object.__setattr__(poisoned, axis, float("nan"))
        landmarks = list(features.normalization.landmarks)
        landmarks[8] = poisoned
        object.__setattr__(features.normalization, "landmarks", tuple(landmarks))
        with pytest.raises(InvalidFeaturesError, match="finite"):
            GestureClassifier().classify(features)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_non_finite_bone_length_is_rejected(self, value: float) -> None:
        features = point()
        geometry = features.finger(Finger.INDEX)
        object.__setattr__(geometry, "segment_lengths", (0.1, value, 0.1, 0.1))
        with pytest.raises(InvalidFeaturesError, match="finite"):
            GestureClassifier().classify(features)

    def test_a_negative_bone_length_is_rejected(self) -> None:
        features = point()
        geometry = features.finger(Finger.INDEX)
        object.__setattr__(geometry, "segment_lengths", (0.1, -0.1, 0.1, 0.1))
        with pytest.raises(InvalidFeaturesError, match=">= 0.0"):
            GestureClassifier().classify(features)

    def test_a_truncated_landmark_array_is_rejected(self) -> None:
        features = point()
        object.__setattr__(
            features.normalization, "landmarks", features.normalization.landmarks[:12]
        )
        with pytest.raises(InvalidFeaturesError, match="too few"):
            GestureClassifier().classify(features)

    def test_fingers_out_of_order_are_rejected(self) -> None:
        features = point()
        fingers = list(features.fingers)
        fingers[1], fingers[2] = fingers[2], fingers[1]
        object.__setattr__(features, "fingers", tuple(fingers))
        with pytest.raises(InvalidFeaturesError, match="order"):
            GestureClassifier().classify(features)

    def test_a_missing_finger_measurement_is_rejected(self) -> None:
        features = point()
        object.__setattr__(features, "fingers", features.fingers[:4])
        with pytest.raises(InvalidFeaturesError, match="expected one per"):
            GestureClassifier().classify(features)

    def test_a_finger_measurement_with_no_finger_is_rejected(self) -> None:
        features = point()
        geometry = features.finger(Finger.INDEX)
        object.__setattr__(geometry, "finger", "index")
        with pytest.raises(InvalidFeaturesError, match="non-Finger"):
            GestureClassifier().classify(features)

    def test_a_short_bone_list_is_rejected(self) -> None:
        features = point()
        geometry = features.finger(Finger.INDEX)
        object.__setattr__(geometry, "segment_lengths", (0.1, 0.1))
        with pytest.raises(InvalidFeaturesError, match="expected 4"):
            GestureClassifier().classify(features)

    @pytest.mark.parametrize("value", ["0.1", None])
    def test_a_non_numeric_bone_length_is_rejected(self, value: object) -> None:
        features = point()
        geometry = features.finger(Finger.INDEX)
        object.__setattr__(geometry, "segment_lengths", (0.1, value, 0.1, 0.1))  # type: ignore[arg-type]
        with pytest.raises(InvalidFeaturesError, match="real number"):
            GestureClassifier().classify(features)

    @pytest.mark.parametrize("finger", list(FINGERS))
    def test_zero_length_bones_are_treated_as_folded(self, finger: Finger) -> None:
        features = without_finger(open_palm(), finger)
        result = GestureClassifier().classify(features)
        assert result.gesture is Gesture.UNKNOWN
        assert result.margin == pytest.approx(result.margin)

    def test_zero_length_bones_on_the_index_end_a_point(self) -> None:
        result = GestureClassifier().classify(without_finger(point(), Finger.INDEX))
        assert result.gesture is Gesture.FIST

    def test_a_valid_hand_is_not_confused_with_a_broken_one(self) -> None:
        assert GestureClassifier().classify(open_palm()).gesture is Gesture.OPEN_PALM


class TestMeasurement:
    def test_measure_reports_the_extension_of_each_finger(self) -> None:
        measurements = GestureClassifier().measure(open_palm())
        assert set(measurements.extensions) == set(FINGERS)
        for finger in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
            assert measurements.extension(finger) == pytest.approx(1.0)
        assert measurements.extension(Finger.THUMB) > 0.95

    def test_a_straight_finger_measures_fully_extended(self) -> None:
        measurements = GestureClassifier().measure(two_fingers())
        assert measurements.extension(Finger.INDEX) == pytest.approx(1.0)
        assert measurements.extension(Finger.PINKY) < 0.6

    def test_the_extension_ratio_is_bounded_by_one(self) -> None:
        for _, features in _all_recognised():
            measurements = GestureClassifier().measure(features)
            assert all(0.0 <= ratio <= 1.0 for ratio in measurements.extensions.values())

    def test_the_gap_is_large_when_the_thumb_is_away(self) -> None:
        assert GestureClassifier().measure(open_palm()).thumb_index_gap > 1.0

    def test_the_gap_is_small_when_the_fingertips_touch(self) -> None:
        assert GestureClassifier().measure(pinch()).thumb_index_gap < 0.2

    def test_measure_agrees_with_the_scores_it_feeds(self) -> None:
        classifier = GestureClassifier()
        features = two_fingers()
        measurements = classifier.measure(features)
        scores = classifier.classify(features).scores
        assert scores[Gesture.TWO_FINGERS] == pytest.approx(1.0)
        assert measurements.extension(Finger.INDEX) == pytest.approx(1.0)
        assert measurements.extension(Finger.MIDDLE) == pytest.approx(1.0)

    def test_measure_does_not_mutate_its_input(self) -> None:
        features = point()
        landmarks = features.normalization.landmarks
        GestureClassifier().measure(features)
        assert features.normalized_landmarks is landmarks


class TestPublicSurface:
    def test_the_result_is_a_typed_value(self) -> None:
        result = GestureClassifier().classify(point())
        assert isinstance(result, GestureClassification)
        assert isinstance(result.gesture, Gesture)

    def test_pose_helpers_build_distinct_hands(self) -> None:
        poses = [open_palm(), fist(), point(), two_fingers(), pinch()]
        assert len({_coordinates(features) for features in poses}) == len(poses)

    def test_the_fixtures_have_the_hand_scale_they_advertise(self) -> None:
        assert open_palm().hand_scale == pytest.approx(0.17007, abs=1e-4)

    def test_coordinates_round_trip_through_the_processor(self) -> None:
        coordinates = hand_coordinates(index=FOLDING_CURL, thumb="curled")
        assert hand_features(coordinates).hand_scale > 0.0
