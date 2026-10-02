"""Unit tests for :class:`LandmarkProcessor`.

The processor is the whole public surface of this phase, so the contracts tested here are
the ones a later stage will depend on: determinism, immutability of the input, metadata
propagation, and honest behaviour when the hand is unusable.

Nothing here needs a webcam, a model, or a sleeping call — every hand is synthetic and
every timestamp is a literal.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from gesturepilot.processing.errors import (
    DegenerateHandError,
    FilterStateError,
    InvalidLandmarksError,
    ProcessingError,
)
from gesturepilot.processing.processor import LandmarkProcessor
from gesturepilot.processing.topology import FINGERS, Finger, LandmarkIndex
from gesturepilot.processing.types import (
    FilterConfig,
    HandFeatures,
    ProcessedLandmark,
    ProcessorConfig,
)
from gesturepilot.tracking.types import LANDMARK_COUNT, Handedness, Landmark, TrackedHand

from .fixtures import (
    HAND_COORDINATES,
    HAND_SCALE,
    collapsed,
    finger_curl,
    make_hand,
    make_landmarks,
    mirrored,
    scaled,
    translated,
    with_collapsed_joint,
)

FRAME_STEP = 1.0 / 30.0

#: The middle knuckle: a landmark whose position moves clearly when the hand is resized.
PROBE = LandmarkIndex.MIDDLE_MCP


def smoothing_processor(**filter_kwargs: float) -> LandmarkProcessor:
    return LandmarkProcessor(ProcessorConfig(smoothing=True, filter=FilterConfig(**filter_kwargs)))


def landmark_values(landmarks: tuple[object, ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple((lm.x, lm.y, lm.z) for lm in landmarks)  # type: ignore[attr-defined]


def normalised_values(features: HandFeatures) -> tuple[tuple[float, float, float], ...]:
    return tuple(landmark.as_tuple for landmark in features.normalized_landmarks)


def flat_normalised(features: HandFeatures) -> tuple[float, ...]:
    """Flatten the normalised landmarks so ``pytest.approx`` can compare them.

    ``approx`` rejects nested sequences, so a per-coordinate check has to be flattened
    into one long sequence of plain floats.
    """
    return tuple(value for landmark in normalised_values(features) for value in landmark)


def finger_values(features: HandFeatures) -> tuple[float, ...]:
    """Flatten every finger measurement into one plain float sequence.

    ``pytest.approx`` cannot compare nested sequences, so a per-coordinate comparison has
    to be spelled out as a single flat sequence. ``None`` angles are dropped rather than
    encoded, which is safe here because every assertion using this helper is on a hand
    whose angles are all defined — :class:`TestFoldedHands` checks the ``None`` case
    directly.
    """
    values: list[float] = []
    for finger in features.fingers:
        for angle in (finger.proximal_angle, finger.middle_angle, finger.distal_angle):
            assert angle is not None, "finger_values() is only valid for fully measured hands"
            values.append(angle)
        values.extend(finger.segment_lengths)
        values.extend((finger.tip_to_palm, finger.tip_to_index_mcp))
    return tuple(values)


def bypass_validation(landmarks: tuple[Landmark, ...]) -> TrackedHand:
    """Build a real ``TrackedHand`` whose landmarks were set past their own validation.

    :class:`Landmark` and :class:`TrackedHand` both refuse malformed data. That is the
    behaviour worth testing separately — but the processor must not *rely* on it, because
    a backend or a caller could hand it a substituted object. This builds exactly that.
    """
    hand = TrackedHand(landmarks=make_landmarks(), handedness=Handedness.LEFT)
    object.__setattr__(hand, "landmarks", landmarks)
    return hand


def with_raw_coordinate(index: int, *, x: float, y: float, z: float) -> TrackedHand:
    landmarks = list(make_landmarks())
    bypassed = object.__new__(Landmark)
    object.__setattr__(bypassed, "x", x)
    object.__setattr__(bypassed, "y", y)
    object.__setattr__(bypassed, "z", z)
    landmarks[index] = bypassed
    return bypass_validation(tuple(landmarks))


class TestConfiguration:
    def test_defaults_to_no_smoothing(self) -> None:
        processor = LandmarkProcessor()
        assert processor.config == ProcessorConfig()
        assert processor.is_filtering is False

    def test_reports_when_smoothing_is_enabled(self) -> None:
        assert smoothing_processor().is_filtering is True

    def test_rejects_a_configuration_of_the_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="ProcessorConfig"):
            LandmarkProcessor(config=True)  # type: ignore[arg-type]

    def test_starts_with_no_filter_state(self) -> None:
        assert LandmarkProcessor().active_slots == ()


class TestDeterminism:
    def test_identical_inputs_give_identical_features(self) -> None:
        processor = LandmarkProcessor()
        assert processor.process(make_hand()) == processor.process(make_hand())

    def test_repeated_processing_does_not_drift(self) -> None:
        """Without smoothing the processor must hold no state at all."""
        processor = LandmarkProcessor()
        first = processor.process(make_hand())
        for _ in range(20):
            again = processor.process(make_hand())
        assert again == first

    def test_a_fresh_processor_matches_a_used_one(self) -> None:
        used = LandmarkProcessor()
        for _ in range(5):
            used.process(make_hand())
        assert LandmarkProcessor().process(make_hand()) == used.process(make_hand())


class TestInputImmutability:
    def test_the_source_hand_is_untouched(self) -> None:
        hand = make_hand()
        before = landmark_values(hand.landmarks)
        LandmarkProcessor().process(hand)
        assert landmark_values(hand.landmarks) == before

    def test_the_source_landmarks_are_the_same_objects(self) -> None:
        hand = make_hand()
        before = tuple(id(landmark) for landmark in hand.landmarks)
        features = LandmarkProcessor().process(hand)

        assert tuple(id(landmark) for landmark in hand.landmarks) == before
        assert features.landmarks is not hand.landmarks

    def test_processed_landmarks_are_frozen_so_they_cannot_be_written_back(self) -> None:
        features = LandmarkProcessor().process(make_hand())
        with pytest.raises(dataclasses.FrozenInstanceError):
            features.landmarks[0].x = 0.0  # type: ignore[misc]

    def test_smoothing_does_not_write_through_to_the_source(self) -> None:
        hand = make_hand()
        before = landmark_values(hand.landmarks)
        processor = smoothing_processor()
        processor.process(hand, timestamp=0.0)
        processor.process(
            make_hand(coordinates=scaled(HAND_COORDINATES, 1.3)), timestamp=FRAME_STEP
        )
        assert landmark_values(hand.landmarks) == before


class TestMetadata:
    @pytest.mark.parametrize("handedness", list(Handedness))
    def test_supports_every_handedness_label(self, handedness: Handedness) -> None:
        features = LandmarkProcessor().process(make_hand(handedness=handedness))
        assert features.handedness is handedness
        assert features.finger(Finger.INDEX).is_fully_measured

    def test_passes_the_scores_through(self) -> None:
        features = LandmarkProcessor().process(
            make_hand(handedness_score=0.87, detection_score=0.42)
        )
        assert features.handedness_score == pytest.approx(0.87)
        assert features.detection_score == pytest.approx(0.42)

    def test_missing_scores_stay_missing(self) -> None:
        features = LandmarkProcessor().process(
            make_hand(handedness_score=None, detection_score=None)
        )
        assert features.handedness_score is None
        assert features.detection_score is None

    def test_passes_the_timestamp_and_sequence_through(self) -> None:
        features = LandmarkProcessor().process(make_hand(), timestamp=12.5, frame_sequence=41)
        assert features.timestamp == pytest.approx(12.5)
        assert features.frame_sequence == 41

    def test_frame_metadata_is_optional(self) -> None:
        features = LandmarkProcessor().process(make_hand(handedness=Handedness.UNKNOWN))
        assert features.timestamp is None
        assert features.frame_sequence is None
        assert features.handedness is Handedness.UNKNOWN

    def test_frame_metadata_does_not_change_the_geometry(self) -> None:
        plain = LandmarkProcessor().process(make_hand())
        annotated = LandmarkProcessor().process(make_hand(), timestamp=3.0, frame_sequence=9)

        assert annotated.normalized_landmarks == plain.normalized_landmarks
        assert annotated.fingers == plain.fingers
        assert annotated.orientation == plain.orientation


class TestGeometry:
    def test_reports_the_hand_scale(self) -> None:
        assert LandmarkProcessor().process(make_hand()).hand_scale == pytest.approx(HAND_SCALE)

    def test_wrist_is_the_origin(self) -> None:
        assert LandmarkProcessor().process(make_hand()).normalized(
            LandmarkIndex.WRIST
        ).as_tuple == (
            0.0,
            0.0,
            0.0,
        )

    def test_measures_every_finger_once_in_canonical_order(self) -> None:
        features = LandmarkProcessor().process(make_hand())
        assert tuple(finger.finger for finger in features.fingers) == FINGERS

    def test_reference_hand_palm_points_up_the_frame(self) -> None:
        orientation = LandmarkProcessor().process(make_hand()).orientation
        assert orientation.tilt == pytest.approx(0.029403288204005135)
        assert orientation.is_fully_measured

    def test_reference_hand_roll_is_a_small_angle_across_the_frame(self) -> None:
        assert LandmarkProcessor().process(make_hand()).orientation.roll == pytest.approx(
            0.08490179344972208
        )

    def test_segment_lengths_are_positive_for_an_extended_finger(self) -> None:
        index = LandmarkProcessor().process(make_hand()).finger(Finger.INDEX)
        assert all(length > 0.0 for length in index.segment_lengths)

    def test_every_angle_is_finite_for_a_well_posed_hand(self) -> None:
        for finger in LandmarkProcessor().process(make_hand()).fingers:
            assert finger.is_fully_measured
            for angle in (finger.proximal_angle, finger.middle_angle, finger.distal_angle):
                assert angle is not None and math.isfinite(angle)

    def test_output_carries_the_full_landmark_count(self) -> None:
        features = LandmarkProcessor().process(make_hand())
        assert len(features.landmarks) == LANDMARK_COUNT
        assert len(features.normalized_landmarks) == LANDMARK_COUNT

    def test_every_landmark_is_a_processed_landmark(self) -> None:
        features = LandmarkProcessor().process(make_hand())
        assert all(isinstance(landmark, ProcessedLandmark) for landmark in features.landmarks)


class TestPositionAndSizeInvariance:
    def test_moving_the_hand_leaves_the_features_alone(self) -> None:
        base = LandmarkProcessor().process(make_hand())
        moved = LandmarkProcessor().process(
            make_hand(coordinates=translated(HAND_COORDINATES, 0.12, -0.2))
        )
        assert flat_normalised(moved) == pytest.approx(flat_normalised(base), abs=1e-12)
        assert finger_values(moved) == pytest.approx(finger_values(base), abs=1e-9)

    @pytest.mark.parametrize("factor", [0.35, 2.5])
    def test_resizing_the_hand_leaves_the_features_alone(self, factor: float) -> None:
        base = LandmarkProcessor().process(make_hand())
        resized = LandmarkProcessor().process(
            make_hand(coordinates=scaled(HAND_COORDINATES, factor))
        )
        assert flat_normalised(resized) == pytest.approx(flat_normalised(base), abs=1e-9)
        assert finger_values(resized) == pytest.approx(finger_values(base), abs=1e-9)

    def test_the_raw_scale_still_tracks_apparent_size(self) -> None:
        small = LandmarkProcessor().process(make_hand(coordinates=scaled(HAND_COORDINATES, 0.5)))
        large = LandmarkProcessor().process(make_hand(coordinates=scaled(HAND_COORDINATES, 2.0)))
        assert large.hand_scale == pytest.approx(4.0 * small.hand_scale)


class TestHandednessIsMetadataOnly:
    def test_the_label_does_not_change_the_geometry(self) -> None:
        """Nothing may branch on handedness, so the label cannot move a single number."""
        by_label = {
            handedness: LandmarkProcessor().process(make_hand(handedness=handedness))
            for handedness in Handedness
        }
        reference = by_label[Handedness.LEFT]
        for features in by_label.values():
            assert features.normalized_landmarks == reference.normalized_landmarks
            assert features.fingers == reference.fingers
            assert features.orientation == reference.orientation

    def test_a_mirrored_hand_produces_mirrored_coordinates(self) -> None:
        """Handedness never triggers a flip, so a mirrored input measures as mirrored."""
        base = LandmarkProcessor().process(make_hand())
        flipped = LandmarkProcessor().process(make_hand(coordinates=mirrored(HAND_COORDINATES)))

        for original, mirror in zip(
            base.normalized_landmarks, flipped.normalized_landmarks, strict=True
        ):
            assert mirror.x == pytest.approx(-original.x, abs=1e-12)
            assert mirror.y == pytest.approx(original.y, abs=1e-12)

    def test_mirroring_preserves_all_lengths_and_angles(self) -> None:
        base = LandmarkProcessor().process(make_hand())
        flipped = LandmarkProcessor().process(make_hand(coordinates=mirrored(HAND_COORDINATES)))

        for original, mirror in zip(base.fingers, flipped.fingers, strict=True):
            assert mirror.segment_lengths == pytest.approx(original.segment_lengths)
            assert mirror.tip_to_palm == pytest.approx(original.tip_to_palm)
            assert mirror.tip_to_index_mcp == pytest.approx(original.tip_to_index_mcp)
            assert mirror.proximal_angle == pytest.approx(original.proximal_angle)

    def test_mirroring_negates_the_orientation_angles(self) -> None:
        base = LandmarkProcessor().process(make_hand())
        flipped = LandmarkProcessor().process(make_hand(coordinates=mirrored(HAND_COORDINATES)))

        assert flipped.orientation.roll == pytest.approx(-base.orientation.roll)
        assert flipped.orientation.tilt == pytest.approx(-base.orientation.tilt)


class TestDegenerateAndMalformedInput:
    def test_a_hand_with_no_measurable_size_is_refused(self) -> None:
        with pytest.raises(DegenerateHandError, match="Hand scale collapsed"):
            LandmarkProcessor().process(make_hand(coordinates=collapsed()))

    def test_a_non_hand_object_is_refused(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="Expected a TrackedHand"):
            LandmarkProcessor().process(HAND_COORDINATES)  # type: ignore[arg-type]

    def test_a_short_hand_that_bypassed_validation_is_refused(self) -> None:
        hand = bypass_validation(make_landmarks()[:5])
        with pytest.raises(InvalidLandmarksError, match="exactly 21 landmarks"):
            LandmarkProcessor().process(hand)

    def test_a_short_hand_is_refused_by_its_own_domain_type(self) -> None:
        with pytest.raises(ValueError, match="exactly 21 landmarks"):
            TrackedHand(landmarks=make_landmarks()[:5])

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_coordinate_is_refused(self, bad: float) -> None:
        """The processor re-checks, rather than trusting the upstream type to have done so."""
        with pytest.raises(InvalidLandmarksError, match="non-finite x"):
            LandmarkProcessor().process(with_raw_coordinate(1, x=bad, y=0.5, z=0.0))

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_a_non_finite_timestamp_is_refused(self, bad: float) -> None:
        with pytest.raises(InvalidLandmarksError, match="Timestamp must be finite"):
            LandmarkProcessor().process(make_hand(), timestamp=bad)

    def test_a_negative_frame_sequence_is_refused(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="frame_sequence"):
            LandmarkProcessor().process(make_hand(), frame_sequence=-1)

    def test_a_non_handedness_label_is_refused(self) -> None:
        hand = TrackedHand(landmarks=make_landmarks(), handedness=Handedness.LEFT)
        object.__setattr__(hand, "handedness", "left")
        with pytest.raises(InvalidLandmarksError, match="handedness"):
            LandmarkProcessor().process(hand)


class TestFoldedHands:
    def test_a_collapsed_tip_reports_an_undefined_angle_instead_of_failing(self) -> None:
        features = LandmarkProcessor().process(make_hand(coordinates=finger_curl()))
        index = features.finger(Finger.INDEX)

        assert index.is_fully_measured is False
        assert index.distal_angle is None

    def test_the_other_fingers_are_unaffected(self) -> None:
        features = LandmarkProcessor().process(make_hand(coordinates=finger_curl()))
        assert features.finger(Finger.THUMB).is_fully_measured is True

    def test_a_folded_hand_still_reports_finite_distances(self) -> None:
        features = LandmarkProcessor().process(make_hand(coordinates=finger_curl()))
        for finger in features.fingers:
            assert math.isfinite(finger.tip_to_palm)
            assert math.isfinite(finger.tip_to_index_mcp)

    def test_a_single_collapsed_joint_does_not_break_the_hand(self) -> None:
        features = LandmarkProcessor().process(
            make_hand(coordinates=with_collapsed_joint(LandmarkIndex.INDEX_DIP))
        )
        assert features.finger(Finger.INDEX).middle_angle is None
        assert features.finger(Finger.INDEX).is_fully_measured is False
        assert features.finger(Finger.MIDDLE).is_fully_measured is True

    @pytest.mark.parametrize(
        "coordinates",
        [HAND_COORDINATES, finger_curl(), with_collapsed_joint(7)],
        ids=["reference", "curled-fingers", "one-collapsed-joint"],
    )
    def test_no_nan_or_infinity_ever_escapes(self, coordinates: tuple) -> None:
        features = LandmarkProcessor().process(make_hand(coordinates=coordinates))
        for landmark in (*features.normalized_landmarks, features.palm_center):
            assert math.isfinite(landmark.x)
            assert math.isfinite(landmark.y)
            assert math.isfinite(landmark.z)
        assert math.isfinite(features.hand_scale)


class TestSmoothing:
    def test_smoothing_requires_a_timestamp(self) -> None:
        with pytest.raises(ProcessingError, match="monotonic timestamp"):
            smoothing_processor().process(make_hand())

    def test_the_first_frame_is_measured_unchanged(self) -> None:
        hand = make_hand()
        features = smoothing_processor().process(hand, timestamp=0.0)
        assert landmark_values(features.landmarks) == landmark_values(hand.landmarks)

    def test_later_frames_are_pulled_back_towards_the_history(self) -> None:
        processor = smoothing_processor(min_cutoff=0.5, beta=0.0)
        base = make_hand()
        resting = landmark_values(base.landmarks)[PROBE][1]
        processor.process(base, timestamp=0.0)
        jumped = make_hand(coordinates=scaled(HAND_COORDINATES, 1.4))
        smoothed = processor.process(jumped, timestamp=FRAME_STEP)

        raw_step = jumped.landmarks[PROBE].y - resting
        applied = smoothed.landmarks[PROBE].y - resting

        # The output moves towards the new measurement without ever reaching it.
        assert applied * raw_step > 0.0
        assert 0.0 < abs(applied) < abs(raw_step)

    def test_smoothing_is_deterministic(self) -> None:
        hands = [make_hand(coordinates=scaled(HAND_COORDINATES, 1.0 + 0.05 * i)) for i in range(6)]

        def run() -> list[tuple[tuple[float, float, float], ...]]:
            processor = smoothing_processor(min_cutoff=1.0, beta=0.3)
            return [
                landmark_values(processor.process(hand, timestamp=i * FRAME_STEP).landmarks)
                for i, hand in enumerate(hands)
            ]

        assert run() == run()

    def test_a_non_increasing_timestamp_is_rejected(self) -> None:
        processor = smoothing_processor()
        processor.process(make_hand(), timestamp=1.0)
        with pytest.raises(FilterStateError, match="strictly increase"):
            processor.process(make_hand(), timestamp=1.0)

    def test_the_error_names_the_slot_and_suggests_a_reset(self) -> None:
        processor = smoothing_processor()
        processor.process(make_hand(handedness=Handedness.LEFT), timestamp=1.0)
        with pytest.raises(FilterStateError, match="reset"):
            processor.process(make_hand(handedness=Handedness.LEFT), timestamp=0.5)

    def test_separate_hands_get_separate_filter_state(self) -> None:
        processor = smoothing_processor(min_cutoff=0.5, beta=0.0)
        left = make_hand(handedness=Handedness.LEFT)

        first = processor.process(left, timestamp=0.0)
        assert processor.active_slots == (Handedness.LEFT,)

        # The right hand's slot is untouched, so its first frame passes straight through
        # even though the left slot has already filtered a frame at the same timestamp.
        right = processor.process(make_hand(handedness=Handedness.RIGHT), timestamp=0.0)
        assert right.landmarks == first.landmarks
        assert processor.active_slots == (Handedness.LEFT, Handedness.RIGHT)

        # Both slots now hold the same single frame of history, so the same next frame
        # must produce the same output in either slot — no state has leaked between them.
        moved = [
            make_hand(coordinates=scaled(HAND_COORDINATES, 1.3), handedness=handedness)
            for handedness in (Handedness.LEFT, Handedness.RIGHT)
        ]
        from_left = processor.process(moved[0], timestamp=FRAME_STEP)
        from_right = processor.process(moved[1], timestamp=FRAME_STEP)

        assert from_left.landmarks == from_right.landmarks
        assert from_left.landmarks != moved[0].landmarks

    def test_active_slots_are_reported_in_declaration_order(self) -> None:
        processor = smoothing_processor()
        processor.process(make_hand(handedness=Handedness.RIGHT), timestamp=0.0)
        processor.process(make_hand(handedness=Handedness.LEFT), timestamp=0.0)
        assert processor.active_slots == (Handedness.LEFT, Handedness.RIGHT)

    def test_reset_discards_every_slot(self) -> None:
        processor = smoothing_processor()
        processor.process(make_hand(handedness=Handedness.LEFT), timestamp=0.0)
        processor.process(make_hand(handedness=Handedness.RIGHT), timestamp=0.0)
        processor.reset()

        assert processor.active_slots == ()

    def test_reset_discards_one_slot_and_keeps_the_other(self) -> None:
        processor = smoothing_processor()
        processor.process(make_hand(handedness=Handedness.LEFT), timestamp=0.0)
        processor.process(make_hand(handedness=Handedness.RIGHT), timestamp=0.0)
        processor.reset(Handedness.LEFT)

        assert processor.active_slots == (Handedness.RIGHT,)

    def test_a_reset_slot_starts_from_a_fresh_passthrough(self) -> None:
        processor = smoothing_processor()
        hand = make_hand(handedness=Handedness.LEFT)
        processor.process(hand, timestamp=0.0)
        processor.reset(Handedness.LEFT)
        features = processor.process(hand, timestamp=FRAME_STEP)

        assert landmark_values(features.landmarks) == landmark_values(hand.landmarks)

    def test_a_reset_processor_reproduces_the_original_trajectory(self) -> None:
        hands = [make_hand(coordinates=scaled(HAND_COORDINATES, 1.0 + 0.05 * i)) for i in range(4)]
        times = [index * FRAME_STEP for index in range(4)]

        def run(processor: LandmarkProcessor) -> list[tuple[tuple[float, float, float], ...]]:
            return [
                landmark_values(processor.process(hand, timestamp=time).landmarks)
                for hand, time in zip(hands, times, strict=True)
            ]

        processor = smoothing_processor(min_cutoff=1.0, beta=0.2)
        first = run(processor)
        assert processor.active_slots != ()

        processor.reset()
        assert processor.active_slots == ()
        assert run(processor) == first

    def test_reset_is_a_no_op_when_smoothing_is_off(self) -> None:
        processor = LandmarkProcessor()
        processor.reset()
        assert processor.active_slots == ()

    def test_reset_of_an_unused_slot_is_harmless(self) -> None:
        smoothing_processor().reset(Handedness.LEFT)
