"""Unit tests for the landmark-processing domain types.

Every output type is frozen and validated on construction, so the invariants are pinned
here: finiteness, landmark counts, angle ranges, and score ranges. These are the checks
that keep a ``NaN`` or a half-built feature set from reaching a future classifier.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from gesturepilot.processing.errors import DegenerateHandError
from gesturepilot.processing.normalization import normalize_hand
from gesturepilot.processing.topology import FINGERS, Finger, LandmarkIndex
from gesturepilot.processing.types import (
    MAX_ANGLE_RADIANS,
    MAX_ROLL_RADIANS,
    FilterConfig,
    FingerGeometry,
    HandFeatures,
    NormalizedHand,
    NormalizedLandmark,
    PalmOrientation,
    ProcessedLandmark,
    ProcessorConfig,
)
from gesturepilot.tracking.types import LANDMARK_COUNT, Handedness

from .fixtures import make_processed


def make_features(**overrides: object) -> HandFeatures:
    normalized = normalize_hand(make_processed())
    defaults: dict[str, object] = {
        "landmarks": make_processed(),
        "normalization": normalized,
        "fingers": tuple(
            FingerGeometry(Finger(finger), 0.5, 0.6, 0.7, (1.0, 1.0, 1.0, 1.0), 2.0, 1.5)
            for finger in FINGERS
        ),
        "orientation": PalmOrientation(roll=0.1, tilt=-0.2),
    }
    defaults.update(overrides)
    return HandFeatures(**defaults)  # type: ignore[arg-type]


class TestProcessedLandmark:
    def test_stores_values(self) -> None:
        landmark = ProcessedLandmark(x=0.1, y=0.2, z=0.3)
        assert landmark.as_tuple == (0.1, 0.2, 0.3)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_components(self, bad: float) -> None:
        for kwargs in ({"x": bad}, {"y": bad}, {"z": bad}):
            with pytest.raises(ValueError, match="finite"):
                ProcessedLandmark(**{"x": 0.0, "y": 0.0, "z": 0.0, **kwargs})

    def test_allows_coordinates_outside_the_unit_range(self) -> None:
        """A landmark at the edge of the frame can legitimately sit slightly outside."""
        landmark = ProcessedLandmark(x=-0.02, y=1.03, z=-0.4)
        assert landmark.x < 0.0
        assert landmark.y > 1.0

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            ProcessedLandmark(x=0.0, y=0.0, z=0.0).x = 1.0  # type: ignore[misc]


class TestNormalizedLandmark:
    def test_stores_values(self) -> None:
        assert NormalizedLandmark(x=0.5, y=-0.5, z=0.0).as_tuple == (0.5, -0.5, 0.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_components(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            NormalizedLandmark(x=bad, y=0.0, z=0.0)

    def test_allows_values_beyond_one_hand_scale(self) -> None:
        """A fingertip can sit more than one palm length from the wrist."""
        assert NormalizedLandmark(x=2.5, y=-3.0, z=0.0).x == 2.5

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            NormalizedLandmark(x=0.0, y=0.0, z=0.0).y = 1.0  # type: ignore[misc]


class TestNormalizedHand:
    def test_requires_the_full_landmark_count(self) -> None:
        normalized = normalize_hand(make_processed())
        with pytest.raises(ValueError, match="exactly 21 landmarks"):
            NormalizedHand(
                origin=normalized.origin,
                scale=normalized.scale,
                landmarks=normalized.landmarks[:5],
                palm_center=normalized.palm_center,
            )

    @pytest.mark.parametrize("bad", [0.0, -0.1])
    def test_rejects_a_non_positive_scale(self, bad: float) -> None:
        normalized = normalize_hand(make_processed())
        with pytest.raises(ValueError, match="scale must be > 0"):
            NormalizedHand(
                origin=normalized.origin,
                scale=bad,
                landmarks=normalized.landmarks,
                palm_center=normalized.palm_center,
            )

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_a_non_finite_scale(self, bad: float) -> None:
        normalized = normalize_hand(make_processed())
        with pytest.raises(ValueError, match="finite"):
            NormalizedHand(
                origin=normalized.origin,
                scale=bad,
                landmarks=normalized.landmarks,
                palm_center=normalized.palm_center,
            )

    def test_landmark_accessor(self) -> None:
        normalized = normalize_hand(make_processed())
        assert normalized.landmark(LandmarkIndex.WRIST) is normalized.landmarks[0]

    def test_landmark_accessor_out_of_range(self) -> None:
        normalized = normalize_hand(make_processed())
        with pytest.raises(IndexError):
            normalized.landmark(LANDMARK_COUNT)

    def test_is_frozen(self) -> None:
        normalized = normalize_hand(make_processed())
        with pytest.raises(dataclasses.FrozenInstanceError):
            normalized.scale = 1.0  # type: ignore[misc]


class TestFingerGeometry:
    def test_accepts_undefined_angles(self) -> None:
        """A collapsed joint is a real pose, so ``None`` must be constructible."""
        geometry = FingerGeometry(Finger.INDEX, None, None, None, (1.0, 1.0, 1.0, 1.0), 0.0, 0.0)
        assert geometry.is_fully_measured is False

    def test_is_fully_measured_when_every_angle_exists(self) -> None:
        geometry = FingerGeometry(Finger.INDEX, 0.1, 0.2, 0.3, (1.0, 1.0, 1.0, 1.0), 0.0, 0.0)
        assert geometry.is_fully_measured is True

    @pytest.mark.parametrize("angle", [-0.001, MAX_ANGLE_RADIANS + 0.001])
    def test_rejects_angles_outside_the_valid_range(self, angle: float) -> None:
        with pytest.raises(ValueError, match="within"):
            FingerGeometry(Finger.INDEX, angle, 0.2, 0.3, (1.0, 1.0, 1.0, 1.0), 0.0, 0.0)

    def test_accepts_the_range_boundaries(self) -> None:
        for angle in (0.0, MAX_ANGLE_RADIANS):
            FingerGeometry(Finger.INDEX, angle, 0.0, 0.0, (1.0, 1.0, 1.0, 1.0), 0.0, 0.0)

    def test_rejects_a_non_finite_angle(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FingerGeometry(Finger.INDEX, math.nan, 0.2, 0.3, (1.0, 1.0, 1.0, 1.0), 0.0, 0.0)

    def test_requires_exactly_four_segment_lengths(self) -> None:
        with pytest.raises(ValueError, match="exactly 4 entries"):
            FingerGeometry(Finger.INDEX, 0.1, 0.2, 0.3, (1.0, 1.0, 1.0), 0.0, 0.0)

    @pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
    def test_rejects_an_invalid_segment_length(self, bad: float) -> None:
        with pytest.raises(ValueError, match="segment_lengths"):
            FingerGeometry(Finger.INDEX, 0.1, 0.2, 0.3, (1.0, 1.0, 1.0, bad), 0.0, 0.0)

    @pytest.mark.parametrize("field", ["tip_to_palm", "tip_to_index_mcp"])
    def test_rejects_a_negative_tip_distance(self, field: str) -> None:
        distances = {"tip_to_palm": 0.0, "tip_to_index_mcp": 0.0, field: -1.0}
        with pytest.raises(ValueError, match=field):
            FingerGeometry(Finger.INDEX, 0.1, 0.2, 0.3, (1.0, 1.0, 1.0, 1.0), **distances)

    def test_accepts_zero_tip_distances(self) -> None:
        FingerGeometry(Finger.INDEX, 0.1, 0.2, 0.3, (1.0, 1.0, 1.0, 1.0), 0.0, 0.0)

    def test_is_frozen(self) -> None:
        geometry = FingerGeometry(Finger.INDEX, 0.1, 0.2, 0.3, (1.0, 1.0, 1.0, 1.0), 0.0, 0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            geometry.finger = Finger.THUMB  # type: ignore[misc]


class TestPalmOrientation:
    def test_accepts_undefined_components(self) -> None:
        assert PalmOrientation(roll=None, tilt=None).is_fully_measured is False

    def test_is_fully_measured_when_both_exist(self) -> None:
        assert PalmOrientation(roll=0.1, tilt=0.2).is_fully_measured is True

    def test_roll_is_folded_into_a_half_turn(self) -> None:
        """Roll measures an axis, so it can never exceed a quarter turn either way."""
        for bad in (-MAX_ROLL_RADIANS - 0.001, MAX_ROLL_RADIANS + 0.001):
            with pytest.raises(ValueError, match="roll must be within"):
                PalmOrientation(roll=bad, tilt=0.0)

    def test_roll_accepts_its_boundaries(self) -> None:
        for good in (-MAX_ROLL_RADIANS, MAX_ROLL_RADIANS):
            PalmOrientation(roll=good, tilt=0.0)

    @pytest.mark.parametrize("bad", [-MAX_ANGLE_RADIANS - 0.001, MAX_ANGLE_RADIANS + 0.001])
    def test_rejects_tilt_outside_the_valid_range(self, bad: float) -> None:
        """Tilt is a directed axis, so it keeps the full half-turn-plus range."""
        with pytest.raises(ValueError, match="tilt must be within"):
            PalmOrientation(roll=0.0, tilt=bad)

    def test_tilt_accepts_its_boundaries(self) -> None:
        for good in (-MAX_ANGLE_RADIANS, MAX_ANGLE_RADIANS):
            PalmOrientation(roll=0.0, tilt=good)

    @pytest.mark.parametrize("field", ["roll", "tilt"])
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_a_non_finite_angle(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError, match=field):
            PalmOrientation(**{"roll": 0.0, "tilt": 0.0, field: bad})

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            PalmOrientation(roll=0.0, tilt=0.0).roll = 1.0  # type: ignore[misc]


class TestProcessorConfig:
    def test_smoothing_is_off_by_default(self) -> None:
        config = ProcessorConfig()
        assert config.smoothing is False
        assert config.filter == FilterConfig()

    def test_rejects_a_filter_of_the_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="FilterConfig"):
            ProcessorConfig(filter=FilterConfig().min_cutoff)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            ProcessorConfig().smoothing = True  # type: ignore[misc]


class TestHandFeatures:
    def test_exposes_the_normalised_view_without_copying_it(self) -> None:
        features = make_features()
        assert features.normalized_landmarks is features.normalization.landmarks
        assert features.palm_center is features.normalization.palm_center
        assert features.hand_scale == features.normalization.scale

    def test_wrist_accessors(self) -> None:
        features = make_features()
        assert features.wrist == features.landmark(LandmarkIndex.WRIST)
        assert features.wrist_normalized == features.normalized(LandmarkIndex.WRIST)
        assert features.wrist_normalized == NormalizedLandmark(x=0.0, y=0.0, z=0.0)

    def test_finger_accessor(self) -> None:
        features = make_features()
        assert features.finger(Finger.MIDDLE).finger is Finger.MIDDLE

    def test_finger_accessor_rejects_a_non_finger(self) -> None:
        with pytest.raises(KeyError):
            make_features().finger("index")  # type: ignore[arg-type]

    def test_index_accessors_out_of_range(self) -> None:
        features = make_features()
        with pytest.raises(IndexError):
            features.landmark(LANDMARK_COUNT)
        with pytest.raises(IndexError):
            features.normalized(LANDMARK_COUNT)

    def test_requires_the_full_landmark_count(self) -> None:
        features = make_features()
        with pytest.raises(ValueError, match="exactly 21 landmarks"):
            HandFeatures(
                landmarks=features.landmarks[:5],
                normalization=features.normalization,
                fingers=features.fingers,
                orientation=features.orientation,
            )

    def test_requires_one_measurement_per_finger(self) -> None:
        features = make_features()
        with pytest.raises(ValueError, match="exactly 5 finger measurements"):
            HandFeatures(
                landmarks=features.landmarks,
                normalization=features.normalization,
                fingers=features.fingers[:2],
                orientation=features.orientation,
            )

    def test_rejects_a_duplicated_finger(self) -> None:
        features = make_features()
        duplicated = (features.fingers[0],) + features.fingers[:-1]
        with pytest.raises(ValueError, match="every Finger exactly once"):
            HandFeatures(
                landmarks=features.landmarks,
                normalization=features.normalization,
                fingers=duplicated,
                orientation=features.orientation,
            )

    def test_covers_every_declared_finger(self) -> None:
        assert len(FINGERS) == 5

    @pytest.mark.parametrize("field", ["handedness_score", "detection_score"])
    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan])
    def test_rejects_an_invalid_score(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError, match=field):
            make_features(**{field: bad})

    def test_accepts_missing_scores(self) -> None:
        features = make_features(handedness_score=None, detection_score=None)
        assert features.handedness_score is None
        assert features.detection_score is None

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_a_non_finite_timestamp(self, bad: float) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            make_features(timestamp=bad)

    def test_rejects_a_negative_frame_sequence(self) -> None:
        with pytest.raises(ValueError, match="frame_sequence"):
            make_features(frame_sequence=-1)

    def test_metadata_is_optional(self) -> None:
        features = make_features()
        assert features.timestamp is None
        assert features.frame_sequence is None
        assert features.handedness is Handedness.UNKNOWN

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_features().timestamp = 1.0  # type: ignore[misc]


class TestDegenerateHandErrorIsUsable:
    def test_is_a_processing_error(self) -> None:
        from gesturepilot.processing.errors import ProcessingError

        assert issubclass(DegenerateHandError, ProcessingError)
