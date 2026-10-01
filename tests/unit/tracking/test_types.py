"""Unit tests for the hand-tracking domain types.

These types are the contract every later phase codes against, so the invariants are
pinned hard: finiteness, landmark count, ordering, and the "no hands is valid" rule.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from gesturepilot.tracking.types import (
    LANDMARK_COUNT,
    Handedness,
    Landmark,
    RunningMode,
    TrackedHand,
    TrackerConfig,
    TrackingResult,
)


def make_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> Landmark:
    return Landmark(x=x, y=y, z=z)


def make_hand(**kwargs: object) -> TrackedHand:
    defaults: dict[str, object] = {
        "landmarks": tuple(make_landmark() for _ in range(LANDMARK_COUNT)),
        "handedness": Handedness.LEFT,
    }
    defaults.update(kwargs)
    return TrackedHand(**defaults)  # type: ignore[arg-type]


class TestLandmark:
    def test_stores_values(self) -> None:
        landmark = Landmark(x=0.1, y=0.2, z=0.3)
        assert (landmark.x, landmark.y, landmark.z) == (0.1, 0.2, 0.3)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            Landmark(x=bad, y=0.0, z=0.0)
        with pytest.raises(ValueError, match="finite"):
            Landmark(x=0.0, y=bad, z=0.0)
        with pytest.raises(ValueError, match="finite"):
            Landmark(x=0.0, y=0.0, z=bad)

    def test_allows_edge_coordinates_outside_unit_range(self) -> None:
        # Landmarks at a frame edge can sit marginally outside [0, 1].
        landmark = Landmark(x=-0.02, y=1.03, z=-0.4)
        assert landmark.x < 0.0
        assert landmark.y > 1.0

    def test_is_frozen(self) -> None:
        landmark = Landmark(x=0.1, y=0.2, z=0.3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            landmark.x = 0.9  # type: ignore[misc]


class TestHandedness:
    def test_known_values(self) -> None:
        assert Handedness.LEFT.value == "Left"
        assert Handedness.RIGHT.value == "Right"
        assert Handedness.UNKNOWN.value == "Unknown"

    def test_unknown_is_a_distinct_member(self) -> None:
        assert len(Handedness) == 3
        assert Handedness.UNKNOWN is not Handedness.LEFT


class TestTrackedHand:
    def test_requires_exact_landmark_count(self) -> None:
        with pytest.raises(ValueError, match="exactly 21 landmarks"):
            make_hand(landmarks=())

    def test_rejects_extra_landmarks(self) -> None:
        with pytest.raises(ValueError, match="exactly 21 landmarks"):
            make_hand(landmarks=tuple(make_landmark() for _ in range(22)))

    def test_landmark_count_property(self) -> None:
        assert make_hand().landmark_count == LANDMARK_COUNT

    def test_landmark_accessor_preserves_index(self) -> None:
        landmarks = tuple(Landmark(x=i / 21.0, y=0.5, z=0.0) for i in range(LANDMARK_COUNT))
        hand = make_hand(landmarks=landmarks)
        assert hand.landmark(0).x == 0.0
        assert hand.landmark(7).x == pytest.approx(7 / 21.0)

    def test_landmark_accessor_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            make_hand().landmark(LANDMARK_COUNT)

    def test_defaults_to_unknown_handedness(self) -> None:
        hand = TrackedHand(landmarks=tuple(make_landmark() for _ in range(LANDMARK_COUNT)))
        assert hand.handedness is Handedness.UNKNOWN
        assert hand.handedness_score is None
        assert hand.detection_score is None

    def test_preserves_landmark_order(self) -> None:
        landmarks = tuple(Landmark(x=i, y=0.0, z=0.0) for i in range(LANDMARK_COUNT))
        assert make_hand(landmarks=landmarks).landmarks == landmarks


class TestTrackingResult:
    def test_empty_hands_is_valid(self) -> None:
        result = TrackingResult(hands=(), timestamp=1.5, frame_sequence=7)
        assert result.hand_count == 0
        assert result.has_hands is False
        assert result.primary_hand is None

    def test_passes_through_frame_metadata(self) -> None:
        result = TrackingResult(hands=(make_hand(),), timestamp=2.25, frame_sequence=11)
        assert result.timestamp == 2.25
        assert result.frame_sequence == 11

    def test_primary_hand_is_first_in_tracker_order(self) -> None:
        first = make_hand(handedness=Handedness.LEFT)
        second = make_hand(handedness=Handedness.RIGHT)
        result = TrackingResult(hands=(first, second), timestamp=0.0, frame_sequence=0)
        assert result.hand_count == 2
        assert result.has_hands is True
        assert result.primary_hand is first

    def test_preserves_backend_hand_order(self) -> None:
        hands = tuple(make_hand(handedness=h) for h in (Handedness.RIGHT, Handedness.LEFT))
        result = TrackingResult(hands=hands, timestamp=0.0, frame_sequence=0)
        assert [h.handedness for h in result.hands] == [
            Handedness.RIGHT,
            Handedness.LEFT,
        ]


class TestTrackerConfig:
    def test_defaults(self) -> None:
        config = TrackerConfig()
        assert config.max_hands == 2
        assert config.running_mode is RunningMode.VIDEO
        assert config.model_path is None
        assert config.model_path_str is None

    def test_model_path_str_converts(self) -> None:
        config = TrackerConfig(model_path="models/hand_landmarker.task")
        assert config.model_path_str == "models/hand_landmarker.task"

    @pytest.mark.parametrize("bad", [0, -1, 11, 100])
    def test_rejects_out_of_range_max_hands(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_hands"):
            TrackerConfig(max_hands=bad)

    @pytest.mark.parametrize(
        "field",
        [
            "min_detection_confidence",
            "min_presence_confidence",
            "min_tracking_confidence",
        ],
    )
    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan])
    def test_rejects_out_of_range_confidence(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError, match=field):
            TrackerConfig(**{field: bad})

    @pytest.mark.parametrize("boundary", [0.0, 1.0])
    def test_accepts_confidence_boundaries(self, boundary: float) -> None:
        config = TrackerConfig(min_detection_confidence=boundary)
        assert config.min_detection_confidence == boundary

    def test_running_mode_values(self) -> None:
        assert RunningMode.IMAGE.value == "image"
        assert RunningMode.VIDEO.value == "video"

    def test_live_stream_is_not_offered(self) -> None:
        # LIVE_STREAM needs async callbacks, deliberately out of scope.
        assert {mode.name for mode in RunningMode} == {"IMAGE", "VIDEO"}
