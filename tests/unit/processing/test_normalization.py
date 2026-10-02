"""Unit tests for wrist-relative and scale normalisation.

The contract being pinned: a hand that moves in the frame, or that grows as the user
leans towards the camera, must produce the *same* normalised features — while the raw
scale that was divided out is still reported, because "how big did the hand look" is
itself a useful signal.
"""

from __future__ import annotations

import math

import pytest

from gesturepilot.processing.errors import DegenerateHandError, InvalidLandmarksError
from gesturepilot.processing.geometry import hand_scale
from gesturepilot.processing.normalization import (
    normalize_hand,
    normalize_landmarks,
    normalize_point,
)
from gesturepilot.processing.topology import LandmarkIndex
from gesturepilot.processing.types import ProcessedLandmark
from gesturepilot.tracking.types import LANDMARK_COUNT

from .fixtures import (
    HAND_COORDINATES,
    HAND_SCALE,
    PALM_CENTRE,
    PALM_CENTRE_Z,
    collapsed,
    make_processed,
    mirrored,
    scaled,
    translated,
)


def _tuples(landmarks: tuple[object, ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple((item.x, item.y, item.z) for item in landmarks)  # type: ignore[attr-defined]


def _flat(landmarks: tuple[object, ...]) -> tuple[float, ...]:
    """Flatten landmarks to a single tuple of floats.

    ``pytest.approx`` cannot compare nested structures, so a per-coordinate comparison has
    to be spelled out as one flat sequence.
    """
    return tuple(value for landmark in _tuples(landmarks) for value in landmark)


class TestNormalizePoint:
    def test_subtracts_the_origin_and_divides_by_the_scale(self) -> None:
        point = ProcessedLandmark(x=3.0, y=6.0, z=9.0)
        origin = ProcessedLandmark(x=1.0, y=2.0, z=3.0)
        assert normalize_point(point, origin, 2.0).as_tuple == pytest.approx((1.0, 2.0, 3.0))

    def test_a_point_at_the_origin_normalises_to_zero(self) -> None:
        origin = ProcessedLandmark(x=0.5, y=0.5, z=0.0)
        assert normalize_point(origin, origin, 0.17).as_tuple == pytest.approx((0.0, 0.0, 0.0))

    def test_rejects_a_zero_scale(self) -> None:
        origin = ProcessedLandmark(x=0.0, y=0.0, z=0.0)
        with pytest.raises(DegenerateHandError, match="finite positive"):
            normalize_point(ProcessedLandmark(x=1.0, y=1.0, z=0.0), origin, 0.0)


class TestNormalizeLandmarks:
    def test_preserves_order_and_count(self) -> None:
        result = normalize_landmarks(make_processed(), make_processed()[0], HAND_SCALE)
        assert len(result) == LANDMARK_COUNT

    def test_first_landmark_is_the_origin(self) -> None:
        source = make_processed()
        result = normalize_landmarks(source, source[0], HAND_SCALE)
        assert result[0].as_tuple == (0.0, 0.0, 0.0)

    def test_scales_all_components(self) -> None:
        source = make_processed()
        result = normalize_landmarks(source, source[0], HAND_SCALE)
        for original, normalised in zip(source, result, strict=True):
            assert normalised.x == pytest.approx((original.x - source[0].x) / HAND_SCALE)


class TestNormalizeHand:
    def test_reports_the_wrist_as_the_origin(self) -> None:
        source = make_processed()
        assert normalize_hand(source).origin == source[LandmarkIndex.WRIST]

    def test_reports_the_hand_scale_it_divided_by(self) -> None:
        assert normalize_hand(make_processed()).scale == pytest.approx(HAND_SCALE)
        assert normalize_hand(make_processed()).scale == pytest.approx(hand_scale(make_processed()))

    def test_wrist_normalises_to_exactly_the_origin(self) -> None:
        wrist = normalize_hand(make_processed()).landmarks[LandmarkIndex.WRIST]
        assert (wrist.x, wrist.y, wrist.z) == (0.0, 0.0, 0.0)

    def test_palm_center_matches_the_reference_hand(self) -> None:
        centre = normalize_hand(make_processed()).palm_center
        assert centre.x == pytest.approx((PALM_CENTRE[0] - 0.500) / HAND_SCALE)
        assert centre.y == pytest.approx((PALM_CENTRE[1] - 0.800) / HAND_SCALE)
        assert centre.z == pytest.approx(PALM_CENTRE_Z / HAND_SCALE)

    def test_palm_center_is_consistent_with_the_normalised_landmarks(self) -> None:
        """The centroid of the normalised palm landmarks must equal the reported centre."""
        result = normalize_hand(make_processed())
        palm_indices = (
            LandmarkIndex.WRIST,
            LandmarkIndex.INDEX_MCP,
            LandmarkIndex.MIDDLE_MCP,
            LandmarkIndex.RING_MCP,
            LandmarkIndex.PINKY_MCP,
        )
        for axis in ("x", "y", "z"):
            expected = sum(getattr(result.landmarks[i], axis) for i in palm_indices) / 5
            assert getattr(result.palm_center, axis) == pytest.approx(expected)

    def test_rejects_an_empty_sequence(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="empty landmark sequence"):
            normalize_hand(())

    def test_raises_for_a_hand_with_no_measurable_size(self) -> None:
        with pytest.raises(DegenerateHandError, match="Hand scale collapsed"):
            normalize_hand(make_processed(collapsed()))

    def test_rejects_a_short_hand(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="landmarks"):
            normalize_hand(make_processed()[:5])

    def test_does_not_modify_its_input(self) -> None:
        source = make_processed()
        before = _tuples(source)
        normalize_hand(source)
        assert _tuples(source) == before

    def test_all_components_are_finite(self) -> None:
        result = normalize_hand(make_processed())
        for landmark in (*result.landmarks, result.palm_center):
            assert math.isfinite(landmark.x)
            assert math.isfinite(landmark.y)
            assert math.isfinite(landmark.z)


class TestPositionInvariance:
    def test_moving_the_hand_in_the_frame_changes_nothing_normalised(self) -> None:
        base = normalize_hand(make_processed())
        moved = normalize_hand(make_processed(translated(HAND_COORDINATES, 0.1, -0.07)))

        assert _flat(moved.landmarks) == pytest.approx(_flat(base.landmarks), abs=1e-12)
        assert moved.palm_center.as_tuple == pytest.approx(base.palm_center.as_tuple, abs=1e-12)

    def test_the_origin_tracks_the_moved_wrist(self) -> None:
        moved = normalize_hand(make_processed(translated(HAND_COORDINATES, 0.1, 0.0)))
        assert moved.origin.x == pytest.approx(0.6)


class TestSizeInvariance:
    @pytest.mark.parametrize("factor", [0.4, 1.0, 2.4])
    def test_apparent_size_cancels_out(self, factor: float) -> None:
        base = normalize_hand(make_processed())
        resized = normalize_hand(make_processed(scaled(HAND_COORDINATES, factor)))

        assert _flat(resized.landmarks) == pytest.approx(_flat(base.landmarks), abs=1e-9)
        assert resized.palm_center.as_tuple == pytest.approx(base.palm_center.as_tuple, abs=1e-9)

    def test_the_raw_scale_still_reports_the_apparent_size(self) -> None:
        """Normalisation cancels apparent size from the features, not from the measurement."""
        small = normalize_hand(make_processed(scaled(HAND_COORDINATES, 0.5)))
        large = normalize_hand(make_processed(scaled(HAND_COORDINATES, 2.0)))
        assert large.scale == pytest.approx(4.0 * small.scale)


class TestMirrorSymmetry:
    def test_a_mirrored_hand_produces_mirrored_normalised_coordinates(self) -> None:
        base = normalize_hand(make_processed())
        reflected = normalize_hand(make_processed(mirrored(HAND_COORDINATES)))
        for original, flipped in zip(base.landmarks, reflected.landmarks, strict=True):
            assert flipped.x == pytest.approx(-original.x, abs=1e-12)
            assert flipped.y == pytest.approx(original.y, abs=1e-12)

    def test_a_mirrored_hand_keeps_the_same_scale(self) -> None:
        reflected = normalize_hand(make_processed(mirrored(HAND_COORDINATES)))
        assert reflected.scale == pytest.approx(HAND_SCALE)
