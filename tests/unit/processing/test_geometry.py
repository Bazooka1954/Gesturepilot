"""Unit tests for the pure geometry helpers.

These are the primitives every feature is built from, so the boundary cases are pinned
hard: zero-length vectors, exact collinearity, and the clamp that keeps ``acos`` from
raising on a rounding artefact.
"""

from __future__ import annotations

import math

import pytest

from gesturepilot.processing.errors import DegenerateHandError, InvalidLandmarksError
from gesturepilot.processing.geometry import (
    add,
    angle_at,
    angle_between,
    centroid,
    distance,
    distance_2d,
    dot,
    hand_scale,
    landmark_point,
    magnitude,
    optional_angle_at,
    optional_angle_between,
    orientation_from_landmarks,
    palm_center,
    relative_to,
    subtract,
)
from gesturepilot.processing.topology import LandmarkIndex
from gesturepilot.processing.types import ProcessedLandmark

from .fixtures import (
    HAND_COORDINATES,
    HAND_SCALE,
    PALM_CENTRE,
    PALM_CENTRE_Z,
    collapsed,
    make_processed,
    mirrored,
)


def _rotated_reference_hand(degrees: float) -> tuple[ProcessedLandmark, ...]:
    """Rotate the reference hand about its wrist by ``degrees`` in the image plane.

    Distances are preserved exactly, so the hand scale is untouched and the only thing the
    rotation can change is direction.
    """
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    wrist_x, wrist_y, _ = HAND_COORDINATES[0]
    return tuple(
        ProcessedLandmark(
            x=wrist_x + (x - wrist_x) * cos - (y - wrist_y) * sin,
            y=wrist_y + (x - wrist_x) * sin + (y - wrist_y) * cos,
            z=z,
        )
        for x, y, z in HAND_COORDINATES
    )


class TestVectorArithmetic:
    def test_subtract_is_component_wise(self) -> None:
        assert subtract((3.0, 4.0, 5.0), (1.0, 1.0, 1.0)) == (2.0, 3.0, 4.0)

    def test_add_is_component_wise(self) -> None:
        assert add((1.0, 2.0), (0.5, 0.5)) == (1.5, 2.5)

    @pytest.mark.parametrize("operation", [subtract, add, dot])
    def test_mismatched_dimensions_are_rejected(self, operation: object) -> None:
        with pytest.raises(ValueError, match="equal length"):
            operation((1.0, 2.0), (1.0, 2.0, 3.0))  # type: ignore[operator]

    def test_magnitude_of_a_zero_vector_is_zero(self) -> None:
        assert magnitude((0.0, 0.0, 0.0)) == 0.0

    def test_magnitude_of_the_three_four_triangle(self) -> None:
        assert magnitude((3.0, 4.0)) == pytest.approx(5.0)

    def test_dot_product(self) -> None:
        assert dot((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == pytest.approx(32.0)

    def test_dot_product_of_orthogonal_vectors_is_zero(self) -> None:
        assert dot((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_distance_ignores_nothing_in_three_dimensions(self) -> None:
        assert distance((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)) == pytest.approx(2.0)

    def test_distance_2d_ignores_depth(self) -> None:
        near = (0.1, 0.2, 0.0)
        far = (0.1, 0.2, 5.0)
        assert distance_2d(near, far) == pytest.approx(0.0)
        assert distance(near, far) == pytest.approx(5.0)


class TestAngleBetween:
    def test_perpendicular_vectors(self) -> None:
        assert angle_between((1.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi / 2)

    def test_collinear_vectors(self) -> None:
        assert angle_between((2.0, 0.0), (5.0, 0.0)) == pytest.approx(0.0)

    def test_opposed_vectors(self) -> None:
        assert angle_between((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(math.pi)

    def test_is_symmetric(self) -> None:
        assert angle_between((1.0, 2.0), (-3.0, 4.0)) == pytest.approx(
            angle_between((-3.0, 4.0), (1.0, 2.0))
        )

    def test_is_scale_invariant(self) -> None:
        enlarged = tuple(component * 1000.0 for component in (-3.0, 4.0))
        assert angle_between((1.0, 2.0), (-3.0, 4.0)) == pytest.approx(
            angle_between((1.0, 2.0), enlarged)
        )

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ((1.0, 0.0), (0.0, 1.0)),
            ((1.0, 0.0), (-1.0, 0.0)),
            ((1.0, 1.0), (1.0, -1.0)),
            ((3.0, 4.0), (-4.0, 3.0)),
            ((1.0, 1.0), (7.0, 7.0)),
            ((1e150, 1e150), (1e150, 1e150)),
            ((1e150, 1e150), (-1e150, -1e150)),
        ],
    )
    def test_result_always_lands_in_range(self, a: tuple[float, ...], b: tuple[float, ...]) -> None:
        """The clamp keeps rounding from pushing acos out of its domain."""
        angle = angle_between(a, b)
        assert 0.0 <= angle <= math.pi
        assert math.isfinite(angle)

    def test_large_magnitude_collinear_pair_is_still_zero(self) -> None:
        assert angle_between((1e150, 1e150), (1e150, 1e150)) == pytest.approx(0.0, abs=1e-9)


class TestDegenerateVectors:
    def test_angle_against_a_zero_vector_raises(self) -> None:
        with pytest.raises(DegenerateHandError, match="zero-length"):
            angle_between((0.0, 0.0), (1.0, 0.0))

    def test_angle_against_a_zero_second_vector_raises(self) -> None:
        with pytest.raises(DegenerateHandError, match="zero-length"):
            angle_between((1.0, 0.0), (0.0, 0.0))

    def test_optional_angle_returns_none_for_a_zero_vector(self) -> None:
        assert optional_angle_between((0.0, 0.0), (1.0, 0.0)) is None

    def test_optional_angle_returns_none_for_a_zero_second_vector(self) -> None:
        assert optional_angle_between((1.0, 0.0), (0.0, 0.0)) is None

    def test_optional_angle_matches_the_strict_version_otherwise(self) -> None:
        assert optional_angle_between((1.0, 0.0), (0.0, 1.0)) == pytest.approx(
            angle_between((1.0, 0.0), (0.0, 1.0))
        )


class TestAngleAt:
    def test_right_angle_at_the_vertex(self) -> None:
        assert angle_at((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)) == pytest.approx(math.pi / 2)

    def test_straight_line_is_a_straight_angle(self) -> None:
        assert angle_at((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) == pytest.approx(math.pi)

    def test_coincident_vertex_raises(self) -> None:
        with pytest.raises(DegenerateHandError, match="zero-length"):
            angle_at((1.0, 1.0), (1.0, 1.0), (0.0, 0.0))

    def test_optional_variant_returns_none_for_a_coincident_vertex(self) -> None:
        assert optional_angle_at((1.0, 1.0), (1.0, 1.0), (0.0, 0.0)) is None


class TestRelativeTo:
    def test_translates_without_scaling_by_default(self) -> None:
        assert relative_to((2.0, 4.0), (1.0, 1.0)) == (1.0, 3.0)

    def test_divides_by_the_scale(self) -> None:
        assert relative_to((2.0, 4.0), (1.0, 1.0), 2.0) == (0.5, 1.5)

    def test_handles_three_dimensions(self) -> None:
        assert relative_to((1.0, 2.0, 3.0), (1.0, 1.0, 1.0)) == (0.0, 1.0, 2.0)

    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_rejects_a_non_positive_scale(self, scale: float) -> None:
        with pytest.raises(DegenerateHandError, match="finite positive"):
            relative_to((1.0, 1.0), (0.0, 0.0), scale)

    @pytest.mark.parametrize("scale", [math.nan, math.inf])
    def test_rejects_a_non_finite_scale(self, scale: float) -> None:
        with pytest.raises(DegenerateHandError, match="finite positive"):
            relative_to((1.0, 1.0), (0.0, 0.0), scale)


class TestCentroid:
    def test_averages_components(self) -> None:
        assert centroid([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]) == pytest.approx((1.0, 5.0 / 3.0))

    def test_single_point_is_its_own_centroid(self) -> None:
        assert centroid([(0.25, 0.75, 0.5)]) == (0.25, 0.75, 0.5)

    def test_rejects_an_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            centroid([])

    def test_rejects_mixed_dimensions(self) -> None:
        with pytest.raises(ValueError, match="same dimension"):
            centroid([(1.0, 2.0), (1.0, 2.0, 3.0)])


class TestPalmCenter:
    def test_matches_the_reference_hand(self) -> None:
        centre = palm_center(make_processed())
        assert centre[0] == pytest.approx(PALM_CENTRE[0])
        assert centre[1] == pytest.approx(PALM_CENTRE[1])
        assert centre[2] == pytest.approx(PALM_CENTRE_Z)

    def test_is_the_mean_of_the_wrist_and_four_knuckles(self) -> None:
        landmarks = make_processed()
        palm = palm_center(landmarks)
        expected_x = (
            sum(
                landmarks[index].x
                for index in (
                    LandmarkIndex.WRIST,
                    LandmarkIndex.INDEX_MCP,
                    LandmarkIndex.MIDDLE_MCP,
                    LandmarkIndex.RING_MCP,
                    LandmarkIndex.PINKY_MCP,
                )
            )
            / 5
        )
        assert palm[0] == pytest.approx(expected_x)

    def test_ignores_the_fingertips(self) -> None:
        """Moving a fingertip must not move the palm centre."""
        base = make_processed()
        nudged = list(base)
        nudged[LandmarkIndex.INDEX_TIP] = ProcessedLandmark(x=0.05, y=0.05, z=0.0)
        assert palm_center(tuple(nudged)) == pytest.approx(palm_center(base))

    def test_rejects_a_short_landmark_sequence(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="landmarks"):
            palm_center(make_processed()[:3])


class TestHandScale:
    def test_matches_the_reference_hand(self) -> None:
        assert hand_scale(make_processed()) == pytest.approx(HAND_SCALE)

    def test_scales_with_apparent_hand_size(self) -> None:
        base = make_processed()
        doubled = tuple(
            ProcessedLandmark(x=landmark.x * 2.0, y=landmark.y * 2.0, z=landmark.z)
            for landmark in base
        )
        assert hand_scale(doubled) == pytest.approx(2.0 * HAND_SCALE)

    def test_raises_when_the_reference_points_coincide(self) -> None:
        with pytest.raises(DegenerateHandError, match="Hand scale collapsed"):
            hand_scale(make_processed(collapsed()))

    def test_rejects_a_short_landmark_sequence(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="landmarks"):
            hand_scale(make_processed()[:3])


class TestOrientation:
    def test_reference_hand_knuckle_line_runs_almost_across_the_frame(self) -> None:
        """The reference hand's knuckles sit in a near-horizontal row, so roll is near zero."""
        roll, _ = orientation_from_landmarks(make_processed())
        assert roll == pytest.approx(0.08490179344972208)

    def test_reference_hand_tilt_is_small_when_the_palm_points_up(self) -> None:
        _, tilt = orientation_from_landmarks(make_processed())
        assert tilt == pytest.approx(0.029403288204005135)

    def test_roll_is_folded_into_a_half_turn(self) -> None:
        """An undirected axis has no sign, so ``roll`` can never leave ``[-pi/2, pi/2]``."""
        for rotation in range(0, 360, 15):
            turned = _rotated_reference_hand(rotation)
            roll, _ = orientation_from_landmarks(turned)
            assert roll is not None
            assert -math.pi / 2 - 1e-9 <= roll <= math.pi / 2 + 1e-9, rotation

    def test_rolling_the_hand_by_pi_leaves_the_axis_angle_unchanged(self) -> None:
        """Half a turn turns the ray around but leaves the axis it lies on alone."""
        straight = orientation_from_landmarks(_rotated_reference_hand(0))[0]
        turned = orientation_from_landmarks(_rotated_reference_hand(180))[0]
        assert turned == pytest.approx(straight, abs=1e-9)

    def test_a_mirrored_hand_negates_both_angles(self) -> None:
        from gesturepilot.processing.normalization import normalize_hand

        base = normalize_hand(make_processed())
        flipped = normalize_hand(make_processed(mirrored(HAND_COORDINATES)))
        assert orientation_from_landmarks(flipped.landmarks)[0] == pytest.approx(
            -orientation_from_landmarks(base.landmarks)[0]
        )
        assert orientation_from_landmarks(flipped.landmarks)[1] == pytest.approx(
            -orientation_from_landmarks(base.landmarks)[1]
        )

    def test_collapsed_hand_yields_no_orientation(self) -> None:
        assert orientation_from_landmarks(make_processed(collapsed())) == (None, None)

    def test_orientation_survives_normalisation_unchanged(self) -> None:
        """Angles do not care about translation or scaling, so this is a free consistency check."""
        from gesturepilot.processing.normalization import normalize_hand

        raw = orientation_from_landmarks(make_processed())
        normalised = orientation_from_landmarks(normalize_hand(make_processed()).landmarks)
        assert normalised[0] == pytest.approx(raw[0])
        assert normalised[1] == pytest.approx(raw[1])

    def test_rejects_a_short_landmark_sequence(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="landmarks"):
            orientation_from_landmarks(make_processed()[:3])


class TestLandmarkPoint:
    def test_exposes_the_components_as_a_tuple(self) -> None:
        landmarks = make_processed()
        assert landmark_point(landmarks[0]) == pytest.approx((0.5, 0.8, 0.0))
