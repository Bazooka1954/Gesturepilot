"""Unit tests for the One Euro filter and the landmark smoother.

Every test drives the filter with literal timestamps. There is no sleeping anywhere, no
wall clock, and no randomness, so every assertion here is exact rather than approximate.
That is the point of supplying the timestamp at all.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from gesturepilot.processing.errors import FilterStateError, InvalidLandmarksError
from gesturepilot.processing.filters import LandmarkSmoother, OneEuroFilter
from gesturepilot.processing.topology import LandmarkIndex
from gesturepilot.processing.types import FilterConfig, ProcessedLandmark
from gesturepilot.tracking.types import LANDMARK_COUNT

from .fixtures import make_processed

#: A 30 Hz timeline. Realistic, and exactly representable in binary floating point.
FRAME_STEP = 1.0 / 30.0


def timeline(count: int, step: float = FRAME_STEP) -> list[float]:
    """Return ``count`` evenly spaced timestamps starting at zero."""
    return [index * step for index in range(count)]


def run_filter(
    samples: list[float],
    config: FilterConfig | None = None,
    step: float = FRAME_STEP,
) -> list[float]:
    """Filter ``samples`` over a fixed timeline and return every output."""
    filt = OneEuroFilter(config)
    return [
        filt.filter(value, time)
        for value, time in zip(samples, timeline(len(samples), step), strict=True)
    ]


def jittered(amount: float) -> list[float]:
    """A constant signal with alternating jitter, as a detector would produce."""
    return [0.75 + (amount if index % 2 == 0 else -amount) for index in range(60)]


class TestFilterConfig:
    def test_defaults_match_the_reference_implementation(self) -> None:
        config = FilterConfig()
        assert config.min_cutoff == 1.0
        assert config.beta == 0.0
        assert config.derivative_cutoff == 1.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_a_non_positive_min_cutoff(self, bad: float) -> None:
        with pytest.raises(ValueError, match="min_cutoff"):
            FilterConfig(min_cutoff=bad)

    @pytest.mark.parametrize("bad", [-0.001, math.nan, math.inf])
    def test_rejects_a_negative_beta(self, bad: float) -> None:
        with pytest.raises(ValueError, match="beta"):
            FilterConfig(beta=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
    def test_rejects_a_non_positive_derivative_cutoff(self, bad: float) -> None:
        with pytest.raises(ValueError, match="derivative_cutoff"):
            FilterConfig(derivative_cutoff=bad)

    def test_beta_of_zero_is_valid(self) -> None:
        assert FilterConfig(beta=0.0).beta == 0.0

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            FilterConfig().beta = 1.0  # type: ignore[misc]


class TestOneEuroFilter:
    def test_starts_uninitialised(self) -> None:
        filt = OneEuroFilter()
        assert filt.is_initialized is False
        assert filt.last_timestamp is None

    def test_first_sample_passes_through_unchanged(self) -> None:
        assert OneEuroFilter().filter(0.42, 0.0) == pytest.approx(0.42)

    def test_first_sample_passes_through_whatever_the_settings(self) -> None:
        filt = OneEuroFilter(FilterConfig(min_cutoff=0.01, beta=10.0))
        assert filt.filter(7.5, 1.25) == pytest.approx(7.5)

    def test_records_the_timestamp_and_becomes_initialised(self) -> None:
        filt = OneEuroFilter()
        filt.filter(1.0, 2.5)
        assert filt.is_initialized is True
        assert filt.last_timestamp == pytest.approx(2.5)

    def test_output_never_overshoots_its_inputs(self) -> None:
        """An exponential filter is convex, so the output is always bracketed."""
        filt = OneEuroFilter(FilterConfig(min_cutoff=1.0, beta=0.5))
        output = filt.filter(0.0, 0.0)
        for index, sample in enumerate([0.5, -0.2, 3.0, -3.0], start=1):
            previous = output
            output = filt.filter(sample, index * FRAME_STEP)
            assert min(previous, sample) <= output <= max(previous, sample)

    def test_is_deterministic_for_a_fixed_timeline(self) -> None:
        samples = [0.0, 0.1, -0.05, 0.3, 0.3, 0.29, 0.31, 0.30]
        config = FilterConfig(min_cutoff=1.5, beta=0.7)

        assert run_filter(samples, config) == run_filter(samples, config)

    def test_a_constant_signal_converges_to_it(self) -> None:
        output = run_filter([0.75] * 200, FilterConfig(min_cutoff=4.0, beta=0.0))[-1]
        assert output == pytest.approx(0.75, abs=1e-3)

    def test_attenuates_jitter(self) -> None:
        raw = jittered(0.05)
        smoothed = run_filter(raw, FilterConfig(min_cutoff=1.0, beta=0.0))
        steady_state = smoothed[20:]

        assert max(steady_state) - min(steady_state) < max(raw) - min(raw)

    def test_beta_makes_the_filter_track_a_step_more_closely(self) -> None:
        """The whole point of the adaptive cutoff: fast motion gets less smoothing."""

        def settled(beta: float) -> float:
            return run_filter([0.0] * 10 + [1.0] * 10, FilterConfig(min_cutoff=1.0, beta=beta))[-1]

        assert settled(beta=2.0) > settled(beta=0.0)

    def test_a_zero_cutoff_is_refused_before_any_sample(self) -> None:
        with pytest.raises(ValueError, match="derivative_cutoff"):
            OneEuroFilter(FilterConfig(derivative_cutoff=0.0))

    def test_a_repeated_timestamp_is_rejected(self) -> None:
        filt = OneEuroFilter()
        filt.filter(1.0, 1.0)
        with pytest.raises(FilterStateError, match="strictly increase"):
            filt.filter(1.0, 1.0)

    def test_a_decreasing_timestamp_is_rejected(self) -> None:
        filt = OneEuroFilter()
        filt.filter(1.0, 2.0)
        with pytest.raises(FilterStateError, match="strictly increase"):
            filt.filter(1.0, 1.5)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_timestamp_is_rejected(self, bad: float) -> None:
        with pytest.raises(FilterStateError, match="finite"):
            OneEuroFilter().filter(1.0, bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_sample_is_rejected(self, bad: float) -> None:
        with pytest.raises(InvalidLandmarksError, match="non-finite sample"):
            OneEuroFilter().filter(bad, 0.0)

    def test_a_rejected_timestamp_leaves_the_state_untouched(self) -> None:
        filt = OneEuroFilter()
        filt.filter(1.0, 1.0)
        with pytest.raises(FilterStateError):
            filt.filter(9.0, 0.5)

        assert filt.last_timestamp == pytest.approx(1.0)
        # The next accepted sample continues from the unperturbed history.
        assert filt.filter(1.0, 2.0) == pytest.approx(1.0)

    def test_reset_clears_every_piece_of_state(self) -> None:
        filt = OneEuroFilter()
        filt.filter(1.0, 1.0)
        filt.filter(2.0, 2.0)
        filt.reset()

        assert filt.is_initialized is False
        assert filt.last_timestamp is None
        assert filt.filter(9.0, 0.0) == pytest.approx(9.0)

    def test_reset_allows_a_repeated_timestamp(self) -> None:
        filt = OneEuroFilter()
        filt.filter(1.0, 1.0)
        filt.reset()
        assert filt.filter(1.0, 1.0) == pytest.approx(1.0)

    def test_exposes_its_configuration(self) -> None:
        config = FilterConfig(min_cutoff=2.0, beta=0.5, derivative_cutoff=0.7)
        assert OneEuroFilter(config).config == config

    def test_rejects_a_configuration_of_the_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="FilterConfig"):
            OneEuroFilter({"min_cutoff": 1.0})  # type: ignore[arg-type]

    def test_repr_names_the_parameters(self) -> None:
        text = repr(OneEuroFilter(FilterConfig(min_cutoff=2.0, beta=0.5)))
        assert "min_cutoff=2.0" in text
        assert "beta=0.5" in text


class TestLandmarkSmoother:
    def test_starts_uninitialised(self) -> None:
        assert LandmarkSmoother().is_initialized is False
        assert LandmarkSmoother().last_timestamp is None

    def test_first_frame_passes_through_unchanged(self) -> None:
        source = make_processed()
        smoothed = LandmarkSmoother().filter(source, 0.0)

        assert len(smoothed) == LANDMARK_COUNT
        for original, produced in zip(source, smoothed, strict=True):
            assert produced.as_tuple == pytest.approx(original.as_tuple)

    def test_gives_each_coordinate_its_own_filter(self) -> None:
        """A shared filter would drag every coordinate toward the previous frame's value."""
        source = make_processed()
        smoother = LandmarkSmoother(FilterConfig(min_cutoff=1.0, beta=0.0))
        first = smoother.filter(source, 0.0)
        moved_x = tuple(
            ProcessedLandmark(x=landmark.x + 0.01, y=landmark.y, z=landmark.z)
            for landmark in source
        )
        second = smoother.filter(moved_x, FRAME_STEP)

        assert second[LandmarkIndex.MIDDLE_TIP].x != pytest.approx(
            first[LandmarkIndex.MIDDLE_TIP].x
        )
        assert second[LandmarkIndex.MIDDLE_TIP].y == pytest.approx(
            first[LandmarkIndex.MIDDLE_TIP].y, abs=1e-12
        )

    def test_smooths_jitter_across_a_sequence_of_frames(self) -> None:
        """A fast, small oscillation must come out with a visibly smaller spread."""
        source = make_processed()
        jitter = 0.004
        smoother = LandmarkSmoother(FilterConfig(min_cutoff=1.0, beta=0.0))
        smoother.filter(source, 0.0)

        resting = source[LandmarkIndex.MIDDLE_TIP].x
        outputs = []
        for step in range(1, 30):
            offset = jitter if step % 2 else -jitter
            shifted = tuple(
                ProcessedLandmark(x=landmark.x + offset, y=landmark.y, z=landmark.z)
                for landmark in source
            )
            outputs.append(smoother.filter(shifted, step * FRAME_STEP)[LandmarkIndex.MIDDLE_TIP].x)

        assert max(outputs) - min(outputs) < 2.0 * jitter
        assert max(outputs) < resting + jitter
        assert min(outputs) > resting - jitter

    def test_rejects_a_wrong_landmark_count(self) -> None:
        with pytest.raises(InvalidLandmarksError, match="exactly 21 landmarks"):
            LandmarkSmoother().filter(make_processed()[:5], 0.0)

    def test_rejects_a_non_increasing_timestamp(self) -> None:
        smoother = LandmarkSmoother()
        source = make_processed()
        smoother.filter(source, 1.0)
        with pytest.raises(FilterStateError, match="strictly increase"):
            smoother.filter(source, 1.0)

    def test_reset_returns_to_the_passthrough_behaviour(self) -> None:
        smoother = LandmarkSmoother()
        source = make_processed()
        smoother.filter(source, 0.0)
        smoother.filter(source, FRAME_STEP)
        smoother.reset()

        assert smoother.is_initialized is False
        assert smoother.last_timestamp is None
        moved = tuple(
            ProcessedLandmark(x=landmark.x + 0.1, y=landmark.y, z=landmark.z) for landmark in source
        )
        assert smoother.filter(moved, 0.0) == tuple(moved)

    def test_does_not_modify_its_input(self) -> None:
        source = make_processed()
        before = tuple(landmark.as_tuple for landmark in source)
        smoother = LandmarkSmoother()
        smoother.filter(source, 0.0)
        smoother.filter(source, FRAME_STEP)

        assert tuple(landmark.as_tuple for landmark in source) == before

    def test_exposes_its_configuration(self) -> None:
        config = FilterConfig(min_cutoff=3.0)
        assert LandmarkSmoother(config).config == config

    def test_rejects_a_configuration_of_the_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="FilterConfig"):
            LandmarkSmoother(1.0)  # type: ignore[arg-type]

    def test_repr_names_the_configuration(self) -> None:
        assert "LandmarkSmoother" in repr(LandmarkSmoother())
