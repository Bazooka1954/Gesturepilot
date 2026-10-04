"""Behaviour tests for the temporal confidence filter.

The filter's promises are what these tests are organised around, not the code that implements
them:

* a new candidate is never accepted immediately, and is accepted exactly when the run
  reaches the configured length
* repeated observations of one valid gesture satisfy the requirement, and a later flicker
  cannot quietly keep a run alive
* ``UNKNOWN`` and sub-threshold classifications are never evidence for a gesture, and never
  accepted as one
* what the filter reports is the present, not the past: a gesture that stops being seen stops
  being accepted, with no remembered state surviving
* a caller can start over completely with :meth:`ConfidenceFilter.reset`
* timestamps are optional, checked for monotonicity when given, and a long gap breaks the
  run rather than bridging it
* an input is never modified, and the same inputs always give the same reports
* rubbish in raises instead of becoming a plausible-looking report

Confidence is the axis most likely to be got wrong, so the threshold tests are deliberately
exhaustive: below, exactly at, and above, at both sides of the boundary.
"""

from __future__ import annotations

import math

import pytest

from gesturepilot.classifier.types import Gesture, GestureClassification
from gesturepilot.confidence.errors import (
    ConfidenceError,
    FilterStateError,
    InvalidClassificationError,
)
from gesturepilot.confidence.filter import ConfidenceFilter
from gesturepilot.confidence.types import (
    CandidateChangePolicy,
    ConfidenceFilterConfig,
    FilterState,
    StabilityReport,
)

from .fixtures import classification, corrupt_confidence, unknown

POINT = classification(Gesture.POINT, 0.8)
FIST = classification(Gesture.FIST, 0.8)
PINCH = classification(Gesture.PINCH, 0.8)


def _feed(
    filter_: ConfidenceFilter,
    observations: list[GestureClassification],
) -> list[StabilityReport]:
    """Observe every classification in turn and collect the reports."""
    return [filter_.observe(observation) for observation in observations]


class TestFirstObservation:
    def test_a_lone_observation_is_not_accepted(self) -> None:
        """One frame cannot show a stable gesture, however confident it is."""
        filter_ = ConfidenceFilter()
        report = filter_.observe(POINT)
        assert report.state is FilterState.CANDIDATE
        assert report.candidate is Gesture.POINT
        assert report.candidate_observations == 1
        assert report.accepted is None
        assert not report.is_stable
        assert not report.has_accepted_gesture

    def test_a_fresh_filter_has_no_state_at_all(self) -> None:
        filter_ = ConfidenceFilter()
        assert filter_.state is FilterState.IDLE
        assert filter_.candidate is None
        assert filter_.candidate_observations == 0
        assert filter_.accepted is None
        assert filter_.last_timestamp is None

    def test_one_observation_may_be_configured_to_be_enough(self) -> None:
        """``min_stable_observations=1`` is the explicit opt-in to immediate acceptance."""
        report = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=1)).observe(POINT)
        assert report.state is FilterState.STABLE
        assert report.accepted is Gesture.POINT

    def test_a_very_confident_single_observation_is_still_not_accepted(self) -> None:
        report = ConfidenceFilter().observe(classification(Gesture.POINT, 1.0))
        assert report.state is FilterState.CANDIDATE
        assert report.accepted is None


class TestStabilityThreshold:
    @pytest.mark.parametrize("required", [2, 3, 5])
    def test_acceptance_happens_exactly_on_the_nth_observation(self, required: int) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=required))
        reports = _feed(filter_, [POINT] * required)
        for index, report in enumerate(reports[:-1]):
            assert report.state is FilterState.CANDIDATE, f"accepted after {index + 1} observations"
            assert report.accepted is None
        assert reports[-1].state is FilterState.STABLE
        assert reports[-1].accepted is Gesture.POINT
        assert reports[-1].candidate_observations == required

    def test_one_short_of_the_requirement_is_not_accepted(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        reports = _feed(filter_, [POINT, POINT])
        assert all(report.accepted is None for report in reports)
        assert filter_.accepted is None

    def test_a_gesture_stays_accepted_while_it_keeps_being_seen(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        reports = _feed(filter_, [POINT] * 10)
        assert all(report.state is FilterState.STABLE for report in reports[2:])
        assert reports[-1].candidate_observations == 10
        assert filter_.accepted is Gesture.POINT

    def test_the_run_length_never_exceeds_the_observations_seen(self) -> None:
        filter_ = ConfidenceFilter()
        reports = _feed(filter_, [POINT] * 5)
        for index, report in enumerate(reports):
            assert report.candidate_observations == index + 1


class TestCandidateChange:
    def test_a_different_gesture_restarts_the_run(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        _feed(filter_, [POINT, POINT])
        assert filter_.state is FilterState.CANDIDATE
        report = filter_.observe(FIST)
        assert report.candidate is Gesture.FIST
        assert report.candidate_observations == 1
        assert report.accepted is None

    def test_the_new_gesture_must_earn_its_own_stability(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        _feed(filter_, [POINT, POINT, POINT])
        assert filter_.accepted is Gesture.POINT
        report = filter_.observe(FIST)
        assert filter_.accepted is None
        assert report.accepted is None
        assert report.state is FilterState.CANDIDATE

    def test_switching_back_does_not_resume_the_old_run(self) -> None:
        """A gesture that flickers away and back has not been seen three times running."""
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        _feed(filter_, [POINT, POINT, FIST, POINT])
        assert filter_.candidate_observations == 1
        assert filter_.accepted is None

    def test_the_hold_policy_keeps_the_incumbent_through_a_flicker(self) -> None:
        filter_ = ConfidenceFilter(
            ConfidenceFilterConfig(
                min_stable_observations=3,
                candidate_change=CandidateChangePolicy.HOLD,
            )
        )
        _feed(filter_, [POINT, POINT, POINT])
        assert filter_.accepted is Gesture.POINT
        report = filter_.observe(FIST)
        assert report.candidate is Gesture.POINT
        assert report.accepted is Gesture.POINT
        assert report.candidate_observations == 3
        assert report.state is FilterState.STABLE

    def test_the_hold_policy_ignores_the_new_gesture_entirely(self) -> None:
        filter_ = ConfidenceFilter(
            ConfidenceFilterConfig(
                min_stable_observations=2,
                candidate_change=CandidateChangePolicy.HOLD,
            )
        )
        _feed(filter_, [POINT, POINT])
        for _ in range(5):
            filter_.observe(FIST)
        assert filter_.candidate is Gesture.POINT
        assert filter_.accepted is Gesture.POINT

    def test_the_reset_policy_is_the_default(self) -> None:
        assert ConfidenceFilterConfig().candidate_change is CandidateChangePolicy.RESET
        filter_ = ConfidenceFilter()
        _feed(filter_, [POINT, POINT, POINT])
        assert filter_.observe(FIST).candidate is Gesture.FIST


class TestUnknown:
    def test_a_lone_unknown_is_not_accepted(self) -> None:
        report = ConfidenceFilter().observe(unknown(0.9))
        assert report.state is FilterState.IDLE
        assert report.accepted is None
        assert report.candidate is None

    def test_repeated_unknowns_never_accumulate(self) -> None:
        """No number of repeats turns "no gesture applies" into a gesture."""
        filter_ = ConfidenceFilter()
        reports = _feed(filter_, [unknown(0.95)] * 10)
        assert all(report.accepted is None for report in reports)
        assert all(report.state is FilterState.IDLE for report in reports)
        assert filter_.accepted is None

    def test_unknown_ends_a_run_in_progress(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        _feed(filter_, [POINT, POINT])
        report = filter_.observe(unknown(0.9))
        assert report.state is FilterState.IDLE
        assert report.candidate_observations == 0

    def test_unknown_ends_an_accepted_gesture(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=2))
        _feed(filter_, [POINT, POINT])
        assert filter_.accepted is Gesture.POINT
        report = filter_.observe(unknown(0.9))
        assert report.accepted is None
        assert filter_.accepted is None

    def test_a_confident_unknown_reports_its_own_confidence_unchanged(self) -> None:
        """The classifier's confidence in an ``UNKNOWN`` means weak evidence, not strength.

        The filter passes it through rather than inverting it, so a caller reading
        ``latest_confidence`` on its own cannot mistake it for a strong gesture.
        """
        report = ConfidenceFilter().observe(unknown(0.95))
        assert report.latest_confidence == 0.95
        assert report.accepted is None
        assert report.mean_confidence is None

    def test_a_run_restarts_from_scratch_after_an_unknown(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=2))
        _feed(filter_, [POINT])
        filter_.observe(unknown(0.9))
        report = filter_.observe(POINT)
        assert report.candidate_observations == 1
        assert report.accepted is None


class TestConfidenceThreshold:
    @pytest.fixture
    def filter_(self) -> ConfidenceFilter:
        return ConfidenceFilter(ConfidenceFilterConfig(min_confidence=0.7))

    @pytest.mark.parametrize(
        ("confidence", "expected_state"),
        [
            (0.69, FilterState.IDLE),
            (0.69999, FilterState.IDLE),
            (0.7, FilterState.CANDIDATE),
            (0.70001, FilterState.CANDIDATE),
            (1.0, FilterState.CANDIDATE),
        ],
    )
    def test_the_threshold_boundary_is_inclusive(
        self, filter_: ConfidenceFilter, confidence: float, expected_state: FilterState
    ) -> None:
        """At the threshold exactly counts; strictly below does not."""
        report = filter_.observe(classification(Gesture.POINT, confidence))
        assert report.state is expected_state
        assert report.accepted is None

    def test_below_threshold_observations_never_reach_a_stable_run(self) -> None:
        filter_ = ConfidenceFilter(
            ConfidenceFilterConfig(min_confidence=0.7, min_stable_observations=3)
        )
        reports = _feed(filter_, [classification(Gesture.POINT, 0.69)] * 5)
        assert all(report.state is FilterState.IDLE for report in reports)
        assert filter_.accepted is None

    def test_a_low_confidence_observation_does_not_reinforce_the_candidate(self) -> None:
        """It is not evidence, so it cannot be counted toward stability."""
        filter_ = ConfidenceFilter(
            ConfidenceFilterConfig(min_confidence=0.7, min_stable_observations=3)
        )
        _feed(filter_, [classification(Gesture.POINT, 0.8), classification(Gesture.POINT, 0.8)])
        report = filter_.observe(classification(Gesture.POINT, 0.2))
        assert report.state is FilterState.IDLE
        assert report.candidate is None
        assert report.candidate_observations == 0

    def test_a_low_confidence_observation_does_not_count_toward_the_run_length(self) -> None:
        filter_ = ConfidenceFilter(
            ConfidenceFilterConfig(min_confidence=0.7, min_stable_observations=3)
        )
        _feed(filter_, [classification(Gesture.POINT, 0.8)])
        filter_.observe(classification(Gesture.POINT, 0.2))
        report = filter_.observe(classification(Gesture.POINT, 0.8))
        assert report.candidate_observations == 1

    def test_a_low_confidence_observation_ends_an_accepted_gesture(self) -> None:
        filter_ = ConfidenceFilter(
            ConfidenceFilterConfig(min_confidence=0.7, min_stable_observations=2)
        )
        _feed(filter_, [classification(Gesture.POINT, 0.8), classification(Gesture.POINT, 0.8)])
        assert filter_.accepted is Gesture.POINT
        report = filter_.observe(classification(Gesture.POINT, 0.69))
        assert report.accepted is None
        assert filter_.accepted is None

    def test_a_zero_confidence_is_never_relabelled_as_a_strong_one(self) -> None:
        """Zero confidence in every gesture must not be reported as a strong result."""
        filter_ = ConfidenceFilter()
        report = filter_.observe(unknown(0.0))
        assert report.latest_confidence == 0.0
        assert report.mean_confidence is None
        assert report.accepted is None

    def test_the_lowest_usable_confidence_is_still_not_accepted(self) -> None:
        report = ConfidenceFilter().observe(classification(Gesture.POINT, 0.01))
        assert report.latest_confidence == 0.01
        assert report.accepted is None

    def test_the_latest_confidence_is_the_classifier_s_own_number(self) -> None:
        for confidence in (0.01, 0.25, 0.6, 0.99, 1.0):
            report = ConfidenceFilter().observe(classification(Gesture.FIST, confidence))
            assert report.latest_confidence == confidence

    def test_the_mean_confidence_is_the_mean_of_the_qualifying_observations(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_confidence=0.5))
        reports = _feed(
            filter_,
            [
                classification(Gesture.POINT, 0.6),
                classification(Gesture.POINT, 0.9),
                classification(Gesture.POINT, 0.8),
            ],
        )
        assert [report.mean_confidence for report in reports] == pytest.approx(
            [0.6, 0.75, pytest.approx((0.6 + 0.9 + 0.8) / 3)]
        )

    def test_excluded_observations_do_not_pull_the_mean_down(self) -> None:
        """A weak frame is dropped, not averaged in, so it cannot be smoothed away."""
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_confidence=0.7))
        filter_.observe(classification(Gesture.POINT, 0.9))
        report = filter_.observe(classification(Gesture.POINT, 0.1))
        assert report.mean_confidence is None

    def test_the_mean_is_absent_while_there_is_no_run(self) -> None:
        assert ConfidenceFilter().observe(unknown(0.9)).mean_confidence is None


class TestReset:
    def test_reset_clears_a_candidate_run(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        _feed(filter_, [POINT, POINT])
        filter_.reset()
        assert filter_.state is FilterState.IDLE
        assert filter_.candidate is None
        assert filter_.candidate_observations == 0
        assert filter_.accepted is None

    def test_reset_clears_an_accepted_gesture(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=2))
        _feed(filter_, [POINT, POINT])
        assert filter_.accepted is Gesture.POINT
        filter_.reset()
        assert filter_.accepted is None

    def test_after_reset_the_run_starts_over(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=2))
        _feed(filter_, [POINT])
        filter_.reset()
        report = filter_.observe(POINT)
        assert report.candidate_observations == 1
        assert report.accepted is None

    def test_reset_forgets_the_last_timestamp(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(max_interruption_seconds=0.5))
        filter_.observe(POINT, timestamp=1.0)
        assert filter_.last_timestamp == 1.0
        filter_.reset()
        assert filter_.last_timestamp is None

    def test_reset_on_a_never_used_filter_is_harmless(self) -> None:
        filter_ = ConfidenceFilter()
        filter_.reset()
        assert filter_.state is FilterState.IDLE

    def test_reset_returns_the_filter_to_its_constructed_state(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=3))
        _feed(filter_, [POINT] * 5)
        fresh = ConfidenceFilter(filter_.config)
        filter_.reset()
        for attribute in ("state", "candidate", "candidate_observations", "accepted"):
            assert getattr(filter_, attribute) == getattr(fresh, attribute)

    def test_observation_counts_are_not_kept_across_a_reset(self) -> None:
        """A run cannot be pieced together from before and after a reset."""
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(min_stable_observations=2))
        _feed(filter_, [POINT])
        filter_.reset()
        assert filter_.observe(POINT).candidate_observations == 1


class TestInterruption:
    @pytest.fixture
    def filter_(self) -> ConfidenceFilter:
        return ConfidenceFilter(ConfidenceFilterConfig(max_interruption_seconds=0.5))

    def test_a_gap_within_the_limit_keeps_the_run(self, filter_: ConfidenceFilter) -> None:
        _feed_with_time(filter_, [0.0, 0.5, 1.0])
        assert filter_.candidate_observations == 3

    def test_a_gap_beyond_the_limit_discards_the_run(self, filter_: ConfidenceFilter) -> None:
        _feed_with_time(filter_, [0.0, 0.1, 5.0])
        assert filter_.candidate_observations == 1
        assert filter_.accepted is None

    def test_an_interruption_is_reported(self, filter_: ConfidenceFilter) -> None:
        reports = _feed_with_time(filter_, [0.0, 5.0])
        assert not reports[0].interrupted
        assert reports[1].interrupted

    def test_an_interruption_breaks_a_run_that_would_otherwise_stabilise(
        self, filter_: ConfidenceFilter
    ) -> None:
        """Two observations an hour apart are not two observations of the same gesture."""
        reports = _feed_with_time(filter_, [0.0, 3600.0])
        assert reports[1].candidate_observations == 1
        assert reports[1].state is FilterState.CANDIDATE

    def test_an_interruption_drops_an_accepted_gesture(self, filter_: ConfidenceFilter) -> None:
        """A hand that left the frame has not been holding the gesture all along."""
        filter_ = ConfidenceFilter(
            ConfidenceFilterConfig(min_stable_observations=2, max_interruption_seconds=0.5)
        )
        _feed_with_time(filter_, [0.0, 0.1])
        assert filter_.accepted is Gesture.POINT
        report = filter_.observe(POINT, timestamp=30.0)
        assert report.accepted is None
        assert report.interrupted

    def test_the_observation_after_an_interruption_is_still_judged(
        self, filter_: ConfidenceFilter
    ) -> None:
        """The gap breaks the run; it does not swallow the frame that followed it."""
        report = _feed_with_time(filter_, [0.0, 5.0])[-1]
        assert report.candidate is Gesture.POINT
        assert report.candidate_observations == 1

    def test_an_interruption_is_reported_even_with_nothing_to_discard(
        self, filter_: ConfidenceFilter
    ) -> None:
        """The sequence was still broken, which is worth logging even when idle."""
        filter_.observe(unknown(0.9), timestamp=0.0)
        report = filter_.observe(POINT, timestamp=5.0)
        assert report.state is FilterState.CANDIDATE
        assert report.interrupted

    def test_no_interruption_is_reported_without_a_limit(self) -> None:
        filter_ = ConfidenceFilter()
        reports = _feed_with_time(filter_, [0.0, 10_000.0])
        assert not any(report.interrupted for report in reports)
        assert filter_.candidate_observations == 2

    def test_interruption_needs_timestamps_on_both_sides(self) -> None:
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(max_interruption_seconds=0.5))
        filter_.observe(POINT, timestamp=0.0)
        report = filter_.observe(POINT)
        assert not report.interrupted
        assert report.candidate_observations == 2

    def test_a_missing_timestamp_stops_the_filter_measuring_gaps(self) -> None:
        """A gap that was not witnessed cannot be measured across the period it spans."""
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(max_interruption_seconds=0.5))
        filter_.observe(POINT, timestamp=0.0)
        filter_.observe(POINT)
        assert filter_.last_timestamp is None
        report = filter_.observe(POINT, timestamp=60.0)
        assert not report.interrupted
        assert report.candidate_observations == 3

    def test_a_timestamp_is_remembered_when_given(self, filter_: ConfidenceFilter) -> None:
        filter_.observe(POINT, timestamp=1.5)
        assert filter_.last_timestamp == 1.5

    def test_a_gap_just_inside_the_limit_does_not_break_a_run(
        self, filter_: ConfidenceFilter
    ) -> None:
        reports = _feed_with_time(filter_, [0.0, 0.4999, 0.9])
        assert all(not report.interrupted for report in reports)
        assert reports[-1].candidate_observations == 3


class TestTimestamps:
    def test_observations_need_no_timestamp_at_all(self) -> None:
        filter_ = ConfidenceFilter()
        reports = _feed(filter_, [POINT, POINT, POINT])
        assert reports[-1].is_stable
        assert filter_.last_timestamp is None

    def test_equal_timestamps_are_allowed(self) -> None:
        """A zero gap is shorter than any limit, so two observations may share an instant."""
        filter_ = ConfidenceFilter(ConfidenceFilterConfig(max_interruption_seconds=0.5))
        reports = _feed_with_time(filter_, [1.0, 1.0, 1.0])
        assert [report.candidate_observations for report in reports] == [1, 2, 3]
        assert reports[-1].is_stable

    def test_a_backwards_timestamp_raises(self) -> None:
        filter_ = ConfidenceFilter()
        filter_.observe(POINT, timestamp=1.0)
        with pytest.raises(FilterStateError, match="backwards"):
            filter_.observe(POINT, timestamp=0.5)

    def test_a_backwards_timestamp_does_not_corrupt_the_state(self) -> None:
        """The rejected observation leaves the filter exactly as it was."""
        filter_ = ConfidenceFilter()
        filter_.observe(POINT, timestamp=1.0)
        with pytest.raises(FilterStateError):
            filter_.observe(FIST, timestamp=0.5)
        assert filter_.candidate_observations == 1
        assert filter_.candidate is Gesture.POINT
        assert filter_.last_timestamp == 1.0

    @pytest.mark.parametrize("timestamp", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_timestamp_raises(self, timestamp: float) -> None:
        with pytest.raises(FilterStateError, match="finite"):
            ConfidenceFilter().observe(POINT, timestamp=timestamp)

    def test_a_negative_timestamp_raises(self) -> None:
        with pytest.raises(FilterStateError, match=">= 0.0"):
            ConfidenceFilter().observe(POINT, timestamp=-0.001)

    @pytest.mark.parametrize("timestamp", ["1.0", True, [1.0]])
    def test_a_non_numeric_timestamp_raises(self, timestamp: object) -> None:
        with pytest.raises(FilterStateError, match="real number"):
            ConfidenceFilter().observe(POINT, timestamp=timestamp)

    def test_a_timestamp_is_keyword_only(self) -> None:
        """Positional timestamps would silently change meaning if the signature grew."""
        with pytest.raises(TypeError):
            ConfidenceFilter().observe(POINT, 1.0)  # type: ignore[misc]

    def test_an_integer_timestamp_is_accepted(self) -> None:
        assert ConfidenceFilter().observe(POINT, timestamp=2).latest_confidence == 0.8


class TestMalformedInput:
    @pytest.mark.parametrize("value", [None, "POINT", 0.8, Gesture.POINT, object()], ids=repr)
    def test_a_non_classification_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidClassificationError, match="GestureClassification"):
            ConfidenceFilter().observe(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("confidence", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_confidence_is_rejected(self, confidence: float) -> None:
        forged = corrupt_confidence(POINT, confidence)
        with pytest.raises(InvalidClassificationError, match="finite"):
            ConfidenceFilter().observe(forged)

    @pytest.mark.parametrize("confidence", [-0.5, 1.5])
    def test_an_out_of_range_confidence_is_rejected(self, confidence: float) -> None:
        """The domain type refuses these; a forged one must not reach the comparison."""
        forged = corrupt_confidence(POINT, confidence)
        with pytest.raises(InvalidClassificationError, match=r"\[0.0, 1.0\]"):
            ConfidenceFilter().observe(forged)

    def test_a_rejected_input_leaves_the_filter_untouched(self) -> None:
        filter_ = ConfidenceFilter()
        filter_.observe(POINT)
        with pytest.raises(InvalidClassificationError):
            filter_.observe(corrupt_confidence(POINT, math.nan))
        assert filter_.candidate_observations == 1
        assert filter_.candidate is Gesture.POINT

    def test_a_non_numeric_confidence_is_rejected(self) -> None:
        forged = corrupt_confidence(POINT, "high")
        with pytest.raises(InvalidClassificationError, match="real number"):
            ConfidenceFilter().observe(forged)

    def test_every_failure_is_a_confidence_error(self) -> None:
        """One ``except`` catches the whole hierarchy, as the other layers promise."""
        assert issubclass(InvalidClassificationError, ConfidenceError)
        assert issubclass(FilterStateError, ConfidenceError)


class TestImmutability:
    def test_the_input_classification_is_not_modified(self) -> None:
        observation = classification(Gesture.POINT, 0.8)
        snapshot = (observation.gesture, observation.confidence, observation.margin)
        scores = dict(observation.scores)
        ConfidenceFilter().observe(observation)
        assert (observation.gesture, observation.confidence, observation.margin) == snapshot
        assert dict(observation.scores) == scores

    def test_the_same_object_is_carried_into_the_report(self) -> None:
        observation = classification(Gesture.POINT, 0.8)
        report = ConfidenceFilter().observe(observation)
        assert report.classification is observation
        assert report.classification.margin == observation.margin

    def test_the_report_is_frozen(self) -> None:
        report = ConfidenceFilter().observe(POINT)
        with pytest.raises(AttributeError):
            report.accepted = Gesture.FIST

    def test_the_config_is_frozen_and_shared(self) -> None:
        config = ConfidenceFilterConfig()
        assert ConfidenceFilter(config).config is config

    def test_the_input_score_mapping_stays_read_only(self) -> None:
        report = ConfidenceFilter().observe(POINT)
        with pytest.raises(TypeError):
            report.classification.scores[Gesture.FIST] = 0.1


class TestDeterminism:
    SEQUENCE = [
        classification(Gesture.POINT, 0.8),
        unknown(0.9),
        classification(Gesture.FIST, 0.55),
        classification(Gesture.FIST, 0.95),
        classification(Gesture.FIST, 0.7),
        classification(Gesture.PINCH, 0.8),
        classification(Gesture.PINCH, 0.8),
        classification(Gesture.PINCH, 0.8),
    ]

    def _reports(self, config: ConfidenceFilterConfig | None = None) -> list[str]:
        return [str(report) for report in _feed(ConfidenceFilter(config), self.SEQUENCE)]

    def test_two_filters_agree_every_time(self) -> None:
        assert self._reports() == self._reports()
        assert self._reports(ConfidenceFilterConfig(min_stable_observations=5)) == self._reports(
            ConfidenceFilterConfig(min_stable_observations=5)
        )

    def test_the_sequence_exercises_a_real_transition(self) -> None:
        """The shared sequence must actually reach acceptance, or it proves nothing."""
        states = [report.state for report in _feed(ConfidenceFilter(), self.SEQUENCE)]
        assert FilterState.STABLE in states
        assert FilterState.IDLE in states
        assert FilterState.CANDIDATE in states

    def test_two_filters_are_independent(self) -> None:
        first, second = ConfidenceFilter(), ConfidenceFilter()
        _feed(first, [POINT, POINT, POINT])
        assert second.state is FilterState.IDLE
        assert first.accepted is Gesture.POINT

    def test_the_same_observations_give_the_same_counts(self) -> None:
        counts = [
            report.candidate_observations for report in _feed(ConfidenceFilter(), [POINT] * 4)
        ]
        assert counts == [1, 2, 3, 4]


class TestIntrospection:
    def test_the_filter_exposes_its_configuration(self) -> None:
        config = ConfidenceFilterConfig(min_confidence=0.9, min_stable_observations=7)
        assert ConfidenceFilter(config).config is config

    def test_a_non_configuration_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="ConfidenceFilterConfig"):
            ConfidenceFilter("strict")  # type: ignore[arg-type]

    def test_the_default_configuration_is_used_when_none_is_given(self) -> None:
        assert ConfidenceFilter().config == ConfidenceFilterConfig()

    def test_is_stable_agrees_with_state(self) -> None:
        filter_ = ConfidenceFilter()
        assert not filter_.is_stable
        _feed(filter_, [POINT, POINT, POINT])
        assert filter_.is_stable

    def test_repr_names_the_state_and_the_candidate(self) -> None:
        filter_ = ConfidenceFilter()
        filter_.observe(POINT)
        summary = repr(filter_)
        assert "candidate" in summary
        assert "point" in summary

    def test_repr_names_the_policy(self) -> None:
        summary = repr(ConfidenceFilter(ConfidenceFilterConfig(min_confidence=0.42)))
        assert "0.42" in summary


def _feed_with_time(
    filter_: ConfidenceFilter,
    timestamps: list[float],
) -> list[StabilityReport]:
    """Observe the same point at each timestamp, in order."""
    return [filter_.observe(POINT, timestamp=timestamp) for timestamp in timestamps]
