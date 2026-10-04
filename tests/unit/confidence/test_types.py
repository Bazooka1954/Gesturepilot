"""Validation and invariant tests for the confidence filter's domain types.

These are the contracts a caller can rely on without reading the filter: a threshold that
could not produce a sensible answer is rejected where it was written, and a report cannot
describe a state its own fields contradict. The invariants are the point of the module. A
report that said ``STABLE`` while naming no accepted gesture, or accepted a gesture that was
not its candidate, would be worse than no report at all — and the position where that mistake
is easy to make is when a report is constructed by hand rather than by the filter.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from gesturepilot.classifier.types import Gesture, GestureClassification
from gesturepilot.confidence.types import (
    CandidateChangePolicy,
    ConfidenceFilterConfig,
    FilterState,
    StabilityReport,
)

from .fixtures import classification, unknown


def _report(
    *,
    state: FilterState = FilterState.STABLE,
    candidate: Gesture | None = Gesture.POINT,
    candidate_observations: int = 3,
    accepted: Gesture | None = Gesture.POINT,
    latest_confidence: float = 0.8,
    mean_confidence: float | None = 0.8,
    interrupted: bool = False,
    classification_result: GestureClassification | None = None,
) -> StabilityReport:
    """A report whose fields agree with each other, for tests to break one at a time."""
    return StabilityReport(
        state=state,
        candidate=candidate,
        candidate_observations=candidate_observations,
        accepted=accepted,
        latest_confidence=latest_confidence,
        mean_confidence=mean_confidence,
        interrupted=interrupted,
        classification=classification_result or classification(Gesture.POINT, 0.8),
    )


class TestFilterState:
    def test_values_are_unique(self) -> None:
        values = [state.value for state in FilterState]
        assert len(set(values)) == len(values)

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (FilterState.IDLE, "idle"),
            (FilterState.CANDIDATE, "candidate"),
            (FilterState.STABLE, "stable"),
        ],
    )
    def test_str_is_the_value(self, state: FilterState, expected: str) -> None:
        assert str(state) == expected

    def test_the_three_states_are_exactly_the_documented_set(self) -> None:
        assert {state.name for state in FilterState} == {"IDLE", "CANDIDATE", "STABLE"}


class TestCandidateChangePolicy:
    def test_values_are_unique(self) -> None:
        values = [policy.value for policy in CandidateChangePolicy]
        assert len(set(values)) == len(values)

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            (CandidateChangePolicy.RESET, "reset"),
            (CandidateChangePolicy.HOLD, "hold"),
        ],
    )
    def test_str_is_the_value(self, policy: CandidateChangePolicy, expected: str) -> None:
        assert str(policy) == expected


class TestConfidenceFilterConfig:
    def test_defaults_do_not_accept_immediately(self) -> None:
        """The default must require a run, so a candidate cannot appear stable at once."""
        config = ConfidenceFilterConfig()
        assert config.min_stable_observations > 1
        assert 0.0 <= config.min_confidence <= 1.0
        assert config.candidate_change is CandidateChangePolicy.RESET
        assert config.max_interruption_seconds is None

    def test_is_frozen(self) -> None:
        config = ConfidenceFilterConfig()
        with pytest.raises(AttributeError):
            config.min_confidence = 0.1

    @pytest.mark.parametrize("value", [-0.1, 1.1, -1e-9, 1.0 + 1e-9])
    def test_confidence_threshold_outside_the_unit_interval_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            ConfidenceFilterConfig(min_confidence=value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_confidence_threshold_at_the_edges_is_accepted(self, value: float) -> None:
        assert ConfidenceFilterConfig(min_confidence=value).min_confidence == value

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_confidence_threshold_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            ConfidenceFilterConfig(min_confidence=value)

    @pytest.mark.parametrize("value", ["high", None, [0.5], True])
    def test_non_numeric_confidence_threshold_is_rejected(self, value: object) -> None:
        with pytest.raises(TypeError, match="min_confidence"):
            ConfidenceFilterConfig(min_confidence=value)

    @pytest.mark.parametrize("value", [0, -1, -50])
    def test_a_stability_requirement_below_one_is_rejected(self, value: int) -> None:
        with pytest.raises(ValueError, match="min_stable_observations"):
            ConfidenceFilterConfig(min_stable_observations=value)

    def test_one_observation_is_explicitly_permitted(self) -> None:
        """``1`` is the documented way to allow acceptance on the first observation."""
        assert ConfidenceFilterConfig(min_stable_observations=1).min_stable_observations == 1

    @pytest.mark.parametrize("value", [1.5, 3.0, "3", True, None])
    def test_a_non_integer_stability_requirement_is_rejected(self, value: object) -> None:
        with pytest.raises(TypeError, match="min_stable_observations"):
            ConfidenceFilterConfig(min_stable_observations=value)

    @pytest.mark.parametrize("value", ["reset", 1, None, True])
    def test_a_non_policy_candidate_change_is_rejected(self, value: object) -> None:
        with pytest.raises(TypeError, match="candidate_change"):
            ConfidenceFilterConfig(candidate_change=value)

    @pytest.mark.parametrize("value", [0.0, -0.5, -1e-9])
    def test_a_non_positive_interruption_limit_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="max_interruption_seconds"):
            ConfidenceFilterConfig(max_interruption_seconds=value)

    def test_a_positive_interruption_limit_is_accepted(self) -> None:
        assert ConfidenceFilterConfig(max_interruption_seconds=0.5).max_interruption_seconds == 0.5

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_interruption_limit_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="max_interruption_seconds"):
            ConfidenceFilterConfig(max_interruption_seconds=value)

    @pytest.mark.parametrize("value", ["0.5", True])
    def test_a_non_numeric_interruption_limit_is_rejected(self, value: object) -> None:
        with pytest.raises(TypeError, match="max_interruption_seconds"):
            ConfidenceFilterConfig(max_interruption_seconds=value)

    def test_every_field_is_documented(self) -> None:
        """Each knob appears in the class docstring, so no threshold is a bare field.

        A threshold nobody can find explained is a threshold nobody can tune on purpose.
        """
        docstring = ConfidenceFilterConfig.__doc__ or ""
        for field in dataclasses.fields(ConfidenceFilterConfig):
            assert field.name in docstring, f"{field.name} is not documented"


class TestStabilityReport:
    def test_a_consistent_report_is_accepted(self) -> None:
        report = _report()
        assert report.is_stable
        assert report.has_accepted_gesture
        assert report.accepted is Gesture.POINT

    def test_a_non_state_label_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="state"):
            _report(state="stable")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        report = _report()
        with pytest.raises(AttributeError):
            report.state = FilterState.IDLE

    def test_idle_names_no_candidate(self) -> None:
        report = _report(
            state=FilterState.IDLE,
            candidate=None,
            candidate_observations=0,
            accepted=None,
            mean_confidence=None,
        )
        assert not report.is_stable
        assert not report.has_accepted_gesture
        assert report.mean_confidence is None

    def test_an_idle_report_may_not_name_a_candidate(self) -> None:
        with pytest.raises(ValueError, match="IDLE"):
            _report(state=FilterState.IDLE, accepted=None, mean_confidence=None)

    def test_a_run_may_not_be_counted_without_a_candidate(self) -> None:
        with pytest.raises(ValueError, match="without naming a candidate"):
            _report(
                state=FilterState.CANDIDATE, candidate=None, candidate_observations=3, accepted=None
            )

    def test_only_a_stable_report_accepts_a_gesture(self) -> None:
        """A candidate that has not stabilised accepts nothing."""
        with pytest.raises(ValueError, match="only a STABLE report"):
            _report(state=FilterState.CANDIDATE, accepted=Gesture.POINT)

    def test_a_stable_report_must_name_what_it_accepts(self) -> None:
        with pytest.raises(ValueError, match="only a STABLE report"):
            _report(state=FilterState.STABLE, accepted=None)

    def test_a_report_may_not_accept_a_gesture_other_than_its_candidate(self) -> None:
        with pytest.raises(ValueError, match="other than its candidate"):
            _report(candidate=Gesture.POINT, accepted=Gesture.FIST)

    @pytest.mark.parametrize("field", ["candidate", "accepted"])
    def test_unknown_is_never_a_candidate_or_an_accepted_gesture(self, field: str) -> None:
        """``UNKNOWN`` is the classifier declining to name a gesture, not a gesture."""
        arguments = {
            "state": FilterState.STABLE,
            "candidate": Gesture.POINT,
            "candidate_observations": 3,
            "accepted": Gesture.POINT,
        }
        arguments[field] = Gesture.UNKNOWN
        with pytest.raises(ValueError, match="UNKNOWN"):
            _report(**arguments)

    def test_a_non_gesture_candidate_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="candidate"):
            _report(candidate="point", accepted=None, state=FilterState.CANDIDATE)  # type: ignore[arg-type]

    def test_a_non_gesture_accepted_label_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="accepted"):
            _report(accepted="point")  # type: ignore[arg-type]

    @pytest.mark.parametrize("count", [-1, -100])
    def test_a_negative_run_length_is_rejected(self, count: int) -> None:
        with pytest.raises(ValueError, match="candidate_observations"):
            _report(candidate_observations=count)

    @pytest.mark.parametrize("count", [1.5, "3", True, None])
    def test_a_non_integer_run_length_is_rejected(self, count: object) -> None:
        with pytest.raises(TypeError, match="candidate_observations"):
            _report(candidate_observations=count)

    @pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
    def test_a_latest_confidence_outside_the_unit_interval_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="latest_confidence"):
            _report(latest_confidence=value)

    @pytest.mark.parametrize("value", ["high", None, True])
    def test_a_non_numeric_latest_confidence_is_rejected(self, value: object) -> None:
        with pytest.raises(TypeError, match="latest_confidence"):
            _report(latest_confidence=value)

    def test_the_latest_confidence_may_not_be_absent(self) -> None:
        """It is the classifier's own score for this observation, so it always exists."""
        with pytest.raises(TypeError, match="latest_confidence"):
            _report(latest_confidence=None)

    @pytest.mark.parametrize("value", [-0.1, 1.1, math.nan])
    def test_a_mean_confidence_outside_the_unit_interval_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="mean_confidence"):
            _report(mean_confidence=value)

    def test_an_absent_mean_confidence_is_allowed(self) -> None:
        assert _report(mean_confidence=None).mean_confidence is None

    def test_a_non_boolean_interrupted_flag_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="interrupted"):
            _report(interrupted="yes")  # type: ignore[arg-type]

    def test_a_non_classification_input_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="classification"):
            _report(classification_result="POINT")  # type: ignore[arg-type]

    def test_the_classification_is_carried_through_untouched(self) -> None:
        result = unknown(0.9)
        assert _report(classification_result=result).classification is result

    def test_str_names_the_state_and_the_accepted_gesture(self) -> None:
        summary = str(_report())
        assert "stable" in summary
        assert "point" in summary

    def test_str_reports_no_candidate_in_plain_words(self) -> None:
        summary = str(
            _report(
                state=FilterState.IDLE,
                candidate=None,
                candidate_observations=0,
                accepted=None,
                mean_confidence=None,
            )
        )
        assert summary.count("none") == 2
