"""Temporal confidence filtering domain types.

The confidence filter consumes a stream of
:class:`~gesturepilot.classifier.types.GestureClassification` results, one per
observation, and answers a single question: *has the same gesture now been seen often
enough, and strongly enough, to be called stable?* The classifier answers a different
question about a single frame — *which pose is this, and how strongly?* — so this layer
adds the one thing the classifier deliberately lacks: memory.

Two thresholds, and nothing else
--------------------------------
:attr:`ConfidenceFilterConfig.min_confidence` is a per-observation gate. An observation
whose confidence is below it is not evidence of anything and ends the current run.
:attr:`~ConfidenceFilterConfig.min_stable_observations` is a per-run gate: this many
qualifying observations in a row for the same gesture.

Counting, not clocking
----------------------
Stability is measured in **observations**, not seconds, because the pipeline's frame rate
is not fixed and nothing downstream may assume it is. A caller that knows how long a gap
has grown may additionally supply a ``timestamp`` per observation and configure
``max_interruption_seconds``; a longer gap then discards the accumulated run. Timestamps
are optional throughout and are never read from a clock: see
:class:`~gesturepilot.confidence.filter.ConfidenceFilter` for the monotonicity policy.

Confidence is reported, never invented
--------------------------------------
:class:`StabilityReport` carries the classifier's own score for this observation verbatim
in :attr:`~StabilityReport.latest_confidence`, plus :attr:`StabilityReport.classification`
untouched, so the original evidence and its margin stay available. The one aggregate,
:attr:`StabilityReport.mean_confidence`, is the arithmetic mean of the classifier's scores
over the current run and is named and documented as a run summary rather than a confidence
of its own. A weak classification is never reported as a strong one; below-threshold
observations are excluded from the run entirely instead of being averaged in.

Accepted is not authorised
--------------------------
An accepted gesture means "stable", nothing more. Whether a stable gesture is allowed to
cause an action is the safety state machine's decision, in a later phase. Nothing in this
package dispatches anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from gesturepilot.classifier.types import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    Gesture,
    GestureClassification,
)


class FilterState(Enum):
    """How much the filter currently believes a candidate gesture.

    The state is not stored independently: it is derived from the candidate and its run
    length, so it cannot drift out of step with them. See
    :attr:`~gesturepilot.confidence.filter.ConfidenceFilter.state`.

    ``IDLE``
        No candidate. The last observation did not qualify, or nothing has been observed
        since construction or :meth:`~gesturepilot.confidence.filter.ConfidenceFilter.reset`.
    ``CANDIDATE``
        A gesture is accumulating a run, but the run is shorter than
        ``min_stable_observations``.
    ``STABLE``
        The candidate's run has reached ``min_stable_observations``. Only this state
        reports an accepted gesture.
    """

    IDLE = "idle"
    CANDIDATE = "candidate"
    STABLE = "stable"

    def __str__(self) -> str:
        return self.value


class CandidateChangePolicy(Enum):
    """What a qualifying observation of a *different* gesture does to the run.

    ``RESET``
        The new gesture becomes the candidate and starts a fresh run at one observation.
        The incumbent must earn stability again from nothing. This is the default, and the
        conservative choice: a hand moving from a fist to a point has not demonstrated a
        stable point yet.
    ``HOLD``
        The incumbent keeps its candidate and its accumulated run; the differing
        observation is discarded as a fluctuation. Suited to two poses that genuinely
        flicker between each other, at the cost of ignoring a real gesture change for as
        long as the incumbent keeps qualifying.
    """

    RESET = "reset"
    HOLD = "hold"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ConfidenceFilterConfig:
    """Every threshold the temporal filter uses, validated where it is written.

    Attributes:
        min_confidence: The lowest confidence an observation may report and still count
            toward stability. An observation at exactly this value qualifies; anything
            strictly below it does not. Defaults to the midpoint of the classifier's own
            range rather than to a tuned value — see ``docs/confidence-filter.md``.
        min_stable_observations: How many qualifying observations of the same gesture are
            needed to accept it. Must be at least ``1``; ``1`` explicitly permits accepting
            a gesture on its very first qualifying observation, so the default of ``3`` is
            what makes a new candidate *not* appear stable immediately.
        candidate_change: What to do when a qualifying observation names a different
            gesture from the incumbent candidate. See :class:`CandidateChangePolicy`.
        max_interruption_seconds: The longest gap between two observations that still
            counts as continuous. A longer gap discards the accumulated run before the new
            observation is judged, so evidence cannot be accumulated across a period when
            nothing was seen — the case where the hand left the frame entirely.
            ``None`` (the default) disables time-based interruption, leaving the filter
            purely count-based. Ignored for observations supplied without a timestamp.
    """

    min_confidence: float = 0.6
    min_stable_observations: int = 3
    candidate_change: CandidateChangePolicy = CandidateChangePolicy.RESET
    max_interruption_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_number("ConfidenceFilterConfig.min_confidence", self.min_confidence)
        if not MIN_CONFIDENCE <= self.min_confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"ConfidenceFilterConfig.min_confidence must be within "
                f"[{MIN_CONFIDENCE}, {MAX_CONFIDENCE}], got {self.min_confidence!r}"
            )
        if isinstance(self.min_stable_observations, bool) or not isinstance(
            self.min_stable_observations, int
        ):
            raise TypeError(
                f"ConfidenceFilterConfig.min_stable_observations must be an int, got "
                f"{self.min_stable_observations!r}"
            )
        if self.min_stable_observations < 1:
            raise ValueError(
                f"ConfidenceFilterConfig.min_stable_observations must be >= 1, got "
                f"{self.min_stable_observations!r}"
            )
        if not isinstance(self.candidate_change, CandidateChangePolicy):
            raise TypeError(
                f"ConfidenceFilterConfig.candidate_change must be a CandidateChangePolicy, "
                f"got {type(self.candidate_change).__name__}"
            )
        if self.max_interruption_seconds is not None:
            _require_number(
                "ConfidenceFilterConfig.max_interruption_seconds",
                self.max_interruption_seconds,
            )
            if self.max_interruption_seconds <= 0.0:
                raise ValueError(
                    f"ConfidenceFilterConfig.max_interruption_seconds must be > 0.0 when "
                    f"set, got {self.max_interruption_seconds!r}"
                )


@dataclass(frozen=True)
class StabilityReport:
    """The filter's answer for one observation.

    Attributes:
        state: How much the filter currently believes the candidate. See
            :class:`FilterState`.
        candidate: The gesture being accumulated evidence for, or ``None`` when the state
            is ``IDLE``. Never :attr:`~gesturepilot.classifier.types.Gesture.UNKNOWN`.
        candidate_observations: How many qualifying observations of ``candidate`` have
            been seen in the current run. ``0`` when ``candidate`` is ``None``.
        accepted: The gesture currently accepted as stable, or ``None``. Exactly the
            candidate when the state is ``STABLE``, and ``None`` in every other state, so
            "no gesture is accepted" is an explicit value rather than an absence.
        latest_confidence: The classifier's own confidence for this observation, verbatim.
            For an ``UNKNOWN`` this is the classifier's *confidence that no gesture
            applies*, which is high when the evidence was weak; it is passed through
            unchanged and never reinterpreted.
        mean_confidence: The arithmetic mean of the classifier's confidence over the
            qualifying observations of the current run, or ``None`` when there are none.
            A summary of how consistent the run was, **not** a confidence in the classifier's
            sense and not comparable with :attr:`latest_confidence`'s meaning.
        interrupted: Whether a gap longer than ``max_interruption_seconds`` separated this
            observation from the previous one, so the run was discarded before this
            observation was judged. Always ``False`` when the filter is count-based.
        classification: The classification this report is about, unmodified. The whole
            classifier result stays available, margin included, so nothing downstream has
            to guess what the filter saw.
    """

    state: FilterState
    candidate: Gesture | None
    candidate_observations: int
    accepted: Gesture | None
    latest_confidence: float
    mean_confidence: float | None
    interrupted: bool
    classification: GestureClassification

    def __post_init__(self) -> None:
        if not isinstance(self.state, FilterState):
            raise TypeError(f"StabilityReport.state must be a FilterState, got {self.state!r}")
        for field_name in ("candidate", "accepted"):
            gesture = getattr(self, field_name)
            if gesture is not None:
                if not isinstance(gesture, Gesture):
                    raise TypeError(
                        f"StabilityReport.{field_name} must be a Gesture or None, got {gesture!r}"
                    )
                if not gesture.is_recognized:
                    raise ValueError(
                        f"StabilityReport.{field_name} cannot be "
                        f"{gesture.name}: UNKNOWN is not a gesture that can be accepted"
                    )
        for field_name in ("candidate_observations",):
            count = getattr(self, field_name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"StabilityReport.{field_name} must be an int, got {count!r}")
            if count < 0:
                raise ValueError(f"StabilityReport.{field_name} must be >= 0, got {count!r}")
        _require_number("StabilityReport.latest_confidence", self.latest_confidence)
        if not MIN_CONFIDENCE <= self.latest_confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"StabilityReport.latest_confidence must be within "
                f"[{MIN_CONFIDENCE}, {MAX_CONFIDENCE}], got {self.latest_confidence!r}"
            )
        if self.mean_confidence is not None:
            _require_number("StabilityReport.mean_confidence", self.mean_confidence)
            if not MIN_CONFIDENCE <= self.mean_confidence <= MAX_CONFIDENCE:
                raise ValueError(
                    f"StabilityReport.mean_confidence must be within "
                    f"[{MIN_CONFIDENCE}, {MAX_CONFIDENCE}] or None, got "
                    f"{self.mean_confidence!r}"
                )
        if not isinstance(self.interrupted, bool):
            raise TypeError(f"StabilityReport.interrupted must be a bool, got {self.interrupted!r}")
        if not isinstance(self.classification, GestureClassification):
            raise TypeError(
                f"StabilityReport.classification must be a GestureClassification, got "
                f"{type(self.classification).__name__}"
            )
        if self.state is FilterState.IDLE and (
            self.candidate is not None or self.candidate_observations != 0
        ):
            raise ValueError(
                "StabilityReport is IDLE but names a candidate; an idle report has no "
                "candidate and no run"
            )
        if self.candidate is None and self.candidate_observations != 0:
            raise ValueError("StabilityReport counts observations without naming a candidate")
        if (self.accepted is not None) is not (self.state is FilterState.STABLE):
            raise ValueError(
                f"StabilityReport.state is {self.state} but accepted is {self.accepted!r}; "
                f"only a STABLE report accepts a gesture"
            )
        if self.accepted is not None and self.accepted is not self.candidate:
            raise ValueError("StabilityReport accepts a gesture other than its candidate")

    @property
    def is_stable(self) -> bool:
        """Whether a gesture is accepted as stable right now."""
        return self.state is FilterState.STABLE

    @property
    def has_accepted_gesture(self) -> bool:
        """Whether an accepted gesture exists, i.e. :attr:`accepted` is not ``None``."""
        return self.accepted is not None

    def __str__(self) -> str:
        candidate = self.candidate if self.candidate is not None else "none"
        return (
            f"{self.state} candidate={candidate} "
            f"observations={self.candidate_observations} "
            f"accepted={self.accepted if self.accepted is not None else 'none'}"
        )


def _require_number(label: str, value: object) -> None:
    """Reject a value that is not a real, finite number, or a bool in its place."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")


__all__ = [
    "CandidateChangePolicy",
    "ConfidenceFilterConfig",
    "FilterState",
    "StabilityReport",
]
