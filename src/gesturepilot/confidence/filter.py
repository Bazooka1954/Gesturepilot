"""The temporal confidence filter: one classification in, one stability report out.

Why this filter
---------------
The classifier answers about one frame and is right to have no memory: it cannot tell a
hand that has been held still for a second from one that flashed past for a single frame.
A control surface needs exactly that difference. A gesture that flickers between two
labels for a handful of frames is a misfire, and a gesture that is seen once and gone is a
false positive. This filter removes both by requiring a *run* of qualifying observations
before it accepts anything.

The whole state is three values
-------------------------------
``_candidate``, ``_candidate_observations``, and ``_confidence_sum``. Everything the caller
sees — :attr:`~gesturepilot.confidence.filter.ConfidenceFilter.state`, the accepted
gesture, the mean confidence — is derived from those three and the configuration, never
stored alongside them. That is deliberate: there is no way for a separate ``is_stable``
flag or an ``accepted`` field to disagree with the run that produced it, and
:meth:`ConfidenceFilter.reset` is a single assignment because there is nothing else to
forget.

No hidden clock, and no frame rate
----------------------------------
Stability is counted in **observations**. The pipeline's frame rate is a property of the
camera and the machine, so a filter measured in seconds would accept a gesture twice as
fast on a fast machine. A caller that wants time-based decay supplies an optional
``timestamp`` per observation and sets
:attr:`~gesturepilot.confidence.types.ConfidenceFilterConfig.max_interruption_seconds`.

The timestamp policy, in full, because a partial one is worse than none:

* **Omitted** is always legal. Omitting it means "no time information"; the observation
  still counts, and the filter forgets the last timestamp so it cannot measure a gap
  across a period it did not witness.
* **Equal** to the last timestamp is legal. Unlike
  :class:`~gesturepilot.processing.filters.OneEuroFilter`, which divides by the interval
  and so cannot tolerate a zero one, the filter only compares a gap against a threshold, and
  a zero gap is shorter than any threshold. Two observations may legitimately share an
  instant when a caller batches, or when the frame clock is coarser than the frame rate.
* **Backwards** raises :class:`~gesturepilot.confidence.errors.FilterStateError`. It means
  the caller handed over observations out of order, and silently producing a stability
  report from unordered evidence would hide that until it surfaced as a gesture that
  inexplicably refuses to stabilise.
* **Too far ahead** is not an error and not special: if the gap exceeds
  ``max_interruption_seconds`` the run is discarded and the report says so via
  :attr:`~gesturepilot.confidence.types.StabilityReport.interrupted`.

What it does not do
-------------------
It does not dispatch actions, emit events, hold a gesture confirmed for a user-visible
duration, or decide whether an accepted gesture may cause anything. This layer answers one
question — is this gesture stable right now — and a later safety state machine answers the
question of what a stable gesture is allowed to do.

Privacy: this module handles numbers only. No file, network, clock, or logging access.
"""

from __future__ import annotations

import math

from gesturepilot.classifier.types import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    Gesture,
    GestureClassification,
)
from gesturepilot.confidence.errors import FilterStateError, InvalidClassificationError
from gesturepilot.confidence.types import (
    CandidateChangePolicy,
    ConfidenceFilterConfig,
    FilterState,
    StabilityReport,
)


class ConfidenceFilter:
    """Accumulates classifications and reports when a gesture is stable.

    Args:
        config: Thresholds and policies. Defaults to
            :class:`~gesturepilot.confidence.types.ConfidenceFilterConfig`'s own, which
            require three qualifying observations and so never accept a gesture on its
            first appearance.

    One filter instance follows one subject. A caller tracking several hands must hold one
    filter per hand, or evidence from different hands could be added up into a stability
    that neither of them ever showed.
    """

    def __init__(self, config: ConfidenceFilterConfig | None = None) -> None:
        if config is not None and not isinstance(config, ConfidenceFilterConfig):
            raise TypeError(f"config must be a ConfidenceFilterConfig, got {type(config).__name__}")
        self._config = config or ConfidenceFilterConfig()
        self._clear_run()
        self._last_timestamp: float | None = None

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> ConfidenceFilterConfig:
        """The parameters this filter was built with."""
        return self._config

    @property
    def state(self) -> FilterState:
        """How much the filter currently believes its candidate, derived from the run."""
        if self._candidate is None:
            return FilterState.IDLE
        if self._candidate_observations >= self._config.min_stable_observations:
            return FilterState.STABLE
        return FilterState.CANDIDATE

    @property
    def candidate(self) -> Gesture | None:
        """The gesture currently accumulating evidence, or ``None``."""
        return self._candidate

    @property
    def candidate_observations(self) -> int:
        """How many qualifying observations the current run holds."""
        return self._candidate_observations

    @property
    def accepted(self) -> Gesture | None:
        """The accepted gesture, or ``None`` when nothing is stable."""
        return self._candidate if self.state is FilterState.STABLE else None

    @property
    def is_stable(self) -> bool:
        """Whether a gesture is accepted as stable right now."""
        return self.state is FilterState.STABLE

    @property
    def last_timestamp(self) -> float | None:
        """Timestamp of the most recent observation, or ``None`` if none carried one."""
        return self._last_timestamp

    # -- filtering --------------------------------------------------------

    def observe(
        self,
        classification: GestureClassification,
        *,
        timestamp: float | None = None,
    ) -> StabilityReport:
        """Fold one observation into the run and report the resulting stability.

        The classification is never modified. An observation that does not qualify — an
        ``UNKNOWN``, or a confidence below ``min_confidence`` — is not evidence of
        anything, so it ends the current run rather than being counted toward it or
        averaged into it. The filter therefore reports the present, not the past: losing
        stability does not mean an action was cancelled, only that there is nothing stable
        to act on right now.

        Args:
            classification: The classifier's result for this observation.
            timestamp: Optional monotonic seconds for this observation. See the module
                docstring for the full policy on equal, missing, and backwards timestamps.
                Required to take effect only when
                ``config.max_interruption_seconds`` is set.

        Returns:
            A :class:`~gesturepilot.confidence.types.StabilityReport` describing the run
            after this observation.

        Raises:
            InvalidClassificationError: If ``classification`` is not a
                :class:`~gesturepilot.classifier.types.GestureClassification`, or its
                confidence is not a finite number in ``[0.0, 1.0]``.
            FilterStateError: If ``timestamp`` is not a real, finite, non-negative number,
                or moves backwards relative to the last one seen.
        """
        self._validate_classification(classification)
        self._validate_timestamp(timestamp)

        interrupted = self._discard_run_if_interrupted(timestamp)

        if self._qualifies(classification):
            self._accumulate(classification)
        else:
            # Not evidence of anything, so it is not evidence *against* the run either: it
            # ends the run rather than being counted toward it or averaged into it.
            self._clear_run()

        report = StabilityReport(
            state=self.state,
            candidate=self._candidate,
            candidate_observations=self._candidate_observations,
            accepted=self.accepted,
            latest_confidence=classification.confidence,
            mean_confidence=self._mean_confidence(),
            interrupted=interrupted,
            classification=classification,
        )
        if timestamp is not None:
            self._last_timestamp = timestamp
        else:
            # A gap that was not witnessed cannot be measured across; forgetting the
            # timestamp keeps a later one from being compared against it.
            self._last_timestamp = None
        return report

    def reset(self) -> None:
        """Discard the accumulated run and the last timestamp.

        For a caller that has lost track of the subject — a hand left the frame, the
        camera was reopened — rather than letting the filter believe it has been watching
        continuously.
        """
        self._clear_run()
        self._last_timestamp = None

    def __repr__(self) -> str:
        return (
            f"ConfidenceFilter(config={self._config!r}, state={self.state}, "
            f"candidate={self._candidate}, "
            f"observations={self._candidate_observations})"
        )

    # -- internals --------------------------------------------------------

    def _clear_run(self) -> None:
        self._candidate: Gesture | None = None
        self._candidate_observations = 0
        self._confidence_sum = 0.0

    def _mean_confidence(self) -> float | None:
        """The mean classifier confidence over the run, or ``None`` when the run is empty."""
        if self._candidate_observations == 0:
            return None
        return self._confidence_sum / self._candidate_observations

    def _qualifies(self, classification: GestureClassification) -> bool:
        """Whether this observation is evidence for a gesture at all.

        ``UNKNOWN`` never is: it is the classifier saying the evidence did not separate
        any gesture, and no number of repeats makes a gesture out of it. A recognised
        gesture qualifies when its confidence reaches ``min_confidence`` — at the
        threshold exactly, not only above it.
        """
        return (
            classification.is_recognized
            and classification.confidence >= self._config.min_confidence
        )

    def _accumulate(self, classification: GestureClassification) -> None:
        """Add a qualifying observation to the run, applying the candidate-change policy."""
        gesture = classification.gesture
        if self._candidate is None or self._candidate is gesture:
            self._candidate = gesture
            self._candidate_observations += 1
            self._confidence_sum += classification.confidence
        elif self._config.candidate_change is CandidateChangePolicy.RESET:
            self._candidate = gesture
            self._candidate_observations = 1
            self._confidence_sum = classification.confidence
        else:
            # HOLD: a differing qualifying observation is a fluctuation and is discarded.
            # The incumbent's run stands, so the candidate cannot be swapped sideways.
            return

    def _discard_run_if_interrupted(self, timestamp: float | None) -> bool:
        """Drop the run when the gap since the last observation is too long to bridge.

        Returns whether an interruption was treated as having happened, which includes the
        case where there was no run to discard: the sequence was still broken, and a caller
        logging gaps wants to see that.
        """
        limit = self._config.max_interruption_seconds
        if limit is None or timestamp is None or self._last_timestamp is None:
            return False
        if timestamp - self._last_timestamp > limit:
            self._clear_run()
            return True
        return False

    @staticmethod
    def _validate_classification(classification: GestureClassification) -> None:
        if not isinstance(classification, GestureClassification):
            raise InvalidClassificationError(
                f"classification must be a GestureClassification, got "
                f"{type(classification).__name__}"
            )
        confidence = classification.confidence
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise InvalidClassificationError(
                f"GestureClassification.confidence must be a real number, got {confidence!r}"
            )
        if not math.isfinite(confidence):
            raise InvalidClassificationError(
                f"GestureClassification.confidence must be finite, got {confidence!r}"
            )
        if not MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE:
            # Unreachable through the constructor, so this only fires on a forged result.
            # Letting one through would be worse than raising: it would be published in
            # StabilityReport.latest_confidence, which is documented as the classifier's own
            # score, and a score outside [0, 1] there is a lie about what the classifier said.
            raise InvalidClassificationError(
                f"GestureClassification.confidence must be within "
                f"[{MIN_CONFIDENCE}, {MAX_CONFIDENCE}], got {confidence!r}"
            )

    def _validate_timestamp(self, timestamp: float | None) -> None:
        if timestamp is None:
            return
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise FilterStateError(f"timestamp must be a real number, got {timestamp!r}")
        if not math.isfinite(timestamp):
            raise FilterStateError(f"timestamp must be finite, got {timestamp!r}")
        if timestamp < 0.0:
            raise FilterStateError(f"timestamp must be >= 0.0, got {timestamp!r}")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise FilterStateError(
                f"ConfidenceFilter timestamps must not go backwards; got {timestamp} after "
                f"{self._last_timestamp}."
            )


__all__ = ["ConfidenceFilter"]
