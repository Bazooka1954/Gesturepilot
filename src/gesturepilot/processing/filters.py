"""A One Euro filter for temporally smoothing landmark coordinates.

Why this filter
---------------

Landmark coordinates jitter frame to frame even when the hand is perfectly still. A
fixed low-pass filter would remove that jitter, but it has to choose between lag and
noise at a single fixed cutoff, and a cutoff quiet enough for a resting hand turns a
deliberate movement into a visible delay. That delay matters here: a gesture that takes
200 ms longer to register than the user's hand actually moved is a gesture that can feel
unresponsive.

The One Euro filter resolves that with an adaptive cutoff. It low-passes the *velocity*
of the signal, and raises the cutoff when that velocity is high — heavy smoothing when
the hand is still, light smoothing when it moves quickly, with no manual tuning per
gesture. It is a few dozen lines, needs no dependencies, and is deterministic given a
timestamp.

Design constraints honoured here
--------------------------------

**No hidden clock.** Every call carries its own timestamp. The filter never reads a
wall clock, so a test drives it with literal numbers and gets the same answer on every
run and every machine. This matches how the camera layer takes timestamps from the frame
rather than re-measuring them.

**Timestamps must increase.** A non-increasing timestamp means the caller has handed the
filter frames out of order or repeated one. Rather than produce a plausible-looking but
wrong number, the filter raises :class:`~gesturepilot.processing.errors.FilterStateError`.
Silence here would surface much later as a gesture that inexplicably misfires.

**Reset is explicit.** State is discarded, not aged out. A caller that loses track of a
hand should call :meth:`OneEuroFilter.reset` rather than let the filter believe it has
been watching a continuous signal.

**One instance per hand.** :class:`LandmarkSmoother` bundles the 21 x 3 scalar filters a
hand needs, so a caller holding two hands holds two smoothers and no state leaks between
them.

Privacy: this module handles numbers only. No file, network, or logging access.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from gesturepilot.processing.errors import FilterStateError, InvalidLandmarksError
from gesturepilot.processing.types import FilterConfig, ProcessedLandmark
from gesturepilot.tracking.types import LANDMARK_COUNT

_TWO_PI = 2.0 * math.pi


def _smoothing_factor(cutoff: float, dt: float) -> float:
    """Return the exponential smoothing factor for a cutoff frequency over ``dt`` seconds.

    This is the standard one-euro-pose smoothing coefficient::

        alpha = 1 / (1 + 1 / (2 * pi * cutoff * dt))

    It behaves as expected at both extremes: ``dt -> 0`` approaches ``1`` (trust the new
    sample almost entirely), and a large ``dt`` approaches ``0`` (hold the history).
    """
    tau = 1.0 / (_TWO_PI * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar stream.

    Args:
        config: Cutoff and speed parameters. Defaults to
            :class:`~gesturepilot.processing.types.FilterConfig`'s own.

    The filter holds the last output value, an estimate of the signal's velocity, and
    the timestamp it last saw. Nothing else, and no wall-clock access.
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        if config is not None and not isinstance(config, FilterConfig):
            raise TypeError(f"config must be a FilterConfig, got {type(config).__name__}")
        self._config = config or FilterConfig()
        self._reset_state()

    def _reset_state(self) -> None:
        self._initialized = False
        self._previous_value = 0.0
        self._previous_output = 0.0
        self._derivative = 0.0
        self._last_timestamp: float | None = None

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> FilterConfig:
        """The parameters this filter was built with."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Whether a sample has been seen since the last reset."""
        return self._initialized

    @property
    def last_timestamp(self) -> float | None:
        """Timestamp of the most recent accepted sample, or ``None`` before the first."""
        return self._last_timestamp

    # -- filtering --------------------------------------------------------

    def filter(self, value: float, timestamp: float) -> float:
        """Filter one sample and return the smoothed value.

        The first sample after construction or :meth:`reset` is passed through
        unchanged: with no history there is nothing to smooth against, and inventing a
        ramp from an assumed zero would be a lie about where the hand was.

        Args:
            value: The new measurement.
            timestamp: Monotonic seconds. Must be strictly greater than the previous
                accepted timestamp.

        Returns:
            The smoothed value, in the same units as ``value``.

        Raises:
            InvalidLandmarksError: If ``value`` is not finite.
            FilterStateError: If ``timestamp`` is not finite, or does not increase.
        """
        value = float(value)
        timestamp = float(timestamp)

        if not math.isfinite(value):
            raise InvalidLandmarksError(f"Cannot filter a non-finite sample: {value!r}")
        if not math.isfinite(timestamp):
            raise FilterStateError(f"Filter timestamp must be finite, got {timestamp!r}")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise FilterStateError(
                f"Filter timestamps must strictly increase; got {timestamp} after "
                f"{self._last_timestamp}."
            )

        if not self._initialized:
            self._initialized = True
            self._derivative = 0.0
            output = value
        else:
            assert self._last_timestamp is not None  # guarded by the check above
            dt = timestamp - self._last_timestamp
            derivative = (value - self._previous_value) / dt
            derivative_alpha = _smoothing_factor(self._config.derivative_cutoff, dt)
            self._derivative = (
                derivative_alpha * derivative + (1.0 - derivative_alpha) * self._derivative
            )
            cutoff = self._config.min_cutoff + self._config.beta * abs(self._derivative)
            alpha = _smoothing_factor(cutoff, dt)
            output = alpha * value + (1.0 - alpha) * self._previous_output

        self._previous_value = value
        self._previous_output = output
        self._last_timestamp = timestamp
        return output

    def reset(self) -> None:
        """Discard all state, so the next sample is again passed through unchanged."""
        self._reset_state()

    def __repr__(self) -> str:
        return (
            f"OneEuroFilter(min_cutoff={self._config.min_cutoff}, "
            f"beta={self._config.beta}, "
            f"derivative_cutoff={self._config.derivative_cutoff}, "
            f"initialized={self._initialized})"
        )


class LandmarkSmoother:
    """One :class:`OneEuroFilter` per landmark coordinate, treated as one unit.

    Holds ``3 x LANDMARK_COUNT`` scalar filters and applies them in step, so a whole hand
    is smoothed coherently: all coordinates share one timestamp, and the landmark count
    is checked once rather than per coordinate.

    A smoother belongs to exactly one hand. Give each hand its own instance and no state
    can leak between them.

    Args:
        config: Passed to every underlying :class:`OneEuroFilter`.
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        self._config = config or FilterConfig()
        if not isinstance(self._config, FilterConfig):
            raise TypeError(f"config must be a FilterConfig, got {type(self._config).__name__}")
        self._filters = tuple(OneEuroFilter(self._config) for _ in range(LANDMARK_COUNT * 3))
        self._landmark_count: int | None = None

    @property
    def config(self) -> FilterConfig:
        """The parameters this smoother was built with."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Whether a full landmark set has been filtered since the last reset."""
        return self._filters[0].is_initialized

    @property
    def last_timestamp(self) -> float | None:
        """Timestamp of the most recent accepted frame, or ``None`` before the first."""
        return self._filters[0].last_timestamp

    def filter(
        self,
        landmarks: Sequence[ProcessedLandmark],
        timestamp: float,
    ) -> tuple[ProcessedLandmark, ...]:
        """Filter one frame's worth of landmarks and return new values.

        The input landmarks are never modified.

        Args:
            landmarks: Exactly ``LANDMARK_COUNT`` landmarks in topology order.
            timestamp: Monotonic seconds for this frame.

        Returns:
            A new tuple of smoothed landmarks, in the same order.

        Raises:
            InvalidLandmarksError: If the landmark count is wrong.
            FilterStateError: If ``timestamp`` does not increase.
        """
        count = len(landmarks)
        if count != LANDMARK_COUNT:
            raise InvalidLandmarksError(
                f"LandmarkSmoother needs exactly {LANDMARK_COUNT} landmarks, got {count}"
            )
        self._landmark_count = count

        smoothed: list[ProcessedLandmark] = []
        for index, landmark in enumerate(landmarks):
            base = index * 3
            smoothed.append(
                ProcessedLandmark(
                    x=self._filters[base].filter(landmark.x, timestamp),
                    y=self._filters[base + 1].filter(landmark.y, timestamp),
                    z=self._filters[base + 2].filter(landmark.z, timestamp),
                )
            )
        return tuple(smoothed)

    def reset(self) -> None:
        """Discard all filter state, so the next frame is again passed through unchanged."""
        for coordinate_filter in self._filters:
            coordinate_filter.reset()

    def __repr__(self) -> str:
        return f"LandmarkSmoother({self._config!r}, initialized={self.is_initialized})"


__all__ = ["LandmarkSmoother", "OneEuroFilter"]
