"""The landmark processor: one tracked hand in, one :class:`HandFeatures` out.

This is the whole public surface of the landmark-processing stage. It takes a
:class:`~gesturepilot.tracking.types.TrackedHand` from the tracking layer, optionally
smooths it, normalises it, and measures its geometry.

What it deliberately does not do
-------------------------------

It makes **no gesture judgement**. It does not know that a hand is open, closed,
pointing, or pinching, and it has no threshold anywhere that could encode such a thing.
Everything it reports is a description: where the joints are relative to the palm, how
long the bones are, how the fingers are angled, how the hand is oriented. Deciding what
those numbers *mean* is the classifier's job, and that is a later phase.

It also does not filter *out* low-confidence detections. Rejecting unreliable
classifications is a separate pipeline stage with its own semantics, and folding that
decision in here would hide it.

Input and output
----------------

The input hand is read, never written. Every output object is newly allocated, so the
caller's :class:`~gesturepilot.tracking.types.TrackedHand` — and the
:class:`~gesturepilot.tracking.types.Landmark` objects inside it — are exactly as they
were before :meth:`LandmarkProcessor.process` was called. Smoothing filters produce new
coordinates instead of writing through.

Determinism
-----------

With smoothing disabled — the default — :meth:`process` is a pure function of its
arguments: the same hand and the same configuration always give the same features. With
smoothing enabled it is stateful, and it is still deterministic, but the caller supplies
the time and therefore controls the trajectory.

Filter state lifecycle
----------------------

When smoothing is on, the processor keeps one
:class:`~gesturepilot.processing.filters.LandmarkSmoother` per *hand slot*. A slot is a
:class:`~gesturepilot.tracking.types.Handedness` label, and that is all it is — a place to
keep state, not a claim about which hand it belongs to. The tracker provides no
persistent hand identity, and inventing one would be a guess.

That has a consequence worth stating plainly: **smoothing state is only as good as the
handedness label.** If two hands swap labels between frames, one smoother sees a jump it
reads as fast motion. Call :meth:`reset` when tracking restarts, when the hands in view
change, or when the label changes unexpectedly — and :meth:`reset` is cheap enough to
call whenever in doubt. Note that ``Handedness.UNKNOWN`` is a slot like any other, so
two unlabelled hands still share one smoother; pass a stable label, or reset, if that
matters.
"""

from __future__ import annotations

import math

from gesturepilot.processing.errors import FilterStateError, InvalidLandmarksError, ProcessingError
from gesturepilot.processing.filters import LandmarkSmoother
from gesturepilot.processing.geometry import (
    distance_2d,
    landmark_point,
    optional_angle_at,
    orientation_from_landmarks,
)
from gesturepilot.processing.normalization import normalize_hand
from gesturepilot.processing.topology import (
    FINGER_CHAINS,
    FingerChain,
    LandmarkIndex,
)
from gesturepilot.processing.types import (
    FingerGeometry,
    HandFeatures,
    NormalizedHand,
    NormalizedLandmark,
    PalmOrientation,
    ProcessedLandmark,
    ProcessorConfig,
)
from gesturepilot.tracking.types import LANDMARK_COUNT, Handedness, TrackedHand


class LandmarkProcessor:
    """Derives gesture-ready geometric features from a tracked hand.

    Args:
        config: Feature and smoothing settings. Defaults to
            :class:`~gesturepilot.processing.types.ProcessorConfig`'s own: smoothing
            disabled, so the processor is a pure function until told otherwise.

    Typical use::

        processor = LandmarkProcessor(ProcessorConfig(smoothing=True))
        features = processor.process(hand, timestamp=result.timestamp)
        ...
        processor.reset()   # e.g. when tracking restarts
    """

    def __init__(self, config: ProcessorConfig | None = None) -> None:
        if config is not None and not isinstance(config, ProcessorConfig):
            raise TypeError(f"config must be a ProcessorConfig, got {type(config).__name__}")
        self._config = config or ProcessorConfig()
        self._smoothers: dict[Handedness, LandmarkSmoother] = {}

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> ProcessorConfig:
        """The configuration this processor was built with."""
        return self._config

    @property
    def is_filtering(self) -> bool:
        """Whether :meth:`process` runs the One Euro filter before measuring.

        ``False`` means the processor is stateless and needs no timestamp.
        """
        return self._config.smoothing

    @property
    def active_slots(self) -> tuple[Handedness, ...]:
        """Hand slots currently holding filter state, in declaration order.

        Always empty when :attr:`is_filtering` is ``False``. Each entry means a
        :class:`~gesturepilot.tracking.types.Handedness` label has been smoothed at least
        once since the last reset.
        """
        return tuple(slot for slot in Handedness if slot in self._smoothers)

    # -- processing -------------------------------------------------------

    def process(
        self,
        hand: TrackedHand,
        *,
        timestamp: float | None = None,
        frame_sequence: int | None = None,
    ) -> HandFeatures:
        """Measure one hand and return its features.

        Args:
            hand: The tracked hand to process. Read only — never modified.
            timestamp: Monotonic seconds for the originating frame. Required when
                smoothing is enabled, since the filter's time step comes from it.
                Optional otherwise, and simply recorded on the output.
            frame_sequence: The originating frame's sequence number. Optional, and
                simply recorded on the output.

        Returns:
            A :class:`~gesturepilot.processing.types.HandFeatures`. Every derived value
            is finite or explicitly ``None``; this never returns a partly-filled result
            to signal that something went wrong.

        Raises:
            InvalidLandmarksError: If ``hand`` is not a usable :class:`TrackedHand`, or
                carries a non-finite coordinate.
            DegenerateHandError: If the hand has no measurable size, so nothing can be
                normalised against it.
            ProcessingError: If smoothing is enabled but no usable ``timestamp`` was
                supplied.
            FilterStateError: If smoothing is enabled and the timestamps do not
                increase.
        """
        source = self._read_landmarks(hand)
        slot = self._validate_handedness(hand.handedness)
        moment = self._validate_timestamp(timestamp)
        sequence = self._validate_sequence(frame_sequence)

        landmarks = self._smooth(source, slot, moment)

        normalization = normalize_hand(landmarks)

        return HandFeatures(
            landmarks=landmarks,
            normalization=normalization,
            fingers=tuple(self._measure_finger(chain, normalization) for chain in FINGER_CHAINS),
            orientation=self._measure_orientation(normalization.landmarks),
            handedness=slot,
            handedness_score=hand.handedness_score,
            detection_score=hand.detection_score,
            timestamp=moment,
            frame_sequence=sequence,
        )

    def reset(self, handedness: Handedness | None = None) -> None:
        """Discard filter state.

        Args:
            handedness: Discard only this slot's state. ``None`` — the default —
                discards every slot, which is what a caller wants when tracking
                restarts or when the set of hands in view changes.

        Does nothing when smoothing is disabled, so it is always safe to call.
        """
        if handedness is None:
            self._smoothers.clear()
            return
        self._smoothers.pop(handedness, None)

    # -- internals --------------------------------------------------------

    def _read_landmarks(self, hand: TrackedHand) -> tuple[ProcessedLandmark, ...]:
        """Validate the input and copy it into processing-owned values."""
        if not isinstance(hand, TrackedHand):
            raise InvalidLandmarksError(f"Expected a TrackedHand, got {type(hand).__name__}.")
        if hand.landmark_count != LANDMARK_COUNT:
            raise InvalidLandmarksError(
                f"A hand needs exactly {LANDMARK_COUNT} landmarks, got {hand.landmark_count}."
            )

        # TrackedHand already rejects non-finite coordinates, but this layer is the last
        # place before the arithmetic, and a duck-typed or substituted object must not be
        # able to slip a NaN into every feature downstream.
        landmarks: list[ProcessedLandmark] = []
        for index, landmark in enumerate(hand.landmarks):
            for name, value in (("x", landmark.x), ("y", landmark.y), ("z", landmark.z)):
                if not math.isfinite(value):
                    raise InvalidLandmarksError(
                        f"Landmark {index} has a non-finite {name} ({value!r})."
                    )
            landmarks.append(ProcessedLandmark(x=landmark.x, y=landmark.y, z=landmark.z))
        return tuple(landmarks)

    def _validate_handedness(self, handedness: object) -> Handedness:
        if not isinstance(handedness, Handedness):
            raise InvalidLandmarksError(
                f"handedness must be a Handedness, got {type(handedness).__name__}."
            )
        return handedness

    def _validate_timestamp(self, timestamp: float | None) -> float | None:
        if timestamp is None:
            return None
        if not math.isfinite(timestamp):
            raise InvalidLandmarksError(f"Timestamp must be finite, got {timestamp!r}.")
        return timestamp

    def _validate_sequence(self, frame_sequence: int | None) -> int | None:
        if frame_sequence is None:
            return None
        if frame_sequence < 0:
            raise InvalidLandmarksError(f"frame_sequence must be >= 0, got {frame_sequence}.")
        return frame_sequence

    def _smooth(
        self,
        landmarks: tuple[ProcessedLandmark, ...],
        slot: Handedness,
        timestamp: float | None,
    ) -> tuple[ProcessedLandmark, ...]:
        """Run the One Euro filter when enabled, otherwise pass the values straight through."""
        if not self._config.smoothing:
            return landmarks

        if timestamp is None:
            raise ProcessingError(
                "Smoothing is enabled, so process() needs the frame's monotonic timestamp. "
                "Pass timestamp=result.timestamp, or disable smoothing with "
                "ProcessorConfig(smoothing=False)."
            )

        smoother = self._smoothers.get(slot)
        if smoother is None:
            smoother = LandmarkSmoother(self._config.filter)
            self._smoothers[slot] = smoother

        try:
            return smoother.filter(landmarks, timestamp)
        except FilterStateError as exc:
            raise FilterStateError(
                f"Smoothing state for handedness '{slot.value}' rejected this frame: {exc} "
                "Call processor.reset() if the tracker was restarted or the hands swapped."
            ) from exc

    def _measure_finger(self, chain: FingerChain, normalized: NormalizedHand) -> FingerGeometry:
        """Measure one finger chain in normalised hand space.

        All measurements are image-plane: ``x`` and ``y`` come from the image and are
        metric relative to each other, whereas the backend's ``z`` is a noisy depth
        estimate only roughly commensurate with them. The depth axis is still carried
        through in the normalised landmarks for any later stage that wants it.
        """
        points = normalized.landmarks
        root, base, middle, distal, tip = tuple(
            landmark_point(points[index]) for index in chain.polyline
        )
        tip_point = landmark_point(points[chain.tip])
        palm_centre = landmark_point(normalized.palm_center)

        return FingerGeometry(
            finger=chain.finger,
            proximal_angle=optional_angle_at(root, base, middle),
            middle_angle=optional_angle_at(base, middle, distal),
            distal_angle=optional_angle_at(middle, distal, tip),
            segment_lengths=(
                distance_2d(root, base),
                distance_2d(base, middle),
                distance_2d(middle, distal),
                distance_2d(distal, tip),
            ),
            tip_to_palm=distance_2d(tip_point, palm_centre),
            tip_to_index_mcp=distance_2d(
                tip_point, landmark_point(points[LandmarkIndex.INDEX_MCP])
            ),
        )

    def _measure_orientation(self, normalized: tuple[NormalizedLandmark, ...]) -> PalmOrientation:
        roll, tilt = orientation_from_landmarks(normalized)
        return PalmOrientation(roll=roll, tilt=tilt)


__all__ = ["LandmarkProcessor"]
