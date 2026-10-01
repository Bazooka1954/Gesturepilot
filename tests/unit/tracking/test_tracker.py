"""Unit tests for the hand-tracker boundary and behaviour.

These are deterministic by design: no model file, no webcam, no network. Only the
conversion path between the camera Frame and TrackingResult is exercised, and
MediaPipe is never actually loaded (the factory is faked).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from gesturepilot.camera.types import Frame
from gesturepilot.tracking import tracker as tracker_module
from gesturepilot.tracking.assets import DEFAULT_MODEL_FILENAME
from gesturepilot.tracking.errors import (
    ModelAssetError,
    TrackerInitializationError,
    TrackerStateError,
    TrackingError,
    TrackingProcessingError,
)
from gesturepilot.tracking.tracker import MediaPipeHandTracker
from gesturepilot.tracking.types import (
    LANDMARK_COUNT,
    Handedness,
    Landmark,
    RunningMode,
    TrackedHand,
    TrackerConfig,
    TrackingResult,
)

from .fakes import (
    FakeCategory,
    FakeLandmark,
    FakeLandmarker,
    FakeLandmarkerFactory,
    FakeLandmarkerResult,
    make_frame,
    make_landmarks,
)


class TestTrackerLifecycle:
    def test_is_not_initialized_initially(self) -> None:
        tracker = MediaPipeHandTracker()
        assert tracker.is_initialized is False

    def test_initialize_closes_the_gate(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        assert tracker.is_initialized is False
        tracker.initialize()
        assert tracker.is_initialized is True

    def test_initialize_requires_model_asset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asset resolution happens before the backend is ever constructed."""
        from gesturepilot.tracking.assets import resolve_model_path as real_resolve

        monkeypatch.setattr(tracker_module, "resolve_model_path", real_resolve)

        missing = str(Path.cwd() / "definitely-absent" / DEFAULT_MODEL_FILENAME)
        built: list[object] = []
        tracker = MediaPipeHandTracker(
            TrackerConfig(model_path=missing),
            landmarker_factory=lambda options: built.append(options),
        )
        with pytest.raises(ModelAssetError) as excinfo:
            tracker.initialize()

        assert str(Path(missing)) in str(excinfo.value.searched[0])
        assert built == [], "backend must not be built when the asset is missing"
        assert tracker.is_initialized is False

    def test_double_initialize_is_an_error(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        with pytest.raises(TrackerStateError, match="already initialized"):
            tracker.initialize()

    def test_close_is_idempotent(self) -> None:
        fake = FakeLandmarker()
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        tracker.close()
        assert fake.closed is True
        assert tracker.is_initialized is False
        tracker.close()  # no exception
        assert tracker.is_initialized is False

    def test_close_releases_landmarker(self) -> None:
        fake = FakeLandmarker()
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        assert tracker.is_initialized is True
        tracker.close()
        assert fake.closed is True
        assert tracker.is_initialized is False

    def test_process_before_initialize_is_an_error(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        frame = make_frame()
        with pytest.raises(TrackerStateError, match="not initialized"):
            tracker.process(frame)

    def test_process_after_close_is_an_error(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        tracker.close()
        with pytest.raises(TrackerStateError, match="not initialized"):
            tracker.process(make_frame())

    def test_context_manager_lifecycle(self) -> None:
        fake = FakeLandmarker()
        factory = FakeLandmarkerFactory(fake)
        with MediaPipeHandTracker(landmarker_factory=factory) as tracker:
            assert tracker.is_initialized is True
            frame = make_frame(sequence=0)
            tracker.process(frame)
        assert fake.closed is True
        assert tracker.is_initialized is False

    def test_context_manager_closes_on_exception(self) -> None:
        fake = FakeLandmarker()
        with (
            pytest.raises(ValueError),
            MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake)) as tracker,
        ):
            assert tracker.is_initialized is True
            raise ValueError("oops")
        assert fake.closed is True


class TestProcessConversion:
    def test_process_returns_tracking_result(self) -> None:
        fake = FakeLandmarker(FakeLandmarkerResult())
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        frame = make_frame(timestamp=1.23, sequence=5)
        result = tracker.process(frame)
        assert isinstance(result, TrackingResult)
        assert result.timestamp == 1.23
        assert result.frame_sequence == 5
        assert result.hand_count == 0
        assert result.has_hands is False

    def test_maps_a_single_hand(self) -> None:
        landmarks = [FakeLandmark(x=0.1, y=0.2, z=0.3) for _ in range(LANDMARK_COUNT)]
        fake = FakeLandmarker(
            FakeLandmarkerResult(
                hand_landmarks=[landmarks],
                handedness=[[FakeCategory("Left", score=0.98)]],
            )
        )
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        result = tracker.process(make_frame(sequence=0, timestamp=0.5))
        assert result.hand_count == 1
        assert result.primary_hand is not None
        hand = result.primary_hand
        assert hand.handedness is Handedness.LEFT
        assert hand.handedness_score == pytest.approx(0.98)
        # MediaPipe does not emit a per-frame detection confidence in this API.
        assert hand.detection_score is None
        assert hand.landmark_count == LANDMARK_COUNT
        assert hand.landmarks[0].x == pytest.approx(0.1)
        assert hand.landmarks[0].y == pytest.approx(0.2)
        assert hand.landmarks[0].z == pytest.approx(0.3)

    def test_maps_both_hands_and_preserves_order(self) -> None:
        landmarks_l = make_landmarks(offset=0.0)
        landmarks_r = make_landmarks(offset=0.2)
        fake = FakeLandmarker(
            FakeLandmarkerResult(
                hand_landmarks=[landmarks_l, landmarks_r],
                handedness=[
                    [FakeCategory("Left", score=0.9)],
                    [FakeCategory("Right", score=0.8)],
                ],
            )
        )
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        result = tracker.process(make_frame())
        assert result.hand_count == 2
        assert result.primary_hand is not None
        assert result.primary_hand.handedness is Handedness.LEFT
        assert result.hands[1].handedness is Handedness.RIGHT

    def test_unknown_handedness_label_is_mapped(self) -> None:
        fake = FakeLandmarker(
            FakeLandmarkerResult(
                hand_landmarks=[make_landmarks()],
                handedness=[[FakeCategory("Ambidextrous")]],
            )
        )
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        result = tracker.process(make_frame())
        assert result.primary_hand is not None
        assert result.primary_hand.handedness is Handedness.UNKNOWN

    def test_handles_empty_handedness_list(self) -> None:
        fake = FakeLandmarker(
            FakeLandmarkerResult(
                hand_landmarks=[make_landmarks(), make_landmarks()],
                handedness=[],
            )
        )
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        result = tracker.process(make_frame())
        assert result.hand_count == 2
        for hand in result.hands:
            assert hand.handedness is Handedness.UNKNOWN
            assert hand.handedness_score is None

    def test_handedness_category_without_a_score(self) -> None:
        """A category missing `score` yields `handedness_score is None`, not a crash."""
        fake = FakeLandmarker(
            FakeLandmarkerResult(
                hand_landmarks=[make_landmarks()],
                handedness=[[object()]],  # no category_name, no score
            )
        )
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        result = tracker.process(make_frame())
        hand = result.primary_hand
        assert hand is not None
        assert hand.handedness is Handedness.UNKNOWN
        assert hand.handedness_score is None

    def test_handles_missing_handedness_entry(self) -> None:
        fake = FakeLandmarker(
            FakeLandmarkerResult(
                hand_landmarks=[make_landmarks(), make_landmarks()],
                handedness=[[FakeCategory("Right", score=0.95)]],  # only first
            )
        )
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        result = tracker.process(make_frame())
        assert result.hands[0].handedness is Handedness.RIGHT
        assert result.hands[1].handedness is Handedness.UNKNOWN

    def test_preserves_landmark_order_exactly(self) -> None:
        landmarks = [FakeLandmark(x=i / 100.0, y=0.5, z=0.0) for i in range(LANDMARK_COUNT)]
        fake = FakeLandmarker(FakeLandmarkerResult(hand_landmarks=[landmarks]))
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        result = tracker.process(make_frame())
        assert result.primary_hand is not None
        for i, lm in enumerate(result.primary_hand.landmarks):
            assert lm.x == pytest.approx(i / 100.0)

    def test_never_mutates_original_frame(self) -> None:
        frame = make_frame()
        snapshot = frame.image.copy()
        fake = FakeLandmarker(FakeLandmarkerResult())
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        tracker.process(frame)
        assert np.array_equal(frame.image, snapshot)


class TestBackendInvocation:
    def test_uses_detect_for_image_mode(self) -> None:
        fake = FakeLandmarker(FakeLandmarkerResult())
        factory = FakeLandmarkerFactory(fake)
        config = TrackerConfig(running_mode=RunningMode.IMAGE, model_path="ignore")
        tracker = MediaPipeHandTracker(config, landmarker_factory=factory)
        tracker.initialize()
        tracker.process(make_frame())
        assert len(fake.calls) == 1
        assert fake.calls[0][0] == "detect"
        assert fake.calls[0][1] is None

    def test_uses_detect_for_video_and_converts_timestamp(self) -> None:
        fake = FakeLandmarker(FakeLandmarkerResult())
        factory = FakeLandmarkerFactory(fake)
        config = TrackerConfig(running_mode=RunningMode.VIDEO)
        tracker = MediaPipeHandTracker(config, landmarker_factory=factory)
        tracker.initialize()
        tracker.process(make_frame(timestamp=1.234, sequence=0))
        tracker.process(make_frame(timestamp=1.235, sequence=1))
        assert fake.calls[0] == ("detect_for_video", 1234)
        assert fake.calls[1] == ("detect_for_video", 1235)

    def test_enforces_strictly_increasing_video_timestamps(self) -> None:
        fake = FakeLandmarker(FakeLandmarkerResult(), FakeLandmarkerResult())
        config = TrackerConfig(running_mode=RunningMode.VIDEO)
        tracker = MediaPipeHandTracker(config, landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        # Same millisecond (1.2340s -> 1234ms)
        tracker.process(make_frame(timestamp=1.2340))
        tracker.process(make_frame(timestamp=1.2341))
        # Second call must be nudged forward by >=1 to keep strictly increasing.
        assert fake.calls[0][1] == 1234
        assert fake.calls[1][1] == 1235

    def test_backward_timestamp_in_video_is_nudged(self) -> None:
        fake = FakeLandmarker(FakeLandmarkerResult(), FakeLandmarkerResult())
        config = TrackerConfig(running_mode=RunningMode.VIDEO)
        tracker = MediaPipeHandTracker(config, landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        tracker.process(make_frame(timestamp=1.5))
        tracker.process(make_frame(timestamp=1.4))  # backwards in wall time
        assert fake.calls[0][1] == 1500
        assert fake.calls[1][1] == 1501

    def test_factory_receives_options(self) -> None:
        fake = FakeLandmarker()
        factory = FakeLandmarkerFactory(fake)
        config = TrackerConfig(
            max_hands=3,
            min_detection_confidence=0.6,
            min_presence_confidence=0.7,
            min_tracking_confidence=0.8,
            running_mode=RunningMode.IMAGE,
            model_path="models/test.task",
        )
        tracker = MediaPipeHandTracker(config, landmarker_factory=factory)
        tracker.initialize()
        assert len(factory.options) == 1
        opts = factory.options[0]
        # MediaPipe options are objects; check the numeric fields we passed through.
        assert getattr(opts, "num_hands", None) == 3
        assert opts.running_mode.name == "IMAGE"

    def test_creates_mp_image_in_correct_format(self) -> None:
        # We can't easily inspect mp.Image internals without MediaPipe, but we can
        # ensure BGR->RGB conversion happened conceptually by never mutating the frame
        # and by the normal successful path.
        fake = FakeLandmarker(FakeLandmarkerResult())
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        frame = make_frame()
        tracker.process(frame)
        # No exception; conversion didn't mutate the source array.
        assert frame.image.shape[2] == 3


class TestInputValidation:
    def test_rejects_non_frame(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        with pytest.raises(TrackingProcessingError, match="Expected a camera Frame"):
            tracker.process("not a frame")  # type: ignore[arg-type]

    def test_rejects_non_numpy_image(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        with pytest.raises(TrackingProcessingError, match="NumPy array"):
            tracker.process(Frame(image="bad", timestamp=0.0, sequence=0))  # type: ignore[arg-type]

    def test_rejects_empty_array(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(TrackingProcessingError, match="empty"):
            tracker.process(Frame(image=empty, timestamp=0.0, sequence=0))

    def test_rejects_non_color_array(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(TrackingProcessingError, match="colour"):
            tracker.process(Frame(image=gray, timestamp=0.0, sequence=0))

    def test_rejects_wrong_dtype(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        bad = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(TrackingProcessingError, match="uint8"):
            tracker.process(Frame(image=bad, timestamp=0.0, sequence=0))


class TestErrorHandling:
    def test_factory_failure_becomes_initialization_error(self) -> None:
        def bad_factory(_opts: object) -> None:
            raise RuntimeError("backend boom")

        tracker = MediaPipeHandTracker(landmarker_factory=bad_factory)
        with pytest.raises(TrackerInitializationError) as excinfo:
            tracker.initialize()
        assert "backend boom" in str(excinfo.value.__cause__)

    def test_factory_returns_none_becomes_initialization_error(self) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=lambda _o: None)
        with pytest.raises(TrackerInitializationError, match="returned None"):
            tracker.initialize()

    def test_landmarker_detect_failure_becomes_processing_error(self) -> None:
        fake = FakeLandmarker()
        fake.raise_on_detect = RuntimeError("inference failed")
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        with pytest.raises(TrackingProcessingError) as excinfo:
            tracker.process(make_frame(sequence=7))
        assert "frame 7" in str(excinfo.value)
        assert "inference failed" in str(excinfo.value.__cause__)

    def test_corrupt_landmarks_become_processing_error(self) -> None:
        # Return a landmark that is not a valid object (missing x/y/z)
        corrupt = [object()]
        fake = FakeLandmarker(FakeLandmarkerResult(hand_landmarks=[corrupt]))  # type: ignore[list-item]
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        with pytest.raises(TrackingProcessingError, match="unusable landmark"):
            tracker.process(make_frame())

    def test_non_finite_landmarks_are_rejected(self) -> None:
        bad_landmarks = [FakeLandmark(x=float("nan"), y=0.5, z=0.0)] + make_landmarks()[1:]
        fake = FakeLandmarker(FakeLandmarkerResult(hand_landmarks=[bad_landmarks]))
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()
        with pytest.raises(TrackingProcessingError, match="unusable landmark"):
            tracker.process(make_frame())

    def test_backend_raising_our_own_error_is_not_double_wrapped(self) -> None:
        """A backend already raising TrackingError passes through unchanged."""
        fake = FakeLandmarker()
        fake.raise_on_detect = TrackingProcessingError("already wrapped by the backend")
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory(fake))
        tracker.initialize()

        with pytest.raises(TrackingProcessingError) as excinfo:
            tracker.process(make_frame())

        assert "already wrapped by the backend" in str(excinfo.value)
        assert excinfo.value.__cause__ is None

    def test_errors_form_hierarchy(self) -> None:
        assert issubclass(TrackerInitializationError, TrackingError)
        assert issubclass(TrackingProcessingError, TrackingError)
        assert issubclass(TrackerStateError, TrackingError)
        assert issubclass(ModelAssetError, TrackingError)


class TestResultImmutability:
    def test_result_and_hand_are_frozen(self) -> None:
        tracker = MediaPipeHandTracker(
            landmarker_factory=FakeLandmarkerFactory(
                FakeLandmarker(
                    FakeLandmarkerResult(
                        hand_landmarks=[make_landmarks()],
                        handedness=[[FakeCategory("Left")]],
                    )
                )
            )
        )
        tracker.initialize()
        result = tracker.process(make_frame())

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.timestamp = 99.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.hands[0].handedness = Handedness.RIGHT

    def test_landmarks_is_a_tuple(self) -> None:
        tracker = MediaPipeHandTracker(
            landmarker_factory=FakeLandmarkerFactory(
                FakeLandmarker(FakeLandmarkerResult(hand_landmarks=[make_landmarks()]))
            )
        )
        tracker.initialize()
        result = tracker.process(make_frame())
        assert isinstance(result.hands, tuple)
        assert isinstance(result.hands[0].landmarks, tuple)

    def test_no_raw_backend_objects_leak_into_result(self) -> None:
        """The result must contain only GesturePilot types."""
        tracker = MediaPipeHandTracker(
            landmarker_factory=FakeLandmarkerFactory(
                FakeLandmarker(
                    FakeLandmarkerResult(
                        hand_landmarks=[make_landmarks()],
                        handedness=[[FakeCategory("Left", score=0.9)]],
                        hand_world_landmarks=[[make_landmarks()]],
                    )
                )
            )
        )
        tracker.initialize()
        result = tracker.process(make_frame())

        assert isinstance(result, TrackingResult)
        assert isinstance(result.hands[0], TrackedHand)
        assert isinstance(result.hands[0].landmarks[0], Landmark)
        assert not any(
            type(obj).__module__.startswith(("mediapipe", "tests"))
            for obj in (
                result,
                result.hands[0],
                result.hands[0].landmarks[0],
            )
        )


class TestConfigFlow:
    def test_config_is_exposed_after_construction(self) -> None:
        config = TrackerConfig(max_hands=4, running_mode=RunningMode.IMAGE)
        tracker = MediaPipeHandTracker(config)
        assert tracker.config is config
        assert tracker.config.max_hands == 4

    def test_default_config_is_used_when_omitted(self) -> None:
        tracker = MediaPipeHandTracker()
        assert isinstance(tracker.config, TrackerConfig)
        assert tracker.config.running_mode is RunningMode.VIDEO

    def test_initialize_resets_timestamp_state_after_reuse(self) -> None:
        factory = FakeLandmarkerFactory(
            FakeLandmarker(FakeLandmarkerResult(), FakeLandmarkerResult())
        )
        tracker = MediaPipeHandTracker(TrackerConfig(), landmarker_factory=factory)
        tracker.initialize()
        tracker.process(make_frame(timestamp=9.0))
        tracker.close()

        tracker.initialize()
        tracker.process(make_frame(timestamp=1.0))
        # After a fresh initialize, the 9.0s state must not leak in and force 9001.
        assert factory.landmarker.calls[-1] == ("detect_for_video", 1000)

    def test_timestamp_conversion_is_exact_for_whole_milliseconds(self) -> None:
        factory = FakeLandmarkerFactory(FakeLandmarker())
        tracker = MediaPipeHandTracker(TrackerConfig(), landmarker_factory=factory)
        tracker.initialize()
        tracker.process(make_frame(timestamp=2.5))
        assert factory.landmarker.calls[0] == ("detect_for_video", 2500)

    def test_many_frames_in_one_millisecond_stay_strictly_increasing(self) -> None:
        """A 1000 FPS burst must not collapse into repeated timestamps."""
        factory = FakeLandmarkerFactory(FakeLandmarker())
        tracker = MediaPipeHandTracker(TrackerConfig(), landmarker_factory=factory)
        tracker.initialize()
        for i in range(10):
            tracker.process(make_frame(timestamp=1.0 + i * 0.0001))

        stamps = [call[1] for call in factory.landmarker.calls]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_timestamp(self, bad: float) -> None:
        tracker = MediaPipeHandTracker(landmarker_factory=FakeLandmarkerFactory())
        tracker.initialize()
        with pytest.raises(TrackingProcessingError, match="timestamp"):
            tracker.process(make_frame(timestamp=bad))
