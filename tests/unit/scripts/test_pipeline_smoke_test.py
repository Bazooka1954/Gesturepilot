"""Unit tests for ``scripts/pipeline_smoke_test.py``.

The script is the one place all five pipeline layers meet, so these tests are organised
around the mistakes only that meeting place can make:

* a frame's own timestamp and sequence reach the next stage unchanged, rather than being
  re-derived from the wall clock
* a frame with no hand clears the smoothing and stability state, so a gesture that stopped
  being held is never still displayed as accepted
* ``UNKNOWN`` is reported as ``UNKNOWN`` and never becomes a candidate, however many times it
  repeats
* the console is throttled without a thread, a timer, or a clock
* a bad option is refused before the camera is opened, and a missing model asset is reported
  without touching hardware at all

The script is a development tool, not a shipped module, so it is loaded by path rather than
imported as part of the package.

The real :class:`ConfidenceFilter` is used throughout, because the interesting behaviour is
how the loop *uses* its reports rather than what the filter does with them (that is already
covered in ``tests/unit/confidence``). Only the hardware boundary is faked: no camera, no
MediaPipe, no model, no console keypress.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import numpy as np
import pytest

from gesturepilot.camera import Frame
from gesturepilot.classifier import GestureClassifier
from gesturepilot.classifier.types import Gesture, GestureClassification
from gesturepilot.confidence import ConfidenceFilter
from gesturepilot.processing.types import HandFeatures
from gesturepilot.tracking.types import LANDMARK_COUNT, Landmark, TrackedHand, TrackingResult

from ..classifier.fixtures import point
from ..confidence.fixtures import classification, unknown

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "pipeline_smoke_test.py"


def _load_script() -> ModuleType:
    """Load the script by path, the way a developer invokes it.

    The module is registered in ``sys.modules`` before execution because ``dataclasses``
    looks the defining module up there, and fails on a module that was never registered.
    """
    spec = importlib.util.spec_from_file_location("pipeline_smoke_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


def make_frame(sequence: int, timestamp: float) -> Frame:
    """A tiny but structurally valid frame. Nothing downstream looks at the pixels."""
    return Frame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=timestamp,
        sequence=sequence,
    )


def make_hand() -> TrackedHand:
    """A hand with the right number of landmarks. The classifier fakes ignore them."""
    return TrackedHand(
        landmarks=tuple(Landmark(x=0.5, y=0.5, z=0.0) for _ in range(LANDMARK_COUNT))
    )


def tracking_with(hand: TrackedHand | None, frame: Frame) -> TrackingResult:
    return TrackingResult(
        hands=() if hand is None else (hand,),
        timestamp=frame.timestamp,
        frame_sequence=frame.sequence,
    )


class FakeCamera:
    """Serves pre-built frames instead of opening a device."""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames
        self.reads = 0

    @property
    def measured_fps(self) -> float | None:
        return 30.0

    def read(self) -> Frame:
        self.reads += 1
        # The last frame repeats, so an unbounded run still has something to serve.
        return self._frames[min(self.reads, len(self._frames)) - 1]


class FakeTracker:
    """Returns a prepared tracking result per frame and records what it was given."""

    def __init__(self, results: list[TrackingResult]) -> None:
        self._results = results
        self.frames: list[Frame] = []

    def process(self, frame: Frame) -> TrackingResult:
        self.frames.append(frame)
        return self._results[min(len(self.frames), len(self._results)) - 1]


class FakeProcessor:
    """Records how the loop called it. Returns a sentinel the fake classifier ignores."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, float | None, int | None]] = []
        self.resets = 0

    def process(
        self,
        hand: object,
        *,
        timestamp: float | None = None,
        frame_sequence: int | None = None,
    ) -> object:
        self.calls.append((hand, timestamp, frame_sequence))
        return "features"

    def reset(self, handedness: object | None = None) -> None:
        self.resets += 1


class FakeClassifier:
    """Returns prepared classifications in order, repeating the last one."""

    def __init__(self, results: list[GestureClassification]) -> None:
        self._results = results
        self.inputs: list[object] = []

    def classify(self, features: object) -> GestureClassification:
        self.inputs.append(features)
        return self._results[min(len(self.inputs), len(self._results)) - 1]


class FakeStream(io.StringIO):
    """A stream that can pretend to be a terminal."""

    def __init__(self, interactive: bool = False) -> None:
        super().__init__()
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


def make_settings(module: ModuleType, **overrides: object) -> object:
    """Default settings for the loop tests, overridable per test."""
    argv = [
        "--frames",
        str(overrides.pop("frames", 10)),
        "--refresh-every",
        str(overrides.pop("refresh_every", 1)),
    ]
    for name, value in overrides.items():
        argv.extend([f"--{name.replace('_', '-')}", str(value)])
    return module.build_settings(module.parse_args(argv))


# ---------------------------------------------------------------------------
# Status formatting
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_a_frame_with_no_hand_says_so(self) -> None:
        line = script.describe(script.no_hand_outcome(7))
        assert line == f"frame     7 | hand  no  | {script.NO_HAND_MESSAGE}"

    def test_no_hand_line_never_claims_a_gesture_is_stable(self) -> None:
        line = script.describe(script.no_hand_outcome(0))
        assert "stable" not in line
        assert "gesture" not in line

    def test_a_hand_reports_gesture_confidence_candidate_and_stability(self) -> None:
        outcome = script.FrameOutcome(
            frame_sequence=12,
            hand_detected=True,
            gesture=Gesture.POINT.value,
            confidence=0.82,
            mean_confidence=0.8,
            candidate=Gesture.POINT.value,
            candidate_observations=2,
            required_observations=3,
        )
        line = script.describe(outcome)
        assert "hand  yes" in line
        assert "POINT" in line  # labels are upper case for legibility
        assert "0.82" in line
        assert "POINT 2/3" in line
        assert "stable no" in line

    def test_an_accepted_gesture_names_itself_as_stable(self) -> None:
        outcome = script.FrameOutcome(
            frame_sequence=1,
            hand_detected=True,
            gesture=Gesture.FIST.value,
            confidence=0.9,
            candidate=Gesture.FIST.value,
            candidate_observations=3,
            required_observations=3,
            accepted=Gesture.FIST.value,
        )
        assert script.describe(outcome).endswith("stable yes (FIST)")

    def test_an_interrupted_run_is_flagged(self) -> None:
        outcome = script.FrameOutcome(
            frame_sequence=1,
            hand_detected=True,
            gesture=Gesture.POINT.value,
            confidence=0.7,
            candidate=Gesture.POINT.value,
            candidate_observations=1,
            required_observations=3,
            interrupted=True,
        )
        assert script.describe(outcome).endswith("gap")

    def test_a_missing_candidate_is_shown_as_a_dash(self) -> None:
        outcome = script.FrameOutcome(
            frame_sequence=1,
            hand_detected=True,
            gesture=Gesture.UNKNOWN.value,
            confidence=0.1,
        )
        assert "candidate -" in script.describe(outcome)


class TestOutcomeFromReport:
    def test_a_real_filter_report_becomes_an_outcome(self) -> None:
        stability = ConfidenceFilter()
        frame = make_frame(4, 10.0)
        report = None
        for _ in range(3):
            report = stability.observe(
                classification(Gesture.POINT, 0.8), timestamp=frame.timestamp
            )
        assert report is not None

        outcome = script.outcome_from_report(frame.sequence, report, required_observations=3)

        assert outcome.hand_detected
        assert outcome.gesture == "point"
        assert outcome.confidence == pytest.approx(0.8)
        assert outcome.candidate == "point"
        assert outcome.candidate_observations == 3
        assert outcome.required_observations == 3
        assert outcome.accepted == "point"
        assert outcome.is_stable
        assert not outcome.interrupted

    def test_an_unknown_observation_stays_an_unknown_candidate(self) -> None:
        report = ConfidenceFilter().observe(unknown(0.2), timestamp=1.0)
        outcome = script.outcome_from_report(0, report, required_observations=3)
        assert outcome.gesture == "unknown"
        assert outcome.candidate is None
        assert outcome.accepted is None
        assert not outcome.is_stable


# ---------------------------------------------------------------------------
# Console throttling
# ---------------------------------------------------------------------------


class TestStatusPrinter:
    def test_a_non_terminal_stream_is_written_only_on_the_refresh_interval(self) -> None:
        stream = FakeStream()
        printer = script.StatusPrinter(stream, refresh_every=5)

        for sequence in range(10):
            printer.show(f"frame {sequence}")

        assert printer.lines_written == 2
        assert stream.getvalue() == "frame 4\nframe 9\n"

    def test_a_salient_line_ignores_the_interval(self) -> None:
        stream = FakeStream()
        printer = script.StatusPrinter(stream, refresh_every=10)

        printer.show("first", salient=True)
        printer.show("second")
        printer.show("the hand left", salient=True)

        assert stream.getvalue() == "first\nthe hand left\n"

    def test_a_terminal_rewrites_the_line_in_place(self) -> None:
        stream = FakeStream(interactive=True)
        printer = script.StatusPrinter(stream, refresh_every=1)

        printer.show("first")
        printer.show("second")

        assert stream.getvalue() == "\rfirst\rsecond"

    def test_a_terminal_pads_a_shorter_line_so_no_debris_is_left(self) -> None:
        stream = FakeStream(interactive=True)
        printer = script.StatusPrinter(stream, refresh_every=1)

        printer.show("a long line")
        printer.show("short")

        assert stream.getvalue() == "\ra long line\rshort      "

    def test_a_terminal_ends_the_line_once(self) -> None:
        stream = FakeStream(interactive=True)
        printer = script.StatusPrinter(stream, refresh_every=1)
        printer.show("one")

        printer.finish()
        printer.finish()

        assert stream.getvalue().count("\n") == 1

    def test_a_non_terminal_ends_with_nothing_extra(self) -> None:
        stream = FakeStream()
        printer = script.StatusPrinter(stream, refresh_every=1)
        printer.show("one")

        printer.finish()

        assert stream.getvalue() == "one\n"

    def test_a_zero_refresh_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="refresh_every must be >= 1"):
            script.StatusPrinter(FakeStream(), refresh_every=0)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestBuildSettings:
    def test_defaults_need_no_camera(self) -> None:
        settings = script.build_settings(script.parse_args([]))
        assert settings.frames == 300
        assert settings.refresh_every == 5
        assert settings.device == 0
        assert not settings.show_preview
        assert not settings.processor.smoothing

    def test_options_reach_the_stage_that_owns_them(self) -> None:
        settings = script.build_settings(
            script.parse_args(
                [
                    "--smoothing",
                    "--min-observations",
                    "5",
                    "--min-confidence",
                    "0.75",
                    "--max-interruption",
                    "0.4",
                    "--device",
                    "1",
                ]
            )
        )
        assert settings.processor.smoothing
        assert settings.stability.min_stable_observations == 5
        assert settings.stability.min_confidence == pytest.approx(0.75)
        assert settings.stability.max_interruption_seconds == pytest.approx(0.4)
        assert settings.device == 1
        assert settings.camera.device_index == 1

    def test_a_negative_frame_limit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--frames must be >= 0"):
            script.build_settings(script.parse_args(["--frames", "-1"]))

    def test_a_zero_refresh_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--refresh-every must be >= 1"):
            script.build_settings(script.parse_args(["--refresh-every", "0"]))

    def test_the_confidence_filter_refuses_an_impossible_threshold(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            script.build_settings(script.parse_args(["--min-confidence", "1.5"]))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class TestRun:
    def test_a_frame_timestamp_reaches_the_next_stage_unchanged(self) -> None:
        hand = make_hand()
        frames = [make_frame(0, 10.0), make_frame(1, 10.05)]
        camera = FakeCamera(frames)
        tracker = FakeTracker([tracking_with(hand, frame) for frame in frames])
        processor = FakeProcessor()
        classifier = FakeClassifier([classification(Gesture.POINT, 0.8)])
        stream = FakeStream()
        settings = make_settings(script, frames=2)

        script.run(
            settings,
            camera=camera,
            tracker=tracker,
            processor=processor,
            classifier=classifier,
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(stream, 1),
            quit_requested=lambda: False,
        )

        assert [call[1:] for call in processor.calls] == [(10.0, 0), (10.05, 1)]
        assert [frame.sequence for frame in tracker.frames] == [0, 1]
        assert len(classifier.inputs) == 2

    def test_only_the_primary_hand_is_processed(self) -> None:
        hand = make_hand()
        frame = make_frame(0, 1.0)
        result = TrackingResult(
            hands=(hand, make_hand(), make_hand()),
            timestamp=frame.timestamp,
            frame_sequence=frame.sequence,
        )
        processor = FakeProcessor()

        script.run(
            make_settings(script, frames=1),
            camera=FakeCamera([frame]),
            tracker=FakeTracker([result]),
            processor=processor,
            classifier=FakeClassifier([classification(Gesture.POINT, 0.8)]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=lambda: False,
        )

        assert processor.calls == [(hand, 1.0, 0)]

    def test_a_frame_with_no_hand_clears_the_state_and_skips_the_classifier(self) -> None:
        frame = make_frame(0, 1.0)
        processor = FakeProcessor()
        classifier = FakeClassifier([classification(Gesture.POINT, 0.8)])

        class CountingFilter(ConfidenceFilter):
            resets: ClassVar[int] = 0

            def reset(self) -> None:
                type(self).resets += 1
                super().reset()

        CountingFilter.resets = 0
        stability = CountingFilter()

        script.run(
            make_settings(script, frames=1),
            camera=FakeCamera([frame]),
            tracker=FakeTracker([tracking_with(None, frame)]),
            processor=processor,
            classifier=classifier,
            stability=stability,
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=lambda: False,
        )

        assert processor.resets == 1
        assert CountingFilter.resets == 1
        assert classifier.inputs == []

    def test_a_stale_gesture_is_not_displayed_after_the_hand_leaves(self) -> None:
        """The bug this guards against: an accepted gesture outliving the hand holding it."""
        hand = make_hand()
        held = [make_frame(sequence, 1.0 + sequence / 30) for sequence in range(4)]
        gone = make_frame(4, 2.0)
        camera = FakeCamera([*held, gone])
        tracker = FakeTracker([*(tracking_with(hand, f) for f in held), tracking_with(None, gone)])
        stream = FakeStream()

        summary = script.run(
            make_settings(script, frames=5, min_observations=3),
            camera=camera,
            tracker=tracker,
            processor=FakeProcessor(),
            classifier=FakeClassifier([classification(Gesture.POINT, 0.8)]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(stream, 1),
            quit_requested=lambda: False,
        )

        lines = stream.getvalue().splitlines()
        assert "stable yes (POINT)" in lines[-2]
        assert lines[-1] == f"frame     4 | hand  no  | {script.NO_HAND_MESSAGE}"
        # The gesture was accepted while it was held, but the final frame reports no hand.
        assert summary.accepted_counts == {"point": 2}

    def test_repeated_unknowns_are_never_accepted(self) -> None:
        hand = make_hand()
        frames = [make_frame(sequence, 1.0 + sequence / 30) for sequence in range(6)]

        summary = script.run(
            make_settings(script, frames=6, min_observations=3),
            camera=FakeCamera(frames),
            tracker=FakeTracker([tracking_with(hand, frame) for frame in frames]),
            processor=FakeProcessor(),
            classifier=FakeClassifier([unknown(0.2)]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=lambda: False,
        )

        assert summary.accepted_counts == {}
        assert summary.longest_run == 0
        assert summary.gesture_counts == {"unknown": 6}

    def test_the_summary_counts_what_was_seen(self) -> None:
        hand = make_hand()
        frames = [make_frame(sequence, 1.0 + sequence / 30) for sequence in range(4)]
        results = [tracking_with(hand, f) for f in frames[:3]] + [tracking_with(None, frames[3])]

        summary = script.run(
            make_settings(script, frames=4, min_observations=3),
            camera=FakeCamera(frames),
            tracker=FakeTracker(results),
            processor=FakeProcessor(),
            classifier=FakeClassifier([classification(Gesture.FIST, 0.8)]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=lambda: False,
        )

        assert summary.frames_processed == 4
        assert summary.frames_with_hands == 3
        assert summary.hand_detection_rate == pytest.approx(0.75)
        assert summary.gesture_counts == {"fist": 3}
        assert summary.accepted_counts == {"fist": 1}
        assert summary.longest_run == 3
        assert not summary.stopped_early

    def test_the_frame_limit_stops_the_loop(self) -> None:
        hand = make_hand()
        frames = [make_frame(sequence, 1.0 + sequence / 30) for sequence in range(20)]
        camera = FakeCamera(frames)

        summary = script.run(
            make_settings(script, frames=5),
            camera=camera,
            tracker=FakeTracker([tracking_with(hand, frames[0])]),
            processor=FakeProcessor(),
            classifier=FakeClassifier([classification(Gesture.POINT, 0.8)]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=lambda: False,
        )

        assert summary.frames_processed == 5
        assert camera.reads == 5

    def test_a_quit_request_stops_the_loop_and_is_reported(self) -> None:
        hand = make_hand()
        frames = [make_frame(sequence, 1.0 + sequence / 30) for sequence in range(10)]
        polls = 0

        def quit_requested() -> bool:
            nonlocal polls
            polls += 1
            return polls > 3

        summary = script.run(
            make_settings(script, frames=0),
            camera=FakeCamera(frames),
            tracker=FakeTracker([tracking_with(hand, frames[0])]),
            processor=FakeProcessor(),
            classifier=FakeClassifier([classification(Gesture.POINT, 0.8)]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=quit_requested,
        )

        assert summary.frames_processed == 3
        assert summary.stopped_early

    def test_quitting_before_the_first_frame_processes_nothing(self) -> None:
        summary = script.run(
            make_settings(script, frames=0),
            camera=FakeCamera([make_frame(0, 1.0)]),
            tracker=FakeTracker([]),
            processor=FakeProcessor(),
            classifier=FakeClassifier([]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=lambda: True,
        )

        assert summary.frames_processed == 0
        assert summary.hand_detection_rate == 0.0
        assert summary.stopped_early

    def test_the_preview_can_stop_the_loop(self) -> None:
        hand = make_hand()
        frames = [make_frame(sequence, 1.0 + sequence / 30) for sequence in range(5)]
        seen: list[int] = []

        def preview(frame: Frame, tracking: TrackingResult) -> bool:
            seen.append(frame.sequence)
            return len(seen) < 2

        summary = script.run(
            make_settings(script, frames=5),
            camera=FakeCamera(frames),
            tracker=FakeTracker([tracking_with(hand, f) for f in frames]),
            processor=FakeProcessor(),
            classifier=FakeClassifier([classification(Gesture.POINT, 0.8)]),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(FakeStream(), 1),
            quit_requested=lambda: False,
            preview=preview,
        )

        assert seen == [0, 1]
        assert summary.frames_processed == 2
        assert summary.stopped_early

    def test_the_real_classifier_and_filter_agree_through_the_loop(self) -> None:
        """The stages this script wires together, with only the hardware faked.

        The fakes above hand the loop classifications that were built to suit the test. This
        one lets the genuine classifier decide, so the loop is checked against the output it
        will actually receive in the field rather than against an idealised one.
        """
        hand = make_hand()
        frames = [make_frame(sequence, 1.0 + sequence / 30) for sequence in range(4)]
        stream = FakeStream()

        class PosedProcessor(FakeProcessor):
            def process(
                self,
                hand: object,
                *,
                timestamp: float | None = None,
                frame_sequence: int | None = None,
            ) -> HandFeatures:
                super().process(hand, timestamp=timestamp, frame_sequence=frame_sequence)
                return point()

        summary = script.run(
            make_settings(script, frames=4, min_observations=3),
            camera=FakeCamera(frames),
            tracker=FakeTracker([tracking_with(hand, frame) for frame in frames]),
            processor=PosedProcessor(),
            classifier=GestureClassifier(),
            stability=ConfidenceFilter(),
            printer=script.StatusPrinter(stream, 1),
            quit_requested=lambda: False,
        )

        assert summary.gesture_counts == {"point": 4}
        assert summary.accepted_counts == {"point": 2}
        assert summary.longest_run == 4
        assert stream.getvalue().splitlines()[-1].endswith("stable yes (POINT)")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestSummaryReporting:
    def test_a_run_that_never_saw_a_hand_says_why(self, capsys: pytest.CaptureFixture) -> None:
        settings = script.build_settings(script.parse_args([]))
        script._print_summary(script.RunSummary(frames_processed=10), settings)

        assert "No hands were ever detected" in capsys.readouterr().out

    def test_a_run_with_no_accepted_gesture_says_why(self, capsys: pytest.CaptureFixture) -> None:
        settings = script.build_settings(script.parse_args([]))
        summary = script.RunSummary(
            frames_processed=10,
            frames_with_hands=8,
            gesture_counts={"point": 8},
        )
        script._print_summary(summary, settings)

        output = capsys.readouterr().out
        assert "No gesture was ever accepted as stable" in output
        assert "80%" in output

    def test_a_clean_run_reports_no_complaint(self, capsys: pytest.CaptureFixture) -> None:
        settings = script.build_settings(script.parse_args([]))
        summary = script.RunSummary(
            frames_processed=10,
            frames_with_hands=10,
            gesture_counts={"point": 10},
            accepted_counts={"point": 4},
            longest_run=10,
        )
        script._print_summary(summary, settings)

        output = capsys.readouterr().out
        assert "No hands were ever detected" not in output
        assert "No gesture was ever accepted" not in output


# ---------------------------------------------------------------------------
# Failure paths that must not need hardware
# ---------------------------------------------------------------------------


class TestMainFailurePaths:
    def test_a_bad_option_exits_before_anything_is_opened(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert script.main(["--frames", "-1"]) == script.EXIT_UNEXPECTED
        assert "Invalid configuration" in capsys.readouterr().out

    def test_a_missing_model_asset_is_reported(self, capsys: pytest.CaptureFixture) -> None:
        exit_code = script.main(["--model", str(Path("no") / "such" / "model.task")])

        assert exit_code == script.EXIT_NO_MODEL
        output = capsys.readouterr().out
        assert "Model asset missing" in output
        assert "download" in output.lower()

    def test_exit_codes_are_distinct(self) -> None:
        codes = [
            script.EXIT_OK,
            script.EXIT_UNEXPECTED,
            script.EXIT_NO_MODEL,
            script.EXIT_PROCESSING,
            script.EXIT_CAMERA,
            script.EXIT_TRACKING,
            script.EXIT_RECOGNITION,
        ]
        assert len(set(codes)) == len(codes)
