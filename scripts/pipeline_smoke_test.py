"""MANUAL DEVELOPMENT TOOL -- not part of the GesturePilot application.

Runs the whole recognition pipeline against a live webcam and prints what each stage
decided, one status line at a time::

    Camera -> Tracking -> Processing -> Classification -> Confidence Filter

This is the first script that exercises all five layers together, so it is where an
integration mistake between them shows up: a frame whose timestamp is not what the next
stage expects, a hand that leaves the frame and leaves stale stability behind it, a
classifier and a filter disagreeing about what "no gesture" means.

**It decides nothing and acts on nothing.** No OS action, no key press, no click, no
system setting, no gesture event. It prints words and numbers. Nothing is written to disk:
no frames, no video, no landmark recordings, no logs, no network. The preview window is off
by default and only draws what the camera already delivered.

**This is a manual real-hardware smoke test.** It needs a webcam and the local
``models/hand_landmarker.task`` asset (see ``models/README.md``). Nothing is ever
downloaded. It is deselected from the normal test suite, which never opens a camera.

Usage::

    uv run python scripts/pipeline_smoke_test.py
    uv run python scripts/pipeline_smoke_test.py --device 1 --max-hands 1
    uv run python scripts/pipeline_smoke_test.py --smoothing --min-observations 5
    uv run python scripts/pipeline_smoke_test.py --frames 0 --show    # run until 'q'

Press ``q`` to quit at any time (``Ctrl+C`` also works). With ``--show``, ``q`` closes the
preview window too.

What it prints, once per line::

    frame  0042 | hand yes | gesture POINT     0.82 | candidate point 2/3 | stable no

* ``hand`` -- whether the tracker found a hand in that frame.
* ``gesture`` -- the classifier's label, including ``UNKNOWN``, and its own confidence.
  Remember a *high* confidence on ``UNKNOWN`` means the evidence was weak, which is a good
  outcome.
* ``candidate`` -- the gesture the confidence filter is accumulating and its run length
  against the requirement.
* ``stable`` -- whether a gesture is currently accepted. This means *consistent*, not
  *confirmed*: no safety state machine exists yet, so nothing is ever acted on.

When no hand is in frame the line says so, and the filter's state is cleared rather than
left showing the last gesture it saw. There is no fake ``UNKNOWN`` for a missing frame --
the classifier was never asked, so inventing an answer for it would be a lie.

Exit codes: ``0`` clean run, ``2`` model asset missing, ``3`` processing error, ``4`` camera
error, ``5`` tracking error, ``6`` classification or confidence-filter error, ``1``
anything else.

The production entry point is ``gesturepilot.main`` and is unaffected by this file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesturepilot.camera import CameraConfig, CameraError, Frame, WebcamCamera  # noqa: E402
from gesturepilot.classifier import (  # noqa: E402
    ClassifierConfig,
    Gesture,
    GestureClassification,
    GestureClassifier,
)
from gesturepilot.classifier.errors import ClassificationError  # noqa: E402
from gesturepilot.confidence import (  # noqa: E402
    ConfidenceFilter,
    ConfidenceFilterConfig,
    StabilityReport,
)
from gesturepilot.confidence.errors import ConfidenceError  # noqa: E402
from gesturepilot.processing import (  # noqa: E402
    LandmarkProcessor,
    ProcessorConfig,
)
from gesturepilot.processing.errors import ProcessingError  # noqa: E402
from gesturepilot.processing.types import HandFeatures  # noqa: E402
from gesturepilot.tracking import (  # noqa: E402
    MediaPipeHandTracker,
    ModelAssetError,
    RunningMode,
    TrackerConfig,
    TrackingError,
    TrackingResult,
    resolve_model_path,
)

#: Shown instead of a gesture when the tracker found no hand at all.
NO_HAND_MESSAGE = "-- no hand in frame: stability reset --"

# Landmark connections, in the MediaPipe hand-model order. Only used for --show.
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm base
)

#: Returned by :func:`main` for each failure mode, so a script runner can tell them apart.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_NO_MODEL = 2
EXIT_PROCESSING = 3
EXIT_CAMERA = 4
EXIT_TRACKING = 5
EXIT_RECOGNITION = 6


# ---------------------------------------------------------------------------
# Collaborator protocols
#
# The pipeline is written against these rather than the concrete classes, so the loop can be
# driven by fakes in the unit tests and no test needs a camera or the 7 MB model. The real
# objects satisfy them structurally.
# ---------------------------------------------------------------------------


class CameraLike(Protocol):
    """The part of :class:`~gesturepilot.camera.camera.WebcamCamera` this script uses."""

    @property
    def measured_fps(self) -> float | None: ...

    def read(self) -> Frame: ...


class TrackerLike(Protocol):
    """The part of :class:`~gesturepilot.tracking.tracker.MediaPipeHandTracker` used here."""

    def process(self, frame: Frame) -> TrackingResult: ...


class ProcessorLike(Protocol):
    """The part of :class:`~gesturepilot.processing.processor.LandmarkProcessor` used here."""

    def process(
        self,
        hand: object,
        *,
        timestamp: float | None = None,
        frame_sequence: int | None = None,
    ) -> HandFeatures: ...

    def reset(self, handedness: object | None = None) -> None: ...


class ClassifierLike(Protocol):
    """The part of :class:`~gesturepilot.classifier.classifier.GestureClassifier` used here."""

    def classify(self, features: HandFeatures) -> GestureClassification: ...


class StabilityLike(Protocol):
    """The part of :class:`~gesturepilot.confidence.filter.ConfidenceFilter` used here."""

    def observe(
        self,
        classification: GestureClassification,
        *,
        timestamp: float | None = None,
    ) -> StabilityReport: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineSettings:
    """Every resolved configuration for one run, so the loop takes no CLI arguments.

    Frozen because a half-applied configuration is the sort of thing that only shows up
    twenty frames later, when a gesture mysteriously fails to stabilise.
    """

    device: int
    frames: int
    refresh_every: int
    show_preview: bool
    debug_camera: bool
    camera: CameraConfig
    tracker: TrackerConfig
    processor: ProcessorConfig
    classifier: ClassifierConfig
    stability: ConfidenceFilterConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line. Camera and model options mirror the tracking smoke test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="requested width")
    parser.add_argument("--height", type=int, default=480, help="requested height")
    parser.add_argument("--fps", type=float, default=30.0, help="requested FPS")
    parser.add_argument("--max-hands", type=int, default=1, help="max hands to detect")
    parser.add_argument(
        "--running-mode",
        choices=("video", "image"),
        default="video",
        help="tracker running mode (default: video)",
    )
    parser.add_argument("--model", type=str, default=None, help="explicit model path")
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="frames to process, 0 to run until 'q' (default: 300)",
    )
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=5,
        help="redraw the status line every N frames (default: 5)",
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="run One Euro smoothing in the processing stage (off by default)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=ConfidenceFilterConfig().min_confidence,
        help="confidence an observation needs to count as evidence",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=ConfidenceFilterConfig().min_stable_observations,
        help="consistent observations needed to accept a gesture",
    )
    parser.add_argument(
        "--max-interruption",
        type=float,
        default=None,
        help="seconds of missing observations that break a run (default: off)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="draw landmarks in a preview window (press 'q' to quit)",
    )
    parser.add_argument(
        "--debug-camera",
        action="store_true",
        help="print frame shape/dtype/channels and measured FPS while running (no save)",
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> PipelineSettings:
    """Turn parsed arguments into validated stage configurations.

    Raises:
        ValueError: If a value is outside the range its own stage accepts. Reported here,
            before the camera opens, so a typo costs nothing.
    """
    if args.frames < 0:
        raise ValueError(f"--frames must be >= 0, got {args.frames}")
    if args.refresh_every < 1:
        raise ValueError(f"--refresh-every must be >= 1, got {args.refresh_every}")

    # Each of these validates itself on construction, which is the point: an out-of-range
    # threshold is refused by the stage that owns it rather than silently clamped here.
    return PipelineSettings(
        device=args.device,
        frames=args.frames,
        refresh_every=args.refresh_every,
        show_preview=args.show,
        debug_camera=args.debug_camera,
        camera=CameraConfig(
            device_index=args.device,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ),
        tracker=TrackerConfig(
            max_hands=args.max_hands,
            model_path=args.model,
            running_mode=RunningMode(args.running_mode),
        ),
        processor=ProcessorConfig(smoothing=args.smoothing),
        classifier=ClassifierConfig(),
        stability=ConfidenceFilterConfig(
            min_confidence=args.min_confidence,
            min_stable_observations=args.min_observations,
            max_interruption_seconds=args.max_interruption,
        ),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameOutcome:
    """What one frame produced, reduced to the facts the status line shows.

    Frozen and label-only, so the formatting can be tested without a camera, a tracker, or a
    model, and so nothing downstream of the classifier can mutate what was recorded.
    """

    frame_sequence: int
    hand_detected: bool
    gesture: str | None = None
    confidence: float | None = None
    margin: float | None = None
    mean_confidence: float | None = None
    candidate: str | None = None
    candidate_observations: int = 0
    required_observations: int = 1
    accepted: str | None = None
    interrupted: bool = False

    @property
    def is_stable(self) -> bool:
        """Whether a gesture was accepted for this frame."""
        return self.accepted is not None


def no_hand_outcome(frame_sequence: int) -> FrameOutcome:
    """The outcome for a frame the tracker found no hand in."""
    return FrameOutcome(frame_sequence=frame_sequence, hand_detected=False)


def outcome_from_report(
    frame_sequence: int,
    report: StabilityReport,
    required_observations: int,
) -> FrameOutcome:
    """Reduce one classifier-and-filter result to a :class:`FrameOutcome`.

    The classifier's confidence and margin are carried through untouched. A confident
    ``UNKNOWN`` stays a confident ``UNKNOWN``: the filter does not reinterpret it, and
    neither does this script.
    """
    classification = report.classification
    return FrameOutcome(
        frame_sequence=frame_sequence,
        hand_detected=True,
        gesture=classification.gesture.value,
        confidence=classification.confidence,
        margin=classification.margin,
        mean_confidence=report.mean_confidence,
        candidate=report.candidate.value if report.candidate is not None else None,
        candidate_observations=report.candidate_observations,
        required_observations=required_observations,
        accepted=report.accepted.value if report.accepted is not None else None,
        interrupted=report.interrupted,
    )


def describe(outcome: FrameOutcome) -> str:
    """Format one status line. Pure, fixed width, and short enough to read at a glance.

    Gesture labels are shown upper case so they read as labels rather than as data; the
    underlying values are the domain's own lowercase strings and are left untouched.
    """
    if not outcome.hand_detected:
        return f"frame {outcome.frame_sequence:>5} | hand  no  | {NO_HAND_MESSAGE}"

    label = (outcome.gesture or Gesture.UNKNOWN.value).upper()
    gesture = f"{label:<8} {outcome.confidence or 0.0:5.2f}"
    if outcome.candidate is None:
        candidate = "-"
    else:
        candidate = (
            f"{outcome.candidate.upper()} "
            f"{outcome.candidate_observations}/{outcome.required_observations}"
        )
        if outcome.mean_confidence is not None:
            candidate += f" @ {outcome.mean_confidence:5.2f}"
    stable = f"yes ({outcome.accepted.upper()})" if outcome.is_stable else "no"
    parts = [
        f"frame {outcome.frame_sequence:>5}",
        "hand  yes ",
        f"gesture {gesture}",
        f"candidate {candidate}",
        f"stable {stable}",
    ]
    if outcome.interrupted:
        parts.append("gap")
    return " | ".join(parts)


class StatusPrinter:
    """Writes status lines without flooding the terminal.

    One behaviour per kind of stream, and both are bounded by the same interval:

    * **Terminal** -- the line is rewritten in place every ``refresh_every`` calls, padded to
      the widest line seen so a shorter line cannot leave debris behind.
    * **Not a terminal** (redirected to a file, or captured by a test) -- one line per
      ``refresh_every`` calls, appended. Waiting for the text to change is not an option
      here: every line carries a new frame number, so the text always changes.

    Either way a line flagged ``salient`` is written immediately rather than waiting for the
    next interval. The loop flags a change in whether a hand is present, so a hand leaving the
    frame is never hidden behind up to ``refresh_every - 1`` frames of silence.

    No thread, no timer, and no clock: the throttle counts calls, so it behaves identically
    on every machine.
    """

    def __init__(self, stream: object, refresh_every: int) -> None:
        if refresh_every < 1:
            raise ValueError(f"refresh_every must be >= 1, got {refresh_every}")
        self._stream = stream
        self._refresh_every = refresh_every
        self._calls = 0
        self._lines = 0
        self._width = 0
        self._finished = False
        self._interactive = bool(getattr(stream, "isatty", lambda: False)())

    @property
    def lines_written(self) -> int:
        """How many status lines actually reached the stream."""
        return self._lines

    def show(self, text: str, *, salient: bool = False) -> None:
        """Display one status line, subject to the throttle.

        Args:
            text: The line to show.
            salient: Whether to write it now, ignoring the throttle. Used for changes worth
                seeing immediately, such as a hand leaving the frame.
        """
        self._calls += 1
        if not salient and self._calls % self._refresh_every:
            return
        if self._interactive:
            padding = " " * max(0, self._width - len(text))
            self._width = max(self._width, len(text))
            self._write(f"\r{text}{padding}")
        else:
            self._write(f"{text}\n")
        self._lines += 1

    def finish(self) -> None:
        """End the rewritten line so the shell prompt does not overwrite the status.

        Idempotent, because the error paths and the ``finally`` block both call it.
        """
        if self._finished:
            return
        self._finished = True
        if self._interactive and self._lines:
            self._write("\n")

    def _write(self, text: str) -> None:
        self._stream.write(text)  # type: ignore[attr-defined]
        flush = getattr(self._stream, "flush", None)
        if callable(flush):
            flush()


def _console_quit_requested() -> bool:
    """Whether ``q`` was pressed in the console. ``False`` where that cannot be read.

    ``msvcrt`` reads a keypress on Windows with no window and no dependency. Elsewhere, and
    when stdin is not a console, this reports ``False`` and the run ends on ``--frames`` or
    ``Ctrl+C`` instead.
    """
    try:
        import msvcrt
    except ImportError:
        return False
    if not msvcrt.kbhit():
        return False
    return msvcrt.getwch().lower() == "q"


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """What the run saw. Built once, at the end, from counters the loop kept."""

    frames_processed: int = 0
    frames_with_hands: int = 0
    stopped_early: bool = False
    gesture_counts: dict[str, int] = field(default_factory=dict)
    accepted_counts: dict[str, int] = field(default_factory=dict)
    longest_run: int = 0

    @property
    def hand_detection_rate(self) -> float:
        """Fraction of frames with a hand in them, or ``0.0`` when nothing was processed."""
        if not self.frames_processed:
            return 0.0
        return self.frames_with_hands / self.frames_processed


def run(
    settings: PipelineSettings,
    *,
    camera: CameraLike,
    tracker: TrackerLike,
    processor: ProcessorLike,
    classifier: ClassifierLike,
    stability: StabilityLike,
    printer: StatusPrinter,
    quit_requested: Callable[[], bool] | None = None,
    preview: Callable[[Frame, TrackingResult], bool] | None = None,
) -> RunSummary:
    """Process frames until the limit, a quit request, or an error.

    One hand is followed: the tracker's primary hand. The confidence filter is per subject,
    and adding a second hand's evidence to the first would report a stability neither of them
    showed. A frame with no hand clears both the smoothing state and the filter's run, so a
    gesture that stopped being held is never still displayed as accepted.

    Args:
        settings: Resolved configuration. ``frames=0`` means run until asked to stop.
        camera: Source of frames.
        tracker: Turns a frame into hands.
        processor: Turns a hand into features.
        classifier: Turns features into one classification.
        stability: Turns classifications into stability reports.
        printer: Receives one status line per update.
        quit_requested: Polled between frames. Defaults to the console ``q``.
        preview: Called after each frame with the frame and its tracking result. Returning
            ``False`` stops the run, which is how ``q`` works in the preview window.

    Returns:
        A :class:`RunSummary` of what was seen.
    """
    should_quit = quit_requested if quit_requested is not None else _console_quit_requested
    required = settings.stability.min_stable_observations

    frames = 0
    frames_with_hands = 0
    stopped_early = False
    gesture_counts: dict[str, int] = {}
    accepted_counts: dict[str, int] = {}
    longest_run = 0
    previous_had_hand: bool | None = None

    while settings.frames == 0 or frames < settings.frames:
        if should_quit():
            stopped_early = True
            break

        frame = camera.read()
        if settings.debug_camera and frames % settings.refresh_every == 0:
            img = frame.image
            try:
                channels = img.shape[2] if img.ndim == 3 else 1
            except Exception:
                channels = 0
            fpsm = camera.measured_fps
            print(
                f"[DEBUG] seq={frame.sequence} ts={frame.timestamp:.4f} shape={img.shape} dtype={img.dtype} ch={channels} cam_fps={fpsm}"
            )
        tracking = tracker.process(frame)
        frames += 1

        hand = tracking.primary_hand if tracking.has_hands else None
        if hand is None:
            # Nothing was classified, so there is nothing to report and nothing to keep:
            # dropping the state here is what stops a stale gesture reading as current.
            processor.reset()
            stability.reset()
            outcome = no_hand_outcome(frame.sequence)
        else:
            frames_with_hands += 1
            features = processor.process(
                hand,
                timestamp=frame.timestamp,
                frame_sequence=frame.sequence,
            )
            report = stability.observe(
                classifier.classify(features),
                timestamp=frame.timestamp,
            )
            outcome = outcome_from_report(frame.sequence, report, required)
            gesture_counts[outcome.gesture or Gesture.UNKNOWN.value] = (
                gesture_counts.get(outcome.gesture or Gesture.UNKNOWN.value, 0) + 1
            )
            if outcome.accepted is not None:
                accepted_counts[outcome.accepted] = accepted_counts.get(outcome.accepted, 0) + 1
            longest_run = max(longest_run, outcome.candidate_observations)

        # The first frame is salient so the console is never blank, and any later change in
        # whether a hand is present is salient so it cannot hide behind the throttle.
        had_hand = hand is not None
        salient = previous_had_hand is None or previous_had_hand != had_hand
        printer.show(describe(outcome), salient=salient)
        previous_had_hand = had_hand

        if preview is not None and not preview(frame, tracking):
            stopped_early = True
            break

    return RunSummary(
        frames_processed=frames,
        frames_with_hands=frames_with_hands,
        stopped_early=stopped_early,
        gesture_counts=gesture_counts,
        accepted_counts=accepted_counts,
        longest_run=longest_run,
    )


def _build_preview() -> Callable[[Frame, TrackingResult], bool]:
    """A preview callback that draws landmarks and reports whether ``q`` was pressed.

    OpenCV is imported here, not at module scope, so importing this file for a unit test
    costs nothing and the drawing code stays out of the way when ``--show`` is unused.
    """
    import cv2

    def preview(frame: Frame, tracking: TrackingResult) -> bool:
        canvas = frame.image.copy()
        height, width = canvas.shape[:2]
        for hand in tracking.hands:
            points = [(int(lm.x * width), int(lm.y * height)) for lm in hand.landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(canvas, points[start], points[end], (0, 200, 255), 2)
            for x, y in points:
                cv2.circle(canvas, (x, y), 4, (0, 200, 255), -1)
        cv2.imshow("GesturePilot pipeline smoke test", canvas)
        return cv2.waitKey(1) & 0xFF != ord("q")

    return preview


def _close_preview() -> None:
    """Tear the preview window down, if one was ever opened."""
    try:
        import cv2
    except ImportError:
        return
    cv2.destroyAllWindows()


def _print_summary(summary: RunSummary, settings: PipelineSettings) -> None:
    """Print the run summary, and say so plainly when nothing was ever seen."""
    print("\n--- summary ---")
    print(f"frames processed        : {summary.frames_processed}")
    print(f"frames with >=1 hand    : {summary.frames_with_hands}")
    print(f"hand detection rate     : {summary.hand_detection_rate:.0%}")
    print(f"longest candidate run   : {summary.longest_run}")
    if summary.gesture_counts:
        print("classifier labels (frames):")
        for label, count in sorted(summary.gesture_counts.items(), key=lambda item: -item[1]):
            print(f"  {label.upper():<14} {count}")
    if summary.accepted_counts:
        print("accepted as stable (frames):")
        for label, count in sorted(summary.accepted_counts.items(), key=lambda item: -item[1]):
            print(f"  {label.upper():<14} {count}")
    if summary.stopped_early:
        print("stopped early on request")
    if not settings.show_preview:
        print("\nNote: 'stable' means seen consistently, not confirmed. No safety state")
        print("machine exists yet, so nothing here can act on a gesture.")

    if summary.frames_with_hands == 0:
        print("\nNo hands were ever detected. Check lighting, keep a full hand in frame,")
        print("and confirm the camera is pointed at you.")
    elif not summary.accepted_counts:
        print("\nNo gesture was ever accepted as stable. Hold one pose still for a moment,")
        print("or lower --min-confidence / --min-observations to see the filter accept one.")


def main(argv: list[str] | None = None) -> int:
    """Run the live pipeline and report what it decided. Never acts on anything."""
    args = parse_args(argv)

    try:
        settings = build_settings(args)
    except (ValueError, TypeError) as exc:
        print(f"Invalid configuration: {exc}")
        return EXIT_UNEXPECTED

    # Resolve the model before touching the camera, so a missing asset is reported plainly
    # instead of after a device has been opened and lit up.
    try:
        model_path = resolve_model_path(args.model)
    except ModelAssetError as exc:
        print(f"Model asset missing.\n\n{exc}")
        return EXIT_NO_MODEL

    print(f"Model: {model_path}")
    print(
        f"Pipeline: camera {settings.device} -> tracking ({settings.tracker.running_mode.value})"
        f" -> processing -> classifier -> confidence filter"
    )
    print(
        f"Stability: {settings.stability.min_stable_observations} consistent observations at"
        f" confidence >= {settings.stability.min_confidence:g}"
    )
    if settings.stability.max_interruption_seconds is not None:
        print(f"Interruption gap: {settings.stability.max_interruption_seconds:g}s")
    print("Press 'q' to quit, Ctrl+C to abort.\n")

    camera = WebcamCamera(settings.camera)
    tracker = MediaPipeHandTracker(settings.tracker)
    processor = LandmarkProcessor(settings.processor)
    classifier = GestureClassifier(settings.classifier)
    stability = ConfidenceFilter(settings.stability)
    printer = StatusPrinter(sys.stdout, settings.refresh_every)

    summary = RunSummary()
    try:
        # WebcamCamera directly: open_camera() returns an already-open device, so it must not
        # be re-entered as a context manager.
        with camera, tracker:
            summary = run(
                settings,
                camera=camera,
                tracker=tracker,
                processor=processor,
                classifier=classifier,
                stability=stability,
                printer=printer,
                preview=_build_preview() if settings.show_preview else None,
            )
    except CameraError as exc:
        printer.finish()
        print(f"\nCamera error: {exc}")
        return EXIT_CAMERA
    except TrackingError as exc:
        printer.finish()
        print(f"\nTracking error: {exc}")
        return EXIT_TRACKING
    except ProcessingError as exc:
        printer.finish()
        print(f"\nProcessing error: {exc}")
        return EXIT_PROCESSING
    except (ClassificationError, ConfidenceError) as exc:
        printer.finish()
        print(f"\nRecognition error: {exc}")
        return EXIT_RECOGNITION
    except KeyboardInterrupt:
        printer.finish()
        print("\nInterrupted.")
    finally:
        printer.finish()
        # Smoothing state lives in the processor; dropping it costs nothing and leaves no
        # half-open resource behind even if the run ended badly.
        processor.reset()
        _close_preview()

    _print_summary(summary, settings)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
