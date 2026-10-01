"""Developer smoke test for the hand-tracking layer (Phase 3).

Prints what the real MediaPipe hand landmarker reports for live webcam frames. This is
a development tool, not part of the application.

Unlike the unit tests, this loads the genuine model and opens a real camera, so it
requires both ``models/hand_landmarker.task`` (see ``models/README.md``) and a webcam.

Usage::

    uv run python scripts/tracking_smoke_test.py
    uv run python scripts/tracking_smoke_test.py --device 1 --max-hands 2 --frames 120
    uv run python scripts/tracking_smoke_test.py --running-mode image

Press ``q`` to quit early. ``--show`` opens a preview window with landmarks drawn on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesturepilot.camera import CameraConfig, CameraError, WebcamCamera  # noqa: E402
from gesturepilot.tracking import (  # noqa: E402
    LANDMARK_COUNT,
    MediaPipeHandTracker,
    ModelAssetError,
    RunningMode,
    TrackerConfig,
    TrackingError,
    resolve_model_path,
)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="requested width")
    parser.add_argument("--height", type=int, default=480, help="requested height")
    parser.add_argument("--fps", type=float, default=30.0, help="requested FPS")
    parser.add_argument("--max-hands", type=int, default=2, help="max hands to detect")
    parser.add_argument("--frames", type=int, default=90, help="frames to process (default: 90)")
    parser.add_argument(
        "--running-mode",
        choices=("video", "image"),
        default="video",
        help="tracker running mode (default: video)",
    )
    parser.add_argument("--model", type=str, default=None, help="explicit model path")
    parser.add_argument("--show", action="store_true", help="draw landmarks in a window")
    return parser.parse_args(argv)


def draw(result_hands, frame, width: int, height: int) -> None:
    """Draw landmarks over a copy of the frame, scaled from normalised coordinates."""
    import cv2

    for hand in result_hands:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand.landmarks]
        colour = {
            "Left": (255, 128, 0),
            "Right": (0, 200, 255),
        }.get(hand.handedness.value, (200, 200, 200))

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], colour, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, colour, -1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    tracker_config = TrackerConfig(
        max_hands=args.max_hands,
        model_path=args.model,
        running_mode=RunningMode(args.running_mode),
    )

    # Check the asset is present before touching the camera, so a missing model gives a
    # clear message instead of opening a device that cannot be used.
    try:
        resolve_model_path(args.model)
    except ModelAssetError as exc:
        print(f"Model asset missing.\n\n{exc}")
        return 2

    # Entering the context manager initialises the tracker; exiting closes it.
    tracker = MediaPipeHandTracker(tracker_config)

    camera_config = CameraConfig(
        device_index=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    print(
        f"Tracking {args.frames} frames from camera {args.device} "
        f"({args.running_mode} mode, max {args.max_hands} hands)"
    )
    print("Press 'q' to quit early.\n")

    frames_with_hands = 0
    total_hands = 0
    handedness_counts: dict[str, int] = {}
    first_hand_latency: list[float] = []

    try:
        # WebcamCamera directly: open_camera() returns an already-open device, so it
        # must not be re-entered as a context manager.
        with WebcamCamera(camera_config) as camera, tracker:
            for _ in range(args.frames):
                frame = camera.read()
                started = time.perf_counter()
                result = tracker.process(frame)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                if result.has_hands:
                    frames_with_hands += 1
                    total_hands += len(result.hands)
                    for hand in result.hands:
                        key = hand.handedness.value
                        if hand.handedness_score is not None:
                            key += f" ({hand.handedness_score:.2f})"
                        handedness_counts[key] = handedness_counts.get(key, 0) + 1

                    primary = result.primary_hand
                    if primary is not None and primary.handedness_score is not None:
                        first_hand_latency.append(elapsed_ms)

                    if args.frames <= 90:
                        print(
                            f"  frame {result.frame_sequence:>4} | "
                            f"hands={result.hand_count} | "
                            f"landmarks={LANDMARK_COUNT} | "
                            f"infer={elapsed_ms:6.1f} ms | "
                            f"cam_fps={camera.measured_fps}"
                        )

                if args.show:
                    import cv2

                    preview = frame.image.copy()
                    draw(result.hands, preview, frame.width, frame.height)
                    cv2.imshow("GesturePilot tracking smoke test", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nStopped early.")
                        break
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 4
    except TrackingError as exc:
        print(f"Tracking error: {exc}")
        return 5
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.show:
            import cv2

            cv2.destroyAllWindows()

    processed = args.frames
    print("\n--- summary ---")
    print(f"frames processed        : {processed}")
    print(f"frames with >=1 hand    : {frames_with_hands}")
    print(f"total hand detections   : {total_hands}")
    if handedness_counts:
        print("handedness (count, score):")
        for key, count in sorted(handedness_counts.items()):
            print(f"  {key:<16} {count}")
    if first_hand_latency:
        mean = sum(first_hand_latency) / len(first_hand_latency)
        print(f"mean inference latency  : {mean:.1f} ms")

    if total_hands == 0:
        print(
            "\nNo hands detected. Check lighting, keep a full hand in frame, and "
            "confirm the camera is pointed at you."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
