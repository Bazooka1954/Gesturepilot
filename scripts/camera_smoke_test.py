"""MANUAL DEVELOPMENT TOOL -- not part of the GesturePilot application.

Opens the default webcam, reads a handful of frames, and prints what the camera
actually reported. Intended for the manual smoke test described in ``docs/camera.md``.

This performs no gesture processing and writes nothing to disk. It exists only to
check that a physical device can be opened on this machine.

Usage::

    uv run python scripts/camera_smoke_test.py
    uv run python scripts/camera_smoke_test.py --device 1 --width 1280 --frames 30

The production entry point is ``gesturepilot.main`` and is unaffected by this file.
"""

from __future__ import annotations

import argparse
import sys

from gesturepilot.camera import CameraConfig, CameraError, WebcamCamera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GesturePilot manual camera smoke test.")
    parser.add_argument("--device", type=int, default=0, help="Camera device index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested FPS.")
    parser.add_argument("--frames", type=int, default=10, help="Frames to read.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CameraConfig(
        device_index=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    camera = WebcamCamera(config)
    print(f"Opening camera {args.device}...")
    try:
        camera.open()
    except CameraError as exc:
        print(f"FAILED: {exc}")
        return 1

    try:
        props = camera.properties
        print(
            f"  requested : {props.requested_width}x{props.requested_height} @ {props.requested_fps}"
        )
        print(f"  device    : {props.actual_resolution} @ {props.actual_fps}")
        print(f"  matched   : {props.matched_request}")

        print(f"Reading {args.frames} frames...")
        for index in range(args.frames):
            frame = camera.read()
            print(
                f"  frame {index:>3}: {frame.width}x{frame.height} "
                f"x{frame.channels}ch  seq={frame.sequence}  measured={camera.measured_fps}"
            )
    except CameraError as exc:
        print(f"FAILED while reading: {exc}")
        return 1
    finally:
        camera.close()
        print("Camera released.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
