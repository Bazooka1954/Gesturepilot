"""Minimal executable entry point for GesturePilot.

This exists solely to prove the package is importable, installable, and
runnable. It intentionally contains no camera, tracking, processing, or
gesture-classification logic — no pipeline stage is wired together here yet.
"""

from gesturepilot import __version__


def main() -> int:
    """Print a startup banner confirming the package runs.

    Intentionally does not open the camera or process frames. Smoke tests live at
    ``scripts/camera_smoke_test.py`` and ``scripts/tracking_smoke_test.py`` instead.

    Returns:
        Process exit code. Always ``0`` at this stage.
    """
    print(f"GesturePilot {__version__}")
    print("Environment OK. Camera, hand-tracking, landmark-processing, and static")
    print("gesture-classification layers available; confidence filtering, temporal")
    print("gestures, and OS control not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
