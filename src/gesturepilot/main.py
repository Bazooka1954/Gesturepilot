"""Minimal executable entry point for GesturePilot.

This exists solely to prove the package is importable, installable, and
runnable. It intentionally contains no camera, tracking, or gesture logic.
"""

from gesturepilot import __version__


def main() -> int:
    """Print a startup banner confirming the package runs.

    Returns:
        Process exit code. Always ``0`` at this stage.
    """
    print(f"GesturePilot {__version__}")
    print("Environment OK. No pipeline stages implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
