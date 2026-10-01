"""GesturePilot: webcam-based hand gesture control for Windows.

This package is a clean v2 rebuild. The v1 project was a Flask web application
that streamed a webcam feed to a browser and fired browser-side actions. That
architecture is retired. GesturePilot runs natively on Windows and drives the
operating system directly.

The target pipeline is::

    Camera
      -> Hand Tracker
      -> Landmark Processing
      -> Gesture Classifier
      -> Confidence Filter
      -> Safety State Machine
      -> Gesture Event
      -> Action Dispatcher
      -> OS-specific Actions

Only the packaging foundation exists at this stage. Each stage of the pipeline
is introduced in a later phase.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
