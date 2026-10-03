"""Static gesture classification: measured features in, one named verdict out.

This layer implements the **Gesture Classifier** stage of the pipeline. It takes the
:class:`~gesturepilot.processing.types.HandFeatures` that the processing stage produced
and returns a :class:`~gesturepilot.classifier.types.GestureClassification`: which of
five static hand configurations the hand most resembles, how strongly, and how much
better that beats the alternatives.

It classifies **one frame at a time**. There is no memory of previous frames, no
smoothing, no debouncing, no cooldown, and no notion of a transition or a swipe — those
are the confidence filter's and the safety state machine's jobs, and doing them here
would mean a classification could no longer be checked from a single input. Nor does
this package reject anything on confidence: it reports what it measured, including
``UNKNOWN``, and leaves deciding what to do with a weak result to the next stage.

The rules are ordinary `if`-shaped arithmetic on normalised geometry, chosen so a
developer can read the whole recogniser and change it deliberately. Nothing is trained,
downloaded, or calibrated against a data set; every threshold lives in
:class:`ClassifierConfig` and is validated where it is written. See
:mod:`gesturepilot.classifier.classifier` for the rule table and
``docs/classifier.md`` for why each rule is shaped the way it is.

Boundaries: this package imports only from ``gesturepilot.processing``. It reaches
neither the camera, nor MediaPipe, nor OpenCV, nor NumPy, and it never touches a clock,
a file, or a network — enforced by AST tests in
``tests/unit/classifier/test_boundaries.py``. Handedness is not an input to any rule; it
is source metadata that the tracking layer may have inverted.

Typical use::

    from gesturepilot.classifier import GestureClassifier, Gesture

    classifier = GestureClassifier()
    result = classifier.classify(features)

    if result.is_recognized and result.gesture is Gesture.POINT:
        print(f"pointing at {result.confidence:.0%} confidence")

No MediaPipe, OpenCV, or NumPy type is exposed anywhere in this API.
"""

from gesturepilot.classifier.classifier import GestureClassifier, HandMeasurements
from gesturepilot.classifier.errors import ClassificationError, InvalidFeaturesError
from gesturepilot.classifier.types import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    RECOGNIZABLE_GESTURES,
    ClassifierConfig,
    Gesture,
    GestureClassification,
)

__all__ = [
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "RECOGNIZABLE_GESTURES",
    "ClassificationError",
    "ClassifierConfig",
    "Gesture",
    "GestureClassification",
    "GestureClassifier",
    "HandMeasurements",
    "InvalidFeaturesError",
]
