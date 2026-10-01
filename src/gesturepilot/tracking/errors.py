"""Hand-tracking exception hierarchy.

Callers handle tracking failures through these types, never raw MediaPipe errors.
This keeps MediaPipe confined to :mod:`gesturepilot.tracking.tracker` and yields
messages a developer can act on.

Four distinct failure modes:

``ModelAssetError``
    The hand-landmarker model file is missing or unusable.
``TrackerInitializationError``
    The tracker could not be set up (model load failure, bad options).
``TrackingProcessingError``
    A frame could not be processed, or was not a usable frame.
``TrackerStateError``
    An operation was invalid for the tracker's current state.

All derive from ``TrackingError``.
"""

from __future__ import annotations

from pathlib import Path


class TrackingError(Exception):
    """Base class for every hand-tracking failure."""


class ModelAssetError(TrackingError):
    """The required hand-landmarker model asset is missing or unusable.

    Attributes:
        searched: Locations that were inspected, in the order they were tried.
    """

    def __init__(self, message: str, searched: list[Path] | None = None) -> None:
        super().__init__(message)
        self.searched: list[Path] = searched or []


class TrackerInitializationError(TrackingError):
    """The tracker could not be initialised.

    Typically the model exists but could not be loaded, or the supplied options were
    rejected. The originating exception is chained.
    """


class TrackingProcessingError(TrackingError):
    """A frame could not be tracked.

    Covers both an unusable input frame and a failure inside the inference backend.
    """


class TrackerStateError(TrackingError):
    """An operation was invalid for the tracker's current state.

    For example, processing before ``initialize()``, processing after ``close()``, or
    initialising twice.
    """


__all__ = [
    "ModelAssetError",
    "TrackerInitializationError",
    "TrackerStateError",
    "TrackingError",
    "TrackingProcessingError",
]
