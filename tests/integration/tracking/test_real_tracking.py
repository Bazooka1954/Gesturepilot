"""Real-model hand-tracking integration tests.

Opt-in only. Excluded from the default run via the ``integration`` marker and the
``-m "not integration"`` default in ``pyproject.toml``.

Requires the real model asset at ``models/hand_landmarker.task`` (see
``models/README.md``). Nothing is ever downloaded automatically, so these tests skip
with an actionable message when the asset is absent::

    uv run pytest -m integration tests/integration/tracking -v

These tests verify the real MediaPipe API behaves as the tracking layer assumes: that
the model loads, that a synthetic frame round-trips, and that a genuinely empty frame
produces zero hands rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesturepilot.camera.types import Frame
from gesturepilot.tracking import (
    LANDMARK_COUNT,
    MediaPipeHandTracker,
    ModelAssetError,
    RunningMode,
    TrackerConfig,
    TrackerInitializationError,
    TrackerStateError,
)
from gesturepilot.tracking.assets import resolve_model_path

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model_available() -> None:
    """Skip the module when the model asset is not on disk."""
    try:
        resolve_model_path()
    except ModelAssetError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def tracker() -> MediaPipeHandTracker:
    """A real, initialized tracker, skipped if the model cannot be loaded."""
    instance = MediaPipeHandTracker(TrackerConfig(running_mode=RunningMode.IMAGE))
    try:
        instance.initialize()
    except (ModelAssetError, TrackerInitializationError) as exc:
        pytest.skip(f"Real hand-landmarker model unavailable: {exc}")
    try:
        yield instance
    finally:
        instance.close()


def make_black_frame(width: int = 640, height: int = 480, sequence: int = 0) -> Frame:
    """A realistic dark frame containing no hand."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(image=image, timestamp=1.0, sequence=sequence)


def test_real_model_initializes(tracker: MediaPipeHandTracker) -> None:
    assert tracker.is_initialized is True


def test_real_model_returns_a_result_for_a_real_frame(tracker: MediaPipeHandTracker) -> None:
    result = tracker.process(make_black_frame(sequence=0))

    # No hand is present, so zero hands is the correct outcome - not an error.
    assert result.hand_count == 0
    assert result.has_hands is False
    assert result.primary_hand is None
    assert result.timestamp == 1.0
    assert result.frame_sequence == 0


def test_real_model_accepts_a_video_mode_sequence(model_available: None) -> None:
    """VIDEO mode must accept a strictly increasing timestamp sequence."""
    tracker = MediaPipeHandTracker(TrackerConfig(running_mode=RunningMode.VIDEO))
    try:
        tracker.initialize()
    except (ModelAssetError, TrackerInitializationError) as exc:
        pytest.skip(f"Real hand-landmarker model unavailable: {exc}")

    try:
        for i in range(5):
            frame = Frame(
                image=np.zeros((240, 320, 3), dtype=np.uint8),
                timestamp=1.0 + i / 30.0,  # 30 FPS cadence
                sequence=i,
            )
            result = tracker.process(frame)
            assert result.hand_count == 0
            assert result.frame_sequence == i
    finally:
        tracker.close()


def test_real_model_rejects_lifecycle_misuse() -> None:
    """A tracker that was never initialized must refuse to process frames."""
    try:
        tracker = MediaPipeHandTracker()
        tracker.initialize()
    except (ModelAssetError, TrackerInitializationError) as exc:
        pytest.skip(f"Real hand-landmarker model unavailable: {exc}")

    tracker.close()
    with pytest.raises(TrackerStateError):
        tracker.process(make_black_frame())


def test_real_model_hand_count_never_exceeds_config(model_available: None) -> None:
    """max_hands must be honoured even for a hand-free frame."""
    config = TrackerConfig(max_hands=1, running_mode=RunningMode.IMAGE)
    tracker = MediaPipeHandTracker(config)
    try:
        tracker.initialize()
    except (ModelAssetError, TrackerInitializationError) as exc:
        pytest.skip(f"Real hand-landmarker model unavailable: {exc}")

    try:
        result = tracker.process(make_black_frame())
        assert result.hand_count <= config.max_hands
    finally:
        tracker.close()


def test_landmark_count_matches_model_contract() -> None:
    """The documented landmark count must match the constant we expose."""
    assert LANDMARK_COUNT == 21
