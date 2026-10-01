"""Tests for measured FPS and frame metadata.

Timing is driven entirely by an injected clock. Nothing here sleeps, and no test
depends on real-world frame rate behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesturepilot.camera import CameraConfig, FpsMeter, WebcamCamera

from .conftest import FakeCapture, FakeCaptureFactory, FakeClock, make_frame

# --------------------------------------------------------------------------
# FpsMeter
# --------------------------------------------------------------------------


def test_fps_is_none_before_any_ticks(clock: FakeClock) -> None:
    meter = FpsMeter(window=1.0, _clock=clock)

    assert meter.fps is None


def test_fps_is_none_after_a_single_tick(clock: FakeClock) -> None:
    meter = FpsMeter(window=1.0, _clock=clock)

    meter.tick()

    assert meter.fps is None


def test_first_tick_returns_no_interval(clock: FakeClock) -> None:
    """There is no preceding sample to measure against."""
    meter = FpsMeter(window=1.0, _clock=clock)

    assert meter.tick() is None


def test_tick_returns_elapsed_since_previous_call(clock: FakeClock) -> None:
    meter = FpsMeter(window=1.0, _clock=clock)

    meter.tick()
    clock.advance(0.25)

    assert meter.tick() == pytest.approx(0.25)


def test_measured_fps_matches_controlled_timing(clock: FakeClock) -> None:
    """Eleven samples a tenth of a second apart span 1.0s => 10 intervals => 10 FPS."""
    meter = FpsMeter(window=1.0, _clock=clock)

    for _ in range(11):
        meter.tick()
        clock.advance(0.1)

    assert meter.fps == pytest.approx(10.0)


def test_measured_fps_tracks_a_slower_capture_rate(clock: FakeClock) -> None:
    """A camera asked for 30 FPS that delivers 5 must report 5, not 30."""
    meter = FpsMeter(window=1.0, _clock=clock)

    for _ in range(11):
        meter.tick()
        clock.advance(0.2)

    assert meter.fps == pytest.approx(5.0)


def test_window_discards_samples_outside_it(clock: FakeClock) -> None:
    meter = FpsMeter(window=0.5, _clock=clock)

    for _ in range(5):
        meter.tick()
        clock.advance(1.0)

    # Old samples fall out of the trailing window instead of skewing the average.
    assert meter.sample_count == 1
    assert meter.fps is None


def test_measured_fps_is_none_when_all_samples_share_a_timestamp(clock: FakeClock) -> None:
    meter = FpsMeter(window=1.0, _clock=clock)

    meter.tick()
    meter.tick()

    assert meter.fps is None


def test_reset_clears_measurements(clock: FakeClock) -> None:
    meter = FpsMeter(window=1.0, _clock=clock)
    meter.tick()
    meter.tick()

    meter.reset()

    assert meter.fps is None
    assert meter.sample_count == 0


def test_invalid_window_is_rejected(clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="window"):
        FpsMeter(window=0.0, _clock=clock)


# --------------------------------------------------------------------------
# FPS through the camera
# --------------------------------------------------------------------------


def test_camera_fps_is_none_before_reading(clock: FakeClock) -> None:
    camera = WebcamCamera(
        CameraConfig(),
        capture_factory=FakeCaptureFactory(FakeCapture()),
        clock=clock,
    )
    camera.open()

    assert camera.measured_fps is None


def test_camera_measures_actual_read_rate(clock: FakeClock) -> None:
    """Read at a steady 10 Hz with an injected clock; measurement must say 10 FPS."""
    camera = WebcamCamera(
        CameraConfig(fps=30.0),
        capture_factory=FakeCaptureFactory(FakeCapture()),
        clock=clock,
    )
    camera.open()

    for _ in range(11):
        camera.read()
        clock.advance(0.1)

    assert camera.measured_fps == pytest.approx(10.0)


def test_measured_fps_is_independent_of_requested_fps(clock: FakeClock) -> None:
    """Requested 60, delivered 10. The camera must not conflate the two."""
    camera = WebcamCamera(
        CameraConfig(fps=60.0),
        capture_factory=FakeCaptureFactory(FakeCapture()),
        clock=clock,
    )
    camera.open()

    for _ in range(11):
        camera.read()
        clock.advance(0.1)

    assert camera.config.fps == 60.0
    assert camera.measured_fps == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Frame metadata
# --------------------------------------------------------------------------


def test_frame_metadata_matches_the_image(clock: FakeClock) -> None:
    capture = FakeCapture(frames=[(True, make_frame(width=1024, height=768))])
    camera = WebcamCamera(
        CameraConfig(),
        capture_factory=FakeCaptureFactory(capture),
        clock=clock,
    )
    camera.open()

    frame = camera.read()

    assert frame.width == 1024
    assert frame.height == 768
    assert frame.channels == 3


def test_frame_timestamp_comes_from_the_injected_clock(clock: FakeClock) -> None:
    clock.advance(12.5)
    camera = WebcamCamera(
        CameraConfig(),
        capture_factory=FakeCaptureFactory(FakeCapture()),
        clock=clock,
    )
    camera.open()

    frame = camera.read()

    assert frame.timestamp == pytest.approx(12.5)


def test_frame_sequence_increments(clock: FakeClock) -> None:
    camera = WebcamCamera(
        CameraConfig(),
        capture_factory=FakeCaptureFactory(FakeCapture()),
        clock=clock,
    )
    camera.open()

    sequences = [camera.read().sequence for _ in range(3)]

    assert sequences == [0, 1, 2]


def test_frame_image_is_a_numpy_array(clock: FakeClock) -> None:
    camera = WebcamCamera(
        CameraConfig(),
        capture_factory=FakeCaptureFactory(FakeCapture()),
        clock=clock,
    )
    camera.open()

    assert isinstance(camera.read().image, np.ndarray)


def test_grayscale_frame_reports_one_channel(clock: FakeClock) -> None:
    capture = FakeCapture(frames=[(True, make_frame(width=320, height=240, channels=1))])
    camera = WebcamCamera(
        CameraConfig(),
        capture_factory=FakeCaptureFactory(capture),
        clock=clock,
    )
    camera.open()

    frame = camera.read()

    assert frame.channels == 1
    assert frame.width == 320
