"""Lifecycle, capture, and error-handling tests for :mod:`gesturepilot.camera.camera`.

No physical webcam is involved: every test injects a ``FakeCapture`` factory.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gesturepilot.camera import (
    CameraConfig,
    CameraOpenError,
    CameraStateError,
    FrameReadError,
    WebcamCamera,
    open_camera,
)
from gesturepilot.camera.errors import CameraError

from .conftest import FakeCapture, FakeCaptureFactory, FakeClock, make_frame


def make_camera(
    config: CameraConfig | None = None,
    capture_factory: FakeCaptureFactory | None = None,
    clock: FakeClock | None = None,
) -> WebcamCamera:
    """Build a camera wired to test doubles."""
    return WebcamCamera(
        config,
        capture_factory=capture_factory or FakeCaptureFactory(FakeCapture()),
        clock=clock or FakeClock(),
    )


# --------------------------------------------------------------------------
# 1. Opening successfully
# --------------------------------------------------------------------------


def test_open_succeeds_and_marks_camera_open(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)

    camera.open()

    assert camera.is_open is True


def test_open_uses_configured_device_index() -> None:
    factory = FakeCaptureFactory(FakeCapture())

    camera = make_camera(CameraConfig(device_index=3), factory)
    camera.open()

    assert factory.requested_indices == [3]


def test_open_camera_helper_returns_open_camera(capture_factory: FakeCaptureFactory) -> None:
    camera = open_camera(CameraConfig(device_index=1), capture_factory=capture_factory)

    assert camera.is_open is True
    camera.close()


# --------------------------------------------------------------------------
# 2. Open failure
# --------------------------------------------------------------------------


def test_open_failure_raises_camera_open_error() -> None:
    factory = FakeCaptureFactory(FakeCapture(opens=False))

    with pytest.raises(CameraOpenError):
        make_camera(capture_factory=factory).open()


def test_open_failure_error_is_a_camera_error() -> None:
    factory = FakeCaptureFactory(FakeCapture(opens=False))

    with pytest.raises(CameraError):
        make_camera(capture_factory=factory).open()


def test_open_failure_leaves_camera_closed() -> None:
    factory = FakeCaptureFactory(FakeCapture(opens=False))

    camera = make_camera(capture_factory=factory)
    with pytest.raises(CameraOpenError):
        camera.open()

    assert camera.is_open is False


def test_open_failure_releases_the_failed_capture() -> None:
    """A half-opened handle must be released, not leaked."""
    capture = FakeCapture(opens=False)
    factory = FakeCaptureFactory(capture)

    with pytest.raises(CameraOpenError):
        make_camera(capture_factory=factory).open()

    assert capture.release_count == 1


def test_open_failure_when_factory_returns_none() -> None:
    def factory(device_index: int) -> None:
        return None

    with pytest.raises(CameraOpenError):
        make_camera(capture_factory=factory).open()


def test_open_failure_message_is_actionable() -> None:
    factory = FakeCaptureFactory(FakeCapture(opens=False))

    with pytest.raises(CameraOpenError) as excinfo:
        make_camera(CameraConfig(device_index=2), factory).open()

    message = str(excinfo.value)
    assert "device 2" in message
    assert "another application" in message


def test_factory_exception_is_not_swallowed() -> None:
    """Programming errors from an injected factory must surface, not be hidden."""
    factory = FakeCaptureFactory(raises=RuntimeError("driver exploded"))

    with pytest.raises(RuntimeError, match="driver exploded"):
        make_camera(capture_factory=factory).open()


# --------------------------------------------------------------------------
# 3. Reading valid frames
# --------------------------------------------------------------------------


def test_read_returns_frame_with_valid_image(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()

    frame = camera.read()

    assert isinstance(frame.image, np.ndarray)
    assert frame.image.shape == (480, 640, 3)
    assert camera.frames_read == 1


def test_read_returns_distinct_frames(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()

    first, second = camera.read(), camera.read()

    assert first.sequence == 0
    assert second.sequence == 1


def test_read_does_not_mutate_the_image(capture_factory: FakeCaptureFactory) -> None:
    """The camera must hand back the captured pixels unmodified."""
    original = make_frame(width=320, height=240)
    capture = FakeCapture(frames=[(True, original)])
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))
    camera.open()

    frame = camera.read()

    assert np.array_equal(frame.image, original)


# --------------------------------------------------------------------------
# 4. Frame read failure
# --------------------------------------------------------------------------


def test_read_failure_raises_frame_read_error() -> None:
    capture = FakeCapture(frames=[(False, None)])
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))
    camera.open()

    with pytest.raises(FrameReadError):
        camera.read()


def test_read_failure_when_image_is_none_despite_ok_flag() -> None:
    capture = FakeCapture(frames=[(True, None)])
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))
    camera.open()

    with pytest.raises(FrameReadError):
        camera.read()


def test_read_failure_on_empty_image() -> None:
    """A zero-byte frame must never be handed back as if it were valid."""
    capture = FakeCapture(frames=[(True, np.empty((0, 0, 3), dtype=np.uint8))])
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))
    camera.open()

    with pytest.raises(FrameReadError):
        camera.read()


def test_read_failure_on_one_dimensional_image() -> None:
    capture = FakeCapture(frames=[(True, np.zeros((10,), dtype=np.uint8))])
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))
    camera.open()

    with pytest.raises(FrameReadError):
        camera.read()


def test_failed_read_does_not_increment_counter() -> None:
    capture = FakeCapture(frames=[(False, None), (True, make_frame())])
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))
    camera.open()

    with pytest.raises(FrameReadError):
        camera.read()

    assert camera.frames_read == 0


def test_frame_read_error_is_a_camera_error() -> None:
    capture = FakeCapture(frames=[(False, None)])
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))
    camera.open()

    with pytest.raises(CameraError):
        camera.read()


# --------------------------------------------------------------------------
# 5 & 6. Closing, and closing more than once
# --------------------------------------------------------------------------


def test_close_releases_the_capture(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()

    camera.close()

    assert capture_factory.capture.release_count == 1
    assert camera.is_open is False


def test_close_is_safe_when_already_closed() -> None:
    camera = make_camera()
    camera.open()

    camera.close()
    camera.close()
    camera.close()

    assert camera.is_open is False


def test_close_is_safe_without_ever_opening() -> None:
    camera = make_camera()

    camera.close()

    assert camera.is_open is False


def test_close_releases_exactly_once_per_open(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()

    camera.close()
    camera.close()

    assert capture_factory.capture.release_count == 1


def test_close_resets_frame_counter(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()
    camera.read()
    camera.read()

    camera.close()

    assert camera.frames_read == 0
    assert camera.measured_fps is None


def test_close_resets_reported_properties(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()
    camera.read()

    camera.close()

    assert camera.properties.actual_width is None
    assert camera.properties.actual_height is None


# --------------------------------------------------------------------------
# 7. Context manager
# --------------------------------------------------------------------------


def test_context_manager_opens_and_closes() -> None:
    factory = FakeCaptureFactory(FakeCapture())

    with make_camera(capture_factory=factory) as camera:
        assert camera.is_open is True
        assert camera.read() is not None

    assert camera.is_open is False
    assert factory.capture.release_count == 1


def test_context_manager_yields_self() -> None:
    camera = make_camera()

    with camera as entered:
        assert entered is camera


def test_context_manager_closes_on_exception() -> None:
    factory = FakeCaptureFactory(FakeCapture())
    camera = make_camera(capture_factory=factory)

    with pytest.raises(ValueError, match="boom"), camera:
        raise ValueError("boom")

    assert camera.is_open is False
    assert factory.capture.release_count == 1


def test_context_manager_releases_when_read_fails_inside_block() -> None:
    capture = FakeCapture(frames=[(False, None)])
    factory = FakeCaptureFactory(capture)

    with pytest.raises(FrameReadError), make_camera(capture_factory=factory) as camera:
        camera.read()

    assert camera.is_open is False
    assert capture.release_count == 1


# --------------------------------------------------------------------------
# 8. Invalid lifecycle operations
# --------------------------------------------------------------------------


def test_read_before_open_raises_state_error() -> None:
    with pytest.raises(CameraStateError):
        make_camera().read()


def test_open_twice_raises_state_error(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()

    with pytest.raises(CameraStateError):
        camera.open()


def test_read_after_close_raises_state_error(capture_factory: FakeCaptureFactory) -> None:
    camera = make_camera(capture_factory=capture_factory)
    camera.open()
    camera.close()

    with pytest.raises(CameraStateError):
        camera.read()


def test_state_error_is_a_camera_error() -> None:
    with pytest.raises(CameraError):
        make_camera().read()


def test_camera_can_be_reopened_after_close() -> None:
    """CLOSED -> OPEN -> CLOSED -> OPEN must be allowed."""
    captures: list[FakeCapture] = []

    def factory(device_index: int) -> FakeCapture:
        capture = FakeCapture()
        captures.append(capture)
        return capture

    camera = WebcamCamera(CameraConfig(), capture_factory=factory, clock=FakeClock())

    camera.open()
    camera.close()
    camera.open()

    assert camera.is_open is True
    assert len(captures) == 2
    assert captures[0].release_count == 1
    assert captures[1].release_count == 0
    camera.close()


# --------------------------------------------------------------------------
# 9. Configuration is passed to the device
# --------------------------------------------------------------------------


def test_configured_resolution_is_requested() -> None:
    capture = FakeCapture()
    camera = make_camera(CameraConfig(width=1920, height=1080), FakeCaptureFactory(capture))

    camera.open()

    assert (cv2.CAP_PROP_FRAME_WIDTH, 1920.0) in capture.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 1080.0) in capture.set_calls


def test_configured_fps_is_requested() -> None:
    capture = FakeCapture()
    camera = make_camera(CameraConfig(fps=60.0), FakeCaptureFactory(capture))

    camera.open()

    assert (cv2.CAP_PROP_FPS, 60.0) in capture.set_calls


def test_fps_request_is_skipped_when_none() -> None:
    capture = FakeCapture()
    camera = make_camera(CameraConfig(fps=None), FakeCaptureFactory(capture))

    camera.open()

    assert not any(prop == cv2.CAP_PROP_FPS for prop, _ in capture.set_calls)


def test_device_reported_properties_are_exposed() -> None:
    capture = FakeCapture(
        props={
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 24.0,
        }
    )
    camera = make_camera(CameraConfig(width=1920, height=1080), FakeCaptureFactory(capture))

    camera.open()

    props = camera.properties
    assert (props.requested_width, props.requested_height) == (1920, 1080)
    assert (props.actual_width, props.actual_height) == (640, 480)
    assert props.actual_fps == 24.0
    assert props.matched_request is False


def test_unsupported_reported_values_become_none() -> None:
    """Drivers report 0.0 or NaN for properties they do not implement."""
    capture = FakeCapture(
        props={
            cv2.CAP_PROP_FRAME_WIDTH: 0.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 0.0,
            cv2.CAP_PROP_FPS: float("nan"),
        }
    )
    camera = make_camera(capture_factory=FakeCaptureFactory(capture))

    camera.open()

    props = camera.properties
    assert props.actual_width is None
    assert props.actual_height is None
    assert props.actual_fps is None
    assert props.matched_request is False


def test_matching_resolution_is_reported_as_matched() -> None:
    capture = FakeCapture(
        props={cv2.CAP_PROP_FRAME_WIDTH: 1280.0, cv2.CAP_PROP_FRAME_HEIGHT: 720.0}
    )
    camera = make_camera(CameraConfig(), FakeCaptureFactory(capture))

    camera.open()

    assert camera.properties.matched_request is True


def test_properties_are_unset_before_opening() -> None:
    camera = make_camera(CameraConfig(width=800, height=600))

    props = camera.properties

    assert props.requested_width == 800
    assert props.requested_height == 600
    assert props.actual_width is None
