"""Real-webcam integration tests.

Opt-in only. Excluded from the default test run via the ``integration`` marker and the
``-m "not integration"`` default in ``pyproject.toml``.

Run manually on Windows::

    uv run pytest -m integration tests/integration/camera -v

Every test skips cleanly when no camera exists or the device is already in use, so
this command never turns into a false failure on another machine.
"""

from __future__ import annotations

import pytest

from gesturepilot.camera import CameraConfig, CameraOpenError, CameraStateError, WebcamCamera

pytestmark = pytest.mark.integration

DEVICE_INDEX = 0
REQUESTED = CameraConfig(device_index=DEVICE_INDEX, width=640, height=480, fps=30.0)


@pytest.fixture
def physical_camera() -> WebcamCamera:
    """Open a real camera, skipping if the device is unavailable."""
    camera = WebcamCamera(REQUESTED)
    try:
        camera.open()
    except CameraOpenError as exc:
        pytest.skip(f"No usable camera at index {DEVICE_INDEX}: {exc}")
    try:
        yield camera
    finally:
        camera.close()


def test_real_camera_opens(physical_camera: WebcamCamera) -> None:
    assert physical_camera.is_open is True


def test_real_camera_reads_a_valid_frame(physical_camera: WebcamCamera) -> None:
    frame = physical_camera.read()

    assert frame.height > 0
    assert frame.width > 0
    assert frame.image.size > 0


def test_real_camera_reports_a_measured_frame_rate(physical_camera: WebcamCamera) -> None:
    """Measures real throughput; no assertion on a specific value, since it varies."""
    for _ in range(10):
        physical_camera.read()

    measured = physical_camera.measured_fps

    assert measured is not None
    assert measured > 0


def test_real_camera_closes_and_refuses_further_reads(physical_camera: WebcamCamera) -> None:
    physical_camera.close()

    assert physical_camera.is_open is False
    with pytest.raises(CameraStateError):
        physical_camera.read()


def test_absent_device_index_raises_camera_open_error() -> None:
    """An out-of-range index must surface the custom error, not a raw OpenCV one."""
    camera = WebcamCamera(CameraConfig(device_index=99))
    try:
        camera.open()
    except CameraOpenError as exc:
        assert "99" in str(exc)
    else:
        camera.close()
        pytest.skip("Device index 99 unexpectedly opened.")
