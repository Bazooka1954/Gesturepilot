"""Validation tests for :class:`CameraConfig` and :class:`CameraProperties`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from gesturepilot.camera import CameraConfig, CameraProperties


def test_defaults_are_reasonable() -> None:
    config = CameraConfig()

    assert config.device_index == 0
    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30.0


def test_fps_may_be_omitted() -> None:
    assert CameraConfig(fps=None).fps is None


def test_config_is_immutable() -> None:
    config = CameraConfig()

    with pytest.raises(FrozenInstanceError):
        config.width = 640  # type: ignore[misc]


def test_resolution_property() -> None:
    assert CameraConfig(width=640, height=480).resolution == (640, 480)


@pytest.mark.parametrize("index", [-1, -10])
def test_negative_device_index_is_rejected(index: int) -> None:
    with pytest.raises(ValueError, match="device_index"):
        CameraConfig(device_index=index)


@pytest.mark.parametrize("width", [0, -640])
def test_non_positive_width_is_rejected(width: int) -> None:
    with pytest.raises(ValueError, match="width"):
        CameraConfig(width=width)


@pytest.mark.parametrize("height", [0, -480])
def test_non_positive_height_is_rejected(height: int) -> None:
    with pytest.raises(ValueError, match="height"):
        CameraConfig(height=height)


@pytest.mark.parametrize("fps", [0.0, -30.0])
def test_non_positive_fps_is_rejected(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        CameraConfig(fps=fps)


def test_unknown_actual_values_are_absent_by_default() -> None:
    props = CameraProperties(requested_width=1280, requested_height=720, requested_fps=30.0)

    assert props.actual_resolution is None
    assert props.actual_fps is None
    assert props.matched_request is False
