"""Shared setup for tracking unit tests.

Every tracking test injects a fake landmarker, so none of them need the real
``models/hand_landmarker.task`` asset or a GPU. Asset *discovery* still runs inside
``MediaPipeHandTracker.initialize()`` before the backend is built, so this fixture
points it at a throwaway file. Tests that specifically exercise discovery behaviour
restore the real resolver themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gesturepilot.tracking import tracker as tracker_module


@pytest.fixture(autouse=True)
def stub_model_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect tracker asset discovery to a throwaway file."""
    asset = tmp_path / "hand_landmarker.task"
    asset.write_bytes(b"not-a-real-model")

    monkeypatch.setattr(tracker_module, "resolve_model_path", lambda explicit=None: asset)
    return asset
