"""Unit tests for model-asset discovery.

Discovery must never hit the network: a missing model is a clear, actionable error,
not a silent download.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gesturepilot.tracking.assets import (
    DEFAULT_MODEL_FILENAME,
    MODEL_PATH_ENV_VAR,
    candidate_paths,
    resolve_model_path,
)
from gesturepilot.tracking.errors import ModelAssetError, TrackingError


@pytest.fixture(autouse=True)
def clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real environment variable out of these tests."""
    monkeypatch.delenv(MODEL_PATH_ENV_VAR, raising=False)


class TestResolveModelPath:
    def test_finds_explicit_existing_path(self, tmp_path: Path) -> None:
        model = tmp_path / DEFAULT_MODEL_FILENAME
        model.write_bytes(b"fake-model-bytes")
        assert resolve_model_path(model) == model

    def test_finds_explicit_path_as_string(self, tmp_path: Path) -> None:
        model = tmp_path / "custom.task"
        model.write_bytes(b"x")
        assert resolve_model_path(str(model)) == model

    def test_finds_via_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model = tmp_path / "env.task"
        model.write_bytes(b"x")
        monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(model))
        assert resolve_model_path() == model

    def test_raises_when_nothing_found(self, tmp_path: Path) -> None:
        with pytest.raises(ModelAssetError):
            resolve_model_path(tmp_path / "absent.task")

    def test_error_lists_searched_paths(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.task"
        with pytest.raises(ModelAssetError) as excinfo:
            resolve_model_path(missing)

        message = str(excinfo.value)
        assert str(missing) in message
        assert str(excinfo.value.searched[0]) == str(missing)

    def test_error_explains_how_to_obtain_the_asset(self, tmp_path: Path) -> None:
        with pytest.raises(ModelAssetError) as excinfo:
            resolve_model_path(tmp_path / "absent.task")

        message = str(excinfo.value)
        assert MODEL_PATH_ENV_VAR in message
        assert DEFAULT_MODEL_FILENAME in message

    def test_directory_is_not_accepted_as_asset(self, tmp_path: Path) -> None:
        # A directory named like the model must not be mistaken for the file.
        decoy_dir = tmp_path / "decoy"
        decoy_dir.mkdir()
        (decoy_dir / DEFAULT_MODEL_FILENAME).mkdir()
        with pytest.raises(ModelAssetError):
            resolve_model_path(decoy_dir / DEFAULT_MODEL_FILENAME)

    def test_model_error_is_tracking_error(self) -> None:
        assert issubclass(ModelAssetError, TrackingError)

    def test_does_not_raise_if_repo_model_present(self) -> None:
        # The repository checkout may legitimately contain the asset; discovery must
        # succeed silently, and must still not attempt any download.
        try:
            resolved = resolve_model_path()
        except ModelAssetError:
            return  # asset absent, which is the normal case
        assert resolved.is_file()


class TestCandidatePaths:
    def test_explicit_path_is_the_only_candidate(self, tmp_path: Path) -> None:
        explicit = tmp_path / "only.task"
        assert candidate_paths(explicit) == [explicit]

    def test_env_var_precedes_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_model = tmp_path / "env.task"
        monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(env_model))
        candidates = candidate_paths()
        assert candidates[0] == env_model

    def test_defaults_include_project_root_and_cwd(self) -> None:
        candidates = candidate_paths()
        assert candidates, "expected at least one default location"
        assert all(path.name == DEFAULT_MODEL_FILENAME for path in candidates)
        assert candidates[-1].parent == Path.cwd() / "models"

    def test_duplicate_locations_are_collapsed(self) -> None:
        """Running from the project root makes two defaults identical."""
        candidates = candidate_paths()
        normalised = [os.path.normcase(os.path.abspath(p)) for p in candidates]
        assert len(normalised) == len(set(normalised))

    def test_explicit_path_ignores_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(tmp_path / "env.task"))
        explicit = tmp_path / "explicit.task"
        assert candidate_paths(explicit) == [explicit]
