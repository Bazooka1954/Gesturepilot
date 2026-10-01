"""Hand-landmarker model asset discovery.

GesturePilot never downloads a model at runtime. The asset is resolved from the local
filesystem, and a missing file produces a :class:`ModelAssetError` naming exactly
where it looked and where to obtain the file.

Search order, first match wins:

1. ``TrackerConfig.model_path`` when set explicitly.
2. The ``GESTUREPILOT_HAND_LANDMARKER_MODEL`` environment variable.
3. ``models/`` beside the project root (the development checkout layout).
4. ``models/`` beneath the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from gesturepilot.tracking.errors import ModelAssetError

#: Filename of the MediaPipe hand-landmarker task bundle.
DEFAULT_MODEL_FILENAME = "hand_landmarker.task"

#: Environment variable holding an explicit path to the model asset.
MODEL_PATH_ENV_VAR = "GESTUREPILOT_HAND_LANDMARKER_MODEL"

_DOWNLOAD_HINT = (
    f"Download 'hand_landmarker.task' from the MediaPipe model catalogue "
    f"(https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and "
    f"place it at models/{DEFAULT_MODEL_FILENAME}, or point "
    f"{MODEL_PATH_ENV_VAR} at its location."
)


def candidate_paths(explicit: str | Path | None = None) -> list[Path]:
    """Return the asset locations that will be tried, in order.

    Duplicates are dropped while preserving order. Running from the project root makes
    the checkout and working-directory locations identical, and naming the same path
    twice in an error message is just noise.
    """
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        env_value = os.environ.get(MODEL_PATH_ENV_VAR)
        if env_value:
            candidates.append(Path(env_value))
        candidates.extend(_default_locations())

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _default_locations() -> list[Path]:
    filename = DEFAULT_MODEL_FILENAME
    locations: list[Path] = []

    # Development checkout: <repo>/models/hand_landmarker.task
    project_root = Path(__file__).resolve().parents[3]
    locations.append(project_root / "models" / filename)

    # Installed / launched-from-elsewhere: ./models/hand_landmarker.task
    locations.append(Path.cwd() / "models" / filename)

    return locations


def resolve_model_path(explicit: str | Path | None = None) -> Path:
    """Locate the hand-landmarker asset.

    Args:
        explicit: A caller-supplied path. When given, it is the only location tried.

    Returns:
        The resolved asset path.

    Raises:
        ModelAssetError: If no candidate exists. The message lists every path that
            was inspected.
    """
    candidates = candidate_paths(explicit)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    inspected = "\n".join(f"  - {path}" for path in candidates)
    raise ModelAssetError(
        f"Hand-landmarker model asset not found.\nSearched:\n{inspected}\n{_DOWNLOAD_HINT}",
        searched=candidates,
    )
