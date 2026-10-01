# Model Assets

GesturePilot loads its models from this directory. **Model binaries are not committed
to Git** — they are large, and keeping them out of history keeps the repository small
and avoids redistribution questions.

---

## Required asset: `hand_landmarker.task`

Used by the hand-tracking layer (Phase 3) for every frame the camera produces.

| Property | Value |
|---|---|
| Filename | `hand_landmarker.task` |
| Expected location | `models/hand_landmarker.task` |
| Source | [MediaPipe hand landmarker model catalogue](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
| Format | MediaPipe Tasks flatbuffer bundle |

### How to obtain it

1. Open the model catalogue page linked above.
2. Download the hand landmarker model bundle.
3. Rename it to `hand_landmarker.task` (if it downloaded under another name).
4. Place it in this directory:

   ```powershell
   # from the repository root
   New-Item -ItemType Directory -Path models -Force
   # then copy the downloaded file to models\hand_landmarker.task
   ```

### How the application finds it

`gesturepilot.tracking.assets.resolve_model_path` searches, first match wins:

1. `TrackerConfig.model_path`, when set explicitly
2. the `GESTUREPILOT_HAND_LANDMARKER_MODEL` environment variable
3. `models/hand_landmarker.task` beside the project root
4. `models/hand_landmarker.task` beneath the current working directory

```powershell
# Option A: default location (models/ at the project root)
# Option B: explicit via config
$env:GESTUREPILOT_HAND_LANDMARKER_MODEL = "D:\models\hand_landmarker.task"
```

If no candidate exists, `ModelAssetError` is raised listing every path that was
inspected and where to get the file.

### Nothing is ever downloaded automatically

There is no runtime fetch. GesturePilot will not reach the network to obtain a model,
so it cannot silently download a large asset during startup or make the application
fail on a machine without connectivity.

---

## Unit tests do not need this asset

The unit suite injects a fake landmarker, so `models/hand_landmarker.task` is not
required to run:

```powershell
uv run pytest
```

Only the opt-in real-MediaPipe smoke test needs the asset:

```powershell
uv run pytest -m integration tests/integration/tracking -v
```

That test reports a clear, actionable error if the model is absent rather than
downloading anything.

---

## Ignoring binaries

`.gitignore` excludes model binaries so they cannot be committed by accident:

```
models/*.task
models/*.tflite
models/*.bin
```
