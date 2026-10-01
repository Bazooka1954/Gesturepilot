#!/usr/bin/env python
"""Developer helper scripts for GesturePilot.

Scripts here are intentionally thin wrappers around project commands. Anything
that belongs in the application belongs in `src/gesturepilot` instead.

## `camera_smoke_test.py`

Prints what a real webcam actually negotiated (resolved resolution, measured FPS).
Requires a physical camera.

```powershell
uv run python scripts/camera_smoke_test.py
uv run python scripts/camera_smoke_test.py --device 1 --width 1280 --height 720 --frames 30
```

## `tracking_smoke_test.py`

Runs the real MediaPipe hand landmarker over live webcam frames and reports detections,
handedness distribution, and inference latency. Requires both a physical camera **and**
the model asset at `models/hand_landmarker.task` (see `models/README.md`).

```powershell
uv run python scripts/tracking_smoke_test.py
uv run python scripts/tracking_smoke_test.py --device 1 --max-hands 2 --frames 120
uv run python scripts/tracking_smoke_test.py --running-mode image
uv run python scripts/tracking_smoke_test.py --show     # opens a landmark preview window
```

Exit codes: `0` completed, `2` model asset missing, `3` model failed to load,
`4` camera error, `5` tracking error.

Nothing here downloads a model or writes to the project; the `--show` preview only draws
over an in-memory copy of the frame.
"""
