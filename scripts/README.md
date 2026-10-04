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

## `pipeline_smoke_test.py`

Runs all five layers together over live webcam frames — camera, tracking, processing,
classification, and confidence filter — and prints one status line per refresh:

```
frame    42 | hand  yes  | gesture POINT     0.82 | candidate POINT 2/3 @  0.80 | stable no
```

Requires a physical camera **and** the model asset, same as the tracking smoke test.

```powershell
uv run python scripts/pipeline_smoke_test.py
uv run python scripts/pipeline_smoke_test.py --device 1 --max-hands 1
uv run python scripts/pipeline_smoke_test.py --smoothing --min-observations 5
uv run python scripts/pipeline_smoke_test.py --max-interruption 0.4
uv run python scripts/pipeline_smoke_test.py --frames 0 --show   # run until 'q'
```

It is the place a mistake *between* layers shows up: a timestamp that does not reach the
next stage, or a gesture that stays on screen after the hand holding it has left the frame.
A frame with no hand clears the state and says so, rather than inventing an `UNKNOWN` for a
frame that was never classified.

It decides nothing and acts on nothing: no OS action, no input, no GUI unless `--show` is
passed, and nothing written to disk — no frames, no recordings, no logs, no network. The
run summary at the end says so too, because `stable` means *seen consistently*, not
*confirmed*.

Exit codes: `0` clean run, `2` model asset missing, `3` processing error, `4` camera error,
`5` tracking error, `6` classification or confidence-filter error, `1` anything else.

Nothing here downloads a model or writes to the project; the `--show` preview only draws
over an in-memory copy of the frame.
"""
