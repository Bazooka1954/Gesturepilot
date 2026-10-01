# Tracking Layer

Phase 3. Implements the **Hand Tracking** stage of the pipeline: one camera `Frame` in,
a structured `TrackingResult` out.

This layer answers one question — *where are the hands in this frame?* It does not
interpret them. Gesture classification, confidence filtering, and smoothing all belong
to later phases.

---

## Layout

```
src/gesturepilot/tracking/
├── __init__.py   # public surface
├── types.py      # Landmark, Handedness, TrackedHand, TrackingResult, TrackerConfig
├── errors.py     # TrackingError hierarchy
├── assets.py     # local model discovery                        (no MediaPipe)
└── tracker.py    # HandTracker protocol + MediaPipeHandTracker  (the only mp import)
```

`tracker.py` is the **only** module that imports `mediapipe`. This is enforced by an
AST-based test (`tests/unit/tracking/test_boundaries.py`) that parses every module in
the package and fails if MediaPipe is imported or referenced anywhere else. Docstrings
may mention MediaPipe; code may not reach for it.

The result of that boundary: swapping or removing the backend cannot ripple upward, and
every later phase codes against `TrackingResult` alone.

---

## API

```python
from gesturepilot.camera import CameraConfig, open_camera
from gesturepilot.tracking import TrackerConfig, MediaPipeHandTracker

config = TrackerConfig(max_hands=2)

# Explicit lifecycle
tracker = MediaPipeHandTracker(config)
tracker.initialize()
result = tracker.process(frame)
tracker.close()

# Or as a context manager (releases the model even if the body raises)
with MediaPipeHandTracker(config) as tracker:
    result = tracker.process(frame)
```

### Lifecycle

```
CREATED --initialize()--> INITIALIZED --close()--> CLOSED
```

- `initialize()` resolves the model asset, then builds the backend. Raises
  `ModelAssetError` if the asset is missing, `TrackerInitializationError` if the model
  cannot be loaded, and `TrackerStateError` if already initialised.
- `close()` releases the model and resets timestamp state. Idempotent and cheap, safe in
  `finally` blocks.
- `process()` before `initialize()` or after `close()` raises `TrackerStateError`.

Asset resolution happens **before** the backend is constructed, so a missing model
never half-initialises the tracker.

No `__del__`. Cleanup is explicit; interpreter-shutdown cleanup is unreliable.

---

## Domain types

All frozen dataclasses, validated on construction.

### `Landmark`

| Field | Meaning |
|---|---|
| `x` | Normalised horizontal position, nominally `0.0`–`1.0` |
| `y` | Normalised vertical position, nominally `0.0`–`1.0` |
| `z` | Depth relative to the wrist; roughly centred on `0.0`, negative toward the camera |

Values are checked for finiteness — a `NaN` would silently corrupt every downstream
calculation. `x` and `y` are *not* range-checked: landmarks at a frame edge legitimately
sit slightly outside `[0, 1]`.

Coordinates stay normalised. Conversion to screen space is a later phase's job, and
that layer is the one that knows the frame dimensions.

### `Handedness`

`LEFT`, `RIGHT`, `UNKNOWN`. An unrecognised backend label maps to `UNKNOWN` rather than
being guessed at, so downstream stages must handle it explicitly.

### `TrackedHand`

| Field | Meaning |
|---|---|
| `landmarks` | Exactly 21 `Landmark` objects, in the model's own order |
| `handedness` | Which hand, or `UNKNOWN` |
| `handedness_score` | Model confidence in that label, or `None` |
| `detection_score` | Per-hand detection confidence, or `None` |

**On `detection_score`:** MediaPipe's Tasks API does **not** emit a per-frame detection
or tracking confidence. Its `min_detection_confidence` / `min_presence_confidence` /
`min_tracking_confidence` are *thresholds applied at init*, not outputs, and they live
in `TrackerConfig`. The field exists so a future backend can populate it without an API
break. Only the handedness label carries a score, and that is exposed.

### `TrackingResult`

| Field | Meaning |
|---|---|
| `hands` | Detected hands, in tracker order. Empty when no hand is present. |
| `timestamp` | The originating frame's monotonic seconds, passed through unchanged |
| `frame_sequence` | The originating frame's sequence number, passed through unchanged |

**Zero hands is a valid result, not an error.** Helpers: `hand_count`, `has_hands`,
`primary_hand` (first in tracker order, or `None`).

Order is preserved exactly as the backend produced it. Associating hands with persistent
identities is a later phase's problem.

### `TrackerConfig`

| Field | Default | Meaning |
|---|---|---|
| `max_hands` | `2` | Maximum hands to detect (1–10) |
| `min_detection_confidence` | `0.5` | Threshold for declaring a hand present |
| `min_presence_confidence` | `0.5` | Threshold for a hand to remain in view |
| `min_tracking_confidence` | `0.5` | Threshold for tracking an existing hand |
| `model_path` | `None` | Explicit model path; otherwise discovered |
| `running_mode` | `VIDEO` | Sequential or time-series processing |

---

## Model asset

Resolved locally, in order, first match wins:

1. `TrackerConfig.model_path`
2. `GESTUREPILOT_HAND_LANDMARKER_MODEL`
3. `models/hand_landmarker.task` at the project root
4. `models/hand_landmarker.task` under the current working directory

**Nothing is downloaded at runtime.** A missing asset raises `ModelAssetError` listing
every path inspected and where to obtain the file. See `models/README.md`.

---

## Errors

| Exception | Raised when |
|---|---|
| `ModelAssetError` | Model file missing or unusable (`.searched` lists the paths tried) |
| `TrackerInitializationError` | Model present but failed to load; original chained |
| `TrackingProcessingError` | Unusable input frame, or the backend raised |
| `TrackerStateError` | Operation invalid for the current state |
| `TrackingError` | Base class; catch this to handle any tracking failure |

Input frames are validated before inference: must be a `Frame`, a non-empty `uint8` 3-D
NumPy array, with a finite timestamp. The backend's own exception types never escape;
they are wrapped with the offending frame's sequence number and the original chained.

---

## Testing

**Unit tests** (`tests/unit/tracking/`) inject a `FakeLandmarker` through
`landmarker_factory`, so the full conversion path runs with no model file, no GPU, and
no webcam. Only the final inference step is faked — everything the project owns is still
exercised, including BGR→RGB conversion and timestamp handling.

`tests/unit/conftest.py` installs two autouse guards: constructing a real
`cv2.VideoCapture` and loading the real `HandLandmarker` both raise. A test that forgets
to inject a fake fails loudly instead of quietly touching the developer's hardware or
model.

```powershell
uv run pytest
```

**Integration tests** (`tests/integration/tracking/`) run the genuine model. Opt in:

```powershell
uv run pytest -m integration tests/integration/tracking -v
```

They skip with an actionable message when the asset is absent, never download anything,
and verify the real API matches the layer's assumptions: the model loads, a frame
round-trips, a hand-free frame yields zero hands, and VIDEO mode accepts a strictly
increasing timestamp sequence.

---

## Design decisions

**VIDEO running mode by default.** Webcam frames arrive as a sequential time series, so
the tracker gets that temporal context: the model carries tracking state between frames,
yielding more stable landmarks than re-detecting each frame in isolation.
`RunningMode.IMAGE` is available for deterministic stateless inspection. `LIVE_STREAM` is
deliberately absent — it requires asynchronous callbacks.

**Timestamps come from the camera, never from wall-clock time.** VIDEO mode needs a
strictly increasing integer millisecond timestamp, and the camera already provides a
monotonic per-frame stamp, so it is converted rather than re-measured. Two frames
landing in the same millisecond are nudged forward by 1 ms, because MediaPipe rejects a
repeated or decreasing stamp.

**BGR→RGB happens here.** The camera emits OpenCV BGR arrays and must stay unaware of
MediaPipe. `cv.cvtColor` allocates a new array, so the caller's frame is never mutated —
asserted by a test.

**The landmarker factory is injected.** This is what makes the layer testable without a
model. Without it, every unit test would need a 7 MB asset and would be far slower.

**A Protocol, not an ABC.** `HandTracker` is structurally typed, so an alternative
backend satisfies it without inheriting anything.

**`UNKNOWN` is a first-class state.** Forcing handedness into two values would push a
guess downstream, where it would be harder to detect.

### Trade-offs

- **Synchronous, blocking.** `process()` blocks on inference. A slow frame stalls the
  loop. A later phase will want capture and processing on separate threads; the API
  does not preclude it, but ordering and backpressure policy is undecided.
- **Handedness assumes mirrored input.** MediaPipe generally expects a mirrored
  (selfie-view) image to label handedness as the user perceives it. The camera layer
  delivers unmirrored frames, so `Handedness` may be inverted relative to the user's
  intent. The tracker does **not** silently mirror the frame — that would change what
  downstream pixel-mapping code sees. Resolve this explicitly before relying on
  handedness.
- **No filtering of low-confidence detections.** Thresholds are set at init and applied
  by the model, but there is no per-frame confidence gate. That belongs to a later
  phase, which can now populate `detection_score` if a backend provides one.
- **A camera-layer fix was needed to use both layers together.**
  `with open_camera(...) as camera:` raised `CameraStateError`, because `open_camera()`
  returns an already-open device and `WebcamCamera.__enter__` then called `open()` a
  second time. `__enter__` now opens only when the camera is closed, and two tests pin
  the behaviour. Found while writing the smoke tool, not while building tracking.
- **Landmark indices are model-defined.** Index 0 is the wrist, and the remaining 20
  follow the model's fixed order. Hard-coding finger semantics into this layer would
  couple it to the model file.

---

## Privacy

Frames are converted and passed to the local inference engine. Nothing is written to
disk, logged, or transmitted. `models/README.md` documents asset handling.

---

## Not in this phase

Gesture classification, temporal smoothing, confidence filtering, the safety state
machine, OS actions, the system tray, and any GUI. The `main` entry point still only
prints a banner.
