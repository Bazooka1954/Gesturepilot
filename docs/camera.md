# Camera Layer

Phase 2. Implements the **Camera** stage of the pipeline: device selection, safe
lifecycle, and frame capture with metadata.

Nothing here knows what a hand or a gesture is.

---

## Layout

```
src/gesturepilot/camera/
├── __init__.py   # public surface
├── types.py      # CameraConfig, CameraProperties, Frame, FpsMeter  (no OpenCV)
├── camera.py     # Camera protocol + WebcamCamera                   (the only cv2 import)
└── errors.py     # CameraError hierarchy                            (no OpenCV)
```

`camera.py` is the **only** module that imports `cv2`. This is enforced by tests
(`tests/unit/camera/test_abstraction.py`), which parse the AST of every module in the
package and assert that `cv2` is imported from exactly one file.

`types.py` and `errors.py` are backend-agnostic and can be imported by anything,
including a future non-OpenCV backend.

---

## API

```python
from gesturepilot.camera import CameraConfig, WebcamCamera, open_camera

config = CameraConfig(device_index=0, width=1280, height=720, fps=30.0)

# Explicit lifecycle
camera = WebcamCamera(config)
camera.open()
frame = camera.read()
camera.close()

# Or as a context manager (releases even if the body raises)
with open_camera(config) as camera:
    frame = camera.read()
```

### Lifecycle

```
CLOSED --open()--> OPEN --close()--> CLOSED
```

- `open()` selects the device, initialises it, verifies it actually opened, and reads
  back what the device negotiated. Raises `CameraOpenError` on failure and
  `CameraStateError` if already open.
- `close()` releases the device. Safe to call repeatedly, safe to call when never
  opened, and resets counters and reported properties.
- Reopening after closing is supported.

The device is released explicitly. Nothing relies on process termination to free the
camera, and a half-opened handle is released rather than leaked.

### `CameraConfig`

Frozen dataclass, validated in `__post_init__`.

| Field | Default | Meaning |
|---|---|---|
| `device_index` | `0` | Capture device index |
| `width` | `1280` | **Requested** width |
| `height` | `720` | **Requested** height |
| `fps` | `30.0` | **Requested** FPS, or `None` to leave the device default |

These are *requests*. Webcams routinely substitute a supported mode. Read
`camera.properties` to see what the device actually gave you.

### `CameraProperties`

Holds requested settings next to device-reported values, so a silently-substituted
resolution is obvious. Driver sentinels (`0.0`, negatives, `NaN`) are normalised to
`None` rather than being passed off as real numbers.

`matched_request` is `True` only when the reported resolution equals the request.

### `Frame`

| Attribute | Meaning |
|---|---|
| `image` | NumPy `ndarray`, BGR, OpenCV-native. Treat as read-only. |
| `timestamp` | **Monotonic** seconds since capture. Monotonic so interval math survives a system clock change. |
| `sequence` | Zero-based capture counter, for detecting gaps or reordering downstream. |
| `width` / `height` / `channels` | Derived from the image. |

No frame is ever persisted, logged, or transmitted.

### `FpsMeter` — requested vs actual

Two different numbers, kept deliberately separate:

- `config.fps` — what we *asked* the driver for.
- `camera.measured_fps` — what we actually measured, over a trailing 1-second window.

`measured_fps` returns `None` until at least two frames land inside the window. A
camera asked for 60 FPS that delivers 10 reports `10.0`. The clock is injectable, so
the measurement is unit tested deterministically.

---

## Errors

| Exception | Raised when |
|---|---|
| `CameraOpenError` | Device could not be initialised — missing, busy, bad index |
| `FrameReadError` | Capture failed, or returned an unusable image |
| `CameraStateError` | Operation invalid for current state — e.g. reading while closed |
| `CameraError` | Base class; catch this to handle any camera failure |

A failed read never yields a partial or empty frame: `read()` validates that the
capture succeeded, that the image is not `None`, non-empty, and has 2 or 3 dimensions,
and raises `FrameReadError` otherwise. Failed reads do not increment the frame counter.

Exceptions raised by an injected capture factory are **not** caught or wrapped — a
programming error stays visible instead of being disguised as a device error.

---

## Testing

Unit tests inject a `FakeCapture` through the `capture_factory` argument and a
`FakeClock` through `clock`, so no hardware is touched and no test sleeps.

`tests/unit/conftest.py` installs an autouse fixture that replaces `cv2.VideoCapture`
with a function that raises. A unit test that forgets to inject a fake fails loudly
instead of quietly opening the developer's webcam.

Real-camera tests live in `tests/integration/camera/`, are marked `integration`, and are
deselected by the default `addopts`. They skip rather than fail when no camera is
available.

---

## Design decisions

**A Protocol, not an ABC.** `Camera` is a `typing.Protocol`, structurally typed. A
mock camera or file-backed implementation satisfies it without inheriting anything,
which keeps test doubles free of production base classes.

**Both the capture factory and the clock are injected.** This is the single decision
that makes the layer testable without hardware. Anything else would force tests to
touch a real device.

**Requested and actual are stored side by side.** Never report a requested FPS as if
it were the real one. Storing both makes the gap unmissable at the call site.

**`sequence` on every frame.** Costs nothing now, and is what a future
producer/consumer pipeline needs to detect dropped or reordered frames.

**No `__del__`.** Cleanup is explicit (`close()` or a `with` block). `__del__` cleanup
is unreliable during interpreter shutdown, and a camera that looks released but is not
is a worse failure mode than one that raises `CameraStateError`.

**`close()` is idempotent and cheap.** Safe in `finally` blocks and teardown paths,
which is where it will mostly be called.

### Trade-offs

- **Blocking read.** `read()` blocks until the device delivers. Correct for now, but
  the final application will want a producer thread feeding gesture processing so a slow
  consumer cannot starve capture. The API does not preclude this — the camera stays
  owned by one reader.
- **Single hand / single device.** Out of scope until the tracker needs it.
- **No reconnection.** A camera unplugged mid-session raises `FrameReadError`; recovery
  policy belongs to the layer that owns the session lifecycle.
- **Requested FPS is a request.** Some drivers ignore `CAP_PROP_FPS` entirely. That is
  why `measured_fps` exists.

---

## Not in this phase

MediaPipe and hand tracking, landmarks, gesture classification, confidence filtering,
the safety state machine, OS actions, the system tray, and any GUI. The `main` entry
point still only prints a banner.
