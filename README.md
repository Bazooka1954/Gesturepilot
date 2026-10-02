# GesturePilot

A professional Windows desktop application that uses a webcam and hand tracking to
recognize **deliberate** hand gestures and, eventually, control the operating system.

GesturePilot is a **clean v2 rebuild**. It takes conceptual inspiration from an older
web-based prototype, but none of its architecture.

---

## What GesturePilot Is

The original project captured a webcam feed, served it to a browser over MJPEG, and
opened websites when the user held up fingers. The webcam → Flask → browser → browser
actions approach is **retired**.

The goal of GesturePilot is different in kind. Gestures should drive **the operating
system** — not links in a browser tab. That means a native desktop application running
on Windows, where every input is a deliberate, confirmed action with clear safety
semantics.

Because the application will drive the OS, gesture recognition must be *conservative*.
A false positive is not a broken link; it is the wrong window closing. Safety is a
first-class stage of the pipeline, not an afterthought.

---

## Architecture

```
Camera
  -> Hand Tracker
  -> Landmark Processing
  -> Gesture Classifier
  -> Confidence Filter
  -> Safety State Machine
  -> Gesture Event
  -> Action Dispatcher
  -> OS-specific Actions
```

Stage responsibilities:

| Stage | Responsibility | Status |
|---|---|---|
| Camera | Capture frames from a device | **Implemented (Phase 2)** |
| Hand Tracker | Produce hand landmarks (MediaPipe) | **Implemented (Phase 3)** |
| Landmark Processing | Normalise, smooth, derive geometric features | **Implemented (Phase 4)** |
| Gesture Classifier | Map features to candidate gestures | Not started |
| Confidence Filter | Reject low-confidence classifications | Not started |
| Safety State Machine | Hold-to-confirm, cooldown, release-to-reset, arming | Not started |
| Gesture Event | Emit a confirmed, validated gesture | Not started |
| Action Dispatcher | Route an event to the correct backend | Not started |
| OS-specific Actions | Perform the action on Windows | Not started |

Each stage will be a separate, independently testable module with no knowledge of the
stages before it. Windows APIs are isolated behind the Action Dispatcher so that
platform-specific code never leaks into recognition logic.

---

## Current Status

**Phase 4 — landmark processing complete.** Phases 2 (camera) and 3 (hand tracking)
remain done.

Implemented so far:

- **Camera capture** (Phase 2) via OpenCV, behind an abstract `Camera` interface with a
  safe `open → read → close` lifecycle, typed frame metadata, and measured FPS
- **Hand tracking** (Phase 3) via MediaPipe, converting a camera `Frame` into a
  backend-independent `TrackingResult` of 21-landmark hands
- **Landmark processing** (Phase 4) via `LandmarkProcessor`, converting a `TrackedHand`
  into a backend-independent `HandFeatures`: wrist-relative, hand-scale-normalised
  coordinates plus per-finger joint angles, bone lengths, tip distances, and palm
  orientation
- Frozen, validated domain types: `Landmark`, `TrackedHand`, `TrackingResult`,
  `TrackerConfig`, `HandFeatures`, `FingerGeometry`, `PalmOrientation`, `ProcessedLandmark`
- Explicit `initialize → process → close` lifecycle with context-manager support
- Optional deterministic One Euro smoothing with caller-supplied timestamps, off by
  default so the processor stays a pure function
- Local model discovery that never downloads anything at runtime
- 512 unit tests that need neither a webcam nor the model file, plus 11 opt-in
  real-hardware / real-model tests

Deliberately **not** implemented yet:

- Gesture recognition or classification
- Confidence filtering, safety state machine
- Windows actions, keyboard/mouse control
- System tray, GUI
- Docker runtime, configuration profiles, calibration, dynamic gestures

No gesture recognition or OS control exists yet. The entry point still only prints a
banner.

See [`docs/camera.md`](docs/camera.md), [`docs/tracking.md`](docs/tracking.md), and
[`docs/processing.md`](docs/processing.md) for the layer designs.

---

## Model Asset

Hand tracking needs one local model file, `hand_landmarker.task`. It is **not** committed
to Git — see [`models/README.md`](models/README.md) for how to download and place it.

Nothing is fetched at runtime. If the asset is missing, startup fails with a
`ModelAssetError` that names every path searched and where to get the file.

---

## Privacy

The camera, tracking, and processing layers are **local only**. Frames are handed to the
running application and the local inference engine, and nowhere else:

- no frame is written to disk
- no frame is uploaded or transmitted over a network
- no frame contents are logged
- no screenshots are created

There is no network code in any of these layers. The webcam is opened, read, and released;
the model is loaded from disk and released on `close()`. The processing layer handles
numbers only — it reads no clock and touches no file, both enforced by test.

---

## Why Python 3.10

Python 3.10 is the project's compatibility baseline. GesturePilot depends on
computer-vision and native/compiled packages such as MediaPipe and OpenCV, which
constrain the set of usable interpreters. Pinning `>=3.10,<3.11` keeps the baseline
explicit and the environment reproducible, and avoids depending on whichever Python
a contributor happens to have installed.

Newer interpreters (3.12, 3.13, 3.14) are deliberately excluded. Raising the baseline
should be a deliberate follow-up decision once the computer-vision dependency set is
validated, not an automatic one.

---

## Prerequisites

- **Windows 10 or 11** (the eventual runtime target)
- **[uv](https://docs.astral.sh/uv/)** — environment and dependency management
- Git (optional)

`uv` provisions the Python interpreter itself, so installing Python separately is not
required. If your system has only a newer Python installed:

```powershell
uv python install 3.10
```

---

## Development Setup

```powershell
# Install uv (if not already available)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create the environment and install dependencies
uv sync

# Activate (optional)
.venv\Scripts\Activate.ps1
```

`uv sync` creates `.venv` using the version pinned in `.python-version` and installs
both runtime and development dependency groups.

---

## Running

```powershell
# Minimal entry point — confirms the package runs
uv run gesturepilot

# Or as a module
uv run python -m gesturepilot.main
```

Expected output:

```
GesturePilot 0.1.0
Environment OK. Camera and hand-tracking layers available;
gesture recognition and OS control not implemented yet.
```

---

## Tests

```powershell
uv run pytest                                    # full suite (no webcam, no model)
uv run pytest --cov=gesturepilot                 # with coverage
uv run pytest tests/unit/camera -v               # camera unit tests only
uv run pytest tests/unit/tracking -v             # tracking unit tests only
uv run pytest tests/unit/processing -v           # processing unit tests only
```

The normal suite **never opens a webcam and never loads the model**. Unit tests inject
fakes, and two autouse fixtures turn any attempt to construct a real `cv2.VideoCapture`
or load the real `HandLandmarker` into an immediate failure. The suite therefore passes
on machines with no camera, no model asset, and no GPU.

### Real-hardware / real-model tests (manual, opt-in)

Both integration directories are marked `integration` and deselected by default:

```powershell
uv run pytest -m integration tests/integration/camera -v      # needs a webcam
uv run pytest -m integration tests/integration/tracking -v   # needs models/hand_landmarker.task
uv run pytest -m integration -v                              # everything
```

They skip rather than fail when hardware or the model is missing. Windows shows a
camera-in-use indicator while the camera tests run.

For a quick manual check that prints what the device actually negotiated:

```powershell
uv run python scripts/camera_smoke_test.py
uv run python scripts/camera_smoke_test.py --device 1 --width 1280 --height 720 --frames 30
```

And for a live check of the tracking layer over real webcam frames:

```powershell
uv run python scripts/tracking_smoke_test.py
uv run python scripts/tracking_smoke_test.py --device 1 --max-hands 2 --frames 120
```

Both scripts are development tools, not part of the application. See
[`scripts/README.md`](scripts/README.md).

---

## Linting

```powershell
uv run ruff check .        # lint
uv run ruff format --check .   # formatting check
uv run ruff format .      # apply formatting
```

---

## Docker

Docker is intended for **development, testing, and CI only** — linting, type checks,
and headless test runs on Linux.

Docker is **not** the runtime for the application. Camera capture, Windows APIs, and
the eventual system tray component all require a native Windows host and cannot be
containerised. The Docker setup lives in `docker/` and is not wired up in this phase.

---

## Layout

```
gesturepilot/
├── src/gesturepilot/    # package source (src layout)
│   ├── camera/          # camera layer (OpenCV isolated to camera.py)
│   ├── tracking/        # hand-tracking layer (MediaPipe isolated to tracker.py)
│   └── processing/      # landmark-processing layer (no backend imports at all)
├── tests/unit/          # fast, hermetic unit tests (no webcam, no model)
├── tests/integration/   # opt-in tests that may need real hardware or the model
├── docs/                # architecture notes and ADRs
├── models/              # local model assets (binaries not committed)
├── scripts/             # developer helper scripts
├── config/              # configuration templates
├── docker/              # development / CI container definitions
├── pyproject.toml       # project metadata and tool config
├── uv.lock              # reproducible dependency lock
└── .python-version      # pinned interpreter
```

---

## License

MIT