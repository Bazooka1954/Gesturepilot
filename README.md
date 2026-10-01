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
| Hand Tracker | Produce hand landmarks (MediaPipe, pending validation) | Not started |
| Landmark Processing | Normalise, smooth, derive geometric features | Not started |
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

**Phase 2 — camera layer complete.**

Implemented so far:

- **Camera capture** via OpenCV (`opencv-python` + `numpy`), behind an abstract
  `Camera` interface with a safe `open → read → close` lifecycle
- Typed frame metadata and measured (not requested) FPS
- Camera-specific exception hierarchy
- 90 unit tests that run without a webcam, plus 5 opt-in real-camera tests

Deliberately **not** implemented yet:

- **MediaPipe / hand tracking / landmarks** — OpenCV is installed for capture only
- Gesture recognition or classification
- Confidence filtering, safety state machine
- Windows actions, keyboard/mouse control
- System tray, GUI
- Docker runtime, configuration profiles, calibration, dynamic gestures

No gesture recognition or OS control exists yet. The entry point still only prints a
banner.

See [`docs/camera.md`](docs/camera.md) for the camera layer's design.

---

## Privacy

The camera layer is **local only**. Frames are handed to the running application and
nothing else:

- no frame is written to disk
- no frame is uploaded or transmitted over a network
- no frame contents are logged
- no screenshots are created

There is no network code anywhere in the camera layer. The webcam is opened, read, and
released.

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
Environment OK. Camera layer available; gesture pipeline not implemented yet.
```

---

## Tests

```powershell
uv run pytest                                    # full suite (no webcam needed)
uv run pytest --cov=gesturepilot                 # with coverage
uv run pytest tests/unit/camera -v               # camera unit tests only
```

The normal suite **never opens a webcam**. Unit tests inject a fake capture device, and
an autouse fixture fails any test that tries to construct a real `cv2.VideoCapture`, so
the suite passes on machines with no camera, on a busy camera, and in CI.

### Real-camera test (manual, opt-in)

The integration tests in `tests/integration/camera/` are marked `integration` and are
deselected by default. To run them on Windows:

```powershell
uv run pytest -m integration tests/integration/camera -v
```

They skip rather than fail when no camera is present or the device is busy. Note that
Windows will show a camera-in-use indicator while they run.

For a quick manual check that prints what the device actually negotiated:

```powershell
uv run python scripts/camera_smoke_test.py
uv run python scripts/camera_smoke_test.py --device 1 --width 1280 --height 720 --frames 30
```

That script is a development tool, not part of the application.

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
│   └── camera/          # camera layer (OpenCV isolated to camera.py)
├── tests/unit/          # fast, hermetic unit tests (no webcam)
├── tests/integration/   # opt-in tests that may need real hardware
├── docs/                # architecture notes and ADRs
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