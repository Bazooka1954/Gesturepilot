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

| Stage | Responsibility |
|---|---|
| Camera | Capture frames from a device |
| Hand Tracker | Produce hand landmarks (MediaPipe, pending validation) |
| Landmark Processing | Normalise, smooth, derive geometric features |
| Gesture Classifier | Map features to candidate gestures |
| Confidence Filter | Reject low-confidence classifications |
| Safety State Machine | Hold-to-confirm, cooldown, release-to-reset, arming |
| Gesture Event | Emit a confirmed, validated gesture |
| Action Dispatcher | Route an event to the correct backend |
| OS-specific Actions | Perform the action on Windows |

Each stage will be a separate, independently testable module with no knowledge of the
stages before it. Windows APIs are isolated behind the Action Dispatcher so that
platform-specific code never leaks into recognition logic.

---

## Current Status

**Phase 1 — project foundation only.**

Deliberately **not** implemented yet:

- MediaPipe tracking, camera capture
- Gesture recognition
- Windows actions
- System tray, GUI
- Docker runtime, configuration profiles, calibration, dynamic gestures

What exists is a Python 3.10 package skeleton with a minimal entry point, a locked
`uv` environment, a test suite, and lint configuration. This validates that the
toolchain and layout are correct before any computer-vision dependency is introduced.

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
Environment OK. No pipeline stages implemented yet.
```

---

## Tests

```powershell
uv run pytest                                    # full suite
uv run pytest --cov=gesturepilot                 # with coverage
uv run pytest tests/unit/test_main.py -v         # single file, verbose
```

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
├── tests/unit/           # fast, hermetic unit tests
├── docs/                 # architecture notes and ADRs
├── scripts/              # developer helper scripts
├── config/               # configuration templates
├── docker/               # development / CI container definitions
├── pyproject.toml        # project metadata and tool config
├── uv.lock               # reproducible dependency lock
└── .python-version       # pinned interpreter
```

---

## License

MIT