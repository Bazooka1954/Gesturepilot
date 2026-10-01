# Architecture Notes

Design records for GesturePilot. Each decision that constrains the system should be
captured here as a short document with context, the decision, and its consequences.

## Planned documents

- `architecture.md` — pipeline stage contracts and data shapes
- `safety-model.md` — hold-to-confirm, cooldown, arming semantics
- `windows-actions.md` — how OS actions are isolated behind the dispatcher

## Written

- [`camera.md`](camera.md) — camera layer design (Phase 2)
- [`tracking.md`](tracking.md) — hand-tracking layer design (Phase 3)

## Current decisions

| Decision | Status | Rationale |
|---|---|---|
| Python 3.10 as compatibility baseline | Accepted | Constrains interpreter choice for MediaPipe/OpenCV |
| `uv` for env and deps | Accepted | Fast, reproducible locking |
| `src/` layout | Accepted | Prevents accidental imports from repo root |
| Native Windows runtime | Accepted | Camera and OS APIs cannot be containerised |
| Docker for CI only | Accepted | Headless tests on Linux |
| Streaming events over actions | Accepted | Recognition must not know about Windows APIs |
| `Camera` as a Protocol | Accepted | Structural typing; test doubles need no base class |
| Injected capture factory + clock | Accepted | Makes the camera layer testable without hardware |
| Requested vs measured FPS stored separately | Accepted | Never report a requested rate as real throughput |
| Explicit `close()`, no `__del__` | Accepted | `__del__` is unreliable at interpreter shutdown |
| OpenCV confined to `camera.py` | Accepted | Enforced by AST-based tests |
| MediaPipe confined to `tracking/tracker.py` | Accepted | Enforced by AST-based tests; backend stays swappable |
| `opencv-contrib-python` replaces `opencv-python` | Accepted | MediaPipe requires the contrib build; both ship the same `cv2` package |
| VIDEO running mode by default | Accepted | Webcam frames are a time series; tracking state stabilises landmarks |
| Timestamps taken from the camera | Accepted | Already monotonic; avoids re-measuring and clock-regression bugs |
| BGR→RGB conversion inside tracking | Accepted | Keeps the camera backend-agnostic; `cvtColor` never mutates the source |
| Injected landmarker factory | Accepted | Makes the tracking layer testable without a 7 MB model asset |
| Local model discovery, no runtime download | Accepted | No silent large downloads or network dependency at startup |
| Model binaries excluded from Git | Accepted | Keeps history small; avoids redistribution questions |
| `Handedness.UNKNOWN` as a real state | Accepted | Forcing a binary guess downstream hides detection problems |
| Normalised landmark coordinates | Accepted | Only the landmark-processing stage knows frame dimensions |
| No per-frame confidence gate yet | Accepted | Thresholds are init-time; a confidence gate belongs to a later phase |