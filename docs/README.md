# Architecture Notes

Design records for GesturePilot. Each decision that constrains the system should be
captured here as a short document with context, the decision, and its consequences.

## Planned documents

- `architecture.md` — pipeline stage contracts and data shapes
- `safety-model.md` — hold-to-confirm, cooldown, arming semantics
- `windows-actions.md` — how OS actions are isolated behind the dispatcher

## Written

- [`camera.md`](camera.md) — camera layer design (Phase 2)

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