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
- [`processing.md`](processing.md) — landmark-processing layer design (Phase 4)
- [`classifier.md`](classifier.md) — static gesture-classification design (Phase 5)
- [`confidence-filter.md`](confidence-filter.md) — temporal stability filtering design (Phase 6)

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
| Wrist-relative, hand-scale normalisation | Accepted | Removes frame position and apparent size from every derived measurement |
| Hand scale measured wrist → middle knuckle | Accepted | Longest stable palm bone; a fingertip-based reference drifts with the measured motion |
| Palm centre = wrist + four knuckles, centroided | Accepted | Sits mid-palm and moves only with the palm; far steadier than any single landmark |
| Image-plane angles only | Accepted | `z` is a noisy depth estimate, not commensurate with `x`/`y`; a planar angle is reliable, a 3-D one is not |
| Collapsed joint angles reported as `None` | Accepted | A folded finger is a real pose, not an error; a `NaN` would poison unrelated comparisons |
| `roll` folded into `[-pi/2, pi/2]` | Accepted | The across-palm axis is undirected; folding makes mirroring negate it exactly |
| Handedness never used for geometry | Accepted | The label may be inverted (see `tracking.md`); the index knuckle is a geometry-chosen reference |
| Optional One Euro smoothing, off by default | Accepted | Keeps `process()` a pure function; adaptive cutoff avoids the lag/noise trade-off |
| Caller-supplied timestamps, no hidden clock | Accepted | Exact, reproducible assertions; matches the camera layer's timestamps |
| Pure float arithmetic, no NumPy | Accepted | Derived measurements stay exactly assertable; forbidden by AST test |
| Static classification, one frame at a time | Accepted | No memory means every verdict is reproducible from a single input |
| Extension ratio per digit, not joint angles | Accepted | Collapsed angles are `None` exactly on the poses a fist is made of; the ratio is always defined |
| Confidence from evidence × margin | Accepted | A confident `UNKNOWN` is a good outcome; `score_for()` answers "how much like a point?" |
| `UNKNOWN` requires both a score and a margin floor | Accepted | Two gestures fitting a pose equally is a coin flip, and naming it would hide that |
| `min` over required conditions, not a product | Accepted | Names the weakest failing finger instead of collapsing it into one small number |
| Thumb excluded from `POINT`/`TWO_FINGERS` | Accepted | A real pointing hand holds its thumb wherever is comfortable |
| Handedness never read by the classifier | Accepted | Enforced by AST test; the label may be inverted (see `tracking.md`) |
| Thresholds defaulted, not calibrated | Accepted | No recorded hands to fit against; every knob is a `ClassifierConfig` field |
| Confidence filtering kept out of the classifier | Accepted | Rejection is its own stage; hiding the threshold inside recognition makes it untunable |
| Stability counted in observations, not seconds | Accepted | Frame rate is a property of the camera; a seconds-based filter would accept twice as fast on a fast machine |
| Timestamps optional, caller-supplied, never read from a clock | Accepted | Matches the camera and smoothing layers; a filter that owns a time is no longer a function of its input |
| Equal timestamps allowed, backwards ones rejected | Accepted | `OneEuroFilter` divides by the interval and cannot tolerate a zero gap; this filter only compares one against a threshold |
| The filter's whole state is three values | Accepted | Nothing derived is stored, so nothing derived can disagree with the run; `reset()` is one assignment |
| `UNKNOWN` and low confidence end a run rather than being ignored | Accepted | They are the classifier declining to name a gesture; accumulating across one would accumulate evidence it withheld |
| `UNKNOWN` and low confidence are not errors | Accepted | The filter answering is its job; only input it cannot evaluate should reach a `try` |
| Losing stability is immediate and silent | Accepted | It reports the present; what a safety stage does about a one-frame dip is that stage's decision, not a threshold side effect |
| `RESET` as the default candidate-change policy | Accepted | A gesture that flickers away and back has not been seen three times running; `HOLD` can pin a stale gesture |
| No run decay after acceptance | Accepted | Nothing reads the count past acceptance; a caller wanting a bounded run calls `reset()` |
| Aggregate confidence named and documented as a run summary | Accepted | `mean_confidence` is not the classifier's confidence and must not be read as one |
| Label-agnostic filter, no gesture named in code | Accepted | Enforced by AST test; a sixth gesture is a classifier change, not a filter change |
| No fake label for a missing frame | Accepted | `reset()` or a long gap; inventing an `UNKNOWN` would lie about what the classifier saw |
| Accepted stability is not authorisation | Accepted | Whether a stable gesture may act is the safety state machine's decision; this layer dispatches nothing |
