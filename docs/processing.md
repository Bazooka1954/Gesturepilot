# Landmark Processing

Phase 4. Implements the **Landmark Processing** stage of the pipeline: one `TrackedHand`
in, a `HandFeatures` out.

This layer answers one question — *what is the geometry of this hand, independently of
where it is and how big it looks?* It recognises **no gestures**. A fist, a pinch, a point,
a swipe: none of those words appear in the package, and no threshold here encodes one.
Classification is a later phase, and so is confidence filtering — rejecting unreliable
classifications is its own stage, and folding that decision in here would hide it.

---

## Layout

```
src/gesturepilot/processing/
├── __init__.py       # public surface
├── types.py          # ProcessedLandmark, HandFeatures, FingerGeometry, configs
├── errors.py         # ProcessingError hierarchy
├── topology.py       # LandmarkIndex, Finger, FingerChain
├── geometry.py       # vector, angle, and hand-measurement helpers
├── normalization.py  # wrist-relative, scale-normalised coordinates
├── filters.py        # OneEuroFilter, LandmarkSmoother
└── processor.py      # LandmarkProcessor — the public entry point
```

`topology.py` is pure data and imports nothing from the rest of the project. The rest
imports only the standard library plus `gesturepilot.tracking.types`.

---

## API

```python
from gesturepilot.processing import LandmarkProcessor, ProcessorConfig, Finger

# Smoothing off (the default): process() is a pure function and needs no timestamp.
processor = LandmarkProcessor()

features = processor.process(hand)

features.hand_scale  # raw apparent size, tracker image units
features.finger(Finger.INDEX).middle_angle  # radians, or None if the joint collapsed
features.finger(Finger.INDEX).tip_to_palm  # normalised hand units
features.orientation.roll  # across-palm axis angle
features.orientation.tilt  # along-palm direction
features.handedness  # pass-through metadata
```

With smoothing:

```python
processor = LandmarkProcessor(ProcessorConfig(smoothing=True))

features = processor.process(hand, timestamp=result.timestamp)
processor.reset()  # when tracking restarts, or the hands in view change
```

### `process(hand, *, timestamp=None, frame_sequence=None) -> HandFeatures`

`hand` is read, never written — every output object is newly allocated, so the caller's
`TrackedHand` and the `Landmark` objects inside it are byte-for-byte what they were.
Smoothing filters produce new coordinates rather than writing through.

`timestamp` and `frame_sequence` are optional pass-through metadata, recorded on the
output and never re-derived. With smoothing off, `timestamp` is ignored entirely, which
is what makes the default configuration a pure function: the same hand and config always
give the same features, with no state to reset.

Every derived value is finite or explicitly `None`. `process()` never returns a
partly-filled result to signal that something went wrong.

---

## Coordinate spaces

Three spaces, each with one job.

| Space | Type | Units | What it is for |
|---|---|---|---|
| Tracker image space | `ProcessedLandmark` | frame-normalised, `z` wrist-relative | smoothing a raw measurement |
| Normalised hand space | `NormalizedLandmark` | hand-scale units | every derived measurement |
| Scalar | `float` | radians, or hand-scale units | angles, lengths, distances |

**Normalised hand space** is the important one. Each landmark is expressed relative to the
wrist and divided by the hand scale, so one unit means "one palm length". Where the hand
sits in the frame and how large it appears both cancel out: the same numbers describe a
hand filling the frame and a hand held at arm's length.

### Normalisation

Two operations, in this order:

1. **Translate.** Subtract the wrist. It is the right origin because it is the one
   landmark that is always present, never hidden by the hand itself, and sits at the base
   of everything being measured.
2. **Scale.** Divide by the hand scale.

### Hand scale

The wrist-to-middle-knuckle distance in the image plane.

That specific pair, rather than a knuckle-to-knuckle width, because it is the longest bone
in the palm and — crucially — because it barely moves as the fingers open and close. A
reference built from fingertips would drift with the very motion a classifier is trying to
measure.

A zero scale is refused with `DegenerateHandError`. There is no sensible fallback: every
normalised coordinate would be a division by zero, and quietly substituting a constant
would hide a broken detection.

### Palm centre

The centroid of the wrist and the four finger knuckles. Averaging the wrist *with* the
knuckles puts the point in the middle of the palm rather than at its base, and it moves
only with the palm — fingertips swinging through it barely register. That makes it a far
steadier reference than any single landmark.

---

## What is measured

`HandFeatures` is descriptive. For each of the five fingers, in `FINGERS` order:

| Field | Meaning |
|---|---|
| `proximal_angle` | Angle at the base joint, between the segment arriving from the wrist and the one leaving toward the middle joint |
| `middle_angle` | Angle at the middle joint |
| `distal_angle` | Angle at the joint nearest the fingertip |
| `segment_lengths` | The four bone lengths: root→base, base→middle, middle→distal, distal→tip |
| `tip_to_palm` | Fingertip to palm centre |
| `tip_to_index_mcp` | Fingertip to the index knuckle |

Plus, for the whole hand:

| Field | Meaning |
|---|---|
| `landmarks` | All 21 landmarks in tracker image space after processing |
| `normalization` | The normalised view, carrying the origin and the scale applied |
| `orientation.roll` | Across-palm axis angle, in `[-pi/2, pi/2]` |
| `orientation.tilt` | Along-palm direction (wrist→middle knuckle), in `[-pi, pi]` |
| `hand_scale` | The raw wrist-to-knuckle span that was divided out |
| `handedness`, `handedness_score`, `detection_score` | Pass-through source metadata |
| `timestamp`, `frame_sequence` | Pass-through frame metadata |

Angles are measured **in the image plane**. `x` and `y` come from the image and are metric
relative to each other; `z` is a noisy depth estimate only roughly commensurate with them.
A reliable planar angle beats an unreliable spatial one. Depth is still carried through in
the normalised landmarks for any later stage that wants it.

### Collapsed joints are `None`, never `NaN`

A tightly curled finger puts two landmarks on the same point. The joint angle there has no
defined direction — that is a real pose, not an error. So the angle is reported as `None`,
and `FingerGeometry.is_fully_measured` answers "is this complete?" without the consumer
testing three fields. A `NaN` would poison every comparison downstream, including ones
that are not looking at that finger.

The same rule applies to `PalmOrientation`, whose components are `None` when their
reference direction collapses.

### `roll` is an axis, `tilt` is a ray

The across-palm axis has no inherent sign — deciding which end is the index side is
precisely the question handedness would answer. So `roll` is folded into `[-pi/2, pi/2]`:
`0.0` means the knuckle line runs across the frame, and the sign says which way the palm
leans. Mirroring a hand then negates `roll` exactly, instead of shifting it by `pi`.

`tilt` is different: the wrist-to-knuckle direction *is* oriented, so it keeps the full
`[-pi, pi]` range. It is measured from "up the frame" (image `-y`) toward `+x`, so a hand
pointing at the ceiling reads `0.0` rather than an arbitrary `atan2` branch cut.

---

## Handedness is metadata, not geometry

`features.handedness` is the tracker's label, passed through unchanged. **Nothing in this
package is computed from it.**

The tempting alternative is to use handedness to decide which side is the thumb side, and
mirror the hand so that all downstream code can assume one orientation. It is also wrong
here, for a concrete reason: [`tracking.md`](tracking.md) records that MediaPipe generally
expects a mirrored input to label handedness as the user perceives it, and this camera
delivers unmirrored frames. So `Handedness` may already be inverted relative to the user's
intent. Branching on it would bake that unresolved ambiguity into every measurement.

Instead, the lateral reference is the **index knuckle**, chosen by geometry rather than by
label: `tip_to_index_mcp` means exactly the same thing for a left hand and a right hand,
without anyone deciding whether the camera is mirrored. The consequence is that a mirrored
hand measures as mirrored — `flip.roll == -base.roll`, and all lengths and interior angles
match — and a test pins that, so the property cannot rot.

Filter *slots* are keyed by handedness, because the tracker offers nothing else to key by.
That is a place to keep state, not a claim about which hand it belongs to; see the
lifecycle section below.

---

## Smoothing

Off by default. `ProcessorConfig(smoothing=False)` keeps `process()` a pure function of its
arguments, with no state and no timestamp requirement.

When enabled, a **One Euro filter** runs over the raw landmarks before anything is measured
— a filter should smooth the measurement, not a quantity derived from it.

The choice is deliberate. A fixed low-pass filter has to pick between lag and noise at one
cutoff, and a cutoff quiet enough for a resting hand turns a deliberate movement into a
visible delay. The One Euro filter low-passes the signal's *velocity* and raises the cutoff
when it is high: heavy smoothing when the hand is still, light smoothing when it moves.
A gesture that registers 200 ms after the user's hand moved is a gesture that feels broken.
It is also a few dozen lines with no dependencies, and deterministic given a timestamp.

| Knob | Default | Effect |
|---|---|---|
| `min_cutoff` | `1.0` | Cutoff in Hz for a still hand. Lower is smoother but laggier. |
| `beta` | `0.0` | How fast the cutoff rises with speed. `0.0` is a fixed cutoff. |
| `derivative_cutoff` | `1.0` | Cutoff for the low-pass on the estimated derivative. |

These are the algorithm's published knobs, not a tuned GesturePilot profile.

**No hidden clock.** Every call carries its own timestamp; nothing here reads a wall clock.
That is what lets the tests drive the filter with literal numbers and assert exact values,
and it matches how the camera layer takes timestamps from the frame rather than
re-measuring them.

**The first frame passes through unchanged.** With no history there is nothing to smooth
against, and inventing a ramp from an assumed zero would be a lie about where the hand was.

**Timestamps must strictly increase.** Otherwise the caller has handed over frames out of
order or repeated one, and the filter raises `FilterStateError` rather than producing a
plausible-looking wrong number. Silence here would surface much later as a gesture that
inexplicably misfires.

### Filter state lifecycle

One `LandmarkSmoother` per *slot*, where a slot is a `Handedness` label. The tracker
provides no persistent hand identity and inventing one would be a guess, so this is worth
stating plainly: **smoothing state is only as good as the handedness label.** If two hands
swap labels between frames, one smoother sees a jump it reads as fast motion.

Call `processor.reset()` when tracking restarts, when the hands in view change, or when a
label changes unexpectedly. It is cheap enough to call whenever in doubt.
`Handedness.UNKNOWN` is a slot like any other, so two unlabelled hands share one smoother.

---

## Errors

| Exception | Raised when |
|---|---|
| `InvalidLandmarksError` | Input is not a usable `TrackedHand`, or carries a non-finite coordinate, timestamp, or sequence |
| `DegenerateHandError` | The hand has no measurable size, so nothing can be normalised against it |
| `FilterStateError` | Smoothing is on and the timestamps do not increase |
| `ProcessingError` | Base class; catch this to handle any processing failure. Also raised when smoothing is enabled with no `timestamp`. |

`DegenerateHandError` and `InvalidLandmarksError` are also `ValueError`s in spirit but not
in fact: a caller that wants to handle *either* bad input catches `ProcessingError`.

---

## Testing

**Unit tests** (`tests/unit/processing/`) need no webcam, no model, no GPU, and no sleeping.
Every hand is synthetic and every timestamp is a literal, so the assertions are exact rather
than approximate.

`fixtures.py` places all 21 landmarks of a reference hand by hand, so its key quantities
are known exactly, and exposes the transformations explicitly rather than hiding them
inside `make_hand()` — a test that means to check mirroring cannot quietly get a translated
hand instead. Coverage of the package is 100%.

```
tests/unit/processing/
├── fixtures.py           # synthetic hands and their transformations
├── test_topology.py      # the index layout, pinned in full
├── test_geometry.py      # vectors, angles, degeneracy, the acos clamp
├── test_normalization.py # translation and scale invariance
├── test_filters.py       # the One Euro filter, driven by literal timestamps
├── test_processor.py     # determinism, metadata, invariants, smoothing lifecycle
└── test_boundaries.py    # architectural guarantees, by AST walk
```

`test_boundaries.py` enforces the three promises that are easy to lose slowly: no backend
or NumPy import anywhere in executable code, no wall-clock or file access, and no internal
import beyond `gesturepilot.tracking.types`. Docstrings may name MediaPipe freely — they
legitimately explain why it is excluded — so only executable code is examined. One test
runs the same detector over `tracking/tracker.py`, which *does* import MediaPipe, so the
detector itself cannot silently stop working.

```powershell
uv run pytest tests/unit/processing -v
```

---

## Design decisions

**Image-plane angles.** `x` and `y` are metric relative to each other; the backend's `z`
is a relative depth estimate, far noisier and only roughly commensurate with them. A
meaningful 3-D joint angle is not available from this data, and pretending otherwise would
produce numbers that look more sophisticated and are less reliable. Depth is preserved in
the normalised landmarks rather than discarded.

**The thumb is rooted at the wrist.** Every finger is modelled as the same five-point
polyline `root → base → middle → distal → tip`. For four fingers the root is the wrist; for
the thumb the first bone genuinely runs wrist→CMC, so rooting it there makes the shape
uniform and no finger needs a special case anywhere downstream.

**Topology knowledge lives here, not in tracking.** `LandmarkIndex` is the one place in the
package where model-specific numbers appear, and the only module that imports nothing.
Tracking hands landmarks over in the model's own order and refuses to attach finger
semantics to them, because that would couple it to the model file. Processing needs to know
which index is a knuckle, and needs to know it in exactly one place.

**Pure float arithmetic, no NumPy.** Every derived measurement is plain `math`, so a test can
pin it to a literal. A vectorised library would be faster and would quietly replace exact
values with library-specific tolerances, and the amount of linear algebra here is small
enough that a hand-rolled framework would be more surface area than it saves. The forbidden
import list includes `numpy` for exactly this reason.

**Immutable everything.** All output types are frozen dataclasses validated on
construction: finiteness, landmark counts, angle ranges, score ranges. These are the checks
that stop a `NaN` or a half-built feature set reaching the classifier.

**Defence in depth on input validation.** `TrackedHand` already rejects non-finite
coordinates, and the processor checks again anyway. It is the last point before the
arithmetic, and a duck-typed or substituted object must not be able to slip a `NaN` into
every feature downstream. Tested by building a hand with `object.__setattr__`, which is
what a faulty backend could produce.

### Trade-offs

- **The tracking import is transitive.** `gesturepilot.processing` imports
  `gesturepilot.tracking.types`, which executes the tracking package's `__init__`, which
  imports the tracker — and so MediaPipe. Reusing the tracking layer's own vocabulary is
  what keeps the two layers agreeing on what a hand is; duplicating `TrackedHand` to avoid
  the import would trade a load-time cost for two definitions of "21 landmarks" that could
  drift. Worth revisiting only if a backend-free install ever becomes a requirement, at
  which point the fix is to make the tracking `__init__` lazy, not to fork the types.
- **Camera-distance invariance is approximate.** Normalised image coordinates conflate
  apparent size with distance, so dividing by the hand scale removes the apparent size and
  with it most of the distance dependence. It cannot recover foreshortening: a hand turned
  edge-on to the camera projects smaller, and no single scalar undoes that. A gesture
  needing fine depth discrimination should not lean on this output.
- **No world landmarks.** `TrackedHand` carries image-space landmarks only, so there is no
  metric 3-D reconstruction available. `NormalizedHand.origin` and `.scale` are retained so
  a consumer *could* convert back to the frame, but the processor never does, because
  nothing downstream of it knows the frame dimensions.
- **Filter slots are keyed on a label the tracker cannot keep.** See the lifecycle section.
  The alternative — smoothing all hands through one filter — would be worse, and the
  alternative of inventing hand identity belongs to a later phase.
- **No confidence gate.** The processor reports `handedness_score` and `detection_score`
  and applies no threshold to either. Filtering is its own pipeline stage; a gate here
  would hide it.

---

## Privacy

This module handles numbers only. No file, network, or logging access — enforced by test,
along with the ban on reading a clock.

---

## Not in this phase

Gesture classification, confidence filtering, the safety state machine, gesture events, OS
actions, persistent hand identity across frames, and any GUI. `main` still only prints a
banner.
