# Gesture Classification

The classifier turns one frame of measured geometry into one named gesture. It is the
first stage that uses the word "gesture", and the last one that is still a pure function
of its input.

```
HandFeatures  ->  extension ratios + thumb/index gap  ->  five rule scores  ->  one label
```

Everything in between is arithmetic on ratios, so where the hand sits in frame and how
large it appears both cancel out, and the same hand always gets the same answer.

## Layout

| File | Contents |
|---|---|
| `types.py` | `Gesture`, `ClassifierConfig`, `GestureClassification`, `HandMeasurements` |
| `classifier.py` | `GestureClassifier`, the measurement derivation, the five rules |
| `errors.py` | `ClassificationError`, `InvalidFeaturesError` |

## API

### `classify(features) -> GestureClassification`

```python
from gesturepilot.classifier import Gesture, GestureClassifier

classifier = GestureClassifier()
result = classifier.classify(features)

if result.is_recognized:
    print(result.gesture, result.confidence, result.runner_up)
```

### `measure(features) -> HandMeasurements`

The evidence the rules would use, without a verdict. Exposed so a decision can be
inspected or reused without re-deriving it — useful when a gesture misbehaves and you
need to know which ratio was wrong.

---

## Evidence

Two measurements, both derived from `HandFeatures` and neither re-implementing any of
the processing layer's geometry.

### Extension ratio

```
extension = |base → tip| / (that digit's own three bones)
```

The numerator is the straight-line reach from a digit's first joint to its fingertip; the
denominator is the summed length of the bones between them. A straight digit scores `1.0`,
and the score falls as the digit folds back on itself. The wrist→knuckle segment is
excluded, because it belongs to the palm rather than the digit and including it would make
the ratio depend on how long each finger happens to be.

Rotating a whole finger at its knuckle leaves this ratio untouched, so the measure answers
"how folded is this finger" and nothing else. Landmarks come from `FINGER_CHAINS`, so no
index is written by hand, and the distances are processing's own `distance_2d`.

Measured on the test fixtures, one degree being half a bend at each interphalangeal joint:

| Curl (°) | Ratio | Reads as |
|---|---|---|
| 0 | 1.000 | extended |
| 30 | 0.978 | extended |
| 60 | 0.913 | extended |
| 90 | 0.811 | **ambiguous** |
| 120 | 0.681 | folded |
| 150 | 0.533 | folded |
| 180 | 0.389 | folded |

The default window is `0.70`–`0.92`, which puts the ambiguous zone at roughly 60°–115° of
bend. That is a wide band on purpose; see *Ramps, not comparisons* below.

### Thumb–index gap

The image-plane distance between the two fingertips in hand-scale units, so `0.0` is
touching and `1.0` is one palm length apart. Image plane only, for the reason the joint
angles are image-plane only: the tracker's `z` is too noisy for a distance that decides
whether two fingertips are touching.

### Joint angles are deliberately unused

`FingerGeometry` reports three joint angles per finger, and none of them appear in a rule.
The extension ratio already encodes folding, it is always defined, and using the angles
would mean handling `None` — which the processing layer reports for a tightly folded
finger, i.e. exactly the poses a fist is made of. Treating that `None` as missing data
would penalise the very gestures most likely to be recognised. So `None` angles are never
read, and they are not treated as malformed: a curled hand from the real processor
classifies normally.

---

## Rules

Each gesture is the **weakest** of its required conditions — a `min`, not a product — so a
score reads as "how close was the least convincing requirement", and one collapsed finger
cannot be averaged away by four perfect ones.

| Gesture | Conditions |
|---|---|
| `OPEN_PALM` | index, middle, ring, pinky extended, and the thumb extended |
| `FIST` | index, middle, ring, pinky folded, and the thumb folded |
| `POINT` | index extended; middle, ring, pinky folded |
| `TWO_FINGERS` | index and middle extended; ring and pinky folded |
| `PINCH` | thumb tip within reach of the index tip; middle, ring, pinky folded |

**The thumb is in `OPEN_PALM` and `FIST` and absent from the other three.** Those three
are about which *fingers* are up, and a real pointing hand holds its thumb wherever is
comfortable — tucked along the palm, or spread for balance. Requiring one position would
reject hands their owner would call a point. For `PINCH` the thumb enters through the gap
instead, which is the thing that actually separates a pinch from a point: in both, the
index is up and the other three are down, and the only real difference is where the thumb
tip ended up.

`POINT` and `TWO_FINGERS` differ by exactly one factor — the middle finger. That is what
makes the margin test meaningful rather than decorative: a middle finger halfway between
the two states puts both rules at almost the same score, and there is no honest way to
choose between them.

### Ramps, not comparisons

A digit's extension score reaches `1.0` at `extended_threshold`, falls to `0.0` at
`curled_threshold`, and is linear in between. Two reasons, both about honesty:

A half-curled finger contributes `0.5`, not a yes or a no, so the margin between
candidates degrades smoothly instead of falling off a cliff at a threshold. And the
reported confidence then tracks how far past the threshold the pose actually is, which is
what makes confidence worth reading at all. A boolean rule can only ever report `0.99` or
`0.0`, which is a confidence value carrying no information.

---

## Choosing a label

The highest score wins, but a win alone is not enough. The label is `UNKNOWN` unless the
score reaches `min_score` **and** the lead over the runner-up reaches `min_margin`.

```
separation = 0.5 + 0.5 × min(1, margin / strong_margin)
confidence  = winning_score × separation        when recognised
confidence  = 1 − winning_score                 when UNKNOWN
```

`min_score` answers "was the gesture actually present?" and `min_margin` answers "was it
uniquely present?". A pose that satisfies `POINT` and `TWO_FINGERS` equally is a hand in
the middle of a movement, and naming it either one would be a coin flip presented as a
fact.

**Confidence means strength of the reported label, not weakness.** For a recognised gesture
it blends how completely the conditions were met with how far the winner beat the field.
For `UNKNOWN` it means the opposite — confidence that *no* gesture applies, which is high
exactly when the evidence was weak or the candidates were level. A confident `UNKNOWN` is
a good outcome, not a weak one, and a caller that wants "how strongly does this look like a
point?" should read `result.score_for(Gesture.POINT)` instead.

`result.margin` is reported separately because it is the half of the confidence a caller
can act on: a high score with no margin means two gestures are equally plausible, which is
the case where acting is most dangerous and doing nothing is safest.

---

## Errors

Recognition does not fail in many ways, and the split is deliberately narrow.

| Error | Meaning |
|---|---|
| `ClassificationError` | Base class. Catch this to handle them all. |
| `InvalidFeaturesError` | Not a `HandFeatures`, or one holding `NaN`/infinite values, a negative or non-numeric bone length, a truncated landmark array, or finger measurements that do not match the topology. |

**A hand that could not be classified is not an error.** It is `Gesture.UNKNOWN`, a normal
result that happens often enough to deserve a first-class label. The exceptions are only
for input the rules could not even be evaluated against.

The domain types already reject non-finite values on construction, so `InvalidFeaturesError`
means the object was built by bypassing validation — `object.__setattr__`, an unpickled
instance, or a future processing layer. Checking anyway is the same last-line-of-defence
the processor makes before dividing by hand scale: one `NaN` would compare false against
every threshold and silently produce `UNKNOWN` for every hand, forever, with no error
anywhere.

A digit whose bones have no length at all is *not* an error. It cannot be straightened, so
it is reported as fully folded — which is both true and harmless, and avoids turning a
degenerate hand into a `ZeroDivisionError`.

---

## Testing

207 tests, no webcam, no model asset, no sleeping. `tests/unit/classifier/fixtures.py` poses
a hand by forward kinematics from a palm skeleton plus one curl number per finger, then runs
the coordinates through the **real** `LandmarkProcessor`. Building genuine processing output
matters: the classifier only ever sees that, so a hand-built `HandFeatures` would test
against a shape the pipeline never produces.

The behaviours that get the most attention:

- Each of the five configurations is recognised from a pose that plainly shows it, and
  nothing else scores above `0.0`.
- An ambiguous pose returns `UNKNOWN` — a half-curled middle finger cannot choose between
  `POINT` and `TWO_FINGERS`, and a half-curled index blocks both.
- A pose matching nothing returns `UNKNOWN` with high confidence and no runner-up.
- Confidence is not constant: a slightly bent hand scores below a perfect one.
- The verdict is unchanged by handedness, by mirroring, by frame position, and by apparent
  size.
- The same input always gives the same answer, timestamps are never consulted, and the
  features are not mutated.
- Every threshold is actually read: changing `pinch_close_threshold` can turn a `PINCH`
  into `UNKNOWN`, and a `curled_threshold` below every real ratio turns a fist into an open
  palm.
- Malformed input is rejected rather than divided by: non-finite values, negative lengths,
  truncated arrays, out-of-order fingers, and a non-`HandFeatures` object.

`test_boundaries.py` enforces the structural promises with AST walks: no backend,
automation, or NumPy imports; no clock, file, or `eval`; internal imports limited to
`gesturepilot.processing`; no rule reading `timestamp`, `frame_sequence`, or `handedness`;
no hand-written landmark indices; a small public API; and a classifier that stores its
config and nothing else.

---

## Design decisions

**Static, one frame at a time.** No memory of previous frames, so every decision is
reproducible from a single input and needs no `reset()`. What to do with a *sequence* —
smoothing, debouncing, hold-to-confirm — belongs to the confidence filter and the safety
state machine, and doing any of it here would mean a classification could no longer be
checked from one input.

**No confidence filtering here.** The classifier reports what it measured, including a
weak result. Deciding to reject it is a separate stage with its own threshold, and hiding
that decision inside recognition would make it invisible and untunable.

**Ratios, not distances.** Both measurements divide by something that scales with the hand,
so a hand at the edge of frame and the same hand close to the camera are identical inputs.
The pipeline's normalisation does the heavy lifting; the ratios only remove what is left.

**`min`, not a product.** A product of five near-perfect conditions and one collapsed one
gives a tiny number with no way to tell which finger failed. A `min` names the weakest
link directly, which is what a caller needs in order to trust or discount the result.

**Handedness is never read.** The tracking layer may hand back an inverted label
(`docs/tracking.md`), and the processing layer already chose a geometry-based reference —
the index knuckle — for exactly this reason. A test asserts the verdict is identical for
every `Handedness`, and an AST test asserts the attribute is never even read.

**Everything is a validated frozen dataclass.** Thresholds, measurements, and results are
all validated where they are written, and `GestureClassification` additionally checks its
own internal consistency: the gesture it names really is the highest-scoring one, and the
margin it reports really is the gap between the top two scores. A result that claimed
`POINT` while `TWO_FINGERS` scored higher would be worse than no result at all.

### Trade-offs

- **`PINCH` and `FIST` genuinely overlap.** When a closed hand's thumb crosses the index
  tip, both rules have their conditions met. This classifier resolves it by margin rather
  than by preferring one, so such a hand comes back as `PINCH` at a visibly lower
  confidence — or `UNKNOWN`. The fixtures include the pose so the behaviour is pinned
  rather than accidental. A real fix needs a thumb-direction cue this data does not carry
  reliably.
- **Foreshortening defeats the extension ratio.** A finger pointing at the camera projects
  short, which reads as folded. A hand held edge-on is the worst case, and no threshold
  change fixes it; it needs a pose prior or a backend that supplies world landmarks.
- **`PINCH` is a distance, not a contact.** Whether fingertips are actually touching is a
  depth question, and `z` is too noisy to answer it. The gap is a good proxy in the image
  plane and a poor one for a pinch that closes towards the camera.
- **The thumb's extension ratio is noisier than a finger's.** It spans the wrist, has the
  widest range of natural positions, and is the joint most often collapsed by
  self-occlusion. It is weighted into `OPEN_PALM` and `FIST` but never allowed to decide
  them alone.
- **The default thresholds are reasoned, not calibrated.** They were chosen from the
  geometry of a palm held toward a camera, and the fixture table above is how they were
  checked. They have not been fitted against recorded hands, and a real deployment should
  expect to tune `curled_threshold` and `extended_threshold` per user. Nothing about the
  API resists that — it is a constructor argument.
- **`classifier.py` imports `gesturepilot.processing`, which transitively imports
  MediaPipe.** This is the same trade-off the processing layer documented: its package
  `__init__` reaches `tracking.types`, which runs the tracking `__init__`. Reusing the
  layer's own domain types is what keeps two layers agreeing on what a hand is; avoiding
  the import at this level would trade a load-time cost for a competing definition of
  `HandFeatures`.

---

## Privacy

This layer handles numbers. It reads no clock, touches no file, opens no network
connection, stores nothing between calls, and sees no frame or image — only the measured
geometry that processing already reduced to ratios and angles. Both facts are enforced by
test.

---

## Not in this phase

- **Temporal behaviour.** Swipes, transitions, gestures that only exist as a movement, and
  anything requiring more than one frame.
- **Smoothing, debouncing, cooldown.** No history is kept, so none of these are possible
  here by design.
- **Confidence filtering.** Reported, never applied.
- **Safety state.** Hold-to-confirm, arming, release-to-reset.
- **Gesture events and OS actions.** No event bus, no keyboard or mouse, no Windows APIs,
  and nothing that changes the state of the machine.
- **Learning.** No model, no training data, no downloads, no runtime calibration. Every
  threshold is a `ClassifierConfig` field.
- **User training.** No profile storage, no per-user geometry.
- **Extra gestures.** The five here are the whole vocabulary; `Gesture` is a closed enum.