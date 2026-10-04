# Confidence Filter

Phase 6. Implements the **Confidence Filter** stage of the pipeline: a stream of
`GestureClassification` results in, a stability verdict out.

The classifier reads one frame and answers "which pose is this, and how strongly?". It is
right to have no memory — but a control surface needs the difference between a hand held
still for a second and one that flashed past for a single frame. This layer supplies exactly
that difference and nothing else.

```
GestureClassification x N  ->  [candidate, run length, confidence sum]  ->  StabilityReport
```

## Layout

| File | Contents |
|---|---|
| `types.py` | `FilterState`, `CandidateChangePolicy`, `ConfidenceFilterConfig`, `StabilityReport` |
| `filter.py` | `ConfidenceFilter`, the state transitions |
| `errors.py` | `ConfidenceError`, `InvalidClassificationError`, `FilterStateError` |

## API

```python
from gesturepilot.confidence import ConfidenceFilter, ConfidenceFilterConfig

stability = ConfidenceFilter(ConfidenceFilterConfig(min_confidence=0.6, min_stable_observations=3))

for hand in hands:
    report = stability.observe(classifier.classify(hand), timestamp=frame.timestamp)
    if report.is_stable:
        print(report.accepted, report.candidate_observations, report.mean_confidence)

stability.reset()  # forget everything
```

One `StabilityReport` comes back per observation:

| Field | Meaning |
|---|---|
| `state` | `IDLE`, `CANDIDATE`, or `STABLE` |
| `candidate` | The gesture being accumulated, or `None` |
| `candidate_observations` | Length of the current run, `0` when `IDLE` |
| `accepted` | The stable gesture, or `None` — non-`None` only when `STABLE` |
| `latest_confidence` | This observation's classifier confidence, verbatim |
| `mean_confidence` | Mean classifier confidence over the run, or `None` |
| `interrupted` | Whether a long gap discarded the run before this observation |
| `classification` | The classifier result itself, unmodified |

---

## The state model

The entire state is three values: **the candidate gesture, the run length, and the sum of
the run's confidences**. Everything a caller sees is derived from those three and the
configuration — `state`, `accepted`, `mean_confidence` are all computed, never stored.

That is the whole design. There is no separate `is_stable` flag to fall out of step with the
run that produced it, and `reset()` is a single assignment because there is nothing else to
forget.

```
                    observation qualifies (recognised, confidence >= min_confidence)
   ┌────────┐      ┌──────────────────────────────────────────────────┐  ┌────────┐
   │  IDLE  │─────▶│ candidate = gesture, run = 1                     │─▶│        │
   │        │      └──────────────────────────────────────────────────┘  │CANDIDATE│
   └────────┘                                                             └────────┘
        ▲                                                                     │ run reaches
        │                                                                     │ min_stable_observations
        │ observation does not qualify                 same gesture, run += 1  ▼
        ├──────────────────────────────┐                          ┌──────────────┐
        │                              │                          │    STABLE    │
        │        different gesture     │      RESET policy        │ accepted =   │
        │        (RESET policy)        │─────────────────────────▶│  candidate   │
        │                              │   candidate = gesture,   └──────────────┘
        └──────────────────────────────┘   run = 1
```

Transitions, in full:

| From | Observation | To |
|---|---|---|
| any | `UNKNOWN`, or confidence `< min_confidence` | `IDLE`, run cleared |
| `IDLE` | qualifying | `CANDIDATE`, run = 1 |
| `CANDIDATE` | same gesture, qualifying | run += 1; `STABLE` once it reaches the requirement |
| `CANDIDATE` | different gesture, qualifying | `RESET`: new candidate, run = 1. `HOLD`: unchanged |
| `STABLE` | same gesture, qualifying | stays `STABLE`, run keeps growing |
| `STABLE` | different gesture, qualifying | `RESET`: `CANDIDATE` on the new gesture, run = 1, acceptance lost. `HOLD`: unchanged |
| `STABLE` | `UNKNOWN` / low confidence / interruption | `IDLE`, acceptance lost |

**A new candidate is never accepted immediately** unless `min_stable_observations=1`, which
is the explicit opt-in. The default of `3` is what makes that true.

### `UNKNOWN` and low confidence are not evidence

Neither counts toward stability, and both **end** the current run rather than being ignored.
A weak frame is not neutral: it is the classifier saying the evidence did not separate a
gesture, and accumulating stability across one would mean accumulating evidence the
classifier explicitly declined to give.

### The filter reports the present, not the past

When an accepted gesture is followed by an `UNKNOWN`, a low-confidence result, a different
gesture, or a long gap, the report immediately stops accepting it. That is a statement about
what is happening now, not a cancellation of anything: **this layer dispatches nothing**, so
there is nothing to cancel. What a downstream safety stage should do when stability is lost —
abort a pending confirmation, or hold the last accepted gesture through a one-frame dip — is
a decision for that stage, not a side effect of a threshold here.

---

## Threshold semantics

**`min_confidence` is inclusive.** An observation at exactly the threshold qualifies;
anything strictly below it does not. It gates *qualifying*, so the rule is `confidence >=
min_confidence`.

**Confidence is reported, never invented.**

- `latest_confidence` is the classifier's own number for this observation, unchanged. For an
  `UNKNOWN` it still means *confidence that no gesture applies* — high when the evidence was
  weak — and the filter does not invert it, so reading that field alone cannot make a weak
  result look like a strong one.
- `mean_confidence` is the arithmetic mean of the classifier's confidence over the run's
  qualifying observations. It is a run summary, **not** a confidence in the classifier's
  sense, and is not comparable with `latest_confidence`'s meaning. Excluded observations are
  excluded from it rather than averaged in, so a weak frame cannot be smoothed away.
- `classification` is the whole classifier result, margin and scores included, so nothing
  downstream has to reconstruct what the filter saw.

---

## Stability measurement semantics

Stability is counted in **observations**, never in seconds. The frame rate is a property of
the camera and the machine; a filter measured in seconds would accept a gesture twice as fast
on a fast machine. A run of `min_stable_observations` therefore means "this many consecutive
qualifying observations of the same gesture", and nothing about the wall clock.

Optional timestamps add interruption detection without changing that:

| `timestamp` | Policy |
|---|---|
| omitted | Always legal. The observation still counts, and the filter forgets its last timestamp so it cannot measure a gap across a period it did not witness. |
| equal to the last | Legal. A zero gap is shorter than any limit, so two observations may share an instant when a caller batches or the frame clock is coarser than the frame rate. Unlike `OneEuroFilter`, which divides by the interval, this filter only compares a gap against a threshold. |
| backwards | Raises `FilterStateError`. The caller handed over observations out of order. |
| further ahead than `max_interruption_seconds` | Not an error. The run is discarded first, and `interrupted=True` says so. |

`max_interruption_seconds` defaults to `None`, which disables the whole mechanism and leaves
the filter purely count-based. It exists for the case counting cannot see: a hand that left
the frame entirely, where the caller keeps feeding observations of *something else*, or
feeds nothing at all and resumes minutes later.

### Missing observations

There is no fake label for a frame that was never classified. The API has two honest ways to
express a break, and the caller picks per situation:

- **The hand left the frame and later returned** — call `reset()`, or rely on
  `max_interruption_seconds` if the caller knows the frame timestamps. This is the same choice
  `OneEuroFilter` offers for the same reason.
- **No classification is available for this frame** — simply do not call `observe()`. The run
  survives, and with `max_interruption_seconds` set, is broken if the gap was long enough.

Inventing a `UNKNOWN` for a missing frame would be a lie about what the classifier saw, and
would also silently count as an observation the pipeline never made.

`scripts/pipeline_smoke_test.py` is the caller that does this in practice: a frame with no hand
resets both the smoothing state and the filter, and prints `no hand in frame` rather than the
last gesture it saw.

---

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `min_confidence` | `0.6` | Lowest confidence that may count as evidence. Inclusive. |
| `min_stable_observations` | `3` | Consecutive qualifying observations needed to accept. `1` explicitly permits immediate acceptance. |
| `candidate_change` | `RESET` | What a qualifying observation of a different gesture does. |
| `max_interruption_seconds` | `None` | Gap beyond which the run is discarded. `None` disables it. |

`CandidateChangePolicy`:

- **`RESET`** — the new gesture becomes the candidate and starts from one observation. The
  incumbent must earn stability again. Conservative, and the default: a hand moving from a
  fist to a point has not demonstrated a stable point yet.
- **`HOLD`** — the incumbent keeps its candidate *and* its accumulated run; the differing
  observation is discarded. Suited to two poses that genuinely flicker between each other, at
  the cost of ignoring a real gesture change for as long as the incumbent keeps qualifying.

All fields are validated on construction, so a threshold that could not produce a sensible
answer is reported where it was written.

The defaults are a starting point, not a tuned profile. Nothing here has been fitted against
recorded hands, and the two numbers that matter most interact with the classifier's own
thresholds: the classifier already refuses weak poses via `min_score` and `min_margin`, so
`min_confidence=0.6` sits above most of what a real hand produces and is mostly there to
reject the tail.

---

## Design decisions

**Counting, not clocking.** Covered above: a seconds-based filter would make stability a
function of the machine.

**The whole state is three values.** No derived value is stored, so no derived value can
disagree with the run. Enforced by a test asserting the filter's exact instance attributes.

**No hidden clock, no threads, no sleeps.** Timestamps are the caller's, as in the camera and
smoothing layers. Enforced by AST tests, along with the ban on OpenCV, MediaPipe, NumPy, and
the automation surface.

**Label-agnostic.** No module names an individual gesture, so adding a sixth gesture is a
classifier change rather than a filter change. Also enforced by AST test.

**A rejected classification is not an exception.** `UNKNOWN` and low confidence are the filter
answering. Only input it could not evaluate raises, so an ordinary ambiguous pose never needs
a `try`.

**Defence in depth on input validation.** `GestureClassification` already rejects a `NaN` or
an out-of-range confidence, so the filter's checks only fire on a forged result. They are
there anyway: a `NaN` compares false against every threshold, which would quietly stop every
gesture being accepted, forever, with no error anywhere.

### Trade-offs

- **Losing stability is immediate.** One ambiguous frame drops an accepted gesture. This is
  the safe direction to be wrong in, and a caller who needs tolerance across brief dips
  should buy it with `CandidateChangePolicy.HOLD` — but `HOLD` forgives only *qualifying*
  label flicker, never a dip below `min_confidence`. Debouncing acceptance across dips
  belongs to the safety state machine.
- **No decay.** A run that reached the requirement stays accepted as long as the same gesture
  keeps qualifying; `candidate_observations` grows without bound. Nothing reads the count
  after acceptance, so this is inert, but a caller wanting a bounded run should call
  `reset()` at the point where the gesture is considered finished.
- **`HOLD` can pin a stale gesture.** If the incumbent keeps qualifying, a genuinely different
  gesture is ignored indefinitely. `RESET` is the default for that reason.
- **Interruption needs timestamps on both sides.** With `timestamp=None` on any observation,
  no gap can be measured across it. Documented rather than guessed at.
- **One filter per subject.** The filter has no hand identity, by design: identity across
  frames is the tracking layer's problem, and a caller tracking two hands must hold two
  filters or their evidence adds up into a stability neither showed.

---

## Known limitations

- **It filters a rule-based classifier.** Every judgement here is downstream of five `if`-shaped
  rules on extension ratios and one thumb-index distance. If those rules misidentify a pose
  confidently, the filter will happily stabilise the wrong label — a *confident* classifier
  error is not what this layer is built to catch.
- **This is not a false-positive guarantee.** Requiring three consecutive qualifying
  observations removes brief flickers and single-frame flashes. It does not eliminate false
  positives, and nothing here has been measured against real hands, real lighting, or real
  camera noise. No reliability claim is made.
- **It has no model of intent.** A user holding a fist to scratch an itch produces exactly the
  same observations as one deliberately holding a fist. Deciding that a gesture was *meant* is
  the safety state machine's problem.
- **It cannot recover a dropped frame rate.** If the pipeline delivers three observations a
  second, three observations still looks stable. Stability is measured in observations by
  design; `max_interruption_seconds` only catches gaps, not slowness.
- **No dynamics.** Swipes, waves, and transitions need more than a repeated label and are out
  of scope; the filter would see a swipe as a candidate that never stabilises.
- **One gesture at a time.** Multi-hand and per-hand gestures are not modelled.

---

## Accepted is not authorised

`StabilityReport.accepted` means one thing: *this gesture has been seen consistently enough
to be called stable.* It is not a confirmation, not an authorisation, and not an instruction
to act on.

Whether a stable gesture may cause an action is decided by the **safety state machine** in a
later phase, together with hold-to-confirm, cooldown, release-to-reset, and arming. This
package contains no event bus, no dispatcher, and no OS code, and there is no method here
that performs an action — which is why a mistake in it cannot close a window.

---

## Privacy

This package handles numbers only. No file, network, clock, or logging access — enforced by
test.

---

## Not in this phase

The safety state machine, gesture events, the action dispatcher, OS-level controls, camera or
tracking changes, classifier rule changes, dynamic gestures such as swipes, any GUI, and any
background thread or timer. `main` still only prints a banner.