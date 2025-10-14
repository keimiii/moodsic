## Music Mapping Status

This document records the current state of music retrieval in the Moodsic POC and
captures design sketches for the therapeutic roadmap. The goal is to make it
obvious which behavior is shipping today versus what remains aspirational.

---

## Current Implementation — Direct Emotional Matching

### Retrieval goal
- Mirror the listener's detected emotional state by selecting songs whose
  valence/arousal scores are closest to the current observation.

### Data and scale alignment
- Upstream perception modules produce valence/arousal in the **reference space**
  `[-1, 1]` (see `utils/emotion_scale_aligner.EmotionScaleAligner`).
- Dataset rows used for retrieval (DEAM static annotations) are pre-converted to
  the same reference space and stored as `valence_ref` / `arousal_ref`.
- When other scales are needed (e.g., FindingEmo inputs), we convert them at the
  pipeline boundary with `EmotionScaleAligner`. The matcher itself expects
  reference-space values.

```python
from utils.emotion_scale_aligner import EmotionScaleAligner
from utils.song_matcher import SongMatcher

# Incoming FindingEmo reading → reference space
aligner = EmotionScaleAligner()
v_ref, a_ref = aligner.findingemo_to_reference(v_fe, a_fe)

# SongMatcher works directly in reference space
matcher = SongMatcher(
    songs_df=songs_df,
    scaler=scaler,
    gmm=gmm,
    min_dwell_time=25.0,
    recent_k=5,
)
result = matcher.recommend(v_ref, a_ref)
```

### What SongMatcher actually does (`utils/song_matcher.py`)
- **Cluster gating**: transforms `(v_ref, a_ref)` with the fitted scaler/GMM and
  limits candidates to the most probable clusters.
- **Distance ranking**: computes Euclidean distance in reference space and keeps
  the top-`k` nearest songs.
- **Repeat avoidance**: tracks a recent song deque and skips songs still in the
  dwell window (`min_dwell_time`).
- **Outputs**: returns the selected `pandas.Series` plus metadata in
  `MatchResult` (`switch` flag, timestamp). No therapeutic progression state is
  tracked today.

### Known limitations
- No concept of mood improvement or regulation; the system mirrors the current
  emotion.
- Therapeutic preferences, progression pacing, and safety constraints are not
  represented in code.
- Candidate scoring relies on a single distance metric; there is no semantic or
  content-based filtering beyond basic gating.

---

## Planned Therapeutic Mapping (Not Implemented Yet)

The following sections capture design sketches for future work. They are **not**
part of the current codebase; treat them as product/engineering notes.

### Proposed `TherapeuticMusicMapper`

```python
# Prototype only – implementation not landed
class TherapeuticMusicMapper:
    def __init__(self, songs_df):
        self.processor = songs_df  # placeholder for DEAM processor
        self.aligner = EmotionScaleAligner()

    def get_therapeutic_music(self, current_v, current_a, improvement_strategy="gradual"):
        quadrant = self._get_quadrant(current_v, current_a)
        if improvement_strategy == "gradual":
            return self._gradual_improvement(current_v, current_a, quadrant)
        if improvement_strategy == "regulation":
            return self._emotional_regulation(current_v, current_a, quadrant)
        if improvement_strategy == "opposite":
            return self._opposite_emotion(current_v, current_a, quadrant)
```

Envisioned responsibilities:
- Map the listener's FindingEmo readings into DEAM space for retrieval.
- Select intermediate emotional targets based on quadrant-specific heuristics.
- Provide helper methods (`_gradual_improvement`, `_emotional_regulation`,
  `_opposite_emotion`) that yield target `(valence, arousal)` pairs.

### Strategy sketches

1. **Gradual improvement** (low valence → stepwise boosts in valence and/or
   arousal depending on the starting quadrant).
2. **Emotional regulation** (high-arousal negative emotions → lower arousal
   before boosting valence).
3. **Opposite emotion** (purposefully contrasting stimulus for distraction when
   user requests it).

Each strategy would convert the desired targets back into DEAM static values and
ask the matcher to retrieve the nearest candidates.

### Integrating with `SongMatcher`

The roadmap envisions a therapeutic wrapper:

```python
# Prototype only – not in repository
class TherapeuticSongMatcher(SongMatcher):
    def recommend_therapeutic_music(self, v_fe, a_fe, strategy="gradual"):
        therapeutic_v, therapeutic_a = self._get_therapeutic_targets(v_fe, a_fe, strategy)
        v_deam, a_deam = self.aligner.findingemo_to_deam_static(therapeutic_v, therapeutic_a)
        return self._choose_candidate_from_deam(v_deam, a_deam)
```

Key design ideas still under discussion:
- Track improvement progress over multiple selections.
- Filter out candidates that could be emotionally risky during transitions.
- Respect user-selected strategies (`validation`, `gradual`, `regulation`,
  `opposite`, `custom`).

### Example therapeutic journey (concept)
- **Session 1**: Depressed user (V=-2.0, A=0.5) → target (-1.5, 1.0): gentle
  uplift.
- **Session 2**: (-1.5, 1.0) → target (-0.8, 1.5): add energy.
- **Session 3**: (-0.8, 1.5) → target (0.0, 2.0): reach neutral/positive.

These examples are illustrative and highlight the intended progression once the
therapeutic mapper exists.

---

## Next Steps
- Decide whether the therapeutic roadmap will extend the existing matcher or
  replace it with a new component.
- Define concrete acceptance criteria for each strategy (e.g., safety bounds,
  dwell behavior across transitions).
- When implementation begins, update this document again to move features from
  "Planned" into "Current" as they ship.
