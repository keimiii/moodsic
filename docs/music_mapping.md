## Current vs. Therapeutic Mapping Approaches

### **Current System: Direct Emotional Matching**
- **Goal**: Validate current emotional state
- **Strategy**: Find music with similar valence-arousal values
- **Example**: Sad person (low valence) → sad music (low valence)

### **Therapeutic System: Mood Improvement Mapping**
- **Goal**: Guide emotional state toward positive change
- **Strategy**: Strategic emotional progression and regulation
- **Example**: Sad person (low valence) → gradually uplifting music (increasing valence)

## Enhanced Therapeutic Mapping Strategies

Note: For this academic POC, we use DEAM static annotations `[1, 9]`. Mapping from FindingEmo to DEAM static:

```python
# FindingEmo: valence in [-3,3], arousal in [0,6]
def fe_to_deam_static(v_fe: float, a_fe: float):
    v_deam = 1.0 + (8.0/6.0) * (v_fe + 3.0)
    a_deam = 1.0 + (8.0/6.0) * a_fe
    return v_deam, a_deam
```

### **1. Valence-Arousal Quadrant-Based Therapy**

Based on the valence-arousal model, we can implement different therapeutic strategies for each emotional quadrant:

```python
class TherapeuticMusicMapper:
    def __init__(self, deam_processor):
        self.processor = deam_processor
        self.current_quadrant = None
        self.target_quadrant = None
        
    def get_therapeutic_music(self, current_v, current_a, improvement_strategy="gradual"):
        current_quadrant = self._get_quadrant(current_v, current_a)
        
        if improvement_strategy == "gradual":
            return self._gradual_improvement(current_v, current_a, current_quadrant)
        elif improvement_strategy == "regulation":
            return self._emotional_regulation(current_v, current_a, current_quadrant)
        elif improvement_strategy == "opposite":
            return self._opposite_emotion(current_v, current_a, current_quadrant)
    
    def _get_quadrant(self, valence, arousal):
        if valence >= 0 and arousal >= 0:
            return "high_high"      # Happy/Excited
        elif valence >= 0 and arousal < 0:
            return "high_low"       # Happy/Calm
        elif valence < 0 and arousal >= 0:
            return "low_high"       # Sad/Anxious
        else:
            return "low_low"        # Sad/Depressed
```

### **2. Gradual Improvement Strategy**

Instead of jumping directly to target emotions, gradually guide users through emotional progression:

```python
def _gradual_improvement(self, current_v, current_a, quadrant):
    if quadrant == "low_low":  # Depressed
        # Step 1: Increase arousal while maintaining low valence
        target_v = current_v + 0.5  # Small valence improvement
        target_a = current_a + 1.0  # Moderate arousal increase
        
    elif quadrant == "low_high":  # Anxious/Stressed
        # Step 1: Reduce arousal first, then improve valence
        target_v = current_v + 0.3  # Small valence improvement
        target_a = current_a - 0.8  # Reduce arousal
        
    elif quadrant == "high_low":  # Calm but could be happier
        # Step 1: Increase arousal to reach high-high quadrant
        target_v = current_v + 0.2  # Maintain high valence
        target_a = current_a + 0.6  # Increase arousal
        
    # Find music segments closest to target emotional state
    v_deam, a_deam = fe_to_deam_static(target_v, target_a)
    return self._find_therapeutic_segment(v_deam, a_deam)
```

### **3. Emotional Regulation Strategy**

For high-arousal negative emotions (anxiety, anger), focus on **calming** rather than mood improvement:

```python
def _emotional_regulation(self, current_v, current_a, quadrant):
    if quadrant == "low_high":  # Anxious/Stressed
        # Primary goal: Reduce arousal to manageable levels
        target_v = current_v + 0.2  # Slight mood improvement
        target_a = current_a - 1.5  # Significant arousal reduction
        
        # Find calming music with gentle positive valence
        v_deam, a_deam = fe_to_deam_static(target_v, target_a)
        return self._find_calming_segment(v_deam, a_deam)
    
    elif quadrant == "high_high":  # Over-excited
        # Primary goal: Moderate arousal while maintaining positivity
        target_v = current_v - 0.3  # Slight valence reduction
        target_a = current_a - 1.0  # Moderate arousal reduction
```

### **4. Opposite Emotion Strategy**

Sometimes the best therapy is to provide the opposite emotional experience:

```python
def _opposite_emotion(self, current_v, current_a, quadrant):
    if quadrant == "low_low":  # Depressed
        # Provide high valence, moderate arousal (uplifting but not overwhelming)
        target_v = 2.0  # High positive valence
        target_a = 2.0  # Moderate arousal (energizing but not stressful)
        
    elif quadrant == "low_high":  # Anxious
        # Provide low arousal, positive valence (calming and positive)
        target_v = 1.5  # Positive valence
        target_a = 0.5  # Low arousal (calm)
```

## Implementation in the Current Architecture

### **Enhanced SegmentMatcher Class**

```python
class TherapeuticSegmentMatcher(SegmentMatcher):
    def __init__(self, segments_df, min_dwell_time=25.0, recent_k=5):
        super().__init__(segments_df, min_dwell_time, recent_k)
        self.therapeutic_mapper = TherapeuticMusicMapper(segments_df)
        self.improvement_progress = 0.0  # Track improvement over time
        self.target_emotions = None
        
    def recommend_therapeutic_music(self, v_fe, a_fe, now=None, strategy="gradual"):
        # Get therapeutic target instead of direct matching
        therapeutic_v, therapeutic_a = self._get_therapeutic_targets(v_fe, a_fe, strategy)
        
        # Scale to DEAM space
        v_deam, a_deam = fe_to_deam(therapeutic_v, therapeutic_a)
        
        # Find segments closest to therapeutic target
        distances, indices = self.kd_tree.query([[v_deam, a_deam]], k=20)
        
        # Apply therapeutic filtering (avoid triggering negative emotions)
        filtered_candidates = self._filter_therapeutic_candidates(indices, distances, v_fe, a_fe)
        
        return self._choose_candidate(filtered_candidates)
    
    def _get_therapeutic_targets(self, current_v, current_a, strategy):
        if strategy == "gradual":
            # Calculate small improvements based on current state
            if current_v < -1.0:  # Very negative
                target_v = current_v + 0.5  # Small improvement
            elif current_v < 0:    # Slightly negative
                target_v = current_v + 0.3  # Smaller improvement
            else:
                target_v = current_v + 0.1  # Maintain positive
                
            if current_a > 4.0:   # Very high arousal
                target_a = current_a - 0.8  # Reduce arousal
            elif current_a < 1.0:  # Very low arousal
                target_a = current_a + 0.6  # Increase arousal
            else:
                target_a = current_a         # Maintain moderate
                
        return target_v, target_a
```

### **User Preference and Strategy Selection**

```python
class TherapeuticStrategySelector:
    def __init__(self):
        self.strategies = {
            "validation": "Match current mood (current system)",
            "gradual": "Gradual mood improvement",
            "regulation": "Emotional regulation (calming)",
            "opposite": "Opposite emotional experience",
            "custom": "User-defined improvement path"
        }
    
    def select_strategy(self, user_preference, current_emotion, improvement_goals):
        if user_preference == "mood_boost":
            return "gradual"
        elif user_preference == "calm_down":
            return "regulation"
        elif user_preference == "distraction":
            return "opposite"
        elif user_preference == "validation":
            return "validation"
        else:
            return "gradual"  # Default to gradual improvement
```

## Key Benefits of Therapeutic Mapping

### **1. Progressive Improvement**
- Avoids overwhelming users with dramatic emotional shifts
- Builds emotional resilience through gradual positive experiences
- Respects individual emotional processing speed

### **2. Context-Aware Therapy**
- Different strategies for different emotional states
- Anxiety: Focus on arousal reduction first
- Depression: Focus on valence improvement first
- Stress: Balance both dimensions appropriately

### **3. User Control and Personalization**
- Users can choose their therapeutic approach
- System learns from user feedback and preferences
- Adapts strategy based on improvement progress

### **4. Safety and Ethics**
- Avoids triggering negative emotional spirals
- Provides emotional validation when needed
- Balances improvement goals with emotional safety

## Example Therapeutic Journey

**Scenario**: User is feeling depressed (low valence, low arousal)

**Session 1 (0-5 minutes)**:
- Current: V=-2.0, A=0.5
- Target: V=-1.5, A=1.0 (slight improvement)
- Music: Gentle, slightly uplifting, moderate energy

**Session 2 (5-15 minutes)**:
- Current: V=-1.5, A=1.0
- Target: V=-0.8, A=1.5 (continued improvement)
- Music: More positive, energizing but not overwhelming

**Session 3 (15-25 minutes)**:
- Current: V=-0.8, A=1.5
- Target: V=0.0, A=2.0 (neutral to positive)
- Music: Clearly positive, moderately energetic

This approach transforms the system from a simple "emotional mirror" into a therapeutic tool that actively helps users improve their emotional well-being while respecting their current state and individual needs.

```python
class TherapeuticMusicMapper:
    def __init__(self, deam_processor):
        self.processor = deam_processor
        self.current_quadrant = None
        self.target_quadrant = None
        
    def get_therapeutic_music(self, current_v, current_a, improvement_strategy="gradual"):
        current_quadrant = self._get_quadrant(current_v, current_a)
        
        if improvement_strategy == "gradual":
            return self._gradual_improvement(current_v, current_a, current_quadrant)
        elif improvement_strategy == "regulation":
            return self._emotional_regulation(current_v, current_a, current_quadrant)
        elif improvement_strategy == "opposite":
            return self._opposite_emotion(current_v, current_a, current_quadrant)
    
    def _get_quadrant(self, valence, arousal):
        if valence >= 0 and arousal >= 0:
            return "high_high"      # Happy/Excited
        elif valence >= 0 and arousal < 0:
            return "high_low"       # Happy/Calm
        elif valence < 0 and arousal >= 0:
            return "low_high"       # Sad/Anxious
        else:
            return "low_low"        # Sad/Depressed
```

```python
def _gradual_improvement(self, current_v, current_a, quadrant):
    if quadrant == "low_low":  # Depressed
        # Step 1: Increase arousal while maintaining low valence
        target_v = current_v + 0.5  # Small valence improvement
        target_a = current_a + 1.0  # Moderate arousal increase
        
    elif quadrant == "low_high":  # Anxious/Stressed
        # Step 1: Reduce arousal first, then improve valence
        target_v = current_v + 0.3  # Small valence improvement
        target_a = current_a - 0.8  # Reduce arousal
        
    elif quadrant == "high_low":  # Calm but could be happier
        # Step 1: Increase arousal to reach high-high quadrant
        target_v = current_v + 0.2  # Maintain high valence
        target_a = current_a + 0.6  # Increase arousal
        
    # Find music segments closest to target emotional state
    v_deam, a_deam = fe_to_deam_static(target_v, target_a)
    return self._find_therapeutic_segment(v_deam, a_deam)
```

```python
def _emotional_regulation(self, current_v, current_a, quadrant):
    if quadrant == "low_high":  # Anxious/Stressed
        # Primary goal: Reduce arousal to manageable levels
        target_v = current_v + 0.2  # Slight mood improvement
        target_a = current_a - 1.5  # Significant arousal reduction
        
        # Find calming music with gentle positive valence
        v_deam, a_deam = fe_to_deam_static(target_v, target_a)
        return self._find_calming_segment(v_deam, a_deam)
    
    elif quadrant == "high_high":  # Over-excited
        # Primary goal: Moderate arousal while maintaining positivity
        target_v = current_v - 0.3  # Slight valence reduction
        target_a = current_a - 1.0  # Moderate arousal reduction
```

```python
def _opposite_emotion(self, current_v, current_a, quadrant):
    if quadrant == "low_low":  # Depressed
        # Provide high valence, moderate arousal (uplifting but not overwhelming)
        target_v = 2.0  # High positive valence
        target_a = 2.0  # Moderate arousal (energizing but not stressful)
        
    elif quadrant == "low_high":  # Anxious
        # Provide low arousal, positive valence (calming and positive)
        target_v = 1.5  # Positive valence
        target_a = 0.5  # Low arousal (calm)
```

```python
class TherapeuticSegmentMatcher(SegmentMatcher):
    def __init__(self, segments_df, min_dwell_time=25.0, recent_k=5):
        super().__init__(segments_df, min_dwell_time, recent_k)
        self.therapeutic_mapper = TherapeuticMusicMapper(segments_df)
        self.improvement_progress = 0.0  # Track improvement over time
        self.target_emotions = None
        
    def recommend_therapeutic_music(self, v_fe, a_fe, now=None, strategy="gradual"):
        # Get therapeutic target instead of direct matching
        therapeutic_v, therapeutic_a = self._get_therapeutic_targets(v_fe, a_fe, strategy)
        
        # Scale to DEAM space
        v_deam, a_deam = fe_to_deam(therapeutic_v, therapeutic_a)
        
        # Find segments closest to therapeutic target
        distances, indices = self.kd_tree.query([[v_deam, a_deam]], k=20)
        
        # Apply therapeutic filtering (avoid triggering negative emotions)
        filtered_candidates = self._filter_therapeutic_candidates(indices, distances, v_fe, a_fe)
        
        return self._choose_candidate(filtered_candidates)
    
    def _get_therapeutic_targets(self, current_v, current_a, strategy):
        if strategy == "gradual":
            # Calculate small improvements based on current state
            if current_v < -1.0:  # Very negative
                target_v = current_v + 0.5  # Small improvement
            elif current_v < 0:    # Slightly negative
                target_v = current_v + 0.3  # Smaller improvement
            else:
                target_v = current_v + 0.1  # Maintain positive
                
            if current_a > 4.0:   # Very high arousal
                target_a = current_a - 0.8  # Reduce arousal
            elif current_a < 1.0:  # Very low arousal
                target_a = current_a + 0.6  # Increase arousal
            else:
                target_a = current_a         # Maintain moderate
                
        return target_v, target_a
```

```python
class TherapeuticStrategySelector:
    def __init__(self):
        self.strategies = {
            "validation": "Match current mood (current system)",
            "gradual": "Gradual mood improvement",
            "regulation": "Emotional regulation (calming)",
            "opposite": "Opposite emotional experience",
            "custom": "User-defined improvement path"
        }
    
    def select_strategy(self, user_preference, current_emotion, improvement_goals):
        if user_preference == "mood_boost":
            return "gradual"
        elif user_preference == "calm_down":
            return "regulation"
        elif user_preference == "distraction":
            return "opposite"
        elif user_preference == "validation":
            return "validation"
        else:
            return "gradual"  # Default to gradual improvement
```

