# Paper Outline: Evidence-MAD with Video Fact-Checking

## Target Venues
- ACL/EMNLP (NLP track)
- AAAI (AI track)
- WWW (Web track)

## Key Research Questions
1. Does evidence-triggered debate reduce computation without sacrificing accuracy?
2. Does conflict-type-aware debate improve resolution quality?
3. Can speaker diarization improve fact-checking accuracy for multi-speaker videos?

## Experiments to Run

### Experiment 1: Evidence Gate Ablation
- Compare: Always debate vs. Evidence-triggered debate
- Metric: Accuracy, LLM calls, Latency
- Expected: 40-60% reduction in LLM calls with <2% accuracy drop

### Experiment 2: Conflict Type Classification
- Compare: Generic debate vs. Conflict-type-aware debate
- Metric: Resolution accuracy, Judge confidence
- Expected: 5-10% improvement with type-aware strategies

### Experiment 3: Number of Agents
- Compare: 2, 3, 5, 8 agents
- Metric: Accuracy, Coverage, Latency
- Expected: Diminishing returns after 5 agents

### Experiment 4: Video Fact-Checking
- Dataset: 100 videos with labeled claims
- Compare: Single LLM, Standard MAD, Evidence-MAD
- Metric: Claim-level F1, Speaker attribution accuracy

## Data Collection
- [ ] Collect 100+ videos with factual claims
- [ ] Label claims as SUPPORTED/REFUTED/NEUTRAL
- [ ] Annotate speaker segments
- [ ] Run all three systems on the dataset

## Writing Timeline
- Week 1-2: Complete experiments 1-3
- Week 3: Collect video dataset
- Week 4: Run video experiments
- Week 5: Write paper
- Week 6: Revise and submit
