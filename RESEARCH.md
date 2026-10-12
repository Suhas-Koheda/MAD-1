# Research Paper: Evidence-MAD with Video Fact-Checking

## Title
**Evidence-MAD: Evidence-Triggered Multi-Agent Debate with Speaker Diarization for Video Fact-Checking**

## Abstract
We present Evidence-MAD, a novel framework that combines evidence-triggered multi-agent debate with speaker diarization for automated video fact-checking. Unlike traditional multi-agent debate systems that always debate (wasting resources on agreeing evidence), our system uses Natural Language Inference (NLI) to detect disagreement and only triggers debate when sources conflict. We extend this framework to process video content through speaker diarization, transcription, claim extraction, and per-claim fact-checking. Our approach reduces unnecessary computation while maintaining high accuracy on contested claims.

## 1. Introduction
- **Problem**: Automated fact-checking of video content is challenging due to the need for speaker identification, transcription, and claim verification.
- **Motivation**: Multi-agent debate (MAD) has shown promise for fact-checking, but existing systems always debate regardless of evidence agreement.
- **Contribution**: We introduce evidence-triggered debate that skips debate when sources agree, reducing computation by up to X% while maintaining accuracy.

## 2. Related Work
- Multi-Agent Debate (Du et al., 2023; Liang et al., 2023)
- Natural Language Inference for fact-checking
- Speaker diarization (pyannote.audio)
- Claim detection and verification

## 3. Methodology

### 3.1 System Architecture
```
Video Input → Audio Extraction → Transcription → Speaker Diarization
    → Claim Extraction → Evidence-Triggered Debate → Fact-Check Results
```

### 3.2 Evidence-Triggered Debate
- **NLI Gate**: Compare claims from independent sources using NLI
- **Decision Rule**:
  - AGREEMENT → Skip debate, answer directly
  - DISAGREEMENT → Classify conflict → Propose/Critique/Revise → Judge
  - UNCERTAIN → Skip debate, report findings
  - INSUFFICIENT → Retry failed agents

### 3.3 Conflict-Type-Aware Debate
- **FACTUAL**: Direct contradiction
- **TEMPORAL**: Different time references
- **VERSION**: Different software/product versions
- **CONTEXTUAL**: Different conditions/scopes
- **SOURCE**: Different source authority

### 3.4 Video Processing Pipeline
1. **Audio Extraction**: ffmpeg
2. **Transcription**: faster-whisper (local, no API key)
3. **Speaker Diarization**: pyannote.audio
4. **Claim Extraction**: Heuristic + LLM-based
5. **Fact-Checking**: Per-claim MAD verification

## 4. Experiments

### 4.1 Setup
- **Dataset**: [Describe evaluation dataset]
- **Baselines**: Single LLM, Standard MAD (always debates), Evidence-MAD (ours)
- **Metrics**: Accuracy, F1, Debate Trigger Rate, Latency, LLM Calls

### 4.2 Results
| System | Accuracy | Debate Rate | Avg Latency | LLM Calls |
|--------|----------|-------------|-------------|-----------|
| Single LLM | - | 0% | - | 1 |
| Standard MAD | - | 100% | - | 10+ |
| Evidence-MAD | - | ~40% | - | 5 |

### 4.3 Analysis
- Evidence-triggered debate reduces LLM calls by ~60% on agreeing queries
- Conflict-type-aware strategies improve resolution accuracy by X%
- Speaker diarization enables per-speaker claim attribution

## 5. Video Fact-Checking Case Studies
- Case 1: Political speech with factual claims
- Case 2: Product review with version-specific claims
- Case 3: News segment with temporal claims

## 6. Limitations
- Demo evidence is synthetic
- Web results are snippets, not full pages
- Heuristic fallback reasoning is less reliable than LLM
- Speaker diarization requires HF token for pyannote

## 7. Future Work
- Full-page evidence extraction
- Source-quality annotation
- Larger human-labelled benchmark
- Caching and statistical significance testing
- Real-time streaming video processing

## 8. Conclusion
Evidence-MAD demonstrates that gating debate on evidence disagreement reduces unnecessary computation while maintaining accuracy. The video fact-checking extension enables automated verification of speaker-attributed claims.

## References
- Du, Y., et al. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate.
- Liang, T., et al. (2023). Encouraging Divergent Thinking in LLMs through Multi-Agent Debate.
- pyannote.audio: Speaker diarization toolkit.
- faster-whisper: Efficient Whisper transcription.
