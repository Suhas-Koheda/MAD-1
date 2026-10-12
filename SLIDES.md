# Evidence-MAD: Presentation Outline

## Slide 1: Title
**Evidence-MAD: Evidence-Triggered Multi-Agent Debate with Video Fact-Checking**
Suhas Koheda | BCSE497J | VIT Vellore | October 2026

## Slide 2: Problem
- Multi-agent debate always debates (wastes resources)
- No video fact-checking with speaker attribution
- Need intelligent gating mechanism

## Slide 3: Solution
- NLI gate: skip debate when sources agree
- 9 independent search agents
- Video pipeline: diarize → transcribe → extract → fact-check

## Slide 4: Architecture
[Architecture diagram from outputs/architecture.png]

## Slide 5: Evidence Gate
- AGREE: All pairs entailment ≥ 0.70 → skip debate
- DISAGREE: Any pair contradiction ≥ 0.70 → trigger debate
- UNCERTAIN: neither → skip
- INSUFFICIENT: < 2 usable agents → retry

## Slide 6: 9 Search Agents
| Agent | Provider |
|-------|----------|
| Web | DuckDuckGo |
| Wiki | Wikipedia API |
| Reference | Wikipedia API |
| Scholarly | Crossref |
| News | Google News RSS |
| Code | GitHub API |
| FactCheck | Snopes |
| Social | Reddit JSON |
| Video | YouTube |

## Slide 7: Conflict Types
- FACTUAL: "100°C" vs "90°C"
- TEMPORAL: "2019" vs "2020"
- VERSION: "Python 3.13" vs "3.14"
- CONTEXTUAL: "adults" vs "children"
- SOURCE: blog vs peer-reviewed

## Slide 8: Video Pipeline
1. Audio extraction (ffmpeg)
2. Transcription (faster-whisper)
3. Speaker diarization (pyannote.audio)
4. Claim extraction
5. Fact-checking (MAD)

## Slide 9: Results
| Test | Result |
|------|--------|
| Evidence Gate | PASS |
| Disagreement Detection | PASS |
| Agent Registration (9/9) | PASS |
| Query Routing (6/6) | PASS |
| Conflict Classifier (3/3) | PASS |
| Debate Manager (3 rounds) | PASS |
| Judge (0.82 confidence) | PASS |
| **Total** | **10/10 (100%)** |

## Slide 10: Live Results
- 6 queries tested with Groq API
- Avg LLM calls: 3.5 per query
- Avg latency: 10.8s
- States: 4 agreement, 2 insufficient (rate limits)

## Slide 11: Outputs
- Text mode: query → evidence → debate → answer
- Video mode: upload → diarize → transcript → fact-check
- RawBlock UI design

## Slide 12: Limitations & Future Work
- Synthetic fixture data
- External API rate limits
- Future: full-page extraction, larger benchmarks, statistical testing

## Slide 13: Conclusion
- Evidence-triggered debate reduces unnecessary computation
- Video fact-checking with speaker attribution is feasible
- 100% on controlled tests, promising live results

## Slide 14: Thank You
Questions?
