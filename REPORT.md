# Evidence-MAD: Project Report

**Author:** Suhas Koheda  
**Course:** BCSE497J – Project I  
**Institution:** VIT Vellore  
**Date:** October 2026

---

## 1. Project Overview

Evidence-MAD is a research prototype for evidence-triggered, conflict-type-aware multi-agent debate with video fact-checking capabilities. The system retrieves evidence independently from 9 sources, compares claims using Natural Language Inference (NLI), and only debates when evidence disagrees.

---

## 2. Objectives

1. Implement evidence-triggered debate (skip debate when sources agree)
2. Build 9 independent search agents with parallel retrieval
3. Classify conflicts into 5 types (Factual, Temporal, Version, Contextual, Source)
4. Add video fact-checking with speaker diarization
5. Create a modern RawBlock UI for both text and video modes

---

## 3. Implementation Details

### 3.1 Core Components

| Component | File | Description |
|-----------|------|-------------|
| Main Pipeline | `app.py` | MADSystem class with solve() method |
| Router | `agents/router.py` | Query routing to 9 agents |
| NLI Detector | `evidence/disagreement_detector.py` | Mock + transformer NLI |
| Aggregator | `evidence/aggregator.py` | N-agent evidence comparison |
| Classifier | `conflict/classifier.py` | 5-type conflict classification |
| Debate | `debate/manager.py` | Propose-Critique-Revise cycle |
| Judge | `debate/judge.py` | Evidence-prioritizing judge |
| Video | `video_processor.py` | Diarization + transcription + fact-check |

### 3.2 Search Agents (9)

| Agent | Provider | Focus |
|-------|----------|-------|
| WebAgent | DuckDuckGo | General web |
| WikiAgent | Wikipedia API | Reference |
| ReferenceAgent | Wikipedia API | Background |
| ScholarlyAgent | Crossref | Academic papers |
| NewsAgent | Google News RSS | Current events |
| CodeAgent | GitHub API | Code repositories |
| FactCheckAgent | Snopes + Google | Fact-checks |
| SocialAgent | Reddit JSON | Community opinions |
| VideoAgent | YouTube | Video content |

### 3.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/solve` | POST | Solve a text query |
| `/api/agents` | GET | List all agents |
| `/api/video/upload` | POST | Async video analysis |
| `/api/video/analyze-sync` | POST | Sync video analysis |
| `/api/video/status/{id}` | GET | Check job status |
| `/api/video/result/{id}` | GET | Get video results |

---

## 4. Results

### 4.1 Evaluation Summary

| Test | Result | Status |
|------|--------|--------|
| Evidence Gate (Agreement) | Debate skipped | PASS |
| Disagreement Detection | 3 rounds, temporal conflict | PASS |
| Agent Registration | 9/9 agents | PASS |
| Query Routing | 6/6 query types | PASS |
| Live Mode (Groq) | 3 LLM calls, 3.19s | PASS |
| NLI Detector | 3/3 predictions | PASS |
| Conflict Classifier | 3/3 classifications | PASS |
| Evidence Aggregator | Agreement + insufficient | PASS |
| Debate Manager | 3 rounds completed | PASS |
| Judge | Winner: candidate_1, 0.82 confidence | PASS |

**Overall: 10/10 tests passed (100%)**

### 4.2 Live Mode Results

| Query | State | Agents | LLM Calls | Latency |
|-------|-------|--------|-----------|---------|
| Capital of Australia | insufficient | 1 | 3 | 6.4s |
| India NEP | agreement | 2 | 3 | 7.1s |
| Python 3.13 | insufficient | 1 | 2 | 9.7s |
| Aspirin safety | agreement | 4 | 5 | 12.5s |
| Water boiling | agreement | 3 | 4 | 8.6s |
| MMR vaccine | agreement | 4 | 4 | 20.3s |

---

## 5. Outputs

All evaluation graphs are in the `outputs/` folder:

- `test_results.png` — Test pass/fail summary
- `routing_agents.png` — Agent routing by query type
- `fixture_outcomes.png` — Expected fixture outcomes
- `conflict_types.png` — Conflict type distribution
- `architecture.png` — System architecture diagram
- `real_evidence_state_distribution.png` — Live mode state distribution
- `real_llm_calls_comparison.png` — LLM call comparison
- `real_latency_comparison.png` — Latency comparison
- `real_usable_agents.png` — Usable agents per query
- `expected_vs_actual.png` — Expected vs actual states

---

## 6. How to Run

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add Groq API key

# Run both servers
./run.sh

# Evaluate
python run_evaluation.py
python run_real_evaluation.py
```

---

## 7. Conclusion

Evidence-MAD successfully demonstrates evidence-triggered multi-agent debate with video fact-checking. The system achieves 100% on controlled tests and shows promising results on live queries. The RawBlock UI provides a clean, functional interface for both text and video modes.
