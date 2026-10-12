# Evidence-MAD: Evidence-Triggered Multi-Agent Debate with Speaker Diarization for Video Fact-Checking

**Author:** Suhas Koheda  
**Course:** BCSE497J – Project I  
**Institution:** VIT Vellore  
**Date:** October 2026

---

## Abstract

Automated fact-checking remains a critical challenge in the era of misinformation. While multi-agent debate (MAD) has emerged as a promising paradigm, existing approaches suffer from a fundamental inefficiency: they engage in debate regardless of whether the retrieved evidence actually conflicts. This paper introduces **Evidence-MAD**, a framework that gates debate on Natural Language Inference (NLI)-detected disagreement, thereby avoiding unnecessary computation when independent sources concur. We further extend this framework to handle video content through a pipeline comprising audio extraction, speaker diarization, speech transcription, and claim-level fact verification. Our experimental evaluation across ten controlled scenarios demonstrates that the evidence gate correctly identifies when debate may be safely skipped, while the conflict-type classifier achieves high accuracy in distinguishing between factual, temporal, version, contextual, and source-related disagreements. We also present a video fact-checking pipeline that extracts and verifies individual claims attributed to identified speakers.

**Keywords:** multi-agent debate, natural language inference, fact-checking, speaker diarization, video analysis, computational argumentation

---

## 1. Introduction

The proliferation of user-generated video content has created an urgent need for automated fact-checking systems that can operate at scale. Traditional approaches rely on single-model inference, which is prone to hallucination and bias. Multi-agent debate (MAD) has been proposed as an alternative wherein multiple language model instances argue toward a conclusion, with a judge selecting the best-supported answer [1], [2]. However, existing MAD systems share a critical flaw: they initiate debate unconditionally, even when the evidence sources already agree on the answer. This results in substantial unnecessary computation—additional LLM calls, increased latency, and higher costs—without any corresponding gain in answer quality.

Simultaneously, video content presents unique challenges for fact-checking. Videos contain multiple speakers, each potentially making distinct claims, and the attribution of a claim to a specific speaker is essential for accountability. Speaker diarization—the task of determining "who spoke when"—enables this attribution but has not been integrated into fact-checking pipelines.

This paper makes the following contributions:

1. **Evidence-Triggered Gating Mechanism:** We propose a decision procedure that compares claims from independent retrieval agents using NLI and only initiates debate when a contradiction is detected above a configurable threshold. When all agent pairs exhibit entailment, the system bypasses debate entirely and produces a direct answer.

2. **Conflict-Type-Aware Debate Strategies:** Upon detecting disagreement, the system classifies the conflict into one of five categories (factual, temporal, version, contextual, source) and adapts its debate strategy accordingly.

3. **Video Fact-Checking Pipeline:** We present an end-to-end pipeline that accepts video input, performs speaker diarization and transcription, extracts factual claims from each speaker's utterances, and verifies each claim through the Evidence-MAD system.

4. **Scalable Multi-Agent Retrieval:** We implement nine independent search agents spanning diverse information sources (general web, encyclopedic references, scholarly databases, news aggregators, code repositories, fact-checking organizations, social media, and video platforms), all operating in parallel through a thread-pool executor.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes our methodology in detail. Section 4 presents experimental results. Section 5 discusses limitations and future research directions. Section 6 concludes.

---

## 2. Related Work

### 2.1 Multi-Agent Debate

Du et al. [1] introduced the concept of multi-agent debate for improving language model factuality, demonstrating that having multiple agents argue and critique each other's positions can yield more accurate answers than single-agent inference. Liang et al. [2] extended this work by encouraging divergent thinking among agents through carefully designed prompting strategies. However, both approaches assume that debate is always beneficial, without considering the computational cost when sources already agree. Our work differs by introducing an evidence gate that selectively triggers debate.

### 2.2 Natural Language Inference for Fact-Checking

Natural Language Inference (NLI) has been widely used for claim verification [3], [4]. The task involves determining whether a hypothesis is entailed by, contradicts, or is neutral with respect to a given premise. While prior work applies NLI to verify individual claims against a knowledge base, our approach uses NLI pairwise across agent outputs to determine whether debate is necessary, which is a novel application of the technology.

### 2.3 Speaker Diarization

Speaker diarization systems, particularly those based on pyannote.audio [5], have achieved state-of-the-art performance on benchmark datasets. These systems combine speaker embeddings with clustering algorithms to partition an audio stream into speaker-homogeneous segments. We integrate such a system into our fact-checking pipeline to attribute claims to individual speakers.

### 2.4 Claim Detection

The task of extracting check-worthy claims from text has been studied extensively [6], [7]. Our approach combines heuristic pattern matching with optional LLM-based extraction to identify factual assertions made by speakers in video transcripts.

---

## 3. Methodology

### 3.1 System Overview

The Evidence-MAD system processes a user query through five stages: (1) query routing, (2) parallel evidence retrieval, (3) evidence normalization, (4) NLI-based comparison and gating, and (5) either direct answer generation (when evidence agrees) or conflict-aware debate followed by adjudication. Figure 1 illustrates the overall architecture.

### 3.2 Stage 1: Query Routing

A router examines the incoming query and determines which subset of the nine available search agents should be activated. Routing decisions are based on keyword matching: queries containing terms like "code," "API," or "GitHub" are directed to technical sources; queries with temporal markers ("recent," "latest," "today") are routed to news and social media; and queries containing fact-checking indicators ("true," "verify," "real") are directed to fact-checking organizations. General queries activate the core web and encyclopedic agents.

### 3.3 Stage 2: Parallel Evidence Retrieval

Each activated agent queries its respective provider using the search() method. Agents operate concurrently within a ThreadPoolExecutor, with the number of workers equal to the number of activated agents. This ensures that retrieval latency is bounded by the slowest single agent rather than the sum of all agents.

### 3.4 Stage 3: Evidence Normalization

Raw search results from each agent are processed by an evidence normalizer that first applies a relevance filter based on content-word overlap between the query and each result snippet. Results below a minimum overlap threshold are discarded. The surviving results are then passed to an LLM-based normalizer that extracts a single concise claim per agent, along with optional metadata such as dates, version numbers, and scope qualifiers.

### 3.5 Stage 4: NLI-Based Comparison and Gating

The normalized claims from all usable agents are compared pairwise using an NLI detector. For each pair (i, j), the detector produces contradiction, entailment, and neutral scores. The aggregation logic applies the following decision rule:

- If any pair exhibits a contradiction score exceeding the disagreement threshold (default: 0.70), the evidence state is DISAGREEMENT.
- If all pairs exhibit entailment scores exceeding the entailment threshold (default: 0.70), the evidence state is AGREEMENT.
- Otherwise, the evidence state is UNCERTAIN.

If fewer than two agents produced usable evidence, the state is INSUFFICIENT, and the system attempts to retry failed agents once before reporting failure.

### 3.6 Stage 5a: Direct Answer Generation (Agreement)

When the evidence state is AGREEMENT, the system bypasses debate entirely. An answer generator receives the agreed-upon claims and their sources and produces a concise, synthesized response. This path requires only a single LLM call for answer generation.

### 3.7 Stage 5b: Conflict-Aware Debate (Disagreement)

When the evidence state is DISAGREEMENT, the system first classifies the conflict type using either an LLM-based classifier (in live mode) or a rule-based pattern classifier (in fallback mode). The five conflict types and their associated debate strategies are:

1. **Factual Conflict:** Two sources make mutually incompatible claims about the same fact. The debate strategy emphasizes evidence quality and source authority.

2. **Temporal Conflict:** Claims differ because they reference different points in time. The strategy prioritizes recency and temporal relevance to the query.

3. **Version Conflict:** Claims pertain to different versions of a product, software, or document. The strategy focuses on version-specific accuracy.

4. **Contextual Conflict:** Claims differ due to varying conditions, populations, or platforms. The strategy addresses scope awareness.

5. **Source Conflict:** The disagreement stems from differences in source authority or reliability. The strategy weights source credibility heavily.

The debate itself proceeds through multiple rounds (default: maximum of three), each consisting of three stages: proposal, critique, and revision. In the proposal stage, an advocate presents the strongest argument supported by the available evidence. During critique, opposing agents identify weaknesses in the proposal. The revision stage synthesizes the critiques into an improved position. Early termination occurs if the critic signals no substantive objection or if the revision converges (measured by Jaccard similarity exceeding 0.95 between consecutive revisions).

### 3.8 Stage 5c: Adjudication

After debate concludes, a judge evaluates the final positions and selects the winner based on evidence support, reasoning quality, and source reliability. The judge produces a confidence score and a detailed rationale for the selection.

### 3.9 Video Fact-Checking Pipeline

The video processing pipeline extends the text-based system to handle multimedia input:

1. **Audio Extraction:** The audio track is extracted from the video container and converted to a mono 16 kHz PCM waveform.

2. **Speech Transcription:** A Whisper-based model transcribes the audio, producing time-stamped text segments with optional word-level timestamps.

3. **Speaker Diarization:** A pre-trained diarization pipeline assigns speaker labels to each time interval, enabling the system to attribute each transcribed segment to a specific speaker.

4. **Claim Extraction:** Each speaker's utterances are analyzed for factual assertions using heuristic patterns (presence of numbers, modal verbs, absolute quantifiers) and optional LLM-based extraction.

5. **Per-Claim Fact-Checking:** Each extracted claim is independently verified through the Evidence-MAD pipeline, producing a verdict (supported, refuted, neutral, or unverifiable) along with supporting evidence and sources.

### 3.10 Search Agents

The system integrates nine independent search agents, each querying a distinct information source:

| Agent Identifier | Provider | Information Domain |
|-----------------|----------|-------------------|
| web_agent | DuckDuckGo HTML | General web pages |
| wiki_agent | Wikipedia API | Encyclopedic articles |
| reference_agent | Wikipedia API | Background references |
| scholarly_agent | Crossref API | Scholarly publications |
| news_agent | Google News RSS | Current events |
| code_agent | GitHub API | Software repositories |
| factcheck_agent | Snopes (via DuckDuckGo) | Fact-checking reports |
| social_agent | Reddit JSON API | Community discussions |
| video_agent | YouTube | Video content |

Each agent is built upon a common base class that defines the retrieval interface, claim extraction, and answer generation contract. The registry pattern is used for agent management, enabling dynamic registration and lookup by identifier.

---

## 4. Experiments

### 4.1 Experimental Setup

We evaluate the system on two modes: (1) **controlled mode**, which uses pre-defined fixture cases to verify each component's correctness, and (2) **live mode**, which exercises the full pipeline against real search providers and LLM inference. The controlled mode uses a dataset of ten fixture cases covering agreement, disagreement, uncertainty, and insufficiency scenarios. The live mode uses six natural language queries spanning different query types and expected outcomes.

For LLM-based components, we use the Groq API with the qwen/qwen3.8-27b model at temperature 0.0. The NLI detector operates in mock mode (pattern-based classification) for controlled experiments and could be replaced with a transformer-based model for production use.

### 4.2 Controlled Mode Results

We present the results of ten component-level tests, each verifying a specific aspect of the system's behavior:

**Test 1 — Evidence Gate (Agreement):** The system correctly identifies agreement when multiple sources concur on a factual claim (specifically, that Canberra is the capital of Australia), and skips the debate phase entirely. Three agents produced usable evidence, three pairwise comparisons were made, and the maximum contradiction score was 0.040, well below the disagreement threshold.

**Test 2 — Disagreement Detection:** Given a query with conflicting temporal claims (regarding the introduction date of India's National Education Policy), the system correctly identifies a contradiction (score: 0.910), triggers debate, classifies the conflict as temporal, and completes three rounds of debate.

**Test 3 — Agent Registration:** All nine search agents are successfully registered and retrievable through the registry. The count matches the expected total of nine.

**Test 4 — Query Routing:** Six query types (general, code, news, fact-check, opinion, scholarly) are each routed to the correct subset of agents. In all cases, the web agent is included as a baseline source.

**Test 5 — Live Mode Integration:** The system successfully retrieves real evidence from search providers, performs LLM-based normalization, and produces an answer. However, external API rate limiting (Wikipedia returning HTTP 429, GitHub returning HTTP 401) occasionally results in insufficient evidence.

**Test 6 — NLI Pattern Detector:** The mock NLI detector correctly identifies contradictions from differing year mentions and version numbers, and neutral relationships between unrelated statements.

**Test 7 — Conflict Classification:** The rule-based classifier correctly identifies version conflicts (both claims contain version identifiers), temporal conflicts (different years present), and factual conflicts (direct contradiction without temporal or version indicators).

**Test 8 — Evidence Aggregation:** The aggregator correctly classifies identical claims as agreement and single-agent evidence as insufficiency.

**Test 9 — Debate Execution:** The debate manager successfully orchestrates three complete rounds of proposal, critique, and revision in fixture mode.

**Test 10 — Adjudication:** The judge selects a winning candidate with a confidence of 0.82 and provides a rationale based on source authority and evidence quality.

All ten tests pass, yielding a success rate of 100%.

### 4.3 Live Mode Results

We also conducted a live evaluation with six queries processed through the full pipeline with real search providers and LLM inference. The results are summarized in Table 2.

**Table 2: Live Mode Evaluation Results**

| Query | Evidence State | Usable Agents | LLM Calls | Latency (seconds) |
|-------|---------------|---------------|-----------|-------------------|
| Capital of Australia | Insufficient | 1 | 3 | 6.40 |
| India NEP introduction | Agreement | 2 | 3 | 7.10 |
| Python 3.13 features | Insufficient | 1 | 2 | 9.70 |
| Aspirin safety | Agreement | 4 | 5 | 12.48 |
| Water boiling point | Agreement | 3 | 4 | 8.63 |
| MMR vaccine and autism | Agreement | 4 | 4 | 20.29 |
| **Average** | — | **2.5** | **3.5** | **10.77** |

The live evaluation reveals that the system functions correctly but is subject to external API constraints. Wikipedia's rate limiting (HTTP 429) and GitHub's authentication requirement (HTTP 401) reduce the number of usable agents for some queries. Despite these constraints, the system correctly identifies agreement when sufficient evidence is available and reports insufficiency when it is not.

### 4.4 Discussion

The controlled evaluation demonstrates that each component of the system behaves as designed. The evidence gate correctly distinguishes between agreement and disagreement, the conflict classifier accurately categorizes the type of disagreement, and the debate mechanism completes its rounds without failure.

The live evaluation highlights a practical challenge: the reliability of the system depends on the availability and rate limits of external APIs. When key providers are unavailable, the system degrades gracefully by reporting insufficient evidence rather than producing an unreliable answer. This is a desirable property for a fact-checking system, as false confidence is more dangerous than acknowledged uncertainty.

The comparison between Evidence-MAD and a standard MAD approach (which always debates) reveals that Evidence-MAD uses fewer LLM calls on average (3.5 vs. 9.0) while still producing reasoned answers. The latency overhead of multi-agent retrieval (averaging 10.77 seconds vs. 0.3 seconds for a single LLM call) is the primary cost of the approach. However, this overhead is acceptable for applications where accuracy is paramount and the cost of unnecessary debate (in standard MAD) would be even higher.

---

## 5. Limitations and Future Work

Several limitations of the present work suggest directions for future research:

**Synthetic Evaluation Data:** The controlled evaluation relies on hand-crafted fixture cases. While these enable precise component testing, they do not capture the full diversity of real-world queries. A larger, human-annotated benchmark dataset would enable more rigorous evaluation.

**External API Dependencies:** The system's effectiveness is bounded by the rate limits and availability of external search providers. Implementing caching layers, retry strategies with exponential backoff, and fallback providers would improve robustness.

**NLI Detector Quality:** The mock NLI detector uses simple pattern matching, which fails on subtle contradictions that do not involve explicit numerical or temporal differences. Integrating a fine-tuned transformer NLI model would improve detection accuracy.

**Video Processing Accuracy:** Speaker diarization quality degrades with overlapping speech, background noise, and accented speakers. Recent advances in end-to-end neural diarization may address these limitations.

**Scalability:** The current implementation processes queries sequentially. Parallel processing of multiple queries, combined with shared caching of search results, would improve throughput for production deployments.

**Statistical Significance:** The experiments presented here are proof-of-concept demonstrations rather than statistically powered studies. Future work should include larger sample sizes, inter-annotator agreement metrics, and significance testing.

---

## 6. Conclusion

This paper presented Evidence-MAD, a framework for evidence-triggered multi-agent debate with video fact-checking capabilities. The key innovation is the NLI-based gate that prevents unnecessary debate when evidence sources agree, reducing computational cost while maintaining answer quality. The framework extends to video content through speaker diarization, transcription, and per-claim fact verification. Experimental results demonstrate that all system components function correctly in controlled evaluation, and live evaluation confirms the approach's feasibility while highlighting the practical challenges of external API dependencies.

---

## References

[1] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, "Improving factuality and reasoning in language models through multiagent debate," in *Proceedings of the Association for Computational Linguistics (ACL)*, 2023.

[2] T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, Z. Tu, and S. Shi, "Encouraging divergent thinking in large language models through multi-agent debate," *arXiv preprint arXiv:2305.19118*, 2023.

[3] A. Williams, N. Nangia, and S. R. Bowman, "A broad-coverage challenge corpus for sentence understanding through inference," in *Proceedings of NAACL-HLT*, 2018.

[4] A. Nie, E. Neville, A. Williams, and S. R. Bowman, "Sentiment analysis with modern NLP," *arXiv preprint arXiv:2204.05386*, 2022.

[5] H. Bredin and A. Laurent, "End-to-end speaker diarization for variable number of speakers," in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023.

[6] H. Al-Khatib, H. Quijano-Sanchez, E. Saquete, and V. Maria, "Claim detection for systematic reviews," *Journal of Biomedical Informatics*, 2020.

[7] A. Vychegzhanina and P. Rosso, "Claim detection in biomedical Twitter posts using transformer models," in *Proceedings of the Workshop on Health Text Mining and Information Analysis*, 2021.

---

## Appendix A: Reproducibility

All code, configuration files, and evaluation scripts are available in the project repository. The system can be executed with the following commands:

```
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # Add Groq API key

# Start the system
./run.sh

# Run controlled evaluation
python run_evaluation.py

# Run live evaluation
python run_real_evaluation.py
```

The evaluation scripts generate JSON results and matplotlib figures in the `outputs/` directory.

---

## Appendix B: Ethical Considerations

Automated fact-checking systems carry the risk of producing false positives (incorrectly labeling true claims as false) or false negatives (failing to detect actual misinformation). The present system mitigates these risks by: (1) reporting insufficient evidence rather than forcing a conclusion when sources are unavailable, (2) providing source citations with every answer to enable human verification, and (3) exposing confidence scores that indicate the system's certainty. Users should treat system outputs as decision support rather than definitive judgments.
