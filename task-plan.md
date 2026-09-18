# Evidence-MAD — Task-Wise Action Plan

Tasks are in **dependency order**. Do not skip ahead — several later tasks will silently appear to "work" if an earlier one is not done (e.g. fixing the UNCERTAIN state before fixing claim normalization will swing your debate trigger rate from 0% to ~100%).

Each task has: **Files** → **Problem** → **Action** → **Done when**.

---

# PHASE A — Make the system honest (Tasks 1–12)

Nothing measured before Phase A is complete is valid data.

---

## Task 1 — Freeze and quarantine existing results

**Files:** `results/`, new `pyproject.toml`, `requirements.txt`

**Problem:** The repo's existing traces and figures were produced by code that fabricates evidence, hides LLM failures, and hardcodes graph bars. If they stay in place you will eventually cite one. Also `requirements.txt` uses `>=` for everything, so two runs a week apart can use different `transformers` versions and produce different NLI scores.

**Action:**
1. `git checkout -b repair/evidence-mad && git tag pre-repair`
2. Move everything in `results/` to `results/archive/`, add `results/archive/INVALID.md` stating these were produced by pre-repair code.
3. Delete all generated figures.
4. Change every `>=` in `requirements.txt` to `==` with the version you currently have installed (`pip freeze` to get them).
5. Add `pyproject.toml` with:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Done when:** `results/` contains only `archive/`, and `pip install -r requirements.txt` in a fresh venv reproduces your exact environment.

---

## Task 2 — Fix LLM provider configuration

**Files:** `config.py`, `.env.example`

**Problem:** Defaults are OpenAI (`https://api.openai.com/v1`, `gpt-4o-mini`) and `.env.example` ships those values. There is **no hardcoded Gemini endpoint anywhere in the committed source** — I grepped the whole repo. Your Gemini traffic came from a stale local `.env` or a shell environment variable, so changing `.env` alone would not fix it if `LLM_BASE_URL` is exported in your shell.

Separate real bug: `self.temperature = temperature or settings.llm_temperature` in `llm_client.py` — `0.0 or 0.3` evaluates to `0.3`. **You cannot set temperature to 0.** Same `or` pattern breaks `threshold=0.0` in `NLIDetector.__init__`.

**Action:**
1. In `config.py` set Groq defaults: `llm_base_url = "https://api.groq.com/openai/v1"`, `llm_model = "llama-3.3-70b-versatile"` (or your chosen Groq model), `llm_temperature = 0.0`.
2. Add `llm_timeout: float = 60.0`, `llm_max_retries: int = 3`, `entailment_threshold: float = 0.70`, `nli_max_claims: int = 3`.
3. Add a `validate()` method that **raises** on startup if `llm_api_key` is empty and `demo_mode` is False. Call it in `MADSystem.__init__`.
4. Rewrite `.env.example` for Groq.
5. Replace every `x or settings.y` with `settings.y if x is None else x` in `llm_client.py` and `evidence/disagreement_detector.py`.
6. Run `env | grep -i "LLM_\|OPENAI\|GEMINI\|GOOGLE"` and unset anything stale. Shell env beats `.env`.

**Done when:** `python -c "from config import settings; print(settings.llm_base_url, settings.llm_temperature)"` prints the Groq URL and `0.0`, and starting the app with an empty key raises immediately instead of running.

---

## Task 3 — Make the LLM client fail loudly and count calls per role

**Files:** `llm_client.py`

**Problem:** Three silent-success paths:
- `if not self.api_key: return "No LLM available - using demo mode"` — a placeholder string returned as if it were a model response. Downstream code treats it as a valid argument.
- `chat_json` catches `JSONDecodeError` and returns `{"raw_response": response}` — a parse failure returned as a result dict.
- `self.call_count += 1` happens **after** `raise_for_status()`, so failed calls cost quota and latency but are never counted.

Also, all components share one `LLMClient`, so there is no per-role breakdown — and your efficiency claim depends on per-role call counts.

**Action:** Rewrite `LLMClient`:

```python
class LLMError(Exception): ...

class LLMClient:
    def __init__(self, ...):
        self.calls = defaultdict(lambda: {"ok": 0, "failed": 0})

    def chat(self, prompt, system_prompt=None, *, role: str) -> str:
        if not self.api_key:
            raise LLMError("No LLM API key configured")
        try:
            resp = self.client.post(...)   # retry ONLY on 429/5xx
            resp.raise_for_status()
        except Exception as e:
            self.calls[role]["failed"] += 1
            raise LLMError(f"{role}: {e}") from e
        self.calls[role]["ok"] += 1
        return resp.json()["choices"][0]["message"]["content"]

    def chat_json(self, prompt, system_prompt=None, *, role: str) -> dict:
        raw = self.chat(prompt, system_prompt, role=role)
        try:
            return json.loads(_extract_json_block(raw))
        except Exception as e:
            self.calls[role]["failed"] += 1
            raise LLMError(f"{role} returned unparseable JSON") from e

    def call_report(self) -> dict:
        return {r: dict(v) for r, v in self.calls.items()}
```

`role` is a **required keyword argument**: `"classifier" | "proposer" | "critic" | "reviser" | "judge" | "answer" | "normalize" | "relevance" | "single"`. Making it required forces you to update every call site, which is what you want.

Do **not** retry on 400 — that is a prompt bug, and retrying burns quota three times per failure.

Delete `DemoLLMClient`. It sniffs the prompt for the words "judge" and "classify" to decide what to return, which is why demo runs produce plausible-looking garbage. It is replaced in Task 22.

**Done when:** `LLMClient().chat("say ok", role="test")` returns a real Groq response; the same call with a bad key raises `LLMError`; `call_report()` shows the failed call.

---

## Task 4 — Delete async from the retrieval layer

**Files:** `agents/search_agent.py`, `agents/base_agent.py`, `agents/router.py`, `app.py`, `api.py`, `evaluation/run_experiments.py`

**Problem (root cause of "Event loop is closed"):** `SearchAgent.__init__` creates `self.client = httpx.AsyncClient(...)`. Agents are stored in the module-global `registry` and `QueryRouter.initialize()` early-returns if already initialized, so those clients live for the whole process. But every caller runs the pipeline through a **fresh** `asyncio.run(...)` — `app.py::run_interactive` does it inside a `while True` loop. `AsyncClient` binds its connection pool to the first loop that uses it; `asyncio.run` closes that loop on return; query #2 hits a pool whose loop is gone.

**Why the smart fix is deletion, not lifecycle management:** nothing downstream of the router is async. `EvidenceComparator.compare`, `ConflictClassifier.classify`, `DebateManager.run_debate`, `Judge.evaluate`, `AnswerGenerator.generate` and `LLMClient` (which uses sync `httpx.Client`) are all synchronous. The only async in the entire system is three concurrent HTTP GETs. That does not justify an async architecture, and keeping it means keeping the bug class.

**Action:**
1. `BaseAgent.retrieve` / `.search` → plain `def`.
2. `SearchAgent`: remove `self.client = httpx.AsyncClient(...)` from `__init__`. Create `httpx.Client` inside `retrieve()` in a `with` block, or use one module-level sync client. Both are loop-safe.
3. `QueryRouter.execute` → plain `def`, using threads for parallelism:
```python
with ThreadPoolExecutor(max_workers=len(agents)) as pool:
    futures = {pool.submit(a.retrieve, query): a for a in agents}
    for fut in as_completed(futures):
        agent = futures[fut]
        try:
            results.append(fut.result())
        except Exception as e:
            results.append(AgentResult.failed(agent.agent_id, reason=repr(e)))
```
4. `MADSystem.solve`, `app.solve`, `run_interactive` → sync, drop `asyncio.run`.
5. `api.py`: keep endpoints `async def`, but call the sync pipeline via `await run_in_threadpool(solve, query)` so you don't block the server loop.
6. `evaluation/run_experiments.py`: plain loops, no `asyncio.run`.
7. **Delete `AgentRegistry.retrieve_all` entirely** (`agents/base_agent.py:130`). It calls `self.logger`, which `AgentRegistry` does not define (instant `AttributeError`), and it fabricates `Evidence(answer=f"Error: {result}", claims=[], sources=[])` for failed agents — a failure rendered as a valid evidence object. It is dead code.
8. Fix the mangled header of `search_agent.py` — the module docstring currently swallows two `import` lines (`"""\nimport html\nimport re\nSearch agent...\n"""`).

**Done when:** you run 5 queries back-to-back in one interactive session with zero `Event loop is closed`.

---

## Task 5 — Remove cross-provider fallbacks

**Files:** `agents/search_agent.py`

**Problem:** `_wikipedia_search` falls back to `_duckduckgo_search` on any exception. `_crossref_search` does the same. `_google_search` falls back to `self._demo_data`. So under load or rate-limiting, all three "independent" agents silently become the same DuckDuckGo agent — and three copies of one source agreeing looks exactly like three independent sources agreeing. This corrupts the central mechanism of the project.

**Action:** Delete every cross-provider fallback. A provider failure is a **failure of that agent**, returned as `status=RETRIEVAL_FAILED` with the reason. Keep retries *within* the same provider (2 attempts, exponential backoff, only on timeout/5xx/429).

**Done when:** blocking `en.wikipedia.org` in `/etc/hosts` makes the reference agent report a failure instead of returning DuckDuckGo results.

---

## Task 6 — Delete fabricated evidence from `app.py`

**Files:** `app.py`

**Problem:** This is the single most damaging function in the repo.

```python
if len(evidence_list) < 2:
    evidence_list = self._create_default_evidence(query)
```

`_create_default_evidence` builds one `Evidence` containing the literal claim `f"Default claim for {query}"` and **returns `[evidence, evidence]` — the same object twice**. The comparator then compares it against itself, which is an automatic entailment.

So your observed sequence — total retrieval failure → "evidence agrees" → "skipping debate" → confident answer citing 0 sources — is not an emergent bug. It is this function behaving as written.

**Action:** Delete `_create_default_evidence` and its call site. Zero or one usable agent is an `INSUFFICIENT` state (Task 9), never a comparison.

**Done when:** `grep -rn "_create_default_evidence" .` returns nothing, and a run with all agents blocked reports insufficient evidence instead of an answer.

---

## Task 7 — Fix the NLI detector

**Files:** `evidence/disagreement_detector.py`

**Problem:** Three separate bugs, any one of which alone would explain your `contradiction=0.000, entailment=0.000` observations.

1. **Wrong input format.** `input_text = f"{premise} </s> {hypothesis}"` passed as a single string to a `text-classification` pipeline. A sequence-pair NLI model needs two segments; a manually inserted `</s>` is tokenized as literal text and produces the wrong segment layout. The model is scoring nonsense.
2. **Failures look like neutral.** The `except` block returns `{"label": NEUTRAL, "score": 0.0}`. A crashed model is indistinguishable from a genuine neutral result — and under the current gate both mean "agree."
3. **`MockNLIDetector` fakes agreement.** `if len(set(premise.split()) & set(hypothesis.split())) > 3: return ENTAILMENT @ 0.85`. Any two English sentences sharing four tokens ("the", "is", "of", "in") are labelled entailment. **Every demo-mode agreement in your current numbers is potentially this heuristic.**

**Action:**
1. `results = self._pipeline({"text": premise, "text_pair": hypothesis})`
2. `top_k=None` instead of the deprecated `return_all_scores=True`.
3. Define `class NLIError(Exception)` and **raise** it on failure. The aggregator (Task 9) catches it and marks that pair as unresolved, which is visible in the trace.
4. Rewrite `MockNLIDetector` as an exact-match fixture lookup keyed on claim text, raising `KeyError` on an unknown pair. Delete the word-overlap heuristic entirely.
5. Run both directions (`predict(a,b)` and `predict(b,a)`) and take the stronger contradiction. NLI is not symmetric and a one-directional check misses roughly half of real contradictions.

**Done when:** a unit test with a known contradiction pair ("The policy was introduced in 2019." / "The policy was introduced in 2020.") returns `CONTRADICTION` above 0.7 from the real model.

---

## Task 8 — Structured, relevance-filtered, normalized evidence

**Files:** `evidence/models.py`, `agents/search_agent.py`, new `evidence/normalizer.py`

**Problem (two parts):**

*Irrelevance:* `retrieve()` wraps every search hit into a `Source` with no relevance check, then `BaseAgent.extract_claims` regex-splits the concatenated text on `[.!?]+` and keeps any fragment over 20 characters. That is how a Python-version question produced evidence about Ubuntu, Minecraft, and Africa.

*Cost and noise:* `extract_claims` keeps up to 10 fragments per agent. The comparator loops over every pair, so that is **10 × 10 = 100 NLI forward passes per agent pair, 300 per query** with three agents — `roberta-large` on CPU. It is most of your latency, and the resulting scores are relationships between arbitrary sentence fragments, not between the agents' answers.

**Action:**

1. Add to `evidence/models.py`:
```python
class EvidenceStatus(str, Enum):
    OK = "ok"
    NO_RELEVANT_RESULTS = "no_relevant_results"
    RETRIEVAL_FAILED = "retrieval_failed"

class NormalizedClaim(BaseModel):
    text: str                      # one sentence that answers the query
    source: Source                 # provenance preserved
    date: Optional[str] = None     # enables temporal reasoning
    version: Optional[str] = None  # enables version reasoning
    scope: Optional[str] = None    # platform / population / edition

class AgentEvidence(BaseModel):
    agent_id: str
    query: str
    status: EvidenceStatus
    failure_reason: Optional[str] = None
    claims: List[NormalizedClaim] = []
    sources: List[Source] = []
    raw_results_count: int = 0
    filtered_out_count: int = 0

    @property
    def is_usable(self) -> bool:
        return self.status is EvidenceStatus.OK and bool(self.claims)
```

2. **Relevance filter**, cheap stage first: content-word overlap between query and `title + snippet`; below threshold → drop and increment `filtered_out_count`. This alone removes the Minecraft/Africa noise. Optionally add one LLM pass (`role="relevance"`) behind a config flag so you can ablate it.

3. **Normalization**: one LLM call per agent (`role="normalize"`) turning surviving snippets into **one** `NormalizedClaim` answering the query, plus extracted `date`/`version`/`scope` and the URL it came from. Instruct it to return `{"answer": null}` if the snippets don't answer the question → `NO_RELEVANT_RESULTS`.

**Why this is the smartest single change in the plan:** +3 LLM calls per query buys you (a) 300 NLI passes → at most 9, (b) a comparison between actual answers instead of random fragments, (c) the `date`/`version`/`scope` attributes that Tasks 18–20 need, and (d) source provenance attached to each claim. Four problems, one change.

Keep the regex `extract_claims` as a no-LLM path for offline tests only — never for experiments.

**Done when:** a Python-version query produces three `AgentEvidence` objects each with exactly one claim about Python versions, and `filtered_out_count > 0` on a noisy query.

---

## Task 9 — N-agent evidence decision with a real UNCERTAIN state

**Files:** new `evidence/aggregator.py`, `evidence/comparator.py`, `app.py`

**Problem (three bugs that must be fixed together):**

1. **No UNCERTAIN state.** `EvidenceComparison` has only `has_disagreement: bool`. `app.py` branches on `if not comparison.has_disagreement:`, so `contradiction=0.0, entailment=0.0` — meaning "unrelated" or "NLI crashed" — takes the AGREE branch.
2. **Only two agents compared.** `app.py` literally does `evidence_a = evidence_list[0]; evidence_b = evidence_list[1]`. Agent C retrieves and is discarded.
3. **`max()` over a cross-product.** Once Task 8 is *not* done, `max()` over 100 noisy fragment pairs is an extreme-value estimator — almost any query will contain one pair scoring >0.70. **If you fix bug 1 without Task 8, your trigger rate flips from ~0% to ~100% and your efficiency claim disappears.** This is why Task 8 comes first.

**Action:** New `evidence/aggregator.py`:

```python
class EvidenceState(str, Enum):
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    UNCERTAIN = "uncertain"
    INSUFFICIENT = "insufficient"

class EvidenceDecision(BaseModel):
    state: EvidenceState
    pairwise: List[PairwiseRelation]        # A-B, A-C, B-C
    usable_agents: List[str]
    failed_agents: List[Dict[str, str]]     # id -> reason; never dropped
    top_conflict: Optional[Dict[str, Any]]  # highest-contradiction pair
    rationale: str
```

The decision rule, written **once, here** — not in `app.py`:

```
usable = [e for e in agent_evidence if e.is_usable]
if len(usable) <= 1:  -> INSUFFICIENT      # this is the case that used to hit
                                           # _create_default_evidence
for each unordered pair in usable: relation = nli(claim_i, claim_j)

if any pair contradiction >= DISAGREEMENT_THRESHOLD  -> DISAGREEMENT
elif all pairs entailment  >= ENTAILMENT_THRESHOLD   -> AGREEMENT
else                                                  -> UNCERTAIN
```

The asymmetry is deliberate and is the core correctness fix: **disagreement needs one pair, agreement needs all pairs.** Two agents entailing plus one neutral is UNCERTAIN, not agreement.

In `app.py`: delete the `[0]`/`[1]` indexing and branch four ways on `decision.state`. Every downstream signature (`compare`, `classify`, `run_debate`, `evaluate`, `generate`) changes from `evidence_a, evidence_b` to `List[AgentEvidence]` — plan for a signature change across five modules, not a one-line edit.

**Done when:** a unit test asserts `contradiction=0.0, entailment=0.0` → `UNCERTAIN` (never `AGREEMENT`), and a 3-agent run shows all three pairwise relations in the trace.

---

## Task 10 — Recovery and insufficient-evidence handling

**Files:** `app.py`, `evidence/aggregator.py`

**Problem:** There is currently no `INSUFFICIENT` path at all — failure falls into fabricated evidence (Task 6).

**Action:**
- On `INSUFFICIENT`: retry **only** the failed/empty agents, **once**. If still insufficient, produce an explicit insufficient-evidence answer naming which agents failed and why.
- On `UNCERTAIN`: **do not debate.** Debating unrelated claims is exactly what the gate exists to prevent. Report that evidence neither agreed nor conflicted. (Make "debate on UNCERTAIN" an ablation in Task 26 if you want to measure it.)
- Cap recovery at one attempt so latency stays bounded and measurable.

**Done when:** all three agents blocked → `INSUFFICIENT`, one retry visible in the trace, no answer content produced.

---

## Task 11 — Honest conflict classification

**Files:** `conflict/classifier.py`, `app.py`

**Problem (three bugs):**

1. **Silent fallback.** `_classify_by_llm`'s `except` returns `self._classify_by_patterns(...)` with no status marker. The output is structurally identical to a successful LLM classification. This is exactly your observed "LLM failure, then `conflict type: temporal`."
2. **The pattern classifier is nearly a constant function.** Its CONTEXTUAL check fires on `but`, `however`, `may`, `will`, `must`, `should`, `often`, `always`, `never` — present in most English sentences — and it is checked *before* SOURCE. Nearly everything not year-mismatched or version-mismatched classifies as CONTEXTUAL.
3. **The classifier and the `Conflict` record describe different claim pairs.** `app.py` passes `comparison.conflicts[0]["claim_a"]` to the classifier but builds the `Conflict` object from `evidence_a.claims[0]`. Meanwhile `find_most_contradictory_pair()` exists in `comparator.py` to select the right pair and is **never called**.

**Action:**
1. Add `class ComponentStatus(str, Enum): LLM_SUCCESS / FALLBACK / FAILED`.
2. `classify()` returns `ClassificationResult(type, confidence, explanation, status, error)`. On LLM exception → `status=FAILED`; only then, and only if `settings.allow_rule_fallback`, run patterns and return `status=FALLBACK`. Never `LLM_SUCCESS`.
3. Tighten `_classify_by_patterns`: remove modal verbs and discourse connectives from the contextual list; reorder checks version → temporal → scope → source → factual.
4. Guard the entry: assert `decision.state is DISAGREEMENT` before classifying.
5. Use `decision.top_conflict` (from Task 9) for **both** the classifier input and the `Conflict` object.

**Done when:** a forced LLM failure produces `status=FALLBACK` in the trace and Task 27's accuracy metric excludes it from the LLM_SUCCESS numerator.

---

## Task 12 — Fix the broken `Conflict` model contract

**Files:** `app.py`, `evidence/models.py`

**Problem:** `Conflict.claim_a` / `claim_b` are typed `Claim` (required, non-optional), but `app.py` passes `evidence_a.claims[0] if evidence_a.claims else None`. Passing `None` raises `ValidationError`, which the broad `try/except` in `solve()` swallows into `trace.error`. Any query where an agent returned zero claims dies with an opaque error message.

**Action:** After Task 9, a `Conflict` can only be constructed on the `DISAGREEMENT` path, where `top_conflict` guarantees two real claims. Change the type to `NormalizedClaim` and keep it required — the guarantee is now structural. Also narrow the `try/except Exception` in `solve()` so `ValidationError` surfaces instead of being logged as a generic failure.

**Done when:** no query produces a bare `trace.error` string; every failure path has a named state.

---

# PHASE B — Make the debate real (Tasks 13–20)

---

## Task 13 — Stop hiding LLM failures in the debate components

**Files:** `debate/proposer.py:239`, `debate/critic.py:198`, `debate/reviser.py:240`, `debate/judge.py:249`

**Problem:** All four have `except Exception: return self._demo_*(...)`. The returned `DebateArgument`/`JudgeResult` is structurally identical to a real one — no flag, no marker. `DebateManager` then increments `total_llm_calls` regardless and logs `"Round N completed"`, ending with `"Debate completed: 3 rounds"`.

That is precisely your observed run: five 400 errors followed by a clean three-round debate and a judge verdict. `_demo_evaluate` even hardcodes `winner`, `evidence_score`, and `confidence`, so a judge that never ran produces a confident numeric decision. **The current traces cannot distinguish a real debate from a total failure.**

**Action:** Remove the `except → _demo_*` path from all four files. Let `LLMError` propagate to the manager. Keep the `_demo_*` methods but call them **only** when `run_mode is FIXTURE`, never as an error handler.

If you keep only one rule from this entire document, keep this one.

**Done when:** `grep -n "except Exception" debate/*.py conflict/*.py` shows no handler that returns a demo/fallback value.

---

## Task 14 — Validate each debate stage and track outcome status

**Files:** `debate/manager.py`

**Problem:** `run_debate` calls proposer → critic → reviser with no check that any returned content. It appends a `DebateRound` unconditionally and always runs the full `max_rounds`.

**Action:**
```python
class DebateOutcome(BaseModel):
    rounds: List[DebateRound]
    status: Literal["completed", "partial", "failed"]
    stop_reason: Literal["converged", "no_objection", "max_rounds", "stage_failure"]
    failed_stages: List[str]
```
Per round: propose → validate → critique → validate → revise → validate. On a stage failure, **stop the loop immediately**, set `partial` (≥1 complete round) or `failed` (round 1 failed), record the stage. Do not log "Round N completed" for an incomplete round.

Shared validator:
```python
def is_valid_argument(arg) -> bool:
    return arg is not None and len(arg.content.strip()) >= 50 \
           and not arg.content.startswith("Demo response")
```

**Done when:** a mock client that raises on the critic yields `status="partial"`, `failed_stages=["critic"]`, and 1 round — not 3.

---

## Task 15 — Judge only judges a real debate

**Files:** `debate/judge.py`, `app.py`

**Problem:** Two `_demo_evaluate` fallbacks (one on unparseable JSON, one on exception) manufacture a confident verdict. The judge also receives only the transcript, not the sources or conflict type in structured form.

**Action:**
- Require `outcome.status == "completed"` (or at minimum a valid final revision). If partial/failed: **do not call the judge**, set `trace.judge_result = None`, and have the final answer state that the debate did not complete.
- Pass the judge the candidate arguments, their `NormalizedClaim`s with sources, and the conflict type.
- Remove both fallbacks. Judge failure = `judge_result = None`, recorded explicitly.

**Done when:** a forced judge failure produces `judge_result is None` and a final answer that does not claim the conflict was resolved.

---

## Task 16 — Keep and verify the conflict-specific strategies

**Files:** `debate/proposer.py` (`CONFLICT_STRATEGIES`), `debate/critic.py`, `debate/reviser.py`

**Problem:** None — this is the strongest part of the repo. The risk is that you break it while fixing everything around it, or that nobody ever verifies the selected strategy actually reaches the prompt.

**Action:** Do not touch `CONFLICT_STRATEGIES` or the Propose → Critique → Revise flow. Add one thing: record `strategy_used` and the **first 200 characters of the actual strategy text sent** on each `DebateArgument`, so a test can assert the temporal strategy text appears in a temporal debate's prompt. Add `ConflictType.UNKNOWN → generic strategy` — you need it for the Standard-MAD baseline in Task 25.

**Done when:** Task 23's tests assert strategy text reaches the proposer for all five conflict types.

---

## Task 17 — Fix routing and agent independence

**Files:** `agents/router.py`, `agents/search_agent.py`

**Problem:**
- `route()` does `agents = registry.list_ids()` with a hardcoded `confidence = 0.9` and the comment "Could add more sophisticated routing here." It returns the whole pool, so Crossref runs on every query.
- **`query_suffix` breaks independence.** Agent A appends `"general web sources"`, B appends `"official government and primary sources"`, C appends `"academic historical timeline sources"`. The three agents are therefore **asking different questions**, which is a direct contributor to the irrelevant-results problem. Independence means different sources for the same question.
- `initialize()` registers agent C only when `not demo_mode`, so **demo mode runs 2 agents and real mode runs 3**. Every demo-vs-real comparison is already invalid.

**Action:**
1. Three fixed profiles: `WEB` (DuckDuckGo), `REFERENCE` (Wikipedia), and a third general source (second web provider with a different provider, or Brave/SerpAPI if you have a key).
2. `SCHOLARLY` (Crossref) replaces the third slot **only** for research-flavoured queries, matched against a small explicit keyword list (`study`, `trial`, `meta-analysis`, `evidence for`, `research on`, DOI/journal mentions).
3. **Delete `query_suffix` entirely.** Every agent gets the query verbatim.
4. Register the **same three agents** in both modes.
5. Delete `RoutingDecision.confidence` (hardcoded) and the unused `ToolType` members (`RAG`, `CALCULATOR`, `DATABASE`), or move them under an explicit `# future work` comment. Record the routing reason in the trace.

**Done when:** "What is the capital of Australia?" routes to web/reference/web (no Crossref); "meta-analysis of statin efficacy" routes to scholarly; all three agents log the identical query string.

---

## Task 18 — Temporal conflicts respect the requested time

**Files:** `debate/proposer.py` (temporal strategy), `evidence/normalizer.py`

**Problem:** The current temporal handling only detects two different years in text (`_classify_by_patterns`) and the strategy prompt says "the more recent information should be weighted higher." For "What was the latest stable Python version in 2024?" that is wrong — it picks the newest source regardless of what was asked.

**Action:** Task 8 already extracts `date` per claim. Add query-time extraction (an explicit year, "as of X", "currently", "in 2024"). The temporal strategy prompt must receive: the query's requested period, each claim's date, and an instruction to discard claims describing a period **after** the requested one.

**Done when:** a fixture case asking about 2024 resolves to the 2024 answer even when a 2026 source is present.

---

## Task 19 — Version conflicts preserve version scope

**Files:** `debate/proposer.py` (version strategy), `evidence/normalizer.py`

**Problem:** Two claims about different releases look like a contradiction to NLI. Nothing currently carries version metadata through the pipeline, so the debate cannot tell "one source is wrong" from "these describe different versions."

**Action:** Use the `version` field from Task 8. The version strategy must first ask whether the disagreement is **explained by** a version difference, and only declare a source wrong if both claims target the same version. If the query does not specify a version, the answer should state the version each claim applies to rather than picking one.

**Done when:** a fixture where Python 3.13 and 3.14 claims conflict resolves as "different versions, both correct" rather than picking a winner.

---

## Task 20 — Contextual conflicts preserve scope

**Files:** `debate/proposer.py` (contextual strategy), `evidence/normalizer.py`

**Problem:** Same shape as Task 19 — platform/population/edition differences are lost before the debate sees them.

**Action:** Use the `scope` field from Task 8. The contextual strategy must identify the conditions under which each claim holds and check whether the contradiction dissolves under scope separation, before any "one side is wrong" reasoning.

**Done when:** the adults/children fixture resolves as "both correct under different conditions."

---

# PHASE C — Output, modes, and tests (Tasks 21–24)

---

## Task 21 — Rewrite final answer generation

**Files:** `output/generator.py`

**Problem:** `_generate_direct` emits `"Based on evidence from N sources:"` followed by raw claim bullets — it concatenates whatever was retrieved. It also hardcodes the reasoning string `"Evidence from both agents was consistent"` regardless of what happened. This is how a 0-source run produced a confident-looking answer.

There is a second, less obvious cost: this snippet dump **inflates your accuracy metric** (see Task 27), because token-recall scoring rewards dumping text.

**Action:** Drive output from `decision.state`:
- `AGREEMENT` → one LLM call (`role="answer"`) writing a direct answer from the agreed claims, then supporting sources. **Delete the bullet dump.**
- `DISAGREEMENT` → conflict type, judge resolution, winning claim, backing sources.
- `UNCERTAIN` → "evidence neither clearly agreed nor conflicted," plus what each agent found.
- `INSUFFICIENT` → explicit statement plus which agents failed and why. No answer content.

Report **`independent_agents_agreeing` separately from `source_count`** — nine URLs behind two agreeing agents is a 2, not a 9. Put both in `trace.metrics` and the displayed output.

**Done when:** the four states produce four visibly different output shapes, and no state can produce a confident answer without usable evidence.

---

## Task 22 — Separate fixture mode from live mode properly

**Files:** `config.py`, `app.py`, all component `__init__`s, new `data/fixtures.json`

**Problem:** `demo_mode` is read independently by eight different `__init__`s via `settings.demo_mode`, then hand-patched in `MADSystem.__init__` (`self.debate_manager.proposer.demo_mode = True`, etc.). Components can end up in mixed states. Meanwhile `SearchAgent.search`'s demo branch matches on keywords (`"policy"`, `"version"`, `"coffee"`, `"blog"`) with a **catch-all fallback returning canned Python text** — which is how an unrelated question gets nonsense evidence.

**Action:**
1. One `RunMode` enum (`FIXTURE` / `LIVE`), passed explicitly into every constructor. No component reads `settings.demo_mode` on its own. In FIXTURE mode, retrieval **and** NLI **and** LLM all come from fixtures; in LIVE, all three are real. No mixing.
2. Replace the keyword buckets with `data/fixtures.json`:
```json
{"cases": [
  {"id": "agree_capital", "query": "What is the capital of Australia?",
   "agents": {"web": {"status":"ok","claim":"The capital of Australia is Canberra.","source":{...}},
              "reference": {...}, "scholarly": {...}},
   "expected_state": "agreement", "expected_conflict_type": null, "expected_debate": false}
]}
```
3. Required cases: `agree_capital`, `factual_boiling`, `temporal_policy`, `version_python`, `contextual_drug`, `source_blog_vs_paper`, `unrelated_evidence` (→ uncertain), `one_agent_fails`, `two_agents_fail`, `all_agents_fail`.
4. **An unknown query in FIXTURE mode raises.** No default bucket — silence is what produced the Minecraft evidence.

**Done when:** `DEMO_MODE=true` with an unlisted query raises a clear error instead of returning canned data.

---

## Task 23 — Deterministic test suite

**Files:** `tests/`

**Problem:** The three existing test files do not cover the failure modes that actually occurred in production runs.

**Action:** One file per concern. Read the existing tests first and fold in reusable assertions rather than deleting wholesale.

| File | Must assert |
|---|---|
| `test_aggregator.py` | 3 agree → AGREEMENT; one contradicting pair → DISAGREEMENT; all neutral → UNCERTAIN; 0 or 1 usable → INSUFFICIENT; **explicitly: contradiction=0 AND entailment=0 must NOT be AGREEMENT** |
| `test_conflict_types.py` | Five parametrised cases: correct state, correct type, **and the matching `CONFLICT_STRATEGIES` text reaches the proposer prompt** |
| `test_retrieval_failure.py` | 1/2/3 agents failing; failed agents appear in `decision.failed_agents` with reasons; no fabricated evidence anywhere in the trace |
| `test_llm_failure.py` | Mock client raising on proposer/critic/reviser/judge; `status != "completed"`; `judge_result is None`; trace never reports 3 successful rounds |
| `test_relevance.py` | Unrelated snippets filtered; agent with only unrelated hits → `NO_RELEVANT_RESULTS` |
| `test_event_loop.py` | `solve()` 5× in one process, no exception (regression test for Task 4) |
| `test_baselines.py` | Single-LLM trace has `agents == []` and exactly 1 LLM call; Standard-MAD always debates; Proposed gates |

**Done when:** `pytest` is green and each of the four production failure modes (event loop, fabricated evidence, fake agreement, fake debate) has a test that would have caught it.

---

## Task 24 — Clean up LLM call tracking

**Files:** `app.py`, `debate/manager.py`, `evaluation/metrics.py`

**Problem:**
```python
def _count_llm_calls(self):
    total = 0
    if self.llm_client: total += self.llm_client.call_count
    if self.debate_manager: total += 0    # dead
    if self.judge: total += 0             # dead
```
Confirmed dead bookkeeping. Combined with Task 3's bug (failures not counted), your call counts are wrong in both directions.

**Action:** Delete `_count_llm_calls`. Use `llm_client.call_report()` from Task 3 directly. Record in `trace.metrics`: per-role `ok`/`failed` counts, `total_llm_calls`, and `llm_calls_by_stage`. Since your efficiency claim is "gating avoids debate calls," report calls **split by whether debate was triggered**.

**Done when:** an agreement-path run and a debate-path run show clearly different per-role breakdowns in the trace.

---

# PHASE D — Valid experiments (Tasks 25–29)

---

## Task 25 — Build genuinely independent baselines

**Files:** `evaluation/run_experiments.py`, new `evaluation/systems/`

**Problem (the most serious research-validity issue):** `run_single_llm_baseline` calls `MADSystem.solve()` — the **full** pipeline with retrieval, NLI, classification, debate, and judge — and then **deletes the evidence of it from the trace**:
```python
result = await system.solve(sample.query)
result.debate_triggered = False; result.conflict = None
result.debate_rounds = []; result.judge_result = None
result.metrics["system"] = "single_llm"
```
`run_standard_mad_baseline` does the inverse and, when no conflict was found, **fabricates one**: `Conflict(..., claim_a=Claim(text="Mock claim A"), ...)`.

All three "systems" are the same code path. Latency, LLM calls, and debate counts are identical **by construction**, and every reported difference is the label edits.

**Action:** A common interface, three separate implementations.

- **`SingleLLMSystem`** — its own file, ~40 lines, importing **nothing** from `agents/`, `evidence/`, `conflict/`, `debate/`:
```python
def run(self, query):
    t0 = time.time()
    answer = self.llm.chat(ANSWER_PROMPT.format(query=query), role="single")
    return ExecutionTrace(query=query, agents=[],
        final_answer=FinalAnswer(answer=answer, sources=[]),
        metrics={"system":"single_llm","llm_calls":1,
                 "latency_seconds":time.time()-t0,"debate_triggered":False})
```
- **`StandardMADSystem`** — real retrieval and normalization, **no gate** (always debate), **one generic strategy** (`ConflictType.UNKNOWN` from Task 16), real judge. Same debate code, same round cap.
- **`ProposedMADSystem`** — gate + conflict-type-aware strategy. The only system with both.

**Enforce the separation with a test**: import the single-LLM module in a subprocess and assert `"debate" not in sys.modules`. Without this, the coupling creeps back.

**Done when:** the three systems produce visibly different `llm_calls`, `latency`, and `debate_triggered` on the same query, with no post-hoc label editing anywhere.

---

## Task 26 — Add ablations

**Files:** `evaluation/systems/`

**Problem:** With only three systems you can show Proposed beats the baselines, but not which of your two novelties did the work.

**Action:** Three ablations, which fall out almost free from Task 25:
- `no_gate` — Proposed minus the gate (= typed strategies, always debate)
- `no_typed_strategy` — Proposed minus typed strategies (= gate + generic prompt)
- `no_relevance_filter` — Proposed with Task 8's filter disabled

That gives a clean 2×2 plus Single-LLM — exactly the table a reviewer asks for.

**Done when:** five to six systems run over the same dataset from one command.

---

## Task 27 — Replace the accuracy metric

**Files:** `evaluation/metrics.py`

**Problem:**
```python
if len(expected_tokens & actual_tokens) / len(expected_tokens) >= 0.5: answer_correct += 1
```
Unweighted token recall with **no precision term**. A system that pastes five whole search snippets scores higher than one that answers correctly in a sentence. It cannot distinguish a correct answer from a verbose wrong one — and it structurally favours your no-debate path (which dumps claims) over your debate path (which returns prose). This is a confound pointing in the direction of your own hypothesis, which is the worst kind.

**Action:** Two metrics, report both:
1. **Primary — LLM-as-judge grading** against the reference answer, fixed prompt, `temperature=0`, cached. Report inter-run agreement on a 20-sample subset so you can defend it.
2. **Check — key-fact matching.** Each dataset sample carries `required_facts: List[str]` and optional `forbidden_facts`. Correct = all required present, none forbidden. Deterministic, no verbosity bonus.

**Done when:** a deliberately verbose wrong answer scores lower than a concise correct one under both metrics.

---

## Task 28 — Honest metrics and figures

**Files:** `evaluation/metrics.py`, `evaluation/run_experiments.py`

**Problem:** `generate_graphs` contains hardcoded bars:
```python
debate_rates = [0, 100, metrics["proposed_mad"].get("debate_trigger_rate", 0) * 100]
```
Two of three bars in that figure are literals, not measurements. Any figure you have already produced from this is fabricated.

**Action:**
1. Delete every hardcoded value in `generate_graphs`. If a number is not measured, the bar does not exist.
2. Add the metrics the fixed trace now supports: `debate_trigger_rate`, `unnecessary_debate_rate` (debated where `expected_conflict=False`), `missed_conflict_rate`, `conflict_classification_accuracy` **computed only over `status == LLM_SUCCESS`** (report FALLBACK separately), `retrieval_failure_rate`, `irrelevant_evidence_rate` (from `filtered_out_count`), per-role LLM calls, `avg_debate_rounds`, `latency_p50/p95`, `insufficient_evidence_rate`.
3. Headline efficiency number: **debates avoided × (LLM calls + latency saved per debate)**.

**Done when:** every number in every figure traces back to a field in a saved trace file.

---

## Task 29 — Rebuild the evaluation dataset

**Files:** `evaluation/dataset.py`, `data/`

**Problem:** The labels are wrong. `"When was the first iPhone released?"` is labelled `expected_conflict=True, temporal` — but it has one uncontested answer, so real retrieval will correctly agree and your gate gets **scored wrong for being right**. Several other samples have the same issue. Bad labels will make a correct system look broken.

**Action:**
1. Re-label every sample by actually checking what three independent sources say today. Delete samples where the label cannot be justified.
2. Target 60–100 samples: roughly half with genuine live disagreement (with a labelled conflict type), half with genuine agreement.
3. Build controlled-conflict cases from questions where sources genuinely differ — superseded statistics, pre/post-update documentation, contested figures, version-scoped behaviour.
4. Add `required_facts` / `forbidden_facts` per sample for Task 27.
5. **Do not touch `DISAGREEMENT_THRESHOLD` to make debates happen.** It was already working — your one successful conflict run scored 0.890 against a 0.70 threshold. The problem was never the threshold.

**Done when:** every label is defensible from sources you can cite.

---

# PHASE E — Optimisation, only after everything above (Tasks 30–31)

---

## Task 30 — Early stopping

**Files:** `debate/manager.py`

**Problem:** The loop always runs `max_rounds` with no convergence check. Doing this before Phase A–D would be optimising a pipeline that produces invalid results.

**Action:** Stop when any holds:
- the reviser's output is ≥0.95 similar (Jaccard or cosine) to the previous round's revision → `converged`
- the critic returns `{"substantive_objection": false}` (add this field to the critic's output schema) → `no_objection`
- otherwise → `max_rounds`

Record `stop_reason` and `rounds_used`. Keep `MAX_DEBATE_ROUNDS=3` as the cap. Report LLM-call savings against the cap.

**Done when:** an easy conflict resolves in 1 round with `stop_reason="converged"`, a hard one still uses 3.

---

## Task 31 — Run the experiments

**Files:** `evaluation/`

**Action, in this order:**
1. FIXTURE mode end-to-end; all Task 23 tests green.
2. LIVE on 10 samples. **Read all 10 traces by hand.** Confirm: no fabricated evidence, every failure visible, `LLM_SUCCESS` where expected.
3. Full grid: 5–6 systems × N samples × 3 seeds. Save every trace.
4. Threshold sweep as a **separate** experiment on Proposed only: `DISAGREEMENT_THRESHOLD ∈ {0.5, 0.6, 0.7, 0.8, 0.9}`, plotting trigger rate against accuracy. This is legitimate sensitivity analysis and pre-empts the obvious reviewer question.

**Done when:** every reported number comes from a trace file produced after Task 29.

---

# Critical ordering constraints

These are the dependencies where getting the order wrong wastes days:

| Do not do | Before | Because |
|---|---|---|
| Task 9 (UNCERTAIN state) | Task 8 (claim normalization) | `max()` over 100 fragment pairs will flip your trigger rate to ~100% and destroy the efficiency claim |
| Task 30 (early stopping) | Tasks 13–15 | You would be optimising a debate that silently fails |
| Task 31 (experiments) | Task 29 (dataset relabel) | Wrong labels make a correct system look broken |
| Any measurement | Task 13 (no fallbacks) | Failed runs currently look identical to successful ones |
| Task 25 (baselines) | Tasks 1–15 | Baselines over a broken pipeline measure nothing |

# Things not to do

- Do not lower `DISAGREEMENT_THRESHOLD` to force debates. Build conflicting cases instead.
- Do not add LangChain / AutoGen / CrewAI. These are lifecycle and control-flow bugs; a framework adds a second lifecycle to debug.
- Do not keep any `except: return self._demo_*()` anywhere.
- Do not reuse any number from the current `results/` or any existing figure.
- Do not delete `CONFLICT_STRATEGIES`, the Propose → Critique → Revise flow, the module layout, or the tracing model. Those parts work.

# If you run out of time

Stop after **Task 15**. At that point the system is honest: no fabricated evidence, no fake agreement, no fake debates, no hidden failures. The results will be real even if the baselines are still missing — which is a defensible position. Stopping anywhere before Task 13 is not, because the traces will still be indistinguishable from failure.
