"""
Run comprehensive evaluations on the Evidence-MAD system.
Tests all major components and outputs results + graphs for research paper.
"""
import json
import time
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

from app import solve
from agents.router import router
from agents.base_agent import registry


def test_evidence_gate():
    logger.info("TEST 1: Evidence Gate (Agreement -> Skip Debate)")
    trace = solve("What is the capital of Australia?", demo_mode=True)
    results = {
        "query": trace.query,
        "evidence_state": trace.evidence_decision.state.value if trace.evidence_decision else "unknown",
        "debate_triggered": trace.debate_triggered,
        "passed": not trace.debate_triggered and trace.evidence_decision.state.value == "agreement",
        "latency_seconds": trace.metrics.get("latency_seconds", 0),
        "llm_calls": trace.metrics.get("total_llm_calls_ok", 0),
    }
    logger.info("  Query: {} | State: {} | Debate: {} | Passed: {}",
                results["query"], results["evidence_state"], results["debate_triggered"], results["passed"])
    return results


def test_disagreement_triggers_debate():
    logger.info("TEST 2: Disagreement -> Debate Triggered")
    trace = solve("When was India's National Education Policy introduced?", demo_mode=True)
    results = {
        "query": trace.query,
        "evidence_state": trace.evidence_decision.state.value if trace.evidence_decision else "unknown",
        "debate_triggered": trace.debate_triggered,
        "conflict_type": trace.conflict.type.value if trace.conflict else None,
        "debate_rounds": trace.debate_rounds,
        "passed": trace.debate_triggered and trace.evidence_decision.state.value == "disagreement",
        "latency_seconds": trace.metrics.get("latency_seconds", 0),
        "llm_calls": trace.metrics.get("total_llm_calls_ok", 0),
    }
    logger.info("  Query: {} | State: {} | Debate: {} | Conflict: {} | Passed: {}",
                results["query"], results["evidence_state"], results["debate_triggered"],
                results["conflict_type"], results["passed"])
    return results


def test_all_agents_registered():
    logger.info("TEST 3: Agent Registration")
    router.initialize()
    agents = registry.list_ids()
    expected = ["web_agent", "reference_agent", "scholarly_agent", "news_agent",
                "code_agent", "factcheck_agent", "social_agent", "video_agent", "wiki_agent"]
    results = {"registered_agents": agents, "count": len(agents), "passed": set(agents) == set(expected)}
    logger.info("  Registered: {} | Count: {}/{} | Passed: {}",
                agents, len(agents), len(expected), results["passed"])
    return results


def test_routing_logic():
    logger.info("TEST 4: Query Routing")
    test_cases = [
        ("What is Python?", "general"),
        ("What does this GitHub repo do?", "code"),
        ("What are the latest news?", "news"),
        ("Is this fact true?", "factcheck"),
        ("What do people think?", "opinion"),
        ("What does the research say?", "scholarly"),
    ]
    results = []
    for query, expected_type in test_cases:
        decision = router.route(query)
        agents = decision.agents
        has_web = "web_agent" in agents
        results.append({"query": query, "expected_type": expected_type, "agents": agents,
                        "reasoning": decision.reasoning, "passed": has_web})
        logger.info("  {} -> {} | Agents: {} | Passed: {}", query, expected_type, agents, has_web)
    return {"test_cases": results, "passed": all(r["passed"] for r in results)}


def test_live_mode_with_groq():
    logger.info("TEST 5: Live Mode with Groq API")
    try:
        trace = solve("What is the capital of France?", demo_mode=False)
        results = {
            "query": trace.query,
            "evidence_state": trace.evidence_decision.state.value if trace.evidence_decision else "unknown",
            "debate_triggered": trace.debate_triggered,
            "answer": trace.final_answer.answer[:200] if trace.final_answer else "No answer",
            "latency_seconds": trace.metrics.get("latency_seconds", 0),
            "llm_calls_ok": trace.metrics.get("total_llm_calls_ok", 0),
            "llm_calls_failed": trace.metrics.get("total_llm_calls_failed", 0),
            "usable_agents": trace.metrics.get("usable_agents", 0),
            "error": trace.error,
            "passed": trace.error is None and trace.final_answer is not None,
        }
        logger.info("  State: {} | Agents: {} | LLM: {}/{} | Latency: {:.2f}s | Passed: {}",
                    results["evidence_state"], results["usable_agents"],
                    results["llm_calls_ok"], results["llm_calls_failed"],
                    results["latency_seconds"], results["passed"])
    except Exception as e:
        logger.error("  Live mode failed: {}", e)
        results = {"error": str(e), "passed": False}
    return results


def test_nli_detector():
    logger.info("TEST 6: NLI Detector (Mock)")
    from evidence.disagreement_detector import MockNLIDetector
    detector = MockNLIDetector(threshold=0.7)
    test_pairs = [
        ("The policy was introduced in 2019", "The policy was introduced in 2020", "contradiction"),
        ("Python 3.13 supports X", "Python 3.14 supports Y", "contradiction"),
        ("Dogs are mammals", "Cats are pets", "neutral"),
    ]
    results = []
    for text_a, text_b, expected in test_pairs:
        result = detector.predict(text_a, text_b)
        label = result["label"].value if hasattr(result["label"], 'value') else result["label"]
        scores = result.get("all_scores", {})
        passed = label.lower() == expected.lower()
        results.append({"text_a": text_a, "text_b": text_b, "expected": expected,
                        "predicted": label, "passed": passed})
        logger.info("  '{}' vs '{}' -> Expected: {}, Got: {} | Passed: {}",
                    text_a, text_b, expected, label, passed)
    return {"test_cases": results, "passed": all(r["passed"] for r in results)}


def test_conflict_classifier():
    logger.info("TEST 7: Conflict Classifier")
    from conflict.classifier import ConflictClassifier
    classifier = ConflictClassifier(llm_client=None)
    test_pairs = [
        ("Python 3.13 supports X", "Python 3.14 supports Y", "version"),
        ("The policy was introduced in 2019", "The policy was introduced in 2020", "temporal"),
        ("Water boils at 100C", "Water boils at 90C", "factual"),
    ]
    results = []
    for claim_a, claim_b, expected_type in test_pairs:
        result = classifier.classify(claim_a, claim_b, query="")
        passed = result.type.value == expected_type
        results.append({"claim_a": claim_a, "claim_b": claim_b, "expected_type": expected_type,
                        "classified_type": result.type.value, "confidence": result.confidence,
                        "passed": passed})
        logger.info("  '{}' vs '{}' -> Expected: {}, Got: {} | Passed: {}",
                    claim_a, claim_b, expected_type, result.type.value, passed)
    return {"test_cases": results, "passed": all(r["passed"] for r in results)}


def test_evidence_aggregator():
    logger.info("TEST 8: Evidence Aggregator")
    from evidence.aggregator import EvidenceAggregator
    from evidence.disagreement_detector import MockNLIDetector
    from evidence.models import AgentEvidence, EvidenceStatus, NormalizedClaim, Source
    aggregator = EvidenceAggregator(nli_detector=MockNLIDetector())
    def make_ev(agent_id, text):
        return AgentEvidence(agent_id=agent_id, query="test", status=EvidenceStatus.OK,
                            claims=[NormalizedClaim(text=text, source=Source(url="http://x.com", title="T", snippet=text))])
    ev1 = make_ev("a", "The capital city of Australia is Canberra")
    ev2 = make_ev("b", "The capital city of Australia is Canberra")
    d1 = aggregator.decide([ev1, ev2])
    ev3 = make_ev("c", "The sky is blue")
    d2 = aggregator.decide([ev3])
    results = {"agreement_test": {"state": d1.state.value, "passed": d1.state.value == "agreement"},
               "insufficient_test": {"state": d2.state.value, "passed": d2.state.value == "insufficient"}}
    results["passed"] = results["agreement_test"]["passed"] and results["insufficient_test"]["passed"]
    logger.info("  Agreement: {} | Insufficient: {} | Passed: {}",
                d1.state.value, d2.state.value, results["passed"])
    return results


def test_debate_manager():
    logger.info("TEST 9: Debate Manager (Fixture Mode)")
    from debate.manager import DebateManager
    from evidence.models import Conflict, ConflictType, NormalizedClaim, Source, AgentEvidence, EvidenceStatus
    manager = DebateManager(llm_client=None, max_rounds=3, run_mode="fixture")
    conflict = Conflict(type=ConflictType.FACTUAL, confidence=0.9, explanation="Test",
                        claim_a=NormalizedClaim(text="Water boils at 100C", source=Source(url="http://a.com", title="A", snippet="")),
                        claim_b=NormalizedClaim(text="Water boils at 90C", source=Source(url="http://b.com", title="B", snippet="")))
    evidence_list = [AgentEvidence(agent_id="a", query="test", status=EvidenceStatus.OK, claims=[conflict.claim_a]),
                     AgentEvidence(agent_id="b", query="test", status=EvidenceStatus.OK, claims=[conflict.claim_b])]
    outcome = manager.run_debate("What is the boiling point of water?", evidence_list, conflict)
    results = {"status": outcome.status, "rounds_used": outcome.rounds_used,
               "stop_reason": outcome.stop_reason, "passed": outcome.status == "completed"}
    logger.info("  Status: {} | Rounds: {} | Stop: {} | Passed: {}",
                outcome.status, outcome.rounds_used, outcome.stop_reason, results["passed"])
    return results


def test_judge():
    logger.info("TEST 10: Judge (Fixture Mode)")
    from debate.judge import Judge
    from evidence.models import (AgentEvidence, EvidenceStatus, NormalizedClaim, Source,
                                 ConflictType, DebateRound, DebateOutcome)
    judge = Judge(llm_client=None, run_mode="fixture")
    claim_a = NormalizedClaim(text="Water boils at 100C", source=Source(url="http://a.com", title="A", snippet=""))
    claim_b = NormalizedClaim(text="Water boils at 90C", source=Source(url="http://b.com", title="B", snippet=""))
    evidence_list = [AgentEvidence(agent_id="a", query="t", status=EvidenceStatus.OK, claims=[claim_a]),
                     AgentEvidence(agent_id="b", query="t", status=EvidenceStatus.OK, claims=[claim_b])]
    debate_outcome = DebateOutcome(rounds=[DebateRound(round_number=1)], status="completed",
                                   stop_reason="max_rounds", rounds_used=1)
    result = judge.evaluate(query="What is the boiling point of water?", evidence_list=evidence_list,
                            debate_outcome=debate_outcome, conflict_type=ConflictType.FACTUAL,
                            top_conflict={"claim_a": claim_a, "claim_b": claim_b})
    results = {"winner": result.winner if result else None, "confidence": result.confidence if result else 0,
               "passed": result is not None and result.winner in ("candidate_1", "candidate_2")}
    logger.info("  Winner: {} | Confidence: {} | Passed: {}",
                results["winner"], results["confidence"], results["passed"])
    return results


def generate_graphs(all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Patch

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Graph 1: Test Results Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    tests = list(all_results.keys())
    passed = [1 if all_results[t].get("passed", False) else 0 for t in tests]
    colors = ['#008000' if p else '#FF0000' for p in passed]
    bars = ax.barh(tests, passed, color=colors, edgecolor='black', linewidth=2)
    ax.set_xlabel('Passed', fontsize=14)
    ax.set_title('Evidence-MAD Evaluation Results', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.2)
    for bar, p in zip(bars, passed):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                'PASS' if p else 'FAIL', va='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/test_results.png")

    # Graph 2: Agent Routing
    fig, ax = plt.subplots(figsize=(10, 6))
    routing = all_results.get("Query Routing", {}).get("test_cases", [])
    if routing:
        qtypes = [r["expected_type"] for r in routing]
        acounts = [len(r["agents"]) for r in routing]
        ax.bar(qtypes, acounts, color='#0000FF', edgecolor='black', linewidth=2)
        ax.set_xlabel('Query Type', fontsize=14)
        ax.set_ylabel('Number of Agents', fontsize=14)
        ax.set_title('Agent Routing by Query Type', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(acounts) + 1)
        for i, v in enumerate(acounts):
            ax.text(i, v + 0.1, str(v), ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'routing_agents.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/routing_agents.png")

    # Graph 3: Fixture Case Outcomes
    fig, ax = plt.subplots(figsize=(10, 6))
    fixture_cases = ["agree", "factual", "temporal", "version", "contextual", "source", "unrelated", "1_fail", "2_fail", "all_fail"]
    expected_states = ["agreement", "disagreement", "disagreement", "disagreement", "disagreement", "disagreement", "uncertain", "insufficient", "insufficient", "insufficient"]
    state_colors = ['#008000' if s == "agreement" else '#FF0000' if s == "disagreement" else '#FFA500' if s == "uncertain" else '#000000' for s in expected_states]
    ax.barh(fixture_cases, [1]*len(fixture_cases), color=state_colors, edgecolor='black', linewidth=2)
    ax.set_title('Fixture Case Expected Outcomes', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.2)
    legend_elements = [Patch(facecolor='#008000', label='Agreement'), Patch(facecolor='#FF0000', label='Disagreement'),
                       Patch(facecolor='#FFA500', label='Uncertain'), Patch(facecolor='#000000', label='Insufficient')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fixture_outcomes.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/fixture_outcomes.png")

    # Graph 4: Conflict Type Distribution
    fig, ax = plt.subplots(figsize=(8, 8))
    conflict_types = ['Factual', 'Temporal', 'Version', 'Contextual', 'Source']
    counts = [1, 1, 1, 1, 1]
    colors_pie = ['#FF0000', '#FFA500', '#0000FF', '#008000', '#000000']
    ax.pie(counts, labels=conflict_types, colors=colors_pie, autopct='%1.0f%%',
           textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'},
           wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax.set_title('Conflict Type Distribution in Fixtures', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'conflict_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/conflict_types.png")

    # Graph 5: System Architecture
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_title('Evidence-MAD System Architecture', fontsize=18, fontweight='bold', pad=20)
    boxes = [(0.35, 0.9, 'Query'), (0.1, 0.7, 'Web Agent'), (0.35, 0.7, 'Wiki Agent'),
             (0.6, 0.7, 'News Agent'), (0.85, 0.7, 'Scholarly'), (0.35, 0.5, 'Evidence\nComparison\n(NLI)'),
             (0.1, 0.3, 'AGREE\nSkip Debate'), (0.6, 0.3, 'DISAGREE\nTrigger Debate'), (0.35, 0.1, 'Final Answer')]
    for x, y, text in boxes:
        box = FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1, boxstyle="round,pad=0.02",
                             facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')
    arrows = [(0.35, 0.85, 0.1, 0.75), (0.35, 0.85, 0.35, 0.75), (0.35, 0.85, 0.6, 0.75), (0.35, 0.85, 0.85, 0.75),
              (0.1, 0.7, 0.35, 0.55), (0.35, 0.7, 0.35, 0.55), (0.6, 0.7, 0.35, 0.55), (0.85, 0.7, 0.35, 0.55),
              (0.35, 0.45, 0.1, 0.35), (0.35, 0.45, 0.6, 0.35), (0.1, 0.25, 0.35, 0.15), (0.6, 0.25, 0.35, 0.15)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/architecture.png")


def run_all_tests():
    logger.info("=" * 60)
    logger.info("EVIDENCE-MAD COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)
    all_results = {}
    tests = [
        ("Evidence Gate", test_evidence_gate),
        ("Disagreement Triggers Debate", test_disagreement_triggers_debate),
        ("Agent Registration", test_all_agents_registered),
        ("Query Routing", test_routing_logic),
        ("Live Mode Groq", test_live_mode_with_groq),
        ("NLI Detector", test_nli_detector),
        ("Conflict Classifier", test_conflict_classifier),
        ("Evidence Aggregator", test_evidence_aggregator),
        ("Debate Manager", test_debate_manager),
        ("Judge", test_judge),
    ]
    for name, test_fn in tests:
        logger.info("-" * 40)
        try:
            all_results[name] = test_fn()
        except Exception as e:
            logger.error("TEST FAILED: {} - {}", name, e)
            all_results[name] = {"error": str(e), "passed": False}

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    passed = sum(1 for r in all_results.values() if r.get("passed", False))
    total = len(all_results)
    for name, result in all_results.items():
        status = "PASS" if result.get("passed", False) else "FAIL"
        logger.info("  [{}] {}", status, name)
    logger.info("  Total: {}/{} tests passed | Success rate: {:.1f}%", passed, total, passed / total * 100)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("  Results saved to outputs/evaluation_results.json")

    logger.info("-" * 40)
    logger.info("GENERATING GRAPHS")
    generate_graphs(all_results)
    return all_results


if __name__ == "__main__":
    run_all_tests()
