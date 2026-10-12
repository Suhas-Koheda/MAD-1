"""
Real evaluation: LIVE mode with actual search + Groq LLM calls.
Compares Evidence-MAD vs Single LLM baseline vs Standard MAD (always debates).
"""
import json
import time
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

from app import solve, MADSystem
from llm_client import LLMClient


# Test queries covering different conflict types
TEST_QUERIES = [
    {"query": "What is the capital of Australia?", "expected": "agreement", "type": "general"},
    {"query": "When was India's National Education Policy introduced?", "expected": "disagreement", "type": "temporal"},
    {"query": "What new features does Python 3.13 have?", "expected": "disagreement", "type": "version"},
    {"query": "Is aspirin safe to take?", "expected": "disagreement", "type": "contextual"},
    {"query": "What is the boiling point of water at sea level?", "expected": "agreement", "type": "factual"},
    {"query": "Does the MMR vaccine cause autism?", "expected": "disagreement", "type": "source"},
]


def run_single_llm_baseline(query: str) -> dict:
    """Single LLM baseline: one call, no debate."""
    client = LLMClient()
    start = time.time()
    try:
        answer = client.chat(f"Answer this question concisely: {query}", role="baseline")
        latency = time.time() - start
        return {
            "query": query,
            "answer": answer,
            "latency": latency,
            "llm_calls": 1,
            "debate_triggered": False,
            "error": None,
        }
    except Exception as e:
        return {"query": query, "answer": "", "latency": time.time() - start,
                "llm_calls": 0, "debate_triggered": False, "error": str(e)}


def run_evidence_mad(query: str, demo_mode: bool = False) -> dict:
    """Evidence-MAD: evidence-triggered debate."""
    start = time.time()
    try:
        trace = solve(query, demo_mode=demo_mode)
        latency = time.time() - start
        return {
            "query": query,
            "answer": trace.final_answer.answer if trace.final_answer else "",
            "evidence_state": trace.evidence_decision.state.value if trace.evidence_decision else "unknown",
            "debate_triggered": trace.debate_triggered,
            "conflict_type": trace.conflict.type.value if trace.conflict else None,
            "debate_rounds": trace.debate_rounds,
            "latency": latency,
            "llm_calls": trace.metrics.get("total_llm_calls_ok", 0),
            "usable_agents": trace.metrics.get("usable_agents", 0),
            "source_count": trace.final_answer.source_count if trace.final_answer else 0,
            "error": trace.error,
        }
    except Exception as e:
        return {"query": query, "answer": "", "latency": time.time() - start,
                "llm_calls": 0, "debate_triggered": False, "error": str(e)}


def run_standard_mad(query: str) -> dict:
    """Standard MAD: always debates (simulated by running debate even on agreement)."""
    start = time.time()
    try:
        trace = solve(query, demo_mode=False)
        # Simulate standard MAD: count what it WOULD cost if it always debated
        state = trace.evidence_decision.state.value if trace.evidence_decision else "unknown"
        # Standard MAD would run debate rounds even on agreement
        simulated_debate_rounds = 3  # max rounds
        simulated_llm_calls = trace.metrics.get("total_llm_calls_ok", 0) + 6  # +2 per stage * 3 rounds
        return {
            "query": query,
            "answer": trace.final_answer.answer if trace.final_answer else "",
            "evidence_state": state,
            "debate_triggered": True,  # Always debates
            "latency": time.time() - start,
            "llm_calls": simulated_llm_calls,
            "note": "Simulated: always runs debate regardless of evidence state",
            "error": trace.error,
        }
    except Exception as e:
        return {"query": query, "answer": "", "latency": time.time() - start,
                "llm_calls": 0, "debate_triggered": True, "error": str(e)}


def generate_graphs(results: dict):
    """Generate honest graphs from real results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Graph 1: Debate Trigger Rate by Query Type
    fig, ax = plt.subplots(figsize=(10, 6))
    queries = [r["query"][:30] + "..." if len(r["query"]) > 30 else r["query"]
               for r in results["evidence_mad"]]
    triggered = [1 if r["debate_triggered"] else 0 for r in results["evidence_mad"]]
    expected = [r["expected"] for r in TEST_QUERIES]

    x = np.arange(len(queries))
    width = 0.35
    ax.bar(x - width/2, triggered, width, label='Evidence-MAD', color='#000000')
    ax.bar(x + width/2, [1]*len(queries), width, label='Standard MAD (always debates)', color='#CCCCCC')

    ax.set_xlabel('Query')
    ax.set_ylabel('Debate Triggered')
    ax.set_title('Debate Trigger Rate: Evidence-MAD vs Standard MAD')
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'debate_trigger_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/debate_trigger_comparison.png")

    # Graph 2: LLM Call Count Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ev_calls = [r["llm_calls"] for r in results["evidence_mad"]]
    std_calls = [r["llm_calls"] for r in results["standard_mad"]]
    baseline_calls = [r["llm_calls"] for r in results["single_llm_baseline"]]

    x = np.arange(len(queries))
    width = 0.25
    ax.bar(x - width, baseline_calls, width, label='Single LLM', color='#0000FF')
    ax.bar(x, ev_calls, width, label='Evidence-MAD', color='#000000')
    ax.bar(x + width, std_calls, width, label='Standard MAD', color='#CCCCCC')

    ax.set_xlabel('Query')
    ax.set_ylabel('LLM Calls')
    ax.set_title('LLM Call Count: Evidence-MAD vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=8)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'llm_calls_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/llm_calls_comparison.png")

    # Graph 3: Latency Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ev_latency = [r["latency"] for r in results["evidence_mad"]]
    baseline_latency = [r["latency"] for r in results["single_llm_baseline"]]

    x = np.arange(len(queries))
    width = 0.35
    ax.bar(x - width/2, baseline_latency, width, label='Single LLM', color='#0000FF')
    ax.bar(x + width/2, ev_latency, width, label='Evidence-MAD', color='#000000')

    ax.set_xlabel('Query')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency: Evidence-MAD vs Single LLM Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=8)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/latency_comparison.png")

    # Graph 4: Evidence State Distribution (pie chart with REAL data)
    fig, ax = plt.subplots(figsize=(8, 8))
    states = [r["evidence_state"] for r in results["evidence_mad"]]
    state_counts = {}
    for s in states:
        state_counts[s] = state_counts.get(s, 0) + 1

    if state_counts:
        colors = {'agreement': '#008000', 'disagreement': '#FF0000', 'uncertain': '#FFA500', 'insufficient': '#000000'}
        pie_colors = [colors.get(s, '#CCCCCC') for s in state_counts.keys()]
        ax.pie(state_counts.values(), labels=state_counts.keys(), colors=pie_colors,
               autopct='%1.0f%%', textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'},
               wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax.set_title('Evidence State Distribution (Real Results)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'evidence_state_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/evidence_state_distribution.png")

    # Graph 5: Conflict Type Distribution (REAL data from results)
    fig, ax = plt.subplots(figsize=(8, 8))
    conflict_types = [r["conflict_type"] for r in results["evidence_mad"] if r["conflict_type"]]
    if conflict_types:
        type_counts = {}
        for t in conflict_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        colors_map = {'factual': '#FF0000', 'temporal': '#FFA500', 'version': '#0000FF',
                      'contextual': '#008000', 'source': '#000000', 'unknown': '#CCCCCC'}
        pie_colors = [colors_map.get(t, '#CCCCCC') for t in type_counts.keys()]
        ax.pie(type_counts.values(), labels=type_counts.keys(), colors=pie_colors,
               autopct='%1.0f%%', textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'},
               wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax.set_title('Conflict Type Distribution (Real Results)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'conflict_type_distribution_real.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: outputs/conflict_type_distribution_real.png")


def main():
    logger.info("=" * 60)
    logger.info("REAL EVALUATION: LIVE MODE WITH GROQ API")
    logger.info("=" * 60)
    logger.info("")

    results = {
        "single_llm_baseline": [],
        "evidence_mad": [],
        "standard_mad": [],
    }

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        logger.info("-" * 60)
        logger.info("QUERY {}/{}: {} (expected: {})", i+1, len(TEST_QUERIES), query, test["expected"])
        logger.info("-" * 60)

        # 1. Single LLM baseline
        logger.info("  Running Single LLM baseline...")
        baseline = run_single_llm_baseline(query)
        results["single_llm_baseline"].append(baseline)
        logger.info("    Answer: {}... | Latency: {:.2f}s | Calls: {}",
                    baseline["answer"][:60], baseline["latency"], baseline["llm_calls"])

        # 2. Evidence-MAD (LIVE mode with real search + Groq)
        logger.info("  Running Evidence-MAD (LIVE)...")
        ev_result = run_evidence_mad(query, demo_mode=False)
        results["evidence_mad"].append(ev_result)
        logger.info("    State: {} | Debate: {} | Agents: {} | Calls: {} | Latency: {:.2f}s",
                    ev_result["evidence_state"], ev_result["debate_triggered"],
                    ev_result["usable_agents"], ev_result["llm_calls"], ev_result["latency"])
        if ev_result["error"]:
            logger.warning("    Error: {}", ev_result["error"])

        # 3. Standard MAD (simulated)
        logger.info("  Running Standard MAD (simulated)...")
        std_result = run_standard_mad(query)
        results["standard_mad"].append(std_result)
        logger.info("    Always debates | Simulated calls: {}", std_result["llm_calls"])

        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("REAL EVALUATION SUMMARY")
    logger.info("=" * 60)

    ev_states = [r["evidence_state"] for r in results["evidence_mad"]]
    ev_debates = [r["debate_triggered"] for r in results["evidence_mad"]]
    ev_calls = [r["llm_calls"] for r in results["evidence_mad"]]
    ev_latency = [r["latency"] for r in results["evidence_mad"]]
    baseline_calls = [r["llm_calls"] for r in results["single_llm_baseline"]]
    baseline_latency = [r["latency"] for r in results["single_llm_baseline"]]
    std_calls = [r["llm_calls"] for r in results["standard_mad"]]

    logger.info("  Evidence States: {}", ev_states)
    logger.info("  Debates Triggered: {}/{}", sum(ev_debates), len(ev_debates))
    logger.info("  Avg LLM Calls: Evidence-MAD={:.1f} | Single LLM={:.1f} | Standard MAD={:.1f}",
                np.mean(ev_calls), np.mean(baseline_calls), np.mean(std_calls))
    logger.info("  Avg Latency: Evidence-MAD={:.2f}s | Single LLM={:.2f}s",
                np.mean(ev_latency), np.mean(baseline_latency))
    logger.info("  Errors: {}", sum(1 for r in results["evidence_mad"] if r["error"]))

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "real_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("  Results saved to outputs/real_evaluation_results.json")

    # Generate graphs
    logger.info("-" * 40)
    logger.info("GENERATING GRAPHS")
    generate_graphs(results)

    return results


if __name__ == "__main__":
    main()
