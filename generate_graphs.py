"""
Generate graphs from the real evaluation results.
"""
import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Real evaluation data (from the run we just did)
real_data = {
    "queries": [
        "What is the capital of Australia?",
        "When was India's NEP introduced?",
        "What new features does Python 3.13 have?",
        "Is aspirin safe to take?",
        "What is the boiling point of water?",
        "Does the MMR vaccine cause autism?",
    ],
    "expected": ["agreement", "disagreement", "disagreement", "disagreement", "agreement", "disagreement"],
    "evidence_mad": {
        "states": ["insufficient", "agreement", "insufficient", "agreement", "agreement", "agreement"],
        "debate_triggered": [False, False, False, False, False, False],
        "conflict_types": [None, None, None, None, None, None],
        "llm_calls": [3, 3, 2, 5, 4, 4],
        "latency": [6.40, 7.10, 9.70, 12.48, 8.63, 20.29],
        "usable_agents": [1, 2, 1, 4, 3, 4],
        "errors": [None, None, None, None, None, None],
    },
    "single_llm_baseline": {
        "llm_calls": [1, 1, 0, 1, 1, 1],
        "latency": [0.61, 0.24, 6.13, 0.26, 6.22, 0.16],
    },
    "standard_mad": {
        "llm_calls": [9, 10, 8, 9, 11, 10],
    },
}

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

queries = real_data["queries"]
ev = real_data["evidence_mad"]
baseline = real_data["single_llm_baseline"]
std = real_data["standard_mad"]

# Graph 1: Evidence State Distribution (REAL data)
fig, ax = plt.subplots(figsize=(8, 8))
states = ev["states"]
state_counts = {}
for s in states:
    state_counts[s] = state_counts.get(s, 0) + 1

colors = {'agreement': '#008000', 'disagreement': '#FF0000', 'uncertain': '#FFA500', 'insufficient': '#000000'}
pie_colors = [colors.get(s, '#CCCCCC') for s in state_counts.keys()]
wedges, texts, autotexts = ax.pie(
    state_counts.values(),
    labels=state_counts.keys(),
    colors=pie_colors,
    autopct='%1.0f%%',
    textprops={'fontsize': 14, 'color': 'white', 'weight': 'bold'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 3}
)
ax.set_title('Evidence State Distribution\n(Real Live Mode Results)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'real_evidence_state_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
logger.info("Saved: outputs/real_evidence_state_distribution.png")

# Graph 2: LLM Call Count Comparison (REAL data)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(queries))
width = 0.25

bars1 = ax.bar(x - width, baseline["llm_calls"], width, label='Single LLM', color='#0000FF')
bars2 = ax.bar(x, ev["llm_calls"], width, label='Evidence-MAD', color='#000000')
bars3 = ax.bar(x + width, std["llm_calls"], width, label='Standard MAD (simulated)', color='#CCCCCC')

ax.set_xlabel('Query', fontsize=12)
ax.set_ylabel('LLM Calls', fontsize=12)
ax.set_title('LLM Call Count: Real Evaluation', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([q[:25] + "..." if len(q) > 25 else q for q in queries], rotation=45, ha='right', fontsize=9)
ax.legend()

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'real_llm_calls_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
logger.info("Saved: outputs/real_llm_calls_comparison.png")

# Graph 3: Latency Comparison (REAL data)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(queries))
width = 0.35

bars1 = ax.bar(x - width/2, baseline["latency"], width, label='Single LLM', color='#0000FF')
bars2 = ax.bar(x + width/2, ev["latency"], width, label='Evidence-MAD', color='#000000')

ax.set_xlabel('Query', fontsize=12)
ax.set_ylabel('Latency (seconds)', fontsize=12)
ax.set_title('Latency: Real Evaluation', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([q[:25] + "..." if len(q) > 25 else q for q in queries], rotation=45, ha='right', fontsize=9)
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'real_latency_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
logger.info("Saved: outputs/real_latency_comparison.png")

# Graph 4: Usable Agents per Query (REAL data)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(queries))
bars = ax.bar(x, ev["usable_agents"], color='#000000', edgecolor='black', linewidth=2)

ax.set_xlabel('Query', fontsize=12)
ax.set_ylabel('Usable Agents', fontsize=12)
ax.set_title('Usable Agents per Query (Real Evaluation)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([q[:25] + "..." if len(q) > 25 else q for q in queries], rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 6)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'real_usable_agents.png', dpi=150, bbox_inches='tight')
plt.close()
logger.info("Saved: outputs/real_usable_agents.png")

# Graph 5: Expected vs Actual State (REAL data)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(queries))
width = 0.35

expected_numerical = [1 if s == "agreement" else 2 if s == "disagreement" else 3 if s == "uncertain" else 0 for s in real_data["expected"]]
actual_numerical = [1 if s == "agreement" else 2 if s == "disagreement" else 3 if s == "uncertain" else 0 for s in ev["states"]]

bars1 = ax.bar(x - width/2, expected_numerical, width, label='Expected', color='#CCCCCC', edgecolor='black')
bars2 = ax.bar(x + width/2, actual_numerical, width, label='Actual', color='#000000')

ax.set_xlabel('Query', fontsize=12)
ax.set_ylabel('State', fontsize=12)
ax.set_title('Expected vs Actual Evidence State (Real Evaluation)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([q[:25] + "..." if len(q) > 25 else q for q in queries], rotation=45, ha='right', fontsize=9)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['Insufficient', 'Agreement', 'Disagreement', 'Uncertain'])
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'expected_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
logger.info("Saved: outputs/expected_vs_actual.png")

# Graph 6: System Architecture (updated with real components)
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
ax.set_title('Evidence-MAD System Architecture (9 Agents)', fontsize=18, fontweight='bold', pad=20)

from matplotlib.patches import FancyBboxPatch

# Input
box = FancyBboxPatch((0.35, 0.92), 0.3, 0.06, boxstyle="round,pad=0.02",
                     facecolor='white', edgecolor='black', linewidth=3)
ax.add_patch(box)
ax.text(0.5, 0.95, 'Query', ha='center', va='center', fontsize=12, weight='bold')

# Agents (9 agents in 3 rows)
agents_row1 = ['Web Agent', 'Wiki Agent', 'Reference Agent']
agents_row2 = ['News Agent', 'Code Agent', 'FactCheck Agent']
agents_row3 = ['Social Agent', 'Video Agent', 'Scholarly Agent']

for i, name in enumerate(agents_row1):
    box = FancyBboxPatch((0.05 + i*0.3, 0.75), 0.25, 0.06, boxstyle="round,pad=0.02",
                         facecolor='white', edgecolor='black', linewidth=3)
    ax.add_patch(box)
    ax.text(0.175 + i*0.3, 0.78, name, ha='center', va='center', fontsize=9, weight='bold')

for i, name in enumerate(agents_row2):
    box = FancyBboxPatch((0.05 + i*0.3, 0.65), 0.25, 0.06, boxstyle="round,pad=0.02",
                         facecolor='white', edgecolor='black', linewidth=3)
    ax.add_patch(box)
    ax.text(0.175 + i*0.3, 0.68, name, ha='center', va='center', fontsize=9, weight='bold')

for i, name in enumerate(agents_row3):
    box = FancyBboxPatch((0.05 + i*0.3, 0.55), 0.25, 0.06, boxstyle="round,pad=0.02",
                         facecolor='white', edgecolor='black', linewidth=3)
    ax.add_patch(box)
    ax.text(0.175 + i*0.3, 0.58, name, ha='center', va='center', fontsize=9, weight='bold')

# Evidence Comparison
box = FancyBboxPatch((0.35, 0.42), 0.3, 0.06, boxstyle="round,pad=0.02",
                     facecolor='white', edgecolor='black', linewidth=3)
ax.add_patch(box)
ax.text(0.5, 0.45, 'Evidence Comparison (NLI)', ha='center', va='center', fontsize=10, weight='bold')

# Decision branches
box = FancyBboxPatch((0.1, 0.28), 0.3, 0.06, boxstyle="round,pad=0.02",
                     facecolor='#008000', edgecolor='black', linewidth=3)
ax.add_patch(box)
ax.text(0.25, 0.31, 'AGREE → Skip Debate', ha='center', va='center', fontsize=10, weight='bold', color='white')

box = FancyBboxPatch((0.6, 0.28), 0.3, 0.06, boxstyle="round,pad=0.02",
                     facecolor='#FF0000', edgecolor='black', linewidth=3)
ax.add_patch(box)
ax.text(0.75, 0.31, 'DISAGREE → Debate', ha='center', va='center', fontsize=10, weight='bold', color='white')

# Final answer
box = FancyBboxPatch((0.35, 0.15), 0.3, 0.06, boxstyle="round,pad=0.02",
                     facecolor='white', edgecolor='black', linewidth=3)
ax.add_patch(box)
ax.text(0.5, 0.18, 'Final Answer', ha='center', va='center', fontsize=12, weight='bold')

# Draw arrows
arrow_props = dict(arrowstyle='->', color='black', lw=2)
# Query to agents
for i in range(3):
    ax.annotate('', xy=(0.175 + i*0.3, 0.81), xytext=(0.4 + i*0.1, 0.92), arrowprops=arrow_props)
# Agents to comparison
for i in range(3):
    ax.annotate('', xy=(0.45, 0.48), xytext=(0.175 + i*0.3, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.45, 0.48), xytext=(0.175 + i*0.3, 0.65), arrowprops=arrow_props)
    ax.annotate('', xy=(0.45, 0.48), xytext=(0.175 + i*0.3, 0.55), arrowprops=arrow_props)
# Comparison to branches
ax.annotate('', xy=(0.25, 0.34), xytext=(0.4, 0.42), arrowprops=arrow_props)
ax.annotate('', xy=(0.75, 0.34), xytext=(0.6, 0.42), arrowprops=arrow_props)
# Branches to final answer
ax.annotate('', xy=(0.45, 0.21), xytext=(0.25, 0.28), arrowprops=arrow_props)
ax.annotate('', xy=(0.55, 0.21), xytext=(0.75, 0.28), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(output_dir / 'real_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
logger.info("Saved: outputs/real_architecture.png")

# Summary
logger.info("=" * 60)
logger.info("REAL EVALUATION SUMMARY")
logger.info("=" * 60)
logger.info("  Queries tested: 6")
logger.info("  Evidence states: %s", ev["states"])
logger.info("  Debates triggered: 0/6 (rate limits prevented full evaluation)")
logger.info("  Avg LLM calls: Evidence-MAD=%.1f | Single LLM=%.1f | Standard MAD=%.1f",
            np.mean(ev["llm_calls"]), np.mean(baseline["llm_calls"]), np.mean(std["llm_calls"]))
logger.info("  Avg latency: Evidence-MAD=%.2fs | Single LLM=%.2fs",
            np.mean(ev["latency"]), np.mean(baseline["latency"]))
logger.info("  Key finding: Wikipedia/GitHub/Groq rate limits prevent full live evaluation")
logger.info("  Recommendation: Use demo mode for development, live mode for final paper")
