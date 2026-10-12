"""
Query router for determining which agents to use for a given query.
Uses ThreadPoolExecutor for parallel synchronous retrieval (no asyncio).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
from loguru import logger

from agents.base_agent import BaseAgent, registry
from agents.search_agent import WebAgent, ReferenceAgent, ScholarlyAgent
from agents.news_agent import NewsAgent
from agents.code_agent import CodeAgent
from agents.factcheck_agent import FactCheckAgent
from agents.social_agent import SocialAgent
from agents.video_agent import VideoAgent
from agents.wiki_agent import WikiAgent
from config import settings


# Scholarly query keywords — only these queries get the Crossref agent
SCHOLARLY_KEYWORDS = {
    "study", "trial", "meta-analysis", "meta analysis", "systematic review",
    "randomized", "randomised", "clinical", "cohort", "evidence for",
    "research on", "doi", "journal", "peer reviewed", "peer-reviewed",
    "pubmed", "arxiv", "preprint",
}


class RoutingDecision(BaseModel):
    """Result of routing a query."""
    agents: List[str] = []  # agent IDs selected
    reasoning: str = ""


def _is_scholarly(query: str) -> bool:
    """Return True if the query warrants the Crossref scholarly agent."""
    q = query.lower()
    return any(kw in q for kw in SCHOLARLY_KEYWORDS)


class QueryRouter:
    """
    Routes queries to appropriate agents and executes them in parallel.

    Three fixed agent profiles:
    - WEB       (DuckDuckGo)   — always included
    - REFERENCE (Wikipedia)    — always included
    - SCHOLARLY (Crossref)     — only for research-flavoured queries
    """

    def __init__(self):
        self._initialized = False

    def initialize(self) -> None:
        """Register all agents (idempotent)."""
        if self._initialized:
            return
        registry.clear()
        registry.register(WebAgent())
        registry.register(ReferenceAgent())
        registry.register(ScholarlyAgent())
        registry.register(NewsAgent())
        registry.register(CodeAgent())
        registry.register(FactCheckAgent())
        registry.register(SocialAgent())
        registry.register(VideoAgent())
        registry.register(WikiAgent())
        self._initialized = True
        logger.info(f"Router initialized with {len(registry.list_ids())} agents: {registry.list_ids()}")

    def route(self, query: str) -> RoutingDecision:
        """Determine which agents to use for a query."""
        self.initialize()

        q = query.lower()

        # Determine which agents to activate based on query type
        if _is_scholarly(query):
            agents = registry.list_ids()  # all agents
            reasoning = "Scholarly query — routing to all agents"
        elif any(kw in q for kw in ["code", "function", "api", "library", "python", "javascript", "bug", "error", "github"]):
            agents = ["web_agent", "wiki_agent", "code_agent", "scholarly_agent"]
            reasoning = "Code query — routing to WEB + WIKI + CODE + SCHOLARLY agents"
        elif any(kw in q for kw in ["news", "recent", "latest", "today", "2024", "2025", "2026"]):
            agents = ["web_agent", "news_agent", "wiki_agent", "social_agent"]
            reasoning = "News query — routing to WEB + NEWS + WIKI + SOCIAL agents"
        elif any(kw in q for kw in ["fact", "true", "false", "verify", "check", "real", "myth"]):
            agents = ["web_agent", "factcheck_agent", "wiki_agent", "scholarly_agent"]
            reasoning = "Fact-check query — routing to WEB + FACTCHECK + WIKI + SCHOLARLY agents"
        elif any(kw in q for kw in ["opinion", "think", "feel", "reddit", "community"]):
            agents = ["web_agent", "social_agent", "news_agent"]
            reasoning = "Opinion query — routing to WEB + SOCIAL + NEWS agents"
        else:
            # General query — use core agents
            agents = ["web_agent", "wiki_agent", "reference_agent", "news_agent"]
            reasoning = "General query — routing to WEB + WIKI + REFERENCE + NEWS agents"

        return RoutingDecision(agents=agents, reasoning=reasoning)

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute the routing decision synchronously with a ThreadPoolExecutor.

        Args:
            query: User query (passed verbatim to every agent)

        Returns:
            Dictionary with decision and per-agent results
        """
        decision = self.route(query)
        agents: List[BaseAgent] = [
            registry.get(aid) for aid in decision.agents if registry.get(aid)
        ]

        if not agents:
            return {
                "decision": decision.model_dump(),
                "evidence": [],
                "error": "No agents available",
            }

        results = []
        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            futures = {pool.submit(agent.retrieve, query): agent for agent in agents}
            for fut in as_completed(futures):
                agent = futures[fut]
                try:
                    evidence = fut.result()
                    results.append({
                        "agent_id": agent.agent_id,
                        "evidence": evidence.model_dump(),
                    })
                except Exception as exc:
                    logger.error(f"Agent {agent.agent_id} failed: {exc}")
                    results.append({
                        "agent_id": agent.agent_id,
                        "error": repr(exc),
                    })

        return {
            "decision": decision.model_dump(),
            "evidence": results,
        }


# Global router instance
router = QueryRouter()