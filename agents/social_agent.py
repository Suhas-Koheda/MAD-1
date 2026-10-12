"""
Social media search agent using Reddit JSON API.
Free, no API key required (rate-limited).
"""
import re
from typing import List, Dict, Any

import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim


class SocialAgentConfig(AgentConfig):
    """Configuration for social media search agent."""
    max_results: int = 5
    provider: str = "reddit"


class SocialAgent(BaseAgent):
    """Agent that retrieves evidence using Reddit search."""

    def __init__(self):
        config = SocialAgentConfig(
            agent_id="social_agent",
            name="SocialAgent",
            description="Social media search agent (Reddit)",
            max_results=5,
            provider="reddit",
        )
        super().__init__(config)

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "evidence-mad/1.0 (research)"},
        )

    def _reddit_search(self, query: str) -> List[Dict[str, Any]]:
        """Reddit search via public JSON API."""
        with self._make_client() as client:
            resp = client.get(
                "https://www.reddit.com/search.json",
                params={"q": query, "limit": self.config.max_results, "sort": "relevance"},
            )
            if resp.status_code == 429:
                logger.warning("Reddit API rate limited")
                return []
            resp.raise_for_status()

        items = resp.json().get("data", {}).get("children", [])
        results = []
        for item in items:
            post = item.get("data", {})
            results.append({
                "title": post.get("title", ""),
                "snippet": post.get("selftext", "")[:300] if post.get("selftext") else post.get("title", ""),
                "link": f"https://reddit.com{post.get('permalink', '')}",
                "score": post.get("score", 0),
                "subreddit": post.get("subreddit", ""),
            })

        return results

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._reddit_search(query)

    def retrieve(self, query: str) -> Evidence:
        self.logger.info(f"Retrieving social evidence for: {query!r}")

        search_results = self.search(query)

        sources = []
        retrieved_texts = []
        for result in search_results:
            source = Source(
                url=result.get("link", ""),
                title=result.get("title", ""),
                snippet=result.get("snippet", ""),
                metadata={
                    "search_query": query,
                    "score": result.get("score", 0),
                    "subreddit": result.get("subreddit", ""),
                },
            )
            sources.append(source)
            retrieved_texts.append(f"{source.title}: {source.snippet}")

        combined_text = "\n\n".join(retrieved_texts)
        claims = self.extract_claims(combined_text, sources)
        answer = self.generate_answer(query, claims, sources)

        return Evidence(
            agent_id=self.agent_id,
            query=query,
            answer=answer,
            claims=claims,
            sources=sources,
            retrieved_text=combined_text,
            metadata={
                "search_results_count": len(search_results),
                "provider": self.config.provider,
            },
        )
