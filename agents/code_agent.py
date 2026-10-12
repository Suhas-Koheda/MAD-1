"""
Code search agent using GitHub code search API.
Free tier: 10 requests/minute without auth.
"""
import re
from typing import List, Dict, Any

import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim


class CodeAgentConfig(AgentConfig):
    """Configuration for code search agent."""
    max_results: int = 5
    provider: str = "github"


class CodeAgent(BaseAgent):
    """Agent that retrieves evidence using GitHub code search."""

    def __init__(self):
        config = CodeAgentConfig(
            agent_id="code_agent",
            name="CodeAgent",
            description="Code search agent (GitHub)",
            max_results=5,
            provider="github",
        )
        super().__init__(config)

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=30.0,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "evidence-mad/1.0",
            },
        )

    def _github_search(self, query: str) -> List[Dict[str, Any]]:
        """GitHub code search — free tier available."""
        with self._make_client() as client:
            resp = client.get(
                "https://api.github.com/search/code",
                params={"q": query, "per_page": self.config.max_results},
            )
            if resp.status_code == 403:
                logger.warning("GitHub API rate limit hit")
                return []
            resp.raise_for_status()

        items = resp.json().get("items", [])
        results = []
        for item in items:
            results.append({
                "title": item.get("name", ""),
                "snippet": f"Repository: {item.get('repository', {}).get('full_name', '')} | Path: {item.get('path', '')}",
                "link": item.get("html_url", ""),
            })

        return results

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._github_search(query)

    def retrieve(self, query: str) -> Evidence:
        self.logger.info(f"Retrieving code evidence for: {query!r}")

        search_results = self.search(query)

        sources = []
        retrieved_texts = []
        for result in search_results:
            source = Source(
                url=result.get("link", ""),
                title=result.get("title", ""),
                snippet=result.get("snippet", ""),
                metadata={"search_query": query},
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
