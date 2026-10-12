"""
Wikipedia search agent using the Wikipedia API.
Free, no API key required.
"""
import re
from typing import List, Dict, Any

import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim


class WikiAgentConfig(AgentConfig):
    """Configuration for Wikipedia search agent."""
    max_results: int = 5
    provider: str = "wikipedia"


class WikiAgent(BaseAgent):
    """Agent that retrieves evidence using Wikipedia API."""

    def __init__(self):
        config = WikiAgentConfig(
            agent_id="wiki_agent",
            name="WikiAgent",
            description="Wikipedia search agent",
            max_results=5,
            provider="wikipedia",
        )
        super().__init__(config)

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "EvidenceMAD/1.0 (research; contact@example.com)"},
        )

    def _wikipedia_search(self, query: str) -> List[Dict[str, Any]]:
        """Wikipedia search via public API."""
        with self._make_client() as client:
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": self.config.max_results,
                },
            )
            resp.raise_for_status()

        items = resp.json().get("query", {}).get("search", [])
        return [
            {
                "title": item.get("title", ""),
                "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip(),
                "link": f"https://en.wikipedia.org/?curid={item.get('pageid', '')}",
            }
            for item in items
        ]

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._wikipedia_search(query)

    def retrieve(self, query: str) -> Evidence:
        self.logger.info(f"Retrieving Wikipedia evidence for: {query!r}")

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
