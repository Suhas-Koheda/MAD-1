"""
Fact-check agent using Google Fact Check Tools API.
Free tier available with API key, falls back to DuckDuckGo.
"""
import re
from typing import List, Dict, Any

import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim
from config import settings


class FactCheckAgentConfig(AgentConfig):
    """Configuration for fact-check agent."""
    max_results: int = 5
    provider: str = "factcheck"


class FactCheckAgent(BaseAgent):
    """Agent that retrieves evidence using fact-checking sources."""

    def __init__(self):
        config = FactCheckAgentConfig(
            agent_id="factcheck_agent",
            name="FactCheckAgent",
            description="Fact-check agent (Google Fact Check + Snopes)",
            max_results=5,
            provider="factcheck",
        )
        super().__init__(config)

    def _make_client(self) -> httpx.Client:
        return httpx.Client(timeout=30.0, follow_redirects=True)

    def _factcheck_search(self, query: str) -> List[Dict[str, Any]]:
        """Search fact-checking sources."""
        results = []

        # Try Google Fact Check API if key is available
        if settings.search_api_key:
            try:
                with self._make_client() as client:
                    resp = client.get(
                        "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                        params={"query": query, "pageSize": self.config.max_results},
                    )
                    if resp.status_code == 200:
                        claims = resp.json().get("claims", [])
                        for claim in claims:
                            review = claim.get("claimReview", [{}])[0]
                            results.append({
                                "title": claim.get("text", "")[:100],
                                "snippet": review.get("textualRating", ""),
                                "link": review.get("url", ""),
                                "publisher": review.get("publisher", {}).get("name", ""),
                            })
            except Exception as e:
                logger.warning(f"Google Fact Check API failed: {e}")

        # Fallback: search Snopes via DuckDuckGo
        if not results:
            with self._make_client() as client:
                resp = client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": f"site:snopes.com {query}"},
                    headers={"User-Agent": "evidence-mad/1.0"},
                )
                matches = re.findall(
                    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                    resp.text, re.I | re.S,
                )
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</a>',
                    resp.text, re.I | re.S,
                )
                for i, (link, title) in enumerate(matches[: self.config.max_results]):
                    results.append({
                        "title": re.sub(r"<[^>]+>", "", title).strip(),
                        "snippet": re.sub(r"<[^>]+>", "", snippets[i] if i < len(snippets) else "").strip(),
                        "link": link,
                        "publisher": "Snopes",
                    })

        return results

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._factcheck_search(query)

    def retrieve(self, query: str) -> Evidence:
        self.logger.info(f"Retrieving fact-check evidence for: {query!r}")

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
                    "publisher": result.get("publisher", ""),
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
