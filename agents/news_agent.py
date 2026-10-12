"""
News search agent using Google News RSS feed.
Free, no API key required.
"""
import html
import re
from typing import List, Dict, Any
from xml.etree import ElementTree

import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim


class NewsAgentConfig(AgentConfig):
    """Configuration for news search agent."""
    max_results: int = 5
    language: str = "en"
    provider: str = "news"


class NewsAgent(BaseAgent):
    """Agent that retrieves evidence using Google News RSS search."""

    def __init__(self):
        config = NewsAgentConfig(
            agent_id="news_agent",
            name="NewsAgent",
            description="News search agent (Google News RSS)",
            max_results=5,
            provider="news",
        )
        super().__init__(config)

    def _make_client(self) -> httpx.Client:
        return httpx.Client(timeout=30.0, follow_redirects=True)

    def _news_search(self, query: str) -> List[Dict[str, Any]]:
        """Google News RSS search — free, no API key required."""
        with self._make_client() as client:
            resp = client.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "en", "gl": "US", "ceid": "US:en"},
                headers={"User-Agent": "evidence-mad/1.0"},
            )
            resp.raise_for_status()

        root = ElementTree.fromstring(resp.content)
        items = root.findall(".//item")

        results = []
        for item in items[: self.config.max_results]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            description = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")

            # Clean HTML from description
            clean_desc = re.sub(r"<[^>]+>", "", html.unescape(description)).strip()
            # Remove "Source: ..." prefix if present
            clean_desc = re.sub(r"^[^:]+:\s*", "", clean_desc)

            results.append({
                "title": html.unescape(title),
                "snippet": clean_desc,
                "link": link,
                "pub_date": pub_date,
            })

        return results

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._news_search(query)

    def retrieve(self, query: str) -> Evidence:
        self.logger.info(f"Retrieving news evidence for: {query!r}")

        search_results = self.search(query)

        sources = []
        retrieved_texts = []
        for result in search_results:
            source = Source(
                url=result.get("link", ""),
                title=result.get("title", ""),
                snippet=result.get("snippet", ""),
                metadata={"search_query": query, "pub_date": result.get("pub_date", "")},
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
