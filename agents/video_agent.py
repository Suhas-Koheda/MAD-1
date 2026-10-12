"""
Video search agent using YouTube Data API or fallback to DuckDuckGo.
"""
import re
from typing import List, Dict, Any

import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim
from config import settings


class VideoAgentConfig(AgentConfig):
    """Configuration for video search agent."""
    max_results: int = 5
    provider: str = "youtube"


class VideoAgent(BaseAgent):
    """Agent that retrieves evidence using video search."""

    def __init__(self):
        config = VideoAgentConfig(
            agent_id="video_agent",
            name="VideoAgent",
            description="Video search agent (YouTube)",
            max_results=5,
            provider="youtube",
        )
        super().__init__(config)

    def _make_client(self) -> httpx.Client:
        return httpx.Client(timeout=30.0, follow_redirects=True)

    def _youtube_search(self, query: str) -> List[Dict[str, Any]]:
        """YouTube search — uses API key if available, else DuckDuckGo."""
        results = []

        # Try YouTube Data API if key is available
        if settings.search_api_key:
            try:
                with self._make_client() as client:
                    resp = client.get(
                        "https://www.googleapis.com/youtube/v3/search",
                        params={
                            "part": "snippet",
                            "q": query,
                            "maxResults": self.config.max_results,
                            "type": "video",
                            "key": settings.search_api_key,
                        },
                    )
                    if resp.status_code == 200:
                        items = resp.json().get("items", [])
                        for item in items:
                            snippet = item.get("snippet", {})
                            video_id = item.get("id", {}).get("videoId", "")
                            results.append({
                                "title": snippet.get("title", ""),
                                "snippet": snippet.get("description", "")[:300],
                                "link": f"https://youtube.com/watch?v={video_id}",
                                "channel": snippet.get("channelTitle", ""),
                            })
            except Exception as e:
                logger.warning(f"YouTube API failed: {e}")

        # Fallback: DuckDuckGo video search
        if not results:
            with self._make_client() as client:
                resp = client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": f"youtube {query}"},
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
                        "channel": "",
                    })

        return results

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._youtube_search(query)

    def retrieve(self, query: str) -> Evidence:
        self.logger.info(f"Retrieving video evidence for: {query!r}")

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
                    "channel": result.get("channel", ""),
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
