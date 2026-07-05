# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from typing import Dict, List


SEARCH_ENDPOINTS = [
    "https://duckduckgo.com/html/",
    "https://html.duckduckgo.com/html/",
]


def _strip_tags(value: str) -> str:
    value = re.sub(r"<script.*?</script>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<style.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _clean_url(url: str) -> str:
    url = html.unescape(url)
    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return url


def _extract_results(page: str, limit: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    anchor_pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        flags=re.I | re.S,
    )
    for match in anchor_pattern.finditer(page):
        title = _strip_tags(match.group("title"))
        url = _clean_url(match.group("href"))
        if not title or not url or "duckduckgo.com/y.js" in url:
            continue
        after = page[match.end(): match.end() + 1800]
        snippet_match = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</',
            after,
            flags=re.I | re.S,
        )
        snippet = _strip_tags(snippet_match.group("snippet")) if snippet_match else ""
        if any(item["url"] == url for item in results):
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def search_web(query: str, limit: int = 5, timeout: int = 15) -> List[Dict[str, str]]:
    query = (query or "").strip()
    if not query:
        return []
    params = urllib.parse.urlencode({"q": query})
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    errors = []
    for endpoint in SEARCH_ENDPOINTS:
        req = urllib.request.Request(f"{endpoint}?{params}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                page = resp.read().decode("utf-8", errors="replace")
            results = _extract_results(page, limit)
            if results:
                return results
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        return [{"title": "Search failed", "url": "", "snippet": "; ".join(errors[:2])}]
    return []

