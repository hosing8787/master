# 웹 딥리서치 로직
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import time, re, datetime, os
from dataclasses import asdict
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS  # new pkg
except Exception:
    try:
        from duckduckgo_search import DDGS  # legacy
    except Exception:
        DDGS = None

try:
    from serpapi import GoogleSearch  # optional
    HAS_SERPAPI = True
except Exception:
    HAS_SERPAPI = False

from .utils import domain_of, has_required_hints, good_domain, safe_fetch
from ..models.evidence import EvidenceItem

def search_ddg(q: str, max_results=10, timeframe_days=365) -> List[Dict[str, Any]]:
    if DDGS is None:
        return []
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=timeframe_days)).date().isoformat()
    hint_q = f"{q} after:{since}"
    out: List[Dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(hint_q, region="kr-kr", max_results=max_results):
                if not r or "href" not in r:
                    continue
                out.append({
                    "title": (r.get("title") or "").strip(),
                    "url": (r.get("href") or "").split("#")[0].strip(),
                    "snippet": r.get("body") or r.get("snippet") or "",
                    "source": r.get("source", "duckduckgo"),
                    "published": None,
                })
    except Exception:
        return []
    return out

def search_serpapi(q: str, max_results=10, timeframe_days=365, gl="kr", hl="ko") -> List[Dict[str, Any]]:
    if not HAS_SERPAPI or not os.getenv("SERPAPI_API_KEY"):
        return []
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=timeframe_days)).date().isoformat()
    try:
        params = {"engine": "google", "q": f"{q} after:{since}", "gl": gl, "hl": hl, "num": min(max_results, 20), "api_key": os.getenv("SERPAPI_API_KEY")}
        results = GoogleSearch(params).get_dict()
        out = []
        for it in results.get("organic_results", []):
            out.append({
                "title": it.get("title", ""),
                "url": (it.get("link") or "").split("#")[0].strip(),
                "snippet": it.get("snippet", ""),
                "source": "google",
                "published": it.get("date", ""),
            })
        return out
    except Exception:
        return []

class SourceDoc:
    def __init__(self, url: str, title: str, snippet: str, published: Optional[str], text: str, domain: str):
        self.url = url; self.title = title; self.snippet = snippet; self.published = published; self.text = text; self.domain = domain

REQUIRED_HINTS = {
    "gpu","nvidia","h100","a100","h200","hgx","nvlink","nvswitch",
    "infiniband","ndr","llm","training","pretrain","tco","cluster",
    "xe9680","mi300","cdna3","cuda","pytorch","tensor core","sxm","pcie"
}

def gather_sources(topic: str, timeframe_days=365, min_sources=12) -> List[SourceDoc]:
    queries = [
        topic,
        f"{topic} TCO budget price",
        f"{topic} NVLink InfiniBand cluster",
        f"{topic} site:nvidia.com",
        f"{topic} site:dell.com",
        f"{topic} filetype:pdf",
    ]
    max_each = int(min_sources) * 2
    pool = []
    for q in queries:
        pool.extend(search_ddg(q, max_results=10, timeframe_days=timeframe_days))
        pool.extend(search_serpapi(q, max_results=10, timeframe_days=timeframe_days))
        time.sleep(0.1)
    seen = set()
    cands = []
    for it in pool:
        url = (it.get("url") or "").split("#")[0].strip()
        if not url:
            continue
        dom = domain_of(url)
        if not good_domain(dom):
            continue
        title_snip = (it.get("title","") + " " + it.get("snippet",""))
        if not has_required_hints(title_snip, REQUIRED_HINTS, min_hits=1):
            continue
        if url in seen:
            continue
        seen.add(url)
        base_hint = sum(1 for h in REQUIRED_HINTS if h in title_snip.lower())
        trust_bonus = 3 if dom in {"nvidia.com","dell.com","supermicro.com","hpe.com","lenovo.com","arxiv.org","ieee.org","microsoft.com","azure.microsoft.com"} else 0
        it["_score"] = base_hint + trust_bonus
        cands.append(it)
    cands.sort(key=lambda x: x.get("_score", 0), reverse=True)
    cands = cands[:max_each]
    docs: List[SourceDoc] = []
    for it in cands:
        url = it["url"]
        text, published = safe_fetch(url)
        if not text or len(text.split()) < 120:
            continue
        if not has_required_hints(text, REQUIRED_HINTS, min_hits=2):
            continue
        docs.append(SourceDoc(url=url, title=it.get("title") or url, snippet=it.get("snippet",""),
                              published=published, text=text, domain=domain_of(url)))
        if len(docs) >= max_each:
            break
    return docs

def gather_dr_evidence(topic: str, timeframe_days=365, min_sources=12, keep_len=1200) -> List[EvidenceItem]:
    web_docs = gather_sources(topic, timeframe_days=timeframe_days, min_sources=min_sources)
    ev: List[EvidenceItem] = []
    for d in web_docs:
        ev.append(EvidenceItem(
            text=(d.text or "")[:keep_len],
            source=f"WEB|{d.url}",
            title=d.title or d.url,
            fetched_at=d.published or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            score=None
        ))
    return ev
