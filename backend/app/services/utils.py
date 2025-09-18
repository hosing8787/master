from __future__ import annotations
from typing import Optional, Iterable
import os, re, requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

def normalize_endpoint(url: Optional[str]) -> Optional[str]:
    return (url or None).rstrip('/') if url else url

def get_env_any(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None

def krw_round_manwon(v_krw: float) -> str:
    try:
        man = round(v_krw / 10_000)
        return f"{man:,}만원"
    except Exception:
        return "금액 오류"

def domain_of(u: str) -> str:
    try:
        p = urlparse(u)
        return (p.netloc or '').lower()
    except Exception:
        return ''

BLOCKED_DOMAINS = {"bing.com","youtube.com","aclick"}

def good_domain(domain: str) -> bool:
    d = domain.lower()
    if any(b in d for b in BLOCKED_DOMAINS):
        return False
    return ('.' in d and len(d) > 3)

def has_required_hints(text: str, required: Iterable[str], min_hits: int = 2) -> bool:
    t = text.lower()
    hits = sum(1 for k in required if k in t)
    return hits >= min_hits

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def safe_fetch(url: str, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script","style","noscript"]):
            s.extract()
        txt = soup.get_text("\n", strip=True)
        if len(txt.split()) < 50:
            return "", None
        t = soup.find("meta", {"property":"article:published_time"}) or soup.find("time")
        published = (t.get("content") if t and t.has_attr("content") else (t.get_text(strip=True) if t else None))
        return txt, published
    except Exception:
        return "", None
