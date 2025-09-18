# Evidence → 보고서 생성 (TTD-DR 통합)
from __future__ import annotations
from typing import List, Dict, Any, Optional
import datetime, json, re, os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..models.evidence import EvidenceItem

def replace_usd_to_krw(body_md: str, rate: float = 1400.0) -> str:
    pattern = re.compile(r'(?:(?:USD|usd)\s*)?\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([MmKk]?)', flags=re.IGNORECASE)
    def to_krw(m):
        num = float(m.group(1)); suf = (m.group(2) or "").upper()
        usd = num * (1_000_000 if suf == "M" else 1_000 if suf == "K" else 1.0)
        if usd < 1000: return m.group(0)
        krw = round((usd*rate) / 10_000)
        if f"{m.group(0)} (≈" in body_md: return m.group(0)
        return f"{m.group(0)} (≈{format(krw, ',')}만원)"
    return pattern.sub(to_krw, body_md)

DEFAULT_SECTIONS = [
    "🎯 추천 구성 스펙",
    "📊 성능 및 용량 분석",
    "💰 비용 분석 및 TCO",
    "⚖️ 대안 및 비교 분석",
    "🛠️ 구현 로드맵",
    "✅ 결론 및 권장사항",
    "📚 참고자료(원문)",
]

SECTION_PROMPTS = {
    "🎯 추천 구성 스펙": "표 형식으로 핵심 구성요소/사양/근거 [n]을 제시. 호환성/확장성/벤더 지원 관점 포함.",
    "💰 비용 분석 및 TCO": "CAPEX/OPEX를 3~5년 관점으로 비교. 전력/냉각/유지보수/인건비 포함. 주장 뒤 [n] 인용.",
}

def plan_outline(llm: AzureChatOpenAI, title: str) -> Dict[str, Any]:
    sys = "너는 엔터프라이즈 GPU 시스템 아키텍트다. JSON으로 제목과 섹션 배열만 응답."
    user = f"제목: {title}\n섹션: {', '.join(DEFAULT_SECTIONS)} 포함"
    out = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)]).content
    txt = out.strip().strip('```').replace('json','')
    try:
        data = json.loads(txt)
        heads = [s.get('heading') if isinstance(s, dict) else s for s in data.get('sections', [])]
        heads = [h for h in heads if isinstance(h, str) and h.strip()] or DEFAULT_SECTIONS
        return {'title': data.get('title') or title, 'sections': heads}
    except Exception:
        return {'title': title, 'sections': DEFAULT_SECTIONS}

def compose_markdown(llm: AzureChatOpenAI, title: str, evidence: List[EvidenceItem], rate: float = 1400.0) -> str:
    packed = []
    for i, e in enumerate(evidence, 1):
        ttl = (e.title or e.source or f"evidence-{i}")
        src = (e.source or "")
        snippet = (e.text or "").strip().replace("\n\n","\n")
        packed.append(f"[{i}] {ttl}\nSRC: {src}\n{snippet}")
    ev_text = "\n\n".join(packed) if packed else "No evidence."

    outline = plan_outline(llm, title)
    wanted = set(DEFAULT_SECTIONS)
    sections = [h for h in outline["sections"] if h in wanted]

    body = [f"# {outline['title']}\n",
            f"> **생성 시각(UTC):** {datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')}\n",
            f"> **증거 수:** {len(evidence)}개\n\n"]
    sys = ("'Evidence'만을 근거로 보고서를 작성하라. 모든 주장 뒤에 [n] 인용번호를 붙이고,"
           "불확실한 경우 명시하라. 섹션명/구조는 그대로 사용한다.")
    for h in sections:
        extra = SECTION_PROMPTS.get(h, "")
        prompt = f"제목: {title}\n섹션: {h}\n지침: {extra}\nEvidence:\n{ev_text}"
        try:
            rsp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=prompt)]).content
            body.append(f"## {h}\n\n{rsp.strip()}\n")
        except Exception as e:
            body.append(f"## {h}\n\n(섹션 생성 오류: {e})\n")
    md = "\n".join(body)
    md = replace_usd_to_krw(md, rate=rate)

    # 참고자료 (원문 유지: 환율 병기 금지)
    if evidence:
        refs = ["\n---\n", "## 📚 참고자료(원문)\n", "### 📄 문서 목록\n"]
        seen = set()
        for i, e in enumerate(evidence, 1):
            key = ((e.title or '').strip().lower(), (e.source or '').strip().lower())
            if key in seen: continue
            seen.add(key)
            fetched = f" (수집: {e.fetched_at})" if e.fetched_at else ""
            preview = re.sub(r'\s+', ' ', (e.text or ''))[:200]
            refs.append(f"[{i}] {e.title or '제목 없음'}\n- **출처:** {e.source}{fetched}\n- **미리보기:** {preview}\n")
        md += "\n".join(refs)
    return md
