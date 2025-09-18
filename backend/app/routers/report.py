from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Tuple
from ..core.config import settings
from ..models.evidence import EvidenceItem
from ..services import rag as rag_svc
from ..services import deep_research as dr_svc
from ..services.reporting import compose_markdown
from langchain_openai import AzureChatOpenAI

router = APIRouter(prefix="/report", tags=["report"])

class RequestBody(BaseModel):
    category: str
    gpu: str
    users: str
    cost: str
    special: Optional[str] = None
    natural: Optional[str] = None

def quick_topic(category: str, gpu: str, users: str, cost: str) -> str:
    c = category or "요구사항"; g = gpu or "GPU N/A"; u = users or "사용자 N/A"; b = cost or "예산 N/A"
    return f"{c} | {g} | 사용자 {u} | 예산 {b}"

@router.post("/generate")
def generate_report(req: RequestBody):
    # 1) VectorStore 로드
    vs, docs_info = rag_svc.build_vectorstore(settings.KNOWLEDGE_BASE_PATH)
    # 2) 제목
    title = quick_topic(req.category, req.gpu, req.users, req.cost)
    # 3) RAG
    query = f"성향/용도: {req.category}\nGPU 스펙: {req.gpu}\n예상 사용자 수: {req.users}\n예산/비용 범위: {req.cost}\n"
    docs = rag_svc.run_rag(vs, query) if vs else []
    ev: List[EvidenceItem] = []
    for i, d in enumerate(docs):
        if not getattr(d, 'page_content', None): continue
        meta = getattr(d, 'metadata', {}) or {}
        ev.append(EvidenceItem(
            text=d.page_content[:1200],
            source=f"KB|{meta.get('path') or meta.get('source') or 'doc'}",
            title=meta.get('title') or meta.get('page') or meta.get('source') or meta.get('path'),
            fetched_at=None,
            score=None
        ))
    # 4) 부족 시 Deep Research 보강 (형식상)
    if len(ev) < 7:
        dr_ev = dr_svc.gather_dr_evidence(query, timeframe_days=settings.DR_TIMEFRAME_DAYS, min_sources=settings.DR_MIN_SOURCES)
        ev.extend(dr_ev)
    # Dedup (간단)
    seen = set(); ev2 = []
    for e in ev:
        key = (e.title or '').strip().lower(), (e.source or '').strip().lower()
        if key in seen: continue
        seen.add(key); ev2.append(e)
    ev = ev2[:12]
    # 5) 보고서 생성
    llm = AzureChatOpenAI(
        model=settings.AZURE_CHAT_DEP or "gpt-4o-mini",
        api_key=settings.AZURE_KEY,
        azure_endpoint=settings.AZURE_ENDPOINT,
        api_version=settings.AZURE_API_VERSION,
        temperature=0
    )
    md = compose_markdown(llm, title, ev, rate=settings.USD_KRW_RATE)
    return {"title": title, "evidence_count": len(ev), "markdown": md}
