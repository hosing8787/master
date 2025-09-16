# rag_runner_ui.py — 통합본 (A안 단일 보고서, 참조 환율미적용, 웹수집 강화, 원화(만원) 표기)

import os
import re
import json
import time
import hashlib
import warnings
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# ===== LangChain / Chroma =====
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain.schema import Document

# ===== HTML 본문 추출 =====
from bs4 import BeautifulSoup
try:
    import trafilatura  # optional
    HAS_TRAFILATURA = True
except Exception:
    trafilatura = None
    HAS_TRAFILATURA = False

# ===== DDG / SerpAPI (옵션) =====
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

# ===== (선택) Chroma 캐시 클리어 =====
try:
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()
except Exception:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===== 환경 변수 로드 =====
load_dotenv()

# ===== LangSmith (있으면 사용) =====
langsmith_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ===== 통화 변환 유틸 (본문만 적용) =====
def _krw_round_manwon(v_krw: float) -> str:
    """KRW 값을 만원 단위로 반올림해 '1,234만원' 형태 문자열로 반환"""
    try:
        man = round(v_krw / 10_000)
        return f"{man:,}만원"
    except Exception:
        return "금액 오류"

def _usd_to_krw_amount(usd: float, rate: float) -> float:
    try:
        return usd * rate
    except Exception:
        return 0.0

def _extract_usd_amounts(text: str) -> List[Tuple[str, float]]:
    """
    본문에서 $1.5M, 500K, USD 2M 등 USD 금액 패턴을 추출.
    반환: [(원문표현, 값(USD float)), ...]
    """
    results = []
    # $1.5M / 1.5M / USD 1.5M / 500K / $500K / USD 500K / $2000000
    pattern = re.compile(
        r'(?:(?:USD|usd)\s*)?\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([MmKk]?)',
        flags=re.IGNORECASE
    )
    # 너무 광범위한 매칭 방지: 단독 숫자지만 단위 없는 경우는 제외(문장부호/문맥으로 보정)
    for m in pattern.finditer(text):
        full = m.group(0)
        num = float(m.group(1))
        suf = (m.group(2) or "").upper()
        if suf == "M":
            usd = num * 1_000_000
        elif suf == "K":
            usd = num * 1_000
        else:
            # 단위 없는 순수 숫자는 '$' 또는 'USD'가 붙은 경우만 인정
            if not re.search(r'(USD|usd|\$)', full):
                continue
            usd = num
        # 너무 작은 값(예: $10)은 무시 -> 오검 방지
        if usd < 1000:
            continue
        results.append((full.strip(), usd))
    return results

def _replace_usd_to_krw(body_md: str, rate: Optional[float] = None) -> str:
    """
    본문(BODY)에 한해 USD→KRW(만원) 병기를 수행.
    참고자료(원문) 미리보기에는 적용하지 않는다. (본문만 호출)
    """
    if rate is None:
        # 보수적 고정 환율(원/달러) — 필요시 환경변수로 조정
        try:
            rate = float(os.getenv("USD_KRW_RATE", "1400"))
        except Exception:
            rate = 1400.0

    # 같은 위치에서 중복 변환되지 않도록, 큰 덩어리부터 치환
    found = _extract_usd_amounts(body_md)
    if not found:
        return body_md

    # 긴 문자열에서 여러 번 re.sub하면 오프셋이 흔들릴 수 있으니 순차 치환
    out = body_md
    used_spans: set = set()
    # 뒤에서 앞으로 치환하면 인덱스 보존에 유리하지만, 여기서는 단순 텍스트 치환 사용
    for raw, usd_val in sorted(found, key=lambda x: -len(x[0])):
        krw_val = _usd_to_krw_amount(usd_val, rate)
        krw_text = _krw_round_manwon(krw_val)
        # 이미 병기된 경우 방지
        if f"{raw} (≈" in out or f"{raw} (~" in out:
            continue
        out = out.replace(raw, f"{raw} (≈{krw_text})")
    return out

# ===== 유틸 =====
def _normalize_endpoint(url: Optional[str]) -> Optional[str]:
    return (url or None).rstrip("/") if url else url

def _get_env_any(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

# ===== Azure OpenAI 빌더 =====
def build_embeddings() -> AzureOpenAIEmbeddings:
    endpoint = _normalize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    api_key = _get_env_any("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
    deployment = _get_env_any(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING",
        "AZURE_EMBEDDING_DEPLOYMENT",
        "AZURE_EMBED_DEPLOYMENT",
        "AZURE_EMBEDDING_MODEL",
        "AZURE_EMBEDDINGS_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    missing = []
    if not endpoint:   missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:    missing.append("AZURE_OPENAI_API_KEY or AZURE_OPENAI_KEY")
    if not deployment: missing.append("AZURE_*_EMBEDDING_* (deployment/model) 이름")
    if missing:
        st.error(f"❌ Azure 임베딩 환경변수 부족: {', '.join(missing)}")
        st.stop()

    return AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

def build_chat_llm() -> AzureChatOpenAI:
    endpoint = _normalize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    api_key = _get_env_any("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
    deployment = _get_env_any(
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_LLM_DEPLOYMENT",
        "AZURE_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
        "AZURE_OPENAI_MODEL_DEPLOYMENT"
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    missing = []
    if not endpoint:   missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:    missing.append("AZURE_OPENAI_API_KEY or AZURE_OPENAI_KEY")
    if not deployment: missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT or AZURE_LLM_DEPLOYMENT or AZURE_CHAT_DEPLOYMENT")
    if missing:
        st.error(f"❌ Azure 채팅 환경변수 부족: {', '.join(missing)}")
        st.stop()

    return AzureChatOpenAI(
        model=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        temperature=0,
    )

# ===== 간단 에이전트 =====
class SimpleGraph:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    def invoke(self, state: dict) -> dict:
        msgs = state.get("messages", [])
        reply = self.llm.invoke(msgs).content
        return {"messages": msgs + [AIMessage(content=reply)]}

def create_agent(retriever_tool, llm: AzureChatOpenAI):
    return SimpleGraph(llm)

# ===== 세션 =====
def initialize_session_state():
    keys = [
        ("messages", [AIMessage(content="무엇을 도와드릴까요?")]),
        ("vectorstore", None),
        ("current_files_hash", None),
        ("docs_info", None),
        ("graph", None),
        ("llm", None),
        ("loaded_once", False),
        ("system_prompt", None),
        ("final_report_md", None),
        ("report_topic", ""),
        ("decision_path", []),
        ("dr_policy", {}),
    ]
    for k, v in keys:
        if k not in st.session_state:
            st.session_state[k] = v

# ===== 폴더 처리 =====
ALLOWED_EXTS = {".txt", ".pdf", ".md"}

def list_folder_files(folder_path: str) -> List[Path]:
    base = Path(folder_path).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return []
    files: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return sorted(files)

def get_folder_hash(files: List[Path]) -> str:
    sha = hashlib.sha256()
    for p in files:
        try:
            sha.update(str(p.resolve()).encode("utf-8"))
            sha.update(str(p.stat()).encode("utf-8"))
            with open(p, "rb") as f:
                sha.update(f.read())
        except Exception:
            continue
    return sha.hexdigest()

def _assert_embedding_ready(emb) -> None:
    try:
        vec = emb.embed_query("healthcheck")
        if not isinstance(vec, list) or len(vec) == 0:
            raise RuntimeError("임베딩 결과가 비어 있습니다.")
    except Exception as e:
        st.error(
            "❌ 임베딩 호출 실패: Azure 설정을 확인하세요.\n"
            f"- AZURE_OPENAI_ENDPOINT={os.getenv('AZURE_OPENAI_ENDPOINT')}\n"
            f"- AZURE_OPENAI_API_VERSION={os.getenv('AZURE_OPENAI_API_VERSION')}\n"
            f"- 임베딩 배포/모델 이름(여러 키 허용)\n"
            f"원인: {e}"
        )
        st.stop()

# ===== Vectorstore / Retriever =====
def process_folder(folder_path: str):
    files = list_folder_files(folder_path)
    if not files:
        st.info(f"폴더에 처리할 파일이 없습니다: {folder_path}")
        return None, None

    current_hash = get_folder_hash(files)

    if st.session_state.current_files_hash == current_hash and st.session_state.vectorstore is not None:
        retriever = st.session_state.vectorstore.as_retriever()
        try:
            _ = retriever.invoke("health check")
        except Exception:
            pass
        docs_info = st.session_state.docs_info or [
            {"name": p.name, "type": ("application/pdf" if p.suffix.lower()==".pdf" else "text/plain"),
             "size": p.stat().st_size, "path": str(p)} for p in files
        ]
        desc = f"Search through: {', '.join(p['name'] for p in docs_info)}"
        retriever_tool = create_retriever_tool(retriever, "search_docs", desc)
        return retriever_tool, docs_info

    # 새로운 빌드
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=100)
    all_docs: List[Document] = []
    docs_info: List[dict] = []

    for p in files:
        try:
            info = {"name": p.name, "type": ("application/pdf" if p.suffix.lower()==".pdf" else "text/plain"),
                    "size": p.stat().st_size, "path": str(p)}
            if p.suffix.lower() in (".txt", ".md"):
                text = p.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    raw = [Document(page_content=text, metadata={"source": p.name, "path": str(p)})]
                    all_docs.extend(splitter.split_documents(raw))
                    docs_info.append(info)
            elif p.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(p))
                raw_docs = loader.load()
                raw_docs = [d for d in raw_docs if d.page_content and d.page_content.strip()]
                for d in raw_docs:
                    d.metadata["source"] = p.name
                    d.metadata["path"] = str(p)
                if raw_docs:
                    all_docs.extend(splitter.split_documents(raw_docs))
                    docs_info.append(info)
        except Exception as e:
            st.error(f"Error processing {p.name}: {e}")

    if not all_docs:
        st.error("❌ 인덱싱할 문서가 없습니다. KB 폴더와 확장자(.md/.txt/.pdf)를 확인하세요.")
        return None, None

    embeddings = build_embeddings()
    _assert_embedding_ready(embeddings)

    persist_dir = os.getenv("VECTORSTORE_PATH", "").strip()
    collection_name = f"kb_{current_hash[:12]}"

    if persist_dir:
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            collection_name=collection_name,
        )

    st.session_state.vectorstore = vectorstore
    st.session_state.current_files_hash = current_hash
    st.session_state.docs_info = docs_info

    retriever = vectorstore.as_retriever()
    desc = f"Search through: {', '.join(d['name'] for d in docs_info)}"
    retriever_tool = create_retriever_tool(retriever, "search_docs", desc)
    return retriever_tool, docs_info

# ===== System 프롬프트(강화) =====
def build_system_prompt(docs_info: List[dict]) -> str:
    file_names = ", ".join(d.get("name", "unknown") for d in (docs_info or []))
    # Few-shot 포함, 전문성·정확성 강조
    examples = """
예시 1:
질문: "H100 8GPU 서버의 메모리 대역폭은?"
답: "업로드된 문서에서 해당 수치를 찾지 못했습니다. HBM 대역폭 수치 확인이 필요합니다."

예시 2:
질문: "A100과 H100의 AI 성능 차이는?"
답: "문서에 구체 벤치마크가 없습니다. 추가 성능 비교 자료 필요."
"""
    return (
        "당신은 20년 경력의 HPC/GPU 시스템 아키텍트입니다. 다음 전문 영역을 보유하고 있습니다:\n"
        "- 엔터프라이즈 GPU 클러스터 설계 (NVIDIA HGX, AMD MI 시리즈)\n"
        "- InfiniBand/Ethernet 네트워크 토폴로지 최적화\n"
        "- 전력/냉각 인프라 설계\n"
        "- TCO 모델링 및 벤더 협상 전략\n\n"
        "반드시 업로드된 문서에서만 근거를 사용하세요. 모르면 모른다고 답하세요.\n"
        f"검색 대상 문서: {file_names}\n\n"
        f"{examples}\n"
    )

# ===== 4필드 템플릿 =====
CATEGORIES = [
    "대규모 AI 트레이닝 특화 (Hyperscale AI Training)",
    "중형 연구 HPC (Research HPC Cluster)",
    "중형 기관·산업 HPC (Enterprise/Institutional HPC)",
    "AI/LLM 특화 서버 (Fine-tuning & Inference)",
    "소규모 연구실 서버 (Lab-scale GPU Server)",
    "단품 GPU/소형 서버 (Single-GPU/Small-scale)",
]

DEFAULT_SECTIONS = [
    "🎯 추천 구성 스펙",
    "📊 성능 및 용량 분석",
    "💰 비용 분석 및 TCO",
    "⚖️ 대안 및 비교 분석",
    "🛠️ 구현 로드맵",
    "✅ 결론 및 권장사항",
    "📚 참고자료(원문)",
]

def build_quick_query_prompt(category: str, gpu: str, users: str, cost: str) -> str:
    gpu_hints = []
    g = (gpu or "").lower()
    if "h100" in g:
        gpu_hints.extend(["H100 80GB", "HGX H100", "8x H100", "8× H100", "NVLink", "SXM"])
    if ("8" in g and "100" in g) or "8x" in g or "8×" in g:
        gpu_hints.append("8-GPU")
    parts = [
        f"성향/용도: {category}",
        f"GPU 스택: {gpu}",
        f"예상 사용자 수: {users}",
        f"예산/비용 범위: {cost}",
        "검색 힌트: (HGX H100 OR H100 80GB OR 8x H100 OR 8× H100 OR NVLink) "
        "AND (cost OR budget OR price OR TCO) AND (user OR seat OR concurrency) "
        "AND (training OR LLM OR fine-tune OR pretraining OR inference)",
        f"키워드 힌트: {', '.join(gpu_hints)}" if gpu_hints else "",
    ]
    return "\n".join([p for p in parts if p])

def build_topic_title(category: str, gpu: str, users: str, cost: str) -> str:
    c = (category or "").strip() or "요구사항"
    g = (gpu or "").strip() or "GPU N/A"
    u = (users or "").strip() or "사용자 N/A"
    b = (cost or "").strip() or "예산 N/A"
    return f"{c} | {g} | 사용자 {u} | 예산 {b}"

# ===== 자연어 요구사항 파서 =====
def parse_freeform_requirements(text: str) -> Dict[str, str]:
    t = (text or "").strip()
    out = {"category": "", "gpu": "", "users": "", "cost": "", "special": ""}
    if not t:
        return out

    m = re.search(r"(HGX\s*H100|H100|A100|H200|L40S|RTX\s*6000|MI300)[^\n,;]*", t, re.IGNORECASE)
    out["gpu"] = m.group(0) if m else ""

    m = re.search(r"(\d+\s*[–\-~]\s*\d+|\d+)\s*(?:명|users?)", t, re.IGNORECASE)
    out["users"] = (m.group(1) if m else "").replace(" ", "")

    m = re.search(r"(\$\s?\d+(\.\d+)?\s*[MK]|[0-9]+(?:\s?억)?\s?원|\d+\s?K|\d+\s?M|USD\s?\d+(\.\d+)?\s*[MK])", t, re.IGNORECASE)
    out["cost"] = m.group(1) if m else ""

    low = t.lower()
    if any(k in low for k in ["hyperscale", "대규모", "pretrain"]):
        out["category"] = CATEGORIES[0]
    elif any(k in low for k in ["연구", "research", "hpc"]):
        out["category"] = CATEGORIES[1]
    elif any(k in low for k in ["기관", "enterprise", "institution"]):
        out["category"] = CATEGORIES[2]
    elif any(k in low for k in ["inference", "fine-tune", "추론", "파인튜닝"]):
        out["category"] = CATEGORIES[3]
    elif any(k in low for k in ["연구실", "lab", "소규모"]):
        out["category"] = CATEGORIES[4]
    else:
        out["category"] = CATEGORIES[0]

    out["special"] = t
    return out

# ===== Evidence 구조 =====
@dataclass
class EvidenceItem:
    text: str
    source: str
    title: Optional[str] = None
    fetched_at: Optional[str] = None
    score: Optional[float] = None  # RAG 점수(간이)

# ===== 보고서 컴포저 =====
SECTION_EXPERT_PROMPTS = {
    "🎯 추천 구성 스펙": """
당신은 NVIDIA Elite 파트너사의 수석 솔루션 아키텍트입니다.
다음 관점에서 표를 작성하세요:

기술적 정확성:
- 정확한 파트 넘버 (예: NVIDIA HGX H100 8-GPU, Dell PowerEdge XE9680)
- 호환성 매트릭스 검증
- 인증된 구성만 권장

비즈니스 고려사항:
- 공급 가능성 및 리드타임
- 확장성 로드맵
- 벤더 지원 레벨

표 형식:
| 구성요소 | 권장 사양(모델/수량/토폴로지) | 근거 [n] | 비고 |
""",
    "💰 비용 분석 및 TCO": """
당신은 Fortune 500 IT 조달 담당자입니다.
다음 TCO 모델을 적용하세요:

초기 투자 (CAPEX):
- 하드웨어 정가 vs 협상 예상가
- 설치/구성 비용
- 초기 교육/인증 비용

운영비 (OPEX):
- 전력비 (kWh 단가 포함)
- 냉각비 (PUE 고려)
- 유지보수 (연간 %로 산정)
- 인건비 (FTE 기준)

3-5년 총 비용과 대안 시나리오 비교 필수.
"""
}

def _plan_outline(llm: AzureChatOpenAI, topic: str) -> Dict:
    pythonsys = """당신은 20년 경력의 HPC/GPU 시스템 아키텍트입니다. 다음 전문 영역을 보유하고 있습니다:
- 엔터프라이즈 GPU 클러스터 설계 (NVIDIA HGX, AMD MI 시리즈)
- InfiniBand/Ethernet 네트워크 토폴로지 최적화
- 전력/냉각 인프라 설계
- TCO 모델링 및 벤더 협상 전략

목차 생성 시 다음을 준수하세요:
1. 기술적 정확성: 실제 제품 모델명, 정확한 사양, 호환성 고려
2. 비즈니스 관점: ROI, 확장성, 리스크 분석 포함
3. 구현 가능성: 현실적 일정, 공급망, 인력 요구사항
4. 산업 표준: IEEE, OCP, Green500 등 표준 준수

JSON 형식: {"title": str, "sections": [{"heading": str, "focus": str, "technical_depth": str}]}"""
    sys = pythonsys
    user = (f"주제: {topic}\n요구사항: {', '.join(DEFAULT_SECTIONS)} 포함")
    out = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)]).content
    txt = out.strip().strip("```").replace("json", "")
    try:
        data = json.loads(txt)
        heads = []
        for s in data.get("sections", []):
            if isinstance(s, dict):
                h = s.get("heading") or s.get("title", "")
            else:
                h = str(s)
            if h:
                heads.append(h)
        if not heads:
            heads = DEFAULT_SECTIONS
        return {"title": data.get("title") or f"{topic} — 시스템 구성 전문 보고서", "sections": heads}
    except Exception:
        return {"title": f"{topic} — 시스템 구성 전문 보고서", "sections": DEFAULT_SECTIONS}

def _short_preview(text: str, maxlen: int = 200) -> str:
    if not text:
        return ""
    t = re.sub(r'\s+', ' ', text)
    # 원문 유지(환율 병기 금지), 특수문자는 가볍게 정리만
    t = re.sub(r'[\r\t]', ' ', t)
    return (t[:maxlen] + "...") if len(t) > maxlen else t

def _compose_enhanced_markdown_from_evidence(
    llm: AzureChatOpenAI,
    topic: str,
    evidence: List[EvidenceItem],
    outline: Optional[Dict] = None,
    min_sent: int = 8,
    max_sent: int = 15,
) -> str:
    """
    RAG/DeepResearch 통합 Evidence로 한 번만 작성.
    - 본문(섹션)은 원화 병기 변환 적용
    - 참고자료(원문)는 변환 금지 (원문 그대로)
    """
    # Evidence 패킹(LLM 인용번호 기준)
    packed = []
    for i, e in enumerate(evidence, 1):
        ttl = (e.title or e.source or f"evidence-{i}").replace("KB|", "").replace("WEB|", "")
        snippet = (e.text or "").strip().replace("\n\n", "\n")
        src = (e.source or "").replace("KB|", "").replace("WEB|", "")
        packed.append(f"[{i}] {ttl}\nSRC: {src}\n{snippet}")
    ev_text = "\n\n".join(packed) if packed else "No evidence."

    if outline is None:
        outline = _plan_outline(llm, topic)
    wanted = set(DEFAULT_SECTIONS)
    sections = []
    seen_h = set()
    for h in outline.get("sections", []):
        if h in wanted and h not in seen_h:
            sections.append(h)
            seen_h.add(h)

    # 본문(body) 구성
    body_parts = []
    body_parts.append(f"# {outline['title']}\n")
    body_parts.append(f"> **생성 시각(UTC):** {datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')}\n")
    body_parts.append(f"> **분석 대상:** {topic}\n")
    body_parts.append(f"> **Evidence 소스 수:** {len(evidence)}개\n\n")

    sys = (
        "너는 'HPC/GPU 시스템 아키텍처 전문가'다. 아래 Evidence만을 근거로 한국어 보고서를 작성하라.\n"
        "요구사항: 1) 구체 모델/수량/토폴로지 2) 각 주장 뒤 [n] 인용번호 3) Evidence 밖 지식 금지 "
        "4) 불확실은 '추가 검토 필요' 명시 5) 섹션명 중복 금지 6) 표 섹션은 표만 작성"
    )
    section_prompts = SECTION_EXPERT_PROMPTS

    for heading in sections:
        section_specific = section_prompts.get(heading, "")
        prompt = (
            f"주제: {topic}\n"
            f"섹션: {heading}\n\n"
            f"{section_specific}\n\n"
            f"Evidence:\n{ev_text}\n\n"
            f"요구사항: {min_sent}~{max_sent}문장(표 섹션은 표만), 구체 수치/모델, 모든 주장 끝에 [n] 인용번호, Evidence 밖 지식 금지."
        )
        try:
            rsp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=prompt)]).content
            body_parts.append(f"## {heading}\n\n{rsp.strip()}\n")
        except Exception as e:
            body_parts.append(f"## {heading}\n\n⚠️ 섹션 생성 오류: {e}\n")

    # 본문만 환율 변환
    body_md = "\n".join(body_parts)
    body_md = _replace_usd_to_krw(body_md)

    # 참고자료(원문) — 변환 금지
    refs_parts = []
    if evidence:
        refs_parts.append("\n---\n")
        refs_parts.append("## 📚 참고자료(원문)\n")
        refs_parts.append("### 📄 문서 목록\n")
        seen = set()
        for i, e in enumerate(evidence, 1):
            title = (e.title or "제목 없음").replace("KB|", "").replace("WEB|", "")
            source = (e.source or "").replace("KB|", "").replace("WEB|", "")
            key = (title.strip().lower(), source.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            fetched = f" (수집: {e.fetched_at})" if e.fetched_at else ""
            preview = _short_preview(e.text or "", maxlen=200)
            refs_parts.append(f"[{i}] {title}\n")
            refs_parts.append(f"- **출처:** {source}{fetched}\n")
            if preview:
                refs_parts.append(f"- **미리보기:** {preview}\n")
            refs_parts.append("")
    refs_md = "\n".join(refs_parts)

    return body_md + refs_md

# ===== RAG 검색 =====
def run_rag(query: str) -> List[Document]:
    if st.session_state.vectorstore is None:
        return []
    retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 60})
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)
    return docs

def _evidence_from_docs(docs: List[Document], query: str, keep_len: int = 1200) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    q_terms = [t for t in re.split(r"\W+", query.lower()) if t]
    for i, d in enumerate(docs):
        if not d.page_content:
            continue
        meta = d.metadata or {}
        content_lower = d.page_content.lower()
        score = sum(1 for term in q_terms if term in content_lower) / max(1, len(q_terms))
        items.append(EvidenceItem(
            text=d.page_content[:keep_len],
            source=f"KB|{meta.get('path') or meta.get('source') or meta.get('file_path') or meta.get('url') or f'doc_{i}'}",
            title=meta.get("title") or meta.get("page") or meta.get("source") or meta.get("path"),
            fetched_at=meta.get("fetched_at") or meta.get("date") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            score=score,
        ))
    return items

# ===== Deep Research (웹 수집: 강화 필터 포함) =====
def _domain_of(u: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(u)
        return (p.netloc or "").lower()
    except Exception:
        return ""

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

BLOCKED_DOMAINS = {
    "hyperscale.com", "dupontregistry.com", "funbe", "tkor", "namu.wiki",
    "redrosa.co.kr", "urimal.org", "bing.com", "aclick", "jwd1.org",
    "dataannotation.tech", "fliphtml5.com", "butterflyinvest.com",
    "makeit27.com", "blog.makeit27.com", "wipi.ai.kr", "mindpath.tistory.com",
    "uproger.com", "technical.city", "youtube.com", "www.youtube.com",
    "bing.com/aclick"
}

TRUSTED_DOMAINS = {
    "nvidia.com", "developer.nvidia.com", "docs.nvidia.com", "blogs.nvidia.com",
    "dell.com", "www.dell.com",
    "supermicro.com", "www.supermicro.com",
    "hpe.com", "www.hpe.com",
    "lenovo.com", "www.lenovo.com",
    "microsoft.com", "learn.microsoft.com", "azure.microsoft.com",
    "techcommunity.microsoft.com",
    "arxiv.org", "ieee.org",
    "idtechex.com",
    "ktcloud.com", "tech.ktcloud.com",
    "tomshardware.com"
}

ALLOWED_HINTS = ["nvidia", "dell", "h100", "hgx", "nvlink", "a100", "gpu", "h200", "xe9680", "tco", "price", "budget"]

REQUIRED_HINTS = {
    "gpu","nvidia","h100","a100","h200","hgx","nvlink","nvswitch",
    "infiniband","ndr","llm","training","pretrain","tco","cluster",
    "xe9680","mi300","cdna3","cuda","pytorch","tensor core","sxm","pcie"
}

def _safe_fetch(url: str, timeout=20) -> Tuple[str, Optional[str]]:
    try:
        import requests
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        if HAS_TRAFILATURA:
            try:
                downloaded = trafilatura.fetch_url(r.url, no_ssl=True, timeout=timeout)
                if downloaded:
                    txt = trafilatura.extract(downloaded, include_comments=False, include_tables=False, include_images=False)
                    if txt and len(txt.split()) > 50:
                        soup = BeautifulSoup(r.text, "html.parser")
                        t = soup.find("meta", {"property": "article:published_time"}) or soup.find("time")
                        published = (t.get("content") if t and t.has_attr("content") else (t.get_text(strip=True) if t else None))
                        return txt.strip(), published
            except Exception:
                pass
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        txt = soup.get_text("\n", strip=True)
        if len(txt.split()) < 50:
            return "", None
        t = soup.find("meta", {"property": "article:published_time"}) or soup.find("time")
        published = (t.get("content") if t and t.has_attr("content") else (t.get_text(strip=True) if t else None))
        return txt, published
    except Exception:
        return "", None

def _search_ddg(q: str, max_results=10, timeframe_days=365) -> List[Dict[str, Any]]:
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

def _search_serpapi(q: str, max_results=10, timeframe_days=365, gl="kr", hl="ko") -> List[Dict[str, Any]]:
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

@dataclass
class SourceDoc:
    url: str
    title: str
    snippet: str
    published: Optional[str]
    text: str
    domain: str

def _normalize_topic_for_web(topic: str) -> str:
    """웹 검색용으로 라벨/불용어 제거."""
    lines = [l.strip() for l in topic.splitlines() if l.strip()]
    keep = []
    for l in lines:
        l = re.sub(r"^(성향/용도|GPU\s*스택|GPU\s*스펙|예상\s*사용자\s*수|예산/비용\s*범위|특별\s*요구사항)\s*:\s*", "", l, flags=re.IGNORECASE)
        l = re.sub(r"^(검색\s*힌트|키워드\s*힌트)\s*:\s*", "", l, flags=re.IGNORECASE)
        l = l.replace("성향", "")
        keep.append(l)
    q = " ".join(keep)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _good_domain(domain: str) -> bool:
    d = domain.lower()
    if any(b in d for b in BLOCKED_DOMAINS):
        return False
    if d in TRUSTED_DOMAINS:
        return True
    if len(d) < 4 or "." not in d:
        return False
    return not any(x in d for x in ["aclick", "adservice", "doubleclick", "utm"])

def _has_required_hints(text: str, min_hits: int = 2) -> bool:
    t = text.lower()
    hits = sum(1 for k in REQUIRED_HINTS if k in t)
    return hits >= min_hits

def _gather_sources(topic: str, policy: Dict[str, Any]) -> List[SourceDoc]:
    base = _normalize_topic_for_web(topic)

    tokens = []
    gmatch = re.findall(r"(HGX\s*H100|H100|A100|H200|MI300|XE9680|NVLINK|NVSWITCH|INFINIBAND)", base, flags=re.IGNORECASE)
    if gmatch:
        tokens.extend(gmatch)
    anchor = ' '.join(tokens) if tokens else base

    queries = [
        base,
        f"{anchor} TCO budget price",
        f"{anchor} NVLink InfiniBand cluster",
        f"{anchor} site:nvidia.com",
        f"{anchor} site:dell.com",
        f"{anchor} filetype:pdf",
    ]

    timeframe = int(policy.get("timeframe_days", 365))
    max_each = int(policy.get("min_sources", 12)) * 2

    pool: List[Dict[str, Any]] = []
    for q in queries:
        pool.extend(_search_ddg(q, max_results=10, timeframe_days=timeframe))
        pool.extend(_search_serpapi(q, max_results=10, timeframe_days=timeframe))
        time.sleep(0.15)

    seen = set()
    cands: List[Dict[str, Any]] = []
    for it in pool:
        url = (it.get("url") or "").split("#")[0].strip()
        if not url:
            continue
        dom = _domain_of(url)
        if not _good_domain(dom):
            continue

        title = it.get("title", "") or ""
        snippet = it.get("snippet", "") or ""
        title_snip = (title + " " + snippet)

        # GPU/HPC 키워드 최소 매칭
        if not _has_required_hints(title_snip, min_hits=1):
            continue

        key = url
        if key in seen:
            continue
        seen.add(key)

        base_hint = sum(1 for h in REQUIRED_HINTS if h in title_snip.lower())
        trust_bonus = 3 if dom in TRUSTED_DOMAINS else 0
        it["_score"] = base_hint + trust_bonus

        cands.append(it)

    cands.sort(key=lambda x: x.get("_score", 0), reverse=True)
    cands = cands[:max_each]

    docs: List[SourceDoc] = []
    for it in cands:
        url = it["url"]
        text, published = _safe_fetch(url)
        if not text or len(text.split()) < 120:
            continue
        if not _has_required_hints(text, min_hits=2):
            continue
        dom = _domain_of(url)
        docs.append(
            SourceDoc(
                url=url,
                title=it.get("title") or url,
                snippet=it.get("snippet", ""),
                published=published,
                text=text,
                domain=dom
            )
        )
        if len(docs) >= max_each:
            break
    return docs

# === Deep Research → Evidence 변환 ===
def gather_dr_evidence(topic: str, policy: Dict[str, Any], keep_len: int = 1200) -> List[EvidenceItem]:
    web_docs = _gather_sources(topic, policy)
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

# === Evidence 중복/도메인 제한 ===
def dedup_evidence(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    out: List[EvidenceItem] = []
    for e in evidence:
        key = ((e.title or "").strip().lower(), (e.source or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out

def _limit_refs_by_domain(evidence: List[EvidenceItem], max_total=12, max_per_domain=3) -> List[EvidenceItem]:
    by_domain: Dict[str, int] = {}
    out = []
    for e in evidence:
        src = (e.source or "")
        dom = _domain_of(src.replace("WEB|","")) if "WEB|" in src else "KB"
        cnt = by_domain.get(dom, 0)
        if cnt >= max_per_domain:
            continue
        by_domain[dom] = cnt + 1
        out.append(e)
        if len(out) >= max_total:
            break
    return out

# ===== 요구사항 분석 & 충분성 판정 =====
def analyze_requirements(category: str, gpu: str, users: str, cost: str, special_req: str) -> Dict[str, Any]:
    txt = " ".join([category or "", gpu or "", users or "", cost or "", special_req or ""]).lower()
    kws = [t for t in re.split(r"\W+", txt) if t]
    return {"keywords": kws[:64], "raw": txt}

def assess_sufficiency(docs: List[Document], req: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    if not docs:
        return False, {"hits": 0, "covered_keywords": 0, "need": {"hits": 7, "kw": 6}}
    content = "\n".join([d.page_content[:500].lower() for d in docs if d.page_content])
    covered = sum(1 for k in set(req["keywords"]) if k and k in content)
    enough = (len(docs) >= 7) and (covered >= max(6, len(set(req["keywords"])) // 5))
    return enough, {"hits": len(docs), "covered_keywords": covered, "need": {"hits": 7, "kw": 6}}

# ===== Streamlit 앱 =====
st.set_page_config(page_title="TTD-DR HPC/GPU System Designer", layout="wide")
st.title("🧬 GPU 인프라 TA를 위한 지능형 보고서 Agent")

initialize_session_state()
if st.session_state.llm is None:
    try:
        st.session_state.llm = build_chat_llm()
    except Exception as e:
        st.error(f"LLM 초기화 실패: {e}")

# 사이드바
st.sidebar.header("📁 Knowledge Base")
default_folder = os.getenv("KNOWLEDGE_BASE_PATH") or os.getenv("PDF_DOCUMENTS_DIR") or "./documents"
folder_path = st.sidebar.text_input("문서 폴더 경로", value=str(Path(default_folder).resolve()))
scan = st.sidebar.button("🔄 폴더 스캔/재로딩")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 딥리서치(내장)")
timeframe_days = st.sidebar.number_input("최근 N일", min_value=30, max_value=3650, value=int(os.getenv("DR_TIMEFRAME_DAYS", "365")))
min_sources = st.sidebar.number_input("최소 출처 수", min_value=4, max_value=100, value=int(os.getenv("DR_MIN_SOURCES", "12")))
st.session_state.dr_policy = dict(timeframe_days=timeframe_days, min_sources=min_sources)

# 초기 로드/스캔
if (not st.session_state.loaded_once) or scan:
    with st.spinner("📂 폴더 스캔 중..."):
        retriever_tool, docs_info = process_folder(folder_path)
        if retriever_tool and docs_info:
            try:
                if st.session_state.llm is None:
                    st.session_state.llm = build_chat_llm()
                st.session_state.graph = create_agent(retriever_tool, st.session_state.llm)
            except Exception as e:
                st.error(f"❌ 에이전트 생성 실패: {e}")
            st.session_state.system_prompt = build_system_prompt(docs_info)
            st.success(f"✅ {len(docs_info)}개 문서 처리 완료!")
            msg = "\n".join([
                f"📚 폴더에서 문서를 불러왔습니다: {folder_path}",
                *[f"- {d['name']} ({'PDF' if 'pdf' in d['type'].lower() else 'Text'}, {d['size']/1024:.1f}KB)" for d in docs_info],
                "\n이제 문서 기반으로 질문해주세요!",
            ])
            st.session_state.messages.append(AIMessage(content=msg))
            st.session_state.loaded_once = True

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
        st.write(message.content)

# ===== 입력 UI =====
st.markdown("### 🎯 시스템 구성 요구사항 입력")
with st.form(key="enhanced_query_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("성향/용도", CATEGORIES, index=0)
        gpu = st.text_input("GPU 스펙", value="HGX H100 8×80GB")
    with col2:
        users = st.text_input("예상 사용자 수", value="80–100명")
        cost = st.text_input("총 예산", value="$1.5–2M")
    special_req = st.text_area("특별 요구사항(선택)", placeholder="예: 고가용성, DR, 특정 SW 호환 등", height=80)
    nl_text = st.text_area("✍️ 자연어 요구사항 (선택) — 입력 시 위 4필드를 자동 보완/덮어쓰기", height=100)
    submitted = st.form_submit_button("🚀 TTD-DR 보고서 생성", type="primary")

if submitted:
    st.session_state.decision_path = []

    if nl_text and nl_text.strip():
        parsed = parse_freeform_requirements(nl_text)
        category = parsed["category"] or category
        gpu      = parsed["gpu"] or gpu
        users    = parsed["users"] or users
        cost     = parsed["cost"] or cost
        if parsed["special"]:
            special_req = parsed["special"]

    req = analyze_requirements(category, gpu, users, cost, special_req)
    st.info("요구사항 키워드: " + ", ".join(req["keywords"][:12]))

    search_query_text = build_quick_query_prompt(category, gpu, users, cost)
    if special_req:
        search_query_text += f"\n특별 요구사항: {special_req}"

    topic_title = build_topic_title(category, gpu, users, cost)

    with st.spinner("🔎 RAG 검색 중…"):
        docs = run_rag(search_query_text)
    st.success(f"RAG 결과 문서 수: {len(docs)}")

    enough, diag = assess_sufficiency(docs, req)
    st.write("**충분성 판단** → ", "충분 ✅" if enough else "부족 ❌", diag)

    # A안: 통합 Evidence
    combined_ev: List[EvidenceItem] = []
    # RAG에서 우선 수집
    combined_ev.extend(_evidence_from_docs(docs, search_query_text, keep_len=1200))

    if not enough:
        st.info("충분성 부족 → 웹 딥리서치(Evidence 보강) 수행")
        policy = st.session_state.get("dr_policy", {})
        dr_ev = gather_dr_evidence(search_query_text, policy, keep_len=1200)
        combined_ev.extend(dr_ev)

    # 중복 제거 및 도메인 제한
    combined_ev = dedup_evidence(combined_ev)
    combined_ev = _limit_refs_by_domain(combined_ev, max_total=12, max_per_domain=3)

    # 단일 보고서 생성 (TTD-DR 최종)
    with st.spinner("🧬 TTD-DR 보고서 생성 중…(통합 Evidence)"):
        merged_md = _compose_enhanced_markdown_from_evidence(st.session_state.llm, topic_title, combined_ev)

    st.session_state.final_report_md = merged_md
    st.session_state.report_topic = topic_title
    st.success("✅ 최종 보고서 생성 완료!")

# 보고서 출력
if st.session_state.get("final_report_md"):
    st.markdown("---")
    st.markdown("## 📋 **최종 보고서 (TTD-DR, 통합 Evidence)**")
    st.download_button(
        "📥 다운로드 (Markdown)",
        data=st.session_state.final_report_md.encode("utf-8"),
        file_name=f"final_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )
    with st.expander("📖 보고서 미리보기", expanded=True):
        st.markdown(st.session_state.final_report_md)
