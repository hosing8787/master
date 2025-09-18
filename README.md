# 📊 GPU 인프라 TA를 위한 지능형 보고서 Agent

이 프로젝트는 **GPU 인프라 설계·조달(TA: Technical Assessment)** 를 지원하기 위한  
**RAG (Retrieval-Augmented Generation) + Deep Research 기반 지능형 보고서 생성 에이전트**입니다.  

Azure OpenAI, LangChain, Chroma 등을 활용하여 문서 기반 검색과 웹 검색을 결합하고,  
Streamlit UI를 통해 대화형으로 GPU 인프라 분석 리포트를 출력합니다.  

---

## 🔬 TTD-DR 방법론 상세 설명

본 프로젝트는 단순 RAG(Retrieval-Augmented Generation) 기반 검색에서 한 단계 확장된  
**TTD-DR(Test-Time Diffusion – Deep Research)** 방법론을 적용합니다.  

- **다중 출처 검증**: 최소 출처 수를 확보하고, 도메인당 편향을 줄여 신뢰성 강화  
- **딥 서치 루프**: 2차·3차 검색 라운드를 반복 수행하여 정보 누락 최소화  
- **비용·성능 병기**: GPU/HPC RFP 사례의 달러 단위 비용을 원화(만원)로 자동 환산  
- **근거 기반 리포트**: 단순 요약이 아닌, 교차 검증된 근거와 권고안을 보고서 형식으로 출력  

이 방법론을 통해 단순 질의응답 수준이 아닌,  
**의사결정 지원용 GPU 인프라 설계 보고서**를 자동 생성할 수 있습니다.  

---

## 🚀 주요 기능

- 📂 로컬 문서(`.txt`, `.pdf`, `.md`)를 벡터DB에 임베딩 후 검색  
- 🔎 DuckDuckGo / SerpAPI 기반 웹 검색 지원  
- 🤖 Azure OpenAI (GPT·임베딩) 연동  
- 💱 USD → KRW(만원) 자동 변환 기능  
- 🖥️ Streamlit UI 기반 대화형 인터페이스  
- 📑 GPU RFP 사례 기반 보고서 자동 생성  

---
🏗️ 시스템 아키텍처
mermaidflowchart LR
    A[**요구사항 입력**] --> B[**요구사항 분석**] --> C[**RAG 검색**] --> D{**RAG 결과<br/>충분성 만족**}
    
    D -->|No| E[**Report 생성<br/>Web Search 검색**]
    D -->|Yes| F[**Report 생성<br/>RAG 기반**]
    
    E --> G[**TTD-DR<br>확산형 추론 딥리서치**]
    F --> G
    
    G --> H[**최종 보고서 생성**] --> I{**요구사항과 보고서<br/>적합성 확인**} --> J[**최종 보고서 출력**]
    
    %% 재귀 연결
    I -.->|부적합| B

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#ffe4b5,stroke:#ff9800,stroke-width:2px,color:#000,font-weight:bold
    classDef report fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef reasoning fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:#000,font-weight:bold
    classDef final fill:#e8f5e8,stroke:#4caf50,stroke-width:3px,color:#000,font-weight:bold
    classDef validation fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000,font-weight:bold
    classDef output fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:#000,font-weight:bold
    
    class D,I decision
    class E,F report
    class G reasoning
    class H final
    class I validation
    class J output
모듈 구조

Backend (FastAPI): RESTful API 서버, 비즈니스 로직 처리
Frontend (Streamlit): 웹 UI, 사용자 인터페이스
Services: 핵심 기능별 독립 서비스 모듈
Models: 데이터 스키마 및 비즈니스 엔티티
Utils: 공통 유틸리티 (통화 변환, 텍스트 처리 등)

## 📂 프로젝트 구조

```
구성 개요
gpu_ttd_dr_refactored/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                 # FastAPI 엔트리포인트
│  │  ├─ routers/
│  │  │  └─ report.py            # 보고서 생성 라우터
│  │  ├─ services/
│  │  │  ├─ rag.py               # RAG 관련 로직
│  │  │  ├─ deep_research.py     # 웹 딥리서치 로직
│  │  │  └─ reporting.py         # Evidence → 보고서 생성
│  │  ├─ core/config.py          # 설정/환경변수 관리
│  │  ├─ models/evidence.py      # 데이터 클래스/스키마
│  │  └─ __init__.py
│  └─ tests/test_smoke.py
├─ frontend/
│  ├─ streamlit_app.py           # Streamlit UI (FastAPI 연동)
│  └─ requirements.txt
├─ legacy/
│  └─ rag_runner_ui.py           # 사용자가 업로드한 원본(참조용 보존)
├─ .env.example
├─ pyproject.toml
└─ README.md
```

---

## ⚙️ 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

2. 가상환경 생성 및 활성화 (선택)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```
(주요 패키지: streamlit, langchain, langchain-openai, chromadb, ddgs, tiktoken 등)

---

## 🔑 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래와 같이 설정합니다:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_LLM_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-06-01

# 경로 설정
KNOWLEDGE_BASE_PATH=./knowledge_base
VECTORSTORE_PATH=./vectorstore
PDF_DOCUMENTS_DIR=./pdf_documents
```

---

## ▶️ 실행 방법

```bash
streamlit run rag_runner_ui.py
```

브라우저에서 `http://localhost:8501` 접속 → GPU 인프라 보고서 Agent 실행 가능.  

---

## 📘 사용 예시

1. `knowledge_base/` 폴더에 분석할 문서를 넣습니다.  
2. Streamlit UI에서 문서 기반 질문을 입력합니다.  
3. GPU 인프라 설계 보고서가 자동으로 생성됩니다.  

---

## 📄 데이터셋

- `GPU_RFP_All_46.txt`: 글로벌 GPU/HPC 도입 사례를 정리한 데이터셋으로, RAG 검색과 Deep Research에 활용됩니다.  
