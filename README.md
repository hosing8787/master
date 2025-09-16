# 📊 GPU 인프라 TA를 위한 지능형 보고서 Agent

이 프로젝트는 **GPU 인프라 설계·조달(TA: Technical Assessment)** 를 지원하기 위한  
**RAG (Retrieval-Augmented Generation) + Deep Research 기반 지능형 보고서 생성 에이전트**입니다.  

Azure OpenAI, LangChain, Chroma 등을 활용하여 문서 기반 검색과 웹 검색을 결합하고,  
Streamlit UI를 통해 대화형으로 GPU 인프라 분석 리포트를 출력합니다.  

---

## 🚀 주요 기능

- 📂 로컬 문서(`.txt`, `.pdf`, `.md`)를 벡터DB에 임베딩 후 검색  
- 🔎 DuckDuckGo / SerpAPI 기반 웹 검색 지원  
- 🤖 Azure OpenAI (GPT·임베딩) 연동  
- 💱 USD → KRW(만원) 자동 변환 기능  
- 🖥️ Streamlit UI 기반 대화형 인터페이스  
- 📑 GPU RFP 사례 기반 보고서 자동 생성  

---

## 📂 프로젝트 구조

```
.
├── rag_runner_ui.py       # Streamlit 실행 메인 스크립트
├── requirements.txt       # 패키지 의존성 목록
├── .env                   # Azure OpenAI 및 환경 변수 설정
├── GPU_RFP_All_46.txt     # GPU RFP 분석용 데이터셋
└── knowledge_base/        # 분석용 문서 저장 폴더 (사용자 생성)
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
