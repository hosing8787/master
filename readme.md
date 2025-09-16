TTD-DR: Test-Time Diffusion Deep Research for HPC/GPU Systems
GPU 인프라 설계를 위한 지능형 분석 도구로, RAG와 웹 딥리서치를 결합한 TTD-DR 방법론을 통해 전문가 수준의 시스템 설계 보고서를 자동 생성합니다.

개요
TTD-DR(Test-Time Diffusion Deep Research)은 확산 모델의 점진적 노이즈 제거 원리를 연구 과정에 적용한 혁신적 방법론입니다. GPU 시스템 설계 요구사항을 입력받아 Knowledge Base 검색과 실시간 웹 연구를 통해 종합적인 기술 보고서를 생성합니다.

주요 특징
적응적 정보 수집: RAG 충분성 판정을 통한 조건부 웹 딥리서치
전문가 수준 보고서: 6개 표준 섹션 자동 생성 (스펙/성능/비용/대안/로드맵/결론)
Evidence 기반 인용: 모든 기술적 주장에 대한 출처 추적
다중 소스 통합: Knowledge Base + 웹 크롤링 결과 융합
실시간 환율 변환: USD 금액의 KRW(만원) 자동 병기
시스템 요구사항
필수 환경
Python 3.8+
Azure OpenAI API 키
최소 8GB RAM (대용량 Knowledge Base 처리 시)
지원 플랫폼
Windows 10/11
macOS 10.15+
Linux (Ubuntu 18.04+, CentOS 7+)
설치 방법
1. 저장소 클론
bash
git clone https://github.com/your-org/ttd-dr-hpc-gpu.git
cd ttd-dr-hpc-gpu
2. 가상환경 생성 및 활성화
bash
python -m venv ttd_dr_env
source ttd_dr_env/bin/activate  # Linux/macOS
# 또는
ttd_dr_env\Scripts\activate     # Windows
3. 의존성 설치
bash
pip install -r requirements.txt
4. 환경변수 설정
.env 파일을 프로젝트 루트에 생성하고 다음 내용을 입력:

env
# Azure OpenAI 설정 (필수)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-06-01

# 채팅 모델 배포명 (필수)
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4
AZURE_LLM_DEPLOYMENT=gpt-4

# 임베딩 모델 배포명 (필수)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002

# Knowledge Base 경로 (선택)
KNOWLEDGE_BASE_PATH=./documents
PDF_DOCUMENTS_DIR=./documents

# 벡터스토어 경로 (선택)
VECTORSTORE_PATH=./vectorstore

# Deep Research 설정 (선택)
DR_TIMEFRAME_DAYS=365
DR_MIN_ROUNDS=3
DR_MIN_SOURCES=12
DR_MAX_PER_DOMAIN=2

# 환율 설정 (선택)
KRW_RATE=1400

# 웹 검색 API (선택)
SERPAPI_API_KEY=your-serpapi-key

# LangSmith 추적 (선택)
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
사용법
1. Knowledge Base 준비
bash
mkdir documents
# PDF, Markdown, 텍스트 파일을 documents 폴더에 복사
지원 파일 형식:

PDF (.pdf)
Markdown (.md)
텍스트 파일 (.txt)
2. 애플리케이션 실행
bash
streamlit run rag_runner_ui.py
브라우저에서 http://localhost:8501에 접속합니다.

3. 요구사항 입력
구조화된 입력
시스템 카테고리: 대규모 AI 트레이닝, 중형 연구 HPC, 기관 HPC 등
GPU 사양: "HGX H100 8×80GB", "A100 4×80GB" 등
예상 사용자 수: "80-100명", "50명" 등
예산: "$1.5-2M", "200만 달러" 등
자연어 입력 (선택)
LLM 파인튜닝용 H100 클러스터가 필요합니다. 
동시 사용자는 약 100명이고, 예산은 150만 달러입니다.
고가용성과 확장성을 고려해주세요.
4. TTD-DR 실행
"TTD-DR 실행" 버튼을 클릭하면 다음 과정이 자동 진행됩니다:

요구사항 분석: 입력 내용을 구조화된 키워드로 변환
RAG 검색: Knowledge Base에서 관련 문서 검색
충분성 판정: 검색 결과의 충분성 평가
조건부 웹 연구: 필요 시 실시간 웹 정보 수집
Evidence 통합: 모든 소스 정보 결합 및 중복 제거
보고서 생성: 6개 섹션 전문 보고서 자동 작성
5. 결과 확인 및 다운로드
생성된 보고서는 Markdown 형식으로 즉시 다운로드 가능합니다.

프로젝트 구조
ttd-dr-hpc-gpu/
├── rag_runner_ui.py          # 메인 Streamlit 애플리케이션
├── requirements.txt          # Python 의존성
├── .env.example             # 환경변수 템플릿
├── README.md               # 이 파일
├── documents/              # Knowledge Base 문서 폴더
├── vectorstore/           # Chroma 벡터스토어 (자동 생성)
└── reports/              # 생성된 보고서 저장 (선택)
고급 설정
Knowledge Base 최적화
python
# 청킹 파라미터 조정
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 검색 파라미터
SEARCH_K = 15          # 반환할 문서 수
FETCH_K = 60          # MMR 계산용 문서 수
Deep Research 정책
python
# 웹 검색 설정
TIMEFRAME_DAYS = 365   # 최근 N일 이내 정보만 수집
MIN_SOURCES = 12       # 최소 웹 소스 수
MAX_PER_DOMAIN = 2     # 도메인당 최대 문서 수
도메인 블랙리스트
신뢰할 수 없는 도메인은 자동 차단됩니다:

python
BLOCKED_DOMAINS = {
    "hyperscale.com", "funbe", "tkor", 
    "namu.wiki", "redrosa.co.kr"
}
API 참조
주요 함수
run_rag(query: str) -> List[Document]
Knowledge Base에서 RAG 검색을 수행합니다.

assess_sufficiency(docs: List[Document], req: Dict) -> Tuple[bool, Dict]
검색 결과의 충분성을 평가합니다.

gather_dr_evidence(topic: str, policy: Dict) -> List[EvidenceItem]
웹 딥리서치를 통해 Evidence를 수집합니다.

dedup_evidence(evidence: List[EvidenceItem]) -> List[EvidenceItem]
Evidence 목록에서 중복을 제거합니다.

문제 해결
일반적인 오류
Azure OpenAI 연결 실패
❌ 임베딩 호출 실패: Azure 설정을 확인하세요.
해결방법: .env 파일의 Azure 설정을 확인하고 API 키와 엔드포인트가 정확한지 검증

Knowledge Base 빌드 실패
❌ 인덱싱할 문서가 없습니다.
해결방법: documents 폴더에 지원 형식(.pdf, .md, .txt) 파일이 있는지 확인

메모리 부족
MemoryError: Unable to allocate array
해결방법:

청킹 크기 줄이기 (CHUNK_SIZE = 400)
배치 크기 줄이기
RAM 증설 또는 더 작은 Knowledge Base 사용
성능 최적화
벡터스토어 캐싱 활용
python
# 벡터스토어 영구 저장 설정
VECTORSTORE_PATH = "./vectorstore"
세션 상태 최적화
python
# Streamlit 세션 캐싱 활성화
st.cache_data
st.cache_resource
기여 방법
개발 환경 설정
bash
# 개발용 의존성 설치
pip install -r requirements-dev.txt

# 코드 스타일 검사
flake8 rag_runner_ui.py
black rag_runner_ui.py

# 테스트 실행
pytest tests/
버그 리포트
GitHub Issues에 다음 정보와 함께 제출:

오류 메시지 전문
환경 정보 (OS, Python 버전)
재현 가능한 최소 예제
기능 요청
새로운 기능 제안 시 다음을 포함:

사용 사례 설명
예상 사용자 그룹
구현 복잡도 추정
라이선스
MIT License. 자세한 내용은 LICENSE 파일을 참조하세요.

지원 및 문의
이슈 트래킹: GitHub Issues
문서: 프로젝트 Wiki
이메일: ttd-dr-support@company.com
변경 이력
v1.0.0 (2024-xx-xx)
TTD-DR 방법론 초기 구현
Streamlit 기반 UI
Azure OpenAI 통합
RAG + 웹 딥리서치 결합
v0.9.0 (2024-xx-xx)
베타 릴리스
핵심 기능 구현 완료
내부 테스트 완료
TTD-DR로 GPU 인프라 설계의 새로운 패러다임을 경험해보세요.

