# GPU 인프라 TA를 위한 지능형 보고서 Agent — 모듈형 리팩토링

이 리포지토리는 **심사위원들에게 구조적으로 전문적인 인상을 주기 위해** 모듈화/레이어 분리를 적용한 샘플 구조입니다.

## 구성 개요

```
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

- **형식 우선**: 실제 동작보다 **전문적인 구조/모듈화**에 초점
- **백엔드/프론트엔드 분리**: FastAPI ←→ Streamlit
- **TTD‑DR 모듈화**: `services/` 하위로 RAG / Deep Research / Reporting 분리
- **테스트 스텁**: 최소 smoke 테스트 포함

> 원본 단일 파일은 `legacy/rag_runner_ui.py`로 보존되어 비교·참조 가능합니다.

---

## 빠른 시작

### 1) 가상환경 & 설치
```bash
cd gpu_ttd_dr_refactored
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r frontend/requirements.txt
pip install -e .
```

### 2) 환경변수 설정
`.env.example`를 `.env`로 복사 후 본인 값으로 채웁니다.

### 3) 백엔드 실행 (FastAPI)
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) 프론트엔드 실행 (Streamlit)
```bash
streamlit run frontend/streamlit_app.py
```
> 기본적으로 `http://localhost:8000`의 백엔드와 통신합니다.

---

## 설계 원칙

- **모듈화(최우선)**: 기능별 파일 분리, 의존성 역전(서비스 레이어) 지향
- **관심사 분리**: REST 라우팅 / 비즈니스 로직 / 프롬프트·리포팅 / 설정
- **테스트 용이성**: 서비스 레이어는 의존성 주입과 순수 함수 위주
- **문서화**: README, 주석, 타입힌트로 가독성 강화

## 라이선스
MIT
