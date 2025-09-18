import os, requests, streamlit as st
from dotenv import load_dotenv
load_dotenv()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="TTD-DR GPU Report (Refactored)", layout="wide")
st.title("🧬 GPU 인프라 TA 지능형 보고서 (Refactored UI)")

st.sidebar.success("백엔드: " + BACKEND)
st.sidebar.info("FastAPI + Streamlit (형식 우선 구조 데모)")

st.markdown("### 🎯 시스템 구성 요구사항 입력")
with st.form(key="query_form"):
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("성향/용도", [
            "대규모 AI 트레이닝 특화 (Hyperscale AI Training)",
            "중형 연구 HPC (Research HPC Cluster)",
            "중형 기관·산업 HPC (Enterprise/Institutional HPC)",
            "AI/LLM 특화 서버 (Fine-tuning & Inference)",
            "소규모 연구실 서버 (Lab-scale GPU Server)",
            "단품 GPU/소형 서버 (Single-GPU/Small-scale)",
        ], index=0)
        gpu = st.text_input("GPU 스펙", value="HGX H100 8×80GB")
    with col2:
        users = st.text_input("예상 사용자 수", value="80–100명")
        cost = st.text_input("총 예산", value="$1.5–2M")
    special = st.text_area("특별 요구사항(선택)", height=80, placeholder="예: 고가용성, DR, 특정 SW 호환 등")
    submitted = st.form_submit_button("🚀 보고서 생성", type="primary")

if submitted:
    with st.spinner("백엔드 보고서 생성 호출 중..."):
        try:
            r = requests.post(f"{BACKEND}/api/report/generate", json={
                "category": category, "gpu": gpu, "users": users, "cost": cost, "special": special or ""
            }, timeout=120)
            r.raise_for_status()
            data = r.json()
            st.success(f"보고서 생성 완료 — Evidence: {data.get('evidence_count')}개")
            st.download_button("📥 다운로드 (Markdown)",
                               data.get("markdown","").encode("utf-8"),
                               file_name="final_report.md",
                               mime="text/markdown")
            st.markdown("---")
            st.markdown("## 📋 **미리보기**")
            st.markdown(data.get("markdown",""))
        except Exception as e:
            st.error(f"보고서 생성 실패: {e}")
