from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers.report import router as report_router

app = FastAPI(title="TTD-DR GPU Report Backend", version="0.1.0")

# CORS (Streamlit 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(report_router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok", "service": "ttd-dr-backend"}
