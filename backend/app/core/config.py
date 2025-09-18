import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AZURE_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_EMBED_DEP: str | None = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    AZURE_CHAT_DEP: str | None = os.getenv("AZURE_LLM_DEPLOYMENT")

    VECTORSTORE_PATH: str | None = os.getenv("VECTORSTORE_PATH") or "./.chroma_cache"
    KNOWLEDGE_BASE_PATH: str | None = os.getenv("KNOWLEDGE_BASE_PATH") or "./documents"

    SERPAPI_API_KEY: str | None = os.getenv("SERPAPI_API_KEY")
    DR_TIMEFRAME_DAYS: int = int(os.getenv("DR_TIMEFRAME_DAYS", "365"))
    DR_MIN_SOURCES: int = int(os.getenv("DR_MIN_SOURCES", "12"))

    USD_KRW_RATE: float = float(os.getenv("USD_KRW_RATE", "1400"))

settings = Settings()
