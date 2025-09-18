# RAG 관련 로직 (임베딩/벡터스토어/리트리버)
from typing import List, Tuple, Dict, Any
import os, hashlib, re
from pathlib import Path
from dataclasses import asdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from .utils import get_env_any, normalize_endpoint, krw_round_manwon
from ..core.config import settings

ALLOWED_EXTS = {".txt", ".pdf", ".md"}

def list_folder_files(folder_path: str):
    base = Path(folder_path).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return []
    files = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return sorted(files)

def get_folder_hash(files):
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

def build_embeddings() -> AzureOpenAIEmbeddings:
    endpoint = normalize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    api_key = get_env_any("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
    deployment = get_env_any(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT","AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING","AZURE_EMBEDDING_DEPLOYMENT","AZURE_EMBED_DEPLOYMENT",
        "AZURE_EMBEDDING_MODEL","AZURE_EMBEDDINGS_DEPLOYMENT","AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not (endpoint and api_key and deployment):
        raise RuntimeError("Azure 임베딩 환경변수 부족")
    return AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

def build_vectorstore(folder_path: str):
    files = list_folder_files(folder_path)
    if not files:
        return None, []
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=100)
    all_docs: List[Document] = []
    docs_info = []
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
            # log only
            pass
    if not all_docs:
        return None, []
    embeddings = build_embeddings()
    collection_name = "kb_" + get_folder_hash(files)[:12]
    path = settings.VECTORSTORE_PATH or None
    if path:
        vs = Chroma.from_documents(documents=all_docs, embedding=embeddings, collection_name=collection_name, persist_directory=path)
    else:
        vs = Chroma.from_documents(documents=all_docs, embedding=embeddings, collection_name=collection_name)
    return vs, docs_info

def run_rag(vectorstore, query: str) -> List[Document]:
    if vectorstore is None:
        return []
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 60})
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)
    return docs
