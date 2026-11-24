"""API FastAPI de l'assistant documentaire."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .embeddings import OllamaEmbedder, TfidfEmbedder
from .ingest import chunk_pdf
from .llm import ExtractiveLLM, OllamaLLM
from .rag import RagService
from .store import VectorStore

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024  # 20 Mo

app = FastAPI(title="RAG Document Assistant", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache
def get_service() -> RagService:
    if settings.embedder == "tfidf":
        embedder, llm = TfidfEmbedder(), ExtractiveLLM()
    else:
        embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
        llm = OllamaLLM(settings.ollama_url, settings.chat_model)
    store = VectorStore(settings.data_dir, embedder)
    return RagService(store, llm, settings.top_k, settings.min_score)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", status_code=201)
async def upload(file: UploadFile = File(...), service: RagService = Depends(get_service)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Fichier trop volumineux (20 Mo maximum)")

    doc_id = hashlib.sha1(data).hexdigest()[:12]
    if any(d["doc_id"] == doc_id for d in service.store.documents()):
        raise HTTPException(409, "Ce document est déjà indexé")

    chunks = chunk_pdf(doc_id, file.filename, data, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "Aucun texte lisible dans ce PDF (document scanné ?)")
    service.store.add(chunks)
    return {"doc_id": doc_id, "source": file.filename, "chunks": len(chunks)}


@app.get("/documents")
def list_documents(service: RagService = Depends(get_service)) -> list[dict]:
    return service.store.documents()


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, service: RagService = Depends(get_service)) -> None:
    if service.store.delete(doc_id) == 0:
        raise HTTPException(404, "Document introuvable")


@app.post("/ask")
def ask(q: Question, service: RagService = Depends(get_service)) -> dict:
    return asdict(service.ask(q.question, q.top_k))
