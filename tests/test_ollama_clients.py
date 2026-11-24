"""Tests des clients Ollama avec un serveur simulé (pas besoin d'Ollama en CI)."""
import httpx
import numpy as np

from app.embeddings import OllamaEmbedder
from app.ingest import Chunk
from app.llm import OllamaLLM, build_context


def test_ollama_embedder_normalizes_vectors(monkeypatch):
    def fake_post(url, json, timeout):
        assert url.endswith("/api/embed")
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    vec = OllamaEmbedder("http://ollama:11434", "nomic-embed-text").embed(["bonjour"])
    assert np.allclose(vec, [[0.6, 0.8]])


def test_ollama_llm_sends_context_and_returns_answer(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return httpx.Response(200, json={"message": {"content": " Réponse [1] "}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    passages = [Chunk("d", "doc.pdf", 2, "Le stage dure six mois.")]
    answer = OllamaLLM("http://ollama:11434", "llama3.2").generate("Durée ?", passages)

    assert answer == "Réponse [1]"
    assert "doc.pdf, page 2" in captured["messages"][1]["content"]


def test_build_context_numbers_passages():
    passages = [Chunk("d", "a.pdf", 1, "A"), Chunk("d", "b.pdf", 4, "B")]
    assert build_context(passages).startswith("[1] (a.pdf, page 1) A")
