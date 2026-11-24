"""Génération de la réponse à partir des passages retrouvés."""
from __future__ import annotations

from typing import Protocol

import httpx

from .ingest import Chunk

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir des extraits fournis. "
    "Cite tes sources sous la forme [n]. Si la réponse n'est pas dans les extraits, "
    "dis clairement que tu ne sais pas. Réponds dans la langue de la question."
)


class LLM(Protocol):
    def generate(self, question: str, passages: list[Chunk]) -> str: ...


def build_context(passages: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({p.source}, page {p.page}) {p.text}" for i, p in enumerate(passages, start=1))


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        self.timeout = timeout

    def generate(self, question: str, passages: list[Chunk]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extraits :\n{build_context(passages)}\n\nQuestion : {question}"},
            ],
        }
        res = httpx.post(self.url, json=payload, timeout=self.timeout)
        res.raise_for_status()
        return res.json()["message"]["content"].strip()


class ExtractiveLLM:
    """Sans modèle de langage : renvoie le passage le plus pertinent (mode hors ligne)."""

    def generate(self, question: str, passages: list[Chunk]) -> str:
        best = passages[0]
        return f"{best.text} [1]"
