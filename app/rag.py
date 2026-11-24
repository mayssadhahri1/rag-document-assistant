"""Orchestration du RAG : recherche des passages puis génération de la réponse."""
from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM
from .store import VectorStore

NO_ANSWER = "Je n'ai pas trouvé cette information dans les documents."


@dataclass
class Source:
    index: int
    source: str
    page: int
    score: float
    excerpt: str


@dataclass
class Answer:
    answer: str
    sources: list[Source]


class RagService:
    def __init__(self, store: VectorStore, llm: LLM, top_k: int, min_score: float):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = [(c, s) for c, s in self.store.search(question, top_k or self.top_k) if s >= self.min_score]
        # Garde-fou : sans passage pertinent, on ne laisse pas le modèle inventer
        if not hits:
            return Answer(answer=NO_ANSWER, sources=[])

        passages = [c for c, _ in hits]
        text = self.llm.generate(question, passages)
        sources = [
            Source(index=i, source=c.source, page=c.page, score=round(s, 3), excerpt=c.text[:240])
            for i, (c, s) in enumerate(hits, start=1)
        ]
        return Answer(answer=text, sources=sources)
