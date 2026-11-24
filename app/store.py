"""Base vectorielle simple, sauvegardée sur disque (JSON + NumPy)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .embeddings import Embedder, TfidfEmbedder
from .ingest import Chunk


class VectorStore:
    def __init__(self, directory: Path, embedder: Embedder):
        self.dir = directory
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # --- persistance -------------------------------------------------------
    @property
    def _chunks_file(self) -> Path:
        return self.dir / "chunks.json"

    def _load(self) -> None:
        if self._chunks_file.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self._chunks_file.read_text(encoding="utf-8"))]
            self._reindex()

    def _save(self) -> None:
        data = [asdict(c) for c in self.chunks]
        self._chunks_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _reindex(self) -> None:
        """Recalcule tous les vecteurs (nécessaire pour TF-IDF quand le corpus change)."""
        texts = [c.text for c in self.chunks]
        if not texts:
            self.vectors = np.zeros((0, 0), dtype=np.float32)
            return
        self.embedder.fit(texts)
        self.vectors = self.embedder.embed(texts)

    # --- opérations --------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> None:
        if isinstance(self.embedder, TfidfEmbedder) or self.vectors.size == 0:
            self.chunks.extend(chunks)
            self._reindex()
        else:
            new = self.embedder.embed([c.text for c in chunks])
            self.vectors = np.vstack([self.vectors, new])
            self.chunks.extend(chunks)
        self._save()

    def delete(self, doc_id: str) -> int:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        removed = before - len(self.chunks)
        if removed:
            self._reindex()
            self._save()
        return removed

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "source": c.source, "chunks": 0, "pages": set()})
            d["chunks"] += 1
            d["pages"].add(c.page)
        return [{**d, "pages": len(d["pages"])} for d in docs.values()]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Recherche par similarité cosinus (les vecteurs sont normalisés)."""
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q
        best = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best]
