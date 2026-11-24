"""Calcul des vecteurs (embeddings) des textes.

Deux implémentations avec la même interface :
- OllamaEmbedder : modèle d'embedding local servi par Ollama ;
- TfidfEmbedder  : alternative sans modèle, utile hors ligne et pour les tests.
"""
from __future__ import annotations

from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        self.timeout = timeout

    def fit(self, texts: list[str]) -> None:
        # Le modèle est déjà entraîné : rien à faire
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        res = httpx.post(self.url, json={"model": self.model, "input": texts}, timeout=self.timeout)
        res.raise_for_status()
        return normalize(np.array(res.json()["embeddings"], dtype=np.float32))


# Mots trop fréquents qui faussent la similarité (français + anglais)
STOP_WORDS = """a au aux avec ce ces cette d dans de des du elle en est et il ils je l la le les leur lui ma mais me
mes mon ne nos notre nous on ou par pas pour qu que quel quelle quels quelles qui sa se ses son sur ta te tes ton tu
un une vos votre vous y combien comment est-ce the of and to in is are what how which who for on with""".split()


class TfidfEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
            stop_words=STOP_WORDS, token_pattern=r"(?u)\b\w\w+\b",
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder doit être entraîné avant utilisation")
        return normalize(self.vectorizer.transform(texts).toarray().astype(np.float32))
