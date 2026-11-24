"""Configuration lue depuis les variables d'environnement."""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "storage")))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "llama3.2"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    # "ollama" en production, "tfidf" pour travailler hors ligne ou dans les tests
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "ollama"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.15"))


settings = Settings()
