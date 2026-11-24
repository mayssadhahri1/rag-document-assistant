"""Lecture des PDF et découpage en morceaux (chunks)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    text: str


def read_pdf(data: bytes) -> list[tuple[int, str]]:
    """Retourne la liste (numéro de page, texte) d'un PDF."""
    reader = PdfReader(BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Découpe un texte en morceaux d'environ `size` caractères.

    On coupe de préférence à la fin d'une phrase, et on garde un
    chevauchement pour ne pas perdre le contexte entre deux morceaux.
    """
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_pdf(doc_id: str, source: str, data: bytes, size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for page, text in read_pdf(data):
        for part in split_text(text, size, overlap):
            chunks.append(Chunk(doc_id=doc_id, source=source, page=page, text=part))
    return chunks
