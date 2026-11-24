"""Fixtures communes : PDF de test et client API en mode hors ligne (TF-IDF)."""
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PAGES = [
    "Règlement des stages. La durée du stage de fin d'études est de quatre à six mois. "
    "Le stage commence en février et se termine au plus tard en juillet.",
    "Encadrement. Chaque stagiaire est suivi par un encadrant académique et un encadrant en entreprise. "
    "Une réunion de suivi est organisée toutes les deux semaines.",
    "Évaluation. La soutenance a lieu devant un jury de trois membres. "
    "La note finale tient compte du rapport, de la présentation et de l'avis de l'entreprise.",
]


def make_pdf(pages: list[str]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in [text[i:i + 90] for i in range(0, len(text), 90)]:
            c.drawString(40, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(PAGES)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER", "tfidf")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib

    import app.config
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.main)
    app.main.get_service.cache_clear()
    return TestClient(app.main.app)
