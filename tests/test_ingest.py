from app.ingest import chunk_pdf, read_pdf, split_text


def test_read_pdf_returns_each_page(pdf_bytes):
    pages = read_pdf(pdf_bytes)
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "soutenance" in pages[2][1]


def test_split_text_respects_size_and_overlap():
    text = "Phrase numéro un. " * 100
    parts = split_text(text, size=200, overlap=40)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)


def test_short_text_is_a_single_chunk():
    assert split_text("Texte court.", size=200, overlap=40) == ["Texte court."]


def test_chunk_pdf_keeps_page_numbers(pdf_bytes):
    chunks = chunk_pdf("doc1", "reglement.pdf", pdf_bytes, size=500, overlap=50)
    assert {c.page for c in chunks} == {1, 2, 3}
    assert all(c.source == "reglement.pdf" for c in chunks)
