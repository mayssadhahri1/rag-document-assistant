def upload(client, pdf_bytes, name="reglement.pdf"):
    return client.post("/documents", files={"file": (name, pdf_bytes, "application/pdf")})


def test_upload_and_list(client, pdf_bytes):
    res = upload(client, pdf_bytes)
    assert res.status_code == 201
    docs = client.get("/documents").json()
    assert docs[0]["source"] == "reglement.pdf"
    assert docs[0]["pages"] == 3


def test_duplicate_upload_is_rejected(client, pdf_bytes):
    upload(client, pdf_bytes)
    assert upload(client, pdf_bytes).status_code == 409


def test_non_pdf_is_rejected(client):
    res = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_answer_cites_the_right_page(client, pdf_bytes):
    upload(client, pdf_bytes)
    data = client.post("/ask", json={"question": "Combien de membres dans le jury de soutenance ?"}).json()
    assert "trois membres" in data["answer"]
    assert data["sources"][0]["page"] == 3


def test_unknown_question_returns_no_answer(client, pdf_bytes):
    upload(client, pdf_bytes)
    data = client.post("/ask", json={"question": "Quel est le prix du billet d'avion ?"}).json()
    assert data["sources"] == []
    assert "pas trouvé" in data["answer"]


def test_delete_document(client, pdf_bytes):
    doc_id = upload(client, pdf_bytes).json()["doc_id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc_id}").status_code == 404
