# RAG Document Assistant

Assistant qui répond aux questions posées sur vos documents PDF, en citant ses sources (document et page). Il fonctionne entièrement en local avec [Ollama](https://ollama.com) : aucune donnée ne sort de votre machine.

![CI](https://github.com/mayssadhahri1/rag-document-assistant/actions/workflows/ci.yml/badge.svg)

## Fonctionnement

```
PDF ──► extraction du texte (pypdf) ──► découpage en passages avec chevauchement
                                              │
                                              ▼
                                   embeddings (nomic-embed-text)
                                              │
Question ──► embedding ──► recherche cosinus ─┘──► top-k passages ──► LLM (llama3.2) ──► réponse + sources
```

1. **Ingestion** : le texte de chaque page est découpé en passages d'environ 800 caractères, coupés en fin de phrase quand c'est possible, avec 150 caractères de chevauchement.
2. **Indexation** : chaque passage est transformé en vecteur normalisé et stocké sur disque.
3. **Recherche** : la question est vectorisée, et on garde les passages les plus proches (similarité cosinus).
4. **Génération** : le LLM reçoit uniquement ces passages et doit citer ses sources `[n]`.
5. **Garde-fou** : si aucun passage ne dépasse le score minimal, l'assistant répond qu'il n'a pas trouvé l'information, au lieu d'inventer.

## Fonctionnalités

- Interface web simple : ajout et suppression de PDF, discussion avec sources.
- API REST documentée (Swagger sur `/docs`).
- Refus des doublons, des fichiers non PDF et des PDF scannés sans texte.
- Mode hors ligne (`EMBEDDER=tfidf`) : TF-IDF à la place des embeddings, réponse extractive à la place du LLM. Utile pour la CI et les démonstrations rapides.
- 13 tests automatisés : découpage, API, garde-fou, clients Ollama simulés.

## API

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/documents` | Ajoute un PDF (multipart, champ `file`) |
| `GET` | `/documents` | Liste les documents indexés |
| `DELETE` | `/documents/{doc_id}` | Supprime un document |
| `POST` | `/ask` | `{"question": "...", "top_k": 4}` → réponse + sources |
| `GET` | `/health` | État du service |

## Lancer le projet

### Avec Docker (recommandé)

```bash
docker compose up --build
# Interface : http://localhost:8000
```

Au premier lancement, le service `ollama-init` télécharge `llama3.2` et `nomic-embed-text`.

### En local

```bash
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text
uvicorn app.main:app --reload
```

Sans Ollama, en mode hors ligne :

```bash
EMBEDDER=tfidf uvicorn app.main:app --reload
```

### Tests

```bash
pytest -q
```

## Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Adresse du serveur Ollama |
| `CHAT_MODEL` | `llama3.2` | Modèle de génération |
| `EMBED_MODEL` | `nomic-embed-text` | Modèle d'embedding |
| `EMBEDDER` | `ollama` | `ollama` ou `tfidf` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Découpage des passages |
| `TOP_K` | `4` | Nombre de passages envoyés au LLM |
| `MIN_SCORE` | `0.15` | Score minimal d'un passage |
| `DATA_DIR` | `storage` | Dossier de la base vectorielle |

## Structure

```
app/ingest.py      lecture PDF et découpage
app/embeddings.py  embeddings Ollama ou TF-IDF
app/store.py       base vectorielle (NumPy + JSON)
app/llm.py         génération avec Ollama, mode extractif
app/rag.py         orchestration recherche + génération
app/main.py        API FastAPI
static/index.html  interface web
```

## Pistes d'amélioration

- Remplacer la base NumPy par Qdrant ou pgvector pour de gros volumes.
- Ajouter un re-ranking des passages (cross-encoder).
- Évaluer la qualité des réponses sur un jeu de questions (RAGAS).
- OCR pour les PDF scannés.

## Auteure

Mayssa Dhahri, élève-ingénieure en génie logiciel à l'École Polytechnique de Sousse.
[LinkedIn](https://www.linkedin.com/in/mayssa-dhahri-38b1373bb/)
