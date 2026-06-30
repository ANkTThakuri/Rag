"""
vectorstore.py
--------------
Local vector database (Chroma, persisted to disk) using a free local
sentence-transformers embedding model — no API key needed for embeddings.
"""

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

DB_DIR = Path(__file__).parent / "vectorstore" / "chroma_db"
DB_DIR.mkdir(exist_ok=True, parents=True)

_client = chromadb.PersistentClient(path=str(DB_DIR))

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

COLLECTION_NAME = "pdf_chunks"


def get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunk_records: list[dict]):
    """Add a list of chunk dicts (from chunker.chunk_and_save) to the vector DB."""
    if not chunk_records:
        return
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunk_records],
        documents=[c["text"] for c in chunk_records],
        metadatas=[
            {"source": c["source"], "chunk_index": c["chunk_index"]}
            for c in chunk_records
        ],
    )


def query(question: str, n_results: int = 5):
    """Return top-n most relevant chunks for a question."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[question], n_results=min(n_results, collection.count()))
    matches = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append({"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"], "score": 1 - dist})
    return matches


def list_sources() -> list[str]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    data = collection.get()
    return sorted(set(m["source"] for m in data["metadatas"]))


def reset_collection():
    _client.delete_collection(COLLECTION_NAME)
