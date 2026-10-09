import json
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_FIELDS = {
    "chunk_id",
    "tool",
    "title",
    "symptom",
    "guidance",
    "source_url",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {"error", "failed", "failure", "warning", "exception", "fatal", "panic", "exit"})


def load_knowledge_base(path: Path) -> list[dict[str, str]]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Knowledge base must be a non-empty JSON array.")

    ids = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or not REQUIRED_FIELDS <= chunk.keys():
            raise ValueError("Each knowledge chunk must contain all required fields.")
        if not all(isinstance(chunk[field], str) and chunk[field].strip() for field in REQUIRED_FIELDS):
            raise ValueError("Knowledge chunk fields must be non-empty strings.")
        if chunk["chunk_id"] in ids:
            raise ValueError(f"Duplicate knowledge chunk ID: {chunk['chunk_id']}")
        if not chunk["source_url"].startswith("https://"):
            raise ValueError(f"Knowledge chunk {chunk['chunk_id']} must have an HTTPS source URL.")
        ids.add(chunk["chunk_id"])
    return chunks


def retrieve(query: str, chunks: list[dict[str, str]], top_k: int = 3) -> list[dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not query.strip() or not chunks:
        return []

    documents = [
        " ".join(chunk[field] for field in ("tool", "title", "symptom", "guidance"))
        for chunk in chunks
    ]
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([query, *documents])
    scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()
    ranked = sorted(enumerate(scores), key=lambda result: (-result[1], result[0]))
    return [
        {"chunk": chunks[index], "score": float(score)}
        for index, score in ranked[:top_k]
        if score > 0
    ]
