"""Task 2 – Retrieval layer: embed question, get top-k chunks.

Thin wrapper so rag_pipeline.py and evaluate_retrieval.py share one code path.

Run smoke test:  python retrieval.py c300 "What are the four functions?"
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from vector_store import Hit, VectorStore

_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Lazy singleton – loading MiniLM twice wastes RAM on an 8 GB laptop."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _EMBEDDER


def retrieve(query: str, config_id: str, top_k: int = 5) -> list[Hit]:
    emb = get_embedder().encode(query).astype(np.float32)
    return VectorStore(config_id).search(emb, top_k=top_k)


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "c300"
    q = sys.argv[2] if len(sys.argv) > 2 else "What are the four functions of the AI RMF Core?"
    for h in retrieve(q, cfg):
        print(f"{h.chunk_id} p{h.page} sim={h.score:.3f}\n   {h.text[:120]}...\n")
