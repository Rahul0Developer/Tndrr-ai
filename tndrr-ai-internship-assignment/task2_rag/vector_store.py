"""Task 2 – NumPy-only vector store with cosine similarity search.

Vectors are L2-normalised at index time (embeddings.py), so cosine
similarity reduces to a dot product; we still divide by norms defensively.

Run standalone smoke test:  python vector_store.py c300 "what is the framework"
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "indexes"


@dataclass
class Hit:
    chunk_id: str
    page: int
    score: float
    text: str


class VectorStore:
    """Brute-force dense retrieval over an .npy matrix + JSONL metadata."""

    def __init__(self, config_id: str) -> None:
        vec_path = INDEX_DIR / f"{config_id}_vectors.npy"
        meta_path = INDEX_DIR / f"{config_id}_meta.jsonl"
        if not vec_path.exists():
            raise SystemExit(f"Index missing ({vec_path}); run embeddings.py first.")
        self.vectors = np.load(vec_path)                       # (N, 384) float32
        self.meta = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines()]
        assert len(self.meta) == self.vectors.shape[0]

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[Hit]:
        q = query_vec.astype(np.float32).ravel()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        denom = np.linalg.norm(self.vectors, axis=1) * q_norm
        sims = self.vectors @ q / np.maximum(denom, 1e-12)     # cosine similarities
        k = min(top_k, sims.shape[0])
        # argpartition gets top-k unordered; sort just those k afterwards.
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Hit(self.meta[i]["chunk_id"], self.meta[i]["primary_page"],
                    float(sims[i]), self.meta[i]["text"]) for i in idx]


if __name__ == "__main__":
    import sys
    from sentence_transformers import SentenceTransformer
    cfg = sys.argv[1] if len(sys.argv) > 1 else "c300"
    query = sys.argv[2] if len(sys.argv) > 2 else "what is the AI RMF"
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vs = VectorStore(cfg)
    print(f"store={cfg} chunks={vs.vectors.shape}")
    for h in vs.search(st.encode(query), top_k=3):
        print(f"  {h.chunk_id} p{h.page} sim={h.score:.3f}: {h.text[:90]}...")
