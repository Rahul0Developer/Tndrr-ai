"""Task 2 – Embed chunks with all-MiniLM-L6-v2 and store vectors + metadata.

No vector database: embeddings are saved as a float32 NumPy .npy matrix plus a
parallel JSONL metadata file (chunk_id, page, text). Cosine similarity is
computed manually in vector_store.py.

Memory discipline for an 8 GB laptop: encode in batches of 16.

Run:  python embeddings.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "indexes"
BATCH_SIZE = 16
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONFIGS = {"c300": "chunks_300.jsonl", "c600": "chunks_600.jsonl"}


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    print(f"Loaded {MODEL_NAME} (dim={dim}); batch_size={BATCH_SIZE}")

    for cid, fname in CONFIGS.items():
        src = BASE / "data" / fname
        if not src.exists():
            raise SystemExit(f"Missing {src}; run chunking.py first.")
        chunks = load_chunks(src)
        texts = [c["text"] for c in chunks]
        vecs = model.encode(
            texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)   # L2-normalised => cosine == dot product
        np.save(INDEX_DIR / f"{cid}_vectors.npy", vecs)
        meta = [{"chunk_id": c["chunk_id"], "primary_page": c["primary_page"], "pages": c["pages"],
                 "n_words": c["n_words"], "text": c["text"]} for c in chunks]
        with (INDEX_DIR / f"{cid}_meta.jsonl").open("w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        size_mb = vecs.nbytes / 1e6
        print(f"{cid}: stored {vecs.shape} ({size_mb:.2f} MB) + {len(meta)} metadata rows")


if __name__ == "__main__":
    main()
