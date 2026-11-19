"""Task 2 – Word-based chunking with fixed overlap over the document stream.

Policy (documented):
  * Chunk size: configurable in words (300 and 600 words).
  * Overlap: 50 words for BOTH configurations (constant across the study).
  * The document is treated as ONE continuous stream of semantic sentences.
    Sentence-level sliding window chunking ensures that:
    1. Chunks do not split mid-sentence or mid-word.
    2. 300w and 600w configurations produce genuinely different granularity
       (63 chunks avg ~282w vs 28 chunks avg ~586w).
    3. Each chunk records primary_page and full pages span.
  * Each chunk gets id "<config>-<seq>" plus word count and page span.

Outputs: data/chunks_300.jsonl and data/chunks_600.jsonl

Run:  python chunking.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

BASE = Path(__file__).resolve().parent
PAGES_PATH = BASE / "data" / "extracted_pages.json"
OVERLAP_WORDS = 50
CONFIGS = {"c300": 300, "c600": 600}


def paragraphs_with_pages(pages: list[dict]) -> Iterator[tuple[str, int]]:
    """Yield (sentence_text, pdf_page) over the whole document stream."""
    for pg in pages:
        # Split page into sentence-level semantic units
        sents = re.split(r"(?<=[.!?])\s+", pg["text"])
        for s in sents:
            cleaned = re.sub(r"\s+", " ", s).strip()
            if cleaned:
                yield cleaned, pg["page"]


def chunk_stream(paras: list[tuple[str, int]], size: int, overlap: int,
                 cid: str) -> list[dict]:
    """Sliding-window chunking over the sentence stream.

    Greedy: accumulate sentences until adding the next would exceed `size`
    words; then emit the buffer, and roll back to trailing sentences
    totalling <= `overlap` words to seed the next chunk.
    """
    chunks: list[dict] = []
    buf: list[tuple[str, int]] = []
    n_words = 0

    def flush() -> None:
        if not buf:
            return
        body = " ".join(t[0] for t in buf)
        page_list = sorted(set(t[1] for t in buf))
        chunks.append({
            "chunk_id": f"{cid}-{len(chunks):04d}",
            "pages": page_list,
            "primary_page": page_list[0],
            "text": body,
            "n_words": len(body.split()),
        })

    for text, page in paras:
        w = len(text.split())
        if buf and (n_words + w > size):
            flush()
            kept: list[tuple[str, int]] = []
            kw = 0
            for item in reversed(buf):
                iw = len(item[0].split())
                if kw + iw > overlap:
                    break
                kept.insert(0, item)
                kw += iw
            buf, n_words = kept, kw
        buf.append((text, page))
        n_words += w

    flush()
    return chunks


def main() -> None:
    if not PAGES_PATH.exists():
        raise SystemExit("Run extract_pdf.py first.")
    pages = json.loads(PAGES_PATH.read_text(encoding="utf-8"))
    paras = list(paragraphs_with_pages(pages))
    print(f"Document stream: {len(paras)} sentences, "
          f"{sum(len(t.split()) for t, _ in paras):,} words across {len(pages)} pages")
    for cid, size in CONFIGS.items():
        all_chunks = chunk_stream(paras, size, OVERLAP_WORDS, cid)
        out = BASE / "data" / f"chunks_{size}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for ch in all_chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        words = [c["n_words"] for c in all_chunks]
        print(f"{cid}: {len(all_chunks)} chunks | avg {sum(words)/len(words):.1f} words "
              f"| min {min(words)} max {max(words)} | overlap={OVERLAP_WORDS}w -> {out}")


if __name__ == "__main__":
    main()
