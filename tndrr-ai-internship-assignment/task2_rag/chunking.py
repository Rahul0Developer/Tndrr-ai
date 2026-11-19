"""Task 2 – Word-based chunking with fixed overlap over the document stream.

Policy (documented):
  * Chunk size: configurable in words (we evaluate 300 and 600).
  * Overlap: 50 words for BOTH configurations (constant across the study).
  * The document is treated as ONE continuous paragraph stream. This matters:
    with ~321 words per PDF page, a naive page-wise split would make the 300w
    and 600w configs produce identical chunks. Stream chunking lets chunks
    cross page boundaries; each chunk stores the FULL list of PDF pages it
    spans plus a primary page for citation.
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
    """Yield (paragraph_text, pdf_page) over the whole document stream."""
    for pg in pages:
        for block in re.split(r"\n\s*\n", pg["text"]):
            p = re.sub(r"\s+", " ", block).strip()
            if p:
                yield p, pg["page"]


def chunk_stream(paras: list[tuple[str, int]], size: int, overlap: int,
                 cid: str) -> list[dict]:
    """Sliding-window chunking over the paragraph stream.

    Greedy: accumulate paragraphs until adding the next would exceed `size`
    words; then emit the buffer, and roll back to the trailing paragraphs
    totalling <= `overlap` words to seed the next chunk.
    """
    chunks: list[dict] = []
    buf_idx: list[int] = []          # indices of paragraphs in current buffer
    n_words = 0

    def flush() -> None:
        if not buf_idx:
            return
        body = " ".join(paras[i][0] for i in buf_idx)
        page_list = sorted({paras[i][1] for i in buf_idx})
        chunks.append({
            "chunk_id": f"{cid}-{len(chunks):04d}",
            "pages": page_list,
            "primary_page": page_list[0],
            "text": body,
            "n_words": len(body.split()),
        })

    for i, (para, _page) in enumerate(paras):
        w = len(para.split())
        if buf_idx and n_words + w > size:
            flush()
            kept: list[int] = []
            kw = 0
            for j in reversed(buf_idx):
                qw = len(paras[j][0].split())
                if kw + qw > overlap:
                    break
                kept.insert(0, j)
                kw += qw
            buf_idx, n_words = kept, kw
        buf_idx.append(i)
        n_words += w
    flush()
    return chunks


def main() -> None:
    if not PAGES_PATH.exists():
        raise SystemExit("Run extract_pdf.py first.")
    pages = json.loads(PAGES_PATH.read_text(encoding="utf-8"))
    paras = list(paragraphs_with_pages(pages))
    print(f"Document stream: {len(paras)} paragraphs, "
          f"{sum(len(t.split()) for t, _ in paras):,} words")
    for cid, size in CONFIGS.items():
        all_chunks = chunk_stream(paras, size, OVERLAP_WORDS, cid)
        out = BASE / "data" / f"chunks_{size}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for ch in all_chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        words = [c["n_words"] for c in all_chunks]
        print(f"{cid}: {len(all_chunks)} chunks | avg {sum(words)/len(words):.0f} words "
              f"| min {min(words)} max {max(words)} | overlap={OVERLAP_WORDS}w -> {out}")


if __name__ == "__main__":
    main()
