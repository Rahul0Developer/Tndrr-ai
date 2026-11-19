"""Task 2 – Extract and clean text from the NIST AI RMF PDF, page by page.

Cleaning rules (documented in README):
  * Drop the running header "NIST AI 100-1 AI RMF 1.0" repeated on pages.
  * Collapse hyphenated line breaks ("of- fers" -> "offers").
  * Normalise whitespace but keep paragraph line structure for chunking.
Output: data/extracted_pages.json  = list of {page, text} with 1-based PDF
page numbers preserved as source metadata.

Run:  python extract_pdf.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

BASE = Path(__file__).resolve().parent
PDF_PATH = BASE / "data" / "nist_ai_rmf_1.0.pdf"
OUT_PATH = BASE / "data" / "extracted_pages.json"

HEADER_RE = re.compile(r"^NIST AI 100-1\s*AI RMF 1\.0\s*$", re.MULTILINE)


def clean_page(text: str) -> str:
    """Apply documented cleaning rules to one page's raw text."""
    text = HEADER_RE.sub("", text)
    # Join words split across lines by hyphenation (but not em/en dashes).
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalise exotic unicode spaces and ligatures introduced by extraction.
    text = text.replace("\u00a0", " ").replace("\ufb01", "fi").replace("\ufb02", "fl")
    # Collapse blank-line runs; strip trailing spaces per line.
    lines = [re.sub(r"\s+$", "", ln) for ln in text.splitlines()]
    out, prev_blank = [], False
    for ln in lines:
        blank = not ln.strip()
        if blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = blank
    return "\n".join(out).strip()


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit("PDF missing – run download_document.py first.")
    doc = fitz.open(PDF_PATH)
    pages = []
    for i, page in enumerate(doc):
        txt = clean_page(page.get_text("text"))
        if txt:
            pages.append({"page": i + 1, "text": txt})
    OUT_PATH.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    total_words = sum(len(p["text"].split()) for p in pages)
    print(f"Extracted {len(pages)} non-empty pages, {total_words:,} words -> {OUT_PATH}")
    print(f"Sample page 9 (first 200 chars):\n{pages[5]['text'][:200]!r}")


if __name__ == "__main__":
    main()
