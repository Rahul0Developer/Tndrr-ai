"""Task 2 – Download the NIST AI Risk Management Framework 1.0 PDF.

Verified source: NIST AI 100-1 (April 2023), 48 pages, public domain US
government work. Saved to data/nist_ai_rmf_1.0.pdf.

Run:  python download_document.py   (laptop CPU, ~2 MB download)
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
OUT = Path(__file__).resolve().parent / "data" / "nist_ai_rmf_1.0.pdf"
MIN_PAGES = 30


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 100_000:
        print(f"PDF already present: {OUT} ({OUT.stat().st_size:,} bytes)")
    else:
        print(f"Downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, OUT.open("wb") as f:
            f.write(r.read())
        print(f"Saved {OUT} ({OUT.stat().st_size:,} bytes)")

    # Page-count verification using PyMuPDF
    import fitz  # noqa: E402  (PyMuPDF)
    doc = fitz.open(OUT)
    pages = doc.page_count
    words = sum(len(p.get_text().split()) for p in doc)
    print(f"Pages: {pages} | total extracted words: {words:,} | sha256: {sha256(OUT)[:16]}...")
    if pages < MIN_PAGES:
        sys.exit(f"FAIL: document has only {pages} pages (<{MIN_PAGES} required)")
    print(f"OK: document meets the >= {MIN_PAGES} page requirement.")


if __name__ == "__main__":
    main()
