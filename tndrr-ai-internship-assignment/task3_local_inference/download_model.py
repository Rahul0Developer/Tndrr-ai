"""Task 3 – Download Qwen2.5-0.5B-Instruct GGUF quantization variants.

Downloads:
1. Q8_0   (~524 MB): 8-bit precision baseline
2. Q4_K_M (~352 MB): 4-bit medium k-quantization

Source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
Destination: models/

Run:
    python download_model.py
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODELS = {
    "q8": {
        "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf",
        "expected_bytes": 524_288_000,   # approx 500 MB
        "quant_type": "Q8_0 (8-bit quantization)",
    },
    "q4": {
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "expected_bytes": 352_000_000,   # approx 335 MB
        "quant_type": "Q4_K_M (4-bit medium k-quant)",
    },
}


def download_with_progress(url: str, dest_path: Path) -> None:
    print(f"Connecting to: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    with urllib.request.urlopen(req, timeout=120) as resp:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        start_time = time.time()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(".download")

        with temp_path.open("wb") as out_file:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                elapsed = time.time() - start_time
                speed = downloaded / (1024 * 1024 * max(elapsed, 0.001))
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {pct:5.1f}% [{mb_cur:6.1f} / {mb_tot:6.1f} MB] @ {speed:5.1f} MB/s")
                else:
                    sys.stdout.write(f"\r  Downloaded: {downloaded / (1024 * 1024):.1f} MB @ {speed:.1f} MB/s")
                sys.stdout.flush()

        temp_path.replace(dest_path)
        print(f"\nCompleted: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-0.5B-Instruct GGUF models")
    parser.add_argument("--variant", choices=["all", "q8", "q4"], default="all",
                        help="Which quantization variant to download")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selected = ["q8", "q4"] if args.variant == "all" else [args.variant]

    print("=" * 65)
    print("Task 3 – GGUF Model Downloader")
    print(f"Target Directory: {MODELS_DIR}")
    print("=" * 65)

    for key in selected:
        info = MODELS[key]
        dest = MODELS_DIR / info["filename"]
        print(f"\n[{key.upper()}] {info['quant_type']}")
        if dest.exists() and dest.stat().st_size > 100_000_000:
            print(f"  Model already present: {dest.name} ({dest.stat().st_size / (1024*1024):.1f} MB)")
        else:
            download_with_progress(info["url"], dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
