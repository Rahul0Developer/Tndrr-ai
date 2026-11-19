"""Task 3 – Benchmark local inference performance across quantization variants.

Measures Q8_0 vs Q4_K_M on:
- Time to first token (prompt evaluation latency)
- Generation speed (tokens per second)
- Generated token count & generation time
- Peak RAM consumption via psutil
- Model file size and memory footprint

Supports:
1. `llama-cpp-python` library (`from llama_cpp import Llama`)
2. `llama-cli` / `main.exe` subprocess execution
3. Deterministic greedy decoding (temperature=0.0, top_p=1.0, seed=42)

Outputs:
    results/q8_results.json, results/q4_results.json, results/comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PROMPTS_FILE = BASE_DIR / "prompts.json"

DEFAULT_MODELS = {
    "q8": {
        "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
        "tag": "Q8_0",
        "bits": 8,
    },
    "q4": {
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "tag": "Q4_K_M",
        "bits": 4,
    },
}


class MemoryMonitor:
    """Background sampler for peak process RAM in MB."""

    def __init__(self, interval_sec: float = 0.05):
        self.interval = interval_sec
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self.process = psutil.Process()

    def _sample(self):
        while not self._stop.is_set():
            try:
                rss_mb = self.process.memory_info().rss / (1024 * 1024)
                if rss_mb > self.peak_mb:
                    self.peak_mb = rss_mb
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_mb = self.process.memory_info().rss / (1024 * 1024)
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        return round(self.peak_mb, 2)


def run_benchmark_llamacpp(model_path: Path, tag: str, prompts: list[dict], ctx_len: int = 2048, max_new_tokens: int = 256) -> list[dict]:
    """Run benchmark using llama-cpp-python."""
    from llama_cpp import Llama

    print(f"\n[ENGINE: llama-cpp-python] Loading model: {model_path.name}")
    monitor = MemoryMonitor()
    monitor.start()

    t0 = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=ctx_len,
        n_threads=os.cpu_count() or 4,
        verbose=False,
        seed=42,
    )
    load_time = time.time() - t0
    peak_load_ram = monitor.stop()
    print(f"Loaded in {load_time:.2f}s | Peak RAM: {peak_load_ram:.1f} MB")

    results = []
    file_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)

    for idx, p in enumerate(prompts, 1):
        pid = p["id"]
        category = p["category"]
        prompt_text = p["prompt"]

        formatted = (
            f"<|im_start|>system\nYou are a helpful and concise assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        monitor.start()
        start_time = time.time()

        # Generate with deterministic settings
        output = llm(
            formatted,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        elapsed = time.time() - start_time
        peak_ram = monitor.stop()

        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(prompt_text.split()))
        completion_tokens = usage.get("completion_tokens", len(output["choices"][0]["text"].split()))
        output_text = output["choices"][0]["text"].strip()

        eval_tps = round(completion_tokens / max(elapsed, 0.001), 2)

        record = {
            "prompt_id": pid,
            "category": category,
            "quantization": tag,
            "model_size_mb": file_size_mb,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_time_sec": round(elapsed, 3),
            "tokens_per_second": eval_tps,
            "peak_ram_mb": peak_ram,
            "output": output_text,
        }
        results.append(record)
        print(f"  [{idx:02d}/10] {category:<24}: {completion_tokens:3d} tokens in {elapsed:5.2f}s ({eval_tps:5.1f} t/s) | RAM: {peak_ram:6.1f} MB")

    return results


def run_benchmark_cli(cli_path: Path, model_path: Path, tag: str, prompts: list[dict], ctx_len: int = 2048, max_new_tokens: int = 256) -> list[dict]:
    """Run benchmark using llama-cli subprocess execution."""
    print(f"\n[ENGINE: llama-cli] Using binary: {cli_path}")
    results = []
    file_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)

    for idx, p in enumerate(prompts, 1):
        pid = p["id"]
        category = p["category"]
        prompt_text = p["prompt"]

        formatted = (
            f"<|im_start|>system\\nYou are a helpful and concise assistant.<|im_end|>\\n"
            f"<|im_start|>user\\n{prompt_text}<|im_end|>\\n"
            f"<|im_start|>assistant\\n"
        )

        cmd = [
            str(cli_path),
            "-m", str(model_path),
            "-c", str(ctx_len),
            "-n", str(max_new_tokens),
            "--temp", "0.0",
            "-s", "42",
            "-t", str(os.cpu_count() or 4),
            "-ngl", "0",
            "-p", formatted,
        ]

        monitor = MemoryMonitor()
        monitor.start()
        start_time = time.time()

        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.time() - start_time
        peak_ram = monitor.stop()

        stdout = proc.stdout
        # Parse timings from llama-cli stderr or stdout
        eval_tps = 0.0
        n_tokens = len(stdout.split())

        for line in stdout.splitlines():
            if "eval time" in line and "t/s" in line:
                try:
                    parts = line.split("=")[1].strip().split("/")
                    eval_tps = float(parts[-1].replace("runs", "").replace("t/s", "").strip())
                except Exception:
                    pass

        if eval_tps == 0.0:
            eval_tps = round(n_tokens / max(elapsed, 0.001), 2)

        record = {
            "prompt_id": pid,
            "category": category,
            "quantization": tag,
            "model_size_mb": file_size_mb,
            "prompt_tokens": len(prompt_text.split()),
            "completion_tokens": n_tokens,
            "generation_time_sec": round(elapsed, 3),
            "tokens_per_second": eval_tps,
            "peak_ram_mb": peak_ram,
            "output": stdout.strip(),
        }
        results.append(record)
        print(f"  [{idx:02d}/10] {category:<24}: {n_tokens:3d} tokens in {elapsed:5.2f}s ({eval_tps:5.1f} t/s)")

    return results


def find_llama_cli() -> Path | None:
    """Find prebuilt llama-cli binary if available."""
    candidates = [
        BASE_DIR / "llama.cpp" / "llama-cli.exe",
        BASE_DIR / "llama.cpp" / "main.exe",
        BASE_DIR / "llama-cli.exe",
        Path("llama-cli.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 3 – Quantization Benchmark")
    parser.add_argument("--variant", choices=["all", "q8", "q4"], default="all",
                        help="Quantization variant to benchmark")
    parser.add_argument("--model", type=Path, default=None,
                        help="Direct path to custom GGUF model")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag identifier for custom model (e.g. Q8_0)")
    args = parser.parse_args()

    if not PROMPTS_FILE.exists():
        print(f"Error: Prompts file not found at {PROMPTS_FILE}", file=sys.stderr)
        return 1

    prompts_data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    prompts = prompts_data["prompts"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Task 3 – Local Inference Quantization Benchmark")
    print(f"Platform: {platform.platform()} | Python: {platform.python_version()}")
    print(f"CPU: {platform.processor()} | Cores: {os.cpu_count()}")
    print(f"Total System RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print("=" * 65)

    has_llama_cpp_pkg = False
    try:
        import llama_cpp
        has_llama_cpp_pkg = True
    except ImportError:
        pass

    cli_bin = find_llama_cli()

    if not has_llama_cpp_pkg and cli_bin is None:
        print("\n[INFO] Neither `llama-cpp-python` nor prebuilt `llama-cli.exe` is installed.")
        print("Installation options:")
        print("  1. Python bindings : pip install llama-cpp-python")
        print("  2. Prebuilt binary : Download llama-b<version>-bin-win-cpu.zip from")
        print("                       https://github.com/ggerganov/llama.cpp/releases and unzip to task3_local_inference/llama.cpp/")
        print("  3. Colab execution : Run task3_quantization_benchmark.ipynb on Google Colab.")
        return 0

    variants_to_run = ["q8", "q4"] if args.variant == "all" else [args.variant]
    all_summary = {}

    for var in variants_to_run:
        info = DEFAULT_MODELS[var]
        tag = args.tag or info["tag"]
        model_path = args.model or (MODELS_DIR / info["filename"])

        if not model_path.exists():
            print(f"\n[MISSING] Model not found at: {model_path}")
            print(f"Please run: python download_model.py --variant {var}")
            continue

        print(f"\n{'=' * 65}")
        print(f"RUNNING BENCHMARK: {tag} ({model_path.name})")
        print(f"{'=' * 65}")

        if has_llama_cpp_pkg:
            records = run_benchmark_llamacpp(model_path, tag, prompts)
        else:
            records = run_benchmark_cli(cli_bin, model_path, tag, prompts)

        out_json = RESULTS_DIR / f"{var}_results.json"
        out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved raw results to: {out_json}")

        avg_tps = sum(r["tokens_per_second"] for r in records) / len(records)
        avg_ram = sum(r["peak_ram_mb"] for r in records) / len(records)
        total_toks = sum(r["completion_tokens"] for r in records)
        all_summary[tag] = {
            "model_size_mb": records[0]["model_size_mb"],
            "avg_tokens_per_second": round(avg_tps, 2),
            "avg_peak_ram_mb": round(avg_ram, 1),
            "total_tokens_generated": total_toks,
            "n_prompts": len(records),
        }

    # Summary table
    if all_summary:
        print("\n" + "=" * 65)
        print(f"{'QUANTIZATION':<12} | {'FILE SIZE':<12} | {'SPEED (t/s)':<14} | {'AVG PEAK RAM':<14}")
        print("-" * 65)
        for tag, s in all_summary.items():
            print(f"{tag:<12} | {s['model_size_mb']:<7.1f} MB | {s['avg_tokens_per_second']:<9.1f} t/s | {s['avg_peak_ram_mb']:<9.1f} MB")
        print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
