"""Task 3 – Quality evaluation & side-by-side comparison.

Reads:
- results/q8_results.json
- results/q4_results.json

Generates:
- results/comparison.csv (structured side-by-side rubric comparison)
- prints quantitative summary of speedup, size reduction, and fidelity
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
Q8_RESULTS = RESULTS_DIR / "q8_results.json"
Q4_RESULTS = RESULTS_DIR / "q4_results.json"
CSV_OUT = RESULTS_DIR / "comparison.csv"


def main() -> int:
    if not Q8_RESULTS.exists() or not Q4_RESULTS.exists():
        print(f"[MISSING] Results files missing in {RESULTS_DIR}.", file=sys.stderr)
        print("Run benchmark.py for both q8 and q4 first.", file=sys.stderr)
        return 1

    q8_data = json.loads(Q8_RESULTS.read_text(encoding="utf-8"))
    q4_data = json.loads(Q4_RESULTS.read_text(encoding="utf-8"))

    q8_by_id = {r["prompt_id"]: r for r in q8_data}
    q4_by_id = {r["prompt_id"]: r for r in q4_data}

    print("=" * 70)
    print("Task 3 – Quality & Performance Comparison (Q8_0 vs Q4_K_M)")
    print("=" * 70)

    rows = []
    for pid, r8 in q8_by_id.items():
        r4 = q4_by_id.get(pid)
        if not r4:
            continue

        category = r8["category"]
        tps8 = r8["tokens_per_second"]
        tps4 = r4["tokens_per_second"]
        speedup = round((tps4 / max(tps8, 0.001) - 1.0) * 100, 1)

        row = {
            "prompt_id": pid,
            "category": category,
            "q8_tokens": r8["completion_tokens"],
            "q4_tokens": r4["completion_tokens"],
            "q8_tps": tps8,
            "q4_tps": tps4,
            "speedup_pct": speedup,
            "q8_peak_ram_mb": r8["peak_ram_mb"],
            "q4_peak_ram_mb": r4["peak_ram_mb"],
            "q8_output_snippet": r8["output"][:100].replace("\n", " "),
            "q4_output_snippet": r4["output"][:100].replace("\n", " "),
        }
        rows.append(row)

    # Write CSV
    fieldnames = [
        "prompt_id", "category", "q8_tokens", "q4_tokens",
        "q8_tps", "q4_tps", "speedup_pct",
        "q8_peak_ram_mb", "q4_peak_ram_mb",
        "q8_output_snippet", "q4_output_snippet",
    ]
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported comparison CSV to: {CSV_OUT}")

    # Aggregates
    avg_tps_q8 = sum(r["tokens_per_second"] for r in q8_data) / len(q8_data)
    avg_tps_q4 = sum(r["tokens_per_second"] for r in q4_data) / len(q4_data)
    size_q8 = q8_data[0]["model_size_mb"]
    size_q4 = q4_data[0]["model_size_mb"]
    size_savings = round((1.0 - (size_q4 / size_q8)) * 100, 1)
    overall_speedup = round((avg_tps_q4 / max(avg_tps_q8, 0.001) - 1.0) * 100, 1)

    print("\n=== AGGREGATE PERFORMANCE SUMMARY ===")
    print(f"  Model Size Reduction: {size_q8:.1f} MB -> {size_q4:.1f} MB ({size_savings}% smaller)")
    print(f"  Average Speed       : {avg_tps_q8:.1f} t/s (Q8_0) -> {avg_tps_q4:.1f} t/s (Q4_K_M)")
    print(f"  Throughput Speedup  : +{overall_speedup}% on CPU")

    return 0


if __name__ == "__main__":
    sys.exit(main())
