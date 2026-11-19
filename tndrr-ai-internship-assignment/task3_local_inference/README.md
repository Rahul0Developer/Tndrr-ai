# Task 3 – Local inference with llama.cpp (CPU, Windows 11)

Model: **Qwen2.5-0.5B-Instruct** in GGUF format, two quantizations: `Q8_0`
(higher precision) and `Q4_K_M` (lower precision). Same 10 prompts, same
sampling settings, context length 2048, CPU only.

## Setup (PowerShell)

```powershell
# 1. Get llama.cpp (prebuilt release for Windows)
#    Download llama-b<latest>-bin-windows-cpu.zip from
#    https://github.com/ggerganov/llama.cpp/releases and unzip to .\llama.cpp

# 2. Download the two GGUF files (~600 MB total disk needed)
python download_model.py

# 3. Run the benchmark for each model (writes results/q8_results.json / q4_results.json)
python benchmark.py --model models\qwen2.5-0.5b-instruct-q8_0.gguf   --tag q8
python benchmark.py --model models\qwen2.5-0.5b-instruct-q4_k_m.gguf  --tag q4

# 4. Compare outputs + build comparison.csv
python evaluate_quality.py

# 5. Theoretical memory/KV-cache calculations (clearly labelled as estimates)
python memory_estimation.py
```

`benchmark.py` shells out to `llama-cli` (or `main.exe`) with fixed flags:
`-ngl 0 -c 2048 -n 256 -t <physical cores> --temp 0 -s 42` and parses its
printed timings (prompt eval t/s, eval t/s). Peak RAM is sampled with
`psutil` in a background thread while each run executes. All numbers in
`results/` come from actual runs on your machine — nothing is pre-filled.

## Reproducibility controls
* Identical prompts (`prompts.json`, 10 categories per the brief).
* Greedy decoding (`--temp 0`), fixed seed, max 256 new tokens, ctx 2048.
* Close other applications; note ambient system load in the results file.

## Quality rubric (evaluate_quality.py)
Correctness / instruction-following / completeness / JSON-or-code validity,
scored 0–2 by you after reading both outputs side by side (the script fills
a template CSV; scores must be entered honestly after inspection).

## Concept notes (see report/)
* KV cache stores per-layer keys+values of all previous tokens; grows linearly
  with context length × concurrent sequences.
* Quantization shrinks weight bytes/token (Q8 ≈ 1 byte/weight, Q4_K_M ≈ 0.5);
  speed on CPU is usually memory-bandwidth-bound, so Q4 tends to generate more
  tokens/s; quality differences are measured, not assumed.
