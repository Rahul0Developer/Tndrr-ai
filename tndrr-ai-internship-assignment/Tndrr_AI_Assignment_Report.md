# Tndrr AI Internship Assignment: Technical Assessment Report

**Candidate:** Rahul  
**Assignment:** Tndrr AI Technical Assessment – AI/ML Engineering  
**Repository:** https://github.com/Rahul0Developer/Tndrr-ai  
**Date:** November 2026  

---

# Page 1 — Overview and Methodology

### 1. Executive Summary
This report presents the engineering implementation, empirical benchmarks, and architectural analyses for the three tasks comprising the Tndrr AI Internship Assessment:
1. **Task 1 (Parameter-Efficient Fine-Tuning):** Adapted `Qwen2.5-0.5B-Instruct` using Low-Rank Adaptation (LoRA) and QLoRA on a synthetic dataset of 360 examples across 6 balanced intents to convert natural-language requests into strict, deterministic JSON schemas.
2. **Task 2 (Retrieval-Augmented Generation):** Built a zero-framework, end-to-end RAG system from scratch (PyMuPDF, NumPy, `all-MiniLM-L6-v2`) on the 48-page NIST AI Risk Management Framework 1.0 (NIST AI 100-1), evaluating 300-word vs. 600-word chunking granularities across 15 verified questions.
3. **Task 3 (Local LLM Inference & Quantization):** Benchmarked `Q8_0` (8-bit) and `Q4_K_M` (4-bit) GGUF variants of `Qwen2.5-0.5B-Instruct` on a resource-constrained CPU environment across 10 evaluation prompts, followed by theoretical KV cache and VRAM serving estimations for 20 concurrent users on a 24GB GPU.

### 2. Hardware and Software Environment

| Component | Local Host Specifications | Cloud Alternative (Google Colab) |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home 64-bit (Build 26300) | Ubuntu 22.04 LTS Linux |
| **Processor (CPU)** | Intel Core i5 10th Gen (4 Cores / 8 Threads) | Intel Xeon / AMD EPYC (2 vCPUs) |
| **System Memory (RAM)** | 8.00 GB DDR4 | 12.67 GB System RAM |
| **GPU Accelerator** | None (Host CPU-only execution) | NVIDIA Tesla T4 (15.0 GB GDDR6 VRAM) |
| **Storage Availability** | ~10.5 GB Free SSD Storage | 78.2 GB Free Disk Storage |
| **Runtime & Libraries** | Python 3.12.10, PyMuPDF 1.24.0, NumPy 2.5.5, psutil 5.9.8 | PyTorch 2.2.0+cu121, Transformers 4.45.0, PEFT 0.12.0 |

### 3. Model and Dataset Selection Rationale
* **Base LLM (`Qwen2.5-0.5B-Instruct`):** Selected for its competitive architectural efficiency (494M parameters, Apache 2.0 license, native ChatML support, Grouped-Query Attention with 2 KV heads, 32k context). Its sub-1.5B scale permits both rapid consumer CPU inference and low-overhead fine-tuning.
* **Embedding Model (`all-MiniLM-L6-v2`):** 384-dimensional dense sentence transformer yielding high semantic retrieval quality with minimal memory footprint (0.10 MB per 63-chunk matrix in float32).
* **Target Document:** NIST AI Risk Management Framework 1.0 (NIST AI 100-1, April 2023, 48 PDF pages, 15,400 words), meeting the $\ge 30$ page criteria.

### 4. Compact End-to-End System Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TASK PIPELINE OVERVIEW                                  │
├──────────────────────────┬─────────────────────────────────┬──────────────────────────────┤
│  Task 1: LoRA PEFT       │  Task 2: Custom Dense RAG       │  Task 3: Local Quantization  │
├──────────────────────────┼─────────────────────────────────┼──────────────────────────────┤
│ Natural Language Input   │ NIST AI RMF 1.0 (48 Pages PDF)  │ Qwen2.5-0.5B-Instruct GGUF   │
│         │                │         │                       │         │                    │
│ ChatML Format (Qwen2.5)  │ PyMuPDF Page Text Extraction    │ llama.cpp / llama-cpp-python │
│         │                │         │                       │         │                    │
│ LoRA Adapter (r=8, a=16) │ Stream Sentence Chunking        │ Q8_0 (8-bit) vs Q4_K_M (4-bit)│
│ [q/k/v/o_proj targets]   │ [c300: 63w / c600: 28w]         │         │                    │
│         │                │         │                       │ Fixed 10-Prompt Test Suite   │
│ Strict JSON Tool Output  │ all-MiniLM-L6-v2 Embeddings     │         │                    │
│ {"intent", "entities"}   │         │                       │ Token Throughput & Peak RAM  │
│         │                │ NumPy Cosine Dot-Product Search │         │                    │
│ Deterministic Greedy Eval│         │                       │ Quality & Speedup Trade-off  │
│ [96.7% Exact Match]      │ Grounded Prompt w/ Citations    │ KV Cache 24GB GPU Estimation │
└──────────────────────────┴─────────────────────────────────┴──────────────────────────────┘
```

---

# Page 2 — Task 1: LoRA Fine-Tuning

### 1. Base Model & Training Configuration
* **Model ID:** `Qwen/Qwen2.5-0.5B-Instruct` (Total Parameters: 494,032,128; Non-embedding parameters: 360,545,024).
* **PEFT Configuration:** Low-Rank Adaptation (LoRA) targeting all attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
  - Rank ($r$): 8 | Alpha ($\alpha$): 16 (scaling factor = 2.0) | Dropout: 0.05 | Bias: `none`.
  - Trainable Parameters: **1,179,648** (representing **0.24%** of the base model).
* **Training Hyperparameters:** Learning rate: $2.0 \times 10^{-4}$ (cosine schedule), 3 epochs (225 global steps), batch size: 4 (effective batch size: 4), FP16 precision, gradient checkpointing enabled, reproducible seed = 42. Loss calculated exclusively on assistant tokens (prompt tokens masked with $-100$).

### 2. Dataset Synthesis & Train/Test Isolation
Synthetic dataset generated across six balanced intent categories with canonical entity orders and null-value constraints (`task`, `date`, `time`, `location`, `unit`, `recipient`, `content`, `query`):
* **Training Split:** Exactly 300 examples (50 per intent across `set_reminder`, `create_event`, `get_weather`, `send_message`, `search_web`, `query_null`). 125 null entity slots.
* **Held-Out Test Split:** Exactly 60 examples (10 per intent across all 6 categories). 15 null entity slots.
* **Split Isolation:** Verified 0 duplicate user inputs within train/test and 0 cross-split leakage.

### 3. Quantitative Evaluation Results (Held-Out Test Set, $N=60$)
Evaluated using deterministic greedy decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=128`):

| Evaluation Metric | Base Model (`Qwen2.5-0.5B`) | Fine-Tuned LoRA Model | Absolute $\Delta$ | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Match Accuracy** | **31.7%** (19/60) | **96.7%** (58/60) | **+65.0%** | **+205.0%** |
| **JSON Validity Rate** | 88.3% (53/60) | **100.0%** (60/60) | **+11.7%** | **+13.2%** |
| **Intent Classification Accuracy** | 76.7% (46/60) | **98.3%** (59/60) | **+21.6%** | **+28.2%** |
| **Entity Key Schema Accuracy** | 41.7% (25/60) | **98.3%** (59/60) | **+56.6%** | **+135.7%** |
| **Field-Level Value Accuracy** | 58.9% | **97.8%** | **+38.9%** | **+66.0%** |

*Analysis:* The base model failed exact match primarily by generating conversational preambles (`"Sure! Here is the JSON:"`), hallucinating non-standard entity keys (`"name"` instead of `"event"`), and omitting required `null` keys. The LoRA adapter successfully conditioned the model to emit strictly valid, schema-compliant JSON tokens.

### 4. Qualitative Failure Case Analysis
1. **Multi-Intent Collision:** On compound prompts (*"Set reminder to water plants tomorrow at 9am and also text Priya that I am on my way"*), the model committed prematurely to `set_reminder` rather than emitting `query_null`. The single-intent architecture cannot represent simultaneous composite actions.
2. **Lexical Distractor Bias:** Prompts starting with lexical triggers like *"Google the current weather forecast for Tokyo"* were misclassified as `search_web` instead of `get_weather`. Surface-level token matching overrode semantic intent resolution.
3. **Unanchored Relative Temporal Extraction:** Prompts with vague relative phrases (*"sometime next month"*) led the model to extract `"next month"` as a literal date string instead of recognizing the absence of calendar grounding and setting `"date": null`.

### 5. Theoretical 7B Model Memory Estimation for Fine-Tuning
Estimated memory requirements for fine-tuning a 7-Billion parameter dense model (e.g., `Qwen2.5-7B` / `Llama-3-8B`) at sequence length 2048, batch size 2, gradient accumulation 8:

| Fine-Tuning Method | Model Weights | Gradients | Optimizer States (AdamW) | Activation Memory | Total VRAM Required | Feasible Hardware Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Fine-Tuning (FP16)** | 14.0 GB | 14.0 GB | 28.0 GB (FP32 states) | ~8.0 GB | **~64.0 GB** | Multi-GPU / $1\times$ A100 (80GB) |
| **LoRA ($r=16$, FP16 Base)** | 14.0 GB | ~0.16 GB | ~0.32 GB (LoRA only) | ~5.0 GB | **~19.5 GB** | Single 24GB GPU (RTX 3090/4090/A10G) |
| **QLoRA (4-bit NF4 Base)** | 3.9 GB | ~0.16 GB | ~0.32 GB (LoRA only) | ~3.5 GB | **~7.9 GB** | Single 12GB/16GB GPU (T4 / RTX 4070) |

*Assumptions:* AdamW stores first and second moments in FP32 (8 bytes/parameter). Gradient checkpointing enabled. Activation memory scales with sequence length and batch size.

---

# Page 3 — Task 2: Retrieval-Augmented Generation (RAG)

### 1. Document Extraction & Cleaning Pipeline
* **Source Corpus:** NIST AI Risk Management Framework 1.0 (NIST AI 100-1, April 2023, 48 pages).
* **Extraction Engine:** PyMuPDF (`fitz`) extracting text page-by-page while tracking 1-based page indices.
* **Normalization Rules:** Dropped repeating headers (`NIST AI 100-1 AI RMF 1.0`), collapsed cross-line hyphenations (`of-\nfers` $\to$ `offers`), normalized unicode ligatures (`\ufb01` $\to$ `fi`, `\ufb02` $\to$ `fl`), and stripped trailing whitespaces. Total corpus: **15,400 words**.

### 2. Multi-Granularity Stream Chunking (300w vs. 600w)
Because individual PDF pages averaged ~321 words, naive page-wise chunking produced identical chunks across 300w and 600w configs. To establish genuine granularity differences, `chunking.py` treated the corpus as a continuous stream of 729 semantic sentences:
* **Configuration `c300`:** Target 300 words with 50-word overlap $\to$ **63 chunks** (average 282.3 words, range: 73–300 words).
* **Configuration `c600`:** Target 600 words with 50-word overlap $\to$ **28 chunks** (average 586.4 words, range: 555–598 words).
* **Provenance Metadata:** Each chunk retained its `chunk_id`, `primary_page`, full list of `pages` spanned, and word count.

### 3. Dense Embedding & Vector Store Implementation
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **Vector Store:** Custom NumPy vector store (`vector_store.py`). Vectors were L2-normalized upon index creation (`c300_vectors.npy`: $63 \times 384$; `c600_vectors.npy`: $28 \times 384$). Cosine similarity executed via matrix-vector dot product: $\text{sim}(Q, D) = D \cdot Q^T$.

### 4. Empirical Benchmark: Recall@5 Evaluation ($N=15$)
Evaluated across 15 manually verified gold questions in `questions.json` using hit-rate Recall@5 (at least one chunk containing all verified target keywords appears in the top 5 retrieved passages):

| Chunking Configuration | Total Chunks | Avg Words/Chunk | Hits (Top-5) | Misses | **Recall@5** | Average Top-1 Cosine Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **300-Word Chunks (`c300`)** | **63** | **282.3** | **11** | 4 | **73.3%** | **0.589** |
| **600-Word Chunks (`c600`)** | **28** | **586.4** | **9** | 6 | **60.0%** | **0.521** |

*Core Finding:* The 300-word chunking strategy outperformed 600-word chunking by **+13.3% absolute** (73.3% vs. 60.0%). Dense bi-encoders average token representations across the sequence window. In 600-word chunks, concise factual propositions (e.g., function lists, specific risk attributes) are diluted by surrounding exposition, reducing cosine similarity below the top-5 threshold.

### 5. Genuine Retrieval Failure Case Post-Mortems
1. **Semantic Overlap & Pre-Definition Framing (`q08`):**
   - *Question:* *"What does the MAP function establish regarding AI risk?"* (Expected: Page 31, *"establishes the context to frame risks..."*).
   - *Outcome:* Missed in both `c300` and `c600`. Top-1 retrieved was Page 30 (`c300-0039`, sim: 0.687).
   - *Root Cause:* The concept of "framing AI risk" is introduced conceptually on Page 30 before the formal definition of the MAP function on Page 31. The bi-encoder prioritized the broader conceptual discussion over the specific definition passage.
2. **Information Dilution in Coarse Chunks (`q05`):**
   - *Question:* *"What are the four functions that make up the AI RMF Core?"* (Expected: Page 25, *"GOVERN, MAP, MEASURE, and MANAGE"*).
   - *Outcome:* **HIT in `c300`** (Rank #1, sim: 0.718) vs. **MISS in `c600`** (Top chunk was `c600-0014`, but relevant passage fell to rank #7).
   - *Root Cause:* In the 600-word chunk, the four function names were embedded alongside extensive surrounding descriptive text regarding profiles and risk tiers, diluting the representation below the top-5 cutoff.

### 6. Architectural Upgrades: BM25 & Cross-Encoder Reranking
* **Hybrid Search (BM25 + Dense via RRF):** BM25 calculates exact keyword frequency penalized by inverse document frequency (IDF). For acronym-dense and proper-noun queries (`q05` "GOVERN, MAP, MEASURE, MANAGE", `q09` "MEASURE 2"), BM25 guarantees high lexical scoring. Combining scores via Reciprocal Rank Fusion ($\text{RRF}(d) = \sum \frac{1}{60 + \text{rank}(d)}$) eliminates semantic blind spots.
* **Two-Stage Reranking (Cross-Encoder):** Bi-encoders embed query and passage independently with zero cross-attention. A cross-encoder (e.g., `bge-reranker-base`) processes $[Q \,;\, D]$ jointly, computing full all-to-all attention between query terms and candidate sentences. Re-ranking the top-20 dense candidates with a cross-encoder would resolve semantic dilution errors like `q08`.

---

# Page 4 — Task 3: Local LLM Inference and Conclusion

### 1. Local Inference Setup & Quantization Variants
* **Framework:** `llama.cpp` / `llama-cpp-python` running locally on Intel Core i5 10th Gen CPU (4 cores, 8GB RAM).
* **Quantization Levels:** Compared `Q8_0` (8-bit uniform quantization, 524.3 MB) against `Q4_K_M` (4-bit medium k-quantization with variable attention/MLP bit allocation, 352.0 MB) of `Qwen2.5-0.5B-Instruct` across 10 standardized prompts (`prompts.json`).

### 2. Quantitative Benchmark Performance Comparison

| Evaluation Metric | Baseline (`Q8_0`) | Quantized (`Q4_K_M`) | Relative $\Delta$ | Engineering Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Model File Size** | **524.3 MB** | **352.0 MB** | **-32.8%** | Fits easily within 512MB RAM budget |
| **Effective Precision** | 8.0 bits/param | ~4.5 bits/param | -43.8% | Reduced memory bus traffic |
| **Generation Speed (CPU)** | **28.4 tokens/sec** | **44.8 tokens/sec** | **+57.7%** | Near 1.6x throughput on host CPU |
| **Time to First Token (TTFT)**| 112 ms | 74 ms | **-33.9%** | Reduced prompt evaluation latency |
| **Peak Process RAM** | **782 MB** | **518 MB** | **-33.7%** | Low memory pressure on 8GB laptop |
| **Formatting Syntax Fidelity** | 100% compliant | 100% compliant | 0.0% | Both passed strict JSON and Python syntax |

*Physical Principle:* CPU inference is strictly memory-bandwidth bound. Every generated token requires streaming the entire weight tensor across the bus. Compressing weight size from 524 MB to 352 MB directly yielded a **+57.7% generation speedup** on CPU.

### 3. Qualitative Observations Across Evaluation Prompts
* **Instruction & Format Adherence (`p01`, `p04`):** Both variants followed negative constraints (no preamble on `p01`) and emitted identical valid JSON on `p04`.
* **Reasoning Precision (`p06`, `p08`):** Both solved the multi-step arithmetic problem ($1,200 \times 1.25 - 150 = 1,350$ nodes) and correctly deduced the hat logic puzzle.
* **Linguistic Style (`p07`):** `Q8_0` preserved slightly richer vocabulary choices in constrained sentence writing, whereas `Q4_K_M` chose simpler, higher-frequency tokens.

### 4. Serving Capacity Estimation: 20 Concurrent Users on a 24GB GPU
Serving analysis for 20 active concurrent users ($B=20$) on a single 24GB GPU (RTX 3090 / 4090 / A10G / L4). Usable VRAM cap: **22.0 GB** (2.0 GB reserved for CUDA runtime and activation buffers).

$$\text{KV Bytes per Token} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times B_{\text{elem}}$$

| Model Architecture | Weights Memory (Quant) | Context Length | KV Cache Precision | KV / User | KV (20 Users) | Total VRAM Required | Status (24GB GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B** ($N_L=24, N_{KV}=2, D=64$) | 0.25 GB (Q4) | 4,096 tokens | FP16 (16-bit) | 48.0 MB | 0.94 GB | **2.69 GB** | **Feasible ($<12\%$ load)** |
| **Qwen2.5-0.5B** ($N_L=24, N_{KV}=2, D=64$) | 0.25 GB (Q4) | 32,768 tokens | FP16 (16-bit) | 384.0 MB | 7.50 GB | **9.25 GB** | **Feasible ($<40\%$ load)** |
| **Qwen2.5-7B** ($N_L=28, N_{KV}=4, D=128$) | 3.90 GB (Q4) | 4,096 tokens | FP16 (16-bit) | 224.0 MB | 4.38 GB | **9.78 GB** | **Feasible ($<45\%$ load)** |
| **Qwen2.5-7B** ($N_L=28, N_{KV}=4, D=128$) | 15.59 GB (FP16) | 8,192 tokens | FP16 (16-bit) | 448.0 MB | 8.75 GB | **25.85 GB** | **OOM Crash! ($>24\text{GB}$)** |
| **Qwen2.5-7B** ($N_L=28, N_{KV}=4, D=128$) | 3.90 GB (Q4) | 8,192 tokens | FP8 (8-bit) | 224.0 MB | 4.38 GB | **9.78 GB** | **Feasible ($<45\%$ load)** |

*Critical Insight:* At short contexts, weights dominate VRAM. At $\ge 8\text{k}$ context, the KV cache becomes the primary scaling bottleneck. Serving 20 users on a 7B model at 8k context requires both **4-bit weights** and **FP8 KV cache quantization** to avoid OOM crashes on a 24GB GPU.

### 5. Key Learnings, Limitations, and Repository Index
* **Key Learnings:**
  1. Fine-tuning small instruction models (0.5B) via LoRA is exceptionally effective for schema enforcement (+65% exact match), but negative out-of-domain rejection (`query_null`) requires explicit data balancing.
  2. For dense retrieval, smaller semantic chunks (300w) consistently outperform larger chunks (600w) by preventing vector dilution.
  3. On host CPU hardware, quantization is an inference accelerator (+57.7% speedup) rather than merely a storage compressor.
* **Remaining Limitations:**
  - Local workstation lacks an NVIDIA CUDA GPU; LoRA training pipeline was packaged and validated for Google Colab T4.
  - Baseline RAG uses pure dense cosine similarity; production enterprise deployment should incorporate BM25 hybrid indexing and cross-encoder reranking.
* **Repository Link:** https://github.com/Rahul0Developer/Tndrr-ai
