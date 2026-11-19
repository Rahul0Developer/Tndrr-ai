# Tndrr AI Technical Assessment Report

**Candidate:** Rahul  
**Role:** AI/ML Engineering Assessment  
**Repository:** https://github.com/Rahul0Developer/Tndrr-ai  
**Date:** November 2026  

---

## Page 1 — Overview and Methodology

### 1. Executive Summary & Assessment Scope
This report presents the engineering implementation, empirical benchmarks, and architectural analyses for the three core tasks of the Tndrr AI Internship Assessment. All metrics reported derive from genuine runs across local host CPU and cloud GPU testbeds:
1. **Task 1 (Parameter-Efficient Fine-Tuning):** Adapted `Qwen2.5-0.5B-Instruct` using Low-Rank Adaptation (LoRA) on 300 custom examples across 6 balanced intents. Fine-tuning enforced strict JSON tool output formatting, eliminating conversational boilerplate and lifting exact-match accuracy from **1.7% to 96.7%** on a 60-example held-out test split.
2. **Task 2 (Zero-Framework Retrieval-Augmented Generation):** Implemented an end-to-end RAG system from first principles (PyMuPDF, NumPy, `all-MiniLM-L6-v2`) over the 48-page *NIST AI Risk Management Framework 1.0*. Benchmarked 300-word versus 600-word chunking over 15 manually verified questions, proving that 300-word chunks deliver superior Recall@5 (**73.3% vs. 60.0%**) due to lower semantic dilution.
3. **Task 3 (Local Inference & Serving Optimization):** Benchmarked 8-bit (`Q8_0`) and 4-bit (`Q4_K_M`) quantized GGUF variants on an Intel Core i5 CPU. Demonstrates that memory-bandwidth constraints make 4-bit quantization **1.58x faster** (44.8 vs 28.4 tok/s) with 34% lower RAM. Derived serving requirements for 20 concurrent users on a 24GB GPU using PagedAttention and GQA KV cache math.

---

### 2. Hardware and Software Environments

| Environment Layer | Local Testbed (Laptop Host) | Training & Cloud GPU Testbed (Google Colab) |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home 64-bit (Build 26300) | Ubuntu 22.04 LTS Linux (Kernel 6.6) |
| **Compute / CPU** | Intel Core i5-1035G1 (4 Cores / 8 Threads, 1.0 GHz base) | Intel Xeon / AMD EPYC (2 vCPUs @ 2.2 GHz) |
| **GPU Accelerator** | None (Pure CPU Host Execution) | NVIDIA Tesla T4 (15.0 GB GDDR6 VRAM, 320 Turing Tensor Cores) |
| **Memory (RAM)** | 8.00 GB DDR4-2666 | 12.67 GB System RAM + 15.0 GB GPU VRAM |
| **Software Stack** | Python 3.12, NumPy 2.5, PyMuPDF 1.24, llama.cpp / psutil | Python 3.13, PyTorch 2.14.1+cu130, Transformers 5.22, PEFT 0.21 |

---

### 3. Model & Corpus Selection Rationale
* **Base LLM (`Qwen2.5-0.5B-Instruct`):** Selected for its competitive architectural efficiency (494M parameters, Apache 2.0 license, native ChatML support, Grouped-Query Attention with 2 KV heads, 32k context). Its sub-1.5B scale permits both rapid consumer CPU inference and low-overhead fine-tuning.
* **Embedding Model (`all-MiniLM-L6-v2`):** 384-dimensional dense sentence transformer yielding high semantic retrieval quality with minimal memory footprint (0.10 MB per 63-chunk matrix in float32).
* **Target Document:** NIST AI Risk Management Framework 1.0 (NIST AI 100-1, April 2023, 48 PDF pages, 15,400 words), meeting the $\ge 30$ page criteria.

---

### 4. End-to-End System Architecture

```text
[Task 1 PEFT]  NL Prompt -> ChatML Format -> Qwen2.5-0.5B + LoRA (r=8, a=16) -> Strict JSON Tool Output (96.7% Exact Match)
[Task 2 RAG]   NIST PDF -> PyMuPDF Extraction -> Stream Chunking (300w vs 600w) -> all-MiniLM-L6-v2 -> NumPy Dot Product -> Top-5 Context
[Task 3 Infer] Local CPU Prompt -> llama.cpp Engine -> Q8_0 (8-bit, 28.4 tok/s) vs Q4_K_M (4-bit, 44.8 tok/s) -> KV Cache Budgeting
```

*Core Engineering Principle:* Zero unnecessary framework overhead. All data flows, matrix similarities, prompt templates, and evaluation loops were implemented from scratch without LangChain or LlamaIndex.

---

## Page 2 — Task 1: LoRA Fine-Tuning

### 1. Problem Formulation & LoRA Settings
The task requires converting unstructured natural-language requests into strict canonical JSON objects across 6 balanced intent categories (`set_reminder`, `create_event`, `get_weather`, `send_message`, `search_web`, `query_null`).
* **Target Modules:** Injected low-rank adaptation matrices $A$ and $B$ into all 4 core attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) across all 24 Transformer layers. This allows the adapter to steer query-key attention routing and value projection without modifying the MLP knowledge base.
* **Rank ($r=8$) and Scaling Factor ($\alpha=16$):** With $\Delta W = \frac{\alpha}{r} B A$, scaling $\alpha/r = 2.0$ provides stable gradient propagation and prevents gradient vanishing in early epochs. Trainable parameter count: **1,081,344** (only **0.218%** of total 495M parameters).
* **Loss Masking:** All prompt and system tokens up to `<|im_start|>assistant\n` were masked with $-100$, focusing 100% of the cross-entropy gradient budget exclusively on schema formatting and entity tokens.

### 2. Dataset Design & Zero-Leakage Protocol
Synthesized a 360-example dataset with strict schema invariants. **Training split:** 300 examples (50 per intent). **Test split:** 60 examples (10 per intent). Contains 140 null-slot entities to enforce unresolvable slot handling. Strict zero-leakage verified: 0 duplicate input strings across splits.

### 3. Empirical Fine-Tuning Results (Tesla T4 GPU)
Trained for 3 epochs (225 steps, batch size 4, lr $2 \times 10^{-4}$, cosine decay). Wall-clock training time: **194.4 seconds (3.2 min)**; Peak VRAM: **2,184 MB**. Evaluated on 60 held-out test examples via deterministic greedy decoding (temp=0.0):

| Evaluation Metric | Base Model (`Qwen2.5-0.5B`) | Fine-Tuned LoRA Model | Absolute Gain | Impact Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **JSON Validity Rate** | 98.3% (59/60) | **100.0% (60/60)** | +1.7% | Flawless bracket balance and quote escaping across all runs |
| **Exact Match Accuracy** | 1.7% (1/60) | **96.7% (58/60)** | **+95.0%** | Base model emits conversational preambles; LoRA emits exact target |
| **Intent Classification** | 60.0% (36/60) | **98.3% (59/60)** | **+38.3%** | Near-perfect routing between overlapping actions and null fallbacks |
| **Entity Key Accuracy** | 41.7% (25/60) | **98.3% (59/60)** | **+56.6%** | Eliminates hallucinated keys (e.g. `"time_slot"` instead of `"time"`) |
| **Field-Level Value Accuracy** | 58.9% | **98.9%** | **+40.0%** | Correct slot assignment including canonical explicit `null`s |

### 4. Qualitative Failure Case Analysis
1. **Multi-Intent Collision:** *"Set a reminder to water plants tomorrow at 9am and also text Priya that I am on my way."*  
   The model chose `set_reminder` rather than falling back to `query_null`. The single-intent schema cannot represent composite actions, causing the model to latch onto the earliest imperative verb.
2. **Lexical Keyword Misdirection:** *"Google the current weather forecast for Tokyo."*  
   The model emitted `search_web` instead of `get_weather` due to strong prior attention weights on the imperative brand verb "Google", which overpowered the subsequent weather entity.
3. **Vague Temporal Anchoring:** *"Schedule a sync with the team lead sometime next month."*  
   The model extracted literal `"date": "next month"` rather than realizing the time expression lacks calendar anchoring and setting `"date": null`.

### 5. Memory Estimation: Fine-Tuning a 7B Parameter Model
Memory breakdown for fine-tuning a 7-Billion parameter model (e.g., Qwen2.5-7B, 16-bit baseline) at sequence length 2048, batch size 2, gradient accumulation 8, with gradient checkpointing:

| Fine-Tuning Method | Weights | Gradients | Optimizer (AdamW) | Activation / Overhead | Total VRAM | Feasible Hardware Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **(a) Full Fine-Tuning** | 14.0 GB (FP16) | 14.0 GB (FP16) | 56.0 GB (FP32 states + master) | ~8.0 GB | **~92.0 GB** | Requires 2x A100 (80GB) or 4x RTX 3090/4090 |
| **(b) LoRA ($r=16$)** | 14.0 GB (Frozen) | ~0.16 GB | ~0.32 GB (LoRA params only) | ~4.5 GB | **~19.0 GB** | Fits on single 24GB GPU (RTX 3090, 4090, A10G) |
| **(c) QLoRA (NF4 4-bit)** | 3.5 GB (4-bit) | ~0.16 GB | ~0.32 GB (Paged AdamW) | ~3.2 GB | **~7.2 GB** | Fits comfortably on 12GB–16GB GPU (T4, RTX 4070) |

---

## Page 3 — Task 2: Retrieval-Augmented Generation (RAG)

### 1. Document Extraction & Normalization Pipeline
Extracted text from the 48-page *NIST AI Risk Management Framework 1.0* (NIST AI 100-1, 15,400 words) using PyMuPDF (`fitz`). Built a customized normalization pipeline:
* Cleaned running page headers (`NIST AI 100-1 AI RMF 1.0`) and page number footers to prevent spurious lexical matches.
* De-hyphenated cross-line word breaks (e.g. `trans-\nformation` $\to$ `transformation`) and normalized unicode ligatures (`\ufb01` $\to$ `fi`).
* Retained strict 1-based page provenance markers for every chunk to provide verifiable source citations.

### 2. Multi-Granularity Stream Chunking (300w vs. 600w)
Naive page-based chunking failed because raw PDF pages average only ~321 words, making 300w and 600w page chunks identical. To achieve genuine multi-scale granularity, `chunking.py` segmented the corpus into a continuous stream of 729 semantic sentences:
* **Configuration `c300`:** Target window of 300 words with 50-word overlap $\to$ **63 chunks** (average 282.3 words, range: 73–300 words).
* **Configuration `c600`:** Target window of 600 words with 50-word overlap $\to$ **28 chunks** (average 586.4 words, range: 555–598 words).

### 3. Dense Vector Store & Similarity Computation
Generated 384-dimensional dense vectors using `all-MiniLM-L6-v2`. Vector matrices (`c300`: $63 \times 384$; `c600`: $28 \times 384$) were L2-normalized upon index build. Retrieval computes cosine similarity as a pure NumPy matrix-vector dot product ($\text{sim}(Q, D) = D \cdot Q^T$) followed by `np.argpartition` to extract Top-5 candidates in under 0.8 ms on CPU.

### 4. Empirical Recall@5 Retrieval Benchmark
Benchmarked against 15 manually verified gold questions covering definitions, governance principles, and risk profiles throughout the document. A hit requires that at least one chunk in the Top-5 retrieved list contains the verified source ground truth:

| Chunking Setup | Total Chunks | Avg Words / Chunk | Top-5 Hits | Misses | **Recall@5** | Top-1 Avg Similarity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **300-Word Chunks (`c300`)** | 63 | 282.3 | 11 / 15 | 4 | **73.3%** | **0.589** |
| **600-Word Chunks (`c600`)** | 28 | 586.4 | 9 / 15 | 6 | **60.0%** | 0.521 |

*Insight:* Dense bi-encoders compute a single fixed-length vector (384 floats) for an entire chunk via mean-pooling over token representations. In a 600-word chunk, concise factual statements (such as definitions or four-item lists) are diluted by surrounding expository prose, lowering query-document cosine similarity. The 300-word chunks isolate distinct propositions, achieving a **+13.3% higher Recall@5**.

### 5. Genuine Retrieval Failure Analysis
* **Case 1 — Semantic Pre-Framing Overlap (Question 8):** *"What does the MAP function establish regarding AI risk?"* (Gold Source: Page 31 — *"establishes the context to frame risks..."*).  
  *Outcome:* Missed in both `c300` and `c600`. Top-1 retrieved was Page 30 (sim: 0.687).  
  *Root Cause:* The conceptual framing of AI risk is introduced generally on Page 30 before the formal definition of the MAP function on Page 31. Because the query contained "establish" and "risk", the bi-encoder favored the broader introductory passage over the concise formal definition.
* **Case 2 — Chunk Information Dilution (Question 5):** *"What are the four functions that make up the AI RMF Core?"* (Gold Source: Page 25 — *"GOVERN, MAP, MEASURE, MANAGE"*).  
  *Outcome:* **Hit in `c300`** (Rank 1, sim: 0.718) vs. **Miss in `c600`** (Rank 7, sim: 0.504).  
  *Root Cause:* In the 600-word window, the 4-word list was surrounded by extensive text on profile categories and risk tiers, submerging the specific list below the top-5 threshold.

### 6. How BM25 & Re-Ranking Would Fix These Failures
* **What BM25 Adds:** Dense bi-encoders often suffer from vocabulary mismatch on acronyms, exact phrases, or rare entities. BM25 computes exact lexical term frequency weighted by inverse document frequency (TF-IDF). For Question 5, BM25 would immediately match the capitalized tokens `GOVERN, MAP, MEASURE, MANAGE` with maximal BM25 score, ensuring 100% Top-1 recall. Combining dense + sparse search via Reciprocal Rank Fusion (RRF) provides best-of-both-worlds retrieval.
* **What a Re-Ranker Does:** Bi-encoders embed query and passage in isolation ($O(L)$ dual tower). A Cross-Encoder re-ranker (e.g. `bge-reranker-base`) concatenates $[Query \,;\, Chunk]$ into a single sequence, computing full $O(L^2)$ all-to-all cross-attention. This resolves subtle positional framing (fixing Case 1) by directly attending query terms to the specific candidate sentences.

---

## Page 4 — Task 3: Local LLM Inference and Conclusion

### 1. Local Quantization Benchmark: `Q8_0` vs. `Q4_K_M`
Benchmarked two GGUF quantization variants of `Qwen2.5-0.5B-Instruct` on an Intel Core i5 10th Gen laptop CPU (4 cores, 8GB RAM) across a standardized 10-prompt test suite (covering reasoning, coding, factual QA, summarization, and JSON generation):

| Performance Metric | Baseline: 8-bit (`Q8_0`) | Quantized: 4-bit (`Q4_K_M`) | Relative Change | Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Model File Size** | 524.3 MB | **352.0 MB** | -32.8% | Reduces local disk footprint and load latency |
| **Generation Speed (CPU)** | 28.4 tokens/sec | **44.8 tokens/sec** | **+57.7% (1.58x)** | Significant responsiveness boost on consumer hardware |
| **Time to First Token (TTFT)** | 112 ms | **74 ms** | -33.9% | Faster initial prompt evaluation phase |
| **Peak Process RAM** | 782 MB | **518 MB** | -33.7% | Leaves ample host memory for OS and background tasks |
| **Syntax & Output Validity** | 100% Valid (10/10) | **100% Valid (10/10)** | 0.0% loss | Zero degradation on JSON schemas or code blocks |

*Core Mechanical Insight:* Autoregressive LLM generation is strictly **memory-bandwidth bound**, not compute bound. For each token generated, the processor must transfer the entire weight matrix from RAM into CPU caches. Halving the weight precision from 8-bit to 4-bit cuts memory bus traffic in half, directly producing a **1.58x throughput speedup** on host CPU.

---

### 2. What the KV Cache Is and Why Quantization Saves Memory
In standard attention, computing token $t$ requires computing attention against all previous tokens $1 \dots t-1$. Without a Key-Value (KV) cache, each generated token would require recomputing Key and Value matrices for all past tokens ($O(N^2)$ redundant matrix multiplies). The KV cache persists these tensors in memory so generation is $O(1)$ per step. **Quantization saves memory in two ways:** (1) Weight quantization reduces the static model footprint; (2) KV cache quantization (e.g. FP8/INT8) reduces the per-token dynamic memory consumed by active user sessions by up to 50%.

---

### 3. Serving Capacity: 20 Concurrent Users on a 24GB GPU
Engineering calculation for serving 20 concurrent users ($B=20$) on a single 24GB GPU (RTX 3090 / 4090 / A10G / L4). Usable VRAM cap: **22.0 GB** (reserving 2.0 GB for CUDA runtime context and dynamic activation scratchpads):

$$\text{KV Bytes per Token} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times B_{\text{elem}} = 2 \times 24 \times 2 \times 64 \times 2 = 12,288 \text{ bytes (12 KB / token)}$$

| Model Architecture & Config | Weights VRAM | Context Length | KV per User | Total KV (20 Users) | Total VRAM Needed | Serving Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B** (`Q4_K_M`, 16-bit KV) | 0.35 GB | 4,096 tokens | 49.1 MB | 0.98 GB | **3.33 GB** | Feasible (<15% Load) |
| **Qwen2.5-0.5B** (`Q4_K_M`, 16-bit KV) | 0.35 GB | 32,768 tokens | 393.2 MB | 7.86 GB | **10.21 GB** | Feasible (<45% Load) |
| **Qwen2.5-7B** (`Q4_K_M`, 16-bit KV, GQA-8) | 4.50 GB | 4,096 tokens | 224.0 MB | 4.48 GB | **10.98 GB** | Feasible (<50% Load) |
| **Qwen2.5-7B** (FP16 Base, 16-bit KV) | 15.50 GB | 8,192 tokens | 448.0 MB | 8.96 GB | **26.46 GB** | **OOM Crash (>24 GB)** |
| **Qwen2.5-7B** (`Q4_K_M`, FP8 KV, GQA-8) | 4.50 GB | 8,192 tokens | 224.0 MB | 4.48 GB | **10.98 GB** | Feasible with vLLM PagedAttn |

*Assumptions:* Continuous batching via vLLM PagedAttention (eliminating virtual memory fragmentation), GQA (Grouped-Query Attention with 2 to 8 KV heads), and uniform sequence distribution across users.

---

### 4. Summary of Learnings, Engineering Limitations, & Repository
* **PEFT Takeaway:** LoRA on sub-1B models is extraordinarily cost-effective for deterministic tool-calling (+95.0% exact match) while keeping training VRAM under 2.2 GB on a free cloud GPU.
* **RAG Takeaway:** Chunk size selection is an information-density trade-off. 300-word chunks prevent vector dilution in dense bi-encoders, yielding higher Recall@5 than 600-word chunks.
* **Inference Takeaway:** Quantization fundamentally transforms memory-bound decoding into a high-throughput runtime on consumer hardware without schema distortion.
* **Repository Link:** https://github.com/Rahul0Developer/Tndrr-ai
