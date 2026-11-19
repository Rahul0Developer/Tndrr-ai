# Task 2: Retrieval-Augmented Generation (RAG) on NIST AI RMF 1.0

**Tndrr AI Internship Assignment – Task 2**  
**Component:** Explicit RAG Architecture (No LangChain / No LlamaIndex)  
**Target Document:** NIST AI Risk Management Framework 1.0 (NIST AI 100-1, April 2023, 48 pages)  
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)  
**Vector Engine:** Pure NumPy Cosine Similarity (`matrix @ vector`)  
**Generation Target:** Grounded Prompts with `[chunk_id, page N]` Citations & Qwen2.5-0.5B-Instruct  

---

## 1. Architectural Overview

Many production RAG systems suffer from brittleness and vendor lock-in due to heavyweight orchestration frameworks. This implementation constructs an explicit, modular, zero-framework RAG pipeline from first principles using:
1. **PyMuPDF (`fitz`):** High-fidelity PDF text extraction with page-level bounding.
2. **Text Normalization Engine:** Header removal, line-break de-hyphenation, and ligature resolution.
3. **Continuous Stream Chunking:** Sentence-level sliding-window chunking across page boundaries to test 300-word vs. 600-word granularities.
4. **NumPy Vector Store:** L2-normalized dense vector dot-product search with zero database dependencies.
5. **Grounded Generation Prompting:** Context assembly with strict character budgeting and per-passage citations.
6. **Reproducible Recall@5 Evaluation:** Evaluation benchmark across 15 manually verified gold questions.

```
[NIST AI RMF 1.0 PDF]
        │
        ▼ (PyMuPDF)
[48 Page Texts + Metadata]
        │
        ▼ (Text Cleaning & Normalization)
[Sentence Stream with Page Provenance]
        │
   ┌────┴──────────────────────────┐
   ▼ (300w / 50w overlap)          ▼ (600w / 50w overlap)
[c300: 63 chunks]               [c600: 28 chunks]
   │                               │
   └───────────────┬───────────────┘
                   ▼ (all-MiniLM-L6-v2)
        [NumPy Vector Store (.npy)]
                   │
                   ▼ (Cosine Similarity Matrix Dot-Product)
        [Top-5 Chunks with [chunk_id, page N]]
                   │
                   ▼
        [Grounded Prompt Construction] ──► [Qwen2.5-0.5B-Instruct / Local Inference]
```

---

## 2. Document Extraction & Normalization

The selected document is the **NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)**, published as NIST AI 100-1 (48 pages, public domain U.S. government document).

### Normalization Pipeline (`extract_pdf.py`)
- **Header Stripping:** Matches and drops repeated running headers (`^NIST AI 100-1\s*AI RMF 1\.0\s*$`).
- **De-hyphenation:** Recombines words split across lines by hyphens (`of-\nfers` $\to$ `offers`) while preserving true compound words and em/en dashes.
- **Ligature & Unicode Normalization:** Maps extraction ligatures (`\ufb01` $\to$ `fi`, `\ufb02` $\to$ `fl`) and non-breaking spaces (`\u00a0` $\to$ `' '`).
- **Page Provenance:** Every extracted segment preserves its 1-based PDF page index (`page: 1..48`). Total corpus size: **15,400 words**.

---

## 3. Chunking Methodology & Empirical Analysis

The specification mandates evaluating two distinct chunking configurations: **300-word chunks** vs. **600-word chunks**, both using a fixed **50-word overlap**.

### The Stream Chunking Problem
In a 48-page PDF of 15,400 words, each page averages ~321 words. A naive page-by-page chunker would produce virtually identical chunks for both 300-word and 600-word configurations (because single pages do not exceed 600 words).

To solve this, `chunking.py` treats the entire document as **one continuous stream of semantic sentences** with page provenance tags:
- Sentences are accumulated into a buffer until adding the next would exceed the target word count.
- Chunks are flushed and trailing sentences totalling $\le 50$ words are rolled back to seed the subsequent chunk.
- Each chunk preserves `primary_page` (first page in the chunk) and `pages` (list of all pages spanned).

### Resulting Chunk Statistics

| Metric | 300-Word Config (`c300`) | 600-Word Config (`c600`) |
| :--- | :---: | :---: |
| **Total Chunks Produced** | **63 chunks** | **28 chunks** |
| **Average Word Count** | **282.3 words** | **586.4 words** |
| **Word Count Range** | 73 – 300 words | 555 – 598 words |
| **Overlap Setting** | 50 words | 50 words |
| **Index Size (`.npy`)** | 63 $\times$ 384 float32 (0.10 MB) | 28 $\times$ 384 float32 (0.04 MB) |

---

## 4. Dense Retrieval Engine (`vector_store.py` & `retrieval.py`)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embedding space).
- **L2 Normalization:** Embeddings are normalized to unit sphere ($\|v\|_2 = 1.0$) upon indexing.
- **Search Complexity:** Cosine similarity reduces to an optimized matrix-vector dot product:
  $$\text{sim}(Q, D_i) = D_i \cdot Q^T$$
- **Top-$K$ Selection:** Uses `np.argpartition` followed by sorting on the top-$K$ indices, achieving sub-millisecond retrieval on CPU.

---

## 5. Recall@5 Evaluation Benchmark

Evaluation was performed against 15 manually verified gold questions in `questions.json` covering definition, architecture, governance, and operational requirements throughout the NIST document.

### Hit-Rate Recall@5 Definition
$$\text{Recall@5} = \frac{\#\text{ questions with } \ge 1 \text{ relevant chunk in top-5 retrieved}}{\text{Total Questions } (N=15)}$$

A chunk is scored as relevant if all verified query-specific target keywords appear in the retrieved passage text.

### Empirical Results

| Configuration | Total Questions | Top-5 Hits | Misses | **Recall@5** |
| :--- | :---: | :---: | :---: | :---: |
| **300-Word Chunks (`c300`)** | 15 | 11 | 4 | **73.3%** |
| **600-Word Chunks (`c600`)** | 15 | 9 | 6 | **60.0%** |

### Why 300-Word Chunks Outperform 600-Word Chunks
1. **Semantic Dilution in Dense Embeddings:** Dense encoders average contextual representations across all sequence tokens. In 600-word chunks, specific factual answers (e.g., the four functions: GOVERN, MAP, MEASURE, MANAGE) are surrounded by hundreds of words of adjacent introductory exposition, diluting the embedding vector and lowering cosine similarity against specific queries.
2. **Information Focus:** 300-word chunks have higher semantic density and information focus, allowing specific passages to match query embeddings with higher fidelity.

---

## 6. Genuine Retrieval Failure Case Post-Mortems

Per the engineering standards of this assignment, the following genuine misses were extracted from the benchmark run and analyzed:

### Failure Case 1: Semantic Overlap & Pre-Definition Framing (`q08`)
- **Question:** *"What does the MAP function establish regarding AI risk?"*
- **Expected Source Passage:** Page 31 (`MAP ... establishes the context to frame risks related to an AI system.`)
- **Keywords Required:** `["establishes the context", "frame"]`
- **Result:** Missed in **both** `c300` and `c600`.
- **Top-1 Retrieved Chunk:** `c300-0039` (Page 30, similarity `0.687`), covering general risk management concepts.
- **Root Cause & Technical Explanation:**  
  The word "MAP" appears repeatedly throughout the Framework. The question contains high semantic similarity with several introductory discussions on Page 30 and Page 26 where "framing AI risk" is introduced conceptually before the formal MAP function definition on Page 31. The bi-encoder ranked the broader introductory discussion higher than the specific definition passage.

### Failure Case 2: Chunk Dilution in Large Chunks (`q05`)
- **Question:** *"What are the four functions that make up the AI RMF Core?"*
- **Expected Source Passage:** Page 25 (`...the Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE.`)
- **Keywords Required:** `["four functions", "govern", "map", "measure", "manage"]`
- **Result:** **HIT in `c300`** (`c300-0031` Page 25 ranked #1, similarity `0.718`) vs. **MISS in `c600`** (top chunk was `c600-0014`, but the exact paragraph was pushed down).
- **Root Cause & Technical Explanation:**  
  In the 600-word window, the short enumeration of the four functions was embedded alongside extensive surrounding descriptive text regarding profiles and risk tiers. This diluted the representation of the specific four-word list, causing its similarity score to drop below broader lifecycle discussion passages.

---

## 7. How BM25 & Reranking Would Enhance the Pipeline

*(Note: In accordance with assignment instructions, the following techniques are proposed and analyzed as engineering enhancements, but were not enabled in the primary benchmark to preserve baseline evaluation integrity.)*

### 1. Hybrid Search with BM25 (Lexical + Dense Retrieval)
- **Problem in Pure Dense Retrieval:** Dense embeddings suffer from the "vocabulary mismatch" problem and frequently miss exact proper nouns, acronyms, and enumeration lists (e.g. `q05` "GOVERN, MAP, MEASURE, MANAGE" or `q09` "MEASURE 2").
- **BM25 Solution:** BM25 scores exact keyword occurrences and penalizes common terms via Inverse Document Frequency (IDF).
- **Hybrid Fusion:** Combining BM25 and dense cosine similarity using **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
  This guarantees that exact technical matches surface to the top even when semantic embeddings are diluted.

### 2. Two-Stage Retrieval with Cross-Encoder Reranking
- **Problem with Bi-Encoders:** Bi-encoders (`all-MiniLM-L6-v2`) embed query and chunk independently, meaning zero cross-attention occurs between question terms and passage words.
- **Cross-Encoder Solution:** A cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or `bge-reranker-base`) processes the query and passage concatenated together:
  $$\text{Score} = \text{CrossEncoder}([Q \,;\, D])$$
- **Pipeline Implementation:** The bi-encoder retrieves the top-20 candidate chunks at low latency ($<5\text{ms}$), and the cross-encoder re-ranks the top 20 to select the final top 5. This would resolve false-positive ranking errors like `q08` by directly scoring the interaction between "MAP function" and "establishes the context".

---

## 8. Grounded Prompt Construction & Generation

`rag_pipeline.py` builds prompt contexts using the top-5 retrieved passages while managing context length:
- **Budget Control:** Context is capped at 6,000 characters (~1,500 tokens), preventing context overflow.
- **Passage Citation:** Every chunk is prepended with its metadata: `[<chunk_id>, page <page>] <text>`.
- **System Prompt:** Instructs the generator to answer **ONLY** from the provided context and cite specific chunk IDs and page numbers.

### Sample Pipeline Output
```text
QUESTION: What are the four functions of the AI RMF Core?
CONFIGURATION: c300 (Top-5 Retrieved Passages)

=== TOP 5 RETRIEVED SOURCES ===
[1] c300-0031 | Page 25 | Cosine Similarity: 0.7177
    Snippet: Part 2: Core and Profiles 5. AI RMF Core The AI RMF Core provides outcomes and actions that enable dialogue, understandi...
[2] c300-0007 | Page 8 | Cosine Similarity: 0.5935
    Snippet: valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable, privacy enha...
[3] c300-0032 | Page 26 | Cosine Similarity: 0.5921
    Snippet: Like the AI RMF, the Playbook is voluntary and organizations can utilize the suggestions according to their needs and in...
[4] c300-0006 | Page 7 | Cosine Similarity: 0.5640
    Snippet: The Framework is designed to equip organizations and individuals – referred to here as AI actors – with approaches that ...
[5] c300-0047 | Page 36 | Cosine Similarity: 0.5259
    Snippet: Page 31 Practices related to managing AI risks are described in the NIST AI RMF Playbook. Table 4 lists the MANAGE funct...

=== GENERATED ANSWER (Qwen2.5-0.5B-Instruct Grounded) ===
The four functions that make up the AI RMF Core are GOVERN, MAP, MEASURE, and MANAGE [c300-0031, page 25].
```

---

## 9. Step-by-Step Reproduction Guide

### Option A: Run via Terminal / Shell

```bash
cd tndrr-ai-internship-assignment/task2_rag

# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract text from PDF with page preservation
python extract_pdf.py

# 3. Generate 300w and 600w sentence-stream chunks
python chunking.py

# 4. Compute dense vector embeddings
python embeddings.py

# 5. Execute Recall@5 retrieval benchmark
python evaluate_retrieval.py

# 6. Test RAG query with grounded prompt and citations
python rag_pipeline.py --question "What are the four functions of the AI RMF Core?" --config c300 --dry-run
```

### Option B: Run in Google Colab

1. Open Google Colab and open `task2_rag_pipeline.ipynb` (or open from GitHub).
2. Run all cells sequentially. The notebook installs dependencies, extracts text, generates embeddings, calculates Recall@5 for both chunk sizes, and runs interactive grounded querying.

---

## 10. Summary of Deliverables & Status

| Deliverable | File Path | Status |
| :--- | :--- | :---: |
| **Requirements** | `task2_rag/requirements.txt` | **COMPLETED** |
| **PDF Extraction** | `task2_rag/download_document.py`, `extract_pdf.py` | **COMPLETED** |
| **Chunking Engine** | `task2_rag/chunking.py` (300w & 600w sliding window) | **COMPLETED** |
| **Embedding Generator** | `task2_rag/embeddings.py` (all-MiniLM-L6-v2) | **COMPLETED** |
| **Dense Vector Store** | `task2_rag/vector_store.py` (NumPy cosine similarity) | **COMPLETED** |
| **Retrieval Layer** | `task2_rag/retrieval.py` | **COMPLETED** |
| **RAG Pipeline** | `task2_rag/rag_pipeline.py` (Context budget + citations) | **COMPLETED** |
| **Evaluation Set** | `task2_rag/questions.json` (15 verified gold questions) | **COMPLETED** |
| **Evaluation Script** | `task2_rag/evaluate_retrieval.py` | **COMPLETED** |
| **Raw Results** | `task2_rag/results/recall_at_5.json`, `sample_answers.jsonl` | **COMPLETED** |
| **Interactive Notebook**| `task2_rag/task2_rag_pipeline.ipynb` | **COMPLETED** |
| **Documentation** | `task2_rag/README.md` | **COMPLETED** |
